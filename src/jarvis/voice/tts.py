"""Text-to-speech: Piper when a model is available, else Windows SAPI.

:func:`create` prefers Piper (more natural speech, GPL-3.0, and a model file
must be configured) and falls back to Windows SAPI, which is always present on
Windows.  ``None`` means no engine could be built, and the caller degrades to
console-only output rather than failing.

SAPI is driven through the ``SAPI.SpVoice`` COM object (``win32com``, already a
hard dependency via ``pywin32``) instead of ``pyttsx3``, for one reason:
**an answer must be interruptible.**  ``ISpVoice`` speaks asynchronously and
can be purged mid-utterance, which is what lets the voice loop cut a long answer
short when the user says the wake word.  ``pyttsx3``'s ``runAndWait()`` cannot
be interrupted from another thread, so it stays only as a last-resort fallback
for an environment without ``win32com`` - and such an engine is explicitly
*not* interruptible.

Two SAPI5 details this module owns deliberately:

* **One ``SpVoice`` per utterance.**  Reusing a single SAPI5 COM voice across
  utterances eventually deadlocks the calling thread in its event loop (the
  long-standing pyttsx3/SAPI5 bug).  Creating and releasing the object per
  utterance costs a few milliseconds and keeps the voice-loop thread from ever
  inheriting a stale COM state.
* **COM is initialised explicitly.**  The voice loop runs in a background
  thread, and ``CoInitialize`` is per-thread: ``Dispatch`` on a thread that has
  not initialised COM fails.  Every engine that touches SAPI therefore
  initialises COM on the calling thread and uninitialises it when the object is
  released.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

#: ``ISpVoice`` flags.  ``SVSFlagsAsync`` returns immediately so the caller can
#: keep listening on the microphone; adding ``SVSFPurgeBeforeSpeak`` to an empty
#: utterance is the documented way to cancel whatever is queued.
_SVS_ASYNC = 0x1
_SVS_PURGE = 0x2
_SVS_ASYNC_PURGE = _SVS_ASYNC | _SVS_PURGE

#: How often a blocking :meth:`SapiTTSEngine.speak` re-checks for completion.
_SAPI_POLL_S = 0.2

#: Ceiling for one blocking SAPI utterance.  A wedged voice must not hold the
#: voice-loop thread (and the microphone) forever: at the ceiling the queue is
#: purged and the call returns.
_SAPI_MAX_S = 300.0

#: ``SpVoice`` by ProgID, then by CLSID, for builds that register only one.
_SAPI_PROGIDS = ("SAPI.SpVoice", "{269316D8-57BD-11D2-9EEE-00C04F797396}")


class PiperTTSEngine:
    """TTS via piper-tts (GPL-3.0)."""

    def __init__(self, model_path: str, *, config_path: str | None = None) -> None:
        self._model_path = model_path
        self._config_path = config_path
        self._synth: Any = None

    def _ensure_loaded(self) -> None:
        if self._synth is not None:
            return
        from piper import PiperVoice

        self._synth = PiperVoice.load(self._model_path, config_path=self._config_path)

    def speak(self, text: str) -> None:
        """Speak *text* synchronously."""
        import numpy as np
        import sounddevice as sd

        self._ensure_loaded()
        logger.info("TTS piper speak start (chars=%d)", len(text))
        # PiperVoice.synthesize_stream_raw returns raw PCM int16 chunks.
        audio_chunks: list[bytes] = list(self._synth.synthesize_stream_raw(text))
        raw = b"".join(audio_chunks)
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        sd.play(audio, self._synth.config.sample_rate)
        sd.wait()
        logger.info("TTS piper speak done")

    def stop(self) -> None:
        """Stop any in-progress speech."""
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:
            logger.debug("sounddevice stop failed", exc_info=True)


class SapiTTSEngine:
    """TTS via Windows SAPI (``SAPI.SpVoice``), and interruptible.

    Implements the blocking :meth:`speak` the ``TextToSpeech`` protocol
    guarantees, plus the non-blocking :meth:`start_speaking` /
    :meth:`is_speaking` pair the voice loop uses for barge-in.  A
    :class:`~jarvis.voice.interfaces.TextToSpeech` consumer that only knows
    ``speak``/``stop`` is unaffected.
    """

    def __init__(self) -> None:
        # Validate at construction time so create() can report
        # "tts-unavailable" early, without opening a voice yet.
        self._lock = threading.Lock()
        self._sapi: Any = None
        self._com_ready = False
        self._probe_dispatch()
        logger.info("TTS sapi backend initialized (win32com SpVoice, stoppable)")

    # ── COM plumbing ──────────────────────────────────────────────────

    @staticmethod
    def _dispatch_voice() -> Any:
        import win32com.client

        for target in _SAPI_PROGIDS:
            try:
                return win32com.client.Dispatch(target)
            except Exception:
                # Deliberate probe: the next registered name may work, and the
                # traceback is logged rather than swallowed.
                logger.debug("SAPI voice creation via %s failed", target, exc_info=True)
        raise RuntimeError(f"could not create an SAPI SpVoice (tried {', '.join(_SAPI_PROGIDS)})")

    def _probe_dispatch(self) -> None:
        """Open and immediately release a voice, to fail fast if SAPI is absent."""
        import pythoncom

        pythoncom.CoInitialize()
        self._com_ready = True
        try:
            self._dispatch_voice()
        finally:
            pythoncom.CoUninitialize()
            self._com_ready = False

    def _acquire(self) -> Any:
        """Start an utterance: COM on this thread, then queue *text*."""
        import pythoncom

        with self._lock:
            self._release_locked()
            pythoncom.CoInitialize()
            self._com_ready = True
            try:
                sapi = self._dispatch_voice()
            except Exception:
                pythoncom.CoUninitialize()
                self._com_ready = False
                raise
            self._sapi = sapi
            return sapi

    def _release_locked(self) -> None:
        self._sapi = None
        if not self._com_ready:
            return
        self._com_ready = False
        try:
            import pythoncom

            pythoncom.CoUninitialize()
        except Exception:  # pragma: no cover - shutdown races
            logger.debug("CoUninitialize failed", exc_info=True)

    def _release(self) -> None:
        with self._lock:
            self._release_locked()

    # ── TextToSpeech + barge-in ────────────────────────────────────────

    def speak(self, text: str) -> None:
        """Speak *text* synchronously (blocks until done or purged)."""
        logger.info("TTS sapi speak start (chars=%d)", len(text))
        started = time.monotonic()
        sapi = self._acquire()
        try:
            sapi.Speak(text, _SVS_ASYNC)
            while not sapi.WaitUntilDone(_SAPI_POLL_S * 1000):
                if time.monotonic() - started >= _SAPI_MAX_S:
                    logger.warning("TTS sapi: utterance ceiling reached; purging")
                    self._purge(sapi)
                    break
        finally:
            self._release()
        logger.info("TTS sapi speak done (elapsed_s=%.2f)", time.monotonic() - started)

    def start_speaking(self, text: str) -> None:
        """Queue *text* and return immediately (barge-in capable)."""
        logger.info("TTS sapi async speak start (chars=%d)", len(text))
        sapi = self._acquire()
        sapi.Speak(text, _SVS_ASYNC)

    def is_speaking(self) -> bool:
        """Whether queued speech is still playing.  Releases the voice when done."""
        with self._lock:
            sapi = self._sapi
            if sapi is None:
                return False
            try:
                if not sapi.WaitUntilDone(0):
                    return True
            except Exception:
                logger.debug("SAPI WaitUntilDone failed", exc_info=True)
            self._release_locked()
            return False

    def stop(self) -> None:
        """Purge any in-progress speech (best effort)."""
        with self._lock:
            sapi = self._sapi
        if sapi is None:
            return
        logger.info("TTS sapi purge requested")
        self._purge(sapi)
        self._release()

    @staticmethod
    def _purge(sapi: Any) -> None:
        try:
            sapi.Speak("", _SVS_ASYNC_PURGE)
        except Exception:
            logger.debug("SAPI purge failed", exc_info=True)


class Pyttsx3TTSEngine:
    """Last-resort SAPI fallback via ``pyttsx3`` — **not interruptible**.

    Used only when ``win32com`` is unavailable.  ``pyttsx3`` speaks through
    ``runAndWait()``, which cannot be cancelled from another thread, so an
    answer spoken by this engine cannot be cut short by a barge-in; the voice
    loop detects the missing ``start_speaking`` and falls back to blocking
    speech.  It still honours the SAPI5 workaround: a **fresh** engine per
    :meth:`speak`, torn down immediately afterwards, because reusing one engine
    eventually deadlocks in the SAPI5 COM event loop.
    """

    def __init__(self) -> None:
        import pyttsx3  # noqa: F401

        logger.info("TTS pyttsx3 fallback initialized (not interruptible)")

    def speak(self, text: str) -> None:
        """Speak *text* synchronously with a fresh COM engine."""
        import pyttsx3

        logger.info("TTS pyttsx3 speak start (chars=%d)", len(text))
        started = time.monotonic()
        engine = pyttsx3.init()
        try:
            engine.say(text)
            engine.runAndWait()
        finally:
            try:
                engine.stop()
            except Exception:
                logger.debug("pyttsx3 engine.stop() failed", exc_info=True)
        logger.info("TTS pyttsx3 speak done (elapsed_s=%.2f)", time.monotonic() - started)

    def stop(self) -> None:
        """No-op: with a per-call engine there is nothing to purge."""


def create(
    *,
    backend: str = "piper",
    model_path: str | None = None,
) -> PiperTTSEngine | SapiTTSEngine | Pyttsx3TTSEngine | None:
    """Create a TTS engine.

    Tries ``piper`` first if requested, then falls back to SAPI: the
    interruptible ``win32com`` engine, or ``pyttsx3`` if ``win32com`` is missing.
    Returns ``None`` if nothing works.
    """
    if backend == "piper":
        try:
            if model_path:
                return PiperTTSEngine(model_path)
            logger.warning(
                "piper backend requested but no model_path provided; falling back to SAPI"
            )
        except ImportError:
            logger.info("piper-tts not installed; falling back to SAPI")
        except Exception:
            logger.warning("piper-tts init failed", exc_info=True)

    try:
        return SapiTTSEngine()
    except ImportError:
        logger.warning("win32com not installed; trying the pyttsx3 SAPI fallback")
    except Exception:
        logger.warning("SAPI SpVoice init failed; trying the pyttsx3 fallback", exc_info=True)

    try:
        return Pyttsx3TTSEngine()
    except ImportError:
        logger.warning("no SAPI engine available (install with: pip install pywin32 or pyttsx3)")
    except Exception:
        logger.warning("pyttsx3 init failed", exc_info=True)

    return None
