"""Text-to-speech via Piper, falling back to Windows SAPI (Phase 5).

Piper produces more natural speech but is GPL-3.0 and may not install
on every Windows/Python combination.  ``pyttsx3`` (wrapping Windows
SAPI5) is always available as a fallback.

:func:`create` tries Piper first; if it fails, falls back to SAPI.
If neither works, returns ``None``.

SAPI5 / pyttsx3 note
--------------------
The Windows SAPI5 COM backend in ``pyttsx3`` has a well-documented bug
where repeated ``runAndWait()`` calls on the **same engine instance**
eventually deadlock the calling thread.  ``SapiTTSEngine`` therefore
creates a **fresh** ``pyttsx3`` engine for every ``speak()`` call and
tears it down immediately afterwards.  This is the recommended
workaround from the pyttsx3 community and avoids the COM event-loop
hang that would otherwise block the voice-loop thread permanently after
the first voice interaction.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


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
    """TTS via Windows SAPI (pyttsx3).

    A **fresh** ``pyttsx3`` engine is created for every :meth:`speak`
    call and torn down immediately afterwards.  Reusing a single
    persistent engine causes the SAPI5 COM event loop inside
    ``runAndWait()`` to deadlock after a small number of calls (a
    well-documented pyttsx3 / Windows SAPI5 bug).  The per-call init
    adds ~30 ms of overhead — negligible compared to the speech
    synthesis itself — and guarantees the voice-loop thread is never
    permanently blocked by a stale COM state.
    """

    def __init__(self) -> None:
        # Validate that pyttsx3 is importable at construction time so
        # create() can report "tts-unavailable" early.  No persistent
        # engine is stored.
        import pyttsx3  # noqa: F401

        logger.info("TTS sapi backend initialized (fresh engine per speak)")

    def speak(self, text: str) -> None:
        """Speak *text* synchronously with a fresh COM engine."""
        import pyttsx3

        logger.info("TTS sapi speak start (chars=%d)", len(text))
        t0 = time.monotonic()
        logger.info("TTS sapi init start")
        engine = pyttsx3.init()
        logger.info("TTS sapi engine initialized")
        try:
            logger.info("TTS sapi say start")
            engine.say(text)
            logger.info("TTS sapi runAndWait start")
            engine.runAndWait()
            logger.info("TTS sapi runAndWait done")
        finally:
            logger.info("TTS sapi cleanup start")
            try:
                engine.stop()
            except Exception:
                logger.debug("pyttsx3 engine.stop() failed", exc_info=True)
            logger.info("TTS sapi speak done (elapsed_s=%.2f)", time.monotonic() - t0)

    def stop(self) -> None:
        """Stop any in-progress speech (best-effort, no persistent engine)."""
        # With per-call engines there is no persistent engine to stop.
        # This method exists to satisfy the TextToSpeech protocol.


def create(
    *,
    backend: str = "piper",
    model_path: str | None = None,
) -> PiperTTSEngine | SapiTTSEngine | None:
    """Create a TTS engine.

    Tries ``piper`` first if requested, then falls back to ``sapi``.
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

    # SAPI fallback
    try:
        return SapiTTSEngine()
    except ImportError:
        logger.warning(
            "pyttsx3 not installed — TTS unavailable (install with: pip install pyttsx3)"
        )
    except Exception:
        logger.warning("pyttsx3 init failed", exc_info=True)

    return None
