"""Voice service lifecycle manager (Phase 5 Stage 2).

Manages the :class:`~jarvis.voice.loop.VoiceLoop` thread and provides
a clean start/stop interface for the CLI (``jarvis on/off``) and the
daemon.

Voice state machine (Phase 5 Stage 2):

    off ──start()──▶ starting ──components ok + mic opens──▶ on
                     │                                         │
                     └──missing component / mic open failed──▶ error
                                                               │
    on ──crash / on_exit(reason)──▶ error    on ──clean exit──▶ off
    error ──start() retry──▶ starting (retries allowed)

``error`` always carries one of the fixed :data:`VOICE_ERROR_CODES`.
The daemon surfaces ``state`` / ``error_code`` through ``StatusResponse``
so ``jarvis status`` and ``jarvis on`` can report the same reason.
"""

from __future__ import annotations

import logging
import threading
import traceback
from typing import Any, Literal

from jarvis.voice.interfaces import (
    AudioInput,
    SpeechToText,
    TextToSpeech,
    WakeWordDetector,
    WindowFocusChecker,
)
from jarvis.voice.loop import DEFAULT_MAX_DICTATION_CHARS, DEFAULT_MAX_SESSION_S, VoiceLoop

logger = logging.getLogger(__name__)

#: Fixed voice error codes (Phase 5 Stage 2).  No other code may be emitted
#: by the voice pipeline; see tests/unit/test_invariants.py.
VOICE_ERROR_CODES = frozenset(
    {
        "no-audio-library",
        "mic-open-failed",
        "wake-model-missing",
        "stt-model-missing",
        "tts-unavailable",
        "loop-crashed",
    }
)

#: Valid voice states (Phase 5 Stage 2 state machine).
VoiceState = Literal["off", "starting", "on", "error"]

#: User-facing hint for each fixed error code (surfaced by ``jarvis on``).
VOICE_HINTS: dict[str, str] = {
    "no-audio-library": "install the voice extras with: pip install jarvis-agent[voice]",
    "mic-open-failed": "check that a microphone is connected and enabled in Windows",
    "wake-model-missing": "install openwakeword or set [voice] wake_word in config.toml",
    "stt-model-missing": "install faster-whisper or set [voice] stt_model in config.toml",
    "tts-unavailable": "install piper-tts / pyttsx3 or set [voice] tts_backend in config.toml",
    "loop-crashed": "see the daemon log for the loop traceback; text and CLI keep working",
}


def voice_error_message(code: str, *, hint: bool = True) -> str:
    """Render the human-readable reason for a voice error code.

    Any unknown code falls back to ``loop-crashed`` so an unqualified
    string can never escape the fixed set (fail closed).
    """
    if code not in VOICE_ERROR_CODES:
        code = "loop-crashed"
    text = f"voice is in error: {code}"
    if hint:
        text += f" — {VOICE_HINTS[code]}"
    return text


class VoiceService:
    """Lifecycle manager for the voice pipeline.

    Create once in the daemon or CLI, then call :meth:`start` /
    :meth:`stop` / :meth:`is_active` as needed.  All lifecycle transitions
    are serialised under a lock so concurrent ``start``/``stop`` calls (e.g.
    a daemon toggle racing a shutdown) cannot corrupt the loop state.
    """

    def __init__(
        self,
        *,
        audio: AudioInput | None = None,
        wake_detector: WakeWordDetector | None = None,
        stt: SpeechToText | None = None,
        tts: TextToSpeech | None = None,
        focus_checker: WindowFocusChecker | None = None,
        submit_task: Any | None = None,
        on_confirmation: Any | None = None,
        on_dictation: Any | None = None,
        wake_word: str = "hey jarvis",
        idle_timeout_s: float = 120.0,
        listen_timeout_s: float = 30.0,
        max_session_s: float = DEFAULT_MAX_SESSION_S,
        max_dictation_chars: int = DEFAULT_MAX_DICTATION_CHARS,
        silence_timeout_s: float = 0.7,
        max_segment_s: float = 30.0,
        silence_threshold: float = 0.01,
    ) -> None:
        self._audio = audio
        self._wake_detector = wake_detector
        self._stt = stt
        self._tts = tts
        self._focus_checker = focus_checker
        self._submit_task = submit_task
        self._on_confirmation = on_confirmation
        self._on_dictation = on_dictation
        self._wake_word = wake_word
        self._idle_timeout_s = idle_timeout_s
        self._listen_timeout_s = listen_timeout_s
        self._max_session_s = max_session_s
        self._max_dictation_chars = max_dictation_chars
        self._silence_timeout_s = silence_timeout_s
        self._max_segment_s = max_segment_s
        self._silence_threshold = silence_threshold
        self._state: VoiceState = "off"
        self._error_code: str | None = None
        self._loop: VoiceLoop | None = None
        self._lock = threading.Lock()

    # ── state accessors ────────────────────────────────────────────────

    @property
    def state(self) -> VoiceState:
        """Current voice state (off | starting | on | error)."""
        with self._lock:
            return self._state

    @property
    def error_code(self) -> str | None:
        """Fixed error code when ``state == "error"``, else ``None``."""
        with self._lock:
            return self._error_code

    # ── lifecycle ──────────────────────────────────────────────────────

    def start(self) -> str:
        """Start the voice loop.  Returns a status message.

        On success state becomes ``on``; on any failure state becomes
        ``error`` with the fixed reason code embedded in the message, so
        the caller can distinguish a real error from a graceful message.
        """
        with self._lock:
            return self._start_locked()

    def _start_locked(self) -> str:
        self._state = "starting"
        print("[DIAG] voice.service._start_locked(): state=starting", flush=True)
        if self._loop is not None and self._loop.is_active():
            self._state = "on"
            return "voice is already active"

        error = self._missing_component_code()
        print(f"[DIAG] voice.service._start_locked(): missing_component={error!r}", flush=True)
        if error is None and self._audio is not None:
            # Open the microphone synchronously so a broken device is
            # reported here (mic-open-failed), not silently in the thread.
            try:
                self._audio.open()
                print("[DIAG] voice.service._start_locked(): audio.open() OK", flush=True)
            except Exception:
                logger.warning("voice audio could not be opened", exc_info=True)
                print("[DIAG] voice.service._start_locked(): audio.open() FAILED", flush=True)
                error = "mic-open-failed"

        if error is not None:
            return self._fail_locked(error)

        try:
            # Components were verified non-None above; assert to satisfy the
            # type checker and fail loudly if the guard ever regresses.
            assert self._audio is not None
            assert self._stt is not None
            assert self._tts is not None
            self._loop = VoiceLoop(
                audio=self._audio,
                wake_detector=self._wake_detector,
                stt=self._stt,
                tts=self._tts,
                focus_checker=self._focus_checker,
                submit_task=self._submit_task,
                on_confirmation=self._on_confirmation,
                on_dictation=self._on_dictation,
                wake_word=self._wake_word,
                listen_timeout_s=self._listen_timeout_s,
                idle_timeout_s=self._idle_timeout_s,
                max_session_s=self._max_session_s,
                max_dictation_chars=self._max_dictation_chars,
                silence_timeout_s=self._silence_timeout_s,
                max_segment_s=self._max_segment_s,
                silence_threshold=self._silence_threshold,
                on_exit=self._on_loop_exit,
            )
            self._loop.start()
        except Exception:
            logger.exception("voice loop failed to start")
            print("[DIAG] voice.service._start_locked(): VoiceLoop.start() RAISED", flush=True)
            traceback.print_exc()
            if self._audio is not None:
                try:
                    self._audio.close()
                except Exception:
                    logger.debug("error closing audio after failed start", exc_info=True)
            return self._fail_locked("loop-crashed")

        self._state = "on"
        self._error_code = None
        return "voice activated"

    def stop(self) -> str:
        """Stop the voice loop.  Returns a status message.

        The loop's ``stop()`` waits for the thread, so it must not run
        while holding the service lock — the loop thread reports its
        clean exit through :meth:`_on_loop_exit`, which needs the lock.
        """
        with self._lock:
            loop = self._loop
            if loop is None or not loop.is_active():
                return "voice is not active"
            self._loop = None
        loop.stop()
        with self._lock:
            self._state = "off"
            self._error_code = None
        return "voice deactivated"

    def is_active(self) -> bool:
        """Return True if the voice loop is running (state ``on``)."""
        with self._lock:
            return self._state == "on"

    @property
    def loop(self) -> VoiceLoop | None:
        """Direct access to the loop (for dictation tools, etc.)."""
        with self._lock:
            return self._loop

    # ── internals ──────────────────────────────────────────────────────

    def _missing_component_code(self) -> str | None:
        """Identify the missing pipeline component as a fixed error code.

        ``audio`` being ``None`` means sounddevice was not importable
        (``no-audio-library``); device failure is caught later at
        :meth:`open`.  ``wake``/``stt``/``tts`` being ``None`` means the
        corresponding backend/model could not be loaded.
        """
        if self._audio is None:
            return "no-audio-library"
        if self._wake_detector is None:
            return "wake-model-missing"
        if self._stt is None:
            return "stt-model-missing"
        if self._tts is None:
            return "tts-unavailable"
        return None

    def _fail_locked(self, code: str) -> str:
        self._state = "error"
        self._error_code = code
        return voice_error_message(code)

    def _on_loop_exit(self, reason: str | None) -> None:
        """Called by the loop thread when it exits.

        ``reason is None`` → clean exit (idle timeout / "stop listening" /
        explicit stop) → ``off``.  A string reason → the loop crashed →
        ``error`` with that fixed code.  A voice failure never kills the
        daemon: this callback only transitions state.
        """
        with self._lock:
            if reason is None:
                if self._state == "on":
                    self._state = "off"
                return
            self._state = "error"
            self._error_code = reason
