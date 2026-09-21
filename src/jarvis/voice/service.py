"""Voice service lifecycle manager (Phase 5).

Manages the :class:`~jarvis.voice.loop.VoiceLoop` thread and provides
a clean start/stop interface for the CLI (``jarvis on/off``) and the
daemon.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from jarvis.voice.interfaces import (
    AudioInput,
    SpeechToText,
    TextToSpeech,
    WakeWordDetector,
    WindowFocusChecker,
)
from jarvis.voice.loop import DEFAULT_MAX_DICTATION_CHARS, DEFAULT_MAX_SESSION_S, VoiceLoop

logger = logging.getLogger(__name__)


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
        self._loop: VoiceLoop | None = None
        self._lock = threading.Lock()

    def start(self) -> str:
        """Start the voice loop.  Returns a status message."""
        with self._lock:
            return self._start_locked()

    def _start_locked(self) -> str:
        if self._loop is not None and self._loop.is_active():
            return "voice is already active"

        if self._audio is None or self._stt is None or self._tts is None:
            return (
                "voice dependencies not available — install with: pip install jarvis-agent[voice]"
            )

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
        )
        self._loop.start()
        return "voice activated"

    def stop(self) -> str:
        """Stop the voice loop.  Returns a status message."""
        with self._lock:
            return self._stop_locked()

    def _stop_locked(self) -> str:
        if self._loop is None or not self._loop.is_active():
            return "voice is not active"
        self._loop.stop()
        self._loop = None
        return "voice deactivated"

    def is_active(self) -> bool:
        """Return True if the voice loop is running."""
        with self._lock:
            loop = self._loop
        return loop is not None and loop.is_active()

    @property
    def loop(self) -> VoiceLoop | None:
        """Direct access to the loop (for dictation tools, etc.)."""
        with self._lock:
            return self._loop
