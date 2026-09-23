"""Voice pipeline interfaces (protocols and data classes).

Every hardware/OS dependency (sounddevice, openWakeWord, faster-whisper,
piper-tts, pyttsx3, pywinauto) is behind a protocol so tests use fakes.
Real implementations live in sibling modules and are lazy-imported only
when the ``voice`` extra is installed.

Audio never leaves the machine; only text goes to LLM APIs.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

# ── data classes ────────────────────────────────────────────────────────


@dataclass
class AudioSegment:
    """A chunk of raw PCM audio (float32 mono)."""

    samples: list[float] = field(default_factory=list)
    sample_rate: int = 16_000

    @property
    def duration_s(self) -> float:
        """Duration in seconds."""
        return len(self.samples) / self.sample_rate if self.sample_rate else 0.0


@dataclass
class WakeWordResult:
    """Result from a wake-word detection cycle."""

    detected: bool
    confidence: float = 0.0
    model_name: str = ""
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class STTResult:
    """Result from speech-to-text."""

    text: str
    language: str = ""
    confidence: float = 0.0
    duration_s: float = 0.0


@dataclass
class VoiceCommand:
    """Parsed voice command after STT."""

    raw_text: str
    intent: str = ""  # e.g. "open_notepad", "dictate", "stop_listening", "unknown"
    target_app: str = ""
    parameters: dict[str, str] = field(default_factory=dict)


# ── protocols ───────────────────────────────────────────────────────────


@runtime_checkable
class VoiceActivityDetector(Protocol):
    """Pure voice-activity detection over audio frames it is *given*.

    Implementations must **not** own a microphone stream.  They receive
    chunks (via the ``read_chunk`` callable supplied by the caller) and
    return the contiguous speech segment once the utterance ends — so the
    single :class:`AudioInput` stream is owned by exactly one component.
    """

    def listen_for_speech(
        self,
        read_chunk: Callable[[int], AudioSegment],
    ) -> AudioSegment:
        """Return one spoken segment, read via *read_chunk* (blocking)."""
        ...

    def close(self) -> None:
        """Release any resources held by the detector."""
        ...


@runtime_checkable
class AudioInput(Protocol):
    """Captures audio from the microphone.

    Implementations must be context-manager compatible (``with`` block)
    so tests can inject a fake stream.
    """

    def open(self, sample_rate: int = 16_000, channels: int = 1) -> None:
        """Open the audio stream."""
        ...

    def read(self, num_frames: int) -> AudioSegment:
        """Read *num_frames* samples from the stream (blocking)."""
        ...

    def close(self) -> None:
        """Close the audio stream."""
        ...

    def is_open(self) -> bool:
        """Return True if the stream is currently open."""
        ...

    def flush(self) -> None:
        """Discard all buffered audio (e.g. stale pre-prompt mic data)."""
        ...


@runtime_checkable
class WakeWordDetector(Protocol):
    """Detects a wake word (e.g. ``"hey jarvis"``) in an audio segment."""

    def detect(self, segment: AudioSegment) -> WakeWordResult:
        """Return detection result for the given audio chunk."""
        ...

    def reset(self) -> None:
        """Reset internal state (e.g. rolling buffer)."""
        ...


@runtime_checkable
class SpeechToText(Protocol):
    """Converts an audio segment to text."""

    def transcribe(self, segment: AudioSegment) -> STTResult:
        """Transcribe the audio segment."""
        ...


@runtime_checkable
class TextToSpeech(Protocol):
    """Speaks text aloud through the speakers."""

    def speak(self, text: str) -> None:
        """Speak *text* synchronously (blocks until done)."""
        ...

    def stop(self) -> None:
        """Interrupt any in-progress speech."""
        ...


@runtime_checkable
class WindowFocusChecker(Protocol):
    """Checks whether a specific application window has focus."""

    def is_foreground(self, app_name: str) -> bool:
        """Return True if *app_name* is the current foreground window."""
        ...

    def get_foreground_title(self) -> str:
        """Return the title of the current foreground window."""
        ...
