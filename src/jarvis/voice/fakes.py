"""Deterministic fake voice backends for tests (no hardware, no network).

Every class implements the corresponding protocol from
:mod:`jarvis.voice.interfaces` so the voice loop, tools, and daemon
can be tested without microphones, speakers, or model files.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from jarvis.voice.interfaces import (
    AudioSegment,
    STTResult,
    VoiceCommand,
    WakeWordResult,
)


class FakeAudioInput:
    """Deterministic audio source: returns pre-loaded segments in order."""

    def __init__(
        self,
        segments: list[AudioSegment] | None = None,
        *,
        sample_rate: int = 16_000,
    ) -> None:
        self._segments = list(segments or [])
        self._idx = 0
        self._sample_rate = sample_rate
        self._open = False
        self.open_calls: list[dict[str, Any]] = []
        self.read_calls: list[int] = []
        self.close_calls: int = 0
        self.flush_calls: int = 0

    def open(self, sample_rate: int = 16_000, channels: int = 1) -> None:
        self._open = True
        self.open_calls.append({"sample_rate": sample_rate, "channels": channels})

    def read(self, num_frames: int) -> AudioSegment:
        self.read_calls.append(num_frames)
        if self._idx < len(self._segments):
            seg = self._segments[self._idx]
            self._idx += 1
            return seg
        return AudioSegment(samples=[0.0] * num_frames, sample_rate=self._sample_rate)

    def close(self) -> None:
        self._open = False
        self.close_calls += 1

    def is_open(self) -> bool:
        return self._open

    def flush(self) -> None:
        self.flush_calls += 1


class FakeWakeWord:
    """Returns a scripted sequence of wake-word results."""

    def __init__(
        self,
        results: list[WakeWordResult] | None = None,
        *,
        detect_fn: Callable[[AudioSegment], WakeWordResult] | None = None,
    ) -> None:
        self._results = list(results or [])
        self._idx = 0
        self._detect_fn = detect_fn
        self.detect_calls: list[AudioSegment] = []
        self.reset_calls: int = 0

    def detect(self, segment: AudioSegment) -> WakeWordResult:
        self.detect_calls.append(segment)
        if self._detect_fn is not None:
            return self._detect_fn(segment)
        if self._idx < len(self._results):
            result = self._results[self._idx]
            self._idx += 1
            return result
        return WakeWordResult(detected=False)

    def reset(self) -> None:
        self._idx = 0
        self.reset_calls += 1


class FakeSTT:
    """Returns scripted transcription results."""

    def __init__(
        self,
        results: list[STTResult] | None = None,
        *,
        transcribe_fn: Callable[[AudioSegment], STTResult] | None = None,
    ) -> None:
        self._results = list(results or [])
        self._idx = 0
        self._transcribe_fn = transcribe_fn
        self.transcribe_calls: list[AudioSegment] = []

    def transcribe(self, segment: AudioSegment) -> STTResult:
        self.transcribe_calls.append(segment)
        if self._transcribe_fn is not None:
            return self._transcribe_fn(segment)
        if self._idx < len(self._results):
            result = self._results[self._idx]
            self._idx += 1
            return result
        return STTResult(text="")


class FakeTTS:
    """Records spoken text without actually producing audio."""

    def __init__(
        self,
        speak_fn: Callable[[str], None] | None = None,
    ) -> None:
        self._speak_fn = speak_fn
        self.spoken: list[str] = []
        self.stop_calls: int = 0

    def speak(self, text: str) -> None:
        self.spoken.append(text)
        if self._speak_fn is not None:
            self._speak_fn(text)

    def stop(self) -> None:
        self.stop_calls += 1


class FakeFocusChecker:
    """Deterministic window focus checker for tests."""

    def __init__(
        self,
        foreground: str = "",
        *,
        foreground_map: dict[str, bool] | None = None,
    ) -> None:
        self._foreground = foreground
        self._map = foreground_map or {}
        self.is_foreground_calls: list[str] = []
        self.get_foreground_title_calls: int = 0

    def set_foreground(self, app: str) -> None:
        """Change which app is 'foreground' (test helper)."""
        self._foreground = app

    def is_foreground(self, app_name: str) -> bool:
        self.is_foreground_calls.append(app_name)
        if app_name in self._map:
            return self._map[app_name]
        return app_name.lower() in self._foreground.lower()

    def get_foreground_title(self) -> str:
        self.get_foreground_title_calls += 1
        return self._foreground


class FakeVoiceCommandParser:
    """Parses text into VoiceCommand with a scripted sequence."""

    def __init__(
        self,
        commands: list[VoiceCommand] | None = None,
    ) -> None:
        self._commands = list(commands or [])
        self._idx = 0
        self.parse_calls: list[str] = []

    def parse(self, text: str) -> VoiceCommand:
        self.parse_calls.append(text)
        if self._idx < len(self._commands):
            cmd = self._commands[self._idx]
            self._idx += 1
            return cmd
        return VoiceCommand(raw_text=text, intent="unknown")


def make_silence(duration_s: float = 1.0, sample_rate: int = 16_000) -> AudioSegment:
    """Create a silence segment of the given duration."""
    n = int(duration_s * sample_rate)
    return AudioSegment(samples=[0.0] * n, sample_rate=sample_rate)


def make_speech(text: str, duration_s: float = 2.0, sample_rate: int = 16_000) -> AudioSegment:
    """Create a fake 'speech' segment (non-zero samples)."""
    n = int(duration_s * sample_rate)
    return AudioSegment(samples=[0.5] * n, sample_rate=sample_rate)
