"""Tests for the one-shot voice slow-response notice (Stage 5a sub-step 10).

Uses only a fake TTS and a fake clock: no real audio, no daemon, no LLM.
"""

from __future__ import annotations

from jarvis.voice.fakes import FakeTTS
from jarvis.voice.service import VoiceService
from jarvis.voice.slow_notice import (
    SLOW_NOTICE_TEXT,
    SLOW_THRESHOLD_S,
    maybe_speak,
)


class _FakeClock:
    """Deterministic monotonic-style clock under the test's control."""

    def __init__(self, now: float = 0.0) -> None:
        self._now = now

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


def test_default_threshold_is_four_seconds() -> None:
    assert SLOW_THRESHOLD_S == 4.0


def test_below_threshold_is_silent() -> None:
    tts = FakeTTS()
    assert maybe_speak(tts, 0.0, clock=_FakeClock(now=3.99)) is False
    assert tts.spoken == []


def test_speaks_the_exact_phrase_at_threshold() -> None:
    tts = FakeTTS()
    assert maybe_speak(tts, 0.0, clock=_FakeClock(now=4.0)) is True
    assert tts.spoken == ["This is taking a moment"]


def test_custom_threshold_is_honoured() -> None:
    tts = FakeTTS()
    assert maybe_speak(tts, 0.0, threshold=2.0, clock=_FakeClock(now=2.0)) is True
    assert tts.spoken == [SLOW_NOTICE_TEXT]


def test_one_shot_per_task() -> None:
    tts = FakeTTS()
    clock = _FakeClock()
    clock.advance(10.0)
    assert maybe_speak(tts, 0.0, clock=clock) is True
    clock.advance(30.0)
    assert maybe_speak(tts, 0.0, already_spoken=True, clock=clock) is False
    assert tts.spoken == ["This is taking a moment"]


def test_suppressed_while_confirmation_owns_mic() -> None:
    tts = FakeTTS()
    spoken = maybe_speak(tts, 0.0, confirmation_owns_mic=True, clock=_FakeClock(now=10.0))
    assert spoken is False
    assert tts.spoken == []


def test_suppressed_while_clarification_owns_mic() -> None:
    tts = FakeTTS()
    spoken = maybe_speak(tts, 0.0, clarification_owns_mic=True, clock=_FakeClock(now=10.0))
    assert spoken is False
    assert tts.spoken == []


def test_suppressed_when_slot_already_done() -> None:
    tts = FakeTTS()
    assert maybe_speak(tts, 0.0, done=True, clock=_FakeClock(now=10.0)) is False
    assert tts.spoken == []


def test_suppressed_while_voice_stopping() -> None:
    tts = FakeTTS()
    assert maybe_speak(tts, 0.0, stopping=True, clock=_FakeClock(now=10.0)) is False
    assert tts.spoken == []


def test_none_tts_is_silent() -> None:
    assert maybe_speak(None, 0.0, clock=_FakeClock(now=10.0)) is False


def test_voice_service_exposes_the_pipeline_tts() -> None:
    tts = FakeTTS()
    assert VoiceService(tts=tts).tts is tts
    assert VoiceService().tts is None
