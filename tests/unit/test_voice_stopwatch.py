"""The speaking-state onset trigger (owner decision **D22**).

``OnsetTrigger`` is the only new piece of logic D22 adds: while JARVIS speaks, the
loop already reads the microphone for the wake word, and an utterance that is not
the wake word used to be discarded unread.  A bare ``"stop"`` has no wake-word
score to find, so the trigger watches ordinary speech energy instead.

These tests pin the state machine itself - no microphone, no engine, no loop -
because the safety property D22 rests on is *"a noise, not a person, must never
start a transcription"*.  The loop that consumes it is covered in
``test_voice_loop.TestBareStopWhileSpeaking``.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from jarvis.voice import loop as loop_mod
from jarvis.voice.interfaces import AudioSegment
from jarvis.voice.stopwatch import (
    _MIN_ONSET_FRAMES,
    _MIN_SEGMENT_S,
    BARE_CANCEL_MAX_S,
    OnsetTrigger,
)


def _speech(duration_s: float = 0.1, value: float = 0.5) -> AudioSegment:
    """A segment loud enough to pass the loop's RMS gate."""
    return AudioSegment(samples=[value] * int(duration_s * 16_000), sample_rate=16_000)


def _silence(duration_s: float = 0.1) -> AudioSegment:
    """A segment the loop's RMS gate calls silence."""
    return AudioSegment(samples=[0.0] * int(duration_s * 16_000), sample_rate=16_000)


#: The same gate the loop uses, so the trigger cannot drift from it.
_LOOP_GATE: Callable[[AudioSegment], bool] = lambda segment: loop_mod._chunk_is_speech(segment)


def test_one_speech_frame_is_not_an_onset() -> None:
    """A single loud frame is a click, a door or a creak - not a person."""
    trigger = OnsetTrigger(_LOOP_GATE)

    assert trigger.feed(_speech()) is None
    assert trigger.fired is False


def test_two_consecutive_speech_frames_fire() -> None:
    trigger = OnsetTrigger(_LOOP_GATE)

    assert trigger.feed(_speech()) is None
    onset = trigger.feed(_speech())

    assert onset is not None
    assert len(onset) == 2 * int(0.1 * 16_000)
    assert trigger.fired is True


def test_a_non_speech_frame_clears_the_run() -> None:
    """Frames must be *consecutive*: click-then-word is two onsets, not one."""
    trigger = OnsetTrigger(_LOOP_GATE)

    trigger.feed(_speech())
    assert trigger.feed(_silence()) is None
    # The run restarted, so one more frame is still not enough.
    assert trigger.feed(_speech()) is None
    assert trigger.feed(_speech()) is not None


def test_the_trigger_latches_after_firing() -> None:
    """One utterance is never transcribed twice."""
    trigger = OnsetTrigger(_LOOP_GATE)

    trigger.feed(_speech())
    assert trigger.feed(_speech()) is not None
    for _ in range(10):
        assert trigger.feed(_speech()) is None
    assert trigger.fired is True


def test_reset_re_arms_the_trigger() -> None:
    """After a non-cancel onset the mic keeps watching for a later "stop"."""
    trigger = OnsetTrigger(_LOOP_GATE)

    trigger.feed(_speech())
    assert trigger.feed(_speech()) is not None
    trigger.reset()

    assert trigger.fired is False
    assert trigger.feed(_speech()) is None
    assert trigger.feed(_speech()) is not None


def test_a_short_segment_never_starts_an_onset() -> None:
    """A stalled read carries no audio and must not look like an utterance."""
    trigger = OnsetTrigger(_LOOP_GATE)

    for _ in range(5):
        assert trigger.feed(_speech(_MIN_SEGMENT_S / 2)) is None
    assert trigger.fired is False


def test_the_onset_buffer_stays_bounded() -> None:
    """Memory is bounded by the frame count, not by how long we keep feeding."""
    trigger = OnsetTrigger(_LOOP_GATE)

    trigger.feed(_speech(duration_s=0.5))
    onset = trigger.feed(_speech(duration_s=0.5))

    assert onset is not None
    assert len(onset) == _MIN_ONSET_FRAMES * int(0.5 * 16_000)
    # Latched, so 500 further frames are dropped - and nothing accumulates.
    for _ in range(500):
        assert trigger.feed(_speech(duration_s=0.5)) is None
    assert len(trigger._frames) <= _MIN_ONSET_FRAMES


def test_a_custom_frame_count_is_honoured() -> None:
    trigger = OnsetTrigger(_LOOP_GATE, min_frames=3)

    trigger.feed(_speech())
    assert trigger.feed(_speech()) is None
    assert len(trigger.feed(_speech()) or []) == 3 * int(0.1 * 16_000)


def test_a_non_callable_gate_is_rejected() -> None:
    """Failing loudly beats a trigger that can never see speech."""
    with pytest.raises(TypeError):
        OnsetTrigger(None)  # type: ignore[arg-type]


def test_bare_cancel_max_is_shorter_than_the_in_flight_watch() -> None:
    """The speaking path shares the mic with wake barge-in, so it hands back fast."""
    assert BARE_CANCEL_MAX_S == 2.0
    assert BARE_CANCEL_MAX_S < loop_mod._STOP_CAPTURE_MAX_S
