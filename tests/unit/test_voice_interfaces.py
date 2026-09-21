"""Voice interface protocol contracts and fake behaviour tests.

Verifies that:
- Each protocol is correctly defined and runtime-checkable.
- Every fake implements its protocol.
- Data classes behave correctly.
"""

from __future__ import annotations

import pytest

from jarvis.voice.fakes import (
    FakeAudioInput,
    FakeFocusChecker,
    FakeSTT,
    FakeTTS,
    FakeWakeWord,
    make_silence,
    make_speech,
)
from jarvis.voice.interfaces import (
    AudioInput,
    AudioSegment,
    SpeechToText,
    STTResult,
    TextToSpeech,
    WakeWordDetector,
    WakeWordResult,
    WindowFocusChecker,
)

# ── AudioSegment ────────────────────────────────────────────────────────


class TestAudioSegment:
    def test_empty_segment_duration(self) -> None:
        seg = AudioSegment()
        assert seg.duration_s == 0.0

    def test_duration_calculation(self) -> None:
        seg = AudioSegment(samples=[0.1] * 1600, sample_rate=16_000)
        assert seg.duration_s == pytest.approx(0.1)

    def test_default_sample_rate(self) -> None:
        seg = AudioSegment(samples=[0.0] * 100)
        assert seg.sample_rate == 16_000


# ── WakeWordResult ──────────────────────────────────────────────────────


class TestWakeWordResult:
    def test_default_values(self) -> None:
        r = WakeWordResult(detected=False)
        assert r.confidence == 0.0
        assert r.model_name == ""
        assert isinstance(r.timestamp, float)

    def test_detected_with_confidence(self) -> None:
        r = WakeWordResult(detected=True, confidence=0.95, model_name="test")
        assert r.detected is True
        assert r.confidence == 0.95


# ── STTResult ───────────────────────────────────────────────────────────


class TestSTTResult:
    def test_default_values(self) -> None:
        r = STTResult(text="")
        assert r.language == ""
        assert r.confidence == 0.0
        assert r.duration_s == 0.0


# ── Protocol runtime checks ─────────────────────────────────────────────


class TestProtocolContracts:
    def test_fake_audio_is_audio_input(self) -> None:
        assert isinstance(FakeAudioInput(), AudioInput)

    def test_fake_wake_is_wake_detector(self) -> None:
        assert isinstance(FakeWakeWord(), WakeWordDetector)

    def test_fake_stt_is_speech_to_text(self) -> None:
        assert isinstance(FakeSTT(), SpeechToText)

    def test_fake_tts_is_text_to_speech(self) -> None:
        assert isinstance(FakeTTS(), TextToSpeech)

    def test_fake_focus_is_window_focus_checker(self) -> None:
        assert isinstance(FakeFocusChecker(), WindowFocusChecker)


# ── FakeAudioInput ──────────────────────────────────────────────────────


class TestFakeAudioInput:
    def test_open_close(self) -> None:
        audio = FakeAudioInput()
        assert not audio.is_open()
        audio.open()
        assert audio.is_open()
        audio.close()
        assert not audio.is_open()
        assert audio.close_calls == 1

    def test_read_returns_preloaded_segments(self) -> None:
        seg1 = AudioSegment(samples=[0.1, 0.2])
        seg2 = AudioSegment(samples=[0.3, 0.4])
        audio = FakeAudioInput(segments=[seg1, seg2])
        audio.open()
        assert audio.read(100) == seg1
        assert audio.read(100) == seg2
        # Exhausted → returns silence
        result = audio.read(100)
        assert result.samples == [0.0] * 100

    def test_read_records_calls(self) -> None:
        audio = FakeAudioInput()
        audio.open()
        audio.read(480)
        audio.read(960)
        assert audio.read_calls == [480, 960]


# ── FakeWakeWord ────────────────────────────────────────────────────────


class TestFakeWakeWord:
    def test_returns_scripted_results(self) -> None:
        wake = FakeWakeWord(
            results=[
                WakeWordResult(detected=False),
                WakeWordResult(detected=True, confidence=0.9),
            ]
        )
        r1 = wake.detect(make_silence())
        assert not r1.detected
        r2 = wake.detect(make_silence())
        assert r2.detected
        assert r2.confidence == 0.9
        # Exhausted → returns not detected
        r3 = wake.detect(make_silence())
        assert not r3.detected

    def test_reset(self) -> None:
        wake = FakeWakeWord(results=[WakeWordResult(detected=True)])
        wake.detect(make_silence())
        wake.reset()
        r = wake.detect(make_silence())
        assert r.detected  # script reset

    def test_custom_detect_fn(self) -> None:
        def always_detect(seg: AudioSegment) -> WakeWordResult:
            return WakeWordResult(detected=True, confidence=1.0)

        wake = FakeWakeWord(detect_fn=always_detect)
        r = wake.detect(make_silence())
        assert r.detected


# ── FakeSTT ─────────────────────────────────────────────────────────────


class TestFakeSTT:
    def test_returns_scripted_results(self) -> None:
        stt = FakeSTT(results=[STTResult(text="hello world")])
        r = stt.transcribe(make_speech("hello world"))
        assert r.text == "hello world"

    def test_custom_transcribe_fn(self) -> None:
        def echo(seg: AudioSegment) -> STTResult:
            return STTResult(text=f"len={len(seg.samples)}")

        stt = FakeSTT(transcribe_fn=echo)
        r = stt.transcribe(AudioSegment(samples=[0.1] * 100))
        assert r.text == "len=100"


# ── FakeTTS ─────────────────────────────────────────────────────────────


class TestFakeTTS:
    def test_records_spoken_text(self) -> None:
        tts = FakeTTS()
        tts.speak("hello")
        tts.speak("world")
        assert tts.spoken == ["hello", "world"]

    def test_stop(self) -> None:
        tts = FakeTTS()
        tts.stop()
        assert tts.stop_calls == 1


# ── FakeFocusChecker ────────────────────────────────────────────────────


class TestFakeFocusChecker:
    def test_is_foreground_match(self) -> None:
        fc = FakeFocusChecker(foreground="Notepad - test.txt")
        assert fc.is_foreground("notepad")

    def test_is_foreground_no_match(self) -> None:
        fc = FakeFocusChecker(foreground="Chrome")
        assert not fc.is_foreground("notepad")

    def test_explicit_map_overrides(self) -> None:
        fc = FakeFocusChecker(foreground="Chrome", foreground_map={"chrome": False})
        assert not fc.is_foreground("chrome")


# ── Helper functions ────────────────────────────────────────────────────


class TestHelpers:
    def test_make_silence(self) -> None:
        seg = make_silence(0.5)
        assert seg.duration_s == pytest.approx(0.5)
        assert all(s == 0.0 for s in seg.samples)

    def test_make_speech(self) -> None:
        seg = make_speech("test", duration_s=1.0)
        assert seg.duration_s == pytest.approx(1.0)
        assert all(s == 0.5 for s in seg.samples)
