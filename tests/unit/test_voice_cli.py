"""Voice CLI command tests (jarvis on/off).

Tests the CLI commands and VoiceService integration without hardware.
"""

from __future__ import annotations

from typing import Any

from jarvis.config import Settings, VoiceSettings
from jarvis.voice.fakes import make_speech
from jarvis.voice.service import VoiceService

# ── VoiceService ────────────────────────────────────────────────────────


class TestVoiceService:
    def test_start_without_deps_identifies_missing_library(self) -> None:
        """Starting without audio/stt/tts identifies the component (Stage 2)."""
        svc = VoiceService()
        msg = svc.start()
        assert "voice is in error: no-audio-library" in msg
        assert "pip install jarvis-agent[voice]" in msg  # useful hint
        assert not svc.is_active()
        assert svc.state == "error"
        assert svc.error_code == "no-audio-library"

    def test_start_stop_lifecycle(self) -> None:
        """Starting with fakes, then stopping."""
        import time

        from jarvis.voice.fakes import FakeAudioInput, FakeSTT, FakeTTS, FakeWakeWord
        from jarvis.voice.interfaces import STTResult, WakeWordResult

        audio = FakeAudioInput(segments=[make_speech("x", duration_s=0.5)] * 50)
        wake = FakeWakeWord(results=[WakeWordResult(detected=True)])
        stt = FakeSTT(results=[STTResult(text="hello")])
        tts = FakeTTS()

        svc = VoiceService(audio=audio, wake_detector=wake, stt=stt, tts=tts)
        msg = svc.start()
        assert msg == "voice activated"
        time.sleep(0.1)  # give thread a moment
        assert svc.is_active()

        msg = svc.stop()
        assert msg == "voice deactivated"
        assert not svc.is_active()

    def test_start_is_idempotent(self) -> None:
        from jarvis.voice.fakes import FakeAudioInput, FakeSTT, FakeTTS, FakeWakeWord

        audio = FakeAudioInput()
        wake = FakeWakeWord()
        stt = FakeSTT()
        tts = FakeTTS()

        svc = VoiceService(audio=audio, wake_detector=wake, stt=stt, tts=tts)
        svc.start()
        msg = svc.start()
        assert "already active" in msg
        assert svc.state == "on"
        svc.stop()

    def test_stop_when_not_active(self) -> None:
        svc = VoiceService()
        msg = svc.stop()
        assert "not active" in msg


# ── Stage 2: voice state machine off | starting | on | error ───────────


class _MicThatFailsToOpen:
    """AudioInput whose open() always fails — a broken/unplugged mic."""

    def __init__(self) -> None:
        self.close_calls = 0
        self.open_calls = 0

    def open(self, sample_rate: int = 16_000, channels: int = 1) -> None:
        self.open_calls += 1
        raise RuntimeError("no audio device available")

    def read(self, num_frames: int) -> Any:
        raise AssertionError("should never be read")

    def close(self) -> None:
        self.close_calls += 1

    def is_open(self) -> bool:
        return False


class _FlakyMic(_MicThatFailsToOpen):
    """Fails the first open (mic not ready), succeeds afterwards."""

    def __init__(self) -> None:
        super().__init__()
        self._fail = True

    def open(self, sample_rate: int = 16_000, channels: int = 1) -> None:
        self.open_calls += 1
        if self._fail:
            self._fail = False
            raise RuntimeError("device busy")
        from jarvis.voice.interfaces import AudioSegment

        self.segments = [AudioSegment(samples=[0.0] * 160, sample_rate=16_000)]
        self._idx = 0

    def read(self, num_frames: int) -> Any:
        from jarvis.voice.interfaces import AudioSegment

        if self._idx < len(self.segments):
            seg = self.segments[self._idx]
            self._idx += 1
            return seg
        return AudioSegment(samples=[0.0] * num_frames, sample_rate=16_000)


class TestVoiceStateMachine:
    def test_states_off_on_after_start_then_stop(self) -> None:
        from jarvis.voice.fakes import FakeAudioInput, FakeSTT, FakeTTS, FakeWakeWord

        svc = VoiceService(
            audio=FakeAudioInput(),
            wake_detector=FakeWakeWord(),
            stt=FakeSTT(),
            tts=FakeTTS(),
            idle_timeout_s=600.0,
        )
        assert svc.state == "off"
        assert svc.start() == "voice activated"
        assert svc.state == "on"
        assert svc.is_active()
        assert svc.error_code is None
        assert svc.stop() == "voice deactivated"
        assert svc.state == "off"
        assert not svc.is_active()

    def test_missing_component_codes_each_identified(self) -> None:
        from jarvis.voice.fakes import FakeAudioInput, FakeSTT, FakeTTS, FakeWakeWord

        cases: list[tuple[Any, str]] = [
            (VoiceService(), "no-audio-library"),
            (
                VoiceService(audio=FakeAudioInput(), stt=FakeSTT(), tts=FakeTTS()),
                "wake-model-missing",
            ),
            (
                VoiceService(audio=FakeAudioInput(), wake_detector=FakeWakeWord(), tts=FakeTTS()),
                "stt-model-missing",
            ),
            (
                VoiceService(audio=FakeAudioInput(), wake_detector=FakeWakeWord(), stt=FakeSTT()),
                "tts-unavailable",
            ),
        ]
        for svc, code in cases:
            msg = svc.start()
            assert f"voice is in error: {code}" in msg, code
            assert svc.state == "error"
            assert svc.error_code == code
            assert not svc.is_active()

    def test_mic_open_failure_is_synchronous_error(self) -> None:
        mic = _MicThatFailsToOpen()
        from jarvis.voice.fakes import FakeSTT, FakeTTS, FakeWakeWord

        svc = VoiceService(
            audio=mic,
            wake_detector=FakeWakeWord(),
            stt=FakeSTT(),
            tts=FakeTTS(),
        )
        msg = svc.start()
        assert "voice is in error: mic-open-failed" in msg
        assert "microphone is connected" in msg  # useful hint
        assert svc.state == "error"
        assert svc.error_code == "mic-open-failed"
        assert not svc.is_active()
        assert mic.open_calls == 1  # never retried on the same start() call

    def test_retry_after_error_starts(self) -> None:
        """`jarvis on` after an error retries: a transient mic failure then works."""
        mic = _FlakyMic()
        from jarvis.voice.fakes import FakeSTT, FakeTTS, FakeWakeWord

        svc = VoiceService(
            audio=mic,
            wake_detector=FakeWakeWord(),
            stt=FakeSTT(),
            tts=FakeTTS(),
            idle_timeout_s=600.0,
        )
        first = svc.start()
        assert "mic-open-failed" in first
        assert svc.state == "error"
        second = svc.start()
        assert second == "voice activated"
        assert svc.state == "on"
        assert svc.error_code is None
        svc.stop()
        assert svc.state == "off"

    def test_loop_crash_moves_state_to_error(self) -> None:
        import time

        from jarvis.voice.fakes import FakeSTT, FakeTTS, FakeWakeWord

        class _CrashOnRead(_MicThatFailsToOpen):
            def open(self, sample_rate: int = 16_000, channels: int = 1) -> None:
                self.open_calls += 1  # open succeeds

            def read(self, num_frames: int) -> Any:
                raise RuntimeError("device disconnected")

        audio = _CrashOnRead()
        svc = VoiceService(
            audio=audio,
            wake_detector=FakeWakeWord(results=[]),
            stt=FakeSTT(),
            tts=FakeTTS(),
        )
        assert svc.start() == "voice activated"  # thread starts, open is fine
        for _ in range(100):
            if svc.state == "error":
                break
            time.sleep(0.02)
        assert svc.state == "error"
        assert svc.error_code == "loop-crashed"
        assert not svc.is_active()
        assert audio.close_calls >= 1  # audio closed in the loop's finally

    def test_clean_idle_exit_moves_state_to_off(self) -> None:
        import time

        from jarvis.voice.fakes import FakeAudioInput, FakeSTT, FakeTTS, FakeWakeWord

        svc = VoiceService(
            audio=FakeAudioInput(),
            wake_detector=FakeWakeWord(results=[]),
            stt=FakeSTT(),
            tts=FakeTTS(),
            idle_timeout_s=0.05,
        )
        assert svc.start() == "voice activated"
        for _ in range(100):
            if svc.state == "off":
                break
            time.sleep(0.02)
        assert svc.state == "off"
        assert svc.error_code is None
        assert not svc.is_active()


# ── Config integration ──────────────────────────────────────────────────


class TestVoiceConfig:
    def test_voice_settings_defaults(self) -> None:
        settings = Settings()
        assert settings.voice.enabled is False
        assert settings.voice.wake_word == "hey_jarvis"
        assert settings.voice.stt_model == "base"
        assert settings.voice.tts_backend == "piper"
        assert settings.voice.silence_threshold == 0.01
        assert settings.voice.silence_timeout_s == 0.7
        assert settings.voice.max_segment_s == 30.0
        assert settings.voice.listen_timeout_s == 30.0
        assert settings.voice.idle_timeout_s == 120.0
        assert settings.voice.max_session_s == 1800.0
        assert settings.voice.max_dictation_chars == 20000

    def test_voice_settings_override(self) -> None:
        settings = Settings(voice=VoiceSettings(enabled=True, wake_word="custom"))
        assert settings.voice.enabled is True
        assert settings.voice.wake_word == "custom"
