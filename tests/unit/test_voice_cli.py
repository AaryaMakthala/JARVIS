"""Voice CLI command tests (jarvis on/off).

Tests the CLI commands and VoiceService integration without hardware.
"""

from __future__ import annotations

from jarvis.config import Settings, VoiceSettings
from jarvis.voice.fakes import make_speech
from jarvis.voice.service import VoiceService

# ── VoiceService ────────────────────────────────────────────────────────


class TestVoiceService:
    def test_start_without_deps(self) -> None:
        """Starting without audio/stt/tts returns an error message."""
        svc = VoiceService()
        msg = svc.start()
        assert "not available" in msg
        assert not svc.is_active()

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
        from jarvis.voice.fakes import FakeAudioInput, FakeSTT, FakeTTS

        audio = FakeAudioInput()
        stt = FakeSTT()
        tts = FakeTTS()

        svc = VoiceService(audio=audio, stt=stt, tts=tts)
        svc.start()
        msg = svc.start()
        assert "already active" in msg
        svc.stop()

    def test_stop_when_not_active(self) -> None:
        svc = VoiceService()
        msg = svc.stop()
        assert "not active" in msg


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
