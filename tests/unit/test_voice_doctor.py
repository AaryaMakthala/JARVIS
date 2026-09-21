"""Voice diagnostics (``jarvis voice doctor``).

Hermetic unit tests: the hardware/library factories are monkeypatched so the
suite runs identically with or without the voice extras installed.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from jarvis import cli, config
from jarvis.checks import Check
from jarvis.voice.doctor import run_voice_doctor
from jarvis.voice.interfaces import AudioSegment, STTResult, WakeWordResult

runner = CliRunner()


def _settings(**voice: object) -> config.Settings:
    """In-memory settings with the voice block overridable per test."""
    return config.Settings(voice={"enabled": True, "wake_word": "hey_jarvis", **voice})


class FakeAudio:
    """Stands in for SoundDeviceAudioInput during probes."""

    def __init__(self, *, raise_open: bool = False, empty: bool = False) -> None:
        self.raise_open = raise_open
        self.empty = empty
        self.opened = False
        self.closed = False

    def open(self) -> None:
        if self.raise_open:
            raise RuntimeError("mic open boom")
        self.opened = True

    def read(self, num_frames: int) -> AudioSegment:
        if self.empty:
            return AudioSegment(samples=[], sample_rate=16_000)
        return AudioSegment(samples=[0.0] * int(num_frames), sample_rate=16_000)

    def close(self) -> None:
        self.closed = True


class FakeWake:
    """Stands in for OpenWakeWordDetector during probes."""

    def __init__(self, *, raise_detect: bool = False) -> None:
        self.raise_detect = raise_detect
        self.reset_calls = 0
        self.detect_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def detect(self, segment: AudioSegment) -> WakeWordResult:
        self.detect_calls += 1
        if self.raise_detect:
            raise RuntimeError("wake predict boom")
        return WakeWordResult(detected=False, confidence=0.01, model_name="hey_jarvis")


class FakeSTT:
    """Stands in for FasterWhisperSTT during probes."""

    def __init__(self, *, raise_transcribe: bool = False) -> None:
        self.raise_transcribe = raise_transcribe

    def transcribe(self, segment: AudioSegment) -> STTResult:
        if self.raise_transcribe:
            raise RuntimeError("stt boom")
        return STTResult(text="", language="en", duration_s=segment.duration_s)


def _patch_factories(
    monkeypatch: pytest.MonkeyPatch,
    *,
    audio: FakeAudio | None = None,
    wake: FakeWake | None = None,
    stt: FakeSTT | None = None,
    tts: object | None = None,
) -> None:
    """Route every backend factory to a fake (or None for "not installed")."""
    monkeypatch.setattr("jarvis.voice.audio_input.create", lambda *a, **k: audio)
    monkeypatch.setattr("jarvis.voice.wake.create", lambda *a, **k: wake)
    monkeypatch.setattr("jarvis.voice.stt.create", lambda *a, **k: stt)
    monkeypatch.setattr("jarvis.voice.tts.create", lambda *a, **k: tts)
    monkeypatch.setattr("jarvis.voice.doctor._input_device_name", lambda dev: ("test mic", None))


def _by_name(checks: list[Check]) -> dict[str, Check]:
    return {c.name: c for c in checks}


def test_all_backends_missing_reports_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_factories(monkeypatch)
    checks = _by_name(run_voice_doctor(_settings()))

    assert checks["voice_config"].status == "PASS"
    assert checks["audio_library"].status == "FAIL"
    assert checks["wake_model"].status == "FAIL"
    assert checks["stt_model"].status == "FAIL"
    assert checks["tts"].status == "FAIL"
    # the failing library short-circuits its deeper probes
    assert "mic_device" not in checks and "mic_open" not in checks


def test_full_pass_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_factories(monkeypatch, audio=FakeAudio(), wake=FakeWake(), stt=FakeSTT(), tts=object())
    checks = _by_name(run_voice_doctor(_settings()))

    for name in (
        "voice_config",
        "audio_library",
        "mic_device",
        "mic_open",
        "wake_model",
        "wake_detection",
        "stt_model",
        "tts",
    ):
        assert checks[name].status == "PASS", f"{name} should PASS: {checks[name]}"
    assert "16000 samples" in checks["mic_open"].detail
    assert "silent probe transcribed" in checks["stt_model"].detail


def test_mic_open_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_factories(monkeypatch, audio=FakeAudio(raise_open=True), wake=FakeWake())
    checks = _by_name(run_voice_doctor(_settings()))
    assert checks["mic_open"].status == "FAIL"
    assert checks["wake_model"].status == "PASS"  # other probes still run


def test_mic_returns_no_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_factories(monkeypatch, audio=FakeAudio(empty=True), wake=FakeWake())
    checks = _by_name(run_voice_doctor(_settings()))
    assert checks["mic_open"].status == "FAIL"
    assert "no audio" in checks["mic_open"].detail


def test_wake_detection_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_factories(monkeypatch, audio=FakeAudio(), wake=FakeWake(raise_detect=True))
    checks = _by_name(run_voice_doctor(_settings()))
    assert "wake_model" not in checks  # model presence is not confirmed
    assert checks["wake_detection"].status == "FAIL"


def test_stt_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_factories(
        monkeypatch,
        audio=FakeAudio(),
        wake=FakeWake(),
        stt=FakeSTT(raise_transcribe=True),
    )
    checks = _by_name(run_voice_doctor(_settings()))
    assert checks["stt_model"].status == "FAIL"


def test_voice_disabled_is_a_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_factories(monkeypatch, wake=FakeWake())
    checks = _by_name(run_voice_doctor(_settings(enabled=False)))
    assert checks["voice_config"].status == "WARN"


def test_probes_can_be_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_factories(monkeypatch, wake=FakeWake())
    checks = _by_name(run_voice_doctor(_settings(), mic=False, stt=False, tts=False))
    assert "audio_library" not in checks and "mic_open" not in checks
    assert "stt_model" not in checks and "tts" not in checks
    assert checks["wake_model"].status == "PASS"


def test_unset_stt_model_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_factories(monkeypatch, wake=FakeWake())
    checks = _by_name(run_voice_doctor(_settings(stt_model="")))
    assert checks["stt_model"].status == "FAIL"


def test_cli_renders_table_and_fail_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_doctor(**kwargs: object) -> list[Check]:
        return [Check("ok_one", "PASS", "fine"), Check("bad_one", "FAIL", "broken")]

    monkeypatch.setattr("jarvis.voice.doctor.run_voice_doctor", fake_doctor)
    result = runner.invoke(cli.app, ["voice", "doctor"])
    assert result.exit_code == 1
    assert "bad_one" in result.output and "broken" in result.output


def test_cli_passes_when_all_green(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_doctor(**kwargs: object) -> list[Check]:
        return [Check("ok_one", "PASS", "fine")]

    monkeypatch.setattr("jarvis.voice.doctor.run_voice_doctor", fake_doctor)
    result = runner.invoke(cli.app, ["voice", "doctor"])
    assert result.exit_code == 0
    assert "ok_one" in result.output
