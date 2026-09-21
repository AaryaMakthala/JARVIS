"""Contract tests: every real voice class vs its Protocol (Stage 1 i).

For each real implementation the public method names and parameters must
match the Protocol's signature (via :mod:`inspect`), so a caller built
against the Protocol can drive the real class without surprises.

One *recorded* gap: :class:`SoundDeviceVAD` still owns a stream and its
``listen_for_speech`` takes no ``read_chunk`` argument.  Stage 3 migrates
it to a pure detector; the test below pins the gap explicitly instead of
pretending conformance.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from jarvis.voice.audio_input import SoundDeviceAudioInput
from jarvis.voice.fakes import FakeAudioInput
from jarvis.voice.focus import WindowFocusChecker as RealFocusChecker
from jarvis.voice.interfaces import (
    AudioInput,
    SpeechToText,
    TextToSpeech,
    VoiceActivityDetector,
    WakeWordDetector,
    WindowFocusChecker,
)
from jarvis.voice.stt import FasterWhisperSTT
from jarvis.voice.tts import PiperTTSEngine, SapiTTSEngine
from jarvis.voice.vad import SoundDeviceVAD
from jarvis.voice.wake import OpenWakeWordDetector


def _params(func: Any) -> set[str]:
    return set(inspect.signature(func).parameters)


def _protocol_methods(proto_cls: Any) -> list[str]:
    return [n for n in dir(proto_cls) if not n.startswith("_")]


CONFORMING: list[tuple[type[Any], type[Any]]] = [
    (SoundDeviceAudioInput, AudioInput),
    (OpenWakeWordDetector, WakeWordDetector),
    (FasterWhisperSTT, SpeechToText),
    (PiperTTSEngine, TextToSpeech),
    (SapiTTSEngine, TextToSpeech),
    (RealFocusChecker, WindowFocusChecker),
]


class TestRealClassesConform:
    @pytest.mark.parametrize("real_cls,proto_cls", CONFORMING)
    def test_protocol_methods_exist_and_accept_protocol_params(
        self, real_cls: type[Any], proto_cls: type[Any]
    ) -> None:
        for name in _protocol_methods(proto_cls):
            assert hasattr(real_cls, name), f"{real_cls.__name__} missing {name}"
            proto_params = _params(getattr(proto_cls, name))
            real_params = _params(getattr(real_cls, name))
            missing = {p for p in proto_params if p != "self"} - real_params
            assert not missing, (
                f"{real_cls.__name__}.{name} lacks protocol params {sorted(missing)}"
            )

    @pytest.mark.parametrize("real_cls,proto_cls", CONFORMING)
    def test_runtime_isinstance_check(self, real_cls: type[Any], proto_cls: type[Any]) -> None:
        instance = _instantiate(real_cls)
        assert isinstance(instance, proto_cls)


def _instantiate(cls: type[Any]) -> Any:
    """Build a hardware-free instance for runtime-checkable isinstance."""
    if cls is SoundDeviceAudioInput:
        return cls(stream_factory=lambda **kw: None)
    if cls is OpenWakeWordDetector:
        return cls(object())
    if cls is FasterWhisperSTT:
        return cls(object())
    if cls is PiperTTSEngine:
        return cls("test-model-path")
    if cls is SapiTTSEngine:
        return cls()
    if cls is RealFocusChecker:
        return cls()
    raise AssertionError(f"no dummy-construction implemented for {cls.__name__}")


class TestRecordedVADGap:
    """SoundDeviceVAD is intentionally NOT yet protocol-conformant."""

    def test_vad_methods_exist_but_listen_for_speech_lacks_read_chunk(self) -> None:
        assert hasattr(SoundDeviceVAD, "listen_for_speech")
        assert hasattr(SoundDeviceVAD, "close")
        # Stage 3 will make the VAD pure: chunks come via read_chunk and the
        # detector owns no stream.  Until then this gap is recorded, not hidden.
        assert "read_chunk" not in _params(SoundDeviceVAD.listen_for_speech)

    def test_vad_passes_runtime_structure_check_only(self) -> None:
        # runtime_checkable isinstance cannot see signatures — this passes
        # today, and is exactly why the inspect-level gap above must stay.
        assert isinstance(SoundDeviceVAD(object()), VoiceActivityDetector)


class TestFakeStrictness:
    def test_fake_audio_open_rejects_unknown_kwargs(self) -> None:
        # Mirrors the real library: unknown kwargs must raise TypeError.
        with pytest.raises(TypeError):
            FakeAudioInput().open(dtype="int16")
