"""Contract tests: every real voice class vs its Protocol.

For each real implementation the public method names and parameters must
match the Protocol's signature (via :mod:`inspect`), so a caller built
against the Protocol can drive the real class without surprises.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from jarvis.voice.audio_input import SoundDeviceAudioInput
from jarvis.voice.fakes import FakeAsyncTTS, FakeAudioInput
from jarvis.voice.focus import WindowFocusChecker as RealFocusChecker
from jarvis.voice.interfaces import (
    AudioInput,
    SpeechToText,
    TextToSpeech,
    WakeWordDetector,
    WindowFocusChecker,
)
from jarvis.voice.stt import FasterWhisperSTT
from jarvis.voice.tts import PiperTTSEngine, Pyttsx3TTSEngine, SapiTTSEngine
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
    (Pyttsx3TTSEngine, TextToSpeech),
    (FakeAudioInput, AudioInput),
    (FakeAsyncTTS, TextToSpeech),
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
    if cls in (SapiTTSEngine, Pyttsx3TTSEngine, FakeAsyncTTS):
        return cls()
    if cls is FakeAudioInput:
        return cls()
    if cls is RealFocusChecker:
        return cls()
    raise AssertionError(f"no dummy-construction implemented for {cls.__name__}")


class TestBargeInCapability:
    """The barge-in pair is an optional capability, not part of the protocol.

    ``TextToSpeech`` only requires ``speak``/``stop``, so an engine without the
    pair is still conformant — it just cannot be interrupted.  The loop probes
    for it, so the pair must exist on the engine that ships interruptible
    speech, and must not be required of the others.
    """

    def test_only_the_sapi_engine_is_interruptible(self) -> None:
        assert hasattr(SapiTTSEngine, "start_speaking")
        assert hasattr(SapiTTSEngine, "is_speaking")
        # Documented non-capable engines: still conformant, blocking only.
        for cls in (PiperTTSEngine, Pyttsx3TTSEngine):
            assert not hasattr(cls, "start_speaking"), f"{cls.__name__} grew a barge-in pair"


class TestFakeStrictness:
    def test_fake_audio_open_rejects_unknown_kwargs(self) -> None:
        # Mirrors the real library: unknown kwargs must raise TypeError.
        with pytest.raises(TypeError):
            FakeAudioInput().open(dtype="int16")
