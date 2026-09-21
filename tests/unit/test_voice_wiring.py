"""Daemon voice wiring regression (Stage 1 h).

The daemon must hand the voice loop a real :class:`AudioInput` — never the
VAD, which owns its own ``sd.InputStream`` and cannot share a microphone
with the loop (Phase 5 single-owner rule).

This is the exact regression that produced the original crash: the server
from ``756b953^`` (``326727b``) built ``audio = create_vad(...)`` and passed
a :class:`SoundDeviceVAD` into :class:`VoiceService` as the ``audio``
argument.  The test below extracts that old method verbatim from git and
shows its output violates the current ``AudioInput`` contract; the current
server satisfies it.
"""

from __future__ import annotations

import inspect
import logging
import subprocess
import textwrap
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from jarvis.config import Settings, VoiceSettings
from jarvis.daemon.server import DaemonServer
from jarvis.voice.audio_input import SoundDeviceAudioInput
from jarvis.voice.fakes import FakeFocusChecker, FakeSTT, FakeTTS, FakeWakeWord
from jarvis.voice.interfaces import AudioInput
from jarvis.voice.vad import SoundDeviceVAD

_TOKEN = "voicetest-token-000000000000"


class _FakeStore:
    def __init__(self) -> None:
        self._kv: dict[str, str] = {"ipc_token": _TOKEN}

    def get(self, name: str) -> str | None:
        return self._kv.get(name)

    def set(self, name: str, value: str) -> None:
        self._kv[name] = value

    def has(self, name: str) -> bool:
        return name in self._kv

    def check_store_access(self) -> str:
        return "FakeStore"


def _patch_factories(monkeypatch: Any) -> None:
    """Route every lazy-imported voice backend factory to a fake."""
    monkeypatch.setattr(
        "jarvis.voice.audio_input.create",
        lambda **kw: SoundDeviceAudioInput(stream_factory=lambda **kw2: None),
    )
    monkeypatch.setattr("jarvis.voice.wake.create", lambda **kw: FakeWakeWord())
    monkeypatch.setattr("jarvis.voice.stt.create", lambda **kw: FakeSTT())
    monkeypatch.setattr("jarvis.voice.tts.create", lambda **kw: FakeTTS())
    monkeypatch.setattr("jarvis.voice.focus.WindowFocusChecker", FakeFocusChecker)


def _extract_method(src: str, name: str) -> str:
    """Slice ``def name(self)`` (4-space indent) out of a server.py source."""
    lines = src.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"    def {name}("))
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("    def ") and not lines[i].startswith("        "):
            end = i
            break
    return "\n".join(lines[start:end])


def _load_legacy_voice_builder(tmp_path: Path) -> Callable[[Any], Any]:
    """Load ``_build_voice_service`` verbatim from the pre-fix server commit.

    ``756b953^`` == ``326727b`` (the daemon-managed voice commit that wired
    ``audio=create_vad(...)``).  The body is exec'd so the regression runs
    the committed code, not a recollection of it.
    """
    blob = subprocess.check_output(
        ["git", "show", "326727b:src/jarvis/daemon/server.py"]
    )
    method = textwrap.dedent(_extract_method(blob.decode("utf-8"), "_build_voice_service"))
    namespace: dict[str, Any] = {"logger": logging.getLogger("legacy-server")}
    exec(method, namespace)  # noqa: S102  # deliberate: run the committed old source
    return namespace["_build_voice_service"]


class TestCurrentWiring:
    def test_built_audio_is_audio_input_and_never_a_vad(self, monkeypatch: Any) -> None:
        _patch_factories(monkeypatch)
        server = DaemonServer(
            settings=Settings(voice=VoiceSettings(enabled=True)), store=_FakeStore()
        )
        service = server._build_voice_service()
        audio = service._audio
        assert audio is not None
        # Runtime structural check against the AudioInput protocol.
        assert isinstance(audio, AudioInput)
        assert not isinstance(audio, SoundDeviceVAD)
        # Parameter-list comparison vs the protocol, method by method.
        for name in ("open", "read", "close", "is_open"):
            protocol_params = inspect.signature(getattr(AudioInput, name)).parameters
            real_params = inspect.signature(getattr(type(audio), name)).parameters
            missing = {p for p in protocol_params if p != "self"} - set(real_params)
            assert not missing, f"{type(audio).__name__}.{name} lacks {sorted(missing)}"

    def test_voice_service_holds_no_sounddevice_vad_instance(self, monkeypatch: Any) -> None:
        _patch_factories(monkeypatch)
        service = DaemonServer(
            settings=Settings(voice=VoiceSettings(enabled=True)), store=_FakeStore()
        )._build_voice_service()
        for attr in ("_audio", "_wake_detector", "_stt", "_tts"):
            comp = getattr(service, attr)
            assert not isinstance(comp, SoundDeviceVAD), f"{attr} is a VAD"


class TestLegacyWiringRegression:
    def test_legacy_audio_was_a_vad_and_violates_audio_input(self, monkeypatch: Any, tmp_path: Path) -> None:
        """The pre-fix server passed a VAD as the loop's audio — must never
        satisfy today's AudioInput contract (that is why it crashed)."""
        _patch_factories(monkeypatch)
        monkeypatch.setattr(
            "jarvis.voice.vad.create", lambda **kw: SoundDeviceVAD(object())
        )
        builder = _load_legacy_voice_builder(tmp_path)
        legacy_self = SimpleNamespace(
            _settings=Settings(voice=VoiceSettings(enabled=True)),
            _voice_submit=lambda text, source="voice": None,
        )
        service = builder(legacy_self)
        legacy_audio = service._audio
        assert legacy_audio is not None
        # Regression recorded: the old wiring produced a VAD, which the loop
        # cannot read from (no open/read/is_open) → not an AudioInput.
        assert isinstance(legacy_audio, SoundDeviceVAD)
        assert not hasattr(legacy_audio, "open")
        assert not hasattr(legacy_audio, "read")
        assert not isinstance(legacy_audio, AudioInput)