"""Dictation and type_text tool tests (no hardware, no Windows GUI).

Tests tool registration, args validation, dry-run, error paths, and
the fake implementations.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.tools.base import ToolContext
from jarvis.tools.dictation import (
    SaveDictationArgs,
    StartDictationArgs,
    StopDictationArgs,
    make_save_dictation_spec,
    make_start_dictation_spec,
    make_stop_dictation_spec,
)
from jarvis.tools.keyboard import TypeTextArgs, make_type_text_spec
from jarvis.tools.registry import build_default_registry

# ── Registration ────────────────────────────────────────────────────────


class TestToolRegistration:
    def test_new_tools_in_registry(self) -> None:
        registry = build_default_registry()
        expected = {"type_text", "start_dictation", "stop_dictation", "save_dictation"}
        assert expected.issubset(set(registry.names()))

    def test_type_text_spec_properties(self) -> None:
        spec = make_type_text_spec()
        assert spec.name == "type_text"
        assert spec.base_tier == 1
        assert spec.windows_only is True

    def test_start_dictation_spec_properties(self) -> None:
        spec = make_start_dictation_spec()
        assert spec.name == "start_dictation"
        assert spec.base_tier == 1

    def test_stop_dictation_spec_properties(self) -> None:
        spec = make_stop_dictation_spec()
        assert spec.name == "stop_dictation"
        assert spec.base_tier == 0

    def test_save_dictation_spec_properties(self) -> None:
        spec = make_save_dictation_spec()
        assert spec.name == "save_dictation"
        assert spec.base_tier == 1
        assert "path" in spec.path_args


# ── Args validation ─────────────────────────────────────────────────────


class TestDictationArgs:
    def test_start_dictation_defaults(self) -> None:
        args = StartDictationArgs()
        assert args.target_app == "notepad"

    def test_start_dictation_custom_app(self) -> None:
        args = StartDictationArgs(target_app="chrome")
        assert args.target_app == "chrome"

    def test_stop_dictation_args(self) -> None:
        args = StopDictationArgs()
        assert args.model_dump() == {}

    def test_save_dictation_requires_path(self) -> None:
        with pytest.raises(Exception, match="String should have at least 1 character"):
            SaveDictationArgs(path="")

    def test_type_text_min_length(self) -> None:
        with pytest.raises(Exception, match="String should have at least 1 character"):
            TypeTextArgs(text="", target_app="notepad")

    def test_type_text_max_length(self) -> None:
        with pytest.raises(Exception, match="String should have at most 2000 characters"):
            TypeTextArgs(text="x" * 2001, target_app="notepad")

    def test_type_text_valid(self) -> None:
        args = TypeTextArgs(text="hello", target_app="notepad")
        assert args.text == "hello"


# ── Dry-run ─────────────────────────────────────────────────────────────


class TestDictationDryRun:
    def test_start_dictation_dry_run(self) -> None:
        spec = make_start_dictation_spec()
        ctx = ToolContext(settings=None, dry_run=True)  # type: ignore[arg-type]
        result = spec.execute(StartDictationArgs(), ctx)
        assert result.ok
        assert "[dry-run]" in result.output

    def test_stop_dictation_dry_run(self) -> None:
        spec = make_stop_dictation_spec()
        ctx = ToolContext(settings=None, dry_run=True)  # type: ignore[arg-type]
        result = spec.execute(StopDictationArgs(), ctx)
        assert result.ok
        assert "[dry-run]" in result.output

    def test_type_text_dry_run(self) -> None:
        spec = make_type_text_spec()
        ctx = ToolContext(settings=None, dry_run=True)  # type: ignore[arg-type]
        result = spec.execute(TypeTextArgs(text="hi", target_app="notepad"), ctx)
        assert result.ok
        assert "[dry-run]" in result.output


# ── Error paths ─────────────────────────────────────────────────────────


class TestDictationErrors:
    def test_type_text_rejects_unknown_app(self, tmp_path: object) -> None:
        from jarvis.config import Settings

        spec = make_type_text_spec()
        ctx = ToolContext(settings=Settings(), dry_run=False)
        result = spec.run(TypeTextArgs(text="hi", target_app="unknown_app"), ctx)
        assert not result.ok
        assert "not in the apps allowlist" in result.error

    def test_start_dictation_rejects_unknown_app(self) -> None:
        from jarvis.config import Settings

        spec = make_start_dictation_spec()
        ctx = ToolContext(settings=Settings(), dry_run=False)
        result = spec.run(StartDictationArgs(target_app="nonexistent"), ctx)
        assert not result.ok
        assert "not in the apps allowlist" in result.error


# ── Describe ────────────────────────────────────────────────────────────


class TestDictationDescribe:
    def test_start_dictation_describe(self) -> None:
        spec = make_start_dictation_spec()
        desc = spec.describe(StartDictationArgs(target_app="notepad"))
        assert "notepad" in desc

    def test_stop_dictation_describe(self) -> None:
        spec = make_stop_dictation_spec()
        desc = spec.describe(StopDictationArgs())
        assert "stop" in desc.lower()

    def test_type_text_describe(self) -> None:
        spec = make_type_text_spec()
        desc = spec.describe(TypeTextArgs(text="hello world", target_app="notepad"))
        assert "11" in desc  # length of "hello world"
        assert "notepad" in desc


# ── F3: save_dictation writes only to resolved, policy-checked paths ────


class TestDictationSavePathPolicy:
    def _allow_only(self, root: str) -> object:
        from jarvis.config import PolicySettings, Settings

        return Settings(policy=PolicySettings(allowed_roots=[root]))

    def _install_gui_fakes(self, monkeypatch: object, content: str = "hello world") -> None:
        """Fake pyperclip/pywinauto/pyautogui so the save tool runs anywhere."""
        import sys
        from types import SimpleNamespace

        win = SimpleNamespace()
        win.set_focus = lambda: None
        desktop = SimpleNamespace(windows=lambda **kwargs: [win])

        def _paste() -> str:
            return content

        monkeypatch.setitem(
            sys.modules,
            "pywinauto",
            SimpleNamespace(Desktop=lambda backend="uia": desktop),
        )
        monkeypatch.setitem(
            sys.modules, "pyperclip", SimpleNamespace(paste=_paste, copy=lambda value: None)
        )
        monkeypatch.setitem(sys.modules, "pyautogui", SimpleNamespace(hotkey=lambda *keys: None))

    def test_save_writes_to_resolved_path_inside_roots(
        self, tmp_path: object, monkeypatch: object
    ) -> None:
        self._install_gui_fakes(monkeypatch)
        spec = make_save_dictation_spec()
        ctx = ToolContext(settings=self._allow_only(str(tmp_path)), dry_run=False)  # type: ignore[arg-type]
        target = tmp_path / "drafts" / "note.txt"  # type: ignore[operator]
        result = spec.run_verified(SaveDictationArgs(path=str(target)), ctx)
        assert result.ok, result.error
        assert result.data["path"] == str(target.resolve())  # type: ignore[attr-defined]
        assert result.verified is True
        assert Path(target).read_text("utf-8") == "hello world"

    def test_save_outside_roots_rejected_before_gui(self, tmp_path: object) -> None:
        spec = make_save_dictation_spec()
        ctx = ToolContext(settings=self._allow_only(str(tmp_path)), dry_run=False)  # type: ignore[arg-type]
        outside = tmp_path.parent / "outside.txt"  # type: ignore[attr-defined]
        result = spec.run(SaveDictationArgs(path=str(outside)), ctx)
        assert not result.ok
        assert "outside the allowed roots" in result.error

    def test_save_escape_via_symlink_rejected(self, tmp_path: object, monkeypatch: object) -> None:
        """A symlink inside a root pointing outside must not widen the root."""
        import os

        if not hasattr(os, "symlink"):
            pytest.skip("symlink not available on this platform")
        link = tmp_path / "link"  # type: ignore[operator]
        try:
            os.symlink(str(tmp_path.parent), str(link))  # type: ignore[attr-defined]
        except OSError:
            pytest.skip("cannot create symlinks on this platform")
        spec = make_save_dictation_spec()
        ctx = ToolContext(settings=self._allow_only(str(tmp_path)), dry_run=False)  # type: ignore[arg-type]
        result = spec.run(SaveDictationArgs(path=str(link / "secret.txt")), ctx)
        assert not result.ok
        assert result.error


# ── F13: dictation tools behave honestly without a live voice loop ───────


class TestDictationVoiceBridge:
    def test_start_honest_when_no_voice_loop(self, monkeypatch: object) -> None:
        from jarvis.config import Settings

        opened: list[object] = []
        monkeypatch.setattr(  # type: ignore[attr-defined]
            "subprocess.Popen", lambda cmd, **kwargs: opened.append(cmd)
        )
        spec = make_start_dictation_spec()
        ctx = ToolContext(settings=Settings(), dry_run=False)
        result = spec.run(StartDictationArgs(), ctx)
        assert result.ok, result.error
        assert opened  # the app was opened first
        assert result.data["dictation_active"] is False
        assert "not active" in result.output

    def test_start_wires_active_loop(self, monkeypatch: object) -> None:
        from types import SimpleNamespace

        from jarvis.agent.context import VoiceToolsFacade
        from jarvis.config import Settings

        monkeypatch.setattr(  # type: ignore[attr-defined]
            "subprocess.Popen", lambda cmd, **kwargs: None
        )
        started: list[str] = []
        loop = SimpleNamespace(start_dictation=lambda target: started.append(target))
        svc = SimpleNamespace(is_active=lambda: True, loop=loop)
        ctx = ToolContext(settings=Settings(), dry_run=False, voice=VoiceToolsFacade(service=svc))
        spec = make_start_dictation_spec()
        result = spec.run(StartDictationArgs(target_app="notepad"), ctx)
        assert result.ok, result.error
        assert result.data["dictation_active"] is True
        assert started == ["notepad"]

    def test_stop_honest_when_no_voice_loop(self) -> None:
        from jarvis.config import Settings

        spec = make_stop_dictation_spec()
        ctx = ToolContext(settings=Settings(), dry_run=False)
        result = spec.run(StopDictationArgs(), ctx)
        assert result.ok
        assert "no active voice loop" in result.output
        assert result.data["dictation_active"] is False

    def test_stop_wires_active_loop(self) -> None:
        from types import SimpleNamespace

        from jarvis.agent.context import VoiceToolsFacade
        from jarvis.config import Settings

        stopped: list[bool] = []
        loop = SimpleNamespace(stop_dictation=lambda: stopped.append(True))
        svc = SimpleNamespace(is_active=lambda: True, loop=loop)
        ctx = ToolContext(settings=Settings(), dry_run=False, voice=VoiceToolsFacade(service=svc))
        spec = make_stop_dictation_spec()
        result = spec.run(StopDictationArgs(), ctx)
        assert result.ok
        assert stopped == [True]
        assert result.data["dictation_active"] is False
