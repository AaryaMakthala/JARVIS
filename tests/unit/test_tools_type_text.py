"""type_text tool tests (focus verification, clipboard, error paths).

The real type_text requires Windows GUI (pywinauto, ctypes, pyautogui).
These tests verify the args model, describe, dry-run, and the app-allowlist
check.  Full GUI tests are marked @pytest.mark.windows_only.
"""

from __future__ import annotations

import pytest

from jarvis.config import Settings
from jarvis.tools.base import ToolContext
from jarvis.tools.keyboard import TypeTextArgs, make_type_text_spec

# ── Args model ──────────────────────────────────────────────────────────


class TestTypeTextArgs:
    def test_valid_args(self) -> None:
        args = TypeTextArgs(text="hello", target_app="notepad")
        assert args.text == "hello"
        assert args.target_app == "notepad"

    def test_empty_text_rejected(self) -> None:
        with pytest.raises(Exception, match="at least 1 character"):
            TypeTextArgs(text="", target_app="notepad")

    def test_text_max_length(self) -> None:
        args = TypeTextArgs(text="x" * 2000, target_app="notepad")
        assert len(args.text) == 2000

    def test_text_too_long(self) -> None:
        with pytest.raises(Exception, match="at most 2000 characters"):
            TypeTextArgs(text="x" * 2001, target_app="notepad")

    def test_empty_target_app_rejected(self) -> None:
        with pytest.raises(Exception, match="at least 1 character"):
            TypeTextArgs(text="hello", target_app="")


# ── Tool spec ───────────────────────────────────────────────────────────


class TestTypeTextSpec:
    def test_properties(self) -> None:
        spec = make_type_text_spec()
        assert spec.name == "type_text"
        assert spec.base_tier == 1
        assert spec.windows_only is True
        assert spec.timeout_s == 10
        assert spec.verify is not None

    def test_describe(self) -> None:
        spec = make_type_text_spec()
        desc = spec.describe(TypeTextArgs(text="test", target_app="notepad"))
        assert "4" in desc
        assert "notepad" in desc

    def test_dry_run(self) -> None:
        spec = make_type_text_spec()
        ctx = ToolContext(settings=Settings(), dry_run=True)
        result = spec.execute(TypeTextArgs(text="hi", target_app="notepad"), ctx)
        assert result.ok
        assert "[dry-run]" in result.output


# ── Allowlist check (no Windows GUI needed) ─────────────────────────────


class TestTypeTextAllowlist:
    def test_rejects_unknown_app(self) -> None:
        spec = make_type_text_spec()
        ctx = ToolContext(settings=Settings(), dry_run=False)
        result = spec.run(TypeTextArgs(text="hi", target_app="unknown"), ctx)
        assert not result.ok
        assert "not in the apps allowlist" in result.error

    def test_accepts_known_app(self) -> None:
        """notepad is in the default allowlist."""
        spec = make_type_text_spec()
        ctx = ToolContext(settings=Settings(), dry_run=False)
        # This will fail on non-Windows (platform guard) but should NOT
        # fail on the allowlist check.
        result = spec.run(TypeTextArgs(text="hi", target_app="notepad"), ctx)
        # On non-Windows: fails with "only supported on Windows"
        # On Windows: may fail with focus/window error (expected in CI)
        assert "apps allowlist" not in (result.error or "")


# ── Invariant: no direct keyboard/mouse tools ───────────────────────────


class TestNoGenericKeyboardTools:
    def test_registry_has_no_press_key_tool(self) -> None:
        """v1 must not have a generic press-key or click-at-XY tool."""
        from jarvis.tools.registry import build_default_registry

        registry = build_default_registry()
        for name in registry.names():
            assert "press" not in name.lower()
            assert "click" not in name.lower()
            assert "hotkey" not in name.lower()


# ── F1: foreground-window verification before any paste ─────────────────


class TestTypeTextFocusVerification:
    """With fake GUI modules installed, type_text must verify focus, and
    fail closed when the foreground window is wrong or unreadable
    (docs/03 §5, invariant 5)."""

    def _install_gui_fakes(
        self, monkeypatch: pytest.MonkeyPatch, foreground_title: str = "Untitled - Notepad"
    ) -> tuple[object, list[str], list[tuple[str, str]]]:
        import sys
        from types import SimpleNamespace

        win = SimpleNamespace()
        win.focused = False
        win.visible = True
        win.is_visible = lambda: True
        win.set_focus = lambda: setattr(win, "focused", True)

        desktop = SimpleNamespace(windows=lambda **kwargs: [win])
        pywinauto = SimpleNamespace(Desktop=lambda backend="uia": desktop)

        clipboard: dict[str, str] = {"value": ""}
        copies: list[str] = []
        pyperclip = SimpleNamespace(
            paste=lambda: clipboard["value"],
            copy=lambda value: (copies.append(value), clipboard.__setitem__("value", value)),
        )
        hotkeys: list[tuple[str, str]] = []
        pyautogui = SimpleNamespace(hotkey=lambda *keys: hotkeys.append(keys))

        monkeypatch.setitem(sys.modules, "pywinauto", pywinauto)
        monkeypatch.setitem(sys.modules, "pyperclip", pyperclip)
        monkeypatch.setitem(sys.modules, "pyautogui", pyautogui)

        import jarvis.platform_guard

        monkeypatch.setattr(jarvis.platform_guard, "is_windows", lambda: True)
        monkeypatch.setattr(
            "jarvis.tools.keyboard._read_foreground_title", lambda: foreground_title
        )
        return win, copies, hotkeys

    def test_types_after_foreground_verified(self, monkeypatch: pytest.MonkeyPatch) -> None:
        win, copies, hotkeys = self._install_gui_fakes(monkeypatch)
        spec = make_type_text_spec()
        ctx = ToolContext(settings=Settings(), dry_run=False)
        result = spec.run(TypeTextArgs(text="hi", target_app="notepad"), ctx)
        assert result.ok, result.error
        assert "typed 2" in result.output
        assert win.focused  # the target window was brought to front
        assert hotkeys == [("ctrl", "v")]  # paste happened, exactly once
        assert copies[0] == "hi"  # the new text was copied for the paste

    def test_foreground_mismatch_fails_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._install_gui_fakes(monkeypatch, foreground_title="Calculator")
        spec = make_type_text_spec()
        ctx = ToolContext(settings=Settings(), dry_run=False)
        result = spec.run(TypeTextArgs(text="hi", target_app="notepad"), ctx)
        assert not result.ok
        assert "focus verification failed" in result.error
        assert "Calculator" in result.error

    def test_unreadable_foreground_fails_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._install_gui_fakes(monkeypatch)

        def _boom() -> str:
            raise RuntimeError("desktop session not available")

        from jarvis.tools import keyboard

        monkeypatch.setattr(keyboard, "_read_foreground_title", _boom)
        spec = make_type_text_spec()
        ctx = ToolContext(settings=Settings(), dry_run=False)
        result = spec.run(TypeTextArgs(text="hi", target_app="notepad"), ctx)
        assert not result.ok
        assert "could not read foreground window" in result.error

    def test_config_added_app_accepted_by_allowlist(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """F11: apps added via config extra entries must pass the allowlist."""
        self._install_gui_fakes(monkeypatch, foreground_title="MyCustomApp - Window")
        from jarvis.config import AppSettings

        settings = Settings(apps=AppSettings(notepad="notepad.exe", myApp="custom.exe"))
        spec = make_type_text_spec()
        ctx = ToolContext(settings=settings, dry_run=False)
        result = spec.run(TypeTextArgs(text="hi", target_app="MyApp"), ctx)
        # Passes the allowlist; then fails on the *focus* check (foreground is
        # "MyCustomApp", not "myapp") — proving the allowlist never rejected it.
        assert not result.ok
        assert "apps allowlist" not in result.error
        assert "focus verification failed" in result.error
