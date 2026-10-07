"""4.17b: type_text focus is a bounded poll (12 x 0.25 s <= 3 s), never a hang.

A freshly launched window can take a moment to exist, so the focus block
retries — but strictly bounded and fail-closed: on timeout the *same* error
the last attempt produced is returned (no generic "timeout" message).  The
sleep goes through ``keyboard._pause`` so tests never actually wait.  All
GUI modules are faked — no real windows.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

import jarvis.platform_guard
from jarvis.config import Settings
from jarvis.tools import keyboard
from jarvis.tools.base import ToolContext, ToolResult
from jarvis.tools.keyboard import TypeTextArgs, make_type_text_spec


class ElementNotFoundError(Exception):
    """Same __name__ as pywinauto.findwindows.ElementNotFoundError."""


class _FakeWindow:
    def __init__(self) -> None:
        self.focused = False

    def is_visible(self) -> bool:
        return True

    def set_focus(self) -> None:
        self.focused = True


def _install(
    monkeypatch: pytest.MonkeyPatch,
    *,
    windows_impl: Callable[[dict[str, Any]], list[Any]],
    enum_registry: tuple[tuple[int, str, bool, int], ...] = (),
    process_name: str = "notepad.exe",
    foreground_title: str = "Untitled - Notepad",
) -> tuple[SimpleNamespace, list[float]]:
    """Install fake GUI modules; return (captures, recorded _pause sleeps)."""
    captured: list[dict[str, Any]] = []

    def windows(**kwargs: Any) -> list[Any]:
        captured.append(kwargs)
        return windows_impl(kwargs)

    pywinauto = SimpleNamespace(Desktop=lambda backend="uia": SimpleNamespace(windows=windows))

    focused: list[int] = []

    class _HwndWrapper:
        def __init__(self, hwnd: int) -> None:
            self.hwnd = hwnd

        def set_focus(self) -> None:
            focused.append(self.hwnd)

    hwnd_mod = ModuleType("pywinauto.controls.hwndwrapper")
    hwnd_mod.HwndWrapper = _HwndWrapper  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pywinauto", pywinauto)
    monkeypatch.setitem(sys.modules, "pywinauto.controls", ModuleType("pywinauto.controls"))
    monkeypatch.setitem(sys.modules, "pywinauto.controls.hwndwrapper", hwnd_mod)

    titles = {h: t for h, t, _v, _p in enum_registry}
    visibles = {h: v for h, _t, v, _p in enum_registry}
    pids = {h: p for h, _t, _v, p in enum_registry}

    def _enum(visit: Callable[[int, object], bool], _extra: object) -> None:
        for hwnd in titles:
            visit(hwnd, None)

    monkeypatch.setitem(
        sys.modules,
        "win32gui",
        SimpleNamespace(
            EnumWindows=_enum,
            GetWindowText=lambda hwnd: titles[hwnd],
            IsWindowVisible=lambda hwnd: visibles[hwnd],
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "win32process",
        SimpleNamespace(GetWindowThreadProcessId=lambda hwnd: (7, pids[hwnd])),
    )
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(Process=lambda pid: SimpleNamespace(name=lambda: process_name)),
    )
    monkeypatch.setitem(
        sys.modules, "pyperclip", SimpleNamespace(paste=lambda: "", copy=lambda _v: None)
    )
    monkeypatch.setitem(sys.modules, "pyautogui", SimpleNamespace(hotkey=lambda *_k: None))

    monkeypatch.setattr(jarvis.platform_guard, "is_windows", lambda: True)
    monkeypatch.setattr(keyboard, "_read_foreground_title", lambda: foreground_title)

    sleeps: list[float] = []
    monkeypatch.setattr(keyboard, "_pause", sleeps.append)
    return SimpleNamespace(captured=captured, focused=focused), sleeps


def _run(target: str = "notepad") -> ToolResult:
    spec = make_type_text_spec()
    ctx = ToolContext(settings=Settings(), dry_run=False)
    return spec.run(TypeTextArgs(text="hi", target_app=target), ctx)


def _raise_not_found(_kwargs: dict[str, Any]) -> list[Any]:
    raise ElementNotFoundError("found_index is specified as 0, but 0 window/s found")


# ── the poll retries, then succeeds ──────────────────────────────────────


def test_polls_until_the_window_appears(monkeypatch: pytest.MonkeyPatch) -> None:
    win = _FakeWindow()
    results: list[list[Any]] = [[], [], [win]]

    fake, sleeps = _install(monkeypatch, windows_impl=lambda _kw: results.pop(0))
    result = _run()
    assert result.ok, result.error
    assert win.focused
    assert len(fake.captured) == 3, "one lookup per attempt"
    assert sleeps == [0.25, 0.25], "paused only *between* the failed attempts"
    assert sum(sleeps) <= 3.0


def test_no_pause_when_focus_succeeds_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    win = _FakeWindow()
    fake, sleeps = _install(monkeypatch, windows_impl=lambda _kw: [win])
    result = _run()
    assert result.ok, result.error
    assert len(fake.captured) == 1
    assert sleeps == [], "no sleep after the first, successful attempt"


# ── bounded fail-closed on timeout ───────────────────────────────────────


def test_gives_up_after_the_bounded_attempts_with_the_same_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake, sleeps = _install(monkeypatch, windows_impl=_raise_not_found, enum_registry=())
    start = time.monotonic()
    result = _run()  # must return, not raise or hang
    elapsed = time.monotonic() - start
    assert not result.ok
    assert result.error is not None
    assert result.error.startswith("failed to focus 'notepad':")
    assert "0 window/s found" in result.error, "timeout keeps the last attempt's error"
    assert len(fake.captured) == keyboard._FOCUS_ATTEMPTS == 12
    assert len(sleeps) == 11, "12 attempts -> 11 inter-attempt pauses"
    assert sum(sleeps) <= 3.0, "poll budget is <= 3 s"
    assert elapsed < 1.0, "tests must not really wait (sleep is injected)"


def test_timeout_keeps_the_original_no_window_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake, sleeps = _install(monkeypatch, windows_impl=lambda _kw: [])
    result = _run()
    assert not result.ok
    assert result.error == "no window matching 'notepad' found", (
        "timeout must surface the real fail-closed error, not a generic timeout"
    )
    assert len(fake.captured) == 12
    assert sum(sleeps) <= 3.0
