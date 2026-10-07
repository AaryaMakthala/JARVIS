"""4.17a: type_text window lookup — case-insensitive title + local fallback.

The live failure was ``found_index is specified as 0, but 0 window/s found``:
pywinauto compiles ``title_re`` without IGNORECASE, so lowercase ``notepad``
never matched ``Untitled - Notepad``.  When the title lookup still finds
nothing, a local EnumWindows + psutil walk (title casefold OR process name)
focuses the window instead; if that also finds nothing the original fail-closed
error is returned, never raised.  All GUI modules are faked — no real windows.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

import jarvis.platform_guard
from jarvis.config import AppSettings, Settings
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
) -> SimpleNamespace:
    """Install fake pywinauto/win32gui/win32process/psutil/clipboard modules."""
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
    return SimpleNamespace(captured=captured, focused=focused)


def _run(settings: Settings | None = None, target: str = "notepad", text: str = "hi") -> ToolResult:
    spec = make_type_text_spec()
    ctx = ToolContext(settings=settings or Settings(), dry_run=False)
    return spec.run(TypeTextArgs(text=text, target_app=target), ctx)


def _raise_not_found(_kwargs: dict[str, Any]) -> list[Any]:
    raise ElementNotFoundError("found_index is specified as 0, but 0 window/s found")


# ── primary lookup: the title regex ──────────────────────────────────────


def test_title_re_is_case_insensitive_and_matches_the_real_title(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    win = _FakeWindow()
    fake = _install(monkeypatch, windows_impl=lambda _kw: [win])
    result = _run()
    assert result.ok, result.error
    pattern = fake.captured[0]["title_re"]
    assert pattern.startswith("(?i)")
    assert re.match(pattern, "Untitled - Notepad"), "lowercase target must match 'Notepad'"
    assert fake.captured[0]["found_index"] == 0
    assert win.focused


def test_title_re_escapes_regex_metacharacters(monkeypatch: pytest.MonkeyPatch) -> None:
    win = _FakeWindow()
    fake = _install(
        monkeypatch,
        windows_impl=lambda _kw: [win],
        foreground_title="my.app Window",
    )
    settings = Settings(apps=AppSettings(**{"my.app": "custom.exe"}))
    result = _run(settings=settings, target="my.app")
    assert result.ok, result.error
    assert fake.captured[0]["title_re"] == r"(?i).*my\.app.*"


# ── fallback: local EnumWindows + psutil walk ────────────────────────────


def test_falls_back_to_process_name_when_title_lookup_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Title deliberately contains no "notepad": only the process name matches.
    fake = _install(
        monkeypatch,
        windows_impl=_raise_not_found,
        enum_registry=((101, "MSCTFIME UI", True, 4242),),
        process_name="notepad.exe",
    )
    result = _run()
    assert result.ok, result.error
    assert fake.focused == [101]


def test_falls_back_to_title_casefold_when_process_does_not_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _install(
        monkeypatch,
        windows_impl=_raise_not_found,
        enum_registry=((102, "Untitled - Notepad", True, 99),),
        process_name="otherapp.exe",
    )
    result = _run()
    assert result.ok, result.error
    assert fake.focused == [102]


def test_ignores_invisible_and_untitled_windows_in_the_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _install(
        monkeypatch,
        windows_impl=_raise_not_found,
        enum_registry=(
            (103, "", True, 4242),  # untitled
            (104, "Untitled - Notepad", False, 4242),  # invisible
            (105, "Untitled - Notepad", True, 4242),  # the one that counts
        ),
        process_name="notepad.exe",
    )
    result = _run()
    assert result.ok, result.error
    assert fake.focused == [105]


# ── fail-closed when nothing can be found ────────────────────────────────


def test_focus_failure_returns_the_fail_closed_error_when_nothing_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, windows_impl=_raise_not_found, enum_registry=())
    result = _run()  # must return, not raise
    assert not result.ok
    assert result.error is not None
    assert result.error.startswith("failed to focus 'notepad':")
    assert "0 window/s found" in result.error
