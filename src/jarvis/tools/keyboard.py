"""Keyboard automation tools (Phase 5).

``type_text`` brings the target application to the foreground, verifies
focus, pastes text via clipboard (Ctrl+V), then restores the clipboard.

No generic "press keys", "click at x,y", or "run command" tools exist
in v1 (docs/04_TOOLS_SPEC.md section 2.3).
"""

from __future__ import annotations

import logging
import re
import time

from pydantic import BaseModel, Field

from jarvis.tools.base import ToolContext, ToolResult, ToolSpec

logger = logging.getLogger(__name__)

# Bounded focus poll: a freshly launched window may not exist yet (4.17b).
_FOCUS_ATTEMPTS = 12
_FOCUS_POLL_S = 0.25

# Bounded foreground re-read (4.19): set_focus() can return before Windows
# has finished activating the window, so the first read can transiently see
# an empty title.  8 attempts x 0.25 s <= 1.75 s of sleeping, well inside
# type_text's 10 s tool timeout.
_VERIFY_ATTEMPTS = 8
_VERIFY_POLL_S = 0.25


def _pause(seconds: float) -> None:
    """Sleep between focus attempts; a module function so tests never wait."""
    time.sleep(seconds)


class TypeTextArgs(BaseModel):
    """Arguments for the ``type_text`` tool."""

    text: str = Field(min_length=1, max_length=2000, description="Text to type into the target app")
    target_app: str = Field(
        min_length=1,
        max_length=100,
        description="Allowlisted application name to type into",
    )


def _read_foreground_title() -> str:
    """Return the current foreground window title; raises when unreadable.

    Kept as a small function so tests can override it: if this read ever
    fails the *caller* must fail closed (docs/03 §1), never skip the focus
    check.
    """
    import ctypes

    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    hwnd = user32.GetForegroundWindow()
    buf = ctypes.create_unicode_buffer(256)
    user32.GetWindowTextW(hwnd, buf, 256)
    return buf.value or ""


def _verify_focus(target: str) -> ToolResult | None:
    """Return an error :class:`ToolResult` unless the foreground window is ``target``.

    Re-reads the foreground title up to ``_VERIFY_ATTEMPTS`` times (4.19):
    ``set_focus()`` can return before Windows has finished activating the
    window, so a single read can transiently see ``foreground=''``.  The
    acceptance condition is unchanged (the same substring check as before,
    evaluated per read), and the error returned when the budget is exhausted
    is byte-identical to the single-read error, so 4.17c's non-retryable
    classification keeps matching.  Fails closed: a read that raises stops
    immediately and is never polled again, and an unreadable or still-
    mismatched foreground window is a verification failure, never an excuse
    to paste into an unknown window.
    """
    title = ""
    for attempt in range(_VERIFY_ATTEMPTS):
        try:
            title = _read_foreground_title()
        except Exception as exc:  # noqa: BLE001 - fail closed when focus cannot be read
            return ToolResult(
                ok=False,
                error=f"focus verification failed: could not read foreground window: {exc}",
            )
        if target.lower() in title.lower():
            return None
        if attempt < _VERIFY_ATTEMPTS - 1:
            _pause(_VERIFY_POLL_S)
    return ToolResult(
        ok=False,
        error=f"focus verification failed: foreground is {title!r}, expected {target!r}",
    )


def _focus_fallback(target: str) -> bool:
    """Focus the first visible window matching ``target`` by title or process.

    Local EnumWindows + psutil walk (same enumeration as ``windows_list``,
    kept local so its ``WindowInfo`` contract is untouched).  ``False`` means
    nothing matched or nothing could be focused: the caller keeps the original
    fail-closed error, so this can never widen what gets typed into.
    """
    import psutil
    import win32gui
    import win32process
    from pywinauto.controls.hwndwrapper import HwndWrapper

    needle = target.casefold()
    matches: list[int] = []

    def _visit(hwnd: int, _extra: object) -> bool:
        try:
            title = (win32gui.GetWindowText(hwnd) or "").strip()
            if not title or not win32gui.IsWindowVisible(hwnd):
                return True
            _thread_id, pid = win32process.GetWindowThreadProcessId(hwnd)
            process = psutil.Process(pid).name().casefold()
            if needle in title.casefold() or process.startswith(needle):
                matches.append(hwnd)
        except Exception:  # noqa: BLE001 - a vanished/denied window is skipped
            return True
        return True

    win32gui.EnumWindows(_visit, None)
    try:
        if not matches:
            return False
        HwndWrapper(matches[0]).set_focus()
        return True
    except Exception:  # stale handle: the caller fails closed
        logger.debug("fallback focus failed for %r", target, exc_info=True)
        return False


def _focus_window(target: str) -> ToolResult | None:
    """Focus ``target`` with a bounded poll; ``None`` means focused.

    12 attempts x 0.25 s (<= 3 s, well inside ``type_text``'s 10 s budget):
    a window the planner just launched may take a moment to exist.  Returns
    the *last* fail-closed error unchanged on timeout, never a different one.
    """
    last_error = ToolResult(ok=False, error=f"failed to focus {target!r}")
    for attempt in range(_FOCUS_ATTEMPTS):
        try:
            import pywinauto

            desktop = pywinauto.Desktop(backend="uia")
            windows = desktop.windows(title_re=f"(?i).*{re.escape(target)}.*", found_index=0)
            if not windows:
                last_error = ToolResult(ok=False, error=f"no window matching {target!r} found")
            else:
                win = windows[0]
                if not win.is_visible():
                    last_error = ToolResult(ok=False, error=f"{target} window is not visible")
                else:
                    win.set_focus()
                    return None
        except Exception as exc:  # noqa: BLE001 - pywinauto raises for 0 title matches
            # pywinauto raises findwindows.ElementNotFoundError when the title
            # regex matched 0 windows (e.g. Windows 11 Store Notepad): try the
            # local walk once, else keep the fail-closed error, never raise.
            recovered = type(exc).__name__ == "ElementNotFoundError" and _focus_fallback(target)
            if recovered:
                return None
            last_error = ToolResult(ok=False, error=f"failed to focus {target!r}: {exc}")
        if attempt < _FOCUS_ATTEMPTS - 1:
            _pause(_FOCUS_POLL_S)
    logger.debug("focus poll timed out for %r after %d attempts", target, _FOCUS_ATTEMPTS)
    return last_error


def _type_text_run(args: TypeTextArgs, ctx: ToolContext) -> ToolResult:
    """Paste text into the foreground window after focus verification."""
    target = args.target_app
    text = args.text

    # Check the app is in the allowlist
    from jarvis.tools.apps import allowlist_names

    allowed_apps = allowlist_names(ctx.settings)
    if target.lower() not in allowed_apps:
        return ToolResult(
            ok=False,
            error=f"{target!r} is not in the apps allowlist. Allowed: {', '.join(allowed_apps)}",
        )

    # Import Windows-only modules behind platform guard
    try:
        from jarvis.platform_guard import is_windows

        if not is_windows():
            return ToolResult(ok=False, error="type_text is only supported on Windows")
    except ImportError:
        return ToolResult(ok=False, error="platform guard not available")

    # Verify focus: bounded poll (a just-launched window may not exist yet).
    focus_error = _focus_window(target)
    if focus_error is not None:
        return focus_error

    # Focus verification: re-check the foreground window.
    focus_error = _verify_focus(target)
    if focus_error is not None:
        return focus_error

    # Paste via clipboard
    try:
        import pyperclip

        old_clipboard = pyperclip.paste()
        pyperclip.copy(text)
    except Exception as exc:  # noqa: BLE001
        return ToolResult(ok=False, error=f"clipboard operation failed: {exc}")

    try:
        import pyautogui

        pyautogui.hotkey("ctrl", "v")
    except Exception as exc:  # noqa: BLE001
        return ToolResult(ok=False, error=f"paste hotkey failed: {exc}")
    finally:
        # Restore clipboard
        try:
            pyperclip.copy(old_clipboard)
        except (OSError, AttributeError):
            logger.debug("clipboard restore failed", exc_info=True)

    return ToolResult(ok=True, output=f"typed {len(text)} chars into {target}")


def _type_text_verify(args: TypeTextArgs, result: ToolResult, ctx: ToolContext) -> ToolResult:
    """Best-effort read-back verification."""
    if not result.ok:
        return result
    # We trust the clipboard + focus check; full read-back requires
    # pywinauto which may not work with every control.
    return result.model_copy(update={"verified": None})


def make_type_text_spec() -> ToolSpec:
    """Build the ``type_text`` tool spec."""
    return ToolSpec(
        name="type_text",
        description=(
            "Type text into a specified application window. Verifies the window "
            "is focused before pasting via clipboard. Fails if the target app "
            "is not in the allowlist or is not the foreground window."
        ),
        args_model=TypeTextArgs,
        base_tier=1,
        windows_only=True,
        timeout_s=10,
        run=_type_text_run,
        verify=_type_text_verify,
        describe=lambda args: f"type {len(args.text)} chars into {args.target_app!r}",
    )
