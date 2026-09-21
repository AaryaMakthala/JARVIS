"""Keyboard automation tools (Phase 5).

``type_text`` brings the target application to the foreground, verifies
focus, pastes text via clipboard (Ctrl+V), then restores the clipboard.

No generic "press keys", "click at x,y", or "run command" tools exist
in v1 (docs/04_TOOLS_SPEC.md section 2.3).
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from jarvis.tools.base import ToolContext, ToolResult, ToolSpec

logger = logging.getLogger(__name__)


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

    Fails closed: an unreadable foreground window is a verification failure,
    never an excuse to paste into an unknown window.
    """
    try:
        title = _read_foreground_title()
    except Exception as exc:  # noqa: BLE001 - fail closed when focus cannot be read
        return ToolResult(
            ok=False,
            error=f"focus verification failed: could not read foreground window: {exc}",
        )
    if target.lower() not in title.lower():
        return ToolResult(
            ok=False,
            error=f"focus verification failed: foreground is {title!r}, expected {target!r}",
        )
    return None


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

    # Verify focus
    try:
        import pywinauto

        desktop = pywinauto.Desktop(backend="uia")
        windows = desktop.windows(title_re=f".*{target}.*", found_index=0)
        if not windows:
            return ToolResult(ok=False, error=f"no window matching {target!r} found")
        win = windows[0]
        if not win.is_visible():
            return ToolResult(ok=False, error=f"{target} window is not visible")
        win.set_focus()
    except Exception as exc:  # noqa: BLE001
        return ToolResult(ok=False, error=f"failed to focus {target!r}: {exc}")

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
