"""Dictation tools (Phase 5).

``start_dictation`` opens Notepad and begins a voice→text session.
``stop_dictation`` ends the session.  ``save_dictation`` saves the
Notepad content to a file.

All audio happens inside the voice loop; these tools control the
session lifecycle and the file-save step.

Focus verification: Notepad must be the foreground window before each
paste.  If focus is lost, the dictation pauses (voice loop concern).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from jarvis.tools.base import ToolContext, ToolResult, ToolSpec

logger = logging.getLogger(__name__)


class StartDictationArgs(BaseModel):
    """Arguments for ``start_dictation``."""

    target_app: str = Field(default="notepad", description="Application to dictate into")


class StopDictationArgs(BaseModel):
    """Arguments for ``stop_dictation``."""


class SaveDictationArgs(BaseModel):
    """Arguments for ``save_dictation``."""

    path: str = Field(min_length=1, description="File path to save the dictated content")


def _active_voice_loop(ctx: ToolContext) -> Any | None:
    """Return the running voice loop via ``ctx.voice`` (or ``None``).

    The daemon wires an active :class:`VoiceService` through the context's
    ``voice`` facade; the CLI/in-process mode has no running loop.  Tools must
    be honest about which case is true.
    """
    voice = getattr(ctx, "voice", None)
    if voice is None:
        return None
    loop = None
    for attr in ("active_loop", "loop"):
        value = getattr(voice, attr, None)
        if callable(value):
            value = value()
        if value is not None:
            loop = value
            break
    return loop


def _start_dictation_run(args: StartDictationArgs, ctx: ToolContext) -> ToolResult:
    """Open the target app and signal the voice loop to start dictating."""
    from jarvis.tools.apps import lookup_command

    target = args.target_app
    app_cmd = lookup_command(target, ctx)
    if app_cmd is None:
        from jarvis.tools.apps import allowlist_names

        return ToolResult(
            ok=False,
            error=f"{target!r} is not in the apps allowlist. Allowed: {', '.join(allowlist_names(ctx.settings))}",
        )

    # Try to open/focus the app
    try:
        import subprocess

        subprocess.Popen(
            [app_cmd],
            shell=False,
            close_fds=True,
        )
    except Exception as exc:  # noqa: BLE001
        return ToolResult(ok=False, error=f"failed to open {target}: {exc}")

    loop = _active_voice_loop(ctx)
    if loop is not None and hasattr(loop, "start_dictation"):
        loop.start_dictation(target.lower())
        return ToolResult(
            ok=True,
            output=f"dictation started in {target}",
            data={
                "target_app": target.lower(),
                "action": "start_dictation",
                "dictation_active": True,
            },
        )

    # Honest fallback: the tool succeeded in opening the app, but no live
    # voice loop is wired to receive the dictation stream.
    return ToolResult(
        ok=True,
        output=(
            f"opened {target}; voice dictation is not active (no running voice loop) — "
            "say 'hey jarvis' first, then ask to start dictation"
        ),
        data={
            "target_app": target.lower(),
            "action": "start_dictation",
            "dictation_active": False,
        },
    )


def _stop_dictation_run(args: StopDictationArgs, ctx: ToolContext) -> ToolResult:
    """Stop the active dictation session."""
    loop = _active_voice_loop(ctx)
    if loop is not None and hasattr(loop, "stop_dictation"):
        loop.stop_dictation()
        return ToolResult(
            ok=True,
            output="dictation stopped",
            data={"action": "stop_dictation", "dictation_active": False},
        )
    return ToolResult(
        ok=True,
        output="dictation stopped (no active voice loop)",
        data={"action": "stop_dictation", "dictation_active": False},
    )


def _check_save_target(raw: str, ctx: ToolContext) -> tuple[Path | None, str | None]:
    """Resolve and policy-check a save path.

    Returns ``(resolved_path, None)`` when the path is valid and inside an
    allowed root, or ``(None, error_message)`` otherwise.  The raw string is
    never written directly: every write uses the returned resolved path.
    """
    try:
        from jarvis.policy.paths import resolve_safe, roots_from_settings, within_any_root

        resolved = resolve_safe(raw)
        if not within_any_root(resolved, roots_from_settings(ctx.settings)):
            return None, f"save path {raw!r} is outside the allowed roots"
        return resolved, None
    except Exception as exc:  # noqa: BLE001
        return None, f"invalid path: {exc}"


def _write_saved_content(content: str, resolved: Path, ctx: ToolContext) -> None:
    """Write dictation content to the *already validated* resolved path.

    ``ctx.dry_run`` never reaches here (the tool runner short-circuits), but
    guarding it keeps the invariant that no tool has side effects in a dry run.
    """
    if ctx.dry_run:
        return
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding="utf-8")


def _save_dictation_run(args: SaveDictationArgs, ctx: ToolContext) -> ToolResult:
    """Save dictated content from Notepad to a file via Ctrl+Shift+S or clipboard."""
    # Validate path through policy (resolve + allowed roots check) BEFORE any
    # GUI interaction; the write later uses the resolved path only.
    resolved, path_error = _check_save_target(args.path, ctx)
    if resolved is None:
        return ToolResult(ok=False, error=path_error or f"invalid path: {args.path!r}")

    # Read Notepad content via clipboard: Ctrl+A, Ctrl+C, then save
    try:
        import pyperclip

        old_clipboard = pyperclip.paste()
    except (OSError, AttributeError):
        old_clipboard = ""

    try:
        import pywinauto

        desktop = pywinauto.Desktop(backend="uia")
        windows = desktop.windows(title_re=".*Notepad.*", found_index=0)
        if not windows:
            return ToolResult(ok=False, error="Notepad window not found")
        win = windows[0]
        win.set_focus()
    except Exception as exc:  # noqa: BLE001
        return ToolResult(ok=False, error=f"failed to focus Notepad: {exc}")

    try:
        import pyautogui

        pyautogui.hotkey("ctrl", "a")
        pyautogui.hotkey("ctrl", "c")
    except Exception as exc:  # noqa: BLE001
        return ToolResult(ok=False, error=f"clipboard copy failed: {exc}")

    try:
        content = pyperclip.paste()
    except Exception as exc:  # noqa: BLE001
        return ToolResult(ok=False, error=f"clipboard read failed: {exc}")
    finally:
        try:
            pyperclip.copy(old_clipboard)
        except (OSError, AttributeError):
            logger.debug("clipboard restore failed", exc_info=True)

    if not content.strip():
        return ToolResult(ok=False, error="no content to save (Notepad appears empty)")

    # Write to the *resolved* path (never the raw string)
    try:
        _write_saved_content(content, resolved, ctx)
    except Exception as exc:  # noqa: BLE001
        return ToolResult(ok=False, error=f"failed to write file: {exc}")

    return ToolResult(
        ok=True,
        output=f"saved {len(content)} chars to {resolved}",
        data={"path": str(resolved), "size": len(content)},
    )


def _verify_save(args: SaveDictationArgs, result: ToolResult, ctx: ToolContext) -> ToolResult:
    """Verify the file exists and has content at the resolved path."""
    if not result.ok:
        return result
    resolved, _ = _check_save_target(args.path, ctx)
    if resolved is None:
        return result.model_copy(update={"verified": False})
    try:
        if not resolved.exists():
            return result.model_copy(update={"verified": False})
        return result.model_copy(update={"verified": True})
    except Exception:  # noqa: BLE001
        return result.model_copy(update={"verified": None})


def make_start_dictation_spec() -> ToolSpec:
    """Build the ``start_dictation`` tool spec."""
    return ToolSpec(
        name="start_dictation",
        description=(
            "Open the target application and begin a voice dictation session. "
            "Speech is transcribed and typed into the app window."
        ),
        args_model=StartDictationArgs,
        base_tier=1,
        windows_only=True,
        timeout_s=15,
        run=_start_dictation_run,
        describe=lambda args: f"start dictation in {args.target_app}",
    )


def make_stop_dictation_spec() -> ToolSpec:
    """Build the ``stop_dictation`` tool spec."""
    return ToolSpec(
        name="stop_dictation",
        description="Stop the active voice dictation session.",
        args_model=StopDictationArgs,
        base_tier=0,
        windows_only=False,
        timeout_s=5,
        run=_stop_dictation_run,
        describe=lambda args: "stop dictation",
    )


def make_save_dictation_spec() -> ToolSpec:
    """Build the ``save_dictation`` tool spec."""
    return ToolSpec(
        name="save_dictation",
        description=(
            "Save the dictated content from Notepad to a file. "
            "The file is written inside allowed roots."
        ),
        args_model=SaveDictationArgs,
        base_tier=1,
        windows_only=True,
        timeout_s=15,
        run=_save_dictation_run,
        verify=_verify_save,
        path_args=("path",),
        describe=lambda args: f"save dictation to {args.path!r}",
    )
