"""open_in_app and open_path tools: open files the safe way.

``open_in_app`` (Tier 1) opens a file inside an allowlisted application.  Like
``open_app``: the launcher comes from the ``[apps]`` allowlist (no shell, no
user-typed arguments); the only extra argv element is the resolved path, which
the policy engine root-checks via ``path_args`` (TOCTOU re-checked in act).
Defence in depth: the launcher must be a direct ``.exe``/``.com`` (a ``.cmd``
shim would re-parse the path through ``cmd.exe``); the target must be absolute,
an existing regular file, with no ADS colon; and the tested
:data:`DENIED_FILE_SUFFIXES` constant (never config-driven) refuses anything
Windows would execute.

``open_path`` (Tier 0) hands a file to the OS default viewer (Windows
``os.startfile``): the only new capability is *viewing* a document/image that
already lives inside an allowed root, so it is safe-tier with no confirmation.
It reuses the same hard resolver as ``open_in_app`` (absolute, existing, no ADS,
not a denied executable/script) and adds a strict :data:`OPENABLE_SUFFIXES`
allowlist - images, ``.txt``/``.md``, and ``.pdf`` only.  ``verify()`` for both
is ``verified=None`` + ``verify_note``: the launch is observable, whether the
file actually opened is not (``lock_computer`` precedent).
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from jarvis.apps_scan import INTERPRETER_STEMS
from jarvis.policy import paths, tiers
from jarvis.tools.apps import lookup_command, resolve_command, split_command
from jarvis.tools.base import ToolContext, ToolResult, ToolSpec

#: Launcher suffixes we run; anything else (batch shims, script hosts) is
#: refused honestly.
ALLOWED_EXE_SUFFIXES = frozenset({".exe", ".com"})

#: File suffixes ``open_path`` may hand to the default viewer.  A strict
#: ALLOWLIST: only images and simple documents, never anything Windows could
#: execute, script, follow as a shortcut, or mount as a container.  Suffixes are
#: matched case-insensitively on the resolved path.
OPENABLE_SUFFIXES = frozenset(
    [
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".webp",
        ".tif",
        ".tiff",
        ".txt",
        ".md",
        ".pdf",
    ]
)

#: File-target suffixes never opened: everything Windows would execute or
#: redirect.  Case-insensitive, checked on the RESOLVED path.
DENIED_FILE_SUFFIXES = frozenset(
    [
        ".exe",
        ".com",
        ".scr",
        ".pif",
        ".bat",
        ".cmd",
        ".ps1",
        ".psm1",
        ".vbs",
        ".vbe",
        ".js",
        ".jse",
        ".wsf",
        ".wsh",
        ".hta",
        ".msi",
        ".msp",
        ".msix",
        ".appx",
        ".application",
        ".appref-ms",
        ".gadget",
        ".reg",
        ".lnk",
        ".url",
        ".website",
        ".cpl",
        ".msc",
        ".jar",
        ".scf",
        ".dll",
        ".ocx",
        ".sys",
        ".html",
        ".htm",
        ".xhtml",
        ".mht",
    ]
)


class OpenInAppArgs(BaseModel):
    """Open one file inside one allowlisted application."""

    model_config = ConfigDict(extra="forbid")

    app: str = Field(min_length=1, max_length=64, description="Allowlisted app name.")
    path: str = Field(description="Absolute path of the file to open.")


def _resolve_target(raw: str) -> tuple[Path | None, str]:
    """``(path, "")`` on success or ``(None, reason)``; relative input is
    refused *before* resolution so nothing here depends on the process CWD.
    """
    expanded = os.path.expandvars(os.path.expanduser(raw))
    if not os.path.isabs(expanded):
        return None, f"path must be absolute, got {raw!r}"
    try:
        resolved = paths.resolve_safe(expanded)
    except paths.PathError as exc:
        return None, str(exc)
    text = str(resolved)
    if len(text) > 1 and text[1] == ":":
        text = text[2:]  # keep the drive letter, e.g. "C:"
    if ":" in text:
        return None, f"alternate data stream (extra ':') not allowed: {resolved}"
    name = resolved.name.rstrip(". ")
    suffix = Path(name).suffix.lower()
    if suffix in DENIED_FILE_SUFFIXES:
        return None, f"refusing to open this file type ({suffix}): {resolved.name}"
    if resolved.is_dir():
        return None, f"path is a folder, not a file: {resolved}"
    if not resolved.is_file():
        return None, f"file does not exist: {resolved}"
    return resolved, ""


def _run_open_in_app(args: OpenInAppArgs, ctx: ToolContext) -> ToolResult:
    if ctx.dry_run:
        return ToolResult(ok=True, output=f"[dry-run] would open {args.path} in {args.app}")
    command = lookup_command(args.app, ctx)
    if command is None:
        known = sorted(str(k) for k in ctx.settings.apps.model_dump())
        return ToolResult(
            ok=False,
            error=f"unknown app {args.app!r}; known apps: {', '.join(known)}",
            data={"known": known},
        )
    argv = split_command(command)
    executable = resolve_command(argv[0]) if argv else None
    if executable is None:
        return ToolResult(
            ok=False,
            error=f"could not find an executable for {args.app!r} ({command!r}); "
            "give an absolute path in [apps]",
        )
    if Path(executable).suffix.lower() not in ALLOWED_EXE_SUFFIXES:
        return ToolResult(
            ok=False,
            error=f"{args.app!r} resolves to {executable!r}; give an absolute "
            "path in [apps] ending in .exe or .com",
        )
    if Path(executable).stem.lower() in INTERPRETER_STEMS:
        return ToolResult(
            ok=False, error=f"refusing {args.app!r}: {executable!r} is an interpreter"
        )
    resolved, refusal = _resolve_target(args.path)
    if resolved is None:
        return ToolResult(ok=False, error=refusal)
    if paths.is_protected(resolved):
        return ToolResult(ok=False, error=f"path is protected: {resolved}")
    roots = paths.roots_from_settings(ctx.settings)
    if not paths.within_any_root(resolved, roots):
        return ToolResult(ok=False, error=f"path is outside the allowed folders: {resolved}")
    try:
        proc = subprocess.Popen([executable, *argv[1:], str(resolved)], close_fds=True)
    except OSError as exc:
        return ToolResult(ok=False, error=f"failed to launch {command!r}: {exc}")
    return ToolResult(
        ok=True,
        output=f"opened {resolved.name} in {args.app}",
        data={"app": args.app, "file": str(resolved), "pid": proc.pid},
    )


def _verify_open_in_app(args: OpenInAppArgs, result: ToolResult, ctx: ToolContext) -> ToolResult:
    """Honest verification: the launch is observable, the file opening is not."""
    if not result.ok:
        return result.model_copy(update={"verified": False})
    if ctx.dry_run:
        return result.model_copy(update={"verified": None})
    file_name = Path(str(result.data.get("file") or args.path)).name or args.path
    note = f"launched {args.app} with {file_name}; opening it cannot be verified"
    return result.model_copy(update={"verified": None, "verify_note": note})


def _describe_open_in_app(args: OpenInAppArgs) -> str:
    """Best-effort resolved path; the raw arg when it cannot be resolved."""
    target = args.path
    try:
        target = str(paths.resolve_safe(args.path))
    except paths.PathError:
        pass
    return f"Open {target} in {args.app}"


def make_open_in_app_spec() -> ToolSpec:
    """Build the ``open_in_app`` tool (Tier 1)."""
    return ToolSpec(
        name="open_in_app",
        description=(
            "Open a file from the allowed folders in a pre-approved application from the "
            "[apps] allowlist (e.g. notepad, vscode). Only the resolved path is passed "
            "to the program; no extra arguments."
        ),
        args_model=OpenInAppArgs,
        base_tier=1,
        windows_only=False,
        path_args=("path",),
        run=_run_open_in_app,
        verify=_verify_open_in_app,
        describe=_describe_open_in_app,
    )


#: A ``launcher(path)`` that opens a file with the default application.
ViewerFunc = Callable[[str], None]


class OpenPathArgs(BaseModel):
    """Open one file with the operating system's default viewer."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Absolute path of the file to open.")


def _launch_default_viewer(path: str) -> None:
    """Open ``path`` with the OS default viewer (Windows ``os.startfile``).

    ``os.startfile`` exists only on Windows; the spec is ``windows_only`` (the
    engine refuses it elsewhere), and this raises a clean ``OSError`` if it is
    ever reached on another platform so the tool fails closed rather than
    guessing a shell command.
    """
    from jarvis.platform_guard import require_windows

    require_windows("Opening a file with the default viewer")
    startfile = getattr(os, "startfile", None)
    if startfile is None:  # pragma: no cover - only on a mis-reported platform
        raise OSError("no default viewer is available on this platform")
    startfile(path)


def _run_open_path(args: OpenPathArgs, ctx: ToolContext, launcher: ViewerFunc) -> ToolResult:
    if ctx.dry_run:
        return ToolResult(ok=True, output=f"[dry-run] would open {args.path} in the default viewer")
    resolved, refusal = _resolve_target(args.path)
    if resolved is None:
        return ToolResult(ok=False, error=refusal)
    name = resolved.name.rstrip(". ")
    suffix = Path(name).suffix.lower()
    if suffix not in OPENABLE_SUFFIXES:
        allowed = ", ".join(sorted(OPENABLE_SUFFIXES))
        return ToolResult(
            ok=False,
            error=f"refusing to open {suffix or 'this file'!r}; the default viewer only "
            f"opens these types: {allowed}",
        )
    if paths.is_protected(resolved):
        return ToolResult(ok=False, error=f"path is protected: {resolved}")
    if not paths.within_any_root(resolved, paths.roots_from_settings(ctx.settings)):
        return ToolResult(ok=False, error=f"path is outside the allowed folders: {resolved}")
    try:
        launcher(str(resolved))
    except OSError as exc:
        return ToolResult(ok=False, error=f"failed to open {resolved.name}: {exc}")
    return ToolResult(
        ok=True,
        output=f"opened {resolved.name} with the default viewer",
        data={"file": str(resolved)},
    )


def _verify_open_path(args: OpenPathArgs, result: ToolResult, ctx: ToolContext) -> ToolResult:
    """Honest verification: the launch is observable, the file opening is not."""
    if not result.ok:
        return result.model_copy(update={"verified": False})
    if ctx.dry_run:
        return result.model_copy(update={"verified": None})
    file_name = Path(str(result.data.get("file") or args.path)).name or args.path
    note = f"handed {file_name} to the default viewer; opening it cannot be verified"
    return result.model_copy(update={"verified": None, "verify_note": note})


def _describe_open_path(args: OpenPathArgs) -> str:
    """Best-effort resolved path; the raw arg when it cannot be resolved."""
    target = args.path
    try:
        target = str(paths.resolve_safe(args.path))
    except paths.PathError:
        pass
    return f"Open {target} with the default viewer"


def make_open_path_spec(launcher: ViewerFunc | None = None) -> ToolSpec:
    """Build the ``open_path`` tool (Tier 0, Windows only).

    ``launcher(path)`` is an injection point for tests; the default is the OS
    default viewer.
    """
    launch = launcher or _launch_default_viewer
    return ToolSpec(
        name="open_path",
        description=(
            "Open a picture or document (.jpg/.png/.txt/.md/.pdf and similar) from the "
            "allowed folders with the operating system's default viewer. It never runs "
            "programs or scripts and cannot open anything outside the allowed folders."
        ),
        args_model=OpenPathArgs,
        base_tier=tiers.TIER_SAFE,
        windows_only=True,
        path_args=("path",),
        run=lambda args, ctx: _run_open_path(args, ctx, launch),
        verify=_verify_open_path,
        describe=_describe_open_path,
    )
