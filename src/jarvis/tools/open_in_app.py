"""open_in_app tool: open a file inside an allowlisted application (Tier 1).

Like ``open_app``: the launcher comes from the ``[apps]`` allowlist (no shell,
no user-typed arguments); the only extra argv element is the resolved path,
which the policy engine root-checks via ``path_args`` (TOCTOU re-checked in
act).  Defence in depth: the launcher must be a direct ``.exe``/``.com`` (a
``.cmd`` shim would re-parse the path through ``cmd.exe``); the target must be
absolute, an existing regular file, with no ADS colon; and the tested
:data:`DENIED_FILE_SUFFIXES` constant (never config-driven) refuses anything
Windows would execute.  ``verify()`` is ``verified=None`` + ``verify_note``:
launch observable, file opening not (``lock_computer`` precedent).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from jarvis.apps_scan import INTERPRETER_STEMS
from jarvis.policy import paths
from jarvis.tools.apps import lookup_command, resolve_command, split_command
from jarvis.tools.base import ToolContext, ToolResult, ToolSpec

#: Launcher suffixes we run; anything else (batch shims, script hosts) is
#: refused honestly.
ALLOWED_EXE_SUFFIXES = frozenset({".exe", ".com"})

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
