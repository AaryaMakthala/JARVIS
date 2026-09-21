"""open_app tool: launch an allowlisted application (Tier 0).

The app must exist in ``settings.apps`` (name -> command).  Nothing is ever
run through a shell and no user-supplied arguments reach the command line -
the whole command string comes from the config allowlist.  ``verify()`` polls
for a running process with a matching image name (best-effort; ``None`` when
the process table is unavailable, e.g. on non-Windows CI).
"""

from __future__ import annotations

import subprocess
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from jarvis.tools.base import ToolContext, ToolResult, ToolSpec

_VERIFY_SECONDS = 8.0
_POLL = 0.25


class OpenAppArgs(BaseModel):
    """Launch an application from the allowlist."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64, description="Allowlisted app name.")


def lookup_command(name: str, ctx: ToolContext) -> str | None:
    """Return the configured command for ``name`` (case-insensitive) or None."""
    key = name.strip().lower()
    for configured_name, command in ctx.settings.apps.model_dump().items():
        if isinstance(command, str) and str(configured_name).lower() == key and command:
            return command
    return None


def allowlist_names(settings: Any) -> list[str]:
    """Lower-cased configured app names via ``model_dump``.

    ``vars()`` on the Pydantic model only sees the declared fields, so apps
    added through the ``extra='allow'`` model config would be invisible to
    tools that read it that way; ``model_dump()`` sees every configured entry.
    """
    return [str(name).lower() for name in settings.apps.model_dump()]


def split_command(command: str) -> list[str]:
    """Split a command into argv, respecting quoted segments (no shell)."""
    import re

    if '"' not in command:
        return [command]
    parts = re.findall(r'"[^"]*"|\S+', command)
    return [part.strip('"') for part in parts]


def _run_open_app(args: OpenAppArgs, ctx: ToolContext) -> ToolResult:
    command = lookup_command(args.name, ctx)
    if command is None:
        known = sorted(str(k) for k in ctx.settings.apps.model_dump())
        return ToolResult(
            ok=False,
            error=f"unknown app {args.name!r}; known apps: {', '.join(known)}",
            data={"known": known},
        )
    if ctx.dry_run:
        return ToolResult(ok=True, output=f"[dry-run] would open {args.name} ({command})")
    try:
        proc = subprocess.Popen(split_command(command), close_fds=True)
    except OSError as exc:
        return ToolResult(ok=False, error=f"failed to launch {command!r}: {exc}")
    return ToolResult(
        ok=True, output=f"opened {args.name}", data={"command": command, "pid": proc.pid}
    )


def process_running(command: str) -> bool:
    """Return ``True`` when a process with the command's image name is running."""
    import psutil

    image = command.rsplit("\\", 1)[-1].rsplit("/", 1)[-1].lower()
    if not image:
        return False
    for proc in psutil.process_iter(["name"]):
        try:
            if proc.info["name"] and str(proc.info["name"]).lower() == image:
                return True
        except (psutil.Error, OSError):  # process died mid-listing: skip it
            continue
    return False


def _verify_open_app(args: OpenAppArgs, result: ToolResult, ctx: ToolContext) -> ToolResult:
    if not result.ok or ctx.dry_run:
        return result.model_copy(update={"verified": False if not result.ok else None})
    command = str(result.data.get("command") or "")
    deadline = time.monotonic() + _VERIFY_SECONDS
    try:
        while time.monotonic() < deadline:
            if process_running(command):
                return result.model_copy(update={"verified": True})
            time.sleep(_POLL)
    except OSError:
        return result.model_copy(update={"verified": None})
    return result.model_copy(update={"verified": False})


def _describe_open_app(args: OpenAppArgs) -> str:
    return f"Open app '{args.name}'"


def make_open_app_spec() -> ToolSpec:
    """Build the ``open_app`` tool (Tier 0)."""
    return ToolSpec(
        name="open_app",
        description=(
            "Launch an installed application by its configured name "
            "(e.g. notepad, calculator, chrome). Only pre-approved apps are allowed."
        ),
        args_model=OpenAppArgs,
        base_tier=0,
        timeout_s=30,
        run=_run_open_app,
        verify=_verify_open_app,
        describe=_describe_open_app,
    )
