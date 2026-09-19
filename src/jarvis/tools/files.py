"""create_file tool: write a text file inside the allowed roots (Tier 1).

Writing/creating a file is a reversible change, hence Tier 1 confirmation.
The write is atomic (temp file + ``os.replace``), encoded UTF-8, capped at
1 MiB, and only ever touches paths the policy engine has already cleared.
``verify()`` re-checks existence, size and SHA-256 deterministically.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from jarvis.policy import paths
from jarvis.tools.base import ToolContext, ToolResult, ToolSpec

_MAX_CHARS = 1_048_576  # 1 MiB
_CAP = f"content exceeds the {_MAX_CHARS} character limit"


class CreateFileArgs(BaseModel):
    """Create a UTF-8 text file at ``path`` (inside allowed roots)."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=4096, description="Absolute file path.")
    content: str = Field(default="", max_length=_MAX_CHARS, description="Text to write.")
    overwrite: bool = Field(default=False, description="Replace the file if it exists.")


def _resolve(args: CreateFileArgs) -> Path:
    return paths.resolve_safe(args.path)


def _run_create_file(args: CreateFileArgs, ctx: ToolContext) -> ToolResult:
    try:
        target = _resolve(args)
        if not paths.within_any_root(target, paths.roots_from_settings(ctx.settings)):
            return ToolResult(ok=False, error="refusing: path is outside the allowed folders")
    except paths.PathError as exc:
        return ToolResult(ok=False, error=str(exc))

    if target.exists() and not args.overwrite:
        return ToolResult(ok=False, error=f"file exists; pass overwrite=true to replace: {target}")

    if ctx.dry_run:
        return ToolResult(ok=True, output=f"[dry-run] would create {target}")

    data = args.content.encode("utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=".jarvis-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, target)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return ToolResult(
        ok=True,
        output=f"wrote {len(data)} bytes to {target}",
        data={"path": str(target), "size": len(data)},
    )


def _verify_create_file(args: CreateFileArgs, result: ToolResult, ctx: ToolContext) -> ToolResult:
    if ctx.dry_run:
        return result.model_copy(update={"verified": None})
    if not result.ok:
        return result.model_copy(update={"verified": False})
    try:
        target = _resolve(args)
    except paths.PathError:
        return result.model_copy(update={"verified": False})
    if not target.is_file():
        return result.model_copy(update={"verified": False})
    expected = args.content.encode("utf-8")
    if target.stat().st_size != len(expected):
        return result.model_copy(update={"verified": False})
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if digest != hashlib.sha256(expected).hexdigest():
        return result.model_copy(update={"verified": False})
    return result.model_copy(update={"verified": True})


def _describe_create_file(args: CreateFileArgs) -> str:
    try:
        target = _resolve(args)
    except paths.PathError:
        target = args.path
    return f"Create file {target} ({len(args.content)} chars)"


def make_create_file_spec() -> ToolSpec:
    """Build the ``create_file`` tool (Tier 1)."""
    return ToolSpec(
        name="create_file",
        description=(
            "Create a UTF-8 text file at a path inside the user's allowed folders. "
            "Refuses to overwrite an existing file unless overwrite is true."
        ),
        args_model=CreateFileArgs,
        base_tier=1,
        timeout_s=30,
        path_args=("path",),
        run=_run_create_file,
        verify=_verify_create_file,
        describe=_describe_create_file,
    )
