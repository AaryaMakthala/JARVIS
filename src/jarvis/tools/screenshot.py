"""Screenshot capture and safe file display (Tier 1 / Tier 0).

``take_screenshot`` captures the active window or the full virtual screen and
writes one PNG into JARVIS's Pictures folder (``photo_dir``).  Only the path,
byte size and pixel dimensions are returned; pixel bytes never enter a result,
the graph state, the task log or a prompt.  ``mss``/Pillow stay optional: the
default capture imports them lazily and a missing backend is a clean tool
failure.

``show_file`` hands one image/PDF/text file inside an allowed root to the OS
default viewer.  It is a strict allowlist (no executables or scripts) and is
deliberately separate from ``open_path`` so the new tool's surface is exactly
the files named by the G1 specification.
"""

from __future__ import annotations

import datetime
import os
import struct
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from jarvis.policy import paths, tiers
from jarvis.tools.base import ToolContext, ToolResult, ToolSpec
from jarvis.tools.default_dirs import photo_dir

__all__ = [
    "ScreenshotError",
    "ShowFileArgs",
    "TakeScreenshotArgs",
    "make_show_file_spec",
    "make_take_screenshot_spec",
]

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_MAX_SCREENSHOT_BYTES = 40 * 1024 * 1024

#: Suffixes ``show_file`` may hand to the OS default viewer.  Matched
#: case-insensitively on the resolved path; everything else is refused.
SHOWABLE_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".pdf", ".txt", ".md"}
)


class ScreenshotError(RuntimeError):
    """The screenshot backend is unavailable or the capture failed."""


class TakeScreenshotArgs(BaseModel):
    """Capture the active window (default) or the full virtual screen."""

    model_config = ConfigDict(extra="forbid")

    scope: str = Field(
        default="active_window",
        pattern="^(active_window|full_screen)$",
        description="active_window (default) or full_screen.",
    )


class ShowFileArgs(BaseModel):
    """Open one allowlisted image/PDF/text file with the default viewer."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Absolute path of the file to open.")


def _import_screen() -> Any:
    """Import the optional capture module or raise a clean tool error."""
    try:
        from jarvis.perception import screen
    except ImportError as exc:
        raise ScreenshotError("screenshot backend not installed") from exc
    return screen


def _default_capture(scope: str) -> bytes:
    """Capture ``scope`` with mss/Pillow; fail clearly when they are absent."""
    screen = _import_screen()
    try:
        if scope == "full_screen":
            return screen.capture_full_screen()
        return screen.capture_foreground()
    except ImportError as exc:
        raise ScreenshotError("screenshot backend not installed") from exc
    except RuntimeError as exc:
        raise ScreenshotError(str(exc)) from exc


def _png_dimensions(data: bytes) -> tuple[int, int] | None:
    """Return ``(width, height)`` from a PNG IHDR, or ``None`` when malformed."""
    if len(data) < 24 or data[:8] != _PNG_SIGNATURE or data[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", data[16:24])
    if width <= 0 or height <= 0:
        return None
    return width, height


def _screenshot_path(directory: Path) -> Path:
    """A ``screenshot-YYYYmmdd-HHMMSS.png`` path, disambiguated on collision."""
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d-%H%M%S")
    candidate = directory / f"screenshot-{stamp}.png"
    if not candidate.exists():
        return candidate
    for suffix in range(1, 100):
        candidate = directory / f"screenshot-{stamp}-{suffix}.png"
        if not candidate.exists():
            return candidate
    return directory / f"screenshot-{stamp}-{os.getpid()}.png"


def _save_png(directory: Path, data: bytes) -> Path:
    """Atomically write ``data`` as a PNG into ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    path = _screenshot_path(directory)
    fd, tmp = tempfile.mkstemp(dir=str(directory), prefix=".jarvis-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, path)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def _run_take_screenshot(
    args: TakeScreenshotArgs,
    ctx: ToolContext,
    capture: Callable[[str], bytes],
) -> ToolResult:
    if ctx.dry_run:
        return ToolResult(
            ok=True, output=f"[dry-run] would take a {args.scope.replace('_', ' ')} screenshot"
        )
    directory = photo_dir(ctx.settings)
    try:
        directory = paths.resolve_safe(str(directory))
    except paths.PathError as exc:
        return ToolResult(ok=False, error=f"the screenshot folder is unusable: {exc}")
    if not paths.within_any_root(directory, paths.roots_from_settings(ctx.settings)):
        return ToolResult(
            ok=False, error=f"the screenshot folder is outside the allowed folders: {directory}"
        )
    if paths.is_protected(directory, settings=ctx.settings):
        return ToolResult(ok=False, error=f"the screenshot folder is protected: {directory}")

    try:
        data = capture(args.scope)
    except ScreenshotError as exc:
        return ToolResult(ok=False, error=str(exc))
    if not isinstance(data, bytes) or not data:
        return ToolResult(ok=False, error="the screenshot backend returned no image data")
    if len(data) > _MAX_SCREENSHOT_BYTES:
        return ToolResult(ok=False, error="the screenshot is unexpectedly large; not saved")
    dimensions = _png_dimensions(data)
    if dimensions is None:
        return ToolResult(ok=False, error="the screenshot backend returned an invalid PNG")

    try:
        path = _save_png(directory, data)
    except OSError as exc:
        return ToolResult(ok=False, error=f"could not save the screenshot: {exc}")
    width, height = dimensions
    return ToolResult(
        ok=True,
        output=f"Saved a screenshot to {path} ({width}x{height}, {len(data)} bytes).",
        data={
            "path": str(path),
            "size": len(data),
            "dimensions": {"width": width, "height": height},
            "artifact": {"path": str(path), "type": "photo"},
        },
    )


def _verify_take_screenshot(
    args: TakeScreenshotArgs, result: ToolResult, ctx: ToolContext
) -> ToolResult:
    """Real verification: exists, non-empty, valid PNG, expected dimensions."""
    del args
    if ctx.dry_run:
        return result.model_copy(update={"verified": None})
    if not result.ok:
        return result.model_copy(update={"verified": False})
    raw = result.data.get("path")
    if not isinstance(raw, str) or not raw:
        return result.model_copy(update={"verified": False})
    path = Path(raw)
    try:
        if not path.is_file():
            return result.model_copy(update={"verified": False})
        size = path.stat().st_size
        if size <= 0 or size > _MAX_SCREENSHOT_BYTES:
            return result.model_copy(update={"verified": False})
        with path.open("rb") as handle:
            head = handle.read(24)
    except OSError:
        return result.model_copy(update={"verified": False})
    dimensions = _png_dimensions(head)
    if dimensions is None:
        return result.model_copy(update={"verified": False})
    expected = result.data.get("dimensions")
    if isinstance(expected, dict):
        okay = width_height_matches(dimensions, expected)
        return result.model_copy(update={"verified": okay})
    return result.model_copy(update={"verified": True})


def width_height_matches(dimensions: tuple[int, int], expected: dict[str, Any]) -> bool:
    """Whether the parsed dimensions equal the result's recorded dimensions."""
    return dimensions == (
        int(expected.get("width") or 0),
        int(expected.get("height") or 0),
    )


def _describe_take_screenshot(args: TakeScreenshotArgs) -> str:
    scope = "active window" if args.scope == "active_window" else "whole screen"
    return f"Take a screenshot of the {scope} and save it locally"


def make_take_screenshot_spec(
    capture: Callable[[str], bytes] | None = None,
) -> ToolSpec:
    """Build the ``take_screenshot`` tool (Tier 1, Windows only)."""
    grab = capture or _default_capture
    return ToolSpec(
        name="take_screenshot",
        description=(
            "Take a screenshot of the active window (default) or the whole screen and "
            "save it as a PNG in JARVIS's Pictures folder. Only the file path, size and "
            "dimensions are reported; the image stays local."
        ),
        args_model=TakeScreenshotArgs,
        base_tier=tiers.TIER_CONFIRM,
        windows_only=True,
        timeout_s=30,
        run=lambda args, ctx: _run_take_screenshot(args, ctx, grab),
        verify=_verify_take_screenshot,
        describe=_describe_take_screenshot,
    )


def _launch_default_viewer(path: str) -> None:
    """Open ``path`` with the OS default viewer (Windows ``os.startfile``)."""
    from jarvis.platform_guard import require_windows

    require_windows("Showing a file")
    startfile = getattr(os, "startfile", None)
    if startfile is None:  # pragma: no cover - only on a mis-reported platform
        raise OSError("no default viewer is available on this platform")
    startfile(path)


def _run_show_file(args: ShowFileArgs, ctx: ToolContext, launcher: Callable[[str], None]) -> ToolResult:
    if ctx.dry_run:
        return ToolResult(ok=True, output=f"[dry-run] would show {args.path}")
    try:
        resolved = paths.resolve_safe(args.path)
    except paths.PathError as exc:
        return ToolResult(ok=False, error=str(exc))
    if not resolved.is_file():
        return ToolResult(ok=False, error=f"file does not exist: {resolved}")
    suffix = resolved.suffix.casefold()
    if suffix not in SHOWABLE_SUFFIXES:
        allowed = ", ".join(sorted(SHOWABLE_SUFFIXES))
        return ToolResult(
            ok=False,
            error=f"refusing to show {suffix or 'this file'!r}; allowed types: {allowed}",
        )
    if paths.is_protected(resolved, settings=ctx.settings):
        return ToolResult(ok=False, error=f"path is protected: {resolved}")
    if not paths.within_any_root(resolved, paths.roots_from_settings(ctx.settings)):
        return ToolResult(ok=False, error=f"path is outside the allowed folders: {resolved}")
    try:
        launcher(str(resolved))
    except OSError as exc:
        return ToolResult(ok=False, error=f"failed to show {resolved.name}: {exc}")
    return ToolResult(
        ok=True,
        output=f"opened {resolved.name}",
        data={"file": str(resolved)},
    )


def _verify_show_file(args: ShowFileArgs, result: ToolResult, ctx: ToolContext) -> ToolResult:
    """Honest verification: the launch is observable, the display is not."""
    if not result.ok:
        return result.model_copy(update={"verified": False})
    if ctx.dry_run:
        return result.model_copy(update={"verified": None})
    file_name = Path(str(result.data.get("file") or args.path)).name or args.path
    note = f"handed {file_name} to the default viewer; opening it cannot be verified"
    return result.model_copy(update={"verified": None, "verify_note": note})


def _describe_show_file(args: ShowFileArgs) -> str:
    target = args.path
    try:
        target = str(paths.resolve_safe(args.path))
    except paths.PathError:
        pass
    return f"Show {target} with the default viewer"


def make_show_file_spec(launcher: Callable[[str], None] | None = None) -> ToolSpec:
    """Build the ``show_file`` tool (Tier 0, Windows only)."""
    launch = launcher or _launch_default_viewer
    return ToolSpec(
        name="show_file",
        description=(
            "Open an image, PDF or text file from the allowed folders with the operating "
            "system's default viewer. Executables and scripts are refused."
        ),
        args_model=ShowFileArgs,
        base_tier=tiers.TIER_SAFE,
        windows_only=True,
        path_args=("path",),
        run=lambda args, ctx: _run_show_file(args, ctx, launch),
        verify=_verify_show_file,
        describe=_describe_show_file,
    )
