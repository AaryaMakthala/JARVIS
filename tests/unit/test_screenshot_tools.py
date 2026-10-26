"""Screenshot/show-file tests with fakes only: no real screen or viewer."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import pytest

from jarvis.agent import tool_groups
from jarvis.config import CameraSettings, PolicySettings, Settings
from jarvis.policy import tiers
from jarvis.tools.base import ToolContext, ToolResult
from jarvis.tools.registry import build_default_registry
from jarvis.tools.screenshot import (
    SHOWABLE_SUFFIXES,
    ScreenshotError,
    ShowFileArgs,
    TakeScreenshotArgs,
    make_show_file_spec,
    make_take_screenshot_spec,
)


def _png(width: int = 2, height: int = 3) -> bytes:
    """A minimal PNG header with a real IHDR, enough for metadata checks."""
    return (
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\rIHDR"
        + struct.pack(">II", width, height)
        + b"\x08\x02\x00\x00\x00"
    )


class FakeCapture:
    """Capture callable recording the requested scope."""

    def __init__(self, data: bytes | None = None, error: ScreenshotError | None = None) -> None:
        self.data = _png() if data is None else data
        self.error = error
        self.scopes: list[str] = []

    def __call__(self, scope: str) -> bytes:
        self.scopes.append(scope)
        if self.error is not None:
            raise self.error
        return self.data


class RecordingLauncher:
    """Viewer callable that records opened paths."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, path: str) -> None:
        self.calls.append(path)


def _ctx(tmp_path: Path, *, dry_run: bool = False) -> ToolContext:
    root = tmp_path / "root"
    root.mkdir(exist_ok=True)
    settings = Settings(
        policy=PolicySettings(allowed_roots=[str(root)]),
        camera=CameraSettings(save_dir=str(tmp_path / "pictures")),
    )
    return ToolContext(settings=settings, dry_run=dry_run)


def _run_shot(tmp_path: Path, capture: FakeCapture) -> ToolResult:
    spec = make_take_screenshot_spec(capture=capture)
    return spec.run_verified(TakeScreenshotArgs(), _ctx(tmp_path))


def _run_show(path: Path, ctx: ToolContext, launcher: RecordingLauncher) -> ToolResult:
    spec = make_show_file_spec(launcher=launcher)
    return spec.run_verified(ShowFileArgs(path=str(path)), ctx)


def _assert_no_bytes(value: Any) -> None:
    """Fail if any bytes leak into the result tree."""
    assert not isinstance(value, bytes), "pixel bytes leaked into the tool result"
    if isinstance(value, dict):
        for item in value.values():
            _assert_no_bytes(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_no_bytes(item)


def test_registered_tiers_names_and_group_assignment() -> None:
    registry = build_default_registry()
    shot = registry.get("take_screenshot")
    show = registry.get("show_file")
    assert shot.base_tier == tiers.TIER_CONFIRM
    assert shot.windows_only is True
    assert show.base_tier == tiers.TIER_SAFE
    assert show.windows_only is True
    assert show.path_args == ("path",)
    assignment = tool_groups.tool_group_map()
    assert assignment["take_screenshot"] == "capture"
    assert assignment["show_file"] == "capture"


@pytest.mark.parametrize("scope", ["active_window", "full_screen"])
def test_screenshot_saves_png_and_reports_metadata(tmp_path: Path, scope: str) -> None:
    fake = FakeCapture()
    spec = make_take_screenshot_spec(capture=fake)
    result = spec.run_verified(TakeScreenshotArgs(scope=scope), _ctx(tmp_path))
    assert result.ok and result.verified is True
    assert fake.scopes == [scope]
    path = Path(result.data["path"])
    assert path.is_file() and path.suffix == ".png"
    assert path.parent == (tmp_path / "pictures").resolve()
    assert result.data["size"] == len(_png())
    assert result.data["dimensions"] == {"width": 2, "height": 3}
    assert "2x3" in result.output
    _assert_no_bytes(result.data)
    _assert_no_bytes(result.output)


def test_artifact_makes_the_screenshot_referable(tmp_path: Path) -> None:
    result = _run_shot(tmp_path, FakeCapture())
    artifact = result.data["artifact"]
    assert artifact["type"] == "photo"
    assert artifact["path"] == result.data["path"]
    assert Path(artifact["path"]).is_file()


def test_missing_backend_is_a_clean_failure(tmp_path: Path) -> None:
    fake = FakeCapture(error=ScreenshotError("screenshot backend not installed"))
    result = _run_shot(tmp_path, fake)
    assert not result.ok
    assert "not installed" in (result.error or "")
    assert not (tmp_path / "pictures").exists()


def test_invalid_png_is_refused(tmp_path: Path) -> None:
    result = _run_shot(tmp_path, FakeCapture(data=b"not a png"))
    assert not result.ok
    assert "invalid PNG" in (result.error or "")


def test_show_file_refuses_outside_allowed_roots(tmp_path: Path) -> None:
    outside = tmp_path / "outside.png"
    outside.write_bytes(_png())
    launcher = RecordingLauncher()
    result = _run_show(outside, _ctx(tmp_path), launcher)
    assert not result.ok
    assert "outside the allowed folders" in (result.error or "")
    assert launcher.calls == []


@pytest.mark.parametrize("name", ["evil.exe", "run.bat", "script.ps1", "shortcut.lnk"])
def test_show_file_refuses_executables_and_scripts(tmp_path: Path, name: str) -> None:
    root = tmp_path / "root"
    root.mkdir(exist_ok=True)
    target = root / name
    target.write_bytes(b"x")
    launcher = RecordingLauncher()
    result = _run_show(target, _ctx(tmp_path), launcher)
    assert not result.ok
    assert "refusing to show" in (result.error or "")
    assert launcher.calls == []


def test_show_file_opens_allowlisted_image(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir(exist_ok=True)
    target = root / "screen.png"
    target.write_bytes(_png())
    launcher = RecordingLauncher()
    result = _run_show(target, _ctx(tmp_path), launcher)
    assert result.ok
    assert result.output == "opened screen.png"
    assert launcher.calls == [str(target.resolve())]
    assert result.verified is None
    assert "cannot be verified" in result.verify_note


def test_showable_suffixes_are_the_g1_allowlist() -> None:
    assert SHOWABLE_SUFFIXES == {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".bmp",
        ".pdf",
        ".txt",
        ".md",
    }
    assert not (SHOWABLE_SUFFIXES & {".exe", ".bat", ".cmd", ".ps1", ".lnk", ".js", ".vbs"})
