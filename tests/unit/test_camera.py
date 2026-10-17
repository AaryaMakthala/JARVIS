"""take_photo tool tests (Stage 8): capture, save, verify, policy boundaries.

No real camera is touched - the capture backend is injected with a fake.  The
decode check is exercised against the OpenCV-less structural path (``_import_cv2``
is monkeypatched to fail), so these tests are deterministic on any machine.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from jarvis.config import CameraSettings, PolicySettings, Settings
from jarvis.policy import tiers
from jarvis.tools import camera
from jarvis.tools.base import ToolContext, ToolResult
from jarvis.tools.camera import CameraError, TakePhotoArgs
from jarvis.tools.default_dirs import photo_dir
from jarvis.tools.registry import _FORBIDDEN_RE, build_default_registry

#: Structurally valid JPEG (SOI ... EOI) for the OpenCV-less decode check.
FAKE_JPEG = b"\xff\xd8" + b"fake-jpeg-payload" + b"\xff\xd9"

_FILENAME_RE = re.compile(r"^photo-\d{8}-\d{6}(-\d+)?\.jpg$")


class FakeCamera:
    """Injectable :class:`CameraBackend` that records captures."""

    def __init__(self, data: bytes = FAKE_JPEG, error: Exception | None = None) -> None:
        self.data = data
        self.error = error
        self.captured = 0

    def capture(self) -> bytes:
        self.captured += 1
        if self.error is not None:
            raise self.error
        return self.data


def _no_cv2() -> Any:
    raise CameraError("OpenCV not installed (test)")


@pytest.fixture(autouse=True)
def _no_opencv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the decode check down the structural path on every test."""
    monkeypatch.setattr(camera, "_import_cv2", _no_cv2)


def _ctx(tmp_path: Path, *, dry_run: bool = False) -> ToolContext:
    settings = Settings(
        policy=PolicySettings(allowed_roots=[str(tmp_path)]),
        camera=CameraSettings(save_dir=str(tmp_path / "photos")),
    )
    return ToolContext(settings=settings, dry_run=dry_run)


def _spec(fake: FakeCamera):
    return camera.make_take_photo_spec(backend_factory=lambda settings: fake)


def _run(fake: FakeCamera, ctx: ToolContext) -> ToolResult:
    return _spec(fake).run_verified(TakePhotoArgs(), ctx)


# ── registration / contract ─────────────────────────────────────────────


def test_registered_tier_and_name_safe() -> None:
    spec = build_default_registry().get("take_photo")
    assert spec.base_tier == tiers.TIER_CONFIRM
    assert spec.windows_only is True
    assert spec.budget_cost == 0
    assert not _FORBIDDEN_RE.search(spec.name)
    assert not _FORBIDDEN_RE.search(spec.description)


# ── success path ────────────────────────────────────────────────────────


def test_success_saves_and_verifies(tmp_path: Path) -> None:
    fake = FakeCamera()
    result = _run(fake, _ctx(tmp_path))
    assert result.ok and result.verified is True
    assert fake.captured == 1
    path = Path(result.data["path"])
    assert path.is_file()
    assert path.read_bytes() == FAKE_JPEG
    assert path.parent == (tmp_path / "photos").resolve()
    assert _FILENAME_RE.match(path.name)


def test_artifact_points_at_saved_file(tmp_path: Path) -> None:
    result = _run(FakeCamera(), _ctx(tmp_path))
    artifact = result.data["artifact"]
    assert artifact["type"] == "photo"
    assert artifact["path"] == result.data["path"]
    assert Path(artifact["path"]).is_file()


def test_camera_dir_is_always_a_root(tmp_path: Path) -> None:
    from jarvis.policy.paths import normalise_key, roots_from_settings

    ctx = _ctx(tmp_path)
    roots = {normalise_key(root) for root in roots_from_settings(ctx.settings)}
    assert normalise_key(photo_dir(ctx.settings)) in roots


# ── dry run ─────────────────────────────────────────────────────────────


def test_dry_run_takes_no_photo(tmp_path: Path) -> None:
    fake = FakeCamera()
    result = _run(fake, _ctx(tmp_path, dry_run=True))
    assert result.ok and result.verified is None
    assert fake.captured == 0
    assert not (tmp_path / "photos").exists()


# ── failure paths ───────────────────────────────────────────────────────


def test_no_camera_is_a_clean_error(tmp_path: Path) -> None:
    fake = FakeCamera(error=CameraError("no camera is available"))
    result = _run(fake, _ctx(tmp_path))
    assert not result.ok
    assert "no camera" in (result.error or "")
    assert not (tmp_path / "photos").exists()


def test_empty_frame_is_rejected(tmp_path: Path) -> None:
    result = _run(FakeCamera(data=b""), _ctx(tmp_path))
    assert not result.ok
    assert "no image data" in (result.error or "")


def test_size_cap_rejects_huge_frame(tmp_path: Path) -> None:
    huge = b"\xff\xd8" + b"x" * (camera._MAX_PHOTO_BYTES + 1) + b"\xff\xd9"
    result = _run(FakeCamera(data=huge), _ctx(tmp_path))
    assert not result.ok
    assert "large" in (result.error or "")


def test_verify_false_when_file_corrupted(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    result = _run(FakeCamera(), ctx)
    assert result.verified is True
    Path(result.data["path"]).write_bytes(b"not a jpeg")
    verified = _spec(FakeCamera()).verify(TakePhotoArgs(), result, ctx)
    assert verified.verified is False


def test_verify_false_when_file_missing(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    result = _run(FakeCamera(), ctx)
    Path(result.data["path"]).unlink()
    verified = _spec(FakeCamera()).verify(TakePhotoArgs(), result, ctx)
    assert verified.verified is False


def test_outside_roots_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(camera.paths, "roots_from_settings", lambda settings: [])
    result = _run(FakeCamera(), _ctx(tmp_path))
    assert not result.ok
    assert "allowed folders" in (result.error or "")


def test_protected_dir_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(camera.paths, "is_protected", lambda path: True)
    result = _run(FakeCamera(), _ctx(tmp_path))
    assert not result.ok
    assert "protected" in (result.error or "")


# ── photo_dir / config ──────────────────────────────────────────────────


def test_photo_dir_uses_configured_save_dir(tmp_path: Path) -> None:
    settings = Settings(camera=CameraSettings(save_dir=str(tmp_path / "shots")))
    assert photo_dir(settings) == (tmp_path / "shots")


def test_photo_dir_defaults_to_pictures_jarvis() -> None:
    location = photo_dir(Settings())
    assert location.name == "JARVIS"
    assert location.parent.name.lower() == "pictures"


def test_camera_save_dir_must_be_absolute() -> None:
    with pytest.raises(ValidationError):
        CameraSettings(save_dir="relative/path")


def test_camera_settings_defaults() -> None:
    camera_settings = Settings().camera
    assert camera_settings.save_dir == ""
    assert camera_settings.device_index == 0
    assert camera_settings.warmup_frames == 5


def test_photo_path_collision_gets_a_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FixedDateTime:
        @staticmethod
        def now(tz: Any = None) -> Any:
            class _Stamp:
                def strftime(self, fmt: str) -> str:
                    return "20260102-030405"

            return _Stamp()

    monkeypatch.setattr(
        camera, "datetime", type("M", (), {"datetime": _FixedDateTime, "UTC": object()})
    )
    first = camera._photo_path(tmp_path)
    assert first.name == "photo-20260102-030405.jpg"
    first.write_bytes(b"x")
    second = camera._photo_path(tmp_path)
    assert second.name == "photo-20260102-030405-1.jpg"
