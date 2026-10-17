"""take_photo tool: capture one still from the computer's camera (Tier 1).

Stage 8.  The photo is written to :func:`jarvis.tools.default_dirs.photo_dir`
(``Pictures/JARVIS`` by default, or ``camera.save_dir``) as a JPEG named
``photo-YYYYmmdd-HHMMSS.jpg``.  That folder is always an allowed root
(``policy.paths.roots_from_settings``), so the picture can be viewed with
``open_path`` and deleted with ``delete_path`` straight afterwards.

Privacy (docs/03): the camera is on only for the moment of the capture, the
image never leaves a local file, and nothing about it is logged (no bytes, no
base64, no path contents beyond the filename).  A Tier-1 confirmation carrying
a spoken notice is required before the camera is touched.

Verification is real (not best-effort): the saved file must exist, be
non-empty, and decode as a JPEG.  OpenCV is imported lazily so merely importing
this module (and every other tool) works where the optional ``[camera]`` extra
is not installed; the capture backend is injectable so unit tests never touch
real hardware.
"""

from __future__ import annotations

import datetime
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from jarvis.policy import paths, tiers
from jarvis.tools.base import ToolContext, ToolResult, ToolSpec
from jarvis.tools.default_dirs import location_label, photo_dir

__all__ = [
    "CameraBackend",
    "CameraError",
    "OpenCvCamera",
    "TakePhotoArgs",
    "make_take_photo_spec",
]

#: Spoken/confirmation notice (the Tier-1 prompt carries it).
PRIVACY_NOTICE = (
    "Take a photo with the camera. The camera is on briefly and the photo is "
    "saved only on this computer."
)

#: Refuse a capture bigger than this (a corrupt/runaway frame must not be saved).
_MAX_PHOTO_BYTES = 25 * 1024 * 1024

#: JPEG start/end-of-image markers, used by the decode check when OpenCV is
#: absent (unit tests) - the live path always has OpenCV and really decodes.
_JPEG_SOI = b"\xff\xd8"
_JPEG_EOI = b"\xff\xd9"


class CameraError(RuntimeError):
    """A camera could not be opened or produced no usable frame."""


class CameraBackend(Protocol):
    """A still-capture source returning encoded JPEG bytes.

    Implemented by :class:`OpenCvCamera` (live) and by test fakes.
    """

    def capture(self) -> bytes:
        """Return JPEG-encoded image bytes; raise :class:`CameraError` on failure."""


@dataclass
class OpenCvCamera:
    """Live camera backend over OpenCV ``VideoCapture`` (Windows ``CAP_DSHOW``).

    A few frames are discarded first so auto-exposure/white-balance settle - the
    first frames out of a webcam are routinely dark or green.
    """

    device_index: int = 0
    warmup_frames: int = 5

    def capture(self) -> bytes:
        """Open the camera, settle it, and return one JPEG frame."""
        cv2 = _import_cv2()
        cap = cv2.VideoCapture(int(self.device_index), cv2.CAP_DSHOW)
        try:
            if not cap.isOpened():
                raise CameraError("no camera is available")
            frame: Any | None = None
            for _ in range(max(1, int(self.warmup_frames)) + 1):
                ok, image = cap.read()
                if ok and image is not None:
                    frame = image
            if frame is None:
                raise CameraError("the camera returned no frame")
            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                raise CameraError("could not encode the photo")
            return bytes(buffer.tobytes())
        finally:
            cap.release()


class TakePhotoArgs(BaseModel):
    """No arguments: one still from the default camera."""

    model_config = ConfigDict(extra="forbid")


def _import_cv2() -> Any:
    """Import OpenCV lazily; raise a clean :class:`CameraError` when missing."""
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise CameraError(
            "OpenCV is not installed; run `pip install jarvis[camera]` so JARVIS can use the camera"
        ) from exc
    return cv2


def _decodes_jpeg(data: bytes) -> bool:
    """Whether ``data`` is a decodable JPEG.

    Uses OpenCV when present (the live case); otherwise a structural start/end
    marker check, so verification is meaningful even in unit tests without the
    optional extra.
    """
    if not data:
        return False
    try:
        cv2 = _import_cv2()
    except CameraError:
        cv2 = None
    if cv2 is not None:
        try:
            import numpy as np  # type: ignore[import-not-found]

            array = np.frombuffer(data, dtype=np.uint8)
            return bool(cv2.imdecode(array, cv2.IMREAD_COLOR) is not None)
        except Exception:  # noqa: BLE001 - an undecomposable image is a failure
            return False
    return data[:2] == _JPEG_SOI and data[-2:] == _JPEG_EOI


def _photo_path(directory: Path) -> Path:
    """A ``photo-YYYYmmdd-HHMMSS.jpg`` path, disambiguated on collision."""
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d-%H%M%S")
    candidate = directory / f"photo-{stamp}.jpg"
    if not candidate.exists():
        return candidate
    for suffix in range(1, 100):
        candidate = directory / f"photo-{stamp}-{suffix}.jpg"
        if not candidate.exists():
            return candidate
    return directory / f"photo-{stamp}-{os.getpid()}.jpg"


def _save_photo(directory: Path, data: bytes) -> Path:
    """Atomically write the JPEG into ``directory`` and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = _photo_path(directory)
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


def _run_take_photo(
    args: TakePhotoArgs,
    ctx: ToolContext,
    backend_factory: Callable[[Any], CameraBackend],
) -> ToolResult:
    del args
    if ctx.dry_run:
        return ToolResult(ok=True, output="[dry-run] would take a photo with the camera")
    directory = photo_dir(ctx.settings)
    try:
        directory = paths.resolve_safe(str(directory))
    except paths.PathError as exc:
        return ToolResult(ok=False, error=f"the photo folder is unusable: {exc}")
    if not paths.within_any_root(directory, paths.roots_from_settings(ctx.settings)):
        return ToolResult(
            ok=False,
            error=(
                f"the photo folder {directory} is outside the allowed folders; "
                "add it to policy.allowed_roots (or set camera.save_dir)"
            ),
        )
    if paths.is_protected(directory):
        return ToolResult(ok=False, error=f"the photo folder is protected: {directory}")

    try:
        backend = backend_factory(ctx.settings)
    except CameraError as exc:
        return ToolResult(ok=False, error=str(exc))
    try:
        data = backend.capture()
    except CameraError as exc:
        return ToolResult(ok=False, error=str(exc))
    if not isinstance(data, bytes) or not data:
        return ToolResult(ok=False, error="the camera returned no image data")
    if len(data) > _MAX_PHOTO_BYTES:
        return ToolResult(ok=False, error="the photo is unexpectedly large; not saved")

    try:
        path = _save_photo(directory, data)
    except OSError as exc:
        return ToolResult(ok=False, error=f"could not save the photo: {exc}")
    return ToolResult(
        ok=True,
        output=f"Took a photo and saved it to {location_label(path, ctx.settings)}.",
        data={
            "path": str(path),
            "size": len(data),
            "artifact": {"path": str(path), "type": "photo"},
        },
    )


def _verify_take_photo(args: TakePhotoArgs, result: ToolResult, ctx: ToolContext) -> ToolResult:
    """Real verification: exists, non-empty, and decodes as a JPEG."""
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
        if size <= 0 or size > _MAX_PHOTO_BYTES:
            return result.model_copy(update={"verified": False})
        with path.open("rb") as handle:
            head = handle.read()
    except OSError:
        return result.model_copy(update={"verified": False})
    return result.model_copy(update={"verified": _decodes_jpeg(head)})


def _describe_take_photo(args: TakePhotoArgs) -> str:
    del args
    return PRIVACY_NOTICE


def make_take_photo_spec(
    backend_factory: Callable[[Any], CameraBackend] | None = None,
) -> ToolSpec:
    """Build the ``take_photo`` tool (Tier 1, Windows only).

    ``backend_factory(settings) -> CameraBackend`` is an injection point for
    tests; the default builds a live :class:`OpenCvCamera` from the camera
    settings.
    """
    factory = backend_factory or (
        lambda settings: OpenCvCamera(
            device_index=int(getattr(settings.camera, "device_index", 0)),
            warmup_frames=int(getattr(settings.camera, "warmup_frames", 5)),
        )
    )
    return ToolSpec(
        name="take_photo",
        description=(
            "Take a photo with the computer's camera and save it to JARVIS's "
            "Pictures folder. The camera is on only for the moment of the shot; "
            "the picture is stored locally and nothing is uploaded."
        ),
        args_model=TakePhotoArgs,
        base_tier=tiers.TIER_CONFIRM,
        windows_only=True,
        timeout_s=30,
        run=lambda args, ctx: _run_take_photo(args, ctx, factory),
        verify=_verify_take_photo,
        describe=_describe_take_photo,
    )
