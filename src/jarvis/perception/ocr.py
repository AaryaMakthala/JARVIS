"""Local OCR via PaddleOCR PP-OCRv6-small (CPU only, Stage 6).

Screenshots never leave the PC on this path.  The backend is genuinely optional
(the ``[vision]`` extra): when PaddleOCR, PaddlePaddle, or the PP-OCRv6-small
weights are missing :func:`ocr_available` is ``False`` and the tools keep the
UIA text and report the limitation - there is never a cloud or alternate-engine
fallback for OCR (owner decision, Stage 6).

The model files are loaded from an explicit local directory; nothing is
downloaded at runtime.  Loading the models is expensive, so the engine is built
once per process under a lock and unloaded after an idle period.  The engine
holds no user data between calls and image bytes are passed straight through,
never cached or written to disk.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from jarvis.config import PerceptionSettings, user_data_dir

logger = logging.getLogger("jarvis.perception.ocr")

#: The pinned PP-OCRv6 tier.  PaddleOCR's defaults select the medium tier, so
#: both names must be passed explicitly.
DETECTION_MODEL = "PP-OCRv6_small_det"
RECOGNITION_MODEL = "PP-OCRv6_small_rec"

_lock = threading.Lock()
_inference_lock = threading.Lock()
_engine: Any | None = None
_engine_key: tuple[str, str, int] | None = None
_idle_timer: threading.Timer | None = None
_last_used = 0.0
_inflight = 0


def _module_available(module: str) -> bool:
    """Return ``True`` when ``module`` can be imported without importing it."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def backend_name() -> str | None:
    """Return ``"paddleocr"`` when PaddleOCR and its CPU runtime are installed."""
    if _module_available("paddleocr") and _module_available("paddle"):
        return "paddleocr"
    return None


def _settings(settings: PerceptionSettings | None) -> PerceptionSettings:
    return settings if settings is not None else PerceptionSettings()


def model_root(settings: PerceptionSettings | None = None) -> Path:
    """Return the directory that holds the two PP-OCRv6-small model folders."""
    current = _settings(settings)
    configured = current.ocr_model_dir.strip()
    if not configured:
        return user_data_dir("models", "paddleocr")
    path = Path(configured).expanduser()
    return path if path.is_absolute() else user_data_dir(path)


def model_paths(settings: PerceptionSettings | None = None) -> tuple[Path, Path]:
    """Return the explicit detection and recognition model directories."""
    root = model_root(settings)
    return root / DETECTION_MODEL, root / RECOGNITION_MODEL


def unavailable_reason(settings: PerceptionSettings | None = None) -> str | None:
    """Return why local OCR cannot run, or ``None`` when it is ready.

    The check is cheap and imports nothing heavy, so doctor and the screen tools
    can call it before touching a desktop.
    """
    if not _module_available("paddleocr"):
        return "PaddleOCR is not installed"
    if not _module_available("paddle"):
        return "PaddlePaddle is not installed"
    missing = [path.name for path in model_paths(settings) if not path.is_dir()]
    if missing:
        return "PP-OCRv6-small models are not installed (" + ", ".join(missing) + ")"
    return None


def ocr_available(settings: PerceptionSettings | None = None) -> bool:
    """Return ``True`` when PaddleOCR and the local PP-OCRv6-small weights exist."""
    return unavailable_reason(settings) is None


@dataclass(frozen=True)
class OcrBlock:
    """One recognised text block, its backend confidence (0..1), and box."""

    text: str
    confidence: float
    box: tuple[float, ...] = ()


@dataclass(frozen=True)
class OcrResult:
    """Recognised OCR text plus per-block and mean confidence."""

    text: str
    blocks: tuple[OcrBlock, ...]
    mean_confidence: float | None


def _load_paddleocr_class() -> Any:
    """Import PaddleOCR with model-source network checks disabled."""
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    from paddleocr import PaddleOCR

    return PaddleOCR


def _decode_image(image_bytes: bytes, *, max_bytes: int, max_side: int) -> Any:
    """Decode, bound, and convert an in-memory screenshot to a BGR array."""
    import numpy as np
    from PIL import Image

    if not image_bytes:
        raise ValueError("image is empty")
    if len(image_bytes) > max_bytes:
        raise ValueError("image is larger than the configured OCR limit")
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
            if image.width > max_side or image.height > max_side:
                image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            rgb = image.convert("RGB")
            array = np.asarray(rgb)
    except (OSError, ValueError) as exc:
        raise ValueError("image could not be decoded") from exc
    return np.ascontiguousarray(array[:, :, ::-1])


class _PaddleEngine:
    """Thin adapter around the PaddleOCR general-OCR pipeline."""

    def __init__(self, settings: PerceptionSettings) -> None:
        detection_dir, recognition_dir = model_paths(settings)
        engine_cls = _load_paddleocr_class()
        self._max_image_bytes = settings.ocr_max_image_bytes
        self._max_image_side = settings.ocr_max_image_side
        self._pipeline = engine_cls(
            text_detection_model_name=DETECTION_MODEL,
            text_detection_model_dir=str(detection_dir),
            text_recognition_model_name=RECOGNITION_MODEL,
            text_recognition_model_dir=str(recognition_dir),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_det_limit_side_len=settings.ocr_max_image_side,
            text_det_limit_type="max",
            text_recognition_batch_size=1,
            device="cpu",
            engine="paddle",
        )

    def __call__(self, image_bytes: bytes) -> OcrResult:
        image = _decode_image(
            image_bytes, max_bytes=self._max_image_bytes, max_side=self._max_image_side
        )
        return _parse_predictions(self._pipeline.predict(image))

    def close(self) -> None:
        """Release the underlying PaddleX pipeline when it supports closing."""
        close = getattr(self._pipeline, "close", None)
        if callable(close):
            close()


def _result_field(result: Any, name: str) -> Any:
    """Read a field from a dict-like PaddleX result (never raises)."""
    try:
        return result[name]
    except (KeyError, TypeError, IndexError):
        return getattr(result, name, None)


def _as_sequence(value: Any) -> tuple[Any, ...]:
    """Normalise a backend field to a tuple, treating a scalar string as one item."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    try:
        return tuple(value)
    except TypeError:
        return ()


def _confidence_at(scores: tuple[Any, ...], index: int) -> float:
    """Read a recognition score at ``index`` and clamp it to 0..1."""
    try:
        value = float(scores[index])
    except (IndexError, TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, value))


def _box_at(boxes: tuple[Any, ...], index: int) -> tuple[float, ...]:
    """Flatten one backend box (corner pairs or four coordinates) to floats."""
    try:
        item = boxes[index]
    except (IndexError, TypeError):
        return ()
    flat: list[float] = []

    def _walk(value: Any) -> None:
        if isinstance(value, (list, tuple)):
            for part in value:
                _walk(part)
            return
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            _walk(tolist())
            return
        try:
            flat.append(float(value))
        except (TypeError, ValueError):
            return

    _walk(item)
    return tuple(flat)


def _parse_predictions(predictions: Any) -> OcrResult:
    """Convert PaddleOCR ``rec_texts``/``rec_scores``/``rec_boxes`` to OcrResult."""
    if predictions is None:
        return OcrResult(text="", blocks=(), mean_confidence=None)
    if isinstance(predictions, Mapping):
        predictions = (predictions,)
    blocks: list[OcrBlock] = []
    for result in predictions:
        texts = _as_sequence(_result_field(result, "rec_texts"))
        scores = _as_sequence(_result_field(result, "rec_scores"))
        boxes = _as_sequence(_result_field(result, "rec_boxes"))
        for index, raw_text in enumerate(texts):
            text = str(raw_text).strip()
            if not text:
                continue
            blocks.append(
                OcrBlock(
                    text=text,
                    confidence=_confidence_at(scores, index),
                    box=_box_at(boxes, index),
                )
            )
    text_out = "\n".join(block.text for block in blocks)
    mean = round(sum(block.confidence for block in blocks) / len(blocks), 2) if blocks else None
    return OcrResult(text=text_out, blocks=tuple(blocks), mean_confidence=mean)


def _engine_key_for(settings: PerceptionSettings) -> tuple[str, str, int]:
    detection_dir, recognition_dir = model_paths(settings)
    return str(detection_dir), str(recognition_dir), settings.ocr_max_image_side


def _close_engine_locked() -> None:
    global _engine, _engine_key
    if _engine is not None:
        try:
            _engine.close()
        except Exception as exc:  # noqa: BLE001 - best-effort unload
            logger.debug("PaddleOCR close failed: %s", type(exc).__name__)
    _engine = None
    _engine_key = None


def _cancel_idle_timer_locked() -> None:
    global _idle_timer
    if _idle_timer is not None:
        _idle_timer.cancel()
        _idle_timer = None


def _idle_unload(timer: threading.Timer) -> None:
    """Drop the engine only when this timer is current and no call is active."""
    global _idle_timer
    with _lock:
        if _idle_timer is not timer:
            return
        _idle_timer = None
        if _inflight == 0:
            _close_engine_locked()


def _schedule_idle_unload_locked(settings: PerceptionSettings) -> None:
    global _idle_timer
    _cancel_idle_timer_locked()
    ttl = float(settings.ocr_idle_unload_seconds)
    if ttl <= 0:
        return
    timer: threading.Timer

    def _expire() -> None:
        _idle_unload(timer)

    timer = threading.Timer(ttl, _expire)
    timer.daemon = True
    _idle_timer = timer
    timer.start()


def _get_engine(settings: PerceptionSettings | None = None) -> Any:
    """Build (once) and check out the PaddleOCR engine; raise when unavailable.

    The in-flight count is raised while the lock is held so the idle timer
    cannot unload the engine between the return and the inference call.
    """
    global _engine, _engine_key, _inflight
    current = _settings(settings)
    reason = unavailable_reason(current)
    if reason is not None:
        raise RuntimeError(f"local OCR unavailable: {reason}")
    key = _engine_key_for(current)
    with _lock:
        _cancel_idle_timer_locked()
        if _engine is None or _engine_key != key:
            _close_engine_locked()
            try:
                _engine = _PaddleEngine(current)
            except Exception as exc:
                logger.warning("PaddleOCR initialisation failed: %s", type(exc).__name__)
                raise RuntimeError(f"PaddleOCR could not start: {type(exc).__name__}") from exc
            _engine_key = key
        _inflight += 1
        return _engine


def ocr_image_with_confidence(
    image_bytes: bytes, settings: PerceptionSettings | None = None
) -> OcrResult:
    """Run local OCR and return the text with block and mean confidence.

    Raises when the backend is unavailable or the engine fails; the caller decides
    whether that is fatal (it is not - UIA text is retained).  The mean is
    rounded to two decimals and is ``None`` when the backend returned no blocks.
    """
    global _inflight, _last_used
    current = _settings(settings)
    engine = _get_engine(current)
    try:
        with _inference_lock:
            return engine(image_bytes)
    finally:
        with _lock:
            _inflight = max(0, _inflight - 1)
            _last_used = time.monotonic()
            if _engine is engine:
                _schedule_idle_unload_locked(current)


def ocr_image(image_bytes: bytes, settings: PerceptionSettings | None = None) -> str:
    """Run local OCR and return the recognised text, one line each.

    Raises when the backend is unavailable or the engine fails; the caller decides
    whether that is fatal (it is not - UIA text is retained).
    """
    return ocr_image_with_confidence(image_bytes, settings).text
