"""Optional local OCR via RapidOCR + ONNX Runtime (Stage 6).

Screenshots never leave the PC on this path.  The backend is genuinely optional
(the ``[vision]`` extra): when it is not installed :func:`ocr_available` is
``False`` and the tools keep the UIA text and report the limitation - there is
never a cloud fallback for OCR (owner decision, Stage 6).

Loading the ONNX models is expensive, so the engine is built once per process
under a lock.  The engine holds no user data between calls and the image bytes
are passed straight through, never cached.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import threading
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("jarvis.perception.ocr")

#: Import names to probe (newest first).  Both expose a ``RapidOCR`` class.
_BACKENDS = ("rapidocr", "rapidocr_onnxruntime")

_lock = threading.Lock()
_engine: Any | None = None
_engine_name: str | None = None


def backend_name() -> str | None:
    """Return the import name of an installed RapidOCR backend, or ``None``."""
    for name in _BACKENDS:
        try:
            if importlib.util.find_spec(name) is not None:
                return name
        except (ImportError, ValueError):
            continue
    return None


def ocr_available() -> bool:
    """Return ``True`` when a local OCR backend is installed."""
    return backend_name() is not None


def _get_engine() -> Any:
    """Build (once) and return the RapidOCR engine; raise when unavailable."""
    global _engine, _engine_name
    name = backend_name()
    if name is None:
        raise RuntimeError("no local OCR backend is installed")
    with _lock:
        if _engine is None or _engine_name != name:
            module = importlib.import_module(name)
            engine_cls = getattr(module, "RapidOCR", None)
            if engine_cls is None:
                raise RuntimeError(f"{name} does not expose a RapidOCR class")
            _engine = engine_cls()
            _engine_name = name
    return _engine


@dataclass(frozen=True)
class OcrBlock:
    """One recognised text block and its backend confidence (0..1)."""

    text: str
    confidence: float


@dataclass(frozen=True)
class OcrResult:
    """Recognised OCR text plus per-block and mean confidence."""

    text: str
    blocks: tuple[OcrBlock, ...]
    mean_confidence: float | None


def ocr_image_with_confidence(image_bytes: bytes) -> OcrResult:
    """Run local OCR and return the text with block and mean confidence.

    Raises when no backend is installed or the engine fails; the caller decides
    whether that is fatal (it is not - UIA text is retained).  The mean is
    rounded to two decimals and is ``None`` when the backend returned no blocks.
    """
    result = _get_engine()(image_bytes)
    boxes = result[0] if isinstance(result, tuple) else result
    if not boxes:
        return OcrResult(text="", blocks=(), mean_confidence=None)
    blocks: list[OcrBlock] = []
    for item in boxes:
        try:
            text = str(item[1]).strip()
        except (IndexError, TypeError):
            continue
        if not text:
            continue
        confidence = _confidence(item)
        blocks.append(OcrBlock(text=text, confidence=confidence))
    text_out = "\n".join(block.text for block in blocks)
    mean = round(sum(block.confidence for block in blocks) / len(blocks), 2) if blocks else None
    return OcrResult(text=text_out, blocks=tuple(blocks), mean_confidence=mean)


def _confidence(item: Any) -> float:
    """Read a block score from ``item[2]`` and clamp it to 0..1 (default 0)."""
    try:
        value = float(item[2])
    except (IndexError, TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, value))


def ocr_image(image_bytes: bytes) -> str:
    """Run local OCR and return the recognised text, one line each.

    Raises when no backend is installed or the engine fails; the caller decides
    whether that is fatal (it is not - UIA text is retained).
    """
    return ocr_image_with_confidence(image_bytes).text
