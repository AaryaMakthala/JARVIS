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


def ocr_image(image_bytes: bytes) -> str:
    """Run local OCR on PNG bytes and return the recognised text, one line each.

    Raises when no backend is installed or the engine fails; the caller decides
    whether that is fatal (it is not - UIA text is retained).
    """
    result = _get_engine()(image_bytes)
    boxes = result[0] if isinstance(result, tuple) else result
    if not boxes:
        return ""
    lines: list[str] = []
    for item in boxes:
        try:
            text = str(item[1]).strip()
        except (IndexError, TypeError):
            continue
        if text:
            lines.append(text)
    return "\n".join(lines)
