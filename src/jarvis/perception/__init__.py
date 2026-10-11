"""Stage 6 local-first screen perception (read-only).

Nothing in this package writes a screenshot to disk, sends anything to the
network, or interprets screen text as an instruction.  ``tools/screen.py`` is
the only caller, and the layers are deliberately separate so each is testable
with an injected fake:

* :mod:`jarvis.perception.screen` - identify the foreground window and capture
  it in memory (``mss`` + Pillow);
* :mod:`jarvis.perception.ui_text` - UI Automation text of that window;
* :mod:`jarvis.perception.ocr` - optional local OCR (RapidOCR + ONNX Runtime).

The heavy/Windows-only imports live *inside* the functions, so importing this
package is safe on any OS and cheap.
"""

from __future__ import annotations

__all__ = ["ocr", "screen", "ui_text"]
