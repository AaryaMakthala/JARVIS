"""Foreground-window info and in-memory capture (Stage 6).

``mss`` grabs the foreground window's on-screen rectangle as raw pixels and
Pillow encodes them to PNG **in memory**; no file is ever written and the bytes
are never logged or checkpointed (I11).  Crop and capture raise on failure so
the caller can fail closed, and the documented behaviour is **no silent
fallback to a full-monitor capture**: if the window rectangle cannot be read
safely, the capture fails and the tools keep the UIA text.  The Windows-only
imports live inside the functions, keeping this module loadable on any host.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

#: GetSystemMetrics indices for the virtual screen origin and size.
_SM_XVIRTUALSCREEN = 76
_SM_YVIRTUALSCREEN = 77
_SM_CXVIRTUALSCREEN = 78
_SM_CYVIRTUALSCREEN = 79


@dataclass(frozen=True)
class ForegroundWindow:
    """Title and owning process name of the window in the foreground.

    ``hwnd`` is resolved once per tool call so the privacy denylist check, the
    UIA read and the rectangle capture always refer to the *same* window: a
    user alt-tab between those steps must never lead to capturing a window
    that was not vetted.  0 means "not a real handle" (tests, no window).
    """

    title: str
    process: str
    hwnd: int = 0


def foreground_window() -> ForegroundWindow:
    """Return the foreground window's title, owning process *name* and handle.

    Raises on failure (the caller fails closed).  ``psutil`` reports a process
    name, never a full path, so a denylist match cannot be fooled by a
    directory.
    """
    import psutil
    import win32gui
    import win32process

    hwnd = ctypes.windll.user32.GetForegroundWindow()
    if not hwnd:
        raise RuntimeError("no foreground window")
    title = (win32gui.GetWindowText(hwnd) or "").strip()
    _thread_id, pid = win32process.GetWindowThreadProcessId(hwnd)
    process = psutil.Process(pid).name()
    return ForegroundWindow(title=title, process=process, hwnd=hwnd)


def _virtual_screen_box(user32: object) -> tuple[int, int, int, int]:
    """Return the virtual screen as ``(left, top, right, bottom)`` in pixels."""
    left = int(user32.GetSystemMetrics(_SM_XVIRTUALSCREEN))
    top = int(user32.GetSystemMetrics(_SM_YVIRTUALSCREEN))
    right = left + int(user32.GetSystemMetrics(_SM_CXVIRTUALSCREEN))
    bottom = top + int(user32.GetSystemMetrics(_SM_CYVIRTUALSCREEN))
    return left, top, right, bottom


def _foreground_bbox(hwnd: int = 0) -> dict[str, int]:
    """Return the foreground window's on-screen pixel box to capture.

    The box is the window rectangle clipped to the virtual screen, in PIXELS.
    Raises ``RuntimeError`` (never falls back to a full-monitor capture) when:

    * there is no foreground window (``hwnd`` 0 and none in focus);
    * the window is minimized (``IsIconic``) - minimised windows report a
      taskbar/restore rectangle that is not the real content;
    * ``GetWindowRect`` fails or returns an empty/inverted rectangle;
    * clipping leaves an empty box (the window is fully off-screen).
    """
    user32 = ctypes.windll.user32
    if not hwnd:
        hwnd = int(user32.GetForegroundWindow())
    if not hwnd:
        raise RuntimeError("no foreground window to capture")
    if user32.IsIconic(hwnd):
        raise RuntimeError("the foreground window is minimized")
    rect = ctypes.wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        raise RuntimeError("could not read the foreground window rectangle")
    left, top, right, bottom = rect.left, rect.top, rect.right, rect.bottom
    if right <= left or bottom <= top:
        raise RuntimeError("the foreground window rectangle is empty or inverted")
    vs_left, vs_top, vs_right, vs_bottom = _virtual_screen_box(user32)
    left, top = max(left, vs_left), max(top, vs_top)
    right, bottom = min(right, vs_right), min(bottom, vs_bottom)
    if right <= left or bottom <= top:
        raise RuntimeError("the foreground window is not visible on the virtual screen")
    return {"left": left, "top": top, "width": right - left, "height": bottom - top}


def capture_foreground(hwnd: int = 0) -> bytes:
    """Capture the foreground window's rectangle as PNG bytes, fully in memory.

    Raises on failure; raises nothing on success.  The returned bytes are never
    written to disk, logged, or put into graph state - they only exist long
    enough to be OCR'd locally or (opt-in) sent to a vision model.  There is
    deliberately **no full-monitor fallback**: a window we cannot crop we do
    not capture.
    """
    return _grab_bbox(_foreground_bbox(hwnd))


def _grab_bbox(bbox: dict[str, int]) -> bytes:
    """Grab pixel ``bbox`` and return PNG bytes, fully in memory."""
    import io

    import mss
    from PIL import Image

    with mss.mss() as sct:
        grabbed = sct.grab(bbox)
        if grabbed.size[0] <= 0 or grabbed.size[1] <= 0:
            raise RuntimeError("the captured rectangle is empty")
        image = Image.frombytes("RGB", grabbed.size, grabbed.rgb)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
