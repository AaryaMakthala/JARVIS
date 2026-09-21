"""Real window-focus checker (daemon side of the dictation focus gate).

Checks whether a given application window currently has foreground focus using
lazy-imported Windows APIs (pywinauto / ctypes).  Every read fails closed:
if the API cannot be reached or the window title cannot be read, the answer is
``False`` so the voice loop pauses dictation rather than typing into the wrong
window (docs/03 §1, docs/04 §2.5).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class WindowFocusChecker:
    """Deterministic wrapper over the Win32 foreground window."""

    def get_foreground_title(self) -> str:
        """Return the current foreground window title ('' when unreadable)."""
        try:
            import ctypes

            user32 = ctypes.windll.user32  # type: ignore[attr-defined]
            hwnd = user32.GetForegroundWindow()
            buf = ctypes.create_unicode_buffer(256)
            user32.GetWindowTextW(hwnd, buf, 256)
            return buf.value or ""
        except Exception:  # fail closed on any read error
            logger.warning("could not read foreground window title", exc_info=True)
            return ""

    def is_foreground(self, app_name: str) -> bool:
        """Return True when ``app_name`` is (part of) the foreground title.

        Fails closed: an unreadable foreground window or a null handle means
        the app is *not* confirmed focused.
        """
        title = self.get_foreground_title()
        if not title:
            return False
        return app_name.lower() in title.lower()
