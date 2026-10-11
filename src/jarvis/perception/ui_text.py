"""UI Automation text of the foreground window (Stage 6, local only).

Best-effort and bounded: a window we cannot read yields empty text, and the
caller keeps whatever local OCR found.  Privacy rules are enforced here:

* secure/password controls are never read (B9) - their value is skipped, not
  masked, because a masked value is still a leak we can avoid entirely;
* invisible elements are skipped (B9: hidden UI is never read);
* the walk stops at ``max_elements`` / ``max_chars`` so a huge window cannot
  stall JARVIS.

Everything is wrapped so UIA failures degrade to the text collected so far, and
tests replace :func:`foreground_uia_text` wholesale without touching a desktop.
"""

from __future__ import annotations

from dataclasses import dataclass

MAX_ELEMENTS = 400
MAX_CHARS = 6000


@dataclass(frozen=True)
class UiaText:
    """Extracted text plus counters the tool reports in its diagnostics."""

    text: str = ""
    controls: int = 0
    secure_controls: int = 0


def _is_password_property_id() -> int:
    """Return the UI Automation ``IsPassword`` property id (30019)."""
    from pywinauto.uia_defines import IUIA

    return int(IUIA().UIA_dll.UIA_IsPasswordPropertyId)


def _is_password(element_info: object) -> bool:
    """Return True when a UIA element is a password/secure control.

    ``IsPassword`` is not surfaced by pywinauto's wrapper, so the raw UI
    Automation property and the ``CurrentIsPassword`` accessor are both tried.
    A failure is treated as *not* a password field: a false negative still
    passes through the redaction layer, and aborting the whole walk over one
    COM hiccup would be worse.
    """
    raw = getattr(element_info, "element", None)
    if raw is None:
        return False
    try:
        prop_id = _is_password_property_id()
        if bool(raw.GetCurrentPropertyValue(prop_id)):
            return True
        return bool(raw.CurrentIsPassword)
    except Exception:  # noqa: BLE001 - a COM hiccup must not abort the walk
        return False


def _is_visible(element_info: object) -> bool:
    """Return ``False`` only when UIA positively reports the element hidden."""
    try:
        return getattr(element_info, "visible", True) is not False
    except Exception:  # noqa: BLE001
        return True


def _attr_text(element: object, attr: str) -> str:
    """Read one attribute as text; ``texts`` is joined, anything else fails soft."""
    try:
        value = getattr(element, attr, None)
        if callable(value):
            value = value()
        if isinstance(value, (list, tuple)):
            return " ".join(str(item) for item in value if item).strip()
        return str(value or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def _element_text(element: object) -> str:
    """Return a control's visible text, or "" when it exposes none."""
    for attr in ("window_text", "texts"):
        text = _attr_text(element, attr)
        if text:
            return text
    try:
        info = getattr(element, "element_info", None)
        return str(getattr(info, "name", "") or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def _uia_elements() -> list[object]:
    """Return the foreground window's UIA descendants (the whole tree).

    Extracted so unit tests can inject a fake element list without touching a
    desktop or importing Windows-only modules.
    """
    import ctypes

    from pywinauto import Desktop

    hwnd = ctypes.windll.user32.GetForegroundWindow()
    wrapper = Desktop(backend="uia").window(handle=hwnd)
    return list(wrapper.descendants())


def foreground_uia_text(max_elements: int = MAX_ELEMENTS, max_chars: int = MAX_CHARS) -> UiaText:
    """Walk the foreground window and return its visible, non-secure text.

    Never raises: any UIA failure degrades to whatever was collected so far.
    """
    try:
        elements = _uia_elements()
    except Exception:  # noqa: BLE001
        return UiaText()

    pieces: list[str] = []
    seen: set[str] = set()
    controls = 0
    secure = 0
    total = 0
    for element in elements:
        if controls >= max_elements or total >= max_chars:
            break
        info = getattr(element, "element_info", None)
        if info is None or not _is_visible(info):
            continue
        if _is_password(info):
            secure += 1
            continue
        text = _element_text(element)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        pieces.append(text)
        controls += 1
        total += len(text) + 1
    return UiaText(text="\n".join(pieces)[:max_chars], controls=controls, secure_controls=secure)
