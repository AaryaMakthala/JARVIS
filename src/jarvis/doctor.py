"""Read-only readiness checks for optional desktop backends.

``jarvis doctor`` calls :func:`optional_backend_checks` so the table says which
optional capability is usable *before* a task tries it.  Every probe is an
``importlib.util.find_spec`` (or the equally cheap
:func:`jarvis.perception.ocr.backend_name`): nothing is imported for its side
effects, no device or network is touched, and no user data is read.

A missing extra is a WARN, never a FAIL - the affected tool reports a clear
error and the rest of JARVIS keeps working.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterable

from jarvis.checks import Check, _check
from jarvis.config import PerceptionSettings
from jarvis.perception.ocr import backend_name, unavailable_reason

__all__ = ["optional_backend_checks"]


def _has(module: str) -> bool:
    """Return ``True`` when ``module`` can be imported (without importing it)."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _required(
    name: str,
    modules: Iterable[str],
    *,
    ready: str,
    fix: str,
) -> Check:
    """PASS when every module is present, otherwise WARN naming what is missing."""
    missing = [module for module in modules if not _has(module)]
    if not missing:
        return _check(name, "PASS", ready)
    return _check(name, "WARN", f"missing {', '.join(missing)} - {fix}")


def optional_backend_checks(
    perception: PerceptionSettings | None = None,
) -> list[Check]:
    """Return one row per optional backend, in a stable order."""
    current = perception if perception is not None else PerceptionSettings()
    reason = unavailable_reason(current)
    if reason is None:
        ocr_check = _check("ocr", "PASS", f"ready ({backend_name()}, PP-OCRv6-small, local CPU)")
    else:
        ocr_check = _check(
            "ocr",
            "WARN",
            f'{reason} - read_screen returns UIA text only; pip install -e ".[vision]"',
        )
    return [
        _required(
            "screenshot",
            ("mss", "PIL"),
            ready="ready (mss + Pillow)",
            fix='pip install -e ".[vision]"',
        ),
        ocr_check,
        _required(
            "volume",
            ("pycaw",),
            ready="ready (pycaw)",
            fix='on Windows: pip install -e ".[tools-extra]"',
        ),
        _required(
            "playwright",
            ("playwright",),
            ready="package installed (run `playwright install chromium` if needed)",
            fix='pip install -e ".[browser]"',
        ),
    ]
