"""Shared pytest setup for the JARVIS test suite.

``langgraph.checkpoint.serde._msgpack`` reads ``LANGGRAPH_STRICT_MSGPACK`` at
module import time, BEFORE any test module loads langgraph (pytest imports this
file first).  Closing that env-var gap is exactly why ``graph.py`` now passes an
explicit secure serializer too -- see ``tests/unit/test_serializer.py``.

The :func:`_sandbox_default_protected_sources` autouse fixture is the other
piece of harness setup.  Pytest sandboxes (``tmp_path``) live under the OS
temp dir, which on Windows sits inside ``LOCALAPPDATA`` - a real container
protected root - and inside the any-user ``C:\\Users\\*\\AppData`` wildcard
region.  Production policy must keep those protected, so the fixture only
stops the *default* region sources (used when a caller passes no explicit
regions) from covering the OS temp subtree: ``is_protected(path)`` still works,
real OS roots stay protected, and explicitly injected regions are untouched.
The exact/container/wildcard machinery itself is still fully asserted by
``tests/unit/test_paths.py`` with explicit regions.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import pytest

os.environ["LANGGRAPH_STRICT_MSGPACK"] = "true"


@pytest.fixture(autouse=True)
def _sandbox_default_protected_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    """Trim the *default* protected regions away from the pytest temp dir.

    Without this, every ``tmp_path`` sandbox on Windows is refused as inside a
    protected region, because the OS temp dir is a descendant of LOCALAPPDATA
    (container) and of the any-user ``Users\\*\\AppData`` wildcard pattern.
    We keep the exact root list unchanged (exact = equality/ancestor only, so
    USERPROFILE still behaves) and only drop default containers / patterns
    whose region covers the temp subtree.  Explicitly injected regions and the
    OS roots themselves are unaffected.
    """
    from jarvis.policy import paths

    temp = Path(tempfile.gettempdir()).resolve(strict=False)
    temp_key = paths.normalise_key(temp)

    def _covers(key: str) -> bool:
        return temp_key == key or temp_key.startswith(key.rstrip(os.sep) + os.sep)

    original_containers = paths.protected_containers
    original_patterns = paths._wildcard_patterns

    def _containers(settings: object = None) -> list[Path]:
        return [
            root for root in original_containers(settings) if not _covers(paths.normalise_key(root))
        ]

    def _patterns() -> tuple[str, ...]:
        kept: list[str] = []
        for pattern in original_patterns():
            if not paths._pattern_region_hit(temp_key, temp, (pattern,)):
                kept.append(pattern)
        return tuple(kept)

    monkeypatch.setattr(paths, "protected_containers", _containers)
    monkeypatch.setattr(paths, "_wildcard_patterns", _patterns)


@pytest.fixture
def _fast_logging() -> None:
    """Stub: keeps phase-1 tests that import logging trivially importable."""
    _ = logging.getLogger(__name__)
