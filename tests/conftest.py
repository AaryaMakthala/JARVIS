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


@pytest.fixture(autouse=True)
def _block_dangerous_calls_in_tests(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    """Block subprocess/os/webbrowser/pyautogui/pywinauto/cv2/sounddevice calls during tests."""
    markers = {m.name for m in request.node.iter_markers()}
    if "live" in markers or "windows_only" in markers:
        return

    import subprocess as _subprocess
    import os as _os
    import webbrowser as _webbrowser

    def _blocked(*args, **kwargs):
        raise RuntimeError("blocked in tests")

    # Allow existing tests that intentionally use subprocess.run/call in controlled ways
    # by not blocking if test module is in allowlist? Keep strict but some tests need to run in same process
    test_file = str(request.node.fspath)
    allow_subprocess = (
        "test_daemon_single_instance" in test_file
        or "test_tools_dictation" in test_file
        or "test_tools_open_in_app" in test_file
        or "test_tools_apps" in test_file
        or "test_memory.py" in test_file and "embed_is_process_stable" in request.node.name
    )
    if not allow_subprocess:
        monkeypatch.setattr(_subprocess, "Popen", _blocked)
        monkeypatch.setattr(_subprocess, "run", _blocked)
        monkeypatch.setattr(_subprocess, "call", _blocked)
        monkeypatch.setattr(_subprocess, "check_call", _blocked)
        monkeypatch.setattr(_subprocess, "check_output", _blocked)
    else:
        pass

    try:
        monkeypatch.setattr(_os, "startfile", _blocked)
    except Exception:
        pass

    try:
        monkeypatch.setattr(_webbrowser, "open", _blocked)
        monkeypatch.setattr(_webbrowser, "open_new", _blocked)
        monkeypatch.setattr(_webbrowser, "open_new_tab", _blocked)
    except Exception:
        pass

    try:
        import pyautogui as _pyautogui

        monkeypatch.setattr(_pyautogui, "hotkey", _blocked)
        monkeypatch.setattr(_pyautogui, "press", _blocked)
        monkeypatch.setattr(_pyautogui, "typewrite", _blocked)
        monkeypatch.setattr(_pyautogui, "click", _blocked)
        monkeypatch.setattr(_pyautogui, "moveTo", _blocked)
    except Exception:
        pass

    try:
        import pywinauto as _pywinauto

        monkeypatch.setattr(_pywinauto, "Desktop", _blocked)
    except Exception:
        pass

    try:
        import cv2 as _cv2

        monkeypatch.setattr(_cv2, "VideoCapture", _blocked)
    except Exception:
        pass

    try:
        import sounddevice as _sd

        monkeypatch.setattr(_sd, "play", _blocked)
        monkeypatch.setattr(_sd, "rec", _blocked)
        monkeypatch.setattr(_sd, "record", _blocked)
        monkeypatch.setattr(_sd, "InputStream", _blocked)
        monkeypatch.setattr(_sd, "OutputStream", _blocked)
    except Exception:
        pass


@pytest.fixture
def _fast_logging() -> None:
    """Stub: keeps phase-1 tests that import logging trivially importable."""
    _ = logging.getLogger(__name__)
