"""Real-Recycle-Bin round-trip (Windows only).

Runs only under ``pytest -m windows_only`` on the dev PC.  The default CI set
excludes it because it touches the real ``$Recycle.Bin`` - everything else in
the suite sticks to :class:`FakeDirTrash` (docs/07_TESTING_AND_BENCHMARK.md).
"""

from __future__ import annotations

import os
import platform
from pathlib import Path

import pytest
import win32api

from jarvis.policy import paths
from jarvis.tools.files import WindowsRecycleBinTrash


@pytest.mark.windows_only
def test_recycle_bin_round_trip(tmp_path: Path) -> None:
    if platform.system() != "Windows":
        pytest.skip("native Recycle Bin only exists on Windows")
    target = tmp_path / "recycle-me.txt"
    target.write_text("into the bin and back", encoding="utf-8")

    trash = WindowsRecycleBinTrash()
    record = trash.send(target)
    assert record.error is None, record.error
    assert target.exists() is False

    assert trash.restore(record) is True
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "into the bin and back"


@pytest.mark.windows_only
def test_long_path_expansion_of_short_names(tmp_path: Path) -> None:
    if platform.system() != "Windows":
        pytest.skip("8.3 short names only exist on Windows")
    long_dir = tmp_path / "A folder with a long name"
    long_dir.mkdir()
    short = win32api.GetShortPathName(str(long_dir))
    if os.path.basename(short) == os.path.basename(str(long_dir)):
        pytest.skip("8.3 short names are disabled on this volume")
    resolved = paths.resolve_safe(short)
    assert resolved == long_dir.resolve(strict=False)


@pytest.mark.windows_only
def test_resolve_safe_follows_junctions(tmp_path: Path) -> None:
    if platform.system() != "Windows":
        pytest.skip("junctions only exist on Windows")
    real_dir = tmp_path / "real_target"
    real_dir.mkdir()
    link = tmp_path / "link_to_real"
    rc = os.system(f'mklink /J "{link}" "{real_dir}"')
    if rc != 0:
        pytest.skip("junction creation is unavailable in this environment")
    assert paths.resolve_safe(str(link)) == real_dir.resolve(strict=False)
