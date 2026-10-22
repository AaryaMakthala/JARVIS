"""The read-only ``system_info`` tool (Stage 9, slice 9.1).

The interesting case is the disk line: the tool used to measure
``psutil.disk_usage("/")`` (the *current* drive on Windows) but always called
it "Disk C:", which is wrong on a machine whose system drive is not C:.  The
label must follow the drive that was actually measured.
"""

from __future__ import annotations

from typing import Any

import pytest

from jarvis.config import Settings
from jarvis.tools.base import ToolContext
from jarvis.tools.system import SystemInfoArgs, _disk_target, make_system_info_spec


class _Usage:
    """A stand-in for ``psutil._common.sdiskusage``."""

    def __init__(self) -> None:
        self.used = 12_000_000_000
        self.total = 100_000_000_000
        self.percent = 12.0


@pytest.fixture()
def measured(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Capture the path the tool measures, without measuring anything."""
    import psutil

    paths: list[str] = []

    def fake_disk_usage(path: str) -> _Usage:
        paths.append(path)
        return _Usage()

    monkeypatch.setattr(psutil, "disk_usage", fake_disk_usage)
    return paths


def _run(what: str, ctx: ToolContext | None = None) -> Any:
    spec = make_system_info_spec()
    return spec.run(SystemInfoArgs(what=what), ctx or ToolContext(settings=Settings()))


# ── which disk is measured ────────────────────────────────────────────────


def test_the_windows_label_is_the_real_system_drive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("jarvis.tools.system.platform.system", lambda: "Windows")
    monkeypatch.setenv("SystemDrive", "D:")
    assert _disk_target() == ("D:", "D:\\")


def test_a_drive_without_a_colon_still_gets_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("jarvis.tools.system.platform.system", lambda: "Windows")
    monkeypatch.setenv("SystemDrive", "E")
    assert _disk_target() == ("E:", "E:\\")


def test_a_missing_system_drive_falls_back_to_c(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("jarvis.tools.system.platform.system", lambda: "Windows")
    monkeypatch.delenv("SystemDrive", raising=False)
    assert _disk_target() == ("C:", "C:\\")


def test_posix_reports_the_root_filesystem(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("jarvis.tools.system.platform.system", lambda: "Linux")
    assert _disk_target() == ("root", "/")


# ── what the tool reports ──────────────────────────────────────────────────


def test_the_disk_line_names_the_drive_it_measured(
    monkeypatch: pytest.MonkeyPatch, measured: list[str]
) -> None:
    monkeypatch.setattr("jarvis.tools.system.platform.system", lambda: "Windows")
    monkeypatch.setenv("SystemDrive", "D:")

    result = _run("disk")

    assert result.ok is True
    assert result.output.startswith("Disk D:")
    assert measured == ["D:\\"]  # the drive that was measured is the one named
    assert result.data["label"] == "D:"
    assert result.data["path"] == "D:\\"


def test_the_disk_numbers_are_reported(
    monkeypatch: pytest.MonkeyPatch, measured: list[str]
) -> None:
    monkeypatch.setattr("jarvis.tools.system.platform.system", lambda: "Linux")
    result = _run("disk")

    assert result.output == "Disk root: 12.0 / 100.0 GB (12.0%)"
    assert result.data["used_gb"] == 12.0
    assert result.data["total_gb"] == 100.0
    assert result.data["percent"] == 12.0


def test_a_dry_run_reads_nothing() -> None:
    ctx = ToolContext(settings=Settings(), dry_run=True)
    result = _run("disk", ctx)

    assert result.ok is True
    assert result.output.startswith("[dry-run]")
    assert not result.data


@pytest.mark.parametrize("what", ["ram", "cpu", "battery", "uptime", "top_processes"])
def test_the_other_readings_work(what: str) -> None:
    result = _run(what)

    assert result.ok is True
    assert result.output
