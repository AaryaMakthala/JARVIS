"""Unit tests for :func:`DaemonServer._check_single_instance`.

Regression coverage for the Windows stale-pid crash: ``os.kill(old_pid, 0)``
on a dead pid raises ``SystemError`` (not plain ``OSError``), which escaped the
``except OSError`` guard and crashed the daemon at startup. The liveness probe
must use ``psutil.pid_exists`` instead, must not kill the probed process, and
must auto-remove a stale ``daemon.json``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import psutil
import pytest

from jarvis.daemon.server import DaemonServer


class _FakeStore:
    def get(self, name: str) -> str:
        return "t"


class _FakeProcess:
    def __init__(self, name: str, cmdline: list[str]) -> None:
        self._name = name
        self._cmdline = cmdline

    def name(self) -> str:
        return self._name

    def cmdline(self) -> list[str]:
        return self._cmdline


def _dead_pid() -> int:
    """Return a pid that does not currently exist on this OS."""
    for _ in range(50):
        proc = subprocess.Popen([sys.executable, "-c", "import os; os._exit(0)"])
        proc.wait(timeout=10)
        if not psutil.pid_exists(proc.pid):
            return proc.pid
    raise AssertionError("could not obtain a dead pid")


def _write_pid_file(pid: int) -> None:
    from jarvis.config import daemon_runtime_file

    pid_file = daemon_runtime_file()
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(json.dumps({"pid": pid, "port": 1}), encoding="utf-8")


@pytest.fixture
def isolated_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("JARVIS_DATA_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def server() -> DaemonServer:
    return DaemonServer(settings=None, store=_FakeStore())


def test_no_pid_file_returns_true(isolated_data_dir: Path, server: DaemonServer) -> None:
    from jarvis.config import daemon_runtime_file

    daemon_runtime_file().unlink(missing_ok=True)
    assert server._check_single_instance() is True


def test_dead_pid_allows_start(isolated_data_dir: Path, server: DaemonServer) -> None:
    _write_pid_file(_dead_pid())
    assert server._check_single_instance() is True


def test_dead_pid_file_auto_removed(isolated_data_dir: Path, server: DaemonServer) -> None:
    from jarvis.config import daemon_runtime_file

    _write_pid_file(_dead_pid())
    server._check_single_instance()
    assert not daemon_runtime_file().is_file()


def test_live_non_jarvis_pid_treated_as_stale(
    isolated_data_dir: Path, server: DaemonServer
) -> None:
    creation_kwargs = {}
    if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        creation_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    victim = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **creation_kwargs,
    )
    try:
        from jarvis.config import daemon_runtime_file

        _write_pid_file(victim.pid)
        assert server._check_single_instance() is True
        assert victim.poll() is None, "liveness probe must not kill the probed process"
        assert not daemon_runtime_file().is_file(), "reused pid file must be removed"
    finally:
        victim.terminate()
        victim.wait(timeout=10)


def test_live_jarvis_daemon_pid_refused(
    isolated_data_dir: Path, server: DaemonServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jarvis.config import daemon_runtime_file

    monkeypatch.setattr(psutil, "pid_exists", lambda pid: True)
    monkeypatch.setattr(
        psutil,
        "Process",
        lambda pid: _FakeProcess(
            "python.exe", ["python", "-m", "jarvis", "daemon", "--foreground"]
        ),
    )
    _write_pid_file(1234)
    assert server._check_single_instance() is False
    assert daemon_runtime_file().is_file()


def test_alive_pid_dies_between_probe_and_process(
    isolated_data_dir: Path, server: DaemonServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jarvis.config import daemon_runtime_file

    monkeypatch.setattr(psutil, "pid_exists", lambda pid: True)

    def _raising_process(pid: int) -> _FakeProcess:
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(psutil, "Process", _raising_process)
    _write_pid_file(1234)
    assert server._check_single_instance() is True
    assert not daemon_runtime_file().is_file()


def test_alive_pid_inaccessible_refused(
    isolated_data_dir: Path, server: DaemonServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jarvis.config import daemon_runtime_file

    monkeypatch.setattr(psutil, "pid_exists", lambda pid: True)

    def _raising_process(pid: int) -> _FakeProcess:
        raise psutil.AccessDenied(pid)

    monkeypatch.setattr(psutil, "Process", _raising_process)
    _write_pid_file(1234)
    assert server._check_single_instance() is False
    assert daemon_runtime_file().is_file()


def test_live_non_python_pid_treated_as_stale(
    isolated_data_dir: Path, server: DaemonServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jarvis.config import daemon_runtime_file

    monkeypatch.setattr(psutil, "pid_exists", lambda pid: True)
    monkeypatch.setattr(
        psutil,
        "Process",
        lambda pid: _FakeProcess("notepad.exe", ["notepad.exe", "C:\\notes.txt"]),
    )
    _write_pid_file(1234)
    assert server._check_single_instance() is True
    assert not daemon_runtime_file().is_file()


def test_own_pid_returns_true(isolated_data_dir: Path, server: DaemonServer) -> None:
    _write_pid_file(os.getpid())
    assert server._check_single_instance() is True


def test_corrupt_pid_file_returns_true(isolated_data_dir: Path, server: DaemonServer) -> None:
    from jarvis.config import daemon_runtime_file

    pid_file = daemon_runtime_file()
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text("not json", encoding="utf-8")
    assert server._check_single_instance() is True
