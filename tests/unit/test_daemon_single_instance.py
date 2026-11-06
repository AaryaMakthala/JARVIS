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

from jarvis.daemon.server import (
    DaemonServer,
    _atomic_write_json,
    _parse_runtime_record,
    _StartupLock,
)


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


# ── Robust process-identity / stale-pid recovery ─────────────────────────
#
# These tests replace the real process table with a deterministic stand-in so
# the decision matrix is exercised without depending on a real PID (in
# particular, no test references PID 1216 or a live ``services.exe``).


class _MatrixProcess:
    """Minimal ``psutil.Process`` stand-in with per-field failure injection."""

    def __init__(
        self,
        *,
        name: str | None = None,
        exe: str | None = None,
        cmdline: list[str] | None = None,
        create_time: float | None = None,
        errors: dict[str, BaseException] | None = None,
    ) -> None:
        self._values: dict[str, object] = {
            "name": name,
            "exe": exe,
            "cmdline": cmdline,
            "create_time": create_time,
        }
        self._errors = errors or {}

    def _read(self, field: str) -> object:
        error = self._errors.get(field)
        if error is not None:
            raise error
        return self._values[field]

    def name(self) -> object:
        return self._read("name")

    def exe(self) -> object:
        return self._read("exe")

    def cmdline(self) -> object:
        return self._read("cmdline")

    def create_time(self) -> object:
        return self._read("create_time")


def _patch_process(
    monkeypatch: pytest.MonkeyPatch,
    process: _MatrixProcess | None = None,
    *,
    factory: object = None,
) -> None:
    monkeypatch.setattr(psutil, "pid_exists", lambda pid: True)
    if factory is None:
        assert process is not None
        factory = lambda pid: process
    monkeypatch.setattr(psutil, "Process", factory)


def _write_record(data: dict) -> Path:
    from jarvis.config import daemon_runtime_file

    path = daemon_runtime_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _read_record() -> dict:
    from jarvis.config import daemon_runtime_file

    return json.loads(daemon_runtime_file().read_text("utf-8"))


def test_legacy_foreign_process_with_denied_cmdline_is_stale(
    isolated_data_dir: Path, server: DaemonServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_process(
        monkeypatch,
        _MatrixProcess(
            name="services.exe",
            exe="C:\\Windows\\System32\\services.exe",
            errors={"cmdline": psutil.AccessDenied(1234)},
        ),
    )
    path = _write_record({"pid": 1234, "port": 1, "started_at": 1.0})
    assert server._check_single_instance() is True
    assert not path.is_file()


def test_legacy_jarvis_daemon_is_refused(
    isolated_data_dir: Path, server: DaemonServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_process(
        monkeypatch,
        _MatrixProcess(
            name="python.exe",
            cmdline=["python", "-m", "jarvis", "daemon", "--foreground"],
        ),
    )
    path = _write_record({"pid": 1234, "port": 1, "started_at": 1.0})
    assert server._check_single_instance() is False
    assert path.is_file()
    assert "another JARVIS daemon" in (server.startup_refusal_reason or "")


def test_legacy_unreadable_identity_fails_closed(
    isolated_data_dir: Path, server: DaemonServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_process(
        monkeypatch,
        _MatrixProcess(errors={"cmdline": psutil.AccessDenied(1234)}),
    )
    path = _write_record({"pid": 1234, "port": 1, "started_at": 1.0})
    assert server._check_single_instance() is False
    assert path.is_file()
    reason = server.startup_refusal_reason or ""
    assert "cannot be confirmed" in reason
    assert str(path) in reason


def test_creation_time_mismatch_is_stale_even_when_cmdline_denied(
    isolated_data_dir: Path, server: DaemonServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_process(
        monkeypatch,
        _MatrixProcess(
            name="python.exe",
            create_time=200.0,
            errors={"cmdline": psutil.AccessDenied(1234)},
        ),
    )
    path = _write_record({"pid": 1234, "port": 1, "started_at": 1.0, "process_create_time": 100.0})
    assert server._check_single_instance() is True
    assert not path.is_file()


def test_creation_time_match_jarvis_is_refused(
    isolated_data_dir: Path, server: DaemonServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_process(
        monkeypatch,
        _MatrixProcess(
            name="python.exe",
            create_time=100.0,
            cmdline=["python", "-m", "jarvis", "daemon"],
        ),
    )
    path = _write_record({"pid": 1234, "port": 1, "started_at": 1.0, "process_create_time": 100.0})
    assert server._check_single_instance() is False
    assert path.is_file()


def test_creation_time_match_foreign_is_stale(
    isolated_data_dir: Path, server: DaemonServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_process(monkeypatch, _MatrixProcess(name="notepad.exe", create_time=100.0))
    path = _write_record({"pid": 1234, "port": 1, "started_at": 1.0, "process_create_time": 100.0})
    assert server._check_single_instance() is True
    assert not path.is_file()


def test_create_time_denied_but_foreign_name_is_stale(
    isolated_data_dir: Path, server: DaemonServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_process(
        monkeypatch,
        _MatrixProcess(
            name="services.exe",
            errors={
                "create_time": psutil.AccessDenied(1234),
                "cmdline": psutil.AccessDenied(1234),
            },
        ),
    )
    path = _write_record({"pid": 1234, "port": 1, "started_at": 1.0, "process_create_time": 100.0})
    assert server._check_single_instance() is True
    assert not path.is_file()


def test_process_disappears_during_inspection_is_stale(
    isolated_data_dir: Path, server: DaemonServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_process(
        monkeypatch,
        _MatrixProcess(create_time=100.0, errors={"name": psutil.NoSuchProcess(1234)}),
    )
    path = _write_record({"pid": 1234, "port": 1, "started_at": 1.0})
    assert server._check_single_instance() is True
    assert not path.is_file()


def test_pid_reused_during_inspection_is_stale(
    isolated_data_dir: Path, server: DaemonServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _MatrixProcess(
        name="python.exe",
        create_time=100.0,
        cmdline=["python", "-m", "jarvis", "daemon"],
    )
    second = _MatrixProcess(name="notepad.exe", create_time=200.0)
    sequence = iter([first, second])
    _patch_process(monkeypatch, factory=lambda pid: next(sequence))
    path = _write_record({"pid": 1234, "port": 1, "started_at": 1.0, "process_create_time": 100.0})
    assert server._check_single_instance() is True
    assert not path.is_file()


def test_malformed_json_is_preserved_for_atomic_replacement(
    isolated_data_dir: Path, server: DaemonServer
) -> None:
    from jarvis.config import daemon_runtime_file

    path = daemon_runtime_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert server._check_single_instance() is True
    assert path.is_file()
    assert path.read_text("utf-8") == "{not json"


def test_invalid_pid_is_preserved(isolated_data_dir: Path, server: DaemonServer) -> None:
    path = _write_record({"pid": "not-a-number", "port": 1})
    assert server._check_single_instance() is True
    assert path.is_file()


def test_invalid_optional_fields_do_not_block_stale_cleanup(
    isolated_data_dir: Path, server: DaemonServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_process(monkeypatch, _MatrixProcess(name="notepad.exe"))
    path = _write_record(
        {
            "pid": 1234,
            "port": "bad",
            "started_at": "nope",
            "process_create_time": "bad",
        }
    )
    assert server._check_single_instance() is True
    assert not path.is_file()


def test_missing_optional_legacy_fields_backward_compatible(
    isolated_data_dir: Path, server: DaemonServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_process(monkeypatch, _MatrixProcess(name="notepad.exe"))
    path = _write_record({"pid": 1234})
    assert server._check_single_instance() is True
    assert not path.is_file()


def test_stale_cleanup_preserves_a_replaced_record(
    isolated_data_dir: Path, server: DaemonServer
) -> None:
    path = _write_record({"pid": 1234, "port": 1, "started_at": 1.0})
    original = _parse_runtime_record(json.loads(path.read_text("utf-8")))
    assert original is not None
    _write_record({"pid": 5678, "port": 2, "started_at": 2.0})
    assert server._take_stale_pid_file(path, original) is False
    assert _read_record()["pid"] == 5678


def test_cleanup_does_not_remove_a_newer_daemons_record(
    isolated_data_dir: Path, server: DaemonServer
) -> None:
    path = _write_record({"pid": os.getpid() + 1, "port": 1, "started_at": 1.0})
    server._pid_file = path
    server._process_create_time = 100.0
    server._cleanup_pid_file()
    assert path.is_file()


def test_cleanup_rejects_same_pid_different_creation_time(
    isolated_data_dir: Path, server: DaemonServer
) -> None:
    path = _write_record(
        {
            "pid": os.getpid(),
            "port": 1,
            "started_at": 1.0,
            "process_create_time": 100.0,
        }
    )
    server._pid_file = path
    server._process_create_time = 200.0
    server._cleanup_pid_file()
    assert path.is_file()


def test_cleanup_removes_own_record(isolated_data_dir: Path, server: DaemonServer) -> None:
    path = _write_record(
        {
            "pid": os.getpid(),
            "port": 1,
            "started_at": 1.0,
            "process_create_time": 100.0,
        }
    )
    server._pid_file = path
    server._process_create_time = 100.0
    server._cleanup_pid_file()
    assert not path.is_file()


def test_write_pid_file_records_process_creation_time(
    isolated_data_dir: Path, server: DaemonServer
) -> None:
    from jarvis.config import daemon_runtime_file

    server._pid_file = daemon_runtime_file()
    server._write_pid_file(43210)
    data = json.loads(daemon_runtime_file().read_text("utf-8"))
    assert data["pid"] == os.getpid()
    assert data["port"] == 43210
    assert data["started_at"] > 0
    assert data["process_create_time"] > 0


def test_atomic_write_failure_leaves_original_record(
    isolated_data_dir: Path, server: DaemonServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jarvis.config import daemon_runtime_file

    path = daemon_runtime_file()
    _atomic_write_json(path, {"pid": 111, "port": 1, "started_at": 1.0})
    original = path.read_text("utf-8")
    server._pid_file = path

    def _boom(src: object, dst: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr("jarvis.daemon.server.os.replace", _boom)
    with pytest.raises(OSError):
        server._write_pid_file(2)
    assert path.read_text("utf-8") == original
    assert list(path.parent.glob("*.tmp")) == []


def test_startup_lock_allows_only_one_owner(isolated_data_dir: Path) -> None:
    path = isolated_data_dir / "daemon.lock"
    first = _StartupLock(path)
    second = _StartupLock(path)
    assert first.acquire() is True
    try:
        assert second.acquire() is False
    finally:
        first.release()
    assert second.acquire() is True
    second.release()


def test_cli_reports_refused_startup_with_nonzero_exit(
    isolated_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    import jarvis.daemon.server as server_module
    from jarvis import cli

    class _RefusingServer:
        def __init__(self, **kwargs: object) -> None:
            self.startup_refusal_reason = "another JARVIS daemon is running (pid 1234)"

        def run(self) -> bool:
            return False

    monkeypatch.setattr(server_module, "DaemonServer", _RefusingServer)
    monkeypatch.setattr(cli.config, "load_settings", lambda: cli.config.Settings())
    monkeypatch.setattr(cli.secret_module, "SecretStore", lambda: _FakeStore())

    result = CliRunner().invoke(cli.app, ["daemon", "--foreground"])
    assert result.exit_code == 1
    assert "did not start" in result.output.lower()
    assert "another JARVIS daemon" in result.output
