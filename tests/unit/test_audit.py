"""Phase 7 audit tests: fixture-driven checks, read-only invariants, report,
tools and CLI.  All checks are exercised with recorded outputs (fakes) so the
suite runs identically on any OS (docs/05_BUILD_PLAN.md Phase 7)."""

from __future__ import annotations

import json
import re
import subprocess
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Self

import pytest

from jarvis.audit import checks
from jarvis.audit.checks import Finding
from jarvis.audit.report import render_markdown, write_report
from jarvis.config import PolicySettings, Settings
from jarvis.tools.audit import (
    _parse_defender_payload,
    make_audit_run_spec,
    make_defender_quick_scan_spec,
    make_defender_status_spec,
)
from jarvis.tools.base import ToolContext

# ── Fakes ────────────────────────────────────────────────────────────────


class _Conn:
    def __init__(self, ip: str, port: int, pid: int | None, status: str = "LISTEN") -> None:
        self.laddr = types.SimpleNamespace(ip=ip, port=port)
        self.pid = pid
        self.status = status


class FakePsutil:
    CONN_LISTEN = "LISTEN"

    def __init__(self, connections: list[_Conn], names: dict[int, str] | None = None) -> None:
        self._connections = list(connections)
        self._names = names or {}

    def net_connections(self, kind: str = "inet") -> list[_Conn]:
        _ = kind
        return self._connections

    def cpu_percent(self, interval: float = 0.1) -> float:
        return 12.5

    def virtual_memory(self) -> types.SimpleNamespace:
        return types.SimpleNamespace(percent=40.0)

    def disk_usage(self, path: str) -> types.SimpleNamespace:
        _ = path
        return types.SimpleNamespace(percent=55.0)

    def Process(self, pid: int | None) -> object:
        proc = object.__new__(type("Proc", (), {}))
        name = self._names.get(pid)
        if pid is None or name is None:
            raise OSError("gone")
        proc.name = lambda: name
        return proc


class _FakeKey:
    def __init__(self, values: list[tuple[str, str, int]]) -> None:
        self._values = values

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def EnumValue(self, index: int) -> tuple[str, str, int]:
        if index >= len(self._values):
            raise OSError
        return self._values[index]


class FakeWinreg:
    def __init__(self, registry: dict[tuple[int, str], list[tuple[str, str, int]]]) -> None:
        self._registry = registry
        self.HKEY_LOCAL_MACHINE = 1
        self.HKEY_CURRENT_USER = 2

    def OpenKey(self, hive: int, key: str) -> _FakeKey:
        values = self._registry.get((hive, key))
        if values is None:
            raise OSError
        return _FakeKey(values)

    def EnumValue(self, reg_key: _FakeKey, index: int) -> tuple[str, str, int]:
        return reg_key.EnumValue(index)


class FakeExecutor:
    """Replays recorded PowerShell outputs; records every command received."""

    def __init__(self, outputs: dict[str, tuple[str, str, int]]) -> None:
        self._outputs = outputs
        self.calls: list[tuple[str, float]] = []

    def __call__(self, command: str, timeout_s: float) -> tuple[str, str, int]:
        self.calls.append((command, timeout_s))
        return self._outputs.get(command, ("", "no fixture", 1))


_RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
_RUN_ONCE_KEY = r"Software\Microsoft\Windows\CurrentVersion\RunOnce"


def _registry_with(
    good: str = r"C:\Windows\System32\mstsc.exe",
    suspicious: str = r"C:\Users\bob\AppData\Roaming\nightly.exe",
) -> FakeWinreg:
    return FakeWinreg(
        {
            (1, _RUN_KEY): [("good", good, 1)],
            (2, _RUN_KEY): [("nightly", suspicious, 1)],
        }
    )


def _defender_json(*, healthy: bool) -> str:
    age = datetime.now(UTC) - timedelta(days=2)
    state = {
        "AntivirusEnabled": healthy,
        "RealTimeProtectionEnabled": healthy,
        "AntivirusSignatureVersion": "1.401.20.0",
        "AntivirusSignatureLastUpdated": age.isoformat(),
    }
    return json.dumps(state)


def _security_fixtures(
    *,
    healthy_defender: bool = True,
    firewall: str | None = None,
) -> dict[str, tuple[str, str, int]]:
    firewall = firewall or (
        '[{"Name":"Domain","Enabled":true},{"Name":"Private","Enabled":true},'
        '{"Name":"Public","Enabled":false}]'
    )
    return {
        checks._PS_DEFENDER_STATUS: (_defender_json(healthy=healthy_defender), "", 0),
        checks._PS_FIREWALL_PROFILES: (firewall, "", 0),
        checks._PS_BITLOCKER: ("", "Access is denied (run as administrator)", 1),
        checks._PS_SCHEDULED_TASKS: (
            json.dumps(
                [
                    {
                        "Author": "Microsoft Corporation",
                        "Task": r"\Microsoft\Windows\Test\job",
                        "Execute": r"C:\Windows\System32\wscript.exe",
                        "State": "Ready",
                    },
                    {
                        "Author": "CustomUser",
                        "Task": r"\MyNonsense",
                        "Execute": r"C:\Users\bob\evil.exe",
                        "State": "Ready",
                    },
                ]
            ),
            "",
            0,
        ),
    }


def _ctx() -> ToolContext:
    settings = Settings(policy=PolicySettings(allowed_roots=[]))
    return ToolContext(settings=settings)


# ── checks: fixture parsing, any OS ──────────────────────────────────────


def test_run_checks_security_fixtures_on_any_os(tmp_path: Path) -> None:
    executor = FakeExecutor(_security_fixtures(healthy_defender=False))
    psutil = FakePsutil(
        [
            _Conn("127.0.0.1", 9000, 1111),
            _Conn("192.168.1.5", 5555, None),
            _Conn("192.168.1.6", 8080, 2222),
        ],
        names={2222: "chrome.exe"},
    )
    findings = checks.run_checks(
        ["security"],
        executor=executor,
        os_name="Windows",
        psutil_module=psutil,
        winreg_module=_registry_with(),
        log_path=tmp_path / "jarvis.jsonl",
    )
    by_title = {f.title: f for f in findings}
    assert by_title["Real-time protection OFF"].severity == "critical"
    assert by_title["Defender antivirus disabled"].severity == "critical"
    assert by_title["Firewall profile(s) disabled"].severity == "warning"
    assert "admin" in by_title["BitLocker"].detail.lower()
    assert by_title["BitLocker"].severity == "unknown"
    assert by_title["Non-loopback listeners by unknown process"].severity == "warning"
    assert by_title["Startup entries from user-writable paths"].severity == "warning"
    assert by_title["Scheduled tasks from user-writable paths"].severity == "warning"
    assert by_title["Non-Microsoft scheduled tasks"].severity == "info"


def test_defender_healthy_yields_ok(tmp_path: Path) -> None:
    executor = FakeExecutor(_security_fixtures(healthy_defender=True))
    findings = checks.run_checks(
        ["security"],
        executor=executor,
        os_name="Windows",
        psutil_module=FakePsutil([]),
        winreg_module=_registry_with(),
        log_path=tmp_path / "jarvis.jsonl",
    )
    assert any(f.title == "Defender protected" and f.severity == "ok" for f in findings)


def test_firewall_disabled_profile_flags_warning(tmp_path: Path) -> None:
    fixtures = _security_fixtures()
    fixtures[checks._PS_FIREWALL_PROFILES] = (
        '[{"Name":"Public","Enabled":false}]',
        "",
        0,
    )
    findings = checks.run_checks(
        ["security"],
        executor=FakeExecutor(fixtures),
        os_name="Windows",
        psutil_module=FakePsutil([]),
        winreg_module=_registry_with(),
        log_path=tmp_path / "jarvis.jsonl",
    )
    flagged = [f for f in findings if f.severity == "warning" and "Firewall" in f.title]
    assert flagged and "Public" in flagged[0].detail


def test_timeout_becomes_unknown_not_exception() -> None:
    def timeout_executor(_command: str, _timeout: float) -> tuple[str, str, int]:
        raise subprocess.TimeoutExpired(cmd="powershell", timeout=_timeout)

    findings = checks.run_checks(
        ["updates"],
        executor=timeout_executor,
        os_name="Windows",
    )
    assert len(findings) == 1
    assert findings[0].severity == "unknown"
    assert "timed out" in findings[0].detail


def test_pending_updates_count() -> None:
    fixtures = {
        checks._PS_PENDING_UPDATES: ("pending=3", "", 0),
    }
    findings = checks.run_checks(["updates"], executor=FakeExecutor(fixtures), os_name="Windows")
    assert findings[0].severity == "warning"
    assert "3" in findings[0].detail

    fixtures[checks._PS_PENDING_UPDATES] = ("pending=0", "", 0)
    clean = checks.run_checks(["updates"], executor=FakeExecutor(fixtures), os_name="Windows")
    assert clean[0].severity == "ok"


def test_non_windows_host_reports_unknown_honestly(tmp_path: Path) -> None:
    findings = checks.run_checks(
        ["security"],
        executor=FakeExecutor({}),
        os_name="Linux",
        psutil_module=FakePsutil([]),
        winreg_module=None,
        log_path=tmp_path / "jarvis.jsonl",
    )
    windows_checks = [f for f in findings if f.section == "security"]
    assert any(
        f.severity == "unknown" and "not applicable on Linux" in f.detail for f in windows_checks
    )


def test_run_checks_rejects_unknown_section() -> None:
    with pytest.raises(ValueError):
        checks.run_checks(["crypto"])


def test_only_whitelisted_constant_commands_are_invoked() -> None:
    """Every command passed to the executor is one of the module constants."""
    executor = FakeExecutor(_security_fixtures())
    checks.run_checks(
        ["security", "updates"],
        executor=executor,
        os_name="Windows",
        psutil_module=FakePsutil([]),
        winreg_module=_registry_with(),
    )
    constants = set(checks._PS_TIMEOUTS)
    assert {cmd for cmd, _ in executor.calls} == constants
    for cmd, timeout_s in executor.calls:
        assert checks._PS_TIMEOUTS.get(cmd) == timeout_s


def test_run_checks_is_read_only(tmp_path: Path) -> None:
    """``run_checks`` never writes: an empty temp dir stays empty."""
    before = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    checks.run_checks(
        ["security", "performance", "updates", "self"],
        executor=FakeExecutor(_security_fixtures()),
        os_name="Windows",
        psutil_module=FakePsutil([]),
        winreg_module=_registry_with(),
        log_path=tmp_path / "jarvis.jsonl",
    )
    after = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    assert after == before


# ── checks: self-review ──────────────────────────────────────────────────


def _log_line(ts: str, level: str, message: str, logger: str = "jarvis.voice.loop") -> str:
    return json.dumps({"ts": ts, "level": level, "logger": logger, "message": message})


def test_self_review_lists_repeated_failures(tmp_path: Path) -> None:
    log = tmp_path / "jarvis.jsonl"
    log.write_text(
        "\n".join(
            [
                _log_line("2026-01-01T00:00:00Z", "INFO", "logging configured"),
                _log_line(
                    "2026-01-02T00:00:00Z",
                    "ERROR",
                    "voice interaction failed at stt; continuing to listen",
                ),
                _log_line(
                    "2026-01-02T00:05:00Z",
                    "ERROR",
                    "voice interaction failed at stt; continuing to listen",
                ),
                _log_line(
                    "2026-01-02T00:06:00Z",
                    "ERROR",
                    "voice interaction failed at stt; continuing to listen",
                ),
                _log_line(
                    "2026-01-03T00:00:00Z",
                    "WARNING",
                    "confirmation by voice refused: wake word not re-detected",
                ),
                _log_line(
                    "2026-01-04T00:00:00Z",
                    "WARNING",
                    "confirmation by voice refused: wake word not re-detected",
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    findings = checks.run_checks(["self"], log_path=log)
    repeats = [f for f in findings if f.severity == "warning"]
    assert len(repeats) == 2
    assert "3x" in repeats[0].detail and "stt" in repeats[0].detail
    assert "2x" in repeats[1].detail and "re-detected" in repeats[1].detail


def test_self_review_ok_with_no_repeats(tmp_path: Path) -> None:
    log = tmp_path / "jarvis.jsonl"
    log.write_text(
        _log_line("2026-01-01T00:00:00Z", "ERROR", "one-off failure", "jarvis.cli") + "\n",
        encoding="utf-8",
    )
    findings = checks.run_checks(["self"], log_path=log)
    assert findings[0].severity == "ok"


def test_self_review_unknown_without_log() -> None:
    findings = checks.run_checks(["self"], log_path=Path("C:/definitely/absent/log.jsonl"))
    assert findings[0].severity == "unknown"


# ── heuristics ───────────────────────────────────────────────────────────


def test_user_writable_path_heuristics() -> None:
    assert checks._is_user_writable_path(r"%TEMP%\x.exe")
    assert checks._is_user_writable_path(r"C:\Users\bob\evil.exe")
    assert checks._is_user_writable_path(r"C:\ProgramData\vendor\thing.exe")
    assert not checks._is_user_writable_path(r"C:\Windows\System32\mstsc.exe")
    assert not checks._is_user_writable_path("C:\\Windows\\System32\\")


# ── report ───────────────────────────────────────────────────────────────


def _sample_findings() -> list[Finding]:
    return [
        Finding("security", "critical", "Real-time protection OFF"),
        Finding("security", "ok", "Firewall profiles enabled", "Domain, Private"),
        Finding("updates", "warning", "Pending Windows updates", "3 update(s) pending"),
        Finding("self", "unknown", "JARVIS failure log", "No log file to review"),
    ]


def test_render_markdown_structure() -> None:
    md = render_markdown(_sample_findings(), generated_at=datetime(2026, 1, 1, tzinfo=UTC))
    assert "# JARVIS System Audit" in md
    assert "## Security" in md or "## security" in md
    assert "[critical] Real-time protection OFF" in md
    assert "Total findings: 4" in md
    assert "critical: 1" in md
    assert "nothing was modified" in md


def test_write_report_creates_timestamped_file(tmp_path: Path) -> None:
    path = write_report(_sample_findings(), tmp_path)
    assert path.is_file()
    assert re.match(r"audit-\d{8}-\d{6}\.md$", path.name)
    assert "Test" not in path.read_text(encoding="utf-8")


# ── tools ────────────────────────────────────────────────────────────────


def test_audit_run_spec_signature() -> None:
    spec = make_audit_run_spec()
    assert spec.name == "audit_run"
    assert spec.base_tier == 0
    assert spec.windows_only is False


def test_audit_run_dry_run_does_not_write(tmp_path: Path) -> None:
    spec = make_audit_run_spec()
    args = spec.args_model.model_validate({"sections": ["performance"]})
    ctx = ToolContext(settings=Settings(policy=PolicySettings(allowed_roots=[])), dry_run=True)
    result = spec.execute(args, ctx)
    assert result.ok
    assert "[dry-run]" in result.output
    assert "report_path" not in result.data


def test_audit_run_writes_report_and_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = make_audit_run_spec()
    args = spec.args_model.model_validate({"sections": ["performance"]})
    monkeypatch.setattr("jarvis.config.reports_dir", lambda: tmp_path)
    monkeypatch.setattr(
        "jarvis.tools.audit.audit_checks.run_checks",
        lambda _sections: [Finding("performance", "info", "CPU usage", "12.5%")],
    )
    result = spec.execute(args, _ctx())
    assert result.ok
    report = Path(result.data["report_path"])
    assert report.is_file()
    assert result.data["count"] == 1
    verified = spec.apply_verify(args, result, _ctx())
    assert verified.verified is True


def test_audit_run_rejects_bad_section() -> None:
    spec = make_audit_run_spec()
    with pytest.raises(ValueError):
        spec.args_model.model_validate({"sections": ["crypto"]})


def test_defender_status_parses_and_verifies() -> None:
    spec = make_defender_status_spec()
    assert spec.name == "defender_status"
    assert spec.base_tier == 0
    assert spec.windows_only is True
    ok, summary, data = _parse_defender_payload(_defender_json(healthy=True), "", 0)
    assert ok and data["real_time"] is True and "signature" in summary
    bad, bad_summary, _ = _parse_defender_payload("", "Access is denied", 1)
    assert not bad and "Could not read" in bad_summary


def test_defender_quick_scan_not_applicable_off_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("jarvis.tools.audit._is_windows", lambda: False)
    spec = make_defender_quick_scan_spec()
    assert spec.base_tier == 1
    args = spec.args_model.model_validate({})
    result = spec.execute(args, _ctx())
    assert result.ok
    assert "not applicable" in result.output


def test_tool_registry_includes_audit_tools() -> None:
    from jarvis.tools.registry import build_default_registry

    registry = build_default_registry()
    names = registry.names()
    for expected in ("audit_run", "defender_status", "defender_quick_scan"):
        assert expected in names
    # The catalogue the planner sees never exposes tiers.
    for entry in registry.catalogue_for_llm():
        if entry["name"] == "defender_quick_scan":
            assert "base_tier" not in entry


# ── real PowerShell smoke (Windows, dev PC only) ─────────────────────────


@pytest.mark.windows_only
def test_real_defender_status_smoke() -> None:
    out, err, rc = checks._run_powershell(checks._PS_DEFENDER_STATUS, 10.0)
    ok, summary, data = _parse_defender_payload(out, err, rc)
    assert ok or rc != 0  # an honest parse, or a reported failure - never a hang
    if ok:
        assert "Defender:" in summary
        assert data["real_time"] is not None


@pytest.mark.windows_only
@pytest.mark.slow
def test_real_quick_scan_flags_are_parseable() -> None:
    # Never start a scan in tests: the constant must only ever be passed to
    # subprocess with the fixed argument list (whitelist rule).
    assert checks._PS_QUICK_SCAN == "Start-MpScan -ScanType QuickScan"
    assert checks._PS_QUICK_SCAN.startswith("Start-MpScan")


def test_spawned_scans_are_kept_alive_list() -> None:
    from jarvis.tools import audit as audit_tools

    assert isinstance(audit_tools._SPAWNED_SCANS, list)


def test_audit_module_imports_on_any_os() -> None:
    import importlib

    for module in ("jarvis.audit.checks", "jarvis.audit.report", "jarvis.tools.audit"):
        importlib.import_module(module)
