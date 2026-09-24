"""Read-only system audit checks (Phase 7).

Every check is strictly read-only and runs a **fixed, hard-coded** PowerShell
command (module-level constant) or a read-only library call (psutil/winreg).
No command string ever originates from the LLM or the user:
docs/04_TOOLS_SPEC.md §2.4 daemon rule.  PowerShell commands are executed with
``subprocess.run([...])`` - never ``shell=True``, never an execution-policy
change.

Checks return :class:`Finding` objects.  A check that needs admin rights but
cannot prove a state reports ``severity="unknown"`` (e.g. "needs admin")
rather than a false "ok" or "warning" - fail closed.  ``run_checks`` accepts an
injectable ``executor`` and ``os_name`` so unit tests can replay recorded
outputs on any host OS.
"""

from __future__ import annotations

import importlib
import json
import logging
import platform
import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TypeAlias

logger = logging.getLogger(__name__)

Severity: TypeAlias = Literal["ok", "info", "warning", "critical", "unknown"]
CheckExecutor: TypeAlias = Callable[[str, float], tuple[str, str, int]]

#: Sections exposed by :func:`run_checks` and the ``jarvis audit`` CLI.
SECTION_CHECKS: dict[str, tuple[str, ...]] = {
    "security": (
        "defender",
        "firewall",
        "bitlocker",
        "listening_ports",
        "startup_entries",
        "scheduled_tasks",
    ),
    "performance": ("performance",),
    "updates": ("pending_updates",),
    "self": ("self_review",),
}
ALLOWED_SECTIONS: tuple[str, ...] = tuple(SECTION_CHECKS)

# ── Fixed, hard-coded PowerShell command constants (never interpolated) ──
_PS_DEFENDER_STATUS = "Get-MpComputerStatus | ConvertTo-Json -Compress"
_PS_FIREWALL_PROFILES = "Get-NetFirewallProfile | ConvertTo-Json -Compress"
_PS_BITLOCKER = "Get-BitLockerVolume | ConvertTo-Json -Compress"
_PS_PENDING_UPDATES = (
    "$s = New-Object -ComObject Microsoft.Update.Session; "
    "$r = $s.CreateUpdateSearcher(); "
    "$u = $r.Search('IsInstalled=0'); "
    "Write-Output ('pending=' + $u.Updates.Count)"
)
_PS_SCHEDULED_TASKS = (
    "Get-ScheduledTask | Where-Object { $_.State -ne 'Disabled' } | "
    "ForEach-Object { "
    "$action = $_.Actions | Where-Object { $_.Execute } | Select-Object -First 1; "
    "[PSCustomObject]@{ "
    "Author = $_.Author; Task = ($_.TaskPath + $_.TaskName); "
    "Execute = $(if ($action) { $action.Execute } else { '' }); "
    "State = $_.State.ToString() } } | ConvertTo-Json -Compress"
)
_PS_QUICK_SCAN = "Start-MpScan -ScanType QuickScan"

#: Per-command timeouts (seconds).  The COM update search gets the spec's 60 s.
_PS_TIMEOUT_S = 20.0
_UPDATES_TIMEOUT_S = 60.0
_PS_TIMEOUTS: dict[str, float] = {
    _PS_DEFENDER_STATUS: _PS_TIMEOUT_S,
    _PS_FIREWALL_PROFILES: _PS_TIMEOUT_S,
    _PS_BITLOCKER: _PS_TIMEOUT_S,
    _PS_SCHEDULED_TASKS: _PS_TIMEOUT_S,
    _PS_PENDING_UPDATES: _UPDATES_TIMEOUT_S,
}

#: Everything below this resource percentage counts as normal.
_HEALTHY_LIMIT_PERCENT: float = 90.0

#: Reports / evidence are capped so a machine with many listeners or tasks
#: cannot produce an unbounded report.
_LIST_CAP = 5


@dataclass(frozen=True)
class Finding:
    """One audit observation with a severity, title, detail and evidence."""

    section: str
    severity: Severity
    title: str
    detail: str = ""
    evidence: str = ""


def _run_powershell(command: str, timeout_s: float) -> tuple[str, str, int]:
    """Run a FIXED PowerShell command; returns (stdout, stderr, returncode).

    Callers MUST pass a module-level constant (whitelist rule).  ``timeout``
    prevents a hung COM/WMI call from wedging an audit.  Raises
    :class:`subprocess.TimeoutExpired` / :class:`OSError` on failure - callers
    turn those into an honest "unknown" finding.
    """
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,  # rc is inspected below; a non-zero exit is a finding, not an error
    )
    return proc.stdout.strip(), proc.stderr.strip(), proc.returncode


def _exec_safe(executor: CheckExecutor, command: str, timeout_s: float) -> tuple[str, str, int]:
    """Run ``executor`` but never let a timeout/OS error escape."""
    try:
        return executor(command, timeout_s)
    except subprocess.TimeoutExpired:
        return "", f"timed out after {timeout_s:g}s", -2
    except OSError as exc:  # e.g. PowerShell not present on non-Windows hosts
        return "", str(exc), -1


def _parse_json(value: str) -> Any:
    """Best-effort JSON parse of PowerShell output (None on failure/empty)."""
    value = value.strip()
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _days_since(iso: str) -> int | None:
    """Whole days between an ISO timestamp and now, or None if unparsable."""
    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return (_now_utc() - dt).days
    except ValueError:
        return None


def _is_user_writable_path(path: str) -> bool:
    """Heuristic: does ``path`` live somewhere an unprivileged user can edit?"""
    upper = path.upper().strip()
    if not upper:
        return False
    if upper.startswith("%"):  # %TEMP%, %APPDATA% ...
        return True
    if "\\USERS\\" in f"{upper}\\" and not upper.startswith(r"C:\USERS\PUBLIC"):
        return True
    if "\\APPDATA\\" in upper or "\\PROGRAMDATA\\" in upper:
        return True
    return not re.match(r"^[A-Z]:\\", upper)


# ── Security checks ──────────────────────────────────────────────────────


def _unknown(section: str, title: str, why: str) -> Finding:
    return Finding(section, "unknown", title, why)


def _check_defender(
    executor: CheckExecutor, timeout_s: float, os_name: str, section: str
) -> list[Finding]:
    if os_name != "Windows":
        return [_unknown(section, "Defender status", f"Windows only; not applicable on {os_name}")]
    out, err, rc = _exec_safe(executor, _PS_DEFENDER_STATUS, timeout_s)
    if rc != 0 or not out:
        reason = err or "no output"
        return [_unknown(section, "Defender status", f"Could not read Defender status ({reason})")]
    data = _parse_json(out)
    if isinstance(data, list):
        data = data[0] if data else None
    if not isinstance(data, dict):
        return [_unknown(section, "Defender status", "Unexpected Defender output")]
    findings: list[Finding] = []
    if not data.get("AntivirusEnabled", True):
        findings.append(Finding(section, "critical", "Defender antivirus disabled"))
    if not data.get("RealTimeProtectionEnabled", True):
        findings.append(Finding(section, "critical", "Real-time protection OFF"))
    signature = data.get("AntivirusSignatureVersion") or data.get("EngineVersion") or "?"
    age = _days_since(str(data.get("AntivirusSignatureLastUpdated", "")))
    if age is not None and age > 7:
        findings.append(
            Finding(section, "warning", "Defender signatures stale", f"{age} days old (> 7)")
        )
    summary = f"Antivirus signature {signature}"
    if age is not None:
        summary += f", updated {age} day(s) ago"
    if not any(f.severity in ("critical", "warning") for f in findings):
        findings.append(Finding(section, "ok", "Defender protected", summary))
    else:
        findings.append(Finding(section, "info", "Defender signatures", summary))
    return findings


def _check_firewall(
    executor: CheckExecutor, timeout_s: float, os_name: str, section: str
) -> list[Finding]:
    if os_name != "Windows":
        return [
            _unknown(section, "Firewall profiles", f"Windows only; not applicable on {os_name}")
        ]
    out, err, rc = _exec_safe(executor, _PS_FIREWALL_PROFILES, timeout_s)
    if rc != 0 or not out:
        reason = err or "no output"
        return [
            _unknown(section, "Firewall profiles", f"Could not read firewall profiles ({reason})")
        ]
    data = _parse_json(out)
    profiles = data if isinstance(data, list) else ([data] if isinstance(data, dict) else [])
    if not profiles:
        return [_unknown(section, "Firewall profiles", "No firewall profile data returned")]
    disabled = [
        f"{p.get('Name', '?')} ({p.get('Profile', '?')})"
        for p in profiles
        if p.get("Enabled") is False
    ]
    if disabled:
        return [Finding(section, "warning", "Firewall profile(s) disabled", ", ".join(disabled))]
    names = [str(p.get("Name", "?")) for p in profiles]
    return [Finding(section, "ok", "Firewall profiles enabled", ", ".join(names))]


def _check_bitlocker(
    executor: CheckExecutor, timeout_s: float, os_name: str, section: str
) -> list[Finding]:
    if os_name != "Windows":
        return [_unknown(section, "BitLocker", f"Windows only; not applicable on {os_name}")]
    out, err, rc = _exec_safe(executor, _PS_BITLOCKER, timeout_s)
    if rc != 0 or "denied" in err.lower():
        note = (
            "needs admin"
            if ("denied" in err.lower() or "deny" in err.lower())
            else (err or "no output")
        )
        return [_unknown(section, "BitLocker", f"Not readable without admin ({note})")]
    if not out:
        return [_unknown(section, "BitLocker", "No BitLocker volume data (report unknown)")]
    data = _parse_json(out)
    volumes = data if isinstance(data, list) else ([data] if isinstance(data, dict) else [])
    if not volumes:
        return [_unknown(section, "BitLocker", "No BitLocker volume data (report unknown)")]
    unencrypted = [
        f"{v.get('VolumeStatus', '?')} on {v.get('DriveLetter', '?')}"
        for v in volumes
        if str(v.get("VolumeStatus", "")).lower() != "fullyencrypted"
    ]
    if unencrypted:
        return [
            Finding(section, "warning", "Volume(s) not fully encrypted", "; ".join(unencrypted))
        ]
    return [
        Finding(section, "ok", "Volumes encrypted", f"{len(volumes)} volume(s) fully encrypted")
    ]


def _check_listening_ports(psutil_module: Any, os_name: str, section: str) -> list[Finding]:
    try:
        connections = psutil_module.net_connections(kind="inet")
    except Exception as exc:  # noqa: BLE001 - a permission/timeout must not kill the audit
        return [_unknown(section, "Listening ports", f"Could not enumerate connections ({exc})")]
    listen_status = getattr(psutil_module, "CONN_LISTEN", "LISTEN")
    listeners = [c for c in connections if c.status == listen_status]

    def owner_name(pid: int | None) -> str | None:
        if pid is None:
            return None
        try:
            return psutil_module.Process(pid).name()
        except Exception:  # noqa: BLE001 - process may have exited meanwhile
            return None

    loopback = ("127.0.0.1", "::1", "0.0.0.0", "::")
    exposed = [
        (c, owner_name(c.pid))
        for c in listeners
        if c.laddr.ip not in loopback and not c.laddr.ip.startswith("127.")
    ]
    unknown_owned = [(c, name) for c, name in exposed if name is None]
    findings: list[Finding] = []
    findings.append(Finding(section, "info", "Listening ports", f"{len(listeners)} port(s) open"))
    if unknown_owned:
        lines = [f"{c.laddr.ip}:{c.laddr.port} (pid {c.pid})" for c, _ in unknown_owned[:_LIST_CAP]]
        more = (
            f" and {len(unknown_owned) - _LIST_CAP} more" if len(unknown_owned) > _LIST_CAP else ""
        )
        findings.append(
            Finding(
                section,
                "warning",
                "Non-loopback listeners by unknown process",
                "; ".join(lines) + more,
            )
        )
    else:
        findings.append(
            Finding(
                section,
                "ok",
                "No non-loopback listener by unknown process",
                f"{len(exposed)} non-loopback listener(s) with a known owner",
            )
        )
    return findings


def _check_startup_entries(winreg_module: Any, os_name: str, section: str) -> list[Finding]:
    if os_name != "Windows":
        return [_unknown(section, "Startup entries", f"Windows only; not applicable on {os_name}")]
    roots = (
        (winreg_module.HKEY_LOCAL_MACHINE, r"Software\Microsoft\Windows\CurrentVersion\Run"),
        (winreg_module.HKEY_LOCAL_MACHINE, r"Software\Microsoft\Windows\CurrentVersion\RunOnce"),
        (winreg_module.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run"),
        (winreg_module.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\RunOnce"),
    )
    entries: list[str] = []
    suspicious: list[str] = []
    for hive, key in roots:
        try:
            with winreg_module.OpenKey(hive, key) as reg_key:
                index = 0
                while True:
                    try:
                        name, value, _kind = winreg_module.EnumValue(reg_key, index)
                    except OSError:
                        break
                    index += 1
                    entries.append(value)
                    if _is_user_writable_path(value):
                        suspicious.append(f"{name} -> {value}")
        except OSError:
            continue  # key absent on this machine: fine
    if not entries and not suspicious:
        return [_unknown(section, "Startup entries", "No startup values readable (report unknown)")]
    findings: list[Finding] = [
        Finding(section, "info", "Startup entries", f"{len(entries)} value(s) in Run/RunOnce")
    ]
    if suspicious:
        lines = ["; ".join(suspicious[:_LIST_CAP])]
        more = f"{len(suspicious) - _LIST_CAP} more" if len(suspicious) > _LIST_CAP else ""
        findings.append(
            Finding(
                section,
                "warning",
                "Startup entries from user-writable paths",
                "; ".join(lines) + more,
            )
        )
    return findings


def _check_scheduled_tasks(
    executor: CheckExecutor, timeout_s: float, os_name: str, section: str
) -> list[Finding]:
    if os_name != "Windows":
        return [_unknown(section, "Scheduled tasks", f"Windows only; not applicable on {os_name}")]
    out, err, rc = _exec_safe(executor, _PS_SCHEDULED_TASKS, timeout_s)
    if rc != 0 or not out:
        reason = err or "no output"
        return [_unknown(section, "Scheduled tasks", f"Could not read scheduled tasks ({reason})")]
    data = _parse_json(out)
    tasks = data if isinstance(data, list) else ([data] if isinstance(data, dict) else [])
    if not tasks:
        return [Finding(section, "ok", "Scheduled tasks", "No non-disabled scheduled tasks found")]
    non_ms = [t for t in tasks if not str(t.get("Author", "")).lower().startswith("microsoft")]
    user_writable = [t for t in tasks if _is_user_writable_path(str(t.get("Execute", "")))]
    findings: list[Finding] = [
        Finding(section, "info", "Scheduled tasks", f"{len(tasks)} task(s) active")
    ]
    if non_ms:
        lines = ["; ".join(str(t.get("Task", "")) for t in non_ms[:_LIST_CAP])]
        more = f"{len(non_ms) - _LIST_CAP} more" if len(non_ms) > _LIST_CAP else ""
        findings.append(
            Finding(section, "info", "Non-Microsoft scheduled tasks", "; ".join(lines) + more)
        )
    if user_writable:
        lines = ["; ".join(str(t.get("Task", "")) for t in user_writable[:_LIST_CAP])]
        more = f"{len(user_writable) - _LIST_CAP} more" if len(user_writable) > _LIST_CAP else ""
        findings.append(
            Finding(
                section,
                "warning",
                "Scheduled tasks from user-writable paths",
                "; ".join(lines) + more,
            )
        )
    return findings


# ── Performance / updates / self ─────────────────────────────────────────


def _check_performance(psutil_module: Any, section: str) -> list[Finding]:
    findings: list[Finding] = []
    try:
        cpu = psutil_module.cpu_percent(interval=0.1)
        mem = psutil_module.virtual_memory().percent
        disk = psutil_module.disk_usage("/").percent
    except Exception as exc:  # noqa: BLE001 - psutil must never kill an audit
        return [_unknown(section, "System resources", f"Could not read resource usage ({exc})")]
    findings.append(Finding(section, "info", "CPU usage", f"{cpu:.1f}%"))
    findings.append(Finding(section, "info", "Memory usage", f"{mem:.1f}%"))
    findings.append(Finding(section, "info", "Disk usage", f"{disk:.1f}%"))
    if max(cpu, mem, disk) > _HEALTHY_LIMIT_PERCENT:
        high = [
            f"{name}:{value:.0f}%"
            for name, value in (("cpu", cpu), ("mem", mem), ("disk", disk))
            if value > _HEALTHY_LIMIT_PERCENT
        ]
        findings.append(Finding(section, "warning", "Resource usage high", ", ".join(high)))
    return findings


def _check_pending_updates(
    executor: CheckExecutor, timeout_s: float, os_name: str, section: str
) -> list[Finding]:
    if os_name != "Windows":
        return [
            _unknown(
                section, "Pending Windows updates", f"Windows only; not applicable on {os_name}"
            )
        ]
    out, err, rc = _exec_safe(executor, _PS_PENDING_UPDATES, timeout_s)
    if rc != 0 or "pending=" not in out:
        reason = err or out or "no output"
        return [
            _unknown(
                section, "Pending Windows updates", f"Could not query Windows Update ({reason})"
            )
        ]
    match = re.search(r"pending=(\d+)", out)
    if match is None:
        return [_unknown(section, "Pending Windows updates", "Unexpected Windows Update output")]
    pending = int(match.group(1))
    if pending == 0:
        return [
            Finding(
                section, "ok", "Pending Windows updates", "No pending updates (last full check)"
            )
        ]
    return [
        Finding(
            section,
            "warning",
            "Pending Windows updates",
            f"{pending} update(s) pending installation",
        )
    ]


def _check_self_review(log_path: Path | None, section: str) -> list[Finding]:
    if log_path is None or not log_path.exists():
        return [_unknown(section, "JARVIS failure log", "No jarvis.jsonl log to review")]
    counts: dict[str, int] = {}
    latest: dict[str, str] = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("level") not in ("ERROR", "WARNING"):
            continue
        message = record.get("message")
        if not message:
            continue
        key = f"{record.get('logger', '')} :: {message}"
        counts[key] = counts.get(key, 0) + 1
        latest[key] = str(record.get("ts", ""))
    repeats = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    repeats = [kv for kv in repeats if kv[1] >= 2][:_LIST_CAP]
    if not repeats:
        return [Finding(section, "ok", "JARVIS failure log", "No repeated ERROR/WARNING lines")]
    return [
        Finding(
            section,
            "warning",
            "Repeated failures in JARVIS log",
            f"{count}x: {key}",
            latest.get(key, ""),
        )
        for key, count in repeats
    ]


# ── Public entry points ──────────────────────────────────────────────────


def run_checks(
    sections: Sequence[str],
    *,
    executor: CheckExecutor | None = None,
    log_path: Path | None = None,
    os_name: str | None = None,
    psutil_module: Any | None = None,
    winreg_module: Any | None = None,
) -> list[Finding]:
    """Run the requested read-only audit sections and return all findings.

    ``sections`` entries must be in :data:`ALLOWED_SECTIONS` else
    :class:`ValueError` is raised.  ``executor`` (PowerShell), ``log_path``
    (self-review source), ``os_name`` and the psutil/winreg modules are
    injectable so unit tests run recorded fixtures on any OS; defaults are the
    live system.  This function never writes to disk.
    """
    real_executor = executor if executor is not None else _run_powershell
    current_os = os_name if os_name is not None else platform.system()
    psutil = psutil_module
    winreg = winreg_module
    if psutil is None:
        psutil = importlib.import_module("psutil")
    if winreg is None and current_os == "Windows":
        winreg = importlib.import_module("winreg")

    findings: list[Finding] = []
    for section in sections:
        if section not in ALLOWED_SECTIONS:
            raise ValueError(
                f"unknown audit section {section!r}; allowed: {', '.join(ALLOWED_SECTIONS)}"
            )
        for check in SECTION_CHECKS[section]:
            if check == "defender":
                findings.extend(
                    _check_defender(
                        real_executor, _PS_TIMEOUTS[_PS_DEFENDER_STATUS], current_os, section
                    )
                )
            elif check == "firewall":
                findings.extend(
                    _check_firewall(
                        real_executor, _PS_TIMEOUTS[_PS_FIREWALL_PROFILES], current_os, section
                    )
                )
            elif check == "bitlocker":
                findings.extend(
                    _check_bitlocker(
                        real_executor, _PS_TIMEOUTS[_PS_BITLOCKER], current_os, section
                    )
                )
            elif check == "listening_ports":
                findings.extend(_check_listening_ports(psutil, current_os, section))
            elif check == "startup_entries":
                findings.extend(_check_startup_entries(winreg, current_os, section))
            elif check == "scheduled_tasks":
                findings.extend(
                    _check_scheduled_tasks(
                        real_executor, _PS_TIMEOUTS[_PS_SCHEDULED_TASKS], current_os, section
                    )
                )
            elif check == "performance":
                findings.extend(_check_performance(psutil, section))
            elif check == "pending_updates":
                findings.extend(
                    _check_pending_updates(
                        real_executor, _PS_TIMEOUTS[_PS_PENDING_UPDATES], current_os, section
                    )
                )
            elif check == "self_review":
                findings.extend(_check_self_review(log_path, section))
    return findings
