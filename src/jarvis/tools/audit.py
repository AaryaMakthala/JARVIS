"""audit_run / defender_status / defender_quick_scan tools (Phase 7).

``audit_run`` and ``defender_status`` are strictly read-only.  ``defender_quick_scan``
starts a Defender quick scan in the background; it is a distinct Tier-1 action and
is NOT part of the read-only audit.  Every PowerShell command used here is taken
from the module-level whitelist constant (the same one the audit checks use);
no command string ever originates from the LLM or user text.
"""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from jarvis import config
from jarvis.audit import checks as audit_checks
from jarvis.audit.report import write_report
from jarvis.tools.base import ToolContext, ToolResult, ToolSpec

#: Sections the planner may request; defaults to all of them.
_SECTIONS = ("security", "performance", "updates", "self")

#: Keep references to spawned scan processes so they are never garbage
#: collected (and therefore never warned/killed) while still running.
_SPAWNED_SCANS: list[subprocess.Popen[bytes]] = []


def _is_windows() -> bool:
    return platform.system() == "Windows"


class AuditRunArgs(BaseModel):
    """Which read-only audit sections to run."""

    model_config = ConfigDict(extra="forbid")

    sections: list[Literal["security", "performance", "updates", "self"]] = []


class DefenderStatusArgs(BaseModel):
    """Read Defender protection state (no arguments)."""

    model_config = ConfigDict(extra="forbid")


class DefenderQuickScanArgs(BaseModel):
    """Start a Defender quick scan in the background (Tier 1)."""

    model_config = ConfigDict(extra="forbid")


def _run_audit(args: AuditRunArgs, ctx: ToolContext) -> ToolResult:
    if ctx.dry_run:
        return ToolResult(
            ok=True,
            output=f"[dry-run] would run read-only audit sections: {args.sections or 'all'}",
        )
    sections = list(args.sections) or list(_SECTIONS)
    findings = audit_checks.run_checks(sections)
    report_path = write_report(findings, config.reports_dir())
    by_severity: dict[str, int] = {}
    for finding in findings:
        by_severity[finding.severity] = by_severity.get(finding.severity, 0) + 1
    summary = (
        f"Audit report: {report_path} ({len(findings)} checks, "
        f"{by_severity.get('critical', 0)} critical, "
        f"{by_severity.get('warning', 0)} warning)"
    )
    return ToolResult(
        ok=True,
        output=summary,
        data={
            "report_path": str(report_path),
            "sections": sections,
            "count": len(findings),
            "critical": by_severity.get("critical", 0),
            "warning": by_severity.get("warning", 0),
            "findings": [
                {
                    "section": f.section,
                    "severity": f.severity,
                    "title": f.title,
                    "detail": f.detail,
                }
                for f in findings
            ],
        },
    )


def _verify_audit(args: AuditRunArgs, result: ToolResult, ctx: ToolContext) -> ToolResult:
    del args, ctx
    report_path = result.data.get("report_path")
    verified = bool(report_path and Path(report_path).is_file())
    return result.model_copy(update={"verified": verified})


def _parse_defender_payload(out: str, err: str, rc: int) -> tuple[bool, str, dict[str, object]]:
    """Parse ``Get-MpComputerStatus`` output; (ok, summary, data)."""
    if rc != 0 or not out:
        reason = err or "no output"
        return False, f"Could not read Defender status ({reason})", {}
    data = audit_checks._parse_json(out)
    if isinstance(data, list):
        data = data[0] if data else None
    if not isinstance(data, dict):
        return False, "Unexpected Defender output", {}
    realtime = bool(data.get("RealTimeProtectionEnabled", False))
    enabled = bool(data.get("AntivirusEnabled", False))
    signature = str(data.get("AntivirusSignatureVersion") or "?")
    summary = (
        f"Defender: antivirus {'enabled' if enabled else 'DISABLED'}, "
        f"real-time {'ON' if realtime else 'OFF'}, signature {signature}"
    )
    return (
        True,
        summary,
        {"enabled": enabled, "real_time": realtime, "signature": signature, "raw": data},
    )


def _run_defender_status(args: DefenderStatusArgs, ctx: ToolContext) -> ToolResult:
    del args
    if ctx.dry_run:
        return ToolResult(ok=True, output="[dry-run] would read Defender status")
    if not _is_windows():
        return ToolResult(ok=True, output="Defender is a Windows feature; not applicable here")
    out, err, rc = audit_checks._run_powershell(
        audit_checks._PS_DEFENDER_STATUS,
        audit_checks._PS_TIMEOUTS[audit_checks._PS_DEFENDER_STATUS],
    )
    ok, summary, data = _parse_defender_payload(out, err, rc)
    if not ok:
        return ToolResult(ok=False, error=summary)
    return ToolResult(ok=True, output=summary, data=data)


def _verify_defender_status(
    args: DefenderStatusArgs, result: ToolResult, ctx: ToolContext
) -> ToolResult:
    del args, ctx
    return result.model_copy(update={"verified": bool(result.ok and result.data.get("raw"))})


def _run_defender_quick_scan(args: DefenderQuickScanArgs, ctx: ToolContext) -> ToolResult:
    del args
    if ctx.dry_run:
        return ToolResult(ok=True, output="[dry-run] would start a Defender quick scan")
    if not _is_windows():
        return ToolResult(ok=True, output="Defender is a Windows feature; not applicable here")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        proc = subprocess.Popen(
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                audit_checks._PS_QUICK_SCAN,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
        )
    except OSError as exc:
        return ToolResult(ok=False, error=f"Could not start quick scan ({exc})")
    _SPAWNED_SCANS.append(proc)
    return ToolResult(
        ok=True,
        output=f"Defender quick scan started (pid {proc.pid}) in the background.",
        data={"pid": proc.pid, "scan": "QuickScan"},
    )


def _verify_defender_quick_scan(
    args: DefenderQuickScanArgs, result: ToolResult, ctx: ToolContext
) -> ToolResult:
    del args, ctx
    if not result.ok or "pid" not in result.data:
        return result.model_copy(update={"verified": False})
    # The scan process runs detached; a live Popen means it started and is
    # still executing (a process that died instantly is a failed start).
    pid = int(result.data["pid"])
    found = None
    for proc in _SPAWNED_SCANS:
        if proc.pid == pid:
            found = proc
            break
    if found is None or found.poll() is not None:
        return result.model_copy(update={"verified": False})
    return result.model_copy(update={"verified": True})


def _describe_audit(args: AuditRunArgs) -> str:
    return f"Run read-only audit sections: {args.sections or 'all'}"


def make_audit_run_spec() -> ToolSpec:
    """Build the ``audit_run`` tool (Tier 0; read-only checks)."""
    return ToolSpec(
        name="audit_run",
        description=(
            "Run a read-only system audit (security, performance, updates, self) "
            "and write a local Markdown report. Never modifies the system."
        ),
        args_model=AuditRunArgs,
        base_tier=0,
        timeout_s=200,
        run=_run_audit,
        verify=_verify_audit,
        describe=_describe_audit,
    )


def make_defender_status_spec() -> ToolSpec:
    """Build the ``defender_status`` tool (Tier 0; read-only)."""
    return ToolSpec(
        name="defender_status",
        description=(
            "Read-only Windows Defender status: antivirus enabled, real-time "
            "protection, signature version. Does not modify anything."
        ),
        args_model=DefenderStatusArgs,
        base_tier=0,
        timeout_s=30,
        windows_only=True,
        run=_run_defender_status,
        verify=_verify_defender_status,
        describe=lambda args: "Read Windows Defender status",
    )


def make_defender_quick_scan_spec() -> ToolSpec:
    """Build the ``defender_quick_scan`` tool (Tier 1)."""
    return ToolSpec(
        name="defender_quick_scan",
        description=(
            "Start a Windows Defender quick scan in the background and return "
            "immediately. This is a separate Tier-1 action, not part of the "
            "read-only audit."
        ),
        args_model=DefenderQuickScanArgs,
        base_tier=1,
        timeout_s=30,
        windows_only=True,
        run=_run_defender_quick_scan,
        verify=_verify_defender_quick_scan,
        describe=lambda args: "Start a Defender quick scan in the background",
    )
