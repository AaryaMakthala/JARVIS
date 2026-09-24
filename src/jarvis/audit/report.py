"""Markdown audit report generation (Phase 7).

Renders :class:`~jarvis.audit.checks.Finding` objects as a deterministic
Markdown document.  The summary is a plain-language roll-up with no LLM
involvement: audit data stays on the machine and is never sent anywhere, and
nothing the audit produced is ever interpreted as instructions by the agent.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from jarvis.audit.checks import Finding, Severity

_SEVERITY_LABELS: dict[Severity, str] = {
    "ok": "ok",
    "info": "info",
    "warning": "warning",
    "critical": "critical",
    "unknown": "unknown",
}


def render_markdown(
    findings: Sequence[Finding],
    *,
    generated_at: datetime | None = None,
    app: str = "JARVIS",
) -> str:
    """Render a Markdown audit report from ``findings`` (no side effects).

    Sections keep the order in which the findings were produced; within a
    section, findings keep their order.  Severity counts are rolled up so the
    report opens with an at-a-glance summary line.
    """
    timestamp = (generated_at or datetime.now(UTC)).astimezone(UTC)
    if not findings:
        return (
            f"# {app} System Audit\n\nGenerated: {timestamp.isoformat()}\n\nNo checks were run.\n"
        )

    counts: dict[str, int] = {}
    by_section: dict[str, list[Finding]] = {}
    for finding in findings:
        counts[finding.severity] = counts.get(finding.severity, 0) + 1
        by_section.setdefault(finding.section, []).append(finding)

    lines = [
        f"# {app} System Audit",
        "",
        f"Generated: {timestamp.isoformat()}",
        "",
        "These checks read the system only; nothing was modified.",
        "",
        "## Summary",
        "",
    ]
    for severity in ("critical", "warning", "info", "ok", "unknown"):
        lines.append(f"- {_SEVERITY_LABELS[severity]}: {counts.get(severity, 0)}")
    lines.extend(["", f"Total findings: {len(findings)}", ""])

    for section, section_findings in by_section.items():
        lines.append(f"## {section}")
        lines.append("")
        for finding in section_findings:
            title = f"[{_SEVERITY_LABELS[finding.severity]}] {finding.title}"
            if finding.detail:
                lines.append(f"- **{title}** — {finding.detail}")
            else:
                lines.append(f"- **{title}**")
            if finding.evidence:
                lines.append(f"  - {finding.evidence}")
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Items marked `unknown` could not be verified (e.g. they need admin); the audit never guesses."
    )
    lines.append("- This report is generated locally and never leaves the machine.")
    return "\n".join(lines).rstrip() + "\n"


def write_report(findings: Sequence[Finding], target_dir: Path) -> Path:
    """Write a timestamped Markdown report and return its path.

    ``target_dir`` is created if absent.  This is the ONLY function in the
    audit package that writes to disk; :func:`run_checks` never writes.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"audit-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}.md"
    path.write_text(render_markdown(findings), encoding="utf-8")
    return path
