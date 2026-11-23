"""Deterministic pre-planning feasibility checks (Phase 8).

The brain may only propose actions from the closed tool registry.  This module
adds a code-owned gate *before* the planner call: it checks the request against
registered capabilities and the runtime integrations wired into
:class:`~jarvis.agent.context.AppContext`.  It never authorizes an action (the
policy engine still does that) and never writes to ``AgentState``.

The checks are deliberately conservative.  Ordinary conversation and generic
action phrasing pass through; only strong, recognizable negatives stop the
planner or become a bounded limitation note.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from jarvis.checks import Check
from jarvis.platform_guard import is_windows

__all__ = [
    "NEEDS_CLARIFICATION",
    "PARTIALLY_SUPPORTED",
    "SUPPORTED",
    "UNSUPPORTED",
    "FeasibilityReport",
    "assess_request",
    "constraint_text",
    "partial_note",
    "tool_availability",
]

Status = Literal["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "NEEDS_CLARIFICATION"]

SUPPORTED: Status = "SUPPORTED"
PARTIALLY_SUPPORTED: Status = "PARTIALLY_SUPPORTED"
UNSUPPORTED: Status = "UNSUPPORTED"
NEEDS_CLARIFICATION: Status = "NEEDS_CLARIFICATION"

_MAX_NOTE_CHARS = 220

_ACTION_RE = re.compile(
    r"\b(?:open|launch|close|quit|run|start|stop|type|write|save|create|make|delete|"
    r"remove|find|search|look\s+up|browse|navigate|click|press|send|message|read|list|"
    r"show|display|take|capture|record|mute|unmute|lock|unlock|scan|audit|check|remember|"
    r"forget|set|turn|disable|enable|book|reserve|buy|purchase|pay|transfer|reset|change)\b",
    re.IGNORECASE,
)

_CLAUSE_SPLIT = re.compile(
    r"\s+(?:and|then|also)\s+|\s*;\s*|\s*,\s*then\s+",
    re.IGNORECASE,
)

_DEICTIC_RE = re.compile(r"\b(?:this|that|it|them|these|those)\b", re.IGNORECASE)
_REFERENT_OBJECT_RE = re.compile(
    r"\b(?:message|photo|picture|file|link|page|ticket|chat|contact|window|item|one)\b",
    re.IGNORECASE,
)

_HARD_UNSUPPORTED: tuple[tuple[str, re.Pattern[str], tuple[str, ...]], ...] = (
    (
        "book a flight",
        re.compile(
            r"\b(?:book|reserve|buy|purchase)\b.{0,40}"
            r"\b(?:flight|airline|air\s+ticket|airline\s+ticket)\b",
            re.IGNORECASE,
        ),
        ("book_flight", "flight_booking", "travel_booking"),
    ),
    (
        "book a hotel",
        re.compile(
            r"\b(?:book|reserve|buy|purchase)\b.{0,40}\b(?:hotel|accommodation)\b",
            re.IGNORECASE,
        ),
        ("book_hotel", "hotel_booking", "travel_booking"),
    ),
    (
        "make a payment",
        re.compile(
            r"\b(?:pay|purchase|buy|transfer)\b.{0,40}"
            r"\b(?:bill|invoice|payment|subscription|order|money)\b",
            re.IGNORECASE,
        ),
        ("pay_bill", "payment", "purchase", "transfer_money"),
    ),
    (
        "change a password",
        re.compile(
            r"\b(?:change|reset|recover)\b.{0,40}\b(?:password|credentials?)\b",
            re.IGNORECASE,
        ),
        ("change_password", "reset_password", "password_manager"),
    ),
    (
        "disable a security control",
        re.compile(
            r"\b(?:disable|turn\s+off|switch\s+off)\b.{0,40}"
            r"\b(?:defender|firewall|antivirus)\b",
            re.IGNORECASE,
        ),
        ("disable_defender", "disable_firewall", "disable_antivirus"),
    ),
)

_WHATSAPP_TOOLS = frozenset(
    {
        "whatsapp_send",
        "whatsapp_send_message",
        "whatsapp_send_file",
        "whatsapp_open_chat",
        "whatsapp_read_messages",
        "whatsapp_list_unread",
        "whatsapp_verify_message",
    }
)


@dataclass(frozen=True, slots=True)
class FeasibilityReport:
    """Transient result of the pre-planner capability check."""

    status: Status
    reason: str
    available_tools: tuple[str, ...] = ()
    unsupported_parts: tuple[str, ...] = ()
    checks: tuple[Check, ...] = ()


def tool_availability(name: str, ctx: Any) -> tuple[bool, str]:
    """Return whether a registered tool can run in this process.

    This is an environment check, not an authorization check.  It never
    launches a browser, opens a chat, or reads a credential; it only reports
    integrations the caller has already wired into the context.
    """
    spec = ctx.registry.get_optional(name)
    if spec is None:
        return False, f"tool {name!r} is not registered"
    if spec.windows_only and not is_windows():
        return False, f"{name} is only available on Windows"
    if name.startswith("browser_") and getattr(ctx, "browser", None) is None:
        return False, "the managed browser is not available in this session"
    if name in _WHATSAPP_TOOLS and getattr(ctx, "whatsapp_browser", None) is None:
        return False, "WhatsApp Web is not available in this session"
    return True, ""


def assess_request(state: dict[str, Any], ctx: Any) -> FeasibilityReport:
    """Classify one request against real registry and runtime state."""
    text = " ".join(str(state.get("user_input") or "").split())
    if not text:
        return FeasibilityReport(SUPPORTED, "No pre-planning restriction.")

    clauses = [part.strip() for part in _CLAUSE_SPLIT.split(text) if part.strip()]
    if not clauses:
        clauses = [text]

    results = [_classify_clause(part, state, ctx) for part in clauses]
    checks = tuple(check for result in results for check in result.checks)
    available = tuple(dict.fromkeys(name for result in results for name in result.available_tools))
    unsupported = tuple(
        dict.fromkeys(part for result in results for part in result.unsupported_parts)
    )

    statuses = {result.status for result in results}
    if statuses == {NEEDS_CLARIFICATION}:
        question = next(result.reason for result in results if result.status == NEEDS_CLARIFICATION)
        return FeasibilityReport(NEEDS_CLARIFICATION, question, available, unsupported, checks)
    if UNSUPPORTED in statuses:
        details = "; ".join(dict.fromkeys(result.reason for result in results if result.reason))
        if available:
            return FeasibilityReport(
                PARTIALLY_SUPPORTED,
                details,
                available,
                unsupported or tuple(part for part in clauses if part),
                checks,
            )
        return FeasibilityReport(UNSUPPORTED, details, (), unsupported, checks)
    return FeasibilityReport(
        SUPPORTED, "Requested capabilities are available.", available, (), checks
    )


def constraint_text(report: FeasibilityReport) -> str | None:
    """Planner-facing instructions derived from a deterministic report."""
    if report.status == SUPPORTED:
        return None
    if report.status == PARTIALLY_SUPPORTED:
        parts = ", ".join(report.unsupported_parts) or "part of the request"
        return (
            "Feasibility check: only plan the parts this PC can do. "
            f"Do not plan or claim these parts: {parts}. "
            "If nothing useful remains, say so without calling a tool."
        )
    return None


def partial_note(report: FeasibilityReport) -> str | None:
    """Short limitation carried by a tool plan's ``dialog_answer``."""
    if report.status != PARTIALLY_SUPPORTED or not report.unsupported_parts:
        return None
    parts = ", ".join(report.unsupported_parts)
    note = f"I can't do this part here: {parts}."
    return " ".join(note.split())[:_MAX_NOTE_CHARS]


def _classify_clause(text: str, state: dict[str, Any], ctx: Any) -> FeasibilityReport:
    if not _ACTION_RE.search(text):
        return FeasibilityReport(SUPPORTED, "Conversation; no capability check needed.")

    hard = _hard_unsupported(text, ctx)
    if hard is not None:
        return FeasibilityReport(UNSUPPORTED, hard, (), (text,))

    if _needs_clarification(text, state):
        return FeasibilityReport(
            NEEDS_CLARIFICATION,
            f"Which one do you mean by {text}?",
            (),
            (),
        )

    from jarvis.agent.capability_index import build_capability_index, discover_candidates

    agent = getattr(getattr(ctx, "settings", None), "agent", None)
    limit = max(1, int(getattr(agent, "tool_candidate_limit", 5) or 5))
    index = getattr(ctx, "capability_index", None) or build_capability_index(ctx.registry)
    names = (
        list(index.names()) if index.count() <= 8 else discover_candidates(index, text, limit=limit)
    )
    available: list[str] = []
    failures: list[Check] = []
    for name in names:
        ok, reason = tool_availability(name, ctx)
        if ok:
            available.append(name)
        elif reason:
            failures.append(Check(name, "WARN", reason))

    if names and not available:
        reason = failures[0].detail if failures else "no usable capability is available"
        return FeasibilityReport(UNSUPPORTED, reason, (), (text,), tuple(failures))
    return FeasibilityReport(
        SUPPORTED, "Capability check passed.", tuple(available), (), tuple(failures)
    )


def _hard_unsupported(text: str, ctx: Any) -> str | None:
    names = tuple(name.casefold() for name in ctx.registry.names())
    for label, pattern, coverage in _HARD_UNSUPPORTED:
        if not pattern.search(text):
            continue
        if any(any(token in name for token in coverage) for name in names):
            continue
        return f"I can't {label} with the tools available here."
    return None


def _needs_clarification(text: str, state: dict[str, Any]) -> bool:
    if not _DEICTIC_RE.search(text) or not _REFERENT_OBJECT_RE.search(text):
        return False
    refs = state.get("resolved_refs")
    if isinstance(refs, dict) and any(value for value in refs.values()):
        return False
    return not str(state.get("clarification_answer") or "").strip()
