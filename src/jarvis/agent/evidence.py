"""Deterministic evaluation of declared step evidence (P5).

A plan step can name one closed evidence contract for its ``expect`` condition.
This module decides whether the observed :class:`~jarvis.tools.base.ToolResult`
actually supplies that evidence.  It never calls a tool, a model or the policy
engine, never executes a side effect, and never copies untrusted output into a
note: every verdict note is a constant drawn from this module.

The vocabulary is deliberately small - only checks JARVIS can evaluate from a
tool result today.  Old plans with no contract are untouched by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jarvis.agent.state import EvidenceStatus

__all__ = [
    "EVIDENCE_KINDS",
    "EvidenceVerdict",
    "evaluate_evidence",
    "normalise_kind",
]

EVIDENCE_KINDS: tuple[str, ...] = (
    "action",
    "resource",
    "information",
    "state",
    "unverifiable",
)

_PRESENCE_KEYS = (
    "artifact",
    "results",
    "items",
    "entries",
    "paths",
    "files",
    "matches",
    "processes",
    "messages",
    "conversations",
)
_RESOURCE_KEYS = ("path", "file", "page_url", "url", "pid")
_INFORMATION_KEYS = ("text", "content", "answer", "output", "question", "question_id")
_ABSENCE_FLAGS = ("found", "present", "available")
_ABSENCE_PHRASES = frozenset(
    {"no results", "no result", "no files found", "nothing found", "not found"}
)

_CONFIRMED = {
    "action": "The action executed; execution itself was the requested evidence.",
    "resource": "The expected resource was reported by the tool.",
    "information": "The expected information was present in the tool result.",
    "state": "The expected state was independently observed.",
}
_CONTRADICTED = {
    "resource": "The expected resource was not present in the tool result.",
    "information": "The expected information was not present in the tool result.",
}
_INCONCLUSIVE = {
    "state": "The expected state could not be independently observed.",
    "resource": "The expected resource could not be independently observed.",
    "information": "The expected information could not be independently observed.",
    "unknown": "No recognised evidence check is available for this step.",
}
_UNVERIFIABLE = "The outcome could not be independently verified."


@dataclass(frozen=True, slots=True)
class EvidenceVerdict:
    """One deterministic verdict and a constant, safe explanation."""

    status: EvidenceStatus
    note: str


def normalise_kind(value: object) -> str:
    """Return a closed kind string, or ``""`` for absent/unknown values."""
    text = str(value or "").strip()
    return text if text in EVIDENCE_KINDS else ""


def evaluate_evidence(kind: object, result: Any) -> EvidenceVerdict:
    """Evaluate ``kind`` against an observed tool result.

    ``result`` is the tool result *after* the tool's own ``verify()`` ran, so a
    ``state`` contract can read its verdict.  A failed action is handled by the
    caller and is never confirmed here.  Unknown kinds fail closed as
    inconclusive instead of raising.
    """
    checked = normalise_kind(kind)
    if checked == "action":
        return EvidenceVerdict("confirmed", _CONFIRMED["action"])
    if checked == "unverifiable":
        return EvidenceVerdict("inconclusive", _UNVERIFIABLE)
    if checked == "state":
        return _state(result)
    if checked == "resource":
        return _resource(result)
    if checked == "information":
        return _information(result)
    return EvidenceVerdict("inconclusive", _INCONCLUSIVE["unknown"])


def _state(result: Any) -> EvidenceVerdict:
    if getattr(result, "verified", None) is True:
        return EvidenceVerdict("confirmed", _CONFIRMED["state"])
    if getattr(result, "verified", None) is False:
        return EvidenceVerdict("contradicted", "An independent check of the expected state failed.")
    return EvidenceVerdict("inconclusive", _INCONCLUSIVE["state"])


def _resource(result: Any) -> EvidenceVerdict:
    data = _data(result)
    if _reports_absence(result, data):
        return EvidenceVerdict("contradicted", _CONTRADICTED["resource"])
    if _has(data, _PRESENCE_KEYS) or _has(data, _RESOURCE_KEYS):
        return EvidenceVerdict("confirmed", _CONFIRMED["resource"])
    return EvidenceVerdict("inconclusive", _INCONCLUSIVE["resource"])


def _information(result: Any) -> EvidenceVerdict:
    data = _data(result)
    if _reports_absence(result, data):
        return EvidenceVerdict("contradicted", _CONTRADICTED["information"])
    if str(getattr(result, "output", "") or "").strip():
        return EvidenceVerdict("confirmed", _CONFIRMED["information"])
    if _has(data, _PRESENCE_KEYS) or _has(data, _INFORMATION_KEYS):
        return EvidenceVerdict("confirmed", _CONFIRMED["information"])
    return EvidenceVerdict("inconclusive", _INCONCLUSIVE["information"])


def _data(result: Any) -> dict[str, Any]:
    data = getattr(result, "data", None)
    return data if isinstance(data, dict) else {}


def _has(data: dict[str, Any], keys: tuple[str, ...]) -> bool:
    return any(key in data and _nonempty(data[key]) for key in keys)


def _nonempty(value: object) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, (str, list, tuple, dict, set)):
        return bool(value)
    if isinstance(value, int):
        return value != 0
    return bool(value)


def _reports_absence(result: Any, data: dict[str, Any]) -> bool:
    if any(key in data and not _nonempty(data[key]) for key in _PRESENCE_KEYS):
        return True
    if any(data.get(key) is False for key in _ABSENCE_FLAGS):
        return True
    count = data.get("count")
    if type(count) is int and count == 0:
        return True
    output = str(getattr(result, "output", "") or "").strip().casefold().rstrip(".")
    return output in _ABSENCE_PHRASES
