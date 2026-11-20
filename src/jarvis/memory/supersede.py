"""Apply a parsed correction to the existing SQLite preference store.

Supersession is deliberately conservative: a correction is only applied when
an exact alias or a well-known value identifies one canonical topic that is
already stored.  Old values are archived under a non-retrievable key, so they
remain inspectable in SQLite without competing with the active value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from jarvis.memory.aliases import (
    canonical_value,
    fact_key_for_topic,
    topic_from_text,
    topic_from_value,
)
from jarvis.memory.correction import CorrectionIntent
from jarvis.memory.prefs import Preference

__all__ = ["SupersessionResult", "apply_correction"]

Status = Literal["applied", "not_found", "ambiguous", "rejected", "error"]


@dataclass(frozen=True)
class SupersessionResult:
    """Outcome of one correction; no user value is ever included."""

    status: Status
    topic: str = ""
    archived: int = 0


def apply_correction(preferences: Any, intent: CorrectionIntent) -> SupersessionResult:
    """Archive matching old facts and store the active replacement.

    ``preferences`` is the existing :class:`~jarvis.memory.prefs.PreferenceStore`
    (or a test double with the same small surface).  No schema change, database
    or LLM call is involved.
    """
    if intent.error:
        return SupersessionResult(status="rejected")
    topic = _resolve_topic(intent)
    if not topic:
        return SupersessionResult(status="ambiguous")

    canonical_key = fact_key_for_topic(topic)
    active = list(preferences.active(limit=preferences.max_rows))
    matches = [pref for pref in active if _pref_topic(pref) == topic]
    if not matches:
        return SupersessionResult(status="not_found")

    targets = (
        matches
        if intent.action == "remove"
        else [pref for pref in matches if pref.key != canonical_key]
    )
    try:
        for pref in targets:
            preferences.archive(pref.key, pref.value)
            preferences.delete(pref.key)
        if intent.action == "replace":
            preferences.set(canonical_key, canonical_value(topic, intent.value))
    except (ValueError, TypeError):
        return SupersessionResult(status="rejected")
    except Exception:  # noqa: BLE001 - memory must never break a user turn
        return SupersessionResult(status="error")
    return SupersessionResult(status="applied", topic=topic, archived=len(targets))


def _resolve_topic(intent: CorrectionIntent) -> str:
    """Combine the stated alias and the value lexicon without guessing."""
    from_value = topic_from_value(intent.value)
    if intent.topic and from_value and intent.topic != from_value:
        return ""
    return intent.topic or from_value or ""


def _pref_topic(pref: Preference) -> str:
    """Identify an existing row from its value, falling back to its key."""
    return (
        topic_from_text(pref.value)
        or topic_from_value(pref.value)
        or topic_from_text(pref.key)
        or ""
    )
