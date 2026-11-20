"""Deterministic recognition of short correction statements.

This parser never writes memory and never calls an LLM.  It only recognises a
small set of explicit correction forms; anything else is left to the normal
conversation path so a casual remark cannot silently overwrite a fact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from jarvis.memory.aliases import topic_from_text
from jarvis.memory.intent import MAX_FACT_CHARS, is_sensitive_fact

__all__ = ["CorrectionIntent", "detect_correction"]

_ADDITIVE = re.compile(
    r"\b(?:also|too|as well|another|in addition|besides|additionally)\b", re.IGNORECASE
)
_QUESTION = re.compile(r"\?|^(?:do|does|did|what|which|who|where|when|why|how)\b", re.IGNORECASE)
_MARKER = re.compile(
    r"\b(?:actually|from now on|now|instead|switched|changed|no longer|stopped|quit)\b",
    re.IGNORECASE,
)
_CHANGE_ONLY = re.compile(r"^(?:actually[, ]*)?my\s+preference\s+has\s+changed\b", re.IGNORECASE)
_REMOVAL = (
    re.compile(
        r"^(?:actually[, ]+)?i\s+(?:no\s+longer|don't|do\s+not)\s+"
        r"(?:really\s+)?(?:use|prefer|like|want)\s+(?P<value>.+?)"
        r"(?:\s+(?:anymore|now))?[.!?]*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:actually[, ]+)?i\s+(?:stopped|quit)\s+(?:using|preferring)\s+"
        r"(?P<value>.+?)[.!?]*$",
        re.IGNORECASE,
    ),
)
_RESPONSE_LANGUAGE = re.compile(
    r"^(?:actually[, ]+)?i\s+want\s+responses\s+in\s+(?P<value>.+?)"
    r"(?:\s+now)?[.!?]*$",
    re.IGNORECASE,
)
_SWITCH = re.compile(
    r"^(?:actually[, ]+)?i(?:'ve|\s+have)?\s+(?:switched|changed)\s+"
    r"(?:from\s+(?P<old>.+?)\s+)?to\s+(?P<value>.+?)[.!?]*$",
    re.IGNORECASE,
)
_PREFERENCE = (
    re.compile(
        r"^from\s+now\s+on[, ]+i\s+(?:prefer|use|want)\s+(?P<value>.+?)[.!?]*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:actually[, ]+)?i\s+(?:prefer|use|want)\s+(?P<value>.+?)"
        r"(?:\s+(?:now|instead))?[.!?]*$",
        re.IGNORECASE,
    ),
)


@dataclass(frozen=True)
class CorrectionIntent:
    """One parsed correction, with no identity resolution or side effects."""

    action: Literal["replace", "remove"]
    value: str
    topic: str
    error: str = ""

    @property
    def valid(self) -> bool:
        """True when the statement is precise enough to resolve and persist."""
        return not self.error and bool(self.value)


def detect_correction(user_input: str) -> CorrectionIntent | None:
    """Recognise a bounded correction form, or return ``None``."""
    text = " ".join((user_input or "").split()).strip()
    if not text or _QUESTION.search(text) or _ADDITIVE.search(text):
        return None
    if not _MARKER.search(text):
        return None

    for pattern in _REMOVAL:
        match = pattern.match(text)
        if match is not None:
            return _build("remove", match.group("value"), text)
    if _CHANGE_ONLY.match(text):
        return CorrectionIntent(action="replace", value="", topic="", error="ambiguous")

    match = _RESPONSE_LANGUAGE.match(text)
    if match is not None:
        return _build("replace", match.group("value"), text)
    match = _SWITCH.match(text)
    if match is not None:
        return _build("replace", match.group("value"), text)
    for pattern in _PREFERENCE:
        match = pattern.match(text)
        if match is not None:
            return _build("replace", match.group("value"), text)
    return None


def _build(action: Literal["replace", "remove"], raw_value: str, text: str) -> CorrectionIntent:
    """Clean, validate and attach an explicit topic hint when one is stated."""
    value = " ".join(raw_value.split()).strip(" \"'`.,!?\u201c\u201d\u2018\u2019")
    if not value:
        return CorrectionIntent(action=action, value="", topic="", error="ambiguous")
    if re.search(r"\b(?:and|or)\b", value, re.IGNORECASE):
        return CorrectionIntent(action=action, value=value, topic="", error="ambiguous")
    if len(value) > MAX_FACT_CHARS:
        return CorrectionIntent(action=action, value=value, topic="", error="too_long")
    if is_sensitive_fact(value):
        return CorrectionIntent(action=action, value=value, topic="", error="sensitive")
    return CorrectionIntent(action=action, value=value, topic=topic_from_text(text) or "")
