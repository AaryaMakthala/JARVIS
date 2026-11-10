"""Deterministic explicit-memory commands ("remember ...", "forget ...").

The LLM is not involved: a clear memory instruction is a local write, just
like ``--dry-run`` or a preference set by the CLI.  Recognition is deliberately
narrow - only an imperative sentence that starts with a supported verb is
treated as a memory command.  Questions ("do you remember ...?") and ordinary
sentences that merely contain the word "remember" fall through to the normal
brain path.

Facts reuse the existing preferences table, so they inherit its secret guard,
per-key overwrite, pruning and deletion.  Keys are deterministic: the same
subject (``my preferred coding language``) always maps to the same row, which
makes an update an overwrite and "forget" a single-key delete.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Literal

from jarvis.memory.db import looks_secret, secret_key
from jarvis.memory.prefs import FACT_KEY_PREFIX

__all__ = ["MAX_FACT_CHARS", "MemoryIntent", "detect_memory_intent"]

#: Bounded like ``PreferenceStore`` values (400 chars).  A longer statement is
#: not truncated (which would silently lose content); the command is rejected
#: with an honest answer instead.
MAX_FACT_CHARS = 400

_LEAD = r"^(?:(?:please|jarvis|hey\s+jarvis)[,\s]+)*"
_ASK = r"(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?)?"
_REMEMBER = re.compile(
    _LEAD + _ASK + r"(?:remember|note|keep\s+in\s+mind)\s+(?:that\s+)?(?P<content>.+)$",
    re.IGNORECASE | re.DOTALL,
)
_FORGET = re.compile(
    _LEAD + _ASK + r"(?:forget|delete\s+the\s+memory|remove\s+the\s+memory)\s+"
    r"(?:that\s+|about\s+)?(?P<content>.+)$",
    re.IGNORECASE | re.DOTALL,
)
_QUERY = re.compile(
    r"^(?:do|did|does)\s+you\s+(?:still\s+)?remember\b"
    r"|^(?:what|which|who|where|when|why|how)\b.*\bremember\b",
    re.IGNORECASE,
)
_IS = re.compile(r"^(?P<subject>.+?)\s+(?:is|are)\s+(?P<value>.+)$", re.IGNORECASE)
_WORD = re.compile(r"[a-z0-9]+")
_TRAILING = " .!?\u2026"

_PREFERENCE = re.compile(
    r"\b(?:prefer(?:s|red|ence)?|favorite|favourite|like|likes|love|usual|default)\b",
    re.IGNORECASE,
)
_PERSONAL = re.compile(
    r"\b(?:my|i|me|we|our|family|friend|sister|brother|mother|father|wife|husband|"
    r"partner|son|daughter|birthday|born|home|address|phone|email)\b",
    re.IGNORECASE,
)
_PROJECT = re.compile(
    r"\b(?:project|jarvis|work|job|school|college|course|team|client|deadline|"
    r"codebase|repository|repo)\b",
    re.IGNORECASE,
)

#: A credential word anywhere in a remembered fact is enough to refuse it.
#: This is stricter than :func:`looks_secret` (which needs a name/value pair),
#: because a remembered fact is replayed verbatim in later prompts.
_SENSITIVE_FACT = re.compile(
    r"\b(?:password|passwd|passcode|passphrase|api[\s_-]?key|access[\s_-]?key|"
    r"secret[\s_-]?key|private[\s_-]?key|client[\s_-]?secret|token|otp|"
    r"one[\s-]?time\s+(?:code|password)|credential|credentials|cookie|"
    r"seed[\s-]?phrase|recovery[\s-]?code|credit[\s-]?card|card[\s-]?number|"
    r"debit[\s-]?card|bank[\s-]?account|routing[\s-]?number|iban|ssn|"
    r"social[\s-]?security|passport[\s-]?number|aadhaar|aadhar|pin)\b",
    re.IGNORECASE,
)
_TOKEN_SHAPE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{8,}|gh[pousr]_[A-Za-z0-9]{8,}|"
    r"AKIA[0-9A-Z]{12,}|AIza[0-9A-Za-z_-]{20,}|"
    r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.)"
)
_LONG_NUMBER = re.compile(r"\b(?:\d[ -]?){13,19}\b")


@dataclass(frozen=True)
class MemoryIntent:
    """One deterministic memory command.

    ``error`` is empty for a valid command, or one of ``"sensitive"`` /
    ``"too_long"``.  The caller still answers the human honestly but writes
    nothing.  ``key`` is only meaningful for ``action="remember"``.
    """

    action: Literal["remember", "forget"]
    content: str
    key: str
    category: str
    error: str = ""

    @property
    def valid(self) -> bool:
        """True when the command may be persisted."""
        return not self.error


def detect_memory_intent(user_input: str) -> MemoryIntent | None:
    """Recognise a clear remember/forget command, or return ``None``.

    Nothing is written here and no exception is raised; this is a pure parse.
    """
    text = (user_input or "").strip()
    if not text or _QUERY.search(text):
        return None
    match = _REMEMBER.match(text)
    action: Literal["remember", "forget"] = "remember"
    if match is None:
        match = _FORGET.match(text)
        action = "forget"
    if match is None:
        return None
    content = _clean(match.group("content"))
    if not content:
        return None
    subject = _subject(content)
    key = _fact_key(subject)
    category = _category(subject)
    error = ""
    if _is_sensitive(content):
        error = "sensitive"
    elif len(content) > MAX_FACT_CHARS:
        error = "too_long"
    return MemoryIntent(action=action, content=content, key=key, category=category, error=error)


def _clean(content: str) -> str:
    text = " ".join(content.split()).strip()
    text = text.strip("\"'`\u201c\u201d\u2018\u2019")
    return text.rstrip(_TRAILING).strip()


def _subject(content: str) -> str:
    match = _IS.match(content)
    if match is None:
        return content
    subject = match.group("subject").strip()
    return subject if len(subject) >= 2 else content


def _category(subject: str) -> str:
    if _PREFERENCE.search(subject):
        return "preference"
    if _PERSONAL.search(subject):
        return "personal"
    if _PROJECT.search(subject):
        return "project"
    return "general"


def _fact_key(subject: str) -> str:
    normalised = " ".join(subject.casefold().split())
    slug = "_".join(_WORD.findall(normalised))[:34].strip("_") or "fact"
    # ``PreferenceStore`` rejects any key containing a credential hint as a
    # substring (for example "shopping" contains "pin").  The content itself
    # was already checked; if only the generated slug trips that guard, fall
    # back to a neutral slug and let the hash preserve identity.
    if secret_key(slug):
        slug = "fact"
    digest = hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:10]
    return f"{FACT_KEY_PREFIX}{_category(subject)}.{slug}_{digest}"


def _is_sensitive(content: str) -> bool:
    return bool(
        looks_secret(content)
        or _TOKEN_SHAPE.search(content)
        or _SENSITIVE_FACT.search(content)
        or _LONG_NUMBER.search(content)
    )
