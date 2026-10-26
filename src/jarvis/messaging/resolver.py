"""Contact name resolution: exact, alias, or a *suggestion* - never a guess.

The old lookup was exact-then-substring, which missed the mistakes people
actually make ("pappa" for "Papa", "der Arzt" for "Arzt") and could match
several contacts without saying so.  This module keeps resolution deterministic
and explainable: it returns a closed ``status``, at most three candidate names
and a short reason code.

``near`` and ``ambiguous`` are deliberately **not** a selection.  The caller
must clarify, and a send tool must refuse rather than pick one - an unambiguous
``exact``/``alias`` hit is the only outcome a send may act on.

Normalisation (NFKC, casefold, emoji/punctuation dropped, spaces collapsed) makes
the comparison insensitive to *how* a name was typed without hiding a real
difference, and queries shorter than three characters are never fuzzily matched.
"""

from __future__ import annotations

import difflib
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from jarvis.messaging.contacts import Contact

__all__ = ["Resolution", "normalize_name", "resolve_contact"]

#: Fuzzy matching starts at this many characters (a 2-character query would
#: "near-match" half the address book).
_MIN_QUERY_CHARS = 3
#: Minimum ``difflib`` similarity for a near match.
_NEAR_RATIO = 0.85
#: Similarity given to a query that is a whole-word subset of the name.
_TOKEN_SCORE = 1.0
#: Similarity given to a query that appears inside the name (>= 3 chars).
_SUBSTRING_SCORE = 0.9
#: At most this many names are ever suggested.
_MAX_MATCHES = 3

STATUS_EXACT = "exact"
STATUS_ALIAS = "alias"
STATUS_NEAR = "near"
STATUS_AMBIGUOUS = "ambiguous"
STATUS_NONE = "none"


@dataclass(frozen=True)
class Resolution:
    """One lookup outcome: a closed status, up to 3 names, a short reason code."""

    status: str
    matches: tuple[str, ...] = ()
    reason: str = ""

    @property
    def selected(self) -> str:
        """The single name this lookup may act on, or ``""`` for a suggestion."""
        if self.status in (STATUS_EXACT, STATUS_ALIAS) and len(self.matches) == 1:
            return self.matches[0]
        return ""


def normalize_name(text: str) -> str:
    """NFKC + casefold, with emoji/punctuation dropped and spaces collapsed."""
    folded = unicodedata.normalize("NFKC", str(text or "")).casefold()
    kept = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in folded)
    return " ".join(kept.split())


def _aliases(contact: Contact) -> tuple[str, ...]:
    """Optional extra names for ``contact`` (dormant until the file format has them)."""
    raw = getattr(contact, "aliases", ())
    if isinstance(raw, str):
        return (raw,)
    return tuple(str(item) for item in raw if str(item).strip()) if raw else ()


def _one_or_many(names: list[str], status: str) -> Resolution | None:
    """``status`` for exactly one distinct name, ``ambiguous`` for several."""
    unique = list(dict.fromkeys(names))
    if not unique:
        return None
    if len(unique) == 1:
        return Resolution(status, (unique[0],), status)
    return Resolution(STATUS_AMBIGUOUS, tuple(unique[:_MAX_MATCHES]), "ambiguous")


def _score(wanted: str, wanted_tokens: set[str], name: str) -> tuple[float, str]:
    """``(score, reason code)`` for one contact name (0.0 when not a candidate)."""
    name_tokens = set(name.split())
    if wanted_tokens and wanted_tokens <= name_tokens:
        return _TOKEN_SCORE, "near_tokens"
    if len(wanted) >= _MIN_QUERY_CHARS and wanted in name:
        return _SUBSTRING_SCORE, "near_substring"
    ratio = difflib.SequenceMatcher(None, wanted, name).ratio()
    if ratio >= _NEAR_RATIO:
        return ratio, "near_ratio"
    return 0.0, ""


def resolve_contact(query: str, contacts: Sequence[Contact]) -> Resolution:
    """Resolve ``query`` against ``contacts``; never choose a near or ambiguous hit.

    Order: exact normalised name, then an exact alias, then fuzzy candidates.
    Fuzzy matching never runs for a query shorter than
    :data:`_MIN_QUERY_CHARS` (after normalisation), and two or more candidates
    are ``ambiguous`` with up to :data:`_MAX_MATCHES` names rather than a pick.
    """
    wanted = normalize_name(query)
    if not wanted:
        return Resolution(STATUS_NONE, (), "empty_query")
    hit = _one_or_many(
        [contact.name for contact in contacts if normalize_name(contact.name) == wanted],
        STATUS_EXACT,
    )
    if hit is not None:
        return hit
    hit = _one_or_many(
        [
            contact.name
            for contact in contacts
            if any(normalize_name(alias) == wanted for alias in _aliases(contact))
        ],
        STATUS_ALIAS,
    )
    if hit is not None:
        return hit
    if len(wanted.replace(" ", "")) < _MIN_QUERY_CHARS:
        return Resolution(STATUS_NONE, (), "query_too_short")
    wanted_tokens = set(wanted.split())
    scored: list[tuple[float, str, str]] = []
    for contact in contacts:
        name = normalize_name(contact.name)
        if not name:
            continue
        score, code = _score(wanted, wanted_tokens, name)
        if score > 0.0:
            scored.append((score, contact.name, code))
    if not scored:
        return Resolution(STATUS_NONE, (), "no_match")
    scored.sort(key=lambda item: (-item[0], normalize_name(item[1])))
    if len(scored) > 1:
        return Resolution(
            STATUS_AMBIGUOUS, tuple(item[1] for item in scored[:_MAX_MATCHES]), "ambiguous"
        )
    score, name, code = scored[0]
    del score
    return Resolution(STATUS_NEAR, (name,), code)
