"""Contact name resolution: exact, alias, or a *suggestion* - never a guess.

The old lookup was exact-then-substring, which missed the mistakes people
actually make ("pappa" for "Papa", "der Arzt" for "Arzt") and could match
several contacts without saying so.  This module keeps resolution deterministic
and explainable: it returns a closed ``status``, at most three candidate names
and a short reason code.

``near`` and ``ambiguous`` are deliberately **not** a selection.  The caller
must clarify, and a send tool must refuse rather than pick one - an unambiguous
``exact``/``alias`` hit is the only outcome a send may act on.

Normalisation is NFC for canonical-equivalence, then a presentation form that
keeps emoji/emoji modifiers/punctuation as comparison noise while still
collapsing carets/spaces/letter-case.  NFKC would turn emoji sequences into
their text/emoji presentation forms and fold skin-tone/family/ZWJ sequences in
unpredictable ways, so it is **not** used as the comparison key: the saved
display name is stored NFC and compared NFC-first, and the more aggressive
normalisation is only ever a lookup helper that never becomes the trusted
identity.

Queries shorter than three characters are never fuzzily matched.
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
    """Presentation normalisation for lookup only: NFC + casefold, then emoji,
    emoji modifiers, punctuation and symbols are treated as comparison noise and
    replaced with spaces so that "Mom ❤️" can still find a person who saved
    the contact as "Mom".

    This is deliberately **not** NFKC and **not** an identity pass: emoji
    presentation selectors, skin-tone modifiers, ZWJ sequences and regional
    indicators are dropped only from this *lookup hint*.  The saved contact name
    itself is stored and compared NFC-first, so two genuinely different
    decorated names ("Mom ❤️" vs "Mom 💙") remain distinct when the caller asks
    for one of them, and both normalise to the same bare word so that a bare
    "Mom" is reported as ambiguous rather than silently choosing one.

    This normalised form is used for *lookup hints*, never as the authoritative
    identity: the exact NFC normalised name is the canonical key, and the trusted
    phone number remains the authority for WhatsApp onward navigation.
    """
    text = unicodedata.normalize("NFC", str(text or ""))
    folded = text.casefold()
    kept = "".join(ch if _is_name_char(ch) else " " for ch in folded)
    return " ".join(kept.split())


def _is_name_char(ch: str) -> bool:
    """Characters that survive presentation normalisation as comparison evidence.

    Letters, combining marks, decimal digits and whitespace keep their meaning
    across real names (an Indic vowel sign or a decomposed accent is part of the
    name, not decoration).  Emoji, emoji presentation selectors, skin-tone
    modifiers, regional indicators, tag characters and punctuation are turned
    into comparison noise for lookup hints only.

    Variation selectors need an explicit rule: ``U+FE0F`` is category ``Mn``, so
    a category-only test would keep it and make ``"Mom \u2764\ufe0f"`` normalise to
    ``"mom \ufe0f"`` while ``"Mom \U0001f499"`` normalises to ``"mom"``.  That split
    is what stopped a decorated name from resolving exactly (and let a bare word
    silently choose the other decorated contact).
    """
    if ch.isspace():
        return True
    code = ord(ch)
    # Emoji presentation/identity noise that is never part of a human name key.
    if 0xFE00 <= code <= 0xFE0F or 0xE0100 <= code <= 0xE01EF:
        return False  # variation selectors (U+FE0F is Mn)
    if code in (0x200D, 0x20E3):  # zero-width joiner, combining enclosing keycap
        return False
    if 0x1F3FB <= code <= 0x1F3FF:  # emoji skin-tone modifiers
        return False
    if 0x1F1E6 <= code <= 0x1F1FF:  # regional indicator flags
        return False
    if 0xE0020 <= code <= 0xE007F:  # emoji tag sequences
        return False
    category = unicodedata.category(ch)
    return category.startswith(("L", "M", "Nd"))


def _aliases(contact: Contact) -> tuple[str, ...]:
    """Optional extra names for ``contact`` (dormant until the file format has them)."""
    raw = getattr(contact, "aliases", ())
    if isinstance(raw, str):
        raw = (raw,)
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

    Order:

    1. Exact saved name, after NFC canonical-equivalence (so the lookup key is
       the same canonical form the store uses and a decorated name still matches
       the stored decorated name exactly).
    2. Exact normalised presentation (emoji/punctuation/case-insensitive) so a
       bare word can find a decorated contact **only when exactly one saved
       contact normalises to that word**.
    3. Exact alias, in the same two-step form.
    4. Fuzzy candidates for a longer query.

    Fuzzy matching never runs for a query shorter than
    :data:`_MIN_QUERY_CHARS` (after normalisation), and two or more candidates
    are ``ambiguous`` with up to :data:`_MAX_MATCHES` names rather than a pick.

    The lookup normalisation is a *hint* only: it can widen the set of queries
    that find a decorated contact, but the exact NFC name and the trusted phone
    number remain the identity proof.  Emoji-stripped forms never become
    authoritative.
    """
    wanted_canonical = unicodedata.normalize("NFC", str(query or "")).casefold()
    wanted_canonical = " ".join(wanted_canonical.split())
    if not wanted_canonical:
        return Resolution(STATUS_NONE, (), "empty_query")
    # The exact NFC name is authoritative and runs first, even for a name made
    # only of emoji (whose presentation form is empty).
    exact = [
        contact.name
        for contact in contacts
        if unicodedata.normalize("NFC", contact.name).casefold() == wanted_canonical
    ]
    hit = _one_or_many(exact, STATUS_EXACT)
    if hit is not None:
        return hit
    wanted = normalize_name(query)
    if not wanted:
        return Resolution(STATUS_NONE, (), "no_name_characters")
    hit = _one_or_many(
        [
            contact.name
            for contact in contacts
            if normalize_name(contact.name) == wanted
        ],
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
