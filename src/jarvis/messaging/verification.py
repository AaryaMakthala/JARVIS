"""Read-only status evidence for a dispatched WhatsApp text message (W5).

Verification is a *reading* problem, not a sending one.  This module turns the
observable evidence of one bounded conversation read into an explicit status,
and refuses to move up the ladder without direct evidence:

``UNKNOWN``     evidence is missing, ambiguous, contradictory or unreadable;
``DISPATCHED``  the Send click happened, but WhatsApp shows no sent evidence
                (or only a "pending/clock" indicator);
``SENT``        the single-check "sent" indicator is visible;
``DELIVERED``   the double-check "delivered" indicator is visible;
``READ``        the blue double-check / "read" indicator is visible.

A message that merely *appears* in the sender's own conversation proves
nothing: it is present whether or not the click reached the server, so the
status is never inferred from presence alone.  A higher status is never
derived from a lower one.

The evidence vocabulary is WhatsApp Web's status node: its ``data-icon`` name
(``msg-check``, ``msg-dblcheck``, ``msg-dblcheck-ack``, ``msg-time``) and the
accessible ``aria-label``/``title`` text of that node.  The page reports the
attributes it actually found; this module only understands those tokens, so
unknown markup degrades to ``UNKNOWN`` instead of being guessed.

Nothing here touches the DOM, the browser or the network, and no message body
is ever returned by these helpers.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from jarvis.browser.whatsapp_web import WhatsAppMessage

__all__ = [
    "MessageMatch",
    "MessageStatus",
    "StatusEvidence",
    "digest_text",
    "match_outgoing",
    "status_from_hints",
]

logger = logging.getLogger(__name__)

#: Cap on how much of one status hint is echoed back as evidence.
_MAX_HINT_CHARS = 80

_RANK: dict[MessageStatus, int] = {}


class MessageStatus(StrEnum):
    """What was actually observed about one outgoing message."""

    UNKNOWN = "unknown"
    DISPATCHED = "dispatched"
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"

    @property
    def rank(self) -> int:
        """Ordering used only to decide whether polling can stop."""
        return _RANK[self]


_RANK.update(
    {
        MessageStatus.UNKNOWN: 0,
        MessageStatus.DISPATCHED: 1,
        MessageStatus.SENT: 2,
        MessageStatus.DELIVERED: 3,
        MessageStatus.READ: 4,
    }
)

#: Tokens of WhatsApp Web's status node.  ``-ack`` (acknowledged) is the read
#: marker; a plain ``dblcheck`` is delivery, ``msg-check`` is sent, and
#: ``msg-time`` (clock) means the message is not on the server yet.
_READ_ICON_TOKENS = ("dblcheck-ack", "double-check-ack")
_DELIVERED_ICON_TOKENS = ("dblcheck", "double-check")
_SENT_ICON_TOKENS = ("msg-check", "single-check")
_PENDING_ICON_TOKENS = ("msg-time",)


def digest_text(text: str) -> str:
    """sha256 of the whitespace-collapsed message text.

    The page read collapses runs of whitespace, so the digest is computed over
    the same normalised form on both sides.  That makes matching independent of
    trailing/duplicated spaces while still separating genuinely different
    bodies.  Only the digest is ever stored.
    """
    normalised = " ".join(str(text or "").split())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class StatusEvidence:
    """One status read from a message's visible metadata."""

    status: MessageStatus
    evidence: tuple[str, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class MessageMatch:
    """Which message was identified, and the status evidence for it."""

    status: MessageStatus
    index: int | None = None
    evidence: tuple[str, ...] = ()
    reason: str = ""


def _classify_hint(hint: str) -> MessageStatus | None:
    """Map one status-node hint to a status, or ``None`` when unrecognised."""
    value = re.sub(r"[^a-z0-9-]+", " ", hint.casefold()).strip()
    if not value:
        return None
    if any(token in value for token in _READ_ICON_TOKENS) or re.search(r"\bread\b", value):
        return MessageStatus.READ
    if any(token in value for token in _DELIVERED_ICON_TOKENS) or re.search(
        r"\bdelivered\b", value
    ):
        return MessageStatus.DELIVERED
    if any(token in value for token in _SENT_ICON_TOKENS) or re.search(r"\bsent\b", value):
        return MessageStatus.SENT
    if any(token in value for token in _PENDING_ICON_TOKENS) or re.search(
        r"\b(pending|waiting|clock)\b", value
    ):
        return MessageStatus.DISPATCHED
    return None


def status_from_hints(
    hints: Sequence[str],
    *,
    is_outgoing: bool | None,
) -> StatusEvidence:
    """Map a message's status hints to one status, failing closed.

    Only outgoing messages carry a delivery indicator.  No recognised hint is
    ``UNKNOWN`` (never ``DISPATCHED``: the caller decides what a missing
    indicator means for a click it knows about).  Two *different* recognised
    indicators on one message are contradictory and therefore ``UNKNOWN``.
    """
    if is_outgoing is not True:
        return StatusEvidence(MessageStatus.UNKNOWN, (), "not_outgoing")
    found: dict[MessageStatus, list[str]] = {}
    for raw in hints or ():
        hint = " ".join(str(raw or "").split())[:_MAX_HINT_CHARS]
        if not hint:
            continue
        status = _classify_hint(hint)
        if status is None:
            continue
        bucket = found.setdefault(status, [])
        if hint not in bucket:
            bucket.append(hint)
    if not found:
        return StatusEvidence(MessageStatus.UNKNOWN, (), "no_status_evidence")
    if len(found) > 1:
        conflicting = tuple(
            hint for hints_for_status in found.values() for hint in hints_for_status
        )
        return StatusEvidence(MessageStatus.UNKNOWN, conflicting, "contradictory_evidence")
    status, evidence = next(iter(found.items()))
    return StatusEvidence(status, tuple(evidence), status.value)


def match_outgoing(
    messages: Iterable[WhatsAppMessage],
    *,
    digest: str,
) -> MessageMatch:
    """Identify the single outgoing message whose text matches ``digest``.

    Every requirement is enforced rather than assumed: the message must be
    outgoing, its text must hash to the recorded digest, it must carry a
    readable timestamp, and it must be unique in the bounded window.  Two
    identical bodies cannot be told apart, so they yield ``UNKNOWN`` instead of
    whichever bubble came first.
    """
    wanted = str(digest or "").strip()
    if not wanted:
        return MessageMatch(MessageStatus.UNKNOWN, reason="no_digest")
    ordered = list(messages)
    digest_hits = [
        (index, message)
        for index, message in enumerate(ordered)
        if digest_text(message.text) == wanted
    ]
    if not digest_hits:
        return MessageMatch(MessageStatus.UNKNOWN, reason="no_matching_message")
    outgoing = [(index, message) for index, message in digest_hits if message.is_outgoing is True]
    if not outgoing:
        return MessageMatch(MessageStatus.UNKNOWN, reason="not_outgoing")
    if len(outgoing) > 1:
        return MessageMatch(MessageStatus.UNKNOWN, reason="duplicate_matching_messages")
    index, message = outgoing[0]
    if not message.timestamp:
        return MessageMatch(MessageStatus.UNKNOWN, index=index, reason="missing_timestamp")
    evidence = status_from_hints(message.status_hints, is_outgoing=True)
    return MessageMatch(
        status=evidence.status,
        index=index,
        evidence=evidence.evidence,
        reason=evidence.reason,
    )
