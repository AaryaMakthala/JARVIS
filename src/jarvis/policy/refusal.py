"""System-written refusal reasons for a confirmation that was never approved.

The policy gate used to refuse with one sentence for every non-approving answer:
*"Refused by you (confirmation answered with 'no' or a mismatched action)."*  That
sentence is wrong whenever the user was **never asked** - a Tier 2 voice action
while ``voice.allow_tier2_by_voice`` is off, or a terminal approval with no
JARVIS password set - and it blames the user for a refusal JARVIS itself
produced.

This module holds the closed set of reason codes and their fixed text.  Two rules
make it safe to carry a reason through the resume answer:

* **Nothing here approves anything.**  A reason only chooses which sentence the
  gate prints.  The approval decision, the action hash and the tier all stay
  where they are - the PolicyEngine and :mod:`jarvis.agent.nodes.policy_gate`.
* **Only known reasons become text.**  :func:`refusal_text` looks the incoming
  value up in :data:`REFUSAL_TEXT`; anything missing, non-string or unknown
  falls back to the legacy sentence.  A client cannot inject free text into
  JARVIS's answer by setting ``reason``.

Every string below is written here, not by the model.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = [
    "DEFAULT_REFUSAL_TEXT",
    "KEY_REASON",
    "REASON_MISMATCH",
    "REASON_NO_PASSWORD",
    "REASON_TIER2_VOICE_DISABLED",
    "REASON_USER_DECLINED",
    "REFUSAL_TEXT",
    "refusal_answer",
    "refusal_text",
]

#: Optional key on a resume answer naming why it does not approve.  Additive: the
#: gate ignores it entirely unless the answer is already a refusal.
KEY_REASON = "reason"

REASON_TIER2_VOICE_DISABLED = "tier2_voice_disabled"
REASON_NO_PASSWORD = "no_password"
REASON_USER_DECLINED = "user_declined"
REASON_MISMATCH = "mismatch"

#: The only reason code -> sentence pairs JARVIS will ever say for a refusal.
REFUSAL_TEXT: dict[str, str] = {
    REASON_TIER2_VOICE_DISABLED: (
        "Deleting by voice is turned off. Enable it in settings or approve in the terminal."
    ),
    REASON_NO_PASSWORD: "No JARVIS password is set. Run jarvis password set.",
    REASON_USER_DECLINED: "Cancelled.",
    REASON_MISMATCH: "That approval did not match, so nothing was done.",
}

#: What the gate said before reasons existed, kept for every refusal we cannot
#: name (including a client that omits or forges ``reason``).
DEFAULT_REFUSAL_TEXT = "Refused by you (confirmation answered with 'no' or a mismatched action)."


def refusal_text(answer: Any) -> str:
    """The sentence for a non-approving resume *answer*.

    Unknown, missing or malformed reasons return :data:`DEFAULT_REFUSAL_TEXT`
    rather than being echoed, so this can never become a channel for injected
    text.
    """
    reason = answer.get(KEY_REASON) if isinstance(answer, Mapping) else None
    text = REFUSAL_TEXT.get(reason) if isinstance(reason, str) else None
    return text or DEFAULT_REFUSAL_TEXT


def refusal_answer(payload: Mapping[str, Any], reason: str | None = None) -> dict[str, Any]:
    """Build the non-approving resume answer for *payload*.

    Mirrors the answer the daemon and CLI already sent - ``approved=False`` plus
    the action hash the gate matches against - and adds ``reason`` only when one
    is known, so an unnamed refusal stays byte-identical to today's payload.
    """
    answer: dict[str, Any] = {"approved": False, "action_hash": payload.get("action_hash", "")}
    if reason is not None:
        answer[KEY_REASON] = reason
    return answer
