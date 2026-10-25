"""Compose rules: the honest short answer, decided from state alone.

The ``compose`` node runs after every terminal path and before ``respond``.  It
turns *what actually happened* into two strings:

* ``screen_answer`` - what the terminal shows.  Built from the same
  :func:`jarvis.agent.nodes.respond.base_answer` the graph has always shown, so
  the assumption prefix, the per-step lines and the failure text are unchanged;
* ``spoken_answer`` - what the voice channel says.  At most two short sentences,
  cut at a sentence boundary, so a long tool output is never read aloud.

Nothing here calls a model, reads a tool or changes a plan.  Every branch is a
function of ``cancelled`` / ``plan`` / ``request_kind`` / ``halted_reason`` /
``results`` / ``final_answer`` - the fields ``respond`` itself reads - so a
composed answer can never claim more than the state proves:

* success is only classified when **every** recorded result is ``ok`` and none
  is ``verified is False``; an unverified success says so;
* refusals and failures are never rewritten into a friendly template - the
  factual text (the halt reason, or the step error ``respond`` would show) is
  spoken as-is, only shortened to two sentences;
* butler wording ("sir", or a stored ``user_name``) is used only on the
  courtesies (success/partial/cancelled/clarify) and at most once per answer -
  never on a refusal, a failure, or a plain action acknowledgment.

Variant choice uses a hash of ``task_id``, so the wording is stable for a task
and different across tasks without any randomness in the graph.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from jarvis.agent.state import StepResult

__all__ = [
    "MAX_SENTENCES",
    "MAX_SPOKEN_CHARS",
    "OUTCOMES",
    "SUCCESS",
    "address_phrase",
    "classify_outcome",
    "first_sentences",
    "spoken_answer",
]

#: Closed outcome vocabulary (the compose equivalents of ``task_status``).
OUTCOMES = (
    "success",
    "partial",
    "failed",
    "refused",
    "cancelled",
    "timeout",
    "policy_blocked",
    "clarify",
    "unsupported",
    "llm_down",
    "conversation",
)

SUCCESS = "success"

#: A spoken answer is at most this many sentences / characters.
MAX_SENTENCES = 2
MAX_SPOKEN_CHARS = 220

#: Longest stored name used for the butler address.
MAX_NAME_CHARS = 40

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

#: Outcomes whose wording is a courtesy and may carry the address.
_ADDRESSABLE = frozenset({"success", "partial", "cancelled", "clarify"})

_TEMPLATES: dict[str, tuple[str, ...]] = {
    "success": (
        "Done.",
        "That's done.",
        "It's done.",
        "All set.",
    ),
    "partial": (
        "That's done, but I could not verify it{address}.",
        "I finished that, though nothing confirmed it worked{address}.",
        "It's finished - I could not verify the result{address}.",
    ),
    "cancelled": (
        "Cancelled.",
        "Stopped{address}.",
        "As you wish, cancelled{address}.",
    ),
}

#: Halt messages that mean "the AI backend", not "the policy layer".
_LLM_MARKERS = (
    "couldn't reach the ai",
    "ai backend is not configured",
    "rate limit",
    "couldn't produce an answer",
    "valid plan",
    "couldn't safely understand",
)

#: Halt messages that mean "a deterministic rule said no".
_POLICY_MARKERS = (
    "i did not perform the action",
    "confirmation timed out",
    "not allowed",
    "blocked",
    "refused",
    "denied",
    "no approval",
    "not permitted",
    "cannot proceed safely",
    "requires",
)


def classify_outcome(state: dict[str, Any]) -> str:
    """One of :data:`OUTCOMES` for this terminal state (deterministic)."""
    if state.get("cancelled"):
        return "cancelled"

    plan = state.get("plan")
    if plan is not None and getattr(plan, "needs_clarification", False):
        return "clarify"

    if str(state.get("request_kind") or "") == "unsupported":
        return "unsupported"

    halt = str(state.get("halted_reason") or "").strip()
    if halt:
        return _halt_outcome(halt)

    if state.get("final_answer") or (
        plan is not None
        and getattr(plan, "kind", None) == "conversation"
        and not getattr(plan, "steps", None)
    ):
        return "conversation"

    results = _results(state)
    if any(not r.ok or r.verified is False for r in results):
        return "failed"
    if any(r.verified is None for r in results):
        return "partial"
    if results:
        return SUCCESS
    # Nothing ran and nothing refused: ``respond`` says "No task performed."
    return "failed"


def _halt_outcome(halt: str) -> str:
    """Which kind of stop a halt message is (message markers only)."""
    text = halt.casefold()
    if "i did not perform the action" in text or "confirmation timed out" in text:
        return "policy_blocked"
    if "timed out" in text or "timeout" in text or "in time" in text:
        return "timeout"
    if any(marker in text for marker in _LLM_MARKERS):
        return "llm_down"
    if any(marker in text for marker in _POLICY_MARKERS):
        return "policy_blocked"
    return "refused"


def first_sentences(text: str, *, limit: int = MAX_SENTENCES) -> str:
    """``text`` cut to at most ``limit`` sentences and :data:`MAX_SPOKEN_CHARS`."""
    flat = " ".join(str(text or "").split())
    if not flat:
        return ""
    sentences = [part for part in _SENTENCE_SPLIT.split(flat) if part.strip()]
    cut = " ".join(sentences[: max(limit, 1)]).strip()
    if len(cut) > MAX_SPOKEN_CHARS:
        cut = cut[: MAX_SPOKEN_CHARS - 1].rstrip() + "…"
    return cut


def address_phrase(user_name: str) -> str:
    """``", <name>"`` for the courtesies, or ``""`` when no name is stored.

    The address is opt-in: it needs a stored ``user_name`` preference, so an
    install without one keeps the plain wording (no invented "sir", no
    guessed name).  The caller has already read and validated the preference;
    this only formats it.
    """
    name = " ".join(str(user_name or "").split())[:MAX_NAME_CHARS].strip()
    return f", {name}" if name else ""


def spoken_answer(state: dict[str, Any], outcome: str, *, address: str = "") -> str:
    """The short, honest spoken form of this outcome (never a new claim).

    Failures and refusals are the factual text the screen shows, shortened -
    never a cheerful template.  Success/partial/cancelled/clarify use a varied
    courtesy chosen reproducibly from ``task_id``, with the address at most
    once.  A conversation is the brain's own answer, untouched.
    """
    if outcome == "conversation":
        return first_sentences(_conversation_text(state))

    if outcome == "clarify":
        question = ""
        plan = state.get("plan")
        if plan is not None:
            question = str(getattr(plan, "clarification_question", "") or "")
        question = first_sentences(question, limit=1)
        if question:
            return first_sentences(f"I need one more detail{address}: {question}")
        return first_sentences(f"I need one more detail{address}.")

    templates = _TEMPLATES.get(outcome)
    if templates is not None and outcome in _ADDRESSABLE:
        chosen = _pick(templates, str(state.get("task_id") or ""))
        return first_sentences(chosen.replace("{address}", address))

    # failed / refused / policy_blocked / timeout / llm_down / unsupported:
    # the facts, verbatim, shortened.  ``respond`` builds the same text.
    from jarvis.agent.nodes.respond import base_answer

    return first_sentences(base_answer(state))


def _conversation_text(state: dict[str, Any]) -> str:
    """The brain's own conversational answer (never reworded)."""
    plan = state.get("plan")
    dialog = getattr(plan, "dialog_answer", None) if plan is not None else None
    if isinstance(dialog, str) and dialog.strip():
        return dialog.strip()
    return str(state.get("final_answer") or "")


def _results(state: dict[str, Any]) -> list[StepResult]:
    return [r for r in (state.get("results") or []) if isinstance(r, StepResult)]


def _pick(options: tuple[str, ...], task_id: str) -> str:
    """Reproducible choice: a hash of ``task_id``, not ``hash()``/randomness."""
    if not options:
        return ""
    digest = hashlib.sha256(task_id.encode("utf-8")).digest()
    return options[digest[0] % len(options)]
