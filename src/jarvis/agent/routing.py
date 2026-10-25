"""The brain -> next-node route contract, as one deterministic table.

``brain`` is the only LLM classification step, so everything the graph does
after it is decided by *code* reading the projected state (never by the model's
own opinion of where to go).  This module is that decision, in one place, so the
table below and the graph's conditional edges cannot drift apart.

Route table (``state`` is what ``brain`` just projected)::

    request_type   | required fields (validated on BrainDecision)     | next node
    ---------------+--------------------------------------------------+-------------------
    conversation   | response_text non-blank, no actions               | respond
    action         | at least one action                              | validate
    clarification  | clarification_question non-blank, no actions      | clarify
    browse         | goal non-blank, no actions                       | specialist_decide
                   | (respond when the specialist flag is off)         |
    unsupported    | no actions; answer = the brain's dialog answer,   | respond
                   | else "I can't do that with the tools I have."     |

    precedence before the table (checked in this order, first match wins):
      * ``halted_reason`` set                          -> respond
      * ``request_kind == "unsupported"``              -> respond (no tools)
      * specialist waiting to decide (flag on)         -> specialist_decide
      * no plan                                        -> respond
      * ``plan.needs_clarification``                   -> clarify
      * ``plan.kind == "conversation"``                -> respond

``unsupported`` is listed explicitly even though a halted task already routes to
``respond``: a request the tools cannot serve must never reach ``validate``, so
the rule is stated here rather than left to the ``plan is None`` fallback.
Nothing in this module reads a tier, a policy decision or a tool result.
"""

from __future__ import annotations

from typing import Any

__all__ = ["UNSUPPORTED_ANSWER", "route_for_brain_result", "unsupported_answer"]

#: What the user hears when the brain says "I can't" without wording a reason.
UNSUPPORTED_ANSWER = "I can't do that with the tools I have."


def route_for_brain_result(state: dict[str, Any]) -> str:
    """Return the next node name for the state ``brain`` just produced.

    Pure: reads only the projected state, writes nothing, calls no LLM.  The
    returned names are exactly the keys ``graph.py`` maps in
    ``add_conditional_edges`` for the ``brain`` node.
    """
    if state.get("halted_reason"):
        return "respond"

    # An unsupported request never carries tool steps, whatever else is in
    # state; it is answered (or refused) by ``respond`` and goes no further.
    if state.get("request_kind") == "unsupported":
        return "respond"

    if state.get("specialist_phase") == "decide" and (
        state.get("specialist_goal") or state.get("request_kind") == "browse"
    ):
        return "specialist_decide"

    plan = state.get("plan")
    if plan is None:
        return "respond"
    if getattr(plan, "needs_clarification", False):
        return "clarify"
    if getattr(plan, "kind", None) == "conversation":
        return "respond"
    return "validate"


def unsupported_answer(state: dict[str, Any]) -> str:
    """The answer shown for an unsupported request (deterministic, never blank).

    Order: the brain's own dialog answer when a conversation plan carries one,
    then the halt text the brain already projected, then the fixed default.
    """
    plan = state.get("plan")
    dialog = getattr(plan, "dialog_answer", None) if plan is not None else None
    if isinstance(dialog, str) and dialog.strip():
        return dialog.strip()
    halt = state.get("halted_reason")
    if isinstance(halt, str) and halt.strip():
        return halt.strip()
    return UNSUPPORTED_ANSWER
