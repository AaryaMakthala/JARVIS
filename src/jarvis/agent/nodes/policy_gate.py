"""policy_gate node: the only place confirmations happen.

Computes the policy :class:`Decision` for the *current* step.  If the step is
Tier 0 it passes straight through; if it needs confirmation it suspends the
graph with ``interrupt(payload)`` (no side effects before the interrupt).  On
resume the node re-runs deterministically, so the decision and its action_hash
are recomputed and the supplied answer is re-checked, preventing a stale or
tampered approval from being trusted.
"""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from jarvis.agent.context import AppContext
from jarvis.agent.state import Decision


def policy_gate(state: dict[str, Any], ctx: AppContext) -> dict[str, Any]:
    """Decide the current step; interrupt when the user must confirm."""
    plan = state.get("plan")
    if plan is None:
        return {"halted_reason": "No plan to gate."}
    if plan.needs_clarification:
        return {"halted_reason": "Clarification needed before I can act."}
    idx = int(state.get("step_index") or 0)
    if idx >= len(plan.steps):
        return {"halted_reason": "No step remaining at the gate."}

    step = plan.steps[idx]
    decision = ctx.engine.decide(step, ctx.policy_ctx)
    decisions = dict(state.get("decisions") or {})
    decisions[step.id] = decision

    if not decision.allowed:
        return {"decisions": decisions, "halted_reason": refusal_text(decision, step.id)}

    if not decision.needs_confirm:
        return {"decisions": decisions}

    payload = confirmation_payload(decision, step.depends_on_untrusted)
    answer = interrupt(payload)  # graph pauses; checkpoint saved; no side effects above

    approved = _answer_matches(answer, decision.action_hash)
    if approved and decision.needs_unlock and (ctx.unlock is None or not ctx.unlock.is_unlocked()):
        return {
            "decisions": decisions,
            "halted_reason": "Refused: Tier 2 action needs an unlocked JARVIS session.",
        }
    if not approved:
        return {
            "decisions": decisions,
            "halted_reason": "Refused by you (confirmation answered with 'no' or a mismatched action).",
        }

    approved_hashes = list(state.get("approved_hashes") or []) + [decision.action_hash]
    return {"decisions": decisions, "approved_hashes": approved_hashes}


def confirmation_payload(decision: Decision, untrusted: bool) -> dict[str, Any]:
    """The interrupt payload the daemon/CLI forwards to the user."""
    return {
        "type": "confirm",
        "step_id": decision.step_id,
        "tier": decision.tier,
        "summary": decision.summary,
        "needs_unlock": decision.needs_unlock,
        "typed_confirmation": decision.needs_typed_confirmation,
        "action_hash": decision.action_hash,
        "untrusted": bool(untrusted),
    }


def _answer_matches(answer: Any, expected_hash: str) -> bool:
    """A resume value only counts as approval when it carries the exact hash."""
    if not isinstance(answer, dict):
        return False
    if answer.get("approved") is not True:
        return False
    provided = answer.get("action_hash")
    return isinstance(provided, str) and provided == expected_hash


def refusal_text(decision: Decision, step_id: str) -> str:
    """Honest, deterministic refusal message for a blocked step."""
    reason = "; ".join(decision.reasons) if decision.reasons else "blocked by policy"
    return f"Refused: {decision.summary} [step {step_id}] ({reason})"
