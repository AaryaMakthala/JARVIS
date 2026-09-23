"""policy_gate node: the only place confirmations happen.

Computes the policy :class:`Decision` for the *current* step.  If the step is
Tier 0 it passes straight through; if it needs confirmation it suspends the
graph with ``interrupt(payload)`` (no side effects before the interrupt).  On
resume the node re-runs deterministically, so the decision and its action_hash
are recomputed and the supplied answer is re-checked, preventing a stale or
tampered approval from being trusted.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from langgraph.types import interrupt

from jarvis.agent.context import AppContext
from jarvis.agent.schemas import ConfirmationRequest
from jarvis.agent.state import Decision, StepResult


def _collect_tainted_fragments(state: dict[str, Any]) -> tuple[str, ...]:
    """Extract tainted output text from previous step results.

    Only fragments from ``StepResult`` objects where ``tainted=True`` are
    included.  This data is trusted (produced by our own tools, not by the
    LLM) and used by the engine for deterministic taint detection.
    """
    fragments: list[str] = []
    for r in state.get("results") or []:
        if isinstance(r, StepResult) and r.tainted and r.output:
            fragments.append(r.output)
    return tuple(fragments)


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

    # Build a PolicyContext enriched with tainted fragments from previous
    # results so the engine can independently verify taint, regardless of
    # what the LLM asserted in ``step.depends_on_untrusted`` (docs/03 §8).
    tainted = _collect_tainted_fragments(state)
    if tainted and ctx.policy_ctx is not None:
        pctx = dataclasses.replace(ctx.policy_ctx, tainted_fragments=tainted)
    else:
        pctx = ctx.policy_ctx

    decision = ctx.engine.decide(step, pctx)
    decisions = dict(state.get("decisions") or {})
    decisions[step.id] = decision

    if not decision.allowed:
        return {"decisions": decisions, "halted_reason": refusal_text(decision, step.id)}

    if not decision.needs_confirm:
        return {"decisions": decisions}

    # Use the engine's deterministic ``warn_untrusted`` flag (which considers
    # both the LLM's ``depends_on_untrusted`` and the engine's own overlap
    # check) rather than the LLM-controlled ``step.depends_on_untrusted``.
    payload = confirmation_payload(decision, decision.warn_untrusted)
    answer = interrupt(payload)  # graph pauses; checkpoint saved; no side effects above

    # TOCTOU guard: compare the stashed pre-interrupt paths against the fresh
    # paths the engine resolved on this resume pass.  The stashed paths come
    # from validate's pre-computation (committed to checkpoint before this
    # node ran) or from a previous resume's return dict.  The comparison uses
    # only engine-produced data; the answer is never consulted.
    gated = dict(state.get("gated_resolved_paths") or {})
    prior_paths = gated.get(step.id)
    # Always update to the current resolution so subsequent resumes
    # (e.g. after a second interrupt) have the latest checkpointed snapshot.
    gated[step.id] = list(decision.resolved_paths)

    if prior_paths is not None and prior_paths != decision.resolved_paths:
        return {
            "decisions": decisions,
            "gated_resolved_paths": gated,
            "halted_reason": "Refused: a file path changed after you confirmed — not acting on it.",
        }

    approved = _answer_matches(answer, decision.action_hash)
    if not approved:
        return {
            "decisions": decisions,
            "gated_resolved_paths": gated,
            "halted_reason": "Refused by you (confirmation answered with 'no' or a mismatched action).",
        }
    if not _typed_confirmation_ok(answer, decision):
        return {
            "decisions": decisions,
            "gated_resolved_paths": gated,
            "halted_reason": "Refused: the typed folder-name confirmation did not match the plan.",
        }
    if decision.needs_unlock and (ctx.unlock is None or not ctx.unlock.is_unlocked()):
        return {
            "decisions": decisions,
            "gated_resolved_paths": gated,
            "halted_reason": "Refused: Tier 2 action needs an unlocked JARVIS session.",
        }

    approved_hashes = list(state.get("approved_hashes") or []) + [decision.action_hash]
    return {
        "decisions": decisions,
        "gated_resolved_paths": gated,
        "approved_hashes": approved_hashes,
    }


def confirmation_payload(decision: Decision, untrusted: bool) -> dict[str, Any]:
    """The interrupt payload the daemon/CLI forwards to the user."""
    payload = ConfirmationRequest(
        step_id=decision.step_id,
        tier=decision.tier,
        summary=decision.summary,
        needs_unlock=decision.needs_unlock,
        typed_confirmation=decision.needs_typed_confirmation,
        resolved_paths=list(decision.resolved_paths),
        action_hash=decision.action_hash,
        untrusted=bool(untrusted),
    )
    return payload.model_dump(mode="json")


def _answer_matches(answer: Any, expected_hash: str) -> bool:
    """A resume value only counts as approval when it carries the exact hash."""
    if not isinstance(answer, dict):
        return False
    if answer.get("approved") is not True:
        return False
    provided = answer.get("action_hash")
    return isinstance(provided, str) and provided == expected_hash


def _typed_confirmation_ok(answer: Any, decision: Decision) -> bool:
    """When the step needs a typed folder-name confirmation the resume value
    must carry exactly that text (bound to the resolved folder at decision
    time).  Fails closed: a missing or wrong answer is a refusal."""
    if not decision.needs_typed_confirmation:
        return True
    if not isinstance(answer, dict):
        return False
    provided = answer.get("typed_confirmation")
    return isinstance(provided, str) and provided == decision.needs_typed_confirmation


def refusal_text(decision: Decision, step_id: str) -> str:
    """Honest, deterministic refusal message for a blocked step."""
    reason = "; ".join(decision.reasons) if decision.reasons else "blocked by policy"
    return f"Refused: {decision.summary} [step {step_id}] ({reason})"
