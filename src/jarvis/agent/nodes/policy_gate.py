"""policy_gate node: the only place confirmations happen.

Computes the policy :class:`Decision` for the *current* step.  If the step is
Tier 0 it passes straight through; if it needs confirmation it suspends the
graph with ``interrupt(payload)`` (no side effects before the interrupt).  On
resume the node re-runs deterministically, so the decision and its action_hash
are recomputed and the supplied answer is re-checked, preventing a stale or
tampered approval from being trusted.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.types import interrupt

from jarvis.agent.batch_approval import (
    TYPE_PLAN_APPROVAL,
    compute_all_decisions,
    decide_and_build_payload,
    recompute_and_resolve_batch,
)
from jarvis.agent.context import AppContext
from jarvis.agent.nodes.util import policy_context_for
from jarvis.agent.schemas import ConfirmationRequest
from jarvis.agent.state import Decision
from jarvis.voice.tier2_voice import allowed as tier2_voice_allowed
from jarvis.voice.tier2_voice import voice_marker_present

__all__ = ["confirmation_payload", "policy_gate", "refusal_text"]

logger = logging.getLogger(__name__)


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

    # Enrich the PolicyContext with tainted fragments from previous results and
    # the user's own words, so the engine can verify taint independently of what
    # the LLM asserted in ``step.depends_on_untrusted`` (docs/03 §8) and the risk
    # classifier sees the original request (docs/08 §3).
    pctx = policy_context_for(state, ctx.policy_ctx)

    decision = ctx.engine.decide(step, pctx)
    decisions = dict(state.get("decisions") or {})
    decisions[step.id] = decision

    if not decision.allowed:
        return {"decisions": decisions, "halted_reason": refusal_text(decision, step.id)}

    if not decision.needs_confirm:
        return {"decisions": decisions}

    already_approved = [str(h) for h in (state.get("approved_hashes") or [])]

    # A step whose exact action_hash the user already approved (typically by an
    # earlier plan-level batch covering this step and the next ones) must never
    # be confirmed a second time: walk straight to act, which re-decides, re-checks
    # the approval binding and the TOCTOU paths itself.  Fail closed: a Tier 2
    # hash can only reach here from an approval the gate itself issued while the
    # session was unlocked, so the same unlock check still applies.
    if str(decision.action_hash) in already_approved:
        if decision.needs_unlock and (ctx.unlock is None or not ctx.unlock.is_unlocked()):
            return {
                "decisions": decisions,
                "halted_reason": "Refused: Tier 2 action needs an unlocked JARVIS session.",
            }
        return {"decisions": decisions}

    # Build the interrupt payload.  Batch eligibility needs a decision for every
    # remaining step, so the whole plan is decided here with the SAME engine and
    # PolicyContext the current step was just decided with (read-only, exactly as
    # `validate` already does for its TOCTOU snapshot).  When 2+ contiguous
    # Tier 1 steps are eligible this is a single ``plan_approval`` payload that
    # names them all; otherwise it is the existing per-step ``confirm`` payload.
    # No side effects before the interrupt: the payload is built from decisions
    # the engine already produced.
    decisions = {**compute_all_decisions(plan, pctx, ctx.engine), step.id: decision}
    payload = decide_and_build_payload(
        plan,
        decisions,
        step_index=idx,
        approved_hashes=already_approved,
    )
    answer = interrupt(payload)  # graph pauses; checkpoint saved; no side effects above

    if _is_plan_approval_payload(payload):
        updates, approved_hashes = recompute_and_resolve_batch(plan, decisions, state, answer)
        if "halted_reason" in updates:
            gated = dict(state.get("gated_resolved_paths") or {})
            if "gated_resolved_paths" not in updates:
                updates["gated_resolved_paths"] = gated
            return updates
        gated = updates.get("gated_resolved_paths") or dict(state.get("gated_resolved_paths") or {})
        return {
            "decisions": decisions,
            "gated_resolved_paths": gated,
            "approved_hashes": updates.get(
                "approved_hashes", list(state.get("approved_hashes") or [])
            ),
        }

    # Single-step path (unchanged semantics; payload shape may differ in fields
    # the daemon already tolerates, but the binding is still the single
    # decision.action_hash).
    gated = dict(state.get("gated_resolved_paths") or {})
    prior_paths = gated.get(step.id)
    gated[step.id] = list(decision.resolved_paths)

    if prior_paths is not None and prior_paths != decision.resolved_paths:
        return {
            "decisions": decisions,
            "gated_resolved_paths": gated,
            "halted_reason": "Refused: a file path changed after you confirmed — not acting on it.",
        }

    approved = _answer_matches(answer, decision.action_hash)
    if not approved:
        if isinstance(answer, dict) and answer.get("timed_out") is True:
            return {
                "decisions": decisions,
                "gated_resolved_paths": gated,
                "halted_reason": "Confirmation timed out. I did not perform the action.",
            }
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
    # Stage 3 opt-in: a voice-approved Tier 2 step may satisfy *this* step's
    # hash without the terminal password — but only when the daemon marked the
    # answer AND the gate re-verifies every condition itself.  This never sets
    # ``ctx.unlock`` and never unlocks the session.
    locked_for_tier2 = decision.needs_unlock and (
        ctx.unlock is None or not ctx.unlock.is_unlocked()
    )
    if locked_for_tier2 and not _voice_tier2_ok(state, answer, step, decision, ctx):
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


def _voice_tier2_ok(
    state: dict[str, Any],
    answer: Any,
    step: Any,
    decision: Decision,
    ctx: AppContext,
) -> bool:
    """May a locked-session Tier 2 step proceed because *voice* approved it?

    Fails closed and unlocks nothing.  Three independent conditions: the task
    came from voice (``state["source"]``, which ``intake`` normalises), the
    daemon set its marker on the answer, and
    :func:`jarvis.voice.tier2_voice.allowed` re-derives the verdict from the
    gate's **own** decision and the step's validated args — so the marker alone
    approves nothing.  ``ctx.unlock`` is read but never written.
    """
    if state.get("source") != "voice":
        return False
    if not voice_marker_present(answer):
        return False
    ok, reason = tier2_voice_allowed(
        str(getattr(step, "tool", "") or ""),
        decision,
        ctx.settings,
        args=getattr(step, "args", None),
    )
    if not ok:
        logger.info("voice Tier 2 approval rejected at the gate: %s", reason)
    return ok


def confirmation_payload(decision: Decision, untrusted: bool) -> dict[str, Any]:
    """The interrupt payload the daemon/CLI forwards to the user.

    Retained for tests and for callers that build a single-step payload
    directly.  The gate node itself now uses ``decide_and_build_payload``
    (batch-aware) so a two-step Tier 1 plan raises one interrupt.
    """
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


def _is_plan_approval_payload(payload: Any) -> bool:
    """Whether *payload* is a plan-level approval interrupt."""
    return isinstance(payload, dict) and payload.get("type") == TYPE_PLAN_APPROVAL


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
