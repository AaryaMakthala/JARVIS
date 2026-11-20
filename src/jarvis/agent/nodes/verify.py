"""verify node: applies each tool's deterministic ``verify()`` to the result.

Never re-runs a side-effecting tool.  Successful-but-non-verifiable steps
(verify() returns None) advance but are reported honestly later.

P5: a step may declare a closed ``evidence`` contract (see
:mod:`jarvis.agent.evidence`).  When it does, the tool's own ``ok`` is not
enough: the observed result must supply the declared evidence.  Contradicted
evidence skips a blind identical retry and goes straight to replan/fail-closed;
inconclusive evidence stays ``verified=None`` with a constant note.

``settings.agent.verify_enabled`` / ``replan_enabled`` turn this node into a
pure bookkeeping step so the benchmark can measure what the two abilities are
worth (docs/07 section 3.4).  An ablated node still never advances past a step
that failed - the switch removes the recovery machinery, not fail-closed
behaviour.
"""

from __future__ import annotations

from typing import Any

from jarvis.agent.context import AppContext
from jarvis.agent.evidence import evaluate_evidence, normalise_kind
from jarvis.agent.state import EvidenceStatus, Plan
from jarvis.tools.base import ToolResult

# 4.17c: focus failures never improve on an identical second attempt (same
# target, same window): skip retry_now and go straight to replan/fail-closed.
_NON_RETRYABLE = ("failed to focus", "focus verification failed")
_UNCHECKED_EXPECTATION = "Planned success condition was not independently checked."
_FAILED_INDEPENDENT_CHECK = "An independent check of the expected outcome failed."
_VERDICT_TO_VERIFIED: dict[str, bool | None] = {
    "confirmed": True,
    "contradicted": False,
    "inconclusive": None,
}


def verify(state: dict[str, Any], ctx: AppContext) -> dict[str, Any]:
    """Verify the last result and advance the step bookkeeping."""
    results = list(state.get("results") or [])
    if not results:
        return {"halted_reason": "Nothing to verify."}

    last = results[-1]
    verify_on = ctx.settings.agent.verify_enabled
    replan_on = ctx.settings.agent.replan_enabled

    if not verify_on:
        return _advance_ablated(state, results, last, ctx)

    plan: Plan | None = state.get("plan")
    step = None
    if plan and plan.steps:
        idx = int(state.get("step_index") or 0)
        if idx < len(plan.steps):
            step = plan.steps[idx]

    verified = last.verified
    note = last.verify_note
    evidence_kind = ""
    evidence_status: EvidenceStatus | None = None
    if step is not None and last.step_id == step.id and last.ok:
        spec = ctx.registry.get_optional(step.tool)
        if spec is not None:
            try:
                args = spec.args_model.model_validate(step.args)
            except Exception:  # noqa: BLE001 - already validated at execution
                args = None
            if args is not None:
                synthetic = ToolResult(
                    ok=last.ok,
                    output=last.output,
                    data=last.data,
                    error=last.error,
                    tainted=last.tainted,
                    verified=last.verified,
                    verify_note=last.verify_note,
                )
                outcome = spec.apply_verify(args, synthetic, ctx.tool_context())
                verified = outcome.verified
                note = outcome.verify_note

        declared = normalise_kind(getattr(step, "evidence", None))
        if declared:
            observed = last.model_copy(update={"verified": verified, "verify_note": note})
            verdict = evaluate_evidence(declared, observed)
            evidence_kind = declared
            if verified is False and verdict.status == "confirmed":
                evidence_status = "contradicted"
                note = _FAILED_INDEPENDENT_CHECK
            else:
                evidence_status = verdict.status
                verified = _VERDICT_TO_VERIFIED[verdict.status]
                note = verdict.note
        elif (
            step.expect.strip() and verified is None and _UNCHECKED_EXPECTATION not in (note or "")
        ):
            note = f"{note} {_UNCHECKED_EXPECTATION}".strip()

    bumped = last.model_copy(
        update={
            "verified": verified,
            "verify_note": note,
            "evidence": evidence_kind or None,
            "evidence_status": evidence_status,
        }
    )
    results[-1] = bumped

    max_retries = ctx.settings.agent.max_retries_per_step
    retry_count = int(state.get("retry_count") or 0)
    if not last.ok or verified is False:
        non_retryable = evidence_status == "contradicted" or (
            (not last.ok)
            and any((last.error or "").lower().startswith(prefix) for prefix in _NON_RETRYABLE)
        )
        if retry_count < max_retries and not non_retryable:
            return {"results": results, "retry_count": retry_count + 1, "retry_now": True}
        # Retries exhausted: try a fresh plan while the replan budget allows it,
        # otherwise fail closed (never move to the next step on a failure).
        if replan_on and int(state.get("replan_count") or 0) < ctx.settings.agent.max_replans:
            return {
                "results": results,
                "retry_count": max_retries,
                "retry_now": False,
                "replan_now": True,
            }
        return {
            "results": results,
            "retry_count": max_retries,
            "retry_now": False,
            "replan_now": False,
        }

    return {
        "results": results,
        "step_index": int(state.get("step_index") or 0) + 1,
        "retry_count": 0,
        "retry_now": False,
        "replan_now": False,
    }


def _advance_ablated(
    state: dict[str, Any], results: list[Any], last: Any, ctx: AppContext
) -> dict[str, Any]:
    """Advance without independent verification, retry or replanning.

    A step the tool itself reported as failed still halts the run: the ablation
    answers "what if JARVIS could not check its own work", not "what if JARVIS
    pretended a failure did not happen".
    """
    if not last.ok:
        return {
            "results": results,
            "step_index": int(state.get("step_index") or 0),
            "retry_count": int(state.get("retry_count") or 0),
            "retry_now": False,
            "replan_now": False,
        }
    return {
        "results": results,
        "step_index": int(state.get("step_index") or 0) + 1,
        "retry_count": 0,
        "retry_now": False,
        "replan_now": False,
    }
