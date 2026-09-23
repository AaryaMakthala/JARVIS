"""validate node: deterministic plan checks before anything executes.

Rejects unknown tools, invalid arguments, and plans over the step cap.  A
rejected *initial* plan goes back to ``plan`` once (one repair retry); a
rejected *replacement* plan goes back to ``replan`` once.  A second rejection
of either halts with an honest explanation.

When it validates a plan produced by :mod:`replan` (``pending_replan`` set),
the already-completed results and approved hashes are preserved and the
TOCTOU path snapshot is recomputed for the new steps.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from jarvis.agent.context import AppContext


def validate(state: dict[str, Any], ctx: AppContext) -> dict[str, Any]:
    """Check the plan; return updates that route to plan/replan/policy_gate/respond."""
    plan = state.get("plan")
    if plan is None:
        return _reject(state, "The planner produced no plan.")

    if plan.needs_clarification:
        return {
            "plan": plan,
            "error": None,
            "validate_attempts": 0,
            "pending_replan": False,
        }

    errors: list[str] = []
    for step in plan.steps:
        spec = ctx.registry.get_optional(step.tool)
        if spec is None:
            errors.append(f"step {step.id}: tool {step.tool!r} is not available")
            continue
        try:
            spec.args_model.model_validate(step.args)
        except ValidationError as exc:
            errors.append(
                f"step {step.id}: invalid args for {step.tool}: {exc.error_count()} error(s)"
            )

    if len(plan.steps) > ctx.settings.agent.max_steps:
        errors.append(f"plan has {len(plan.steps)} steps (max {ctx.settings.agent.max_steps})")

    if errors:
        return _reject(state, "; ".join(errors))

    # A replanned plan has new steps (possibly reusing ids) whose paths must be
    # re-snapshotted; the original plan's snapshot is kept otherwise (resume).
    replaying = bool(state.get("pending_replan"))
    gated = {} if replaying else state.get("gated_resolved_paths")

    if gated is None:
        # Pre-compute and stash the resolved paths for each step so that
        # policy_gate can detect TOCTOU changes after interrupt/resume.
        gated = {}
        for step in plan.steps:
            decision = ctx.engine.decide(step, ctx.policy_ctx)
            gated[step.id] = list(decision.resolved_paths)

    if replaying:
        results = list(state.get("results") or [])
        approved_hashes = list(state.get("approved_hashes") or [])
    else:
        results = []
        approved_hashes = []

    return {
        "plan": plan,
        "error": None,
        "validate_attempts": 0,
        "step_index": 0,
        "retry_count": 0,
        "retry_now": False,
        "replan_now": False,
        "decisions": {},
        "approved_hashes": approved_hashes,
        "results": results,
        "gated_resolved_paths": gated,
        "halted_reason": None,
        "pending_replan": False,
    }


def _reject(state: dict[str, Any], reason: str) -> dict[str, Any]:
    """Return rejection updates; a second rejection halts the task.

    The next planning stage depends on whether the rejected plan was a
    replacement (``pending_replan`` set) or the initial one.
    """
    attempts = int(state.get("validate_attempts") or 0) + 1
    if attempts >= 2:
        return {
            "validate_attempts": attempts,
            "error": reason,
            "halted_reason": f"Could not produce a valid plan (after 2 attempts): {reason}",
        }
    return {"validate_attempts": attempts, "error": reason, "halted_reason": None}
