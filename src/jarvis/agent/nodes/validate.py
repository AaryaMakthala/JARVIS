"""validate node: deterministic plan checks before anything executes.

Rejects unknown tools, invalid arguments, and plans over the step cap.  A
rejected plan goes back to ``plan`` once (one repair retry); a second
rejection halts with an honest explanation.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from jarvis.agent.context import AppContext


def validate(state: dict[str, Any], ctx: AppContext) -> dict[str, Any]:
    """Check the plan; return updates that route to plan/policy_gate/respond."""
    plan = state.get("plan")
    if plan is None:
        return _reject(state, "The planner produced no plan.")

    if plan.needs_clarification:
        return {"plan": plan, "error": None, "validate_attempts": 0}

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

    return {
        "plan": plan,
        "error": None,
        "validate_attempts": 0,
        "step_index": 0,
        "retry_count": 0,
        "retry_now": False,
        "decisions": {},
        "approved_hashes": [],
        "results": [],
        "halted_reason": None,
    }


def _reject(state: dict[str, Any], reason: str) -> dict[str, Any]:
    """Return rejection updates; second rejection halts the task."""
    attempts = int(state.get("validate_attempts") or 0) + 1
    if attempts >= 2:
        return {
            "validate_attempts": attempts,
            "error": reason,
            "halted_reason": f"Could not produce a valid plan (after 2 attempts): {reason}",
        }
    return {"validate_attempts": attempts, "error": reason, "halted_reason": None}
