"""verify node: applies each tool's deterministic ``verify()`` to the result.

Never re-runs a side-effecting tool.  Successful-but-non-verifiable steps
(verify() returns None) advance but are reported honestly later.
"""

from __future__ import annotations

from typing import Any

from jarvis.agent.context import AppContext
from jarvis.agent.state import Plan
from jarvis.tools.base import ToolResult


def verify(state: dict[str, Any], ctx: AppContext) -> dict[str, Any]:
    """Verify the last result and advance the step bookkeeping."""
    results = list(state.get("results") or [])
    if not results:
        return {"halted_reason": "Nothing to verify."}

    last = results[-1]
    plan: Plan | None = state.get("plan")
    step = None
    if plan and plan.steps:
        idx = int(state.get("step_index") or 0)
        if idx < len(plan.steps):
            step = plan.steps[idx]

    verified = last.verified
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
                )
                outcome = spec.apply_verify(args, synthetic, ctx.tool_context())
                verified = outcome.verified

    bumped = last.model_copy(update={"verified": verified})
    results[-1] = bumped

    max_retries = ctx.settings.agent.max_retries_per_step
    retry_count = int(state.get("retry_count") or 0)
    if not last.ok or verified is False:
        if retry_count < max_retries:
            return {"results": results, "retry_count": retry_count + 1, "retry_now": True}
        # Retries exhausted: try a fresh plan while the replan budget allows it,
        # otherwise fail closed (never move to the next step on a failure).
        if int(state.get("replan_count") or 0) < ctx.settings.agent.max_replans:
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
