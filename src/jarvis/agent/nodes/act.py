"""act node: execute the current step through the registry.

Re-runs the deterministic policy check and re-validates args so that a plan
mutated between confirmation and execution can never run.  Only Tier-0 steps
or steps whose exact action_hash was previously approved may execute.  All
side effects happen here and only here.
"""

from __future__ import annotations

import time
from typing import Any

from jarvis.agent.context import AppContext
from jarvis.agent.nodes.util import truncate
from jarvis.agent.state import StepResult


def act(state: dict[str, Any], ctx: AppContext) -> dict[str, Any]:
    """Execute the current step; never trust state alone — re-decide."""
    plan = state.get("plan")
    idx = int(state.get("step_index") or 0)
    if plan is None or not plan.steps or idx >= len(plan.steps):
        return {"halted_reason": "No step to execute."}

    step = plan.steps[idx]
    spec = ctx.registry.get_optional(step.tool)
    if spec is None:
        return append_result(
            state,
            StepResult(step_id=step.id, ok=False, error="unknown tool", verified=False),
            halted="Refused: unknown tool.",
        )

    decision = ctx.engine.decide(step, ctx.policy_ctx)
    if not decision.allowed or decision.tier >= 3:
        return append_result(
            state,
            StepResult(step_id=step.id, ok=False, error="blocked by policy", verified=False),
            halted=f"Refused: blocked by policy ({step.tool}).",
        )

    if decision.needs_confirm and decision.action_hash not in (state.get("approved_hashes") or []):
        return append_result(
            state,
            StepResult(
                step_id=step.id, ok=False, error="approval binding mismatch", verified=False
            ),
            halted="Refused: the approved action no longer matches — the plan was likely altered after confirmation.",
        )

    try:
        args = spec.args_model.model_validate(step.args)
    except Exception as exc:  # noqa: BLE001 - defensive; validate() already checked
        return append_result(
            state,
            StepResult(step_id=step.id, ok=False, error=f"invalid args: {exc}", verified=False),
            halted="Refused: step arguments are invalid.",
        )

    tool_ctx = ctx.tool_context()
    start = time.perf_counter()
    result = spec.execute(args, tool_ctx)
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    step_result = StepResult(
        step_id=step.id,
        ok=result.ok,
        output=truncate(result.output or ""),
        data=result.data,
        error=result.error,
        tainted=result.tainted,
        verified=result.verified,
        duration_ms=elapsed_ms,
    )
    return append_result(state, step_result)


def append_result(
    state: dict[str, Any], step_result: StepResult, halted: str | None = None
) -> dict[str, Any]:
    """Append to results; optionally halt the task."""
    results = list(state.get("results") or []) + [step_result]
    update: dict[str, Any] = {"results": results}
    if halted:
        update["halted_reason"] = halted
    return update
