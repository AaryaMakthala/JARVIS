"""act node: execute the current step through the registry.

Re-runs the deterministic policy check and re-validates args so that a plan
mutated between confirmation and execution can never run.  Only Tier-0 steps
or steps whose exact action_hash was previously approved may execute.  All
side effects happen here and only here.

**Exactly-once execution.**  A step that already produced a ``StepResult`` is
never executed a second time: :func:`_recorded_result` short-circuits and
replays the recorded outcome.  This is what makes provider fallback and
replanning safe.  A fallback provider re-asks only the *LLM* (classification /
planning), never a tool, but a replan, a resumed checkpoint, or a re-entered
node can all re-present the same ``step_id`` — and a non-idempotent step
(``whatsapp_send``, ``file_delete``, ``type_text``) must not be repeated just
because the graph visited this node again.  See docs/03 §7 (invariants 6/7)
and AGENTS.md §5.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from jarvis.agent.context import AppContext
from jarvis.agent.nodes.util import truncate
from jarvis.agent.state import StepResult

logger = logging.getLogger(__name__)


def _recorded_result(state: dict[str, Any], step_id: str) -> StepResult | None:
    """Return the recorded outcome for ``step_id``, or ``None``.

    ``StepResult`` carries no timestamp, so the *last* match is taken as the
    current one: a later retry must not be masked by an earlier result.
    """
    for result in reversed(list(state.get("results") or [])):
        if getattr(result, "step_id", None) == step_id:
            return result  # type: ignore[no-any-return]
    return None


def act(state: dict[str, Any], ctx: AppContext) -> dict[str, Any]:
    """Execute the current step; never trust state alone — re-decide."""
    plan = state.get("plan")
    idx = int(state.get("step_index") or 0)
    if plan is None or not plan.steps or idx >= len(plan.steps):
        return {"halted_reason": "No step to execute."}

    step = plan.steps[idx]

    # Exactly-once guard for *successful* side effects.  A step that already
    # succeeded (and was not rejected by its own verify()) is advanced past
    # rather than executed again, so a re-entered node, a resumed checkpoint,
    # or a replan that proposes the same step_id can never repeat a
    # non-idempotent action (``whatsapp_send``, ``file_delete``, ``type_text``).
    # Provider fallback is safe by the same argument: it re-asks only the LLM,
    # never a tool.  A step that *failed* is still retryable — that is the
    # designed retry path, and verification (not this guard) is what decides
    # whether a failure is recoverable.
    recorded = _recorded_result(state, step.id)
    if recorded is not None and recorded.ok and recorded.verified is not False:
        logger.info("agent step %s already executed; not repeating it", step.id)
        return {
            "step_index": idx + 1,
            "retry_count": 0,
            "retry_now": False,
            "replan_now": False,
        }

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

    # TOCTOU guard: the confirmation was bound to the exact resolved paths the
    # gate stashed before interrupt.  If any of them resolves differently now
    # (file swapped for a junction, folder renamed in between), refuse instead
    # of acting on a path the user never confirmed.
    prior_paths = (state.get("gated_resolved_paths") or {}).get(step.id)
    if prior_paths is not None and list(prior_paths) != decision.resolved_paths:
        return append_result(
            state,
            StepResult(
                step_id=step.id,
                ok=False,
                error="resolved path changed after confirmation",
                verified=False,
            ),
            halted="Refused: a file path changed after you confirmed — not acting on it.",
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
        verify_note=result.verify_note,
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
