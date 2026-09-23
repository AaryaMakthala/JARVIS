"""replan node: propose a fresh plan when a step fails beyond its retries.

Bounded by ``settings.agent.max_replans`` (checked in code, never by prompt).
The LLM receives the completed steps, the failing step and its error, the
remaining tool catalogue (no tiers) and returns a :class:`ReplanDecision`:
either ``continue`` with a replacement :class:`Plan` for the *remaining* work,
or ``stop`` with an honest message.  A stop / exhausted budget / LLM failure
halts the task fail-closed; the ``respond`` node turns that into the final
answer.
"""

from __future__ import annotations

from typing import Any

from jarvis.agent.context import AppContext
from jarvis.agent.nodes.util import truncate
from jarvis.agent.schemas import ReplanDecision, ToolCall
from jarvis.agent.state import StepResult
from jarvis.llm.prompts import replan_system_prompt, replan_user_prompt

_NO_LLM = "No LLM backend configured (run `jarvis init`)."


def replan(state: dict[str, Any], ctx: AppContext) -> dict[str, Any]:
    """Ask the LLM for a bounded replacement plan; never acts by itself."""
    replan_count = int(state.get("replan_count") or 0)
    max_replans = ctx.settings.agent.max_replans
    if replan_count >= max_replans:
        return _stop(
            state,
            ReplanDecision(
                action="stop",
                message="the replan budget is exhausted and the task did not complete",
            ),
        )

    if ctx.llm is None:
        return {"halted_reason": _NO_LLM}

    repair_error = state.get("error") if state.get("pending_replan") else None
    catalogue = ctx.registry.catalogue_for_llm()
    try:
        model, usage = ctx.llm.structured(
            system=replan_system_prompt(catalogue),
            user=replan_user_prompt(
                user_input=state.get("user_input") or "",
                goal=_goal(state),
                memory_context=[
                    x if isinstance(x, dict) else {} for x in (state.get("memory_context") or [])
                ],
                toolcalls=_toolcalls(state),
                repair_error=repair_error,
            ),
            schema=ReplanDecision,
            model_role="planner",
            temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001 - LLM failures stop this task only
        ctx.logger.warning("replanning failed: %s", exc)
        return {"halted_reason": f"Replan error: {exc}"}

    if not isinstance(model, ReplanDecision):
        return {"halted_reason": "Replanner returned an unexpected structure."}

    usage_updates = {
        "api_calls": int(state.get("api_calls") or 0) + usage.calls,
        "tokens": int(state.get("tokens") or 0) + usage.prompt_tokens + usage.completion_tokens,
    }

    if model.action == "stop":
        ctx.logger.info(
            "agent event=replan task_id=%s action=stop replan_count=%d",
            state.get("task_id"),
            replan_count,
        )
        return {**_stop(state, model), **usage_updates, "replan_count": replan_count}

    if model.plan is None:
        return {"halted_reason": "Replanner continued without a replacement plan."}

    ctx.logger.info(
        "agent event=replan task_id=%s action=continue replan_count=%d steps=%d",
        state.get("task_id"),
        replan_count,
        len(model.plan.steps),
    )
    return {
        "plan": model.plan,
        "replan_decision": model,
        "pending_replan": True,
        "replan_count": replan_count + 1,
        "replan_now": False,
        "retry_now": False,
        "retry_count": 0,
        "validate_attempts": 0,
        "step_index": 0,
        "error": None,
        "halted_reason": None,
        # drop failed attempts of the superseded step; keep completed successes
        "results": _completed_results(state),
        **usage_updates,
    }


def _goal(state: dict[str, Any]) -> str:
    plan = state.get("plan")
    if plan is not None and getattr(plan, "goal", ""):
        return str(plan.goal)
    return state.get("user_input") or ""


def _completed_results(state: dict[str, Any]) -> list[StepResult]:
    """Only keep results that already succeeded before the failing step.

    When the first replan is triggered ``step_index`` is the count of fully
    verified successes, so ``results[:step_index]`` is exactly the completed
    prefix (a failed attempt never advances the index).  On a repair call
    (replan again because validate rejected the plan) results were already
    trimmed, so nothing is lost.
    """
    if state.get("pending_replan"):
        return list(state.get("results") or [])
    idx = int(state.get("step_index") or 0)
    results = list(state.get("results") or [])
    return [r for r in results if isinstance(r, StepResult)][:idx]


def _toolcalls(state: dict[str, Any]) -> list[ToolCall]:
    """Pair plan steps with their recorded outcomes for the LLM context."""
    plan = state.get("plan")
    if plan is None:
        return []
    steps = {getattr(s, "id", ""): s for s in plan.steps}
    calls: list[ToolCall] = []
    for r in state.get("results") or []:
        if not isinstance(r, StepResult):
            continue
        step = steps.get(r.step_id)
        if step is None:
            continue
        if r.ok and r.verified is not False:
            status = "succeeded"
        elif r.ok:
            status = "unverified"
        else:
            status = "failed"
        calls.append(
            ToolCall(
                tool=step.tool,
                args=step.args,
                status=status,
                output=truncate(r.output or ""),
                error=r.error,
                tainted=r.tainted,
            )
        )
    return calls


def _stop(state: dict[str, Any], decision: ReplanDecision) -> dict[str, Any]:
    """Fail-closed stop: keep the decision for observability, halt the task."""
    message = (decision.message or "the task could not be completed").strip().rstrip(".")
    return {
        "replan_decision": decision,
        "halted_reason": f"Could not complete the task: {message}.",
    }
