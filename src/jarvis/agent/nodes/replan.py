"""replan node: propose a fresh plan when a step fails beyond its retries.

Bounded by ``settings.agent.max_replans`` (checked in code, never by prompt).
The LLM receives the completed steps, the failing step and its error, the
remaining tool catalogue (no tiers), each step's closed evidence contract and
its deterministic verdict, and returns a :class:`ReplanDecision`:
either ``continue`` with a replacement :class:`Plan` for the *remaining* work,
or ``stop`` with an honest message.  A stop / exhausted budget / LLM failure
halts the task fail-closed; the ``respond`` node turns that into the final
answer.
"""

from __future__ import annotations

from typing import Any

from jarvis.agent.context import AppContext
from jarvis.agent.context_composer import CONTEXT_BUDGET_EXCEEDED, compose_replan_context
from jarvis.agent.evidence import normalise_kind
from jarvis.agent.nodes.telemetry import llm_attempt_meta
from jarvis.agent.nodes.util import truncate
from jarvis.agent.results import effective_results
from jarvis.agent.schemas import ReplanDecision, ToolCall
from jarvis.agent.state import StepResult

_NO_LLM = "No LLM backend configured (run `jarvis init`)."
_MAX_REMAINING_STEPS = 8
_REMAINING_TEXT_CHARS = 120


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

    toolcalls = _toolcalls(state)
    composed = compose_replan_context(
        state,
        ctx,
        toolcalls=[call.model_dump() for call in toolcalls],
        remaining_steps=_remaining_steps(state),
        goal=_goal(state),
    )
    if not composed.stats.fits:
        return {"halted_reason": CONTEXT_BUDGET_EXCEEDED}
    llm_budget: dict[str, Any] = {}
    if (state.get("source") or "terminal") == "voice":
        llm_budget = {
            "max_retries": 0,
            "timeout_s": min(float(ctx.settings.llm.timeout_seconds), 10.0),
        }
    # B8.5: the replanner's provider attempts belong to this task too, so a
    # fallback during replanning is visible in the task history.
    attempt_meta = llm_attempt_meta(state, ctx, "action", context_stats=composed.stats)
    if attempt_meta is not None:
        llm_budget["attempt_meta"] = attempt_meta
    try:
        model, usage = ctx.llm.structured(
            system=composed.system,
            user=composed.user,
            schema=ReplanDecision,
            model_role="planner",
            temperature=0.0,
            **llm_budget,
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
    verified successes.  :func:`~jarvis.agent.results.effective_results` first
    collapses retries to the latest attempt per step, so a step that failed
    once and then succeeded is kept as a completed success.  On a repair call
    (replan again because validate rejected the plan) results were already
    trimmed, so nothing is lost.
    """
    effective = effective_results(state)
    if state.get("pending_replan"):
        return effective
    idx = int(state.get("step_index") or 0)
    return effective[:idx]


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
                expect=step.expect,
                status=status,
                evidence=normalise_kind(getattr(step, "evidence", None)),
                evidence_status=_evidence_status(r),
                output=truncate(r.output or ""),
                error=r.error,
                tainted=r.tainted,
            )
        )
    return calls


def _evidence_status(result: StepResult) -> str:
    """Return the closed evidence verdict, or ``""`` for legacy results."""
    status = getattr(result, "evidence_status", None)
    return status if status in ("confirmed", "contradicted", "inconclusive") else ""


def _remaining_steps(state: dict[str, Any]) -> list[dict[str, object]]:
    """Bounded view of unfinished work; argument values stay out of the prompt."""
    plan = state.get("plan")
    if plan is None:
        return []
    idx = int(state.get("step_index") or 0)
    steps = list(getattr(plan, "steps", None) or [])[idx : idx + _MAX_REMAINING_STEPS]
    return [
        {
            "id": getattr(step, "id", ""),
            "tool": getattr(step, "tool", ""),
            "rationale": truncate(str(getattr(step, "rationale", "") or ""), _REMAINING_TEXT_CHARS),
            "expect": truncate(str(getattr(step, "expect", "") or ""), _REMAINING_TEXT_CHARS),
            "evidence": normalise_kind(getattr(step, "evidence", None)),
            "args_keys": sorted(str(key) for key in (getattr(step, "args", None) or {})),
        }
        for step in steps
    ]


def _stop(state: dict[str, Any], decision: ReplanDecision) -> dict[str, Any]:
    """Fail-closed stop: keep the decision for observability, halt the task."""
    message = (decision.message or "the task could not be completed").strip().rstrip(".")
    return {
        "replan_decision": decision,
        "halted_reason": f"Could not complete the task: {message}.",
    }
