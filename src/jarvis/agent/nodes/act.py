"""act node: execute the current step through the registry.

Re-runs the deterministic policy check and re-validates args so that a plan
mutated between confirmation and execution can never run.  Only Tier-0 steps
or steps whose exact action_hash was previously approved may execute.  All
side effects happen here and only here.

**Slice 9 revalidation.**  act does not trust the gate's decision: it rebuilds
the same enriched :class:`~jarvis.policy.engine.PolicyContext` (taint, user
words, resolved target/origin/scope), re-observes the interaction target, and
re-decides with the one policy engine.  A target-bearing step must additionally
present the one-shot :class:`~jarvis.agent.target_binding.TargetBinding` the
gate recorded, recomputed from that fresh observation; a changed, missing or
unauthorized target fails closed.  The step's task-scope action budget is then
reserved atomically, immediately before dispatch, and is never refunded.

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
from jarvis.agent.nodes.util import (
    active_scope_cid,
    plan_hash_for,
    policy_context_for,
    resolve_step_target,
    truncate,
)
from jarvis.agent.session_context import SESSION_THREAD
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


def _refuse(state: dict[str, Any], step_id: str, error: str, halted: str) -> dict[str, Any]:
    """Append a refused (never-executed) result and halt the task."""
    return append_result(
        state,
        StepResult(step_id=step_id, ok=False, error=error, verified=False),
        halted=halted,
    )


def _consume_target_binding(
    ctx: AppContext,
    state: dict[str, Any],
    step: Any,
    decision: Any,
    target: Any,
    task_id: str,
) -> str | None:
    """Re-verify the gate's one-shot target binding for this action.

    Returns ``None`` when the freshly re-observed target is byte-identical to
    the one the gate authorized, otherwise an error string.  Fails closed: a
    missing, expired, ambiguous or changed target (or a different scope
    confirmation) is a refusal, and consuming the binding invalidates it so it
    can never authorize a second action.
    """
    if not task_id:
        return "missing task id for target authorization"
    try:
        fingerprint = target.fingerprint()
    except Exception:  # noqa: BLE001 - fail closed on an unusable target
        fingerprint = None
    if not fingerprint:
        return "interaction target has ambiguous or missing identity"
    result = ctx.target_bindings.consume(
        task_id,
        str(getattr(step, "id", "") or ""),
        action_hash=str(getattr(decision, "action_hash", "")),
        fingerprint=str(fingerprint),
        origin=str(getattr(target, "origin", "") or ""),
        scope_confirmation_id=active_scope_cid(ctx.scope_registry, task_id),
    )
    if not result.ok:
        return f"target binding {result.reason}"
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
        return _refuse(state, step.id, "unknown tool", "Refused: unknown tool.")

    # Objective B (policy/execution parity): rebuild the SAME enriched
    # PolicyContext the gate decided with - tainted fragments, the user's words,
    # and the resolved interaction target/origin/scope - and re-decide.  The
    # gate is not trusted to have decided for us; act proves it independently.
    base_pctx = policy_context_for(state, ctx.policy_ctx)
    task_id = str(state.get("task_id") or "")
    plan_hash_v = plan_hash_for(plan, base_pctx, ctx.engine)
    resolved = resolve_step_target(
        base_pctx,
        step=step,
        spec=spec,
        settings=ctx.settings,
        tool_ctx=ctx.tool_context(),
        scope_registry=ctx.scope_registry,
        task_id=task_id,
        plan_hash_v=plan_hash_v,
    )
    if resolved.refused:
        return _refuse(
            state,
            step.id,
            "interaction target could not be reliably observed",
            "Refused: I could not reliably observe the target, so I did not act.",
        )

    decision = ctx.engine.decide(step, resolved.context)
    if not decision.allowed or decision.tier >= 3:
        return _refuse(
            state, step.id, "blocked by policy", f"Refused: blocked by policy ({step.tool})."
        )

    if decision.needs_confirm and decision.action_hash not in (state.get("approved_hashes") or []):
        return _refuse(
            state,
            step.id,
            "approval binding mismatch",
            "Refused: the approved action no longer matches — the plan was likely altered after confirmation.",
        )

    # TOCTOU guard: the confirmation was bound to the exact resolved paths the
    # gate stashed before interrupt.  If any of them resolves differently now
    # (file swapped for a junction, folder renamed in between), refuse instead
    # of acting on a path the user never confirmed.
    prior_paths = (state.get("gated_resolved_paths") or {}).get(step.id)
    if prior_paths is not None and list(prior_paths) != decision.resolved_paths:
        return _refuse(
            state,
            step.id,
            "resolved path changed after confirmation",
            "Refused: a file path changed after you confirmed — not acting on it.",
        )

    try:
        args = spec.args_model.model_validate(step.args)
    except Exception as exc:  # noqa: BLE001 - defensive; validate() already checked
        return _refuse(
            state, step.id, f"invalid args: {exc}", "Refused: step arguments are invalid."
        )

    # Objective A (target identity): a target-bearing action must present the
    # binding the gate recorded, recomputed from a *fresh* observation.  A
    # changed, missing, expired or unauthorized target fails closed.  Dry runs
    # execute nothing, so they neither consume the one-shot binding nor budget.
    if resolved.target is not None and not ctx.dry_run:
        binding_error = _consume_target_binding(
            ctx, state, step, decision, resolved.target, task_id
        )
        if binding_error is not None:
            return _refuse(
                state,
                step.id,
                binding_error,
                "Refused: the interaction target changed or was not authorized — not acting on it.",
            )

    # Objective A (atomic task budget): reserve the step's cost immediately
    # before dispatch, after every pre-dispatch check has passed.  The
    # reservation is atomic (shared per-task counter) and is never refunded.
    if spec.budget_cost > 0 and not ctx.dry_run:
        budget = ctx.scope_registry.reserve(task_id, cost=int(spec.budget_cost))
        if not budget.ok:
            return _refuse(
                state,
                step.id,
                f"task action budget: {budget.reason}",
                "Refused: this task has reached its action limit.",
            )

    tool_ctx = ctx.tool_context()
    start = time.perf_counter()
    result = spec.execute(args, tool_ctx)
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    # Session context (Stage 8, D19): remember the artifact a tool just produced
    # so a later "delete it"/"send it" can resolve to the real path.  The store
    # accepts only the trusted ``result.data["artifact"]`` dict; any other
    # action (including an empty artifact, or one carrying only text) closes the
    # current run, invalidating earlier candidates.  Never recorded on a dry run
    # or a failed step, and never written to AgentState.  This is memory, not
    # authorization: the path is re-checked like any other when it is used.
    if not ctx.dry_run and result.ok:
        artifact = result.data.get("artifact") if isinstance(result.data, dict) else None
        ctx.session_context.note_action(SESSION_THREAD, artifact)

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
