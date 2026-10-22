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

**Content binding (8.4d).**  A tool whose spec sets ``bind_content_hash`` may
only run after act has re-read every resolved path and matched the sha256 the
policy gate recorded when the user approved.  The comparison happens after the
path/target checks and immediately before the budget is reserved and the tool
is dispatched; the verified digests are then handed to the tool context.  This
is a narrow window, not an atomic filesystem lock - an opted-in tool re-checks
again immediately before its irreversible action.

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
from pathlib import Path
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
from jarvis.policy.paths import ContentHashError, content_hash

logger = logging.getLogger(__name__)


def _content_hashes_now(decision: Any) -> tuple[list[str] | None, str | None]:
    """Recompute a ``bind_content_hash`` step's content hashes from a fresh read.

    Returns ``(hashes, None)`` on success (one sha256 per resolved path) or
    ``(None, reason)`` when a file is over the size cap, missing or unreadable.
    Mirrors the policy gate's binding so act proves the same content the user
    confirmed (Stage 8.4d).
    """
    hashes: list[str] = []
    for raw in getattr(decision, "resolved_paths", None) or []:
        try:
            hashes.append(content_hash(Path(raw)))
        except ContentHashError as exc:
            return None, str(exc)
    return hashes, None


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

    # Content-hash binding (Stage 8.4d).  A tool that opts in must present the
    # sha256 the gate recorded, recomputed from a fresh read right here, after
    # the path/target checks and immediately before the budget is reserved and
    # the tool is dispatched.  A changed, missing, over-cap, malformed or
    # never-recorded binding fails closed: the file's *content* is what the
    # user confirmed, not just its path.
    #
    # Boundary: this is a re-read, not an atomic filesystem lock.  A writer can
    # still replace the bytes after this check; an opted-in tool therefore
    # re-hashes again immediately before its irreversible action
    # (``whatsapp_send_file`` does, just before Send).
    approved_digests: tuple[str, ...] | None = None
    if spec.bind_content_hash and not ctx.dry_run:
        snapshot = (state.get("gated_content_hashes") or {}).get(step.id)
        if prior_paths is None or list(prior_paths) != list(decision.resolved_paths):
            return _refuse(
                state,
                step.id,
                "content binding is not associated with this action's resolved paths",
                "Refused: I could not match the approval to this file — not acting on it.",
            )
        if (
            not isinstance(snapshot, (list, tuple))
            or not snapshot
            or not all(isinstance(item, str) and item for item in snapshot)
        ):
            return _refuse(
                state,
                step.id,
                "content binding is missing or malformed",
                "Refused: I could not verify the file's content before acting — not acting on it.",
            )
        fresh, hash_error = _content_hashes_now(decision)
        if hash_error is not None:
            return _refuse(
                state,
                step.id,
                f"content could not be re-verified: {hash_error}",
                "Refused: the file's content could not be verified immediately before acting — not acting on it.",
            )
        if list(snapshot) != list(fresh or []):
            return _refuse(
                state,
                step.id,
                "file content changed after confirmation",
                "Refused: the file's content changed after you confirmed — not acting on it.",
            )
        approved_digests = tuple(snapshot)

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
    if approved_digests is not None:
        # The digests the gate bound the approval to, after act's own re-read
        # matched them.  The tool may use them only to re-verify, never as a
        # substitute for its own checks.
        tool_ctx.approved_content_hashes = approved_digests
    if spec.bind_target_identity and not ctx.dry_run and resolved.target is not None:
        # The stable identity of the target the gate authorized.  The tool
        # re-derives it from a fresh observation immediately before it acts and
        # refuses when it no longer holds, so the element that is acted on is
        # the element that was confirmed (Stage 8.6).
        tool_ctx.approved_target_identity = str(resolved.target.ref or "")
    start = time.perf_counter()
    result = spec.execute(args, tool_ctx)
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    # Session context (Stage 8, D19): remember the artifact a tool just produced
    # so a later "delete it"/"send it" can resolve to the real path.  The store
    # accepts only the trusted ``result.data["artifact"]`` dict; any other step
    # that *completes successfully* (including one carrying only text) closes the
    # current run, so an unrelated completed action invalidates earlier
    # candidates.  A failed or refused step and a dry run record nothing and
    # leave the run untouched (nothing happened, so the last real artifact still
    # stands).  Never written to AgentState.  This is memory, not authorization:
    # the path is re-checked like any other when it is used.
    if not ctx.dry_run and result.ok:
        artifact = result.data.get("artifact") if isinstance(result.data, dict) else None
        ctx.session_context.note_action(SESSION_THREAD, artifact)
        page_url = result.data.get("page_url") if isinstance(result.data, dict) else None
        if isinstance(page_url, str) and page_url.strip():
            # The page JARVIS is on (Stage 8.6, slice 8.6.3).  A location, not
            # an authorization: the next navigation is checked like any other.
            ctx.session_context.note_page(SESSION_THREAD, page_url)

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
