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
from urllib.parse import urlsplit

from jarvis.agent.context import AppContext
from jarvis.agent.nodes.util import (
    active_scope_cid,
    browse_grant_check,
    plan_hash_for,
    policy_context_for,
    resolve_step_target,
    specialist_active,
    truncate,
)
from jarvis.agent.precheck import step_precheck_error
from jarvis.agent.session_context import SESSION_THREAD
from jarvis.agent.state import StepResult
from jarvis.browser.final_click import FINAL_CLICK_TOOL, is_hex64
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


def _live_page_host(ctx: AppContext) -> str:
    """Host of the managed browser's live active page, or ``""``.

    ``browser_press_key`` has no target to resolve, so the gate derives its
    host from the typed-target history marker.  That marker records where a
    previous step *was*, not where the browser is now (a redirect, another tab
    or an intervening navigation can move it), so act reads the live URL here
    and compares it with the grant host before pressing the key.
    """
    session = getattr(ctx, "browser", None)
    getter = getattr(session, "current_url", None)
    if getter is None:
        return ""
    try:
        url = str(getter() or "").strip()
    except Exception:  # noqa: BLE001 - a session that cannot report it has none
        return ""
    if not url:
        return ""
    try:
        return (urlsplit(url).hostname or "").casefold().rstrip(".")
    except ValueError:
        return ""


def _consume_target_binding(
    ctx: AppContext,
    state: dict[str, Any],
    step: Any,
    decision: Any,
    target: Any,
    task_id: str,
    scope_confirmation_id: str,
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
        scope_confirmation_id=scope_confirmation_id,
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
        browse_grants=ctx.browse_grants,
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

    # Phase 4: a task-scoped browsing grant may pre-approve this exact step.
    # The coverage rules are re-run independently here, against the freshly
    # re-observed target (`resolve_step_target` above) - a changed or risky
    # element fails closed and the step keeps its normal confirmation path.
    grant_id = ""
    if decision.needs_confirm and specialist_active(state):
        grant, _host, _reason = browse_grant_check(
            state, ctx, step=step, decision=decision, resolved=resolved
        )
        if grant is not None:
            if str(getattr(step, "tool", "") or "") == "browser_press_key":
                live_host = _live_page_host(ctx)
                if not live_host or not grant.covers_host(live_host):
                    return _refuse(
                        state,
                        step.id,
                        "live page host does not match the browsing grant",
                        "Refused: the page changed since you approved browsing - "
                        "not pressing the key.",
                    )
            grant_id = grant.grant_id
            decision = decision.model_copy(
                update={"reasons": [*decision.reasons, f"browse grant {grant_id}"]}
            )

    if (
        decision.needs_confirm
        and not grant_id
        and decision.action_hash not in (state.get("approved_hashes") or [])
    ):
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

    # Read-only precondition (e.g. the messaging recipient), re-run here to
    # close the window between the gate's check and dispatch.  It happens
    # before any target binding is consumed and before budget is reserved, so a
    # refused step never spends either.
    precheck_error = step_precheck_error(spec, step, ctx.tool_context())
    if precheck_error is not None:
        return _refuse(
            state,
            step.id,
            f"precondition failed: {precheck_error}",
            f"Refused: {precheck_error}",
        )

    # Objective A (target identity): a target-bearing action must present the
    # binding the gate recorded, recomputed from a *fresh* observation.  A
    # changed, missing, expired or unauthorized target fails closed.  Dry runs
    # execute nothing, so they neither consume the one-shot binding nor budget.
    if resolved.target is not None and not ctx.dry_run:
        # One identifier per step, derived here from act's own state (never
        # copied from the gate's decision):
        # * browse-covered - the gate recorded the binding under the browse
        #   grant's id (policy_gate.py `_browse_grant_gate`), and `grant_id`
        #   above is the same grant, re-derived from the fresh coverage check;
        # * otherwise - the active ScopeGrant's confirmation id, or "" when the
        #   task has no scope grant, which is what the gate recorded for a plain
        #   confirmed step.
        # A stale, expired or superseded authorization therefore mismatches and
        # is refused below; a host-B step can never present host A's grant.
        observed_id = grant_id if grant_id else active_scope_cid(ctx.scope_registry, task_id)
        binding_error = _consume_target_binding(
            ctx, state, step, decision, resolved.target, task_id, observed_id
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
        # A granted skip charges the browsing registry, not the scope registry;
        # every other step keeps the unchanged task-scope reservation.
        if grant_id:
            budget = ctx.browse_grants.charge(task_id)
        else:
            budget = ctx.scope_registry.reserve(task_id, cost=int(spec.budget_cost))
        if not budget.ok:
            return _refuse(
                state,
                step.id,
                f"task action budget: {budget.reason}",
                "Refused: this task has reached its action limit.",
            )

    tool_ctx = ctx.tool_context()
    # The stable identity of this approved dispatch, for operation-bound
    # duplicate-send guards.  It is derived from the same task id and action
    # hash act just re-checked, never from planner output.
    tool_ctx.operation_id = f"{task_id}:{decision.action_hash}"
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
    # Phase 5C: a consequential browser click must present the trusted readback
    # digest the gate minted for *this* step and action.  It is read from
    # ``gated_browser_readback_hashes``, which only the gate ever writes, only
    # after every approval check has passed, and only with the digest of the
    # *observed* readback (never the value the client echoed back).  A missing,
    # malformed or unassociated binding is refused here, before the tool is
    # dispatched; the tool then re-observes the real target and compares
    # digests again immediately before it clicks.
    if str(getattr(spec, "name", "") or "") == FINAL_CLICK_TOOL and not ctx.dry_run:
        bound_readback = (state.get("gated_browser_readback_hashes") or {}).get(step.id)
        if not is_hex64(bound_readback):
            return _refuse(
                state,
                step.id,
                "readback binding is missing or malformed",
                "Refused: I could not match your approval to the observed action — not clicking.",
            )
        tool_ctx.approved_final_click_readback_hash = str(bound_readback)
    start = time.perf_counter()
    result = spec.execute(args, tool_ctx)
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    # Session context (Stage 8, D19): remember the artifact a tool just produced
    # so a later "delete it"/"send it" can resolve to the real path.
    if not ctx.dry_run and result.ok:
        artifact = result.data.get("artifact") if isinstance(result.data, dict) else None
        ctx.session_context.note_action(SESSION_THREAD, artifact)
        page_url = result.data.get("page_url") if isinstance(result.data, dict) else None
        # Phase 1: do NOT clear page referent on unrelated successful steps.
        # Clear only on browser navigation to a different origin, explicit
        # invalidation, or TTL. Keep SessionContext API working.
        try:
            from jarvis.agent import referents
            from jarvis.agent.referents import (
                REFERENT_STORE,
                ActionRef,
                FileRef,
                PageRef,
                ResultItem,
                ResultSetRef,
                origin_from_url,
            )

            cid = str(state.get("conversation_id") or "local")
            tool_name = str(getattr(step, "tool", "") or "")
            # record action
            try:
                REFERENT_STORE.set_action(
                    cid, ActionRef(tool=tool_name, ok=True, created_at=referents.now_ts())
                )
            except (TypeError, ValueError):
                # Unreachable in practice: the store only takes a lock and
                # assigns, and the ref model was just constructed.  Kept
                # defensive; the message carries no tool or page content.
                logger.debug("referent action not recorded")
            # browser navigation -> set/update page; clear only if different origin? spec says clear only on nav to different origin or explicit/TTL
            if tool_name in ("browser_goto", "browser_open_site", "browser_open_result") and isinstance(page_url, str) and page_url.strip():
                prev = REFERENT_STORE.get(cid).page
                new_origin = origin_from_url(page_url)
                old_origin = origin_from_url(prev.url) if prev else None
                if prev is None or old_origin != new_origin or prev.url != page_url:
                    # different origin or new page -> update page (and per spec clear only in those cases; just set)
                    REFERENT_STORE.set_page(cid, PageRef(url=page_url, title=None, origin=new_origin, created_at=referents.now_ts()))
                else:
                    REFERENT_STORE.set_page(cid, PageRef(url=page_url, title=None, origin=new_origin, created_at=referents.now_ts()))
                ctx.session_context.note_page(SESSION_THREAD, page_url)
            elif isinstance(page_url, str) and page_url.strip() and tool_name in ("browser_read",):
                # reading doesn't necessarily change page; don't force clear
                pass
            # browser_find results
            if tool_name == "browser_find":
                results = None
                if isinstance(result.data, dict):
                    results = result.data.get("results") or result.data.get("items")
                if isinstance(results, list) and results:
                    items = []
                    for i, it in enumerate(results):
                        if isinstance(it, dict):
                            items.append(ResultItem(
                                position=int(it.get("position") or i + 1),
                                title=str(it.get("title") or it.get("name") or ""),
                                host=str(it.get("host") or it.get("origin") or origin_from_url(it.get("url") or page_url or "") or ""),
                                url=str(it.get("url") or ""),
                                ref=it.get("ref"),
                            ))
                    if items and isinstance(page_url, str) and page_url.strip():
                        REFERENT_STORE.set_result_set(cid, ResultSetRef(id=f"rs_{int(referents.now_ts())}", page_url=page_url, items=items, created_at=referents.now_ts()))
            # browser_open_result may produce page_url
            # files/photos
            if tool_name in ("take_photo",):
                p = None
                if isinstance(artifact, dict):
                    p = artifact.get("path") or result.data.get("path")
                if isinstance(p, str) and p:
                    REFERENT_STORE.set_file(cid, FileRef(path=p, kind="photo", created_at=referents.now_ts()))
            if tool_name in ("create_file", "write_file", "append_file", "save_dictation"):
                p = None
                if isinstance(artifact, dict):
                    p = artifact.get("path") or result.data.get("path")
                if isinstance(p, str) and p:
                    REFERENT_STORE.set_file(cid, FileRef(path=p, kind="file", created_at=referents.now_ts()))
        except (ImportError, TypeError, ValueError):
            # Exactly what this block can raise: the optional import, a
            # page-supplied position that is not a number (``int(...)``), or a
            # ref model that fails validation (``ValidationError`` IS a
            # ``ValueError``).  Referent bookkeeping is convenience only - it can
            # never fail the step - and nothing is logged in detail because the
            # values are page and tool content.
            logger.debug("referent bookkeeping skipped")
        # For non-browser unrelated steps, do NOT clear page referent (Phase 1 fix)

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
