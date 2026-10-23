"""policy_gate node: the only place confirmations happen.

Computes the policy :class:`Decision` for the *current* step.  If the step is
Tier 0 it passes straight through; if it needs confirmation it suspends the
graph with ``interrupt(payload)`` (no side effects before the interrupt).  On
resume the node re-runs deterministically, so the decision and its action_hash
are recomputed and the supplied answer is re-checked, preventing a stale or
tampered approval from being trusted.

Stage 7 (Slice 8) adds a resolver + task-scope front end, without weakening the
engine's sole authority:

* the interaction target for the step is resolved *before* the decision (Slice
  5 ``resolve`` hook) and fed to the engine through a :class:`PolicyContext`;
* when a browser step is blocked **only** because its origin is outside the
  approved domain scope, the gate offers a single *scope* confirmation bound to
  the plan hash.  On approval it mints a transient :class:`ScopeGrant` (daemon
  layer issues the id; the gate mints its own grant id) and re-decides;
* a target that cannot be reliably observed is a fail-closed refusal with no
  confirmation, never a silent pass.

The engine remains the only component that decides tiers; nothing here lowers
one, and all new checks can only block or re-confirm.
"""

from __future__ import annotations

import dataclasses
import logging
import uuid
from pathlib import Path
from typing import Any

from langgraph.types import interrupt

from jarvis.agent.batch_approval import (
    TYPE_PLAN_APPROVAL,
    compute_all_decisions,
    decide_and_build_payload,
    recompute_and_resolve_batch,
)
from jarvis.agent.context import AppContext
from jarvis.agent.nodes.util import (
    StepTarget,
    active_scope_cid,
    plan_hash_for,
    policy_context_for,
    resolve_step_target,
)
from jarvis.agent.precheck import (
    blocked_precheck_steps,
    precheck_refusal,
    step_precheck_error,
)
from jarvis.agent.schemas import ConfirmationRequest
from jarvis.agent.state import Decision
from jarvis.policy import rules, tiers
from jarvis.policy.paths import ContentHashError, content_hash
from jarvis.policy.refusal import refusal_text as refusal_sentence
from jarvis.policy.scope import ScopeError
from jarvis.voice.tier2_voice import allowed as tier2_voice_allowed
from jarvis.voice.tier2_voice import voice_marker_present

__all__ = ["confirmation_payload", "policy_gate", "refusal_text"]

logger = logging.getLogger(__name__)


def policy_gate(state: dict[str, Any], ctx: AppContext) -> dict[str, Any]:
    """Decide the current step; interrupt when the user must confirm."""
    plan = state.get("plan")
    if plan is None:
        return {"halted_reason": "No plan to gate."}
    if plan.needs_clarification:
        return {"halted_reason": "Clarification needed before I can act."}
    idx = int(state.get("step_index") or 0)
    if idx >= len(plan.steps):
        return {"halted_reason": "No step remaining at the gate."}

    step = plan.steps[idx]

    # Base context: tainted fragments from previous results and the user's own
    # words, so the engine can verify taint independently of what the LLM
    # asserted in ``step.depends_on_untrusted`` (docs/03 §8) and the risk
    # classifier sees the original request (docs/08 §3).
    base_pctx = policy_context_for(state, ctx.policy_ctx)
    spec = ctx.registry.get_optional(step.tool)

    # Read-only precondition (e.g. the messaging recipient).  It runs *before*
    # any interrupt, so an unknown or ambiguous contact is refused without
    # asking the user to approve the action or unlock the session.  A failure
    # can only block; the engine still owns the tier decision.
    precheck_error = step_precheck_error(spec, step, ctx.tool_context())
    if precheck_error is not None:
        decision = precheck_refusal(step, precheck_error)
        decisions = dict(state.get("decisions") or {})
        decisions[step.id] = decision
        return {"decisions": decisions, "halted_reason": refusal_text(decision, decision.step_id)}

    plan_hash_v = plan_hash_for(plan, base_pctx, ctx.engine)

    # Resolve the interaction target (if any) and build the context the engine
    # must decide with.  A target that cannot be observed is refused outright
    # (fail closed) - no confirmation may paper over an unobservable target.
    resolved = resolve_step_target(
        base_pctx,
        step=step,
        spec=spec,
        settings=ctx.settings,
        tool_ctx=ctx.tool_context(),
        scope_registry=ctx.scope_registry,
        task_id=str(state.get("task_id") or ""),
        plan_hash_v=plan_hash_v,
    )
    if resolved.refused:
        return _resolution_refusal(state, step, resolved.reason)

    decision = ctx.engine.decide(step, resolved.context)
    logger.info(
        "policy gate step=%s allowed=%s reasons=%s", step.id, decision.allowed, decision.reasons
    )
    decisions = dict(state.get("decisions") or {})
    decisions[step.id] = decision

    if not decision.allowed:
        banned = _blocked_or_scope(
            state, ctx, step, idx, decisions, base_pctx, plan_hash_v, resolved, decision
        )
        return banned

    if not decision.needs_confirm:
        return {"decisions": decisions}

    already_approved = [str(h) for h in (state.get("approved_hashes") or [])]

    # A step whose exact action_hash the user already approved (typically by an
    # earlier plan-level batch covering this step and the next ones) must never
    # be confirmed a second time: walk straight to act, which re-decides, re-checks
    # the approval binding and the TOCTOU paths itself.  Fail closed: a Tier 2
    # hash can only reach here from an approval the gate itself issued while the
    # session was unlocked, so the same unlock check still applies.
    if str(decision.action_hash) in already_approved:
        if decision.needs_unlock and (ctx.unlock is None or not ctx.unlock.is_unlocked()):
            return {
                "decisions": decisions,
                "halted_reason": "Refused: Tier 2 action needs an unlocked JARVIS session.",
            }
        return {"decisions": decisions}

    # Build the interrupt payload.  Batch eligibility needs a decision for every
    # remaining step, so the whole plan is decided here with the SAME engine and
    # PolicyContext the current step was just decided with (read-only, exactly as
    # `validate` already does for its TOCTOU snapshot).  When 2+ contiguous
    # Tier 1 steps are eligible this is a single ``plan_approval`` payload that
    # names them all; otherwise it is the existing per-step ``confirm`` payload.
    # No side effects before the interrupt: the payload is built from decisions
    # the engine already produced.
    decisions = {**compute_all_decisions(plan, base_pctx, ctx.engine), step.id: decision}
    # A later step whose read-only precondition already fails must never be
    # folded into this plan-level approval: it ends the contiguous eligible run
    # and is refused on its own when the gate reaches it.
    blocked = blocked_precheck_steps(plan, ctx.registry, ctx.tool_context())
    payload = decide_and_build_payload(
        plan,
        decisions,
        step_index=idx,
        approved_hashes=already_approved,
        blocked_steps=blocked,
    )
    answer = interrupt(payload)  # graph pauses; checkpoint saved; no side effects above

    if _is_plan_approval_payload(payload):
        updates, newly_approved = recompute_and_resolve_batch(
            plan, decisions, state, answer, blocked_steps=blocked
        )
        if "halted_reason" in updates:
            gated = dict(state.get("gated_resolved_paths") or {})
            if "gated_resolved_paths" not in updates:
                updates["gated_resolved_paths"] = gated
            return updates
        gated = updates.get("gated_resolved_paths") or dict(state.get("gated_resolved_paths") or {})
        # A batch approval covers only contiguous Tier 1 steps; an opted-in
        # (Tier 2) tool is never batch-eligible, so no content hash is recorded
        # here.  If one ever were, act would refuse it for a missing binding.
        # Target-bearing steps the batch just approved still need a transient
        # binding: the batch decisions are target-less, so resolve each one now
        # (read-only) and record the binding the act node will re-verify (Slice 9).
        _record_batch_bindings(ctx, state, plan, decisions, newly_approved, base_pctx, plan_hash_v)
        return {
            "decisions": decisions,
            "gated_resolved_paths": gated,
            "approved_hashes": updates.get(
                "approved_hashes", list(state.get("approved_hashes") or [])
            ),
        }

    # Single-step path (unchanged semantics; payload shape may differ in fields
    # the daemon already tolerates, but the binding is still the single
    # decision.action_hash).
    gated = dict(state.get("gated_resolved_paths") or {})
    prior_paths = gated.get(step.id)
    gated[step.id] = list(decision.resolved_paths)

    if prior_paths is not None and prior_paths != decision.resolved_paths:
        return {
            "decisions": decisions,
            "gated_resolved_paths": gated,
            "halted_reason": "Refused: a file path changed after you confirmed — not acting on it.",
        }

    # Content-hash binding (Stage 8.4d) happens inside ``_approve_single_step``,
    # after every approval check has passed: a denied/timed-out/locked
    # confirmation must not leave a recorded content binding behind.
    return _approve_single_step(
        state,
        ctx,
        step,
        decisions,
        gated,
        decision,
        answer,
        resolved.target,
        _active_scope_cid(ctx, state),
    )


# ── Stage 7 (Slice 8): scope front end ────────────────────────────────────


def _blocked_or_scope(
    state: dict[str, Any],
    ctx: AppContext,
    step: Any,
    idx: int,
    decisions: dict[str, Any],
    base_pctx: Any,
    plan_hash_v: str,
    resolved: StepTarget,
    decision: Decision,
) -> dict[str, Any]:
    """Handle a blocked step: offer a scope confirmation when that is the *only* block.

    A browser target is offered a scope approval only when re-deciding with the
    origin's host added to the approved scope (and ``scope_approved`` set) makes
    the step *allowed* below Tier 3.  Payment/password/ambiguous/other blocks
    fail this trial and stay hard refusals.  Returns the node update.
    """
    if resolved.host and resolved.target is not None:
        trial = _scope_missing_tier(step, base_pctx, resolved, ctx)
        if trial is not None:
            if trial.allowed and trial.tier < tiers.TIER_BLOCKED:
                return _scope_confirm(
                    state,
                    ctx,
                    step,
                    idx,
                    decisions,
                    base_pctx,
                    plan_hash_v,
                    resolved,
                    trial,
                )
            # The scope was not the (only) blocker: report the engine's specific
            # reason from the hypothetical approved-scope decision (payment /
            # password / ambiguous target) instead of the less specific scope one.
            decisions = {**decisions, step.id: trial}
            return {"decisions": decisions, "halted_reason": refusal_text(trial, step.id)}
    return {"decisions": decisions, "halted_reason": refusal_text(decision, step.id)}


def _scope_missing_tier(
    step: Any, base_pctx: Any, resolved: StepTarget, ctx: AppContext
) -> Decision | None:
    """Hypothetical re-decide with the origin host approved, or ``None``.

    Using the engine itself (never string-matching on reasons) is how we prove
    that a missing scope is the *only* blocker: the same target still goes
    through the password/payment/ambiguity checks.
    """
    if not resolved.host or resolved.target is None:
        return None
    trial = dataclasses.replace(
        base_pctx,
        origin=resolved.target.origin,
        allowed_domains=(resolved.host,),
        scope_approved=True,
        target=resolved.target,
    )
    return ctx.engine.decide(step, trial)


def _scope_confirm(
    state: dict[str, Any],
    ctx: AppContext,
    step: Any,
    idx: int,
    decisions: dict[str, Any],
    base_pctx: Any,
    plan_hash_v: str,
    resolved: StepTarget,
    trial: Decision,
) -> dict[str, Any]:
    """Interrupt for a domain scope, then execute or re-confirm the first action.

    The grant is issued only *after* the final interrupt.  That is deliberate:
    LangGraph re-runs this node from the top on every resume, and the node must
    reach the same interrupt sequence each time.  Because no grant exists until
    the node actually returns, every re-run is still blocked by scope and takes
    this exact path (scope prompt, then - possibly - the risky-action prompt),
    consuming the stacked resume values in order.
    """
    del idx  # the scope covers the whole task; step index is not part of it
    task_id = str(state.get("task_id") or "")
    answer = interrupt(_scope_payload(step, resolved.host, trial, plan_hash_v))  # index 0

    if not _answer_matches(answer, plan_hash_v):
        if isinstance(answer, dict) and answer.get("timed_out") is True:
            return {
                "decisions": decisions,
                "halted_reason": "Confirmation timed out. I did not perform the action.",
            }
        return {"decisions": decisions, "halted_reason": refusal_sentence(answer)}

    # The target was re-observed at the top of THIS run; decide against a context
    # that treats the origin host as approved (the grant is not issued yet).
    scoped = dataclasses.replace(
        base_pctx,
        origin=resolved.target.origin,  # type: ignore[union-attr]
        allowed_domains=(resolved.host,),
        scope_approved=True,
        target=resolved.target,
    )
    decision = ctx.engine.decide(step, scoped)
    decisions = {**decisions, step.id: decision}
    if not decision.allowed:
        return {"decisions": decisions, "halted_reason": refusal_text(decision, step.id)}

    gated = dict(state.get("gated_resolved_paths") or {})
    prior_paths = gated.get(step.id)
    gated[step.id] = list(decision.resolved_paths)
    if prior_paths is not None and prior_paths != decision.resolved_paths:
        return {
            "decisions": decisions,
            "gated_resolved_paths": gated,
            "halted_reason": "Refused: a file path changed after you confirmed — not acting on it.",
        }

    if not _needs_fresh_confirm(decision, resolved.target):
        # Option A: the scope approval *is* this first action's approval.
        grant = _issue_scope_grant(ctx, task_id, plan_hash_v, resolved.host)
        if grant is None:
            return {
                "decisions": decisions,
                "halted_reason": "Refused: the approved browser scope could not be issued.",
            }
        _record_binding_for(ctx, state, step, decision, resolved.target, grant.confirmation_id)
        approved_hashes = list(state.get("approved_hashes") or []) + [str(decision.action_hash)]
        return {
            "decisions": decisions,
            "gated_resolved_paths": gated,
            "approved_hashes": approved_hashes,
        }

    # A risky control (or unlock / typed confirmation) still needs its own fresh
    # action confirmation, independent of the domain approval (owner §3).
    action_answer = interrupt(_fresh_confirm_payload(decision, step))  # index 1
    grant = _issue_scope_grant(ctx, task_id, plan_hash_v, resolved.host)
    if grant is None:
        return {
            "decisions": decisions,
            "halted_reason": "Refused: the approved browser scope could not be issued.",
        }
    result = _approve_single_step(
        state,
        ctx,
        step,
        decisions,
        gated,
        decision,
        action_answer,
        resolved.target,
        grant.confirmation_id,
    )
    if "halted_reason" in result:
        # The action was refused (rejection/timeout): drop the just-issued grant.
        ctx.scope_registry.revoke(task_id)
        return result
    return result


def _needs_fresh_confirm(decision: Decision, target: Any) -> bool:
    """Whether a scope-approved action still needs its own confirmation."""
    if decision.needs_unlock or decision.needs_typed_confirmation:
        return True
    return target is not None and (
        rules.risky_control(target.name, target.role) == rules.CONTROL_FRESH_CONFIRM
    )


def _scope_payload(step: Any, host: str, decision: Decision, plan_hash_v: str) -> dict[str, Any]:
    """The scope-confirmation interrupt payload (``type="confirm"`` + markers).

    Reuses the daemon's single-step ``confirm`` machinery unchanged: the daemon
    issues the correlation id/plan hash/expiry from the fields it already reads.
    ``scope_confirm``/``scope_domain`` are extra, tolerantly-ignored keys that
    let a consumer render the prompt as a task-scope approval.  The payload's
    ``action_hash`` is the plan hash, so a resume can only ever approve *this*
    task's scope (and the grant the gate mints binds the same plan hash).
    """
    summary = f"Approve browsing {host} for this task? {decision.summary}"
    return {
        "type": "confirm",
        "scope_confirm": True,
        "scope_domain": str(host),
        "step_id": str(getattr(step, "id", "") or ""),
        "tier": int(decision.tier),
        "summary": summary,
        "needs_unlock": False,
        "typed_confirmation": None,
        "resolved_paths": [],
        "action_hash": str(plan_hash_v),
        "untrusted": False,
        "tool": str(getattr(step, "tool", "") or ""),
        "allowed": True,
    }


def _fresh_confirm_payload(decision: Decision, step: Any) -> dict[str, Any]:
    """A plain single-step ``confirm`` payload for one risky action.

    Mirrors the exact shape ``decide_and_build_payload`` emits for the
    single-step case (so the daemon and voice layers see no new fields), but is
    built directly so a following batch-eligible step can never fold this action
    into a plan-level approval.
    """
    payload = {
        "type": "confirm",
        "step_id": str(decision.step_id),
        "tier": int(decision.tier),
        "summary": str(decision.summary),
        "needs_unlock": bool(decision.needs_unlock),
        "typed_confirmation": decision.needs_typed_confirmation,
        "resolved_paths": list(decision.resolved_paths),
        "action_hash": str(decision.action_hash),
        "untrusted": bool(decision.warn_untrusted),
        "tool": str(getattr(step, "tool", "") or ""),
        "allowed": bool(decision.allowed),
    }
    if (
        payload["tier"] == 2
        and payload["needs_unlock"]
        and isinstance(getattr(step, "args", None), dict)
    ):
        payload["readback_args"] = dict(step.args)
    return payload


def _issue_scope_grant(ctx: AppContext, task_id: str, plan_hash_v: str, host: str) -> Any | None:
    """Mint the task's scope grant (fail closed when it cannot be issued)."""
    if not task_id or not host or not plan_hash_v:
        return None
    try:
        return ctx.scope_registry.issue(
            task_id,
            plan_hash=plan_hash_v,
            confirmation_id=uuid.uuid4().hex,
            allowed_domains=[host],
            max_actions=int(ctx.settings.browser.task_scope_max_actions),
            ttl_s=float(ctx.settings.browser.scope_expiry_s),
        )
    except ScopeError:
        logger.warning("browser scope grant not issued for task %s", task_id)
        return None


def _active_scope_cid(ctx: AppContext, state: dict[str, Any]) -> str:
    """The current valid grant's confirmation id for this task, else ``""``."""
    return active_scope_cid(ctx.scope_registry, str(state.get("task_id") or ""))


def _record_binding_for(
    ctx: AppContext,
    state: dict[str, Any],
    step: Any,
    decision: Decision,
    target: Any,
    scope_cid: str,
) -> None:
    """Record a transient, one-shot target binding for a target-bearing step.

    No-op for target-less steps or an unidentifiable target.  In-memory only;
    never goes into ``AgentState`` (owner decision 1/8).
    """
    if target is None:
        return
    try:
        fingerprint = target.fingerprint()
    except Exception:  # noqa: BLE001 - fail closed: an unusable target gets no binding
        return
    task_id = str(state.get("task_id") or "")
    if fingerprint is None or not task_id:
        return
    ctx.target_bindings.put(
        task_id,
        str(getattr(step, "id", "") or ""),
        action_hash=str(getattr(decision, "action_hash", "")),
        fingerprint=str(fingerprint),
        origin=str(getattr(target, "origin", "") or ""),
        scope_confirmation_id=str(scope_cid or ""),
    )


def _record_batch_bindings(
    ctx: AppContext,
    state: dict[str, Any],
    plan: Any,
    decisions: dict[str, Any],
    approved_hashes: Any,
    base_pctx: Any,
    plan_hash_v: str,
) -> None:
    """Record transient target bindings for batch-approved target-bearing steps.

    A plan-level approval approves *hashes* and its decisions are deliberately
    target-less (``plan_hash`` must not depend on a runtime target).  The act
    node still requires a gate-issued binding for every target-bearing step, so
    each newly-approved step is resolved here (read-only) and gets one.  A step
    whose target cannot be observed now simply gets no binding and is refused at
    act (fail closed).  Nothing here authorizes anything: the engine already
    decided, and act re-decides against the resolved context.
    """
    approved = {str(h) for h in (approved_hashes or ())}
    if not approved:
        return
    task_id = str(state.get("task_id") or "")
    scope_cid = _active_scope_cid(ctx, state)
    for step in list(getattr(plan, "steps", None) or []):
        decision = decisions.get(str(getattr(step, "id", "") or ""))
        if decision is None or str(getattr(decision, "action_hash", "")) not in approved:
            continue
        resolved = resolve_step_target(
            base_pctx,
            step=step,
            spec=ctx.registry.get_optional(step.tool),
            settings=ctx.settings,
            tool_ctx=ctx.tool_context(),
            scope_registry=ctx.scope_registry,
            task_id=task_id,
            plan_hash_v=plan_hash_v,
        )
        if resolved.refused or resolved.target is None:
            continue
        _record_binding_for(ctx, state, step, decision, resolved.target, scope_cid)


# ── shared single-step approval ────────────────────────────────────────────


def _bound_content_hashes(decision: Decision) -> tuple[list[str] | None, str | None]:
    """Compute the content hashes a ``bind_content_hash`` tool's approval binds to.

    Returns ``(hashes, None)`` on success, where ``hashes`` holds one sha256 per
    resolved path (aligned with ``decision.resolved_paths``), or ``(None, reason)``
    when a file is over the size cap, missing or unreadable.  Called only after
    the approval has succeeded; approval is never bound to content that cannot be
    read, so the caller refuses (fail closed, before dispatch).  Stage 8.4d.
    """
    hashes: list[str] = []
    for raw in decision.resolved_paths:
        try:
            hashes.append(content_hash(Path(raw)))
        except ContentHashError as exc:
            return None, str(exc)
    return hashes, None


def _approve_single_step(
    state: dict[str, Any],
    ctx: AppContext,
    step: Any,
    decisions: dict[str, Any],
    gated: dict[str, Any],
    decision: Decision,
    answer: Any,
    target: Any,
    scope_cid: str,
) -> dict[str, Any]:
    """Validate a single-step approval, record the binding and approve the action.

    A ``ToolSpec.bind_content_hash`` tool additionally gets one sha256 per
    resolved path recorded **only after** every approval check has passed
    (Stage 8.4d).  A denied, timed-out, typed-mismatch or locked-session answer
    records no content binding; a file that cannot be hashed is a refusal and
    the action is not approved.
    """
    content_hashes = dict(state.get("gated_content_hashes") or {})
    approved = _answer_matches(answer, decision.action_hash)
    if not approved:
        if isinstance(answer, dict) and answer.get("timed_out") is True:
            return {
                "decisions": decisions,
                "gated_resolved_paths": gated,
                "gated_content_hashes": content_hashes,
                "halted_reason": "Confirmation timed out. I did not perform the action.",
            }
        return {
            "decisions": decisions,
            "gated_resolved_paths": gated,
            "gated_content_hashes": content_hashes,
            "halted_reason": refusal_sentence(answer),
        }
    if not _typed_confirmation_ok(answer, decision):
        return {
            "decisions": decisions,
            "gated_resolved_paths": gated,
            "gated_content_hashes": content_hashes,
            "halted_reason": "Refused: the typed folder-name confirmation did not match the plan.",
        }
    # Stage 3 opt-in: a voice-approved Tier 2 step may satisfy *this* step's
    # hash without the terminal password — but only when the daemon marked the
    # answer AND the gate re-verifies every condition itself.  This never sets
    # ``ctx.unlock`` and never unlocks the session.
    locked_for_tier2 = decision.needs_unlock and (
        ctx.unlock is None or not ctx.unlock.is_unlocked()
    )
    if locked_for_tier2 and not _voice_tier2_ok(state, answer, step, decision, ctx):
        return {
            "decisions": decisions,
            "gated_resolved_paths": gated,
            "gated_content_hashes": content_hashes,
            "halted_reason": "Refused: Tier 2 action needs an unlocked JARVIS session.",
        }

    # Bind the approval to the file content only now, when it is certain the
    # action was actually approved.  ``content_hashes`` above deliberately holds
    # only bindings from earlier steps, so every refusal path returns without
    # recording anything for this step.
    spec = ctx.registry.get_optional(str(getattr(step, "tool", "") or ""))
    if spec is not None and spec.bind_content_hash:
        hashes, hash_error = _bound_content_hashes(decision)
        if hash_error is not None:
            return {
                "decisions": decisions,
                "gated_resolved_paths": gated,
                "gated_content_hashes": content_hashes,
                "halted_reason": (
                    "Refused: I could not bind approval to the file's content "
                    f"({hash_error}) — not confirming it."
                ),
            }
        content_hashes[str(step.id)] = list(hashes or [])

    _record_binding_for(ctx, state, step, decision, target, scope_cid)
    approved_hashes = list(state.get("approved_hashes") or []) + [str(decision.action_hash)]
    return {
        "decisions": decisions,
        "gated_resolved_paths": gated,
        "gated_content_hashes": content_hashes,
        "approved_hashes": approved_hashes,
    }


def _resolution_refusal(state: dict[str, Any], step: Any, reason: str) -> dict[str, Any]:
    """Refuse a step whose interaction target could not be reliably observed."""
    decision = Decision(
        step_id=str(getattr(step, "id", "") or ""),
        tier=tiers.TIER_BLOCKED,
        allowed=False,
        needs_confirm=False,
        needs_unlock=False,
        reasons=[reason],
        summary=reason,
        action_hash=rules.action_hash_raw(
            str(getattr(step, "tool", "") or ""), getattr(step, "args", None)
        ),
    )
    decisions = dict(state.get("decisions") or {})
    decisions[decision.step_id] = decision
    return {"decisions": decisions, "halted_reason": refusal_text(decision, decision.step_id)}


def _voice_tier2_ok(
    state: dict[str, Any],
    answer: Any,
    step: Any,
    decision: Decision,
    ctx: AppContext,
) -> bool:
    """May a locked-session Tier 2 step proceed because *voice* approved it?

    Fails closed and unlocks nothing.  Three independent conditions: the task
    came from voice (``state["source"]``, which ``intake`` normalises), the
    daemon set its marker on the answer, and
    :func:`jarvis.voice.tier2_voice.allowed` re-derives the verdict from the
    gate's **own** decision and the step's validated args — so the marker alone
    approves nothing.  ``ctx.unlock`` is read but never written.
    """
    if state.get("source") != "voice":
        return False
    if not voice_marker_present(answer):
        return False
    ok, reason = tier2_voice_allowed(
        str(getattr(step, "tool", "") or ""),
        decision,
        ctx.settings,
        args=getattr(step, "args", None),
    )
    if not ok:
        logger.info("voice Tier 2 approval rejected at the gate: %s", reason)
    return ok


def confirmation_payload(decision: Decision, untrusted: bool) -> dict[str, Any]:
    """The interrupt payload the daemon/CLI forwards to the user.

    Retained for tests and for callers that build a single-step payload
    directly.  The gate node itself now uses ``decide_and_build_payload``
    (batch-aware) so a two-step Tier 1 plan raises one interrupt.
    """
    payload = ConfirmationRequest(
        step_id=decision.step_id,
        tier=decision.tier,
        summary=decision.summary,
        needs_unlock=decision.needs_unlock,
        typed_confirmation=decision.needs_typed_confirmation,
        resolved_paths=list(decision.resolved_paths),
        action_hash=decision.action_hash,
        untrusted=bool(untrusted),
    )
    return payload.model_dump(mode="json")


def _is_plan_approval_payload(payload: Any) -> bool:
    """Whether *payload* is a plan-level approval interrupt."""
    return isinstance(payload, dict) and payload.get("type") == TYPE_PLAN_APPROVAL


def _answer_matches(answer: Any, expected_hash: str) -> bool:
    """A resume value only counts as approval when it carries the exact hash."""
    if not isinstance(answer, dict):
        return False
    if answer.get("approved") is not True:
        return False
    provided = answer.get("action_hash")
    return isinstance(provided, str) and provided == expected_hash


def _typed_confirmation_ok(answer: Any, decision: Decision) -> bool:
    """When the step needs a typed folder-name confirmation the resume value
    must carry exactly that text (bound to the resolved folder at decision
    time).  Fails closed: a missing or wrong answer is a refusal."""
    if not decision.needs_typed_confirmation:
        return True
    if not isinstance(answer, dict):
        return False
    provided = answer.get("typed_confirmation")
    return isinstance(provided, str) and provided == decision.needs_typed_confirmation


def refusal_text(decision: Decision, step_id: str) -> str:
    """Honest, deterministic refusal message for a blocked step."""
    reason = "; ".join(decision.reasons) if decision.reasons else "blocked by policy"
    return f"Refused: {decision.summary} [step {step_id}] ({reason})"
