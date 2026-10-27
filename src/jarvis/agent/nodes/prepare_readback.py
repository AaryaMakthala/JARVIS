"""Phase 5C: prepare the live readback for a consequential browser click.

The readback the user is asked to approve - and the digest their approval is
bound to - must come from a **live observation of the real target**, and it must
survive the confirmation ``interrupt`` in ``policy_gate``.

LangGraph re-runs an *interrupted* node from the top on resume and discards
anything that node mutated before it called ``interrupt``, so the observation
cannot be taken inside ``policy_gate``: the value it wrote would be gone by the
time the resume value arrives.  This node runs one hop *earlier* and returns
``pending_readback`` as an ordinary state update, so the checkpoint the
interrupt saves already contains it.  ``policy_gate`` then uses the **persisted**
observation instead of re-observing (a resume path that re-observed could be
talked into acting on a *changed* target).

What lands in state is plain JSON only - strings, numbers, booleans and dicts -
so the checkpoint serializer's allowlist is unchanged and no new first-party
type has to be revived.  Nothing here authorizes anything: the PolicyEngine
still decides, ``policy_gate`` still gates and ``act`` is still the only node
that executes a tool.  The observation is read-only (``session.resolve``); no
click, type or navigation happens on any path through this module.
"""

from __future__ import annotations

from typing import Any

from jarvis.agent.context import AppContext
from jarvis.browser.final_click import (
    FINAL_CLICK_TOOL,
    canonical_json,
    final_click_limits,
    format_proposed_note,
    format_readback_summary,
    observe_live_readback,
    readback_identity,
)
from jarvis.policy import rules

__all__ = ["PENDING_READBACK_KEY", "prepare_readback"]

#: The state key the pending observation is persisted under.
PENDING_READBACK_KEY = "pending_readback"


def prepare_readback(state: dict[str, Any], ctx: AppContext) -> dict[str, Any]:
    """Observe the current step's real target and persist its readback.

    Always returns an explicit value for ``pending_readback`` (``None`` for
    every step that is not a consequential browser click), so an observation
    can never outlive the step it was made for.  A step whose readback cannot be
    observed is persisted as a *refusal* - ``policy_gate`` then refuses before
    asking the user for anything, instead of confirming an action JARVIS cannot
    describe.  Nothing is fabricated: an element that cannot be resolved is a
    refusal, never a target reconstructed from the plan's own words.
    """
    reset = {PENDING_READBACK_KEY: None}
    plan = state.get("plan")
    if plan is None or getattr(plan, "needs_clarification", False):
        return reset
    steps = list(getattr(plan, "steps", None) or [])
    idx = int(state.get("step_index") or 0)
    if idx < 0 or idx >= len(steps):
        return reset
    step = steps[idx]
    if str(getattr(step, "tool", "") or "") != FINAL_CLICK_TOOL:
        return reset
    spec = ctx.registry.get_optional(FINAL_CLICK_TOOL)
    if spec is None:
        return reset
    raw_args = getattr(step, "args", None)
    try:
        args = spec.args_model.model_validate(dict(raw_args or {}))
    except Exception:  # noqa: BLE001 - a malformed step is refused by validate/act
        return reset
    action = str(getattr(args, "action", "") or "")
    ref = str(getattr(args, "ref", "") or "")
    try:
        action_hash = str(rules.action_hash(spec, args))
    except Exception:  # noqa: BLE001 - no hash means no binding, so no binding is offered
        return reset

    _max_total, spending_cap, currency, _ttl, _voice = final_click_limits(ctx.settings)
    observation = observe_live_readback(
        getattr(ctx, "browser", None),
        ref,
        action,
        proposed={"expect": str(getattr(args, "expect", "") or "")},
        spending_cap=spending_cap,
        spending_currency=currency,
    )
    if not observation.ok:
        return {
            PENDING_READBACK_KEY: {
                "task_id": str(state.get("task_id") or ""),
                "step_id": str(getattr(step, "id", "") or ""),
                "step_index": idx,
                "action_hash": action_hash,
                "refusal": str(observation.reason),
                "halting": bool(observation.halting),
            }
        }
    details = observation.details or {}
    target_identity, origin = readback_identity(details)
    return {
        PENDING_READBACK_KEY: {
            "task_id": str(state.get("task_id") or ""),
            "step_id": str(getattr(step, "id", "") or ""),
            "step_index": idx,
            "action_hash": action_hash,
            "target_identity": target_identity,
            "origin": origin,
            "schema": int(details.get("schema") or 0),
            "canonical_json": canonical_json(details),
            "digest": observation.digest,
            "summary": format_readback_summary(details),
            "proposed_note": format_proposed_note(details),
            "details": details,
        }
    }
