"""Read-only prechecks that must pass *before* a step is confirmed or run.

A tool may declare a :attr:`~jarvis.tools.base.ToolSpec.precheck` hook for a
precondition that is cheap, deterministic and safe to evaluate without side
effects - today the messaging recipient check.  Running it at the policy gate
means an invalid recipient is refused **before** the user is asked to approve a
Tier 2 action or unlock the session; running it again in ``act`` closes the
window between the gate and dispatch.

The hook proposes nothing and authorizes nothing: the PolicyEngine still decides
the tier, and ``act`` still re-decides and re-validates every path and target.
A failed precheck can only block.  An exception is treated as a failure (fail
closed), never as a pass.
"""

from __future__ import annotations

import logging
from typing import Any

from jarvis.agent.state import Decision
from jarvis.platform_guard import is_windows
from jarvis.policy import rules, tiers
from jarvis.tools.base import ToolContext, ToolSpec

__all__ = [
    "blocked_precheck_steps",
    "precheck_refusal",
    "step_precheck_error",
]

logger = logging.getLogger(__name__)


def step_precheck_error(spec: ToolSpec | None, step: Any, ctx: ToolContext) -> str | None:
    """Return the precheck failure for one step, or ``None`` when it passes.

    Invalid args and a ``windows_only`` tool on a non-Windows host are left to
    the engine's own deterministic refusal: a precheck must not mask the more
    specific policy reason.  A hook that raises is a failure (fail closed).
    """
    if spec is None or spec.precheck is None:
        return None
    raw = getattr(step, "args", None)
    if not isinstance(raw, dict):
        return None
    try:
        args = spec.args_model.model_validate(raw)
    except Exception:  # noqa: BLE001 - invalid args are the engine's refusal
        return None
    if spec.windows_only and not is_windows():
        return None
    try:
        error = spec.precheck(args, ctx)
    except Exception:
        logger.warning("precheck for %s raised; refusing", spec.name, exc_info=True)
        return f"the {spec.name} check could not be completed"
    if not error:
        return None
    return str(error).strip() or f"the {spec.name} check failed"


def blocked_precheck_steps(plan: Any, registry: Any, ctx: ToolContext) -> set[str]:
    """Step ids whose precheck fails right now (read-only, never raises)."""
    blocked: set[str] = set()
    for step in list(getattr(plan, "steps", None) or []):
        spec = registry.get_optional(str(getattr(step, "tool", "") or ""))
        if step_precheck_error(spec, step, ctx) is not None:
            blocked.add(str(getattr(step, "id", "") or ""))
    return blocked


def precheck_refusal(step: Any, error: str) -> Decision:
    """A blocked decision for a step whose precheck failed.

    Deliberately not an engine decision: it is a deterministic refusal at the
    gate boundary, with no confirmation and no unlock.  The action hash is
    computed from the raw args exactly like the other refusal paths, so it can
    never match an approval.
    """
    reason = str(error).strip() or "the tool's precondition failed"
    return Decision(
        step_id=str(getattr(step, "id", "") or ""),
        tier=tiers.TIER_BLOCKED,
        allowed=False,
        needs_confirm=False,
        needs_unlock=False,
        reasons=["precondition failed"],
        summary=reason,
        action_hash=rules.action_hash_raw(
            str(getattr(step, "tool", "") or ""), getattr(step, "args", None)
        ),
    )
