"""Deterministic policy decision engine.

The LLM **proposes** actions; this engine **decides** whether they may run
(docs/03_SECURITY_AND_POLICY.md).  ``decide()`` is pure - no filesystem
writes, no network, no LLM - and always returns a :class:`Decision` built from
validated arguments and hard rules.

Invariants honoured here:
* the tier comes from ``base_tier`` and rules only, never from LLM output;
* unknown tools and unparseable args are refused outright;
* the summary shown to the user is generated from canonical args, not from
  the LLM's rationale;
* ``action_hash`` binds approvals to this exact action (re-checked in act).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pydantic import ValidationError

from jarvis.agent.state import Decision, Step
from jarvis.config import Settings
from jarvis.policy import paths, rules, tiers
from jarvis.tools.registry import ToolRegistry, UnknownTool


class RiskClassifier(Protocol):
    """Phase 8: an ML wrapper that may only *raise* a tier."""

    def min_tier(self, step: Step) -> int: ...


class UnlockManager(Protocol):
    """Phase 2: whether the JARVIS session is currently unlocked."""

    def is_unlocked(self) -> bool: ...


@dataclass(frozen=True)
class PolicyContext:
    """Read-only inputs to :func:`decide`.  Never mutated."""

    registry: ToolRegistry
    settings: Settings
    classifier: RiskClassifier | None = None
    unlock: UnlockManager | None = None


def _blocked(step: Step, reasons: list[str], summary: str) -> Decision:
    """Build a refused Decision (tier 3, never executed)."""
    return Decision(
        step_id=step.id,
        tier=tiers.TIER_BLOCKED,
        allowed=False,
        needs_confirm=False,
        needs_unlock=False,
        reasons=reasons,
        summary=summary,
        action_hash=rules.action_hash_raw(step.tool, step.args),
    )


class PolicyEngine:
    """Stateless decision engine.  One instance per application context."""

    def decide(self, step: Step, ctx: PolicyContext) -> Decision:
        """Return the tiered decision for one planned step."""
        try:
            spec = ctx.registry.get(step.tool)
        except UnknownTool as exc:
            return _blocked(step, ["unknown tool not in the registry"], str(exc))

        try:
            args = spec.args_model.model_validate(step.args)
        except ValidationError as exc:
            return _blocked(step, [f"invalid arguments: {exc.error_count()} error(s)"], str(exc))

        tier = spec.base_tier
        reasons: list[str] = []
        blocked: str | None = None

        if rules.matches_blocked(spec, args):
            # Hard block: treat exactly like an unknown/invalid step - Tier 3,
            # recorded reason, never reachable by any confirmation path.
            return _blocked(
                step,
                ["this tool/action is hard-blocked by policy"],
                spec.describe(args),
            )

        # Path rules: apply to every declared path argument.
        if blocked is None:
            roots = paths.roots_from_settings(ctx.settings)
            for name, raw in spec.path_arg_values(args):
                try:
                    resolved = paths.resolve_safe(raw)
                except paths.PathError as exc:
                    blocked = f"bad path in argument {name!r}: {exc}"
                    break
                if paths.is_protected(resolved):
                    blocked = f"path is protected: {resolved}"
                    break
                if not paths.within_any_root(resolved, roots):
                    blocked = f"path is outside the allowed folders: {resolved}"
                    break
                tier = max(tier, rules.path_tier(spec, str(resolved), args))

        tier = max(tier, rules.tool_tier(spec, args))

        if step.depends_on_untrusted and tier >= tiers.TIER_CONFIRM:
            tier = max(tier, tiers.TIER_CONFIRM)
            reasons.append("derived from untrusted content")

        if ctx.classifier is not None:
            tier = max(tier, ctx.classifier.min_tier(step))

        allowed = _matches(blocked, tier)

        summary = spec.describe(args)
        overwrite = rules.overwrite_warning(spec, args)
        if overwrite:
            summary = f"{summary} ({overwrite})"

        return Decision(
            step_id=step.id,
            tier=tier,
            allowed=allowed,
            needs_confirm=tier >= tiers.TIER_CONFIRM,
            needs_unlock=tier >= tiers.TIER_CONFIRM_UNLOCK,
            needs_typed_confirmation=None,
            reasons=reasons,
            summary=summary,
            action_hash=rules.action_hash(spec, args),
        )


def _matches(blocked: str | None, tier: int) -> bool:
    return tier < tiers.TIER_BLOCKED and blocked is None
