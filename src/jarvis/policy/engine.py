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
from jarvis.platform_guard import is_windows
from jarvis.policy import paths, rules, targets, tiers
from jarvis.tools.registry import ToolRegistry, UnknownTool

#: Reason recorded for a ``windows_only`` tool on a host that is not Windows.
#: Deliberately distinct from the generic hard-block text so the human is told
#: the real cause instead of "blocked by policy".
_WINDOWS_ONLY_REASON = "this tool only works on Windows"


class RiskClassifier(Protocol):
    """Phase 8: an ML wrapper that may only *raise* a tier.

    ``min_tier`` returns 0..2 (Tier 3 is never delegated to a model) and the
    engine combines it with ``max()``, so a wrong or broken classifier can make
    JARVIS ask for confirmation but can never let an action through.  The
    optional arguments are the inputs docs/08 section 3 says the model needs;
    they default so a classifier written before them still satisfies the
    protocol.
    """

    def min_tier(self, step: Step, user_input: str = "", tainted: bool | None = None) -> int: ...


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
    #: Tainted output text from previous steps (docs/03 §8).  The engine
    #: checks these independently of ``step.depends_on_untrusted`` so an
    #: LLM omission cannot suppress the taint escalation.
    tainted_fragments: tuple[str, ...] = ()
    #: The user's original request (the graph's ``user_input``).  Only the
    #: classifier reads it: the wording of a command is a risk signal
    #: (docs/08 §3), but the rules must not change because the user phrased
    #: something differently, so no rule consults this field.
    user_input: str = ""

    # ── Stage 7 (Slice 4): optional browser/target authorization context ──
    # All four default to "absent", so every existing caller is byte-for-byte
    # unchanged.  When a browser step supplies them the engine additionally
    # enforces canonical-origin scope, control risk and password-field rules;
    # these checks can only raise a tier or block, never lower one.  The
    # resolved target is transient metadata about what was observed at runtime
    # - it is deliberately *not* part of ``action_hash`` (docs/03 §5 T4).
    #: Canonical origin the browser step targets, resolved by the policy gate.
    #: ``None`` means "not a browser step" and skips all new checks.
    origin: str | None = None
    #: Approved canonical domains for this task's scope.  Empty means no
    #: approved domain scope, which fails closed for a browser action.
    allowed_domains: tuple[str, ...] = ()
    #: The resolved interaction target (browser DOM or UIA control), if any.
    target: targets.ResolvedTarget | None = None
    #: Whether a valid approved task scope currently authorizes browser
    #: actions.  Default ``False`` fails closed.
    scope_approved: bool = False


def _target_checks(ctx: PolicyContext) -> tuple[bool, list[str], int]:
    """Evaluate optional Stage-7 browser/target context (pure, deterministic).

    Returns ``(blocked, reasons, min_tier)``: ``blocked`` means the step must be
    refused (Tier 3) and ``min_tier`` is a floor the engine applies with
    ``max()``.  When no browser origin or target is present this is a no-op, so
    existing decisions are unchanged.  A browser step must present a valid
    canonical origin inside the approved domain scope with an approved task
    scope; a resolved target must have reliable identity, must not be a
    password field, and a payment/purchase control is blocked while other risky
    controls force a fresh confirmation even inside an approved scope.
    """
    target = ctx.target
    is_browser = ctx.origin is not None or (target is not None and target.kind == "browser")
    if not is_browser and target is None:
        return False, [], tiers.TIER_SAFE

    reasons: list[str] = []
    min_tier = tiers.TIER_SAFE

    if is_browser:
        raw_origin = ctx.origin
        if raw_origin is None and target is not None:
            raw_origin = target.origin
        canonical = rules.url_origin(raw_origin) if raw_origin else None
        if canonical is None:
            return True, ["browser action with a missing or invalid origin"], tiers.TIER_BLOCKED
        if not ctx.allowed_domains:
            return True, ["browser action without an approved domain scope"], tiers.TIER_BLOCKED
        if not rules.domain_allowed(canonical, list(ctx.allowed_domains)):
            return True, [f"browser origin not in approved scope: {canonical}"], tiers.TIER_BLOCKED
        if not ctx.scope_approved:
            return True, ["browser action without an approved task scope"], tiers.TIER_BLOCKED

    if target is not None:
        if targets.fingerprint(target) is None:
            return (
                True,
                ["interaction target has ambiguous or missing identity"],
                tiers.TIER_BLOCKED,
            )
        if rules.is_password_target(target):
            return True, ["refusing to interact with a password field"], tiers.TIER_BLOCKED
        control = rules.risky_control(target.name, target.role)
        if control == rules.CONTROL_PAYMENT:
            return True, ["payment/purchase control is blocked"], tiers.TIER_BLOCKED
        if control == rules.CONTROL_FRESH_CONFIRM:
            min_tier = max(min_tier, tiers.TIER_CONFIRM)
            reasons.append("risky control requires fresh confirmation")

    return False, reasons, min_tier


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

        # Platform gate: a ``windows_only`` tool is refused on a host that is not
        # Windows.  Deterministic and pure, and checked before any tier/hash
        # work because there is nothing to decide for a tool that cannot run
        # here.  On Windows this is a no-op, so every existing windows_only
        # tool behaves exactly as it did before.
        if spec.windows_only and not is_windows():
            return _blocked(
                step,
                [_WINDOWS_ONLY_REASON],
                f"{spec.describe(args)} (Windows only)",
            )

        tier = spec.base_tier
        reasons: list[str] = []
        resolved_paths: list[str] = []

        if rules.matches_blocked(spec, args):
            # Hard block: treat exactly like an unknown/invalid step - Tier 3,
            # recorded reason, never reachable by any confirmation path.
            return _blocked(
                step,
                ["this tool/action is hard-blocked by policy"],
                spec.describe(args),
            )

        # Path rules: apply to every declared path argument.  Every resolved
        # path is recorded so act() can re-check nothing changed after the
        # confirmation (TOCTOU, docs/03 section 5 T4).  Every rule failure is a
        # hard block (Tier 3) with its reason recorded on the Decision.
        roots = paths.roots_from_settings(ctx.settings)
        for name, raw in spec.path_arg_values(args):
            try:
                resolved = paths.resolve_safe(raw)
            except paths.PathError as exc:
                return _blocked(step, [f"bad path in argument {name!r}: {exc}"], str(exc))
            if paths.is_protected(resolved):
                return _blocked(step, [f"path is protected: {resolved}"], str(resolved))
            if not paths.within_any_root(resolved, roots):
                return _blocked(
                    step,
                    [f"path is outside the allowed folders: {resolved}"],
                    f"outside allowed folders: {resolved}",
                )
            if spec.name == "delete_path" and paths.covers_any_root(resolved, roots):
                return _blocked(
                    step,
                    [f"refusing to delete an allowed folder (or an ancestor of one): {resolved}"],
                    f"refusing to delete an allowed folder: {resolved}",
                )
            resolved_paths.append(str(resolved))
            tier = max(tier, rules.path_tier(spec, str(resolved), args))

        tier = max(tier, rules.tool_tier(spec, args))

        # ── Taint escalation (docs/03 §8) ────────────────────────────
        # The LLM may set ``depends_on_untrusted`` but an LLM *omission*
        # must not suppress taint.  We therefore also perform a deterministic
        # substring overlap check against known tainted fragments from
        # previous results.  Either signal forces the escalation.
        llm_says_tainted = bool(step.depends_on_untrusted)
        engine_says_tainted = rules.args_overlap_taint(step.args, ctx.tainted_fragments)
        tainted = llm_says_tainted or engine_says_tainted
        if tainted and tier >= tiers.TIER_CONFIRM:
            tier = max(tier, tiers.TIER_CONFIRM)
            reasons.append("derived from untrusted content")

        # ── Optional risk classifier (Phase 8) ─────────────────────────
        # The last word on the tier, and the only `max()` a non-rule layer gets:
        # this can add a confirmation but can never remove one.
        if ctx.classifier is not None:
            classifier_tier = ctx.classifier.min_tier(step, ctx.user_input, tainted)
            if classifier_tier > tier:
                reasons.append(f"risk classifier raised this to tier {classifier_tier}")
            tier = max(tier, classifier_tier)

        # ── Optional Stage-7 browser/target checks (Slice 4) ──────────
        # Applied after every existing raise so they can only add a tier or
        # block.  With no browser/target context present the helper is a no-op
        # and the decision is identical to before.
        target_blocked, target_reasons, target_tier = _target_checks(ctx)
        if target_blocked:
            return _blocked(step, target_reasons, spec.describe(args))
        if target_tier > tiers.TIER_SAFE:
            tier = max(tier, target_tier)
        reasons.extend(target_reasons)

        allowed = tier < tiers.TIER_BLOCKED

        summary = spec.describe(args)
        overwrite = rules.overwrite_warning(spec, args)
        if overwrite:
            summary = f"{summary} ({overwrite})"

        typed = rules.typed_confirmation(spec, resolved_paths, ctx.settings)

        return Decision(
            step_id=step.id,
            tier=tier,
            allowed=allowed,
            needs_confirm=tier >= tiers.TIER_CONFIRM,
            needs_unlock=tier >= tiers.TIER_CONFIRM_UNLOCK,
            needs_typed_confirmation=typed,
            resolved_paths=resolved_paths,
            reasons=reasons,
            summary=summary,
            action_hash=rules.action_hash(spec, args),
            warn_untrusted=llm_says_tainted or engine_says_tainted,
        )
