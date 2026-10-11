"""Small helpers shared by agent nodes."""

from __future__ import annotations

import dataclasses
import secrets
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from jarvis.agent.state import StepResult
from jarvis.config import Settings
from jarvis.platform_guard import is_windows
from jarvis.policy import rules
from jarvis.policy.scope import ScopeRegistry
from jarvis.policy.targets import ResolvedTarget
from jarvis.tools.base import resolve_target

_SNIPPET = 4000


@dataclass(frozen=True)
class StepTarget:
    """Gate-time resolution outcome for one planned step.

    ``context`` is the :class:`~jarvis.policy.engine.PolicyContext` the gate must
    decide with: the base context exactly when the step involves no interaction
    target, otherwise the base context augmented with the resolved origin,
    approved-domain scope and resolved :class:`ResolvedTarget`.  ``target`` is
    the resolved interaction target (``None`` for target-less steps),
    ``host`` the canonical host for scope approval (browser steps only) and
    ``refused``/``reason`` mark a fail-closed resolution refusal that must halt
    the step without any confirmation.
    """

    context: Any
    target: ResolvedTarget | None = None
    host: str = ""
    refused: bool = False
    reason: str = ""


def validated_step_args(spec: Any, step: Any) -> Any | None:
    """Validate ``step.args`` against ``spec``'s args model, or ``None``.

    Invalid arguments are deliberately *not* a resolution refusal: the engine
    rejects them deterministically on its own, so the gate falls through to that
    normal refusal path instead of inventing a target error.
    """
    if spec is None or step is None:
        return None
    raw = getattr(step, "args", None)
    if not isinstance(raw, dict):
        return None
    try:
        return spec.args_model.model_validate(raw if raw is not None else {})
    except Exception:  # noqa: BLE001 - fail closed: treat as unvalidated
        return None


def plan_hash_for(plan: Any, pctx: Any, engine: Any) -> str:
    """Deterministic plan hash over every step's action hash (ordered).

    Computed with the engine's *target-less* decisions so the value is stable
    regardless of resolved targets: the action hash deliberately excludes the
    target and origin (docs/03 section 5 T4).  Falls back to ``""`` when there
    is no plan or no step (fail closed: nothing to bind a scope to).
    """
    from jarvis.agent.batch_approval import compute_all_decisions
    from jarvis.daemon.confirmations import plan_hash

    steps = list(getattr(plan, "steps", None) or [])
    if not steps:
        return ""
    decisions = compute_all_decisions(plan, pctx, engine)
    ordered = [str(getattr(decisions.get(s.id), "action_hash", "") or "") for s in steps]
    if any(not h for h in ordered):
        return ""
    return plan_hash(ordered)


def resolve_step_target(
    pctx: Any,
    *,
    step: Any,
    spec: Any | None,
    settings: Settings,
    tool_ctx: Any,
    scope_registry: ScopeRegistry | None,
    task_id: str,
    plan_hash_v: str,
) -> StepTarget:
    """Resolve and contextualise the current step's interaction target.

    * a step with no tool/resolver is passed through untouched (``refused``
      False, no target);
    * args that do not validate, or a ``windows_only`` tool on a non-Windows
      host, are left to the engine's own deterministic refusal (no target);
    * a resolver that yields ``None`` means "no target could be reliably
      observed" and is a fail-closed refusal;
    * a browser target yields a context whose ``allowed_domains`` is the static
      config **plus** the valid live scope-grant domains, with ``scope_approved``
      set exactly when the origin's host is in that union — so static allowed
      domains need no approval and a valid grant covers its whole host;
    * a UIA target carries no origin: only ``target`` is set on the context,
      keeping the engine's UIA path (no browser-origin checks) byte-identical.
    """
    if spec is None or spec.resolve is None:
        return StepTarget(context=pctx)
    args = validated_step_args(spec, step)
    if args is None:
        return StepTarget(context=pctx)
    if spec.windows_only and not is_windows():
        return StepTarget(context=pctx)

    target = resolve_target(spec, args, tool_ctx)
    if target is None:
        return StepTarget(
            context=pctx,
            refused=True,
            reason="interaction target could not be reliably observed",
        )

    if target.kind != "browser":
        return StepTarget(context=dataclasses.replace(pctx, target=target), target=target)

    host = _host_of(target.origin)
    allowed = _effective_allowed(settings, scope_registry, task_id, plan_hash_v)
    augmented = dataclasses.replace(
        pctx,
        origin=target.origin,
        allowed_domains=tuple(allowed),
        scope_approved=bool(host) and rules.domain_allowed(target.origin, list(allowed)),
        target=target,
    )
    return StepTarget(context=augmented, target=target, host=host)


def active_scope_cid(scope_registry: ScopeRegistry | None, task_id: str) -> str:
    """The current valid grant's confirmation id for ``task_id``, else ``""``.

    Shared by the policy gate (which records a target binding under it) and the
    act node (which recomputes it to prove the same scope authorized the
    action).  Never raises: bookkeeping must not crash execution.
    """
    if scope_registry is None or not task_id:
        return ""
    try:
        grant = scope_registry.active_grant(str(task_id))
    except Exception:  # noqa: BLE001 - fail closed, never crash
        return ""
    return str(getattr(grant, "confirmation_id", "") or "") if grant is not None else ""


def _host_of(origin: str) -> str:
    """Canonical host (lower-cased, trailing dot stripped) of an origin."""
    if not origin:
        return ""
    host = (urlsplit(str(origin)).hostname or "").lower().rstrip(".")
    return host


def _effective_allowed(
    settings: Settings,
    scope_registry: ScopeRegistry | None,
    task_id: str,
    plan_hash_v: str,
) -> list[str]:
    """Static ``[browser] allowed_domains`` plus valid live-grant domains.

    A grant only contributes when it currently validates against ``task_id``
    and ``plan_hash_v`` (expired, superseded, exhausted or mismatched budgets
    are excluded — fail closed), so an old approval can never widen the scope.
    """
    domains = [str(d) for d in (settings.browser.allowed_domains or [])]
    for domain in _valid_grant_domains(scope_registry, task_id, plan_hash_v):
        if domain not in domains:
            domains.append(domain)
    return domains


def _valid_grant_domains(
    scope_registry: ScopeRegistry | None, task_id: str, plan_hash_v: str
) -> list[str]:
    """Domains of the task's current, still-valid grant (else ``[]``)."""
    if scope_registry is None or not task_id or not plan_hash_v:
        return []
    grant = scope_registry.active_grant(task_id)
    if grant is None:
        return []
    check = scope_registry.validate(task_id, plan_hash=plan_hash_v)
    if not check.ok:
        return []
    return [d for d in (grant.allowed_domains or ()) if d]


def truncate(text: str, limit: int = _SNIPPET) -> str:
    """Truncate text to ``limit`` chars for storage in state (docs 4 KB rule)."""
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def new_task_id() -> str:
    """Compact, collision-resistant task id for checkpointer thread_id."""
    return secrets.token_hex(6)


def tainted_fragments(state: dict[str, Any]) -> tuple[str, ...]:
    """Output text of every previous step that was flagged as untrusted.

    Only ``StepResult`` objects with ``tainted=True`` contribute.  This is our
    own data (tool output), not LLM text, which is why the engine trusts it for
    the independent taint check in docs/03 section 8.
    """
    return tuple(
        r.output
        for r in state.get("results") or []
        if isinstance(r, StepResult) and r.tainted and r.output
    )


def policy_context_for(state: dict[str, Any], policy_ctx: Any) -> Any:
    """Return a PolicyContext enriched with the state the engine cannot see.

    Two things are added, both read-only and both from engine-owned data:

    * ``tainted_fragments`` - so the taint check does not rely on the LLM having
      set ``depends_on_untrusted``;
    * ``user_input`` - the original request, which the Phase 8 risk classifier
      needs (docs/08 section 3) and which no rule ever reads.

    The original context is returned unchanged when there is nothing to add, so
    the common path allocates nothing.
    """
    if policy_ctx is None:
        return None
    tainted = tainted_fragments(state)
    user_input = str(state.get("user_input") or "")
    if not tainted and not user_input:
        return policy_ctx
    return dataclasses.replace(policy_ctx, tainted_fragments=tainted, user_input=user_input)
