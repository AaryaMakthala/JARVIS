"""Small helpers shared by agent nodes."""

from __future__ import annotations

import dataclasses
import secrets
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from jarvis.agent.state import StepResult
from jarvis.browser.sites import sites_from_settings
from jarvis.config import Settings
from jarvis.platform_guard import is_windows
from jarvis.policy import rules
from jarvis.policy.browse_grant import BROWSE_TOOLS, coverage, prospective_coverage
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
    browse_grants: Any | None = None,
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
    allowed = _effective_allowed(settings, scope_registry, task_id, plan_hash_v, browse_grants)
    augmented = dataclasses.replace(
        pctx,
        origin=target.origin,
        allowed_domains=tuple(allowed),
        scope_approved=bool(host) and rules.domain_allowed(target.origin, list(allowed)),
        target=target,
    )
    return StepTarget(context=augmented, target=target, host=host)


def browse_step_host(step: Any, settings: Settings) -> str:
    """Host a ``browser_search`` step will search, resolved in code.

    The model supplies only the site name, so the host is looked up in the
    SiteDirectory (defaults + config).  An unknown site returns ``""``, which
    fails closed (never covered by a grant).
    """
    if str(getattr(step, "tool", "") or "") != "browser_search":
        return ""
    args = getattr(step, "args", None)
    if not isinstance(args, dict):
        return ""
    try:
        site = sites_from_settings(settings).resolve(str(args.get("site") or ""))
    except Exception:  # noqa: BLE001 - a lookup failure fails closed
        return ""
    return site.host if site is not None else ""


#: Tools that only read the page.  They cannot move focus, so the loop's own
#: refresh step between a type and the next decision does not break the chain.
_PASSIVE_HISTORY_TOOLS = frozenset({"browser_read"})
#: Roles a preceding typing step must have had for Enter to be coverable
#: (mirrors the pure rule's ``_ENTER_ROLES``; a password field is never here).
_SEARCH_ROLES = frozenset({"searchbox", "combobox"})
#: How far back the typed-target marker is looked for (history is bounded anyway).
_HISTORY_LOOKBACK = 12


def _previous_typed_target(state: dict[str, Any]) -> tuple[str, str, str]:
    """``(tool, role, host)`` of the last step that typed into a search field.

    Walks back over passive page reads only: the loop's automatic refresh
    cannot move focus, so it does not break the chain, while any real action
    (click, navigation, search, another key press) does.  The marker is written
    by ``specialist._record_history`` and is ordinary state, so it is treated as
    untrusted input: unknown or malformed values yield an empty marker and the
    Enter rule then fails closed.
    """
    history = state.get("specialist_history")
    if not isinstance(history, list):
        return "", "", ""
    for entry in reversed(history[-_HISTORY_LOOKBACK:]):
        if not isinstance(entry, dict):
            return "", "", ""
        tool = str(entry.get("tool") or "").strip()
        if tool in _PASSIVE_HISTORY_TOOLS:
            continue
        if tool != "browser_type" or entry.get("ok") is not True:
            return "", "", ""
        role = str(entry.get("role") or "").casefold()
        host = str(entry.get("host") or "").casefold().rstrip(".")
        if role not in _SEARCH_ROLES or not host:
            return "", "", ""
        return "browser_type", role, host
    return "", "", ""


def _browse_grant_fields(
    state: dict[str, Any], ctx: Any, *, step: Any, resolved: Any
) -> dict[str, Any]:
    """The exact coverage inputs for one step, shared by every caller."""
    tool = str(getattr(step, "tool", "") or "")
    target = getattr(resolved, "target", None)
    previous_tool, previous_role, previous_host = _previous_typed_target(state)
    host = str(getattr(resolved, "host", "") or "") or browse_step_host(step, ctx.settings)
    if not host and tool == "browser_press_key":
        # Enter acts on whatever holds focus, so there is no target to resolve and
        # no URL in the step.  The host comes from the marker of the last typing
        # step instead - and coverage() still requires the task's grant to cover
        # exactly that host, so this never widens the approved scope.
        host = previous_host
    if tool == "browser_press_key" and previous_host != host:
        previous_tool, previous_role = "", ""
    args = getattr(step, "args", None)
    return {
        "tool": tool,
        "host": host,
        "label": str(getattr(target, "name", "") or "") if target is not None else "",
        "role": str(getattr(target, "role", "") or "") if target is not None else "",
        "is_password": bool(getattr(target, "is_password", False))
        if target is not None
        else False,
        "key": str(args.get("key") or "") if isinstance(args, dict) else "",
        "previous_tool": previous_tool,
        "previous_role": previous_role,
    }


def browse_grant_check(
    state: dict[str, Any], ctx: Any, *, step: Any, decision: Any, resolved: Any
) -> tuple[Any | None, str, str]:
    """``(grant, host, reason)`` when ``coverage`` covers this exact step.

    Shared by the policy gate and act so both run the identical pure rules
    (act re-observes the target first, so a changed element is refused).  The
    grant is returned only on success; any doubt yields ``None``.
    """
    registry = getattr(ctx, "browse_grants", None)
    if registry is None or not getattr(registry, "enabled", False):
        return None, "", "disabled"
    task_id = str(state.get("task_id") or "")
    if str(getattr(step, "tool", "") or "") not in BROWSE_TOOLS:
        return None, "", "tool_not_allowed"
    fields = _browse_grant_fields(state, ctx, step=step, resolved=resolved)
    ok, reason = coverage(
        registry.active_for_task(task_id),
        task_id=task_id,
        tier=int(getattr(decision, "tier", 0)),
        actions_used=registry.consumed(task_id),
        needs_unlock=bool(getattr(decision, "needs_unlock", False)),
        needs_typed_confirmation=bool(getattr(decision, "needs_typed_confirmation", False)),
        **fields,
    )
    return (registry.active_for_task(task_id) if ok else None), str(fields["host"]), reason


def browse_grant_eligible(
    state: dict[str, Any], ctx: Any, *, step: Any, decision: Any, resolved: Any
) -> tuple[bool, str]:
    """Whether a fresh browsing grant for this step's host would cover it.

    Pure preview for the gate's *before asking* decision: it issues no grant and
    records no binding.  If this is false the step keeps its ordinary action
    confirmation and never causes a browsing prompt.
    """
    registry = getattr(ctx, "browse_grants", None)
    if registry is None or not getattr(registry, "enabled", False):
        return False, "disabled"
    fields = _browse_grant_fields(state, ctx, step=step, resolved=resolved)
    return prospective_coverage(
        task_id=str(state.get("task_id") or ""),
        tier=int(getattr(decision, "tier", 0)),
        needs_unlock=bool(getattr(decision, "needs_unlock", False)),
        needs_typed_confirmation=bool(getattr(decision, "needs_typed_confirmation", False)),
        **fields,
    )


def specialist_active(state: dict[str, Any]) -> bool:
    """Whether this task is inside the (flagged) browser specialist loop.

    ``"idle"`` is the resting phase, so only "decide" and "observe" count.
    Every other task keeps the ordinary graph behaviour.
    """
    return str(state.get("specialist_phase") or "idle") not in ("", "idle")


def record_specialist_note(
    state: dict[str, Any],
    *,
    tool: str,
    note: str,
    ok: bool = False,
    args_summary: str = "",
) -> list[dict[str, Any]]:
    """Append one bounded entry to ``specialist_history`` and return the list.

    The shape matches what :mod:`jarvis.agent.nodes.specialist` writes, so the
    next decision sees a refusal exactly like a failed step: ``tool``,
    ``args-summary``, ``ok`` and a note trimmed to 200 characters.  The repeat
    guard compares ``(tool, args-summary)``, so the same refused action twice in
    a row still stops the loop.
    """
    history = list(state.get("specialist_history") or [])
    history.append(
        {
            "tool": str(tool or "")[:200],
            "args-summary": str(args_summary or "")[:200],
            "ok": bool(ok),
            "note": str(note or "")[:200],
        }
    )
    return history


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
    browse_grants: Any | None = None,
) -> list[str]:
    """Static ``[browser] allowed_domains`` plus valid live-grant domains.

    A grant only contributes when it currently validates against ``task_id``
    and ``plan_hash_v`` (expired, superseded, exhausted or mismatched budgets
    are excluded — fail closed), so an old approval can never widen the scope.
    """
    domains = [str(d) for d in (settings.browser.allowed_domains or [])]
    # A valid task-scoped browsing grant contributes exactly its one host, so the
    # engine's target checks pass for steps the user approved for this task.
    if browse_grants is not None:
        grant = browse_grants.active_for_task(task_id)
        if (
            grant is not None
            and browse_grants.validate(task_id).ok
            and grant.host not in domains
        ):
            domains.append(grant.host)
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
