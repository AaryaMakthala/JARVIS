"""Task-scoped browsing grant (Phase 4).

One approval ("Allow JARVIS to browse <host> for this task?") covers a bounded
run of **Tier 1** browser steps on exactly one host.  It is deliberately
separate from :mod:`jarvis.policy.scope`:

* it binds **no plan hash** - the specialist loop builds a fresh one-step plan
  for every decision, so a plan hash would invalidate the grant on each
  iteration;
* it has its own 30-action / 600-second ceilings, while ``ScopeRegistry`` keeps
  its own (much smaller) hard maximum and is left untouched.

This module only answers two questions: "may this exact step proceed without a
fresh confirmation?" (:func:`coverage`) and "how much budget is left?"
(:class:`BrowseGrantRegistry`).  It never decides a tier and never lowers one:
the policy engine remains the sole authority, and ``act`` re-runs the same pure
checks immediately before dispatch.  Everything here is **in-memory and
task-scoped**, never checkpointed, so a daemon restart requires fresh
authorization.

Fail closed: any doubt or missing datum (no grant, revoked, expired, empty
budget, a different task, an unobserved click label, an unknown role, a
non-allowlisted tool, a host that does not match, a Tier != 1 step) means *not
covered*.  A step whose origin cannot be resolved at gate time (for example
``browser_search``, which derives its URL from the site catalogue inside the
tool) is therefore never covered and keeps its ordinary confirmation.
"""

from __future__ import annotations

import re
import secrets
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from jarvis.policy.rules import FRESH_CONFIRM_PHRASES, FRESH_CONFIRM_TOKENS

__all__ = [
    "BROWSE_TOOLS",
    "RISK_WORDS",
    "BrowseGrant",
    "BrowseGrantError",
    "BrowseGrantRegistry",
    "GrantCheck",
    "coverage",
    "prospective_coverage",
    "risky_label",
]

#: Tools a browsing grant may cover.  Closed on purpose: anything else keeps
#: the ordinary per-step confirmation path.
BROWSE_TOOLS = frozenset(
    {
        "browser_search",
        "browser_open_site",
        "browser_goto",
        "browser_back",
        "browser_forward",
        "browser_tabs",
        "browser_open_result",
        "browser_click",
        "browser_type",
        "browser_press_key",
    }
)

#: Whole-word, case-folded words that mean "this click does something the user
#: must approve every time" (money, publishing, deleting, signing in, ...).
#: Built from the policy module's fresh-confirm set so the two can never drift
#: apart: a word that needs its own confirmation is never browse-covered.
_MONEY_RISK_WORDS = (
    "buy",
    "purchase",
    "pay",
    "payment",
    "payments",
    "remove",
    "check out",
)
RISK_WORDS = tuple(
    dict.fromkeys(
        (*_MONEY_RISK_WORDS, *sorted(FRESH_CONFIRM_TOKENS), *FRESH_CONFIRM_PHRASES)
    )
)

_RISK_PATTERNS = tuple(
    (word, re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)) for word in RISK_WORDS
)

#: Roles that are a form submission (or a password), never covered by a grant.
_FORBIDDEN_CLICK_ROLES = frozenset({"password", "submit"})
#: Typing is only covered into a non-secret text field.
_TYPE_ROLES = frozenset({"textbox", "searchbox", "combobox"})
#: Enter is only covered after typing into a *search* field; a plain textbox
#: might be a one-field form whose Enter submits it.
_ENTER_ROLES = frozenset({"searchbox", "combobox"})
#: Keys that would submit a form; they need a preceding search-box typing step.
_SUBMIT_KEYS = frozenset({"enter", "return"})


class BrowseGrantError(ValueError):
    """Raised when a browsing grant cannot be issued (bad or duplicate input)."""


def _fold(text: object) -> str:
    """Case-fold and collapse whitespace for host/label comparisons."""
    return " ".join(str(text or "").casefold().split())


def risky_label(label: str) -> str:
    """Return the first risk word contained in ``label``, or ``""``.

    Matching is whole-word and case-insensitive, so "Buy now" matches ``buy``
    while "buyer" does not.  Multi-word entries ("check out") match as phrases.
    """
    folded = _fold(label)
    if not folded:
        return ""
    for word, pattern in _RISK_PATTERNS:
        if pattern.search(folded):
            return word
    return ""


@dataclass(frozen=True)
class BrowseGrant:
    """One approved "browse this host for this task" grant.  Transient."""

    task_id: str
    grant_id: str
    host: str
    tools: frozenset[str] = field(default_factory=lambda: BROWSE_TOOLS)
    max_actions: int = 30
    issued_at: float = 0.0
    expires_at: float = 0.0
    revoked: bool = False

    def covers_host(self, host: str) -> bool:
        """Whether ``host`` (canonical or raw) is this grant's single host."""
        candidate = _fold(host)
        return bool(candidate) and candidate == self.host

    def allows_tool(self, tool: str) -> bool:
        """Whether ``tool`` is inside this grant's allowlist."""
        return str(tool or "").strip() in self.tools


@dataclass(frozen=True)
class GrantCheck:
    """Outcome of :meth:`BrowseGrantRegistry.validate` / :meth:`charge`."""

    ok: bool
    reason: str = ""
    remaining: int = 0
    grant: BrowseGrant | None = None


class BrowseGrantRegistry:
    """In-memory, task-scoped browsing grants with an atomic action budget.

    ``clock`` is injectable so expiry is testable without sleeping.  Config may
    only lower the ceilings (:data:`HARD_MAX_ACTIONS` / 600 s); nothing may
    raise them.
    """

    HARD_MAX_ACTIONS = 30
    HARD_MAX_TTL_S = 600.0

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        enabled: bool = True,
        max_actions: int = 30,
        ttl_s: float = 600.0,
    ) -> None:
        self._clock = clock
        self._enabled = bool(enabled)
        self._lock = threading.Lock()
        self._grants: dict[str, BrowseGrant] = {}
        self._consumed: dict[str, int] = {}
        self._seen_ids: set[str] = set()
        self._task_cap: dict[str, int] = {}
        self._default_max_actions = max(1, min(int(max_actions), self.HARD_MAX_ACTIONS))
        self._default_ttl_s = max(1.0, min(float(ttl_s), self.HARD_MAX_TTL_S))

    @property
    def enabled(self) -> bool:
        """Whether new grants may be issued at all."""
        return self._enabled

    # ── issuing ────────────────────────────────────────────────────────────

    def issue(
        self,
        task_id: str,
        *,
        host: str,
        tools: Iterable[str] | None = None,
        max_actions: int | None = None,
        ttl_s: float | None = None,
        grant_id: str | None = None,
        now: float | None = None,
    ) -> BrowseGrant:
        """Mint the task's grant for exactly one host.

        A re-issue supersedes the previous grant but never resets the consumed
        counter, and the task's lifetime ceiling may only shrink - the same
        discipline ``ScopeRegistry`` uses, so a re-approval cannot buy extra
        actions.  A blank task/host, a disabled registry or a replayed grant id
        is rejected (:class:`BrowseGrantError`).
        """
        if not self._enabled:
            raise BrowseGrantError("browsing grants are disabled")
        task = str(task_id or "").strip()
        grant_host = _fold(host)
        if not task:
            raise BrowseGrantError("task_id is required")
        if not grant_host:
            raise BrowseGrantError("host is required")
        gid = str(grant_id or "").strip() or secrets.token_hex(8)
        requested = self._default_max_actions if max_actions is None else int(max_actions)
        if requested <= 0:
            raise BrowseGrantError("max_actions must be positive")
        budget = max(1, min(requested, self._default_max_actions, self.HARD_MAX_ACTIONS))
        ttl = self._default_ttl_s if ttl_s is None else float(ttl_s)
        if ttl <= 0:
            raise BrowseGrantError("ttl_s must be positive")
        ttl = min(ttl, self._default_ttl_s, self.HARD_MAX_TTL_S)
        allowed = BROWSE_TOOLS if tools is None else frozenset(tools) & BROWSE_TOOLS
        with self._lock:
            if gid in self._seen_ids:
                raise BrowseGrantError(f"grant_id {gid!r} has already been used")
            budget = min(budget, self._task_cap.get(task, budget))
            self._task_cap[task] = budget
            current = self._clock() if now is None else now
            grant = BrowseGrant(
                task_id=task,
                grant_id=gid,
                host=grant_host,
                tools=allowed,
                max_actions=budget,
                issued_at=current,
                expires_at=current + ttl,
            )
            self._grants[task] = grant
            self._consumed.setdefault(task, 0)
            self._seen_ids.add(gid)
            return grant

    # ── validating / charging ──────────────────────────────────────────────

    def validate(self, task_id: str, *, now: float | None = None) -> GrantCheck:
        """Check the task's current grant without consuming budget."""
        with self._lock:
            return self._check_locked(task_id, now)

    def charge(self, task_id: str, *, now: float | None = None) -> GrantCheck:
        """Atomically charge one covered action against the grant budget.

        Fails closed and charges nothing when the grant is missing, revoked,
        expired or exhausted.  A successful charge is final and never refunded.
        """
        with self._lock:
            check = self._check_locked(task_id, now)
            if not check.ok:
                return check
            task = str(task_id or "").strip()
            self._consumed[task] = self._consumed.get(task, 0) + 1
            grant = check.grant
            assert grant is not None  # _check_locked guarantees it on ok
            remaining = max(0, grant.max_actions - self._consumed[task])
            return GrantCheck(True, "", remaining, grant)

    def _check_locked(self, task_id: str, now: float | None) -> GrantCheck:
        if not self._enabled:
            return GrantCheck(False, "disabled")
        task = str(task_id or "").strip()
        grant = self._grants.get(task)
        if grant is None:
            return GrantCheck(False, "no_grant")
        if grant.revoked:
            return GrantCheck(False, "revoked")
        current = self._clock() if now is None else now
        if current >= grant.expires_at:
            return GrantCheck(False, "expired")
        remaining = max(0, grant.max_actions - self._consumed.get(task, 0))
        if remaining <= 0:
            return GrantCheck(False, "budget_exhausted", 0, grant)
        return GrantCheck(True, "", remaining, grant)

    # ── introspection / lifecycle ──────────────────────────────────────────

    def active_for_task(self, task_id: str) -> BrowseGrant | None:
        """The task's current, non-revoked grant, or ``None``."""
        with self._lock:
            grant = self._grants.get(str(task_id or "").strip())
        if grant is None or grant.revoked:
            return None
        return grant

    def consumed(self, task_id: str) -> int:
        """Actions charged for ``task_id`` so far."""
        with self._lock:
            return self._consumed.get(str(task_id or "").strip(), 0)

    def remaining(self, task_id: str) -> int:
        """Actions still available (0 when missing, revoked or expired)."""
        with self._lock:
            check = self._check_locked(task_id, None)
        return max(0, check.remaining) if check.ok else 0

    def revoke(self, task_id: str) -> None:
        """Revoke the task's grant; its id can never authorize anything again."""
        with self._lock:
            task = str(task_id or "").strip()
            grant = self._grants.get(task)
            if grant is None:
                return
            self._grants[task] = BrowseGrant(
                task_id=grant.task_id,
                grant_id=grant.grant_id,
                host=grant.host,
                tools=grant.tools,
                max_actions=grant.max_actions,
                issued_at=grant.issued_at,
                expires_at=grant.expires_at,
                revoked=True,
            )


def coverage(
    grant: BrowseGrant | None,
    *,
    task_id: str,
    tool: str,
    tier: int,
    host: str = "",
    label: str = "",
    role: str = "",
    is_password: bool = False,
    key: str = "",
    actions_used: int = 0,
    needs_unlock: bool = False,
    needs_typed_confirmation: bool = False,
    previous_tool: str = "",
    previous_role: str = "",
    now: float | None = None,
) -> tuple[bool, str]:
    """Whether one step is covered by ``grant`` (pure, deterministic).

    Every rule can only *withhold* coverage; passing them does not authorize
    anything by itself and never changes a tier.  ``(True, "")`` means "this
    step may skip its own confirmation and be marked pre-approved"; anything
    else is the fail-closed reason it may not.
    """
    if grant is None:
        return False, "no_grant"
    if grant.revoked:
        return False, "revoked"
    if str(task_id or "").strip() != grant.task_id:
        return False, "task_mismatch"
    tool_name = str(tool or "").strip()
    if tool_name not in BROWSE_TOOLS or not grant.allows_tool(tool_name):
        return False, "tool_not_allowed"
    # Only a Tier 1 browser step needs a grant; Tier 0 needs nothing, and Tier 2
    # (unlock / typed confirmation) and Tier 3 are never covered.
    if int(tier) != 1:
        return False, "tier_not_coverable"
    if needs_unlock or needs_typed_confirmation:
        return False, "tier_not_coverable"
    current = time.monotonic() if now is None else float(now)
    if current >= grant.expires_at:
        return False, "expired"
    if int(actions_used) >= grant.max_actions:
        return False, "budget_exhausted"
    if not host:
        return False, "host_missing"
    if not grant.covers_host(host):
        return False, "cross_domain"

    role_folded = _fold(role)
    if tool_name in ("browser_click", "browser_open_result"):
        if is_password or role_folded in _FORBIDDEN_CLICK_ROLES:
            return False, "target_not_allowed"
        if not _fold(label):
            return False, "unobserved_target"
        word = risky_label(label)
        if word:
            return False, f"risky_label:{word}"
    elif tool_name == "browser_type":
        if is_password or role_folded == "password":
            return False, "password_field"
        if role_folded not in _TYPE_ROLES:
            return False, "target_not_allowed"
    elif tool_name == "browser_press_key":
        pressed = _fold(key)
        if not pressed:
            return False, "unobserved_key"
        if pressed in _SUBMIT_KEYS and not (
            _fold(previous_tool) == "browser_type" and _fold(previous_role) in _ENTER_ROLES
        ):
            return False, "enter_after_non_search_type"
    return True, ""


def prospective_coverage(
    *,
    task_id: str,
    tool: str,
    tier: int,
    host: str = "",
    label: str = "",
    role: str = "",
    is_password: bool = False,
    key: str = "",
    needs_unlock: bool = False,
    needs_typed_confirmation: bool = False,
    previous_tool: str = "",
    previous_role: str = "",
    now: float | None = None,
) -> tuple[bool, str]:
    """Whether a *fresh* grant for ``host`` would cover this exact step.

    Pure and side-effect free: it mints no grant and changes no budget.  The
    gate uses it *before* asking for a host grant, so an ineligible action (for
    example a risky first click) keeps its ordinary per-action confirmation
    instead of triggering a browsing prompt.
    """
    task = str(task_id or "").strip()
    if not task:
        return False, "task_missing"
    current = time.monotonic() if now is None else float(now)
    preview = BrowseGrant(
        task_id=task,
        grant_id="",
        host=_fold(host),
        max_actions=BrowseGrantRegistry.HARD_MAX_ACTIONS,
        issued_at=current,
        expires_at=current + BrowseGrantRegistry.HARD_MAX_TTL_S,
    )
    return coverage(
        preview,
        task_id=task,
        tool=tool,
        tier=tier,
        host=host,
        label=label,
        role=role,
        is_password=is_password,
        key=key,
        actions_used=0,
        needs_unlock=needs_unlock,
        needs_typed_confirmation=needs_typed_confirmation,
        previous_tool=previous_tool,
        previous_role=previous_role,
        now=current,
    )
