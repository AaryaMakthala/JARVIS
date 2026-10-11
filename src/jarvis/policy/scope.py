"""Transient browser/UIA task-scope authorization (D12/I4).

One "proceed" approves a *scope*, not individual clicks: a
:class:`ScopeGrant` binds the task, the plan hash, the daemon-issued
``confirmation_id``, the approved domain scope, an expiry and an action
budget.  Every action inside the scope still gets a fresh deterministic policy
check (the engine is the sole authority); this module only tracks *how much*
scope remains and *whether* the presented confirmation still matches.

Design constraints (owner decisions 1/5/7, docs/03):

* everything here is **in-memory and task-scoped** - grants and counters are
  never written to ``AgentState`` or the checkpoint store, so a daemon restart
  loses them and requires fresh authorization (fail closed);
* consumption is **atomic** (a lock guards the counter), so concurrent
  execution can never exceed the budget and a duplicate/replayed confirmation
  cannot mint extra budget;
* the consumed-action count is preserved across an in-session replan; a changed
  ``plan_hash`` requires fresh authorization with a **new** ``confirmation_id``.

Budget semantics: an action is charged at :meth:`ScopeRegistry.consume`, the
commitment point immediately before dispatch.  A pre-dispatch validation that
fails (stale fingerprint, wrong origin) never reaches ``consume`` and therefore
does not charge; once an action is charged it is **never refunded**, even if the
dispatch later fails, because execution may already have begun.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

__all__ = [
    "ScopeCheck",
    "ScopeError",
    "ScopeGrant",
    "ScopeRegistry",
]


class ScopeError(ValueError):
    """Raised when a scope cannot be issued (bad or duplicate input)."""


@dataclass(frozen=True)
class ScopeGrant:
    """One approved task scope.  Transient; never checkpointed."""

    task_id: str
    plan_hash: str
    confirmation_id: str
    allowed_domains: frozenset[str] = field(default_factory=frozenset)
    max_actions: int = 8
    issued_at: float = 0.0
    expires_at: float = 0.0

    def contains_domain(self, host: str) -> bool:
        """Whether ``host`` (already canonical) is inside the approved scope.

        An empty allowlist approves no domain, so the caller must fall back to
        its own (confirmation-gated) path rather than treating this as "all".
        """
        return str(host or "").strip().lower() in self.allowed_domains


@dataclass(frozen=True)
class ScopeCheck:
    """Outcome of :meth:`ScopeRegistry.validate` / :meth:`consume`."""

    ok: bool
    reason: str = ""
    remaining: int = 0
    grant: ScopeGrant | None = None


class ScopeRegistry:
    """In-memory, task-scoped scope grants with an atomic action budget.

    ``clock`` is injectable so expiry is testable without sleeping; it must
    return a monotonically non-decreasing value in the same domain used at
    issue time (default :func:`time.monotonic`).  A :class:`threading.Lock`
    makes consumption atomic across worker threads.

    A hard maximum of :data:`HARD_MAX_ACTIONS` (8) state-changing actions
    applies to a task across **all** replans.  Configuration may only lower the
    cap; ``max_actions`` overrides and later grants can never raise the task's
    remaining lifetime budget.
    """

    #: Absolute ceiling on state-changing actions per task (owner decision: 8).
    HARD_MAX_ACTIONS = 8

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        default_max_actions: int = HARD_MAX_ACTIONS,
    ) -> None:
        self._clock = clock
        requested_default = int(default_max_actions)
        if requested_default <= 0:
            raise ScopeError("default_max_actions must be positive")
        # Configuration may lower the cap, never raise the hard maximum.
        self._default_max_actions = min(requested_default, self.HARD_MAX_ACTIONS)
        self._lock = threading.Lock()
        self._grants: dict[str, ScopeGrant] = {}
        self._consumed: dict[str, int] = {}
        self._seen_ids: set[str] = set()
        self._superseded: set[str] = set()
        # Monotonically non-increasing lifetime cap per task (across replans).
        self._task_cap: dict[str, int] = {}

    # ── issuing ────────────────────────────────────────────────────────────

    def issue(
        self,
        task_id: str,
        *,
        plan_hash: str,
        confirmation_id: str,
        allowed_domains: Iterable[str] = (),
        max_actions: int | None = None,
        ttl_s: float = 120.0,
    ) -> ScopeGrant:
        """Create the current grant for ``task_id``.

        Supersedes this task's previous grant (its confirmation id can no longer
        authorize anything) but **never resets** the task's consumed counter, so
        a replan inherits the actions already spent.  The task's lifetime cap is
        monotonically non-increasing: a reauthorization may lower it but can
        never raise the remaining budget above what the task already had.  A
        duplicate, unknown or already-superseded ``confirmation_id`` is rejected.
        """
        task = str(task_id or "").strip()
        plan = str(plan_hash or "").strip()
        cid = str(confirmation_id or "").strip()
        if not task:
            raise ScopeError("task_id is required")
        if not plan:
            raise ScopeError("plan_hash is required")
        if not cid:
            raise ScopeError("confirmation_id is required")
        requested = self._default_max_actions if max_actions is None else int(max_actions)
        if requested <= 0:
            raise ScopeError("max_actions must be positive")
        # The configured default is itself a ceiling: a grant override may only
        # lower the cap, and nothing may exceed the hard maximum.
        budget = min(requested, self._default_max_actions, self.HARD_MAX_ACTIONS)

        domains = frozenset(
            host for raw in allowed_domains if (host := str(raw or "").strip().lower())
        )
        with self._lock:
            if cid in self._seen_ids:
                raise ScopeError(f"confirmation_id {cid!r} has already been used")
            now = self._clock()
            # Lifetime cap can only shrink across replans/reauthorizations.
            budget = min(budget, self._task_cap.get(task, budget))
            self._task_cap[task] = budget
            previous = self._grants.get(task)
            if previous is not None:
                self._superseded.add(previous.confirmation_id)
            grant = ScopeGrant(
                task_id=task,
                plan_hash=plan,
                confirmation_id=cid,
                allowed_domains=domains,
                max_actions=budget,
                issued_at=now,
                expires_at=now + float(ttl_s),
            )
            self._grants[task] = grant
            self._seen_ids.add(cid)
            self._consumed.setdefault(task, 0)
            return grant

    # ── validating / consuming ─────────────────────────────────────────────

    def validate(
        self,
        task_id: str,
        *,
        plan_hash: str | None = None,
        confirmation_id: str | None = None,
        now: float | None = None,
    ) -> ScopeCheck:
        """Check the current grant without consuming budget (fail closed)."""
        with self._lock:
            return self._check_locked(task_id, plan_hash, confirmation_id, now)

    def consume(
        self,
        task_id: str,
        *,
        plan_hash: str,
        confirmation_id: str,
        now: float | None = None,
    ) -> ScopeCheck:
        """Atomically charge one action against the task budget.

        Fails closed (and charges nothing) when the grant is missing, expired,
        bundled to a different plan/confirmation, or out of budget.  On success
        the charge is final and is never refunded.
        """
        with self._lock:
            check = self._check_locked(task_id, plan_hash, confirmation_id, now)
            if not check.ok:
                return check
            task = str(task_id or "").strip()
            self._consumed[task] = self._consumed.get(task, 0) + 1
            grant = check.grant
            assert grant is not None  # _check_locked guarantees it on ok
            remaining = max(0, grant.max_actions - self._consumed[task])
            return ScopeCheck(True, "", remaining, grant)

    def _check_locked(
        self,
        task_id: str,
        plan_hash: str | None,
        confirmation_id: str | None,
        now: float | None,
    ) -> ScopeCheck:
        task = str(task_id or "").strip()
        grant = self._grants.get(task)
        if grant is None:
            return ScopeCheck(False, "no_scope")
        cid = None if confirmation_id is None else str(confirmation_id).strip()
        if cid is not None:
            if cid in self._superseded:
                return ScopeCheck(False, "superseded")
            if cid != grant.confirmation_id:
                return ScopeCheck(False, "confirmation_mismatch")
        if plan_hash is not None and str(plan_hash).strip() != grant.plan_hash:
            return ScopeCheck(False, "plan_mismatch")
        current = self._clock() if now is None else now
        if current >= grant.expires_at:
            return ScopeCheck(False, "expired")
        remaining = max(0, grant.max_actions - self._consumed.get(task, 0))
        if remaining <= 0:
            return ScopeCheck(False, "budget_exhausted", 0, grant)
        return ScopeCheck(True, "", remaining, grant)

    # ── introspection / lifecycle ──────────────────────────────────────────

    def active_grant(self, task_id: str) -> ScopeGrant | None:
        """The current grant for ``task_id``, or ``None``."""
        with self._lock:
            return self._grants.get(str(task_id or "").strip())

    def consumed(self, task_id: str) -> int:
        """Actions charged for ``task_id`` so far (survives replans)."""
        with self._lock:
            return self._consumed.get(str(task_id or "").strip(), 0)

    def remaining(self, task_id: str) -> int:
        """Actions still available in the current grant (0 if none/expired)."""
        with self._lock:
            task = str(task_id or "").strip()
            grant = self._grants.get(task)
            if grant is None or self._clock() >= grant.expires_at:
                return 0
            return max(0, grant.max_actions - self._consumed.get(task, 0))

    def revoke(self, task_id: str) -> None:
        """Drop a task's grant (its confirmation id can no longer authorize)."""
        with self._lock:
            task = str(task_id or "").strip()
            grant = self._grants.pop(task, None)
            if grant is not None:
                self._superseded.add(grant.confirmation_id)
