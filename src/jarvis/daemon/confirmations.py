"""Confirmation correlation (I3).

Every confirmation carries a unique ``confirmation_id`` **and** the ``plan_hash``
it belongs to **and** an expiry.  A "proceed" satisfies only the exact pending
confirmation: the id must be known, unexpired, un-superseded, unused, and both
hashes must match.  Anything else fails closed.

This registry is **daemon-transient and in-memory** — it is never written to
``AgentState`` or the checkpoint store, so the serde allowlist stays exactly
``{Plan, Step, Decision, StepResult, ReplanDecision}``.  The ``plan_hash`` is
computed on demand, never stored in the graph.

Expiry reuses the single existing timeout constant
(:data:`jarvis.daemon.task_runtime.CONFIRMATION_TIMEOUT_S`); there is no second
timeout here.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from jarvis.daemon.task_runtime import CONFIRMATION_TIMEOUT_S

__all__ = [
    "CheckResult",
    "ConfirmationRegistry",
    "PendingConfirmation",
    "plan_hash",
]


def plan_hash(action_hashes: list[str]) -> str:
    """sha256 over the **ordered** action hashes of a plan.

    Order-sensitive on purpose: reordering the steps is a different plan.
    """
    payload = "\n".join(str(h) for h in action_hashes).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class PendingConfirmation:
    """One outstanding confirmation.  Transient; never checkpointed."""

    confirmation_id: str
    task_id: str
    step_id: str
    tier: int
    action_hash: str
    plan_hash: str
    issued_at: float
    expires_at: float


@dataclass(frozen=True)
class CheckResult:
    """Outcome of :meth:`ConfirmationRegistry.check`.  ``ok`` is the only gate."""

    ok: bool
    reason: str = ""


class ConfirmationRegistry:
    """Tracks pending confirmations and validates an answer against the exact one.

    ``clock`` is injectable so tests can drive expiry without sleeping; it
    defaults to :func:`time.monotonic` (same domain as the issue time).
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._pending: dict[str, PendingConfirmation] = {}
        self._superseded: set[str] = set()
        self._used: set[str] = set()

    def issue(
        self,
        task_id: str,
        step_id: str,
        tier: int,
        action_hash: str,
        plan_hash: str,
    ) -> PendingConfirmation:
        """Create a pending confirmation, superseding this task's older ones."""
        self._supersede(str(task_id))
        now = self._clock()
        pending = PendingConfirmation(
            confirmation_id=uuid.uuid4().hex,
            task_id=str(task_id),
            step_id=str(step_id),
            tier=int(tier),
            action_hash=str(action_hash),
            plan_hash=str(plan_hash),
            issued_at=now,
            expires_at=now + CONFIRMATION_TIMEOUT_S,
        )
        self._pending[pending.confirmation_id] = pending
        return pending

    def check(self, confirmation_id: str, plan_hash: str, action_hash: str) -> CheckResult:
        """Validate an answer against the exact pending confirmation (fail closed)."""
        cid = str(confirmation_id or "")
        if not cid or cid not in self._pending:
            return CheckResult(False, "unknown_id")
        if cid in self._used:
            return CheckResult(False, "already_used")
        if cid in self._superseded:
            return CheckResult(False, "superseded")

        pending = self._pending[cid]
        if self._clock() >= pending.expires_at:
            return CheckResult(False, "expired")
        if str(plan_hash or "") != pending.plan_hash:
            return CheckResult(False, "plan_hash_mismatch")
        if str(action_hash or "") != pending.action_hash:
            return CheckResult(False, "action_hash_mismatch")
        return CheckResult(True, "")

    def resolve(self, confirmation_id: str) -> bool:
        """Mark a confirmation used (one-shot).  Returns whether it existed."""
        cid = str(confirmation_id or "")
        if cid in self._pending:
            self._used.add(cid)
            return True
        return False

    def supersede(self, task_id: str) -> None:
        """Invalidate every still-usable pending id of ``task_id``."""
        self._supersede(str(task_id))

    def _supersede(self, task_id: str) -> None:
        for cid, pending in self._pending.items():
            if pending.task_id == task_id and cid not in self._used:
                self._superseded.add(cid)
