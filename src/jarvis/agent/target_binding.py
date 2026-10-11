"""Transient target bindings tying a resolved element to a scope action.

Between the policy gate deciding an action and the act node executing it, the
graph is paused at an ``interrupt()`` for confirmation and re-run from the top
on resume.  A :class:`TargetBinding` records, in memory only, exactly which
element the gate authorized so the act node can re-observe the target, recompute
the same binding and prove nothing changed (owner decision 8, docs/03 I6/I7).

The authorization hash mirrors the existing ``plan_hash`` construction: a
``sha256`` over newline-joined canonical fields.  The bound fields are the
**unchanged** existing ``action_hash`` plus the target ``fingerprint``, the
canonical ``origin`` and the scope ``confirmation_id``.  The existing
``action_hash`` implementation and its semantics are not touched here.

Bindings are one-shot and task/step scoped.  They are never written to
``AgentState`` or ``_ALLOWED_MSGPACK_TYPES``; a daemon restart loses them.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

__all__ = [
    "BindingResult",
    "TargetBinding",
    "TargetBindingStore",
    "authorization_hash",
]


def authorization_hash(
    action_hash: str,
    fingerprint: str,
    origin: str,
    scope_confirmation_id: str,
) -> str:
    """Deterministic authorization hash over the reviewed canonical fields.

    ``sha256("\n".join([action_hash, fingerprint, origin, scope_confirmation_id]))``
    - the same shape as ``plan_hash`` - so both are obvious and independently
    reproducible.  No secrets and no user data are part of the input.
    """
    payload = "\n".join(
        [
            str(action_hash or ""),
            str(fingerprint or ""),
            str(origin or ""),
            str(scope_confirmation_id or ""),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TargetBinding:
    """A gate-issued, one-shot binding of an element to an authorized action."""

    task_id: str
    step_id: str
    action_hash: str
    fingerprint: str
    origin: str
    scope_confirmation_id: str
    authorization_hash: str
    created_at: float
    expires_at: float | None = None


@dataclass(frozen=True)
class BindingResult:
    """Outcome of :meth:`TargetBindingStore.consume`."""

    ok: bool
    reason: str = ""
    binding: TargetBinding | None = None


class TargetBindingStore:
    """Transient, in-memory, one-shot store of target bindings.

    ``clock`` is injectable for deterministic expiry tests.  A
    :class:`threading.Lock` keeps put/consume/discard atomic.
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._bindings: dict[tuple[str, str], TargetBinding] = {}

    def put(
        self,
        task_id: str,
        step_id: str,
        *,
        action_hash: str,
        fingerprint: str,
        origin: str,
        scope_confirmation_id: str,
        ttl_s: float | None = None,
    ) -> TargetBinding:
        """Store (or replace) the binding for ``(task_id, step_id)``.

        Replacing a previous binding for the same key supersedes it: the stale
        binding can no longer be consumed.
        """
        key = (str(task_id or "").strip(), str(step_id or "").strip())
        now = self._clock()
        expires_at = None if ttl_s is None else now + float(ttl_s)
        binding = TargetBinding(
            task_id=key[0],
            step_id=key[1],
            action_hash=str(action_hash or ""),
            fingerprint=str(fingerprint or ""),
            origin=str(origin or ""),
            scope_confirmation_id=str(scope_confirmation_id or ""),
            authorization_hash=authorization_hash(
                action_hash, fingerprint, origin, scope_confirmation_id
            ),
            created_at=now,
            expires_at=expires_at,
        )
        with self._lock:
            self._bindings[key] = binding
        return binding

    def consume(
        self,
        task_id: str,
        step_id: str,
        *,
        action_hash: str,
        fingerprint: str,
        origin: str,
        scope_confirmation_id: str,
    ) -> BindingResult:
        """Atomically verify and consume the binding (one-shot).

        Fails closed *and invalidates* the binding when the key is missing, the
        binding expired, or any recomputed field differs: a stale, duplicate or
        mismatched binding is destroyed on the spot, so the caller must obtain a
        fresh resolution and authorization before that target can be used again.
        """
        key = (str(task_id or "").strip(), str(step_id or "").strip())
        with self._lock:
            binding = self._bindings.get(key)
            if binding is None:
                return BindingResult(False, "no_binding")
            if binding.expires_at is not None and self._clock() >= binding.expires_at:
                del self._bindings[key]
                return BindingResult(False, "expired")
            expected = authorization_hash(action_hash, fingerprint, origin, scope_confirmation_id)
            if expected != binding.authorization_hash:
                del self._bindings[key]
                return BindingResult(False, "mismatch")
            del self._bindings[key]
            return BindingResult(True, "", binding)

    def get(self, task_id: str, step_id: str) -> TargetBinding | None:
        """Return the binding without consuming it, or ``None``."""
        key = (str(task_id or "").strip(), str(step_id or "").strip())
        with self._lock:
            return self._bindings.get(key)

    def discard(self, task_id: str, step_id: str) -> None:
        """Drop a binding (e.g. after a refusal) without consuming it."""
        key = (str(task_id or "").strip(), str(step_id or "").strip())
        with self._lock:
            self._bindings.pop(key, None)

    def clear_task(self, task_id: str) -> None:
        """Drop every binding for a task (e.g. on replan or task end)."""
        task = str(task_id or "").strip()
        with self._lock:
            for key in [k for k in self._bindings if k[0] == task]:
                del self._bindings[key]
