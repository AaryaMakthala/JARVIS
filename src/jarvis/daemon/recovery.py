"""Task-store recovery for interrupted daemon instances.

Only one daemon may run at a time, so a non-terminal task owned by a process
that is demonstrably gone cannot still be executing.  Recovery expires such
rows without replaying tools, resuming confirmations, or deleting checkpoints.
Unknown liveness and owner-less rows are deliberately left untouched.
"""

from __future__ import annotations

import logging
import os
import uuid

import psutil

from jarvis.taskstore.models import TaskOwner, TaskReconcileResult, TaskStoreError
from jarvis.taskstore.store import TaskStore

__all__ = ["current_process_create_time", "new_task_owner", "owner_is_alive", "reconcile_task_store"]

_LOG = logging.getLogger(__name__)
_CREATE_TIME_TOLERANCE_S = 1.0


def current_process_create_time() -> float | None:
    """Return this process's creation time, or ``None`` when unavailable."""
    try:
        return float(psutil.Process(os.getpid()).create_time())
    except (psutil.Error, OSError):
        return None


def new_task_owner() -> TaskOwner:
    """Mint the identity for one daemon serving instance."""
    return TaskOwner(
        instance_id=uuid.uuid4().hex,
        pid=os.getpid(),
        process_create_time=current_process_create_time(),
    )


def owner_is_alive(owner: TaskOwner) -> bool | None:
    """Return whether ``owner`` is still the same process.

    ``None`` means liveness or process age could not be verified safely.  In
    that case the caller must not expire the task.
    """
    try:
        if not psutil.pid_exists(owner.pid):
            return False
        process = psutil.Process(owner.pid)
        if process.status() == psutil.STATUS_ZOMBIE:
            return False
        actual = float(process.create_time())
    except psutil.NoSuchProcess:
        return False
    except (psutil.Error, OSError):
        return None
    if owner.process_create_time is None:
        return None
    return abs(actual - owner.process_create_time) <= _CREATE_TIME_TOLERANCE_S


def reconcile_task_store(
    store: TaskStore | None,
    owner: TaskOwner,
    *,
    logger: logging.Logger | None = None,
) -> TaskReconcileResult:
    """Expire stale tasks on startup, never raising into the daemon boot path."""
    log = logger or _LOG
    if store is None:
        return TaskReconcileResult()
    if not store.available or not store.writable:
        log.info("task recovery skipped: task store is unavailable or read-only")
        return TaskReconcileResult()
    try:
        result = store.reconcile_stale_tasks(owner, owner_is_alive)
    except (TaskStoreError, OSError, ValueError) as exc:
        log.warning("task recovery failed without replaying work: %s", exc)
        return TaskReconcileResult()
    if result.expired_count:
        log.warning(
            "task recovery expired %d interrupted task(s): %s",
            result.expired_count,
            ", ".join(result.expired_task_ids),
        )
    else:
        log.info("task recovery found no stale tasks")
    return result
