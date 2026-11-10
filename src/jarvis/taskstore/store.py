"""Independent, durable task projection and append-only event timeline."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, Self, runtime_checkable

from pydantic import ValidationError

from . import schema
from .models import (
    LIFECYCLE_EVENT_TYPES,
    OUTCOME_FOR_STATUS,
    TERMINAL_STAGES,
    TERMINAL_STATUSES,
    WORK_STARTED_STATUSES,
    InvalidTaskTransition,
    TaskEvent,
    TaskEventType,
    TaskNotFound,
    TaskOwner,
    TaskPruneResult,
    TaskReconcileResult,
    TaskSnapshot,
    TaskStage,
    TaskStatus,
    TaskStoreCorruption,
    TaskStoreError,
    TaskStoreStatus,
    TaskStoreSummary,
    TaskStoreUnavailable,
    TaskStoreWriteResult,
    TaskSummary,
    utc_now,
    validate_transition,
)
from .privacy import (
    safe_failure_category,
    safe_failure_code,
    safe_outcome,
    safe_source,
    safe_task_id,
    sanitize_detail,
)
from .vocabulary import TaskOutcome, TaskStoreVocabulary

__all__ = [
    "NullTaskEventStore",
    "OwnerLiveness",
    "TaskStore",
    "open_task_store",
]

OwnerLiveness = Callable[[TaskOwner], bool | None]

_LOG = logging.getLogger(__name__)

MAX_RECENT_LIMIT = 200
DEFAULT_MAX_TASKS = 1000
DEFAULT_MAX_EVENTS_PER_TASK = 500
_UNAVAILABLE_REASON = "task_store_unavailable"
_REASON_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _iso(value: datetime | None) -> str | None:
    normalised = _as_utc(value)
    return None if normalised is None else normalised.isoformat(timespec="microseconds")


def _from_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(str(value))
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _normalise_reason(reason: str) -> str:
    if not isinstance(reason, str):
        raise TypeError("reason must be a string")
    text = reason.strip()
    if not _REASON_RE.fullmatch(text):
        raise ValueError("reason must be a bounded lowercase code")
    return text


@runtime_checkable
class TaskStore(Protocol):
    """Shared public contract for durable and no-op task stores."""

    @property
    def available(self) -> bool: ...

    @property
    def writable(self) -> bool: ...

    def status(self) -> TaskStoreStatus: ...

    def close(self) -> None: ...

    def ensure_task(
        self,
        task_id: str,
        *,
        source: str = "terminal",
        created_at: datetime | None = None,
    ) -> TaskStoreWriteResult[TaskSnapshot]: ...

    def append_event(
        self,
        task_id: str,
        event_type: TaskEventType,
        *,
        detail: Mapping[str, Any] | None = None,
        created_at: datetime | None = None,
    ) -> TaskStoreWriteResult[TaskEvent]: ...

    def update_stage(
        self, task_id: str, stage: TaskStage
    ) -> TaskStoreWriteResult[TaskSnapshot]: ...

    def update_status(
        self, task_id: str, status: TaskStatus
    ) -> TaskStoreWriteResult[TaskSnapshot]: ...

    def finish(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        failure_category: str | None = None,
        failure_code: str | None = None,
        detail: Mapping[str, Any] | None = None,
        created_at: datetime | None = None,
    ) -> TaskStoreWriteResult[TaskSnapshot]: ...

    def get(self, task_id: str) -> TaskSnapshot | None: ...

    def events(self, task_id: str, after_seq: int = 0) -> list[TaskEvent]: ...

    def recent(self, limit: int = 20) -> list[TaskSummary]: ...

    def summary(self) -> TaskStoreSummary: ...

    def reconcile_stale_tasks(
        self,
        owner: TaskOwner,
        is_owner_alive: OwnerLiveness,
    ) -> TaskReconcileResult: ...

    def prune(
        self,
        *,
        max_tasks: int = DEFAULT_MAX_TASKS,
        max_events_per_task: int = DEFAULT_MAX_EVENTS_PER_TASK,
    ) -> TaskPruneResult: ...


class _SqliteTaskEventStore:
    """Private SQLite implementation; open it through :func:`open_task_store`."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        path: str | Path,
        *,
        writable: bool,
        vocabulary: TaskStoreVocabulary,
        owner: TaskOwner | None = None,
    ) -> None:
        self._conn = conn
        self._path = str(path)
        self._writable = writable
        self._vocabulary = vocabulary
        self._owner = owner
        self._lock = threading.RLock()
        self._closed = False
        self._broken = False
        self._unavailable_reason: str | None = None
        schema.verify_connection(
            conn,
            read_only=not writable,
            memory=str(path) == ":memory:",
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        del exc_type, exc, tb
        self.close()

    @property
    def available(self) -> bool:
        """Whether this store is open and usable."""
        return not self._closed

    @property
    def writable(self) -> bool:
        """Whether writes are allowed."""
        return self._writable and not self._closed

    def status(self) -> TaskStoreStatus:
        """Return availability and schema state without opening anything new."""
        if self._closed:
            return TaskStoreStatus(
                available=False,
                reason=self._unavailable_reason or "task store is closed",
            )
        return TaskStoreStatus(
            available=True,
            schema_version=schema.user_version(self._conn),
        )

    def close(self) -> None:
        """Checkpoint and close the connection. Safe to call twice."""
        with self._lock:
            if self._closed:
                return
            if self._writable:
                try:
                    self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except sqlite3.Error:
                    _LOG.debug("taskstore: WAL checkpoint on close failed", exc_info=True)
            try:
                self._conn.close()
            finally:
                self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise TaskStoreUnavailable(self._unavailable_reason or "task store is closed")

    def _require_writable(self) -> None:
        if not self._writable or self._closed:
            raise TaskStoreUnavailable(
                self._unavailable_reason or "task store is read-only or closed"
            )

    def _discard_connection(self, reason: str) -> None:
        self._broken = True
        self._unavailable_reason = reason
        try:
            self._conn.close()
        except sqlite3.Error:
            _LOG.debug("taskstore: close after failure failed", exc_info=True)
        finally:
            self._closed = True

    @contextmanager
    def _transaction(self, *, immediate: bool) -> Iterator[None]:
        with self._lock:
            self._require_open()
            begin = "BEGIN IMMEDIATE" if immediate else "BEGIN"
            try:
                self._conn.execute(begin)
                yield
            except BaseException:
                try:
                    self._conn.rollback()
                except BaseException:  # noqa: BLE001
                    _LOG.debug("taskstore: rollback failed; discarding connection", exc_info=True)
                    self._discard_connection("task_store_rollback_failed")
                raise
            else:
                try:
                    self._conn.commit()
                except BaseException:
                    try:
                        if self._conn.in_transaction:
                            self._conn.rollback()
                    except BaseException:  # noqa: BLE001
                        _LOG.debug(
                            "taskstore: rollback after commit failure failed; discarding connection",
                            exc_info=True,
                        )
                        self._discard_connection("task_store_commit_recovery_failed")
                    raise

    def _get_locked(self, task_id: str) -> TaskSnapshot | None:
        row = self._conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        return None if row is None else self._row_to_snapshot(row)

    def _finished_detail_locked(self, task_id: str) -> dict[str, Any]:
        rows = self._conn.execute(
            """
            SELECT detail_json FROM task_events
            WHERE task_id = ? AND event_type = ?
            ORDER BY seq ASC
            """,
            (task_id, TaskEventType.FINISHED.value),
        ).fetchall()
        if len(rows) != 1:
            raise TaskStoreCorruption("terminal task must have exactly one task_finished event")
        try:
            raw_detail = json.loads(str(rows[0]["detail_json"]))
            if not isinstance(raw_detail, dict):
                raise TypeError("task_finished detail must be an object")
            return sanitize_detail(
                TaskEventType.FINISHED,
                raw_detail,
                self._vocabulary,
                strict=True,
            )
        except (TypeError, ValueError):
            raise TaskStoreCorruption("task_finished event is corrupt") from None

    def _row_to_snapshot(self, row: sqlite3.Row) -> TaskSnapshot:
        try:
            created_at = _from_iso(str(row["created_at"]))
            updated_at = _from_iso(str(row["updated_at"]))
            if created_at is None or updated_at is None:
                raise ValueError("task timestamps are required")
            return TaskSnapshot.model_validate(
                {
                    "task_id": str(row["task_id"]),
                    "source": str(row["source"]),
                    "stage": str(row["stage"]),
                    "status": str(row["status"]),
                    "created_at": created_at,
                    "started_at": _from_iso(row["started_at"]),
                    "finished_at": _from_iso(row["finished_at"]),
                    "updated_at": updated_at,
                    "version": int(row["version"]),
                    "retry_count": int(row["retry_count"]),
                    "active_tool": row["active_tool"],
                    "active_step_id": row["active_step_id"],
                    "approval_state": row["approval_state"],
                    "failure_category": row["failure_category"],
                    "failure_code": row["failure_code"],
                    "owner_instance_id": row["owner_instance"],
                    "owner_pid": row["owner_pid"],
                    "owner_process_create_time": row["owner_process_create_time"],
                },
                context={"vocabulary": self._vocabulary},
            )
        except (ValidationError, TypeError, ValueError, KeyError, IndexError, OverflowError):
            raise TaskStoreCorruption("task projection is corrupt") from None

    def _row_to_summary(self, row: sqlite3.Row) -> TaskSummary:
        try:
            snapshot = self._row_to_snapshot(row)
            return TaskSummary.model_validate(
                {**snapshot.model_dump(), "event_count": int(row["event_count"])},
                context={"vocabulary": self._vocabulary},
            )
        except TaskStoreCorruption:
            raise
        except (ValidationError, TypeError, ValueError, KeyError, IndexError, OverflowError):
            raise TaskStoreCorruption("task summary is corrupt") from None

    def _row_to_event(self, row: sqlite3.Row) -> TaskEvent:
        try:
            raw_detail = json.loads(str(row["detail_json"]))
            if not isinstance(raw_detail, dict):
                raise TypeError("event detail must be an object")
            created_at = _from_iso(str(row["created_at"]))
            if created_at is None:
                raise ValueError("event timestamp is required")
            return TaskEvent.model_validate(
                {
                    "event_id": int(row["event_id"]),
                    "task_id": str(row["task_id"]),
                    "seq": int(row["seq"]),
                    "event_type": str(row["event_type"]),
                    "stage": str(row["stage"]) if row["stage"] is not None else None,
                    "status": str(row["status"]) if row["status"] is not None else None,
                    "detail": raw_detail,
                    "created_at": created_at,
                },
                context={"vocabulary": self._vocabulary, "strict_detail": True},
            )
        except (ValidationError, TypeError, ValueError, KeyError, IndexError, OverflowError):
            raise TaskStoreCorruption("task event is corrupt") from None

    def _append_locked(
        self,
        task_id: str,
        event_type: TaskEventType,
        *,
        detail: Mapping[str, Any] | None,
        created_at: datetime,
        current: TaskSnapshot | None = None,
        allow_task_created: bool = False,
        terminal_status: TaskStatus | None = None,
    ) -> TaskEvent:
        snapshot = current if current is not None else self._get_locked(task_id)
        if snapshot is None:
            raise TaskNotFound(f"task {task_id} does not exist")
        if snapshot.status in TERMINAL_STATUSES:
            raise InvalidTaskTransition("cannot append to a terminal task")

        raw_detail: Mapping[str, Any] = detail if isinstance(detail, Mapping) else {}
        if "stage" in raw_detail and event_type is not TaskEventType.STAGE_CHANGED:
            raise InvalidTaskTransition("stage may only be changed by update_stage()")
        if "status" in raw_detail and event_type is not TaskEventType.STATUS_CHANGED:
            raise InvalidTaskTransition("status may only be changed by update_status()")

        if event_type is TaskEventType.TASK_CREATED:
            if not allow_task_created:
                raise InvalidTaskTransition("task_created may only be written by ensure_task()")
            if terminal_status is not None:
                raise ValueError("task_created cannot carry a terminal status")
            if snapshot.status is not TaskStatus.QUEUED or snapshot.stage is not TaskStage.RECEIVED:
                raise InvalidTaskTransition("task_created requires a queued task at received")
            stage_value = TaskStage.RECEIVED
            status_value = TaskStatus.QUEUED
        elif event_type is TaskEventType.TASK_STARTED:
            if terminal_status is not None:
                raise ValueError("task_started cannot carry a terminal status")
            if snapshot.status is not TaskStatus.QUEUED or snapshot.stage in TERMINAL_STAGES:
                raise InvalidTaskTransition("task_started requires a queued task")
            stage_value = snapshot.stage
            status_value = TaskStatus.RUNNING
        elif event_type is TaskEventType.STAGE_CHANGED:
            if terminal_status is not None:
                raise ValueError("stage_changed cannot carry a terminal status")
            try:
                new_stage = TaskStage(raw_detail.get("stage"))
            except (TypeError, ValueError) as exc:
                raise InvalidTaskTransition("stage_changed requires a valid stage") from exc
            if new_stage in TERMINAL_STAGES:
                raise InvalidTaskTransition("done stage may only be set by finish()")
            stage_value = new_stage
            status_value = snapshot.status
        elif event_type is TaskEventType.STATUS_CHANGED:
            if terminal_status is not None:
                raise ValueError("status_changed cannot carry a terminal status")
            try:
                new_status = TaskStatus(raw_detail.get("status"))
            except (TypeError, ValueError) as exc:
                raise InvalidTaskTransition("status_changed requires a valid status") from exc
            if new_status in TERMINAL_STATUSES:
                raise InvalidTaskTransition("terminal statuses may only be set by finish()")
            validate_transition(snapshot.status, new_status)
            stage_value = snapshot.stage
            status_value = new_status
        elif event_type is TaskEventType.FINISHED:
            if terminal_status is None or terminal_status not in TERMINAL_STATUSES:
                raise InvalidTaskTransition("task_finished requires a terminal status")
            stage_value = TaskStage.DONE
            status_value = terminal_status
        else:
            if terminal_status is not None:
                raise ValueError("only task_finished may carry a terminal status")
            stage_value = snapshot.stage
            status_value = snapshot.status

        started_at = snapshot.started_at
        finished_at = snapshot.finished_at
        if status_value in WORK_STARTED_STATUSES and started_at is None:
            started_at = created_at
        if status_value in TERMINAL_STATUSES and finished_at is None:
            finished_at = created_at

        clean_detail = sanitize_detail(event_type, raw_detail, self._vocabulary)
        if (
            event_type is TaskEventType.STAGE_CHANGED
            and clean_detail.get("stage") != stage_value.value
        ):
            raise InvalidTaskTransition("stage_changed detail must match the new stage")
        if (
            event_type is TaskEventType.STATUS_CHANGED
            and clean_detail.get("status") != status_value.value
        ):
            raise InvalidTaskTransition("status_changed detail must match the new status")
        if event_type is TaskEventType.FINISHED:
            expected_outcome = _outcome_for_status(status_value).value
            if clean_detail.get("outcome") != expected_outcome:
                raise InvalidTaskTransition("task_finished outcome must match the terminal status")

        active_tool = snapshot.active_tool
        active_step_id = snapshot.active_step_id
        approval_state = snapshot.approval_state
        failure_category = snapshot.failure_category
        failure_code = snapshot.failure_code
        retry_count = snapshot.retry_count

        if event_type is TaskEventType.TOOL_STARTED:
            active_tool = clean_detail["tool"]
            active_step_id = clean_detail["step_id"]
        elif event_type is TaskEventType.TOOL_FINISHED:
            active_tool = None
            active_step_id = None
        elif event_type is TaskEventType.CONFIRMATION_REQUESTED:
            approval_state = clean_detail["approval_state"]
            active_step_id = clean_detail["step_id"]
        elif event_type is TaskEventType.CLARIFICATION_REQUESTED:
            active_step_id = clean_detail["step_id"]
        elif event_type is TaskEventType.FAILURE:
            failure_category = clean_detail["failure_category"]
            failure_code = clean_detail["failure_code"]
        elif event_type is TaskEventType.NOTE:
            if clean_detail.get("retry_count") is not None:
                retry_count = int(clean_detail["retry_count"])
        elif event_type is TaskEventType.FINISHED:
            active_tool = None
            active_step_id = None
            approval_state = None
            failure_category = clean_detail.get("failure_category")
            failure_code = clean_detail.get("failure_code")

        seq_row = self._conn.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 FROM task_events WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        seq = int(seq_row[0]) if seq_row else 1
        version = snapshot.version + 1
        event_time = _as_utc(created_at) or utc_now()
        when = _iso(event_time)
        detail_json = json.dumps(clean_detail, sort_keys=True, separators=(",", ":"))

        event = TaskEvent.model_validate(
            {
                "task_id": task_id,
                "seq": seq,
                "event_type": event_type,
                "stage": stage_value,
                "status": status_value,
                "detail": clean_detail,
                "created_at": event_time,
            },
            context={"vocabulary": self._vocabulary},
        )
        cursor = self._conn.execute(
            """
            INSERT INTO task_events (
              task_id, seq, event_type, stage, status, detail_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_id,
                seq,
                event_type.value,
                stage_value.value,
                status_value.value,
                detail_json,
                when,
            ),
        )
        self._conn.execute(
            """
            UPDATE tasks
            SET stage = ?, status = ?, started_at = ?, finished_at = ?, updated_at = ?,
                version = ?, retry_count = ?, active_tool = ?, active_step_id = ?,
                approval_state = ?, failure_category = ?, failure_code = ?
            WHERE task_id = ?
            """,
            (
                stage_value.value,
                status_value.value,
                _iso(started_at),
                _iso(finished_at),
                when,
                version,
                retry_count,
                active_tool,
                active_step_id,
                approval_state,
                failure_category,
                failure_code,
                task_id,
            ),
        )
        event_id = int(cursor.lastrowid) if cursor.lastrowid is not None else None
        return event.model_copy(update={"event_id": event_id})

    def ensure_task(
        self,
        task_id: str,
        *,
        source: str = "terminal",
        created_at: datetime | None = None,
    ) -> TaskStoreWriteResult[TaskSnapshot]:
        """Create a queued task once and return the existing row on repeat calls."""
        self._require_writable()
        safe_id = safe_task_id(task_id)
        when = _as_utc(created_at) or utc_now()
        source_value = safe_source(source)
        if source_value is None:
            raise ValueError("source must be terminal, voice, or benchmark")
        with self._transaction(immediate=True):
            existing = self._get_locked(safe_id)
            if existing is not None:
                return TaskStoreWriteResult(persisted=True, value=existing)
            owner = self._owner
            self._conn.execute(
                """
                INSERT INTO tasks (
                  task_id, source, stage, status, created_at, started_at, finished_at,
                  updated_at, version, retry_count, active_tool, active_step_id,
                  approval_state, failure_category, failure_code,
                  owner_instance, owner_pid, owner_process_create_time
                ) VALUES (?, ?, ?, ?, ?, NULL, NULL, ?, 0, 0, NULL, NULL, NULL, NULL, NULL,
                          ?, ?, ?)
                """,
                (
                    safe_id,
                    source_value,
                    TaskStage.RECEIVED.value,
                    TaskStatus.QUEUED.value,
                    _iso(when),
                    _iso(when),
                    None if owner is None else owner.instance_id,
                    None if owner is None else owner.pid,
                    None if owner is None else owner.process_create_time,
                ),
            )
            current = self._get_locked(safe_id)
            self._append_locked(
                safe_id,
                TaskEventType.TASK_CREATED,
                detail={"source": source_value},
                created_at=when,
                current=current,
                allow_task_created=True,
            )
            created = self._get_locked(safe_id)
            if created is None:
                raise TaskStoreError("task creation did not persist")
            return TaskStoreWriteResult(persisted=True, value=created)

    def append_event(
        self,
        task_id: str,
        event_type: TaskEventType,
        *,
        detail: Mapping[str, Any] | None = None,
        created_at: datetime | None = None,
    ) -> TaskStoreWriteResult[TaskEvent]:
        """Append one non-lifecycle event and update the projection atomically."""
        self._require_writable()
        safe_id = safe_task_id(task_id)
        when = _as_utc(created_at) or utc_now()
        event_kind = TaskEventType(event_type)
        if event_kind in LIFECYCLE_EVENT_TYPES:
            raise InvalidTaskTransition(
                f"{event_kind.value} may only be written by its lifecycle method"
            )
        with self._transaction(immediate=True):
            event = self._append_locked(
                safe_id,
                event_kind,
                detail=detail,
                created_at=when,
            )
            return TaskStoreWriteResult(persisted=True, value=event)

    def update_stage(self, task_id: str, stage: TaskStage) -> TaskStoreWriteResult[TaskSnapshot]:
        """Set the task's current non-terminal stage."""
        self._require_writable()
        safe_id = safe_task_id(task_id)
        new_stage = TaskStage(stage)
        if new_stage in TERMINAL_STAGES:
            raise InvalidTaskTransition("done stage may only be set by finish()")
        with self._transaction(immediate=True):
            current = self._get_locked(safe_id)
            if current is None:
                raise TaskNotFound(f"task {safe_id} does not exist")
            if current.status in TERMINAL_STATUSES:
                raise InvalidTaskTransition("cannot change the stage of a terminal task")
            if current.stage == new_stage:
                return TaskStoreWriteResult(persisted=True, value=current)
            self._append_locked(
                safe_id,
                TaskEventType.STAGE_CHANGED,
                detail={"stage": new_stage.value},
                created_at=utc_now(),
                current=current,
            )
            updated = self._get_locked(safe_id)
            if updated is None:
                raise TaskStoreError("task stage update did not persist")
            return TaskStoreWriteResult(persisted=True, value=updated)

    def update_status(self, task_id: str, status: TaskStatus) -> TaskStoreWriteResult[TaskSnapshot]:
        """Set the task's current status through the closed transition map."""
        self._require_writable()
        safe_id = safe_task_id(task_id)
        new_status = TaskStatus(status)
        if new_status in TERMINAL_STATUSES:
            raise InvalidTaskTransition("terminal statuses may only be set by finish()")
        with self._transaction(immediate=True):
            current = self._get_locked(safe_id)
            if current is None:
                raise TaskNotFound(f"task {safe_id} does not exist")
            if current.status == new_status:
                return TaskStoreWriteResult(persisted=True, value=current)
            if current.status is TaskStatus.QUEUED and new_status is TaskStatus.RUNNING:
                event_type = TaskEventType.TASK_STARTED
                detail = None
            else:
                event_type = TaskEventType.STATUS_CHANGED
                detail = {"status": new_status.value}
            self._append_locked(
                safe_id,
                event_type,
                detail=detail,
                created_at=utc_now(),
                current=current,
            )
            updated = self._get_locked(safe_id)
            if updated is None:
                raise TaskStoreError("task status update did not persist")
            return TaskStoreWriteResult(persisted=True, value=updated)

    def finish(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        failure_category: str | None = None,
        failure_code: str | None = None,
        detail: Mapping[str, Any] | None = None,
        created_at: datetime | None = None,
    ) -> TaskStoreWriteResult[TaskSnapshot]:
        """Move a task to a terminal status without allowing a later overwrite."""
        self._require_writable()
        safe_id = safe_task_id(task_id)
        final_status = TaskStatus(status)
        if final_status not in TERMINAL_STATUSES:
            raise ValueError("finish requires a terminal status")
        when = _as_utc(created_at) or utc_now()
        payload = _normalise_finish_payload(
            final_status,
            failure_category,
            failure_code,
            detail,
            self._vocabulary,
        )
        with self._transaction(immediate=True):
            current = self._get_locked(safe_id)
            if current is None:
                raise TaskNotFound(f"task {safe_id} does not exist")
            if current.status in TERMINAL_STATUSES:
                if current.status != final_status:
                    raise InvalidTaskTransition(
                        f"task is already terminal as {current.status.value}"
                    )
                existing = self._finished_detail_locked(safe_id)
                if existing != payload:
                    raise InvalidTaskTransition("task is already finished with different metadata")
                return TaskStoreWriteResult(persisted=True, value=current)

            self._append_locked(
                safe_id,
                TaskEventType.FINISHED,
                detail=payload,
                created_at=when,
                current=current,
                terminal_status=final_status,
            )
            finished = self._get_locked(safe_id)
            if finished is None:
                raise TaskStoreError("task finish did not persist")
            return TaskStoreWriteResult(persisted=True, value=finished)

    def reconcile_stale_tasks(
        self,
        owner: TaskOwner,
        is_owner_alive: OwnerLiveness,
    ) -> TaskReconcileResult:
        """Expire non-terminal rows whose owning process is proven gone.

        Rows without ownership metadata (older schemas or non-daemon writers)
        are intentionally left alone: absence of proof is not proof of death.
        """
        self._require_writable()
        if not isinstance(owner, TaskOwner):
            raise TypeError("owner must be a TaskOwner")
        if not callable(is_owner_alive):
            raise TypeError("is_owner_alive must be callable")
        terminals = _terminal_values()
        placeholders = ", ".join("?" for _ in terminals)
        with self._lock:
            self._require_open()
            rows = self._conn.execute(
                f"""
                SELECT task_id FROM tasks
                WHERE status NOT IN ({placeholders})
                ORDER BY updated_at ASC, task_id ASC
                """,
                terminals,
            ).fetchall()
            candidates = [str(row["task_id"]) for row in rows]

        expired: list[str] = []
        skipped_active = 0
        skipped_unknown = 0
        skipped_legacy = 0
        for task_id in candidates:
            current = self.get(task_id)
            if current is None or current.status in TERMINAL_STATUSES:
                continue
            row_owner = current.owner_identity
            if row_owner is None:
                skipped_legacy += 1
                continue
            if row_owner.instance_id == owner.instance_id or (
                row_owner.pid == owner.pid
                and row_owner.process_create_time is not None
                and owner.process_create_time is not None
                and abs(row_owner.process_create_time - owner.process_create_time) <= 1.0
            ):
                skipped_active += 1
                continue
            try:
                alive = is_owner_alive(row_owner)
            except (OSError, RuntimeError, TypeError, ValueError):
                alive = None
                _LOG.debug("taskstore: owner liveness probe failed", exc_info=True)
            if alive is True:
                skipped_active += 1
                continue
            if alive is None:
                skipped_unknown += 1
                continue
            latest = self.get(task_id)
            if latest is None or latest.status in TERMINAL_STATUSES:
                continue
            if latest.owner_identity != row_owner:
                skipped_unknown += 1
                continue
            try:
                self.finish(
                    task_id,
                    TaskStatus.EXPIRED,
                    failure_category="timeout",
                    failure_code="expired",
                )
            except InvalidTaskTransition:
                continue
            expired.append(task_id)
        return TaskReconcileResult(
            expired_task_ids=tuple(expired),
            skipped_active=skipped_active,
            skipped_unknown=skipped_unknown,
            skipped_legacy=skipped_legacy,
        )

    def get(self, task_id: str) -> TaskSnapshot | None:
        """Return one task projection, or ``None`` when it does not exist."""
        with self._lock:
            self._require_open()
            return self._get_locked(safe_task_id(task_id))

    def events(self, task_id: str, after_seq: int = 0) -> list[TaskEvent]:
        """Return events in stable sequence order, optionally after a cursor."""
        with self._lock:
            self._require_open()
            safe_id = safe_task_id(task_id)
            if after_seq < 0:
                raise ValueError("after_seq must be non-negative")
            rows = self._conn.execute(
                """
                SELECT * FROM task_events
                WHERE task_id = ? AND seq > ?
                ORDER BY seq ASC
                """,
                (safe_id, int(after_seq)),
            ).fetchall()
            return [self._row_to_event(row) for row in rows]

    def recent(self, limit: int = 20) -> list[TaskSummary]:
        """Return the most recently updated tasks."""
        with self._lock:
            self._require_open()
            if limit < 0:
                raise ValueError("limit must be non-negative")
            count = min(int(limit), MAX_RECENT_LIMIT)
            if count == 0:
                return []
            rows = self._conn.execute(
                """
                SELECT t.*,
                  (SELECT COUNT(*) FROM task_events e WHERE e.task_id = t.task_id)
                    AS event_count
                FROM tasks t
                ORDER BY t.updated_at DESC, t.task_id DESC
                LIMIT ?
                """,
                (count,),
            ).fetchall()
            return [self._row_to_summary(row) for row in rows]

    def summary(self) -> TaskStoreSummary:
        """Return aggregate task counts."""
        with self._lock:
            self._require_open()
            rows = self._conn.execute("SELECT * FROM tasks").fetchall()
            by_status: dict[str, int] = {}
            for row in rows:
                snapshot = self._row_to_snapshot(row)
                by_status[snapshot.status.value] = by_status.get(snapshot.status.value, 0) + 1
            total = sum(by_status.values())
            terminal = sum(
                count for status, count in by_status.items() if status in _terminal_values()
            )
            return TaskStoreSummary(
                total=total,
                active=total - terminal,
                terminal=terminal,
                by_status=by_status,
            )

    def prune(
        self,
        *,
        max_tasks: int = DEFAULT_MAX_TASKS,
        max_events_per_task: int = DEFAULT_MAX_EVENTS_PER_TASK,
    ) -> TaskPruneResult:
        """Prune old terminal tasks and old terminal-task events only."""
        self._require_writable()
        if max_tasks < 0:
            raise ValueError("max_tasks must be non-negative")
        if max_events_per_task < 0:
            raise ValueError("max_events_per_task must be non-negative")
        terminals = _terminal_values()
        placeholders = ", ".join("?" for _ in terminals)
        tasks_deleted = 0
        events_deleted = 0
        with self._transaction(immediate=True):
            total = int(self._conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0])
            if max_tasks > 0 and total > max_tasks:
                rows = self._conn.execute(
                    f"""
                    SELECT task_id FROM tasks
                    WHERE status IN ({placeholders})
                    ORDER BY updated_at ASC, task_id ASC
                    LIMIT ?
                    """,
                    (*terminals, total - max_tasks),
                ).fetchall()
                ids = [str(row["task_id"]) for row in rows]
                if ids:
                    id_placeholders = ", ".join("?" for _ in ids)
                    cascade_row = self._conn.execute(
                        f"SELECT COUNT(*) FROM task_events WHERE task_id IN ({id_placeholders})",
                        ids,
                    ).fetchone()
                    self._conn.execute(
                        f"DELETE FROM tasks WHERE task_id IN ({id_placeholders})",
                        ids,
                    )
                    tasks_deleted = len(ids)
                    events_deleted += int(cascade_row[0]) if cascade_row else 0

            if max_events_per_task > 0:
                rows = self._conn.execute(
                    f"""
                    SELECT t.task_id, MAX(e.seq) AS max_seq, COUNT(e.seq) AS event_count
                    FROM tasks t
                    LEFT JOIN task_events e ON e.task_id = t.task_id
                    WHERE t.status IN ({placeholders})
                    GROUP BY t.task_id
                    """,
                    terminals,
                ).fetchall()
                for row in rows:
                    if int(row["event_count"]) <= max_events_per_task:
                        continue
                    cutoff = int(row["max_seq"]) - max_events_per_task
                    cursor = self._conn.execute(
                        "DELETE FROM task_events WHERE task_id = ? AND seq <= ?",
                        (str(row["task_id"]), cutoff),
                    )
                    events_deleted += int(cursor.rowcount or 0)
        return TaskPruneResult(tasks_deleted=tasks_deleted, events_deleted=events_deleted)


class NullTaskEventStore:
    """Fail-safe stand-in used when task history is intentionally unavailable."""

    def __init__(
        self,
        reason: str = _UNAVAILABLE_REASON,
        *,
        vocabulary: TaskStoreVocabulary,
    ) -> None:
        self.reason = _normalise_reason(reason)
        self._vocabulary = vocabulary

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        del exc_type, exc, tb
        self.close()

    @property
    def available(self) -> bool:
        """Always false."""
        return False

    @property
    def writable(self) -> bool:
        """Always false."""
        return False

    def status(self) -> TaskStoreStatus:
        """Return an explicit unavailable status."""
        return TaskStoreStatus(available=False, reason=self.reason)

    def close(self) -> None:
        """No-op."""

    def _write_result(self) -> TaskStoreWriteResult[Any]:
        return TaskStoreWriteResult(persisted=False, value=None, reason=self.reason)

    def ensure_task(
        self,
        task_id: str,
        *,
        source: str = "terminal",
        created_at: datetime | None = None,
    ) -> TaskStoreWriteResult[TaskSnapshot]:
        """Validate the call and report that nothing was persisted."""
        del created_at
        safe_task_id(task_id)
        if safe_source(source) is None:
            raise ValueError("source must be terminal, voice, or benchmark")
        return self._write_result()

    def append_event(
        self,
        task_id: str,
        event_type: TaskEventType,
        *,
        detail: Mapping[str, Any] | None = None,
        created_at: datetime | None = None,
    ) -> TaskStoreWriteResult[TaskEvent]:
        """Validate the call and report that nothing was persisted."""
        del created_at
        safe_task_id(task_id)
        event_kind = TaskEventType(event_type)
        if event_kind in LIFECYCLE_EVENT_TYPES:
            raise InvalidTaskTransition(
                f"{event_kind.value} may only be written by its lifecycle method"
            )
        if isinstance(detail, Mapping) and ("stage" in detail or "status" in detail):
            raise InvalidTaskTransition("stage and status may only be changed by lifecycle methods")
        sanitize_detail(event_kind, detail, self._vocabulary)
        return self._write_result()

    def update_stage(self, task_id: str, stage: TaskStage) -> TaskStoreWriteResult[TaskSnapshot]:
        """Validate the call and report that nothing was persisted."""
        safe_task_id(task_id)
        new_stage = TaskStage(stage)
        if new_stage in TERMINAL_STAGES:
            raise InvalidTaskTransition("done stage may only be set by finish()")
        return self._write_result()

    def update_status(self, task_id: str, status: TaskStatus) -> TaskStoreWriteResult[TaskSnapshot]:
        """Validate the call and report that nothing was persisted."""
        safe_task_id(task_id)
        status_value = TaskStatus(status)
        if status_value in TERMINAL_STATUSES:
            raise InvalidTaskTransition("terminal statuses may only be set by finish()")
        return self._write_result()

    def finish(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        failure_category: str | None = None,
        failure_code: str | None = None,
        detail: Mapping[str, Any] | None = None,
        created_at: datetime | None = None,
    ) -> TaskStoreWriteResult[TaskSnapshot]:
        """Validate the call and report that nothing was persisted."""
        del created_at
        safe_task_id(task_id)
        final_status = TaskStatus(status)
        if final_status not in TERMINAL_STATUSES:
            raise ValueError("finish requires a terminal status")
        _normalise_finish_payload(
            final_status,
            failure_category,
            failure_code,
            detail,
            self._vocabulary,
        )
        return self._write_result()

    def _unavailable(self) -> TaskStoreUnavailable:
        return TaskStoreUnavailable(self.reason)

    def get(self, task_id: str) -> TaskSnapshot:
        """Raise: an unavailable store must not look empty."""
        del task_id
        raise self._unavailable()

    def events(self, task_id: str, after_seq: int = 0) -> list[TaskEvent]:
        """Raise: an unavailable store must not look empty."""
        del task_id, after_seq
        raise self._unavailable()

    def recent(self, limit: int = 20) -> list[TaskSummary]:
        """Raise: an unavailable store must not look empty."""
        del limit
        raise self._unavailable()

    def summary(self) -> TaskStoreSummary:
        """Raise: an unavailable store must not look empty."""
        raise self._unavailable()

    def reconcile_stale_tasks(
        self,
        owner: TaskOwner,
        is_owner_alive: OwnerLiveness,
    ) -> TaskReconcileResult:
        """Report an empty reconciliation; an unavailable store owns no rows."""
        del owner, is_owner_alive
        return TaskReconcileResult()

    def prune(
        self,
        *,
        max_tasks: int = DEFAULT_MAX_TASKS,
        max_events_per_task: int = DEFAULT_MAX_EVENTS_PER_TASK,
    ) -> TaskPruneResult:
        """Raise: pruning requires a writable store."""
        del max_tasks, max_events_per_task
        raise self._unavailable()


def _normalise_finish_payload(
    status: TaskStatus,
    failure_category: str | None,
    failure_code: str | None,
    detail: Mapping[str, Any] | None,
    vocabulary: TaskStoreVocabulary,
) -> dict[str, Any]:
    if detail is not None and not isinstance(detail, Mapping):
        raise ValueError("task_finished detail must be a mapping")
    payload = dict(detail or {})
    expected_outcome = _outcome_for_status(status).value
    provided_category: str | None = None
    provided_code: str | None = None
    if "outcome" in payload and safe_outcome(payload["outcome"]) != expected_outcome:
        raise ValueError("task_finished outcome must match the terminal status")
    if "failure_category" in payload:
        provided_category = safe_failure_category(payload["failure_category"], vocabulary)
        if provided_category is None:
            raise ValueError("failure_category is not configured")
    if failure_category is not None:
        category = safe_failure_category(failure_category, vocabulary)
        if category is None:
            raise ValueError("failure_category is not configured")
        if "failure_category" in payload and provided_category != category:
            raise ValueError("failure_category conflicts with task_finished detail")
        payload["failure_category"] = category
    if "failure_code" in payload:
        provided_code = safe_failure_code(payload["failure_code"], vocabulary)
        if provided_code is None:
            raise ValueError("failure_code is not configured")
    if failure_code is not None:
        code = safe_failure_code(failure_code, vocabulary)
        if code is None:
            raise ValueError("failure_code is not configured")
        if "failure_code" in payload and provided_code != code:
            raise ValueError("failure_code conflicts with task_finished detail")
        payload["failure_code"] = code
    payload["outcome"] = expected_outcome
    try:
        return sanitize_detail(TaskEventType.FINISHED, payload, vocabulary, strict=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("task_finished detail is invalid") from exc


def _outcome_for_status(status: TaskStatus) -> TaskOutcome:
    try:
        return OUTCOME_FOR_STATUS[status]
    except KeyError as exc:
        raise ValueError("finish requires a terminal status") from exc


def _terminal_values() -> tuple[str, ...]:
    return tuple(sorted(status.value for status in TERMINAL_STATUSES))


def open_task_store(
    path: str | Path | None = None,
    *,
    create: bool = False,
    read_only: bool = False,
    vocabulary: TaskStoreVocabulary | None = None,
    owner: TaskOwner | None = None,
) -> TaskStore:
    """Open the task event store without ever creating it implicitly.

    ``create=True`` initializes a missing database/schema.  ``create=False``
    requires the file and schema to exist.  ``read_only=True`` never creates or
    migrates anything and forbids every write.  ``vocabulary`` is mandatory and
    comes from the trusted application configuration, never from stored data.
    """
    if vocabulary is None:
        raise ValueError("task store vocabulary is required")
    if not isinstance(vocabulary, TaskStoreVocabulary):
        raise TypeError("vocabulary must be a TaskStoreVocabulary")
    if create and read_only:
        raise ValueError("create and read_only are mutually exclusive")
    if path is None:
        from jarvis.config import task_events_db

        path = task_events_db()

    target = str(path)
    resolved: str | Path = target
    if target != ":memory:":
        resolved = Path(target).expanduser().resolve(strict=False)
        exists = resolved.is_file()
        if read_only and not exists:
            raise TaskStoreUnavailable("task store does not exist")
        if not read_only and not create and not exists:
            raise TaskStoreUnavailable("task store does not exist")

    conn: sqlite3.Connection | None = None
    try:
        conn = schema.connect(resolved, create=create, read_only=read_only)
        if read_only:
            schema.validate(conn, read_only=True)
        else:
            missing = schema.REQUIRED_TABLES - schema.table_names(conn)
            if missing:
                if not create:
                    raise schema.SchemaError("task store schema is not initialized")
                schema.initialize(conn)
            schema.validate(conn, read_only=False)
        return _SqliteTaskEventStore(
            conn,
            resolved,
            writable=not read_only,
            vocabulary=vocabulary,
            owner=owner,
        )
    except (sqlite3.Error, schema.SchemaError, OSError) as exc:
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error:
                _LOG.debug("taskstore: close after open failure failed", exc_info=True)
        raise TaskStoreUnavailable(str(exc) or exc.__class__.__name__) from exc
