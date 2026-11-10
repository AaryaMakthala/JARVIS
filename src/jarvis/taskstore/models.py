"""Validated models and lifecycle rules for the task event store."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from .privacy import (
    safe_approval_state,
    safe_failure_category,
    safe_failure_code,
    safe_source,
    safe_step_id,
    safe_task_id,
    safe_tool,
    sanitize_detail,
)
from .vocabulary import TaskOutcome, TaskStoreVocabulary

T = TypeVar("T")


def utc_now() -> datetime:
    """Return the current timezone-aware UTC time."""
    return datetime.now(UTC)


class TaskStage(str, Enum):
    """Closed lifecycle stage vocabulary."""

    RECEIVED = "received"
    RECALLING = "recalling"
    PLANNING = "planning"
    VALIDATING = "validating"
    AUTHORIZING = "authorizing"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    REPLANNING = "replanning"
    CLARIFYING = "clarifying"
    RESPONDING = "responding"
    SAVING = "saving"
    DONE = "done"
    LEGACY = "legacy"


class TaskStatus(str, Enum):
    """Closed task status vocabulary."""

    QUEUED = "queued"
    RUNNING = "running"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    AWAITING_CLARIFICATION = "awaiting_clarification"
    COMPLETED = "completed"
    ANSWERED = "answered"
    PARTIAL = "partial"
    FAILED = "failed"
    HALTED = "halted"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class TaskEventType(str, Enum):
    """Event kinds accepted by the append-only timeline."""

    TASK_CREATED = "task_created"
    TASK_STARTED = "task_started"
    STAGE_CHANGED = "stage_changed"
    STATUS_CHANGED = "status_changed"
    PLAN_CREATED = "plan_created"
    LLM_CALL = "llm_call"
    NOTE = "note"
    WARNING = "warning"
    FAILURE = "failure"
    CONFIRMATION_REQUESTED = "confirmation_requested"
    CLARIFICATION_REQUESTED = "clarification_requested"
    TOOL_STARTED = "tool_started"
    TOOL_FINISHED = "tool_finished"
    FINISHED = "task_finished"


TERMINAL_STATUSES = frozenset(
    {
        TaskStatus.COMPLETED,
        TaskStatus.ANSWERED,
        TaskStatus.PARTIAL,
        TaskStatus.FAILED,
        TaskStatus.HALTED,
        TaskStatus.CANCELLED,
        TaskStatus.EXPIRED,
    }
)

WORK_STARTED_STATUSES = frozenset(
    {
        TaskStatus.RUNNING,
        TaskStatus.AWAITING_CONFIRMATION,
        TaskStatus.AWAITING_CLARIFICATION,
        TaskStatus.COMPLETED,
        TaskStatus.ANSWERED,
        TaskStatus.PARTIAL,
        TaskStatus.FAILED,
        TaskStatus.HALTED,
    }
)

TERMINAL_STAGES = frozenset({TaskStage.DONE})

LIFECYCLE_EVENT_TYPES = frozenset(
    {
        TaskEventType.TASK_CREATED,
        TaskEventType.TASK_STARTED,
        TaskEventType.STAGE_CHANGED,
        TaskEventType.STATUS_CHANGED,
        TaskEventType.FINISHED,
    }
)

OUTCOME_FOR_STATUS: dict[TaskStatus, TaskOutcome] = {
    TaskStatus.COMPLETED: TaskOutcome.SUCCESS,
    TaskStatus.ANSWERED: TaskOutcome.SUCCESS,
    TaskStatus.PARTIAL: TaskOutcome.PARTIAL,
    TaskStatus.FAILED: TaskOutcome.FAILED,
    TaskStatus.HALTED: TaskOutcome.BLOCKED,
    TaskStatus.CANCELLED: TaskOutcome.CANCELLED,
    TaskStatus.EXPIRED: TaskOutcome.TIMEOUT,
}

ALLOWED_STATUS_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.QUEUED: frozenset(
        {
            TaskStatus.RUNNING,
            TaskStatus.AWAITING_CONFIRMATION,
            TaskStatus.AWAITING_CLARIFICATION,
            TaskStatus.CANCELLED,
            TaskStatus.EXPIRED,
        }
    ),
    TaskStatus.RUNNING: frozenset(
        {
            TaskStatus.AWAITING_CONFIRMATION,
            TaskStatus.AWAITING_CLARIFICATION,
            TaskStatus.COMPLETED,
            TaskStatus.ANSWERED,
            TaskStatus.PARTIAL,
            TaskStatus.FAILED,
            TaskStatus.HALTED,
            TaskStatus.CANCELLED,
            TaskStatus.EXPIRED,
        }
    ),
    TaskStatus.AWAITING_CONFIRMATION: frozenset(
        {
            TaskStatus.RUNNING,
            TaskStatus.FAILED,
            TaskStatus.HALTED,
            TaskStatus.CANCELLED,
            TaskStatus.EXPIRED,
        }
    ),
    TaskStatus.AWAITING_CLARIFICATION: frozenset(
        {
            TaskStatus.RUNNING,
            TaskStatus.FAILED,
            TaskStatus.HALTED,
            TaskStatus.CANCELLED,
            TaskStatus.EXPIRED,
        }
    ),
    TaskStatus.COMPLETED: frozenset(),
    TaskStatus.ANSWERED: frozenset(),
    TaskStatus.PARTIAL: frozenset(),
    TaskStatus.FAILED: frozenset(),
    TaskStatus.HALTED: frozenset(),
    TaskStatus.CANCELLED: frozenset(),
    TaskStatus.EXPIRED: frozenset(),
}


class TaskStoreError(RuntimeError):
    """Base class for task-store errors."""


class TaskStoreUnavailable(TaskStoreError):
    """The store cannot be opened or used."""


class TaskStoreCorruption(TaskStoreError):
    """A persisted row does not satisfy the task-store schema or invariants."""


class TaskNotFound(TaskStoreError):
    """The requested task does not exist."""


class InvalidTaskTransition(TaskStoreError):
    """A requested status transition is not in the closed transition map."""


def validate_transition(current: TaskStatus, new: TaskStatus) -> None:
    """Raise when ``current`` may not become ``new``."""
    if current == new:
        return
    if new not in ALLOWED_STATUS_TRANSITIONS[current]:
        raise InvalidTaskTransition(f"cannot transition task from {current.value} to {new.value}")


class TaskEvent(BaseModel):
    """One append-only lifecycle event."""

    model_config = ConfigDict(extra="forbid")

    event_id: int | None = Field(default=None, ge=1)
    task_id: str
    seq: int | None = Field(default=None, ge=1)
    event_type: TaskEventType
    stage: TaskStage | None = None
    status: TaskStatus | None = None
    detail: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("task_id")
    @classmethod
    def _safe_task_id(cls, value: str) -> str:
        return safe_task_id(value)

    @field_validator("detail")
    @classmethod
    def _safe_detail(cls, value: dict[str, Any], info: ValidationInfo) -> dict[str, Any]:
        event_type = info.data.get("event_type")
        vocabulary = (info.context or {}).get("vocabulary")
        if event_type is None:
            raise ValueError("event_type is required before detail")
        if not isinstance(vocabulary, TaskStoreVocabulary):
            raise TypeError("task store vocabulary is required")
        context = info.context or {}
        strict = bool(context.get("strict_detail", False))
        return sanitize_detail(event_type, value, vocabulary, strict=strict)

    @model_validator(mode="after")
    def _consistent_lifecycle_state(self) -> TaskEvent:
        if self.stage is None or self.status is None:
            raise ValueError("event stage and status are required")
        if self.event_type is TaskEventType.TASK_CREATED:
            if self.stage is not TaskStage.RECEIVED or self.status is not TaskStatus.QUEUED:
                raise ValueError("task_created requires the received/queued projection")
        elif self.event_type is TaskEventType.TASK_STARTED:
            if self.status is not TaskStatus.RUNNING or self.stage in TERMINAL_STAGES:
                raise ValueError("task_started requires a running, non-terminal projection")
        elif self.event_type is TaskEventType.STAGE_CHANGED:
            if self.status in TERMINAL_STATUSES or self.stage in TERMINAL_STAGES:
                raise ValueError("stage_changed requires a non-terminal, non-done projection")
            if self.detail.get("stage") != self.stage.value:
                raise ValueError("stage_changed detail must match the event stage")
        elif self.event_type is TaskEventType.STATUS_CHANGED:
            if self.status in TERMINAL_STATUSES or self.stage in TERMINAL_STAGES:
                raise ValueError("status_changed requires a non-terminal, non-done projection")
            if self.detail.get("status") != self.status.value:
                raise ValueError("status_changed detail must match the event status")
        elif self.event_type is TaskEventType.FINISHED:
            if self.status not in TERMINAL_STATUSES or self.stage not in TERMINAL_STAGES:
                raise ValueError("task_finished requires a terminal/done projection")
            expected = OUTCOME_FOR_STATUS[self.status].value
            if self.detail.get("outcome") != expected:
                raise ValueError("task_finished detail outcome must match the terminal status")
        elif self.status in TERMINAL_STATUSES or self.stage in TERMINAL_STAGES:
            raise ValueError("non-lifecycle events require a non-terminal, non-done projection")
        return self


class TaskOwner(BaseModel):
    """Process identity that owns a non-terminal task row."""

    model_config = ConfigDict(extra="forbid")

    instance_id: str
    pid: int = Field(ge=1, le=2_147_483_647)
    process_create_time: float | None = Field(default=None, gt=0)

    @field_validator("instance_id")
    @classmethod
    def _safe_instance_id(cls, value: str) -> str:
        from .privacy import safe_owner_instance

        instance = safe_owner_instance(value)
        if instance is None:
            raise ValueError("instance_id must be 32 lowercase hex characters")
        return instance

    @field_validator("process_create_time")
    @classmethod
    def _finite_create_time(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("process_create_time must be finite")
        return value


class TaskReconcileResult(BaseModel):
    """Outcome of one bounded stale-task reconciliation pass."""

    model_config = ConfigDict(extra="forbid")

    expired_task_ids: tuple[str, ...] = ()
    skipped_active: int = Field(default=0, ge=0)
    skipped_unknown: int = Field(default=0, ge=0)
    skipped_legacy: int = Field(default=0, ge=0)

    @property
    def expired_count(self) -> int:
        """Number of tasks moved to the expired terminal state."""
        return len(self.expired_task_ids)


class TaskSnapshot(BaseModel):
    """Current projection of one task."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    source: str = "terminal"
    stage: TaskStage = TaskStage.RECEIVED
    status: TaskStatus = TaskStatus.QUEUED
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    updated_at: datetime
    version: int = Field(default=0, ge=0)
    retry_count: int = Field(default=0, ge=0)
    active_tool: str | None = None
    active_step_id: str | None = None
    approval_state: str | None = None
    failure_category: str | None = None
    failure_code: str | None = None
    owner_instance_id: str | None = None
    owner_pid: int | None = Field(default=None, ge=1, le=2_147_483_647)
    owner_process_create_time: float | None = Field(default=None, gt=0)

    @field_validator("task_id")
    @classmethod
    def _safe_task_id(cls, value: str) -> str:
        return safe_task_id(value)

    @field_validator("owner_instance_id")
    @classmethod
    def _safe_owner_instance_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        from .privacy import safe_owner_instance

        instance = safe_owner_instance(value)
        if instance is None:
            raise ValueError("owner_instance_id must be 32 lowercase hex characters")
        return instance

    @field_validator("owner_process_create_time")
    @classmethod
    def _finite_owner_create_time(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("owner_process_create_time must be finite")
        return value

    @property
    def owner_identity(self) -> TaskOwner | None:
        """Return the persisted process identity, if this row has one."""
        if self.owner_instance_id is None or self.owner_pid is None:
            return None
        return TaskOwner(
            instance_id=self.owner_instance_id,
            pid=self.owner_pid,
            process_create_time=self.owner_process_create_time,
        )

    @field_validator("source")
    @classmethod
    def _safe_source(cls, value: str) -> str:
        source = safe_source(value)
        if source is None:
            raise ValueError("source must be terminal, voice, or benchmark")
        return source

    @field_validator("active_tool")
    @classmethod
    def _safe_tool(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        vocabulary = (info.context or {}).get("vocabulary")
        if not isinstance(vocabulary, TaskStoreVocabulary):
            raise TypeError("task store vocabulary is required for active_tool")
        tool = safe_tool(value, vocabulary)
        if tool is None:
            raise ValueError("active_tool is not a registered tool")
        return tool

    @field_validator("active_step_id")
    @classmethod
    def _safe_step_id(cls, value: str | None) -> str | None:
        return None if value is None else safe_step_id(value)

    @field_validator("approval_state")
    @classmethod
    def _safe_approval_state(cls, value: str | None) -> str | None:
        if value is None:
            return None
        state = safe_approval_state(value)
        if state is None:
            raise ValueError("approval_state is not a supported approval state")
        return state

    @field_validator("failure_code")
    @classmethod
    def _safe_failure_code(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        vocabulary = (info.context or {}).get("vocabulary")
        if not isinstance(vocabulary, TaskStoreVocabulary):
            raise TypeError("task store vocabulary is required for failure_code")
        code = safe_failure_code(value, vocabulary)
        if code is None:
            raise ValueError("failure_code is not configured")
        return code

    @field_validator("failure_category")
    @classmethod
    def _safe_failure_category(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        vocabulary = (info.context or {}).get("vocabulary")
        if not isinstance(vocabulary, TaskStoreVocabulary):
            raise TypeError("task store vocabulary is required for failure_category")
        category = safe_failure_category(value, vocabulary)
        if category is None:
            raise ValueError("failure_category is not configured")
        return category

    @model_validator(mode="after")
    def _consistent_lifecycle_state(self) -> TaskSnapshot:
        if self.status in TERMINAL_STATUSES:
            if self.stage not in TERMINAL_STAGES:
                raise ValueError("terminal task must have the done stage")
            if self.finished_at is None:
                raise ValueError("terminal task must have finished_at")
            if self.active_tool is not None or self.active_step_id is not None:
                raise ValueError("terminal task must not have active work")
        else:
            if self.stage in TERMINAL_STAGES:
                raise ValueError("non-terminal task must not have the done stage")
            if self.finished_at is not None:
                raise ValueError("non-terminal task must not have finished_at")
        if self.status is TaskStatus.QUEUED and self.started_at is not None:
            raise ValueError("queued task must not have started_at")
        if self.status in WORK_STARTED_STATUSES and self.started_at is None:
            raise ValueError("started task must have started_at")
        if (self.owner_instance_id is None) != (self.owner_pid is None):
            raise ValueError("task owner instance and pid must be set together")
        return self


class TaskSummary(TaskSnapshot):
    """Compact task projection used by list views."""

    event_count: int = Field(default=0, ge=0)


@dataclass(frozen=True, slots=True)
class TaskStoreWriteResult(Generic[T]):
    """Explicit result of a write on a real or unavailable task store."""

    persisted: bool
    value: T | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.persisted and self.value is None:
            raise ValueError("a persisted write must include a value")
        if not self.persisted and self.value is not None:
            raise ValueError("an unpersisted write must not include a value")


class TaskStoreStatus(BaseModel):
    """Availability and schema state of a task store."""

    model_config = ConfigDict(extra="forbid")

    available: bool
    reason: str | None = None
    schema_version: int | None = None


class TaskStoreSummary(BaseModel):
    """Aggregate counts for a task store."""

    model_config = ConfigDict(extra="forbid")

    total: int = Field(default=0, ge=0)
    active: int = Field(default=0, ge=0)
    terminal: int = Field(default=0, ge=0)
    by_status: dict[str, int] = Field(default_factory=dict)


class TaskPruneResult(BaseModel):
    """Rows removed by one bounded prune call."""

    model_config = ConfigDict(extra="forbid")

    tasks_deleted: int = Field(default=0, ge=0)
    events_deleted: int = Field(default=0, ge=0)
