"""Real LLM attempt telemetry (B8.5).

``TaskEventType.LLM_CALL`` used to have no producer, so the operational task
history could say *that* a task ran but never *who* answered it, how long it
took, or which provider had to be skipped.  This module supplies the missing
producer and nothing else:

* :class:`LLMAttemptRecord` - one real provider attempt, reduced to safe
  fields.  No prompt, response, tool argument, credential, screenshot, OCR text
  or file content can be represented by this type.
* :class:`LLMAttemptMeta` - the explicit **per-request** metadata a caller
  (``brain`` / ``replan``) hands to the composite client: task id, task class
  and the sink.  There is no mutable global "current task" anywhere, so two
  concurrent tasks can never write each other's attempts.
* :class:`TaskStoreAttemptSink` - writes ``LLM_CALL`` through the existing
  task-event store (schema unchanged: the allowed detail fields already exist
  in :mod:`jarvis.taskstore.privacy`).  Every failure is swallowed: losing
  observability must never change the answer the user gets.

The composite client records **the model that was actually attempted** (asked
from the role-specific client), plus its own fallback index, so a fallback is
visible as two records (index 0 failed, index 1 succeeded) rather than as one
successful call that hides the failure.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from jarvis.taskstore.models import TaskEventType
from jarvis.taskstore.vocabulary import TaskAttemptOutcome

__all__ = [
    "AttemptSink",
    "LLMAttemptMeta",
    "LLMAttemptRecord",
    "LLMContextStats",
    "TaskStoreAttemptSink",
    "attempt_kwargs",
    "build_attempt_meta",
    "new_attempt_id",
]

_LOG = logging.getLogger(__name__)

#: Closed mapping from the LLM-layer failure vocabulary
#: (:class:`jarvis.llm.failures.FailureCategory`) to the task store's own
#: closed failure vocabulary.  A provider's transport taxonomy is operational
#: detail; what the task history stores is the *safe* category.
_CATEGORY_MAP: dict[str, str] = {
    "timeout": "timeout",
    "cancelled": "cancelled",
}


class AttemptSink(Protocol):
    """Where one attempt record goes."""

    def record(self, attempt: LLMAttemptRecord) -> None: ...


def new_attempt_id() -> str:
    """A fresh canonical UUID for one attempt (correlation only, not a secret)."""
    return str(uuid.uuid4())


@dataclass(frozen=True, slots=True)
class LLMContextStats:
    """Safe, content-free context measurements for one model request (B9).

    Every field is a count or a boolean: no prompt, history, memory, tool
    argument or provider response can be represented here.
    """

    estimated_input_tokens: int = 0
    candidate_tools: int = 0
    schema_tools: int = 0
    memory_candidates: int = 0
    history_turns: int = 0
    trimmed: bool = False


@dataclass(frozen=True, slots=True)
class LLMAttemptRecord:
    """One real request sent to one provider/model for one role.

    ``token counts`` are only filled when the provider reported them; otherwise
    they stay ``0`` (never estimated).  ``failure_category`` is the task-store
    category, so the stored value stays inside its closed vocabulary.
    """

    task_id: str
    provider: str
    model: str
    role: str
    outcome: TaskAttemptOutcome
    attempt_id: str = ""
    task_class: str = ""
    fallback_index: int = 0
    duration_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    failure_category: str = ""
    cancelled: bool = False
    timed_out: bool = False
    context: LLMContextStats | None = None

    def detail(self) -> dict[str, Any]:
        """The allow-listed ``llm_call`` detail payload (safe fields only)."""
        detail: dict[str, Any] = {
            "provider": self.provider,
            "model": self.model,
            "role": self.role,
            "outcome": self.outcome.value,
            "duration_ms": max(int(self.duration_ms), 0),
            "fallback_index": max(int(self.fallback_index), 0),
            "cancelled": bool(self.cancelled),
            "timed_out": bool(self.timed_out),
        }
        if self.attempt_id:
            detail["attempt_id"] = self.attempt_id
        if self.task_class:
            detail["task_class"] = self.task_class
        if self.prompt_tokens or self.completion_tokens:
            detail["prompt_tokens"] = max(int(self.prompt_tokens), 0)
            detail["completion_tokens"] = max(int(self.completion_tokens), 0)
        if self.failure_category:
            detail["failure_category"] = self.failure_category
        if self.context is not None and self.context.estimated_input_tokens:
            detail["estimated_input_tokens"] = max(int(self.context.estimated_input_tokens), 0)
            detail["candidate_tools"] = max(int(self.context.candidate_tools), 0)
            detail["schema_tools"] = max(int(self.context.schema_tools), 0)
            detail["memory_candidates"] = max(int(self.context.memory_candidates), 0)
            detail["history_turns"] = max(int(self.context.history_turns), 0)
            detail["context_trimmed"] = bool(self.context.trimmed)
        return detail


@dataclass(frozen=True, slots=True)
class LLMAttemptMeta:
    """Per-request attempt metadata handed to the composite client."""

    task_id: str
    task_class: str = ""
    sink: AttemptSink | None = None
    context: LLMContextStats | None = None

    def emit(self, attempt: LLMAttemptRecord) -> None:
        """Forward one record; a sink failure is logged and never propagated."""
        if self.sink is None:
            return
        try:
            self.sink.record(attempt)
        except Exception as exc:  # noqa: BLE001 - telemetry must never break a call
            _LOG.debug("llm attempt not recorded: %s", exc)


def build_attempt_meta(
    client: Any,
    task_id: str,
    task_class: str,
    store: Any,
    *,
    context: LLMContextStats | None = None,
) -> LLMAttemptMeta | None:
    """Per-request attempt metadata, or ``None`` when nothing can be recorded.

    Shared by the graph nodes (brain/replan/specialist) and the tool layer
    (``web_answer``, the WhatsApp summariser, contact grounding) so every real
    provider attempt is attributed to the same explicit task id.  ``None`` is
    returned - never an empty id - when the client does not advertise
    ``supports_attempt_meta`` (a test double), when there is no task store, or
    when the task id is missing; the caller then invokes the model exactly as
    before.
    """
    if store is None or not task_id:
        return None
    if not getattr(client, "supports_attempt_meta", False):
        return None
    return LLMAttemptMeta(
        task_id=str(task_id),
        task_class=str(task_class or ""),
        sink=TaskStoreAttemptSink(store),
        context=context,
    )


def attempt_kwargs(client: Any, meta: LLMAttemptMeta | None) -> dict[str, Any]:
    """The ``attempt_meta`` keyword, only for clients that advertise support."""
    if meta is None or not getattr(client, "supports_attempt_meta", False):
        return {}
    return {"attempt_meta": meta}


class TaskStoreAttemptSink:
    """Write attempt records as ``LLM_CALL`` events on the task store."""

    def __init__(self, store: Any, logger: logging.Logger | None = None) -> None:
        self._store = store
        self._logger = logger or _LOG

    def record(self, attempt: LLMAttemptRecord) -> None:
        """Append one event, tolerating a read-only, closed or busy store."""
        try:
            self._store.append_event(
                attempt.task_id,
                TaskEventType.LLM_CALL,
                detail=attempt.detail(),
            )
        except Exception as exc:  # noqa: BLE001 - observability is best effort
            self._logger.debug("llm_call event not written: %s", exc)


def task_failure_category(failure_category: str | None, outcome: TaskAttemptOutcome) -> str:
    """Map an LLM failure category to the task store's closed vocabulary."""
    if outcome is TaskAttemptOutcome.TIMEOUT:
        return "timeout"
    if outcome is TaskAttemptOutcome.CANCELLED:
        return "cancelled"
    if outcome is TaskAttemptOutcome.SUCCESS:
        return ""
    return _CATEGORY_MAP.get(str(failure_category or ""), "llm")
