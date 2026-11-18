"""P0.1: confirmation and clarification events are written after a pause."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jarvis.agent.context import make_app_context
from jarvis.agent.graph import open_sqlite_checkpointer
from jarvis.agent.runner import resume_task, run_task
from jarvis.agent.schemas import ActionIntent
from jarvis.config import Settings
from jarvis.llm.client import FakeLLM
from jarvis.taskstore import TaskEventType, TaskStatus, TaskStore, open_task_store
from jarvis.taskstore.runtime import build_task_vocabulary
from support import (
    approve,
    brain_action,
    brain_clarify,
    brain_steps,
    make_spec,
    registry_with,
)


def _store(tmp_path: Path, registry: Any) -> TaskStore:
    settings = Settings()
    return open_task_store(
        tmp_path / "task_events.db",
        create=True,
        vocabulary=build_task_vocabulary(settings, registry),
    )


def _ctx(
    store: TaskStore,
    registry: Any,
    *,
    llm: Any,
) -> Any:
    return make_app_context(Settings(), llm=llm, registry=registry, tasks=store)


def _events(store: TaskStore, task_id: str, event_type: TaskEventType) -> list[Any]:
    return [event for event in store.events(task_id) if event.event_type is event_type]


def test_confirmation_event_is_written_once_and_resume_does_not_duplicate(
    tmp_path: Path,
) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    registry = registry_with(make_spec("fake_echo", base_tier=1, record=record))
    store = _store(tmp_path, registry)
    saver = open_sqlite_checkpointer(":memory:")
    ctx = _ctx(
        store,
        registry,
        llm=FakeLLM([brain_action("fake_echo", {"text": "hi"})]),
    )
    try:
        first = run_task(ctx, saver, "echo hi")
        assert first.interrupted is True
        assert first.interrupt_kind == "confirm"

        events = _events(store, first.task_id, TaskEventType.CONFIRMATION_REQUESTED)
        assert len(events) == 1
        assert events[0].detail == {
            "approval_state": "pending",
            "step_id": "s1",
            "tool": "fake_echo",
        }
        paused = store.get(first.task_id)
        assert paused is not None
        assert paused.status is TaskStatus.AWAITING_CONFIRMATION
        assert paused.approval_state == "pending"
        assert paused.active_step_id == "s1"
        assert paused.active_tool == "fake_echo"
        assert record == []

        done = resume_task(ctx, saver, first.task_id, approve(first.confirmation or {}))
        assert done.interrupted is False
        events = _events(store, first.task_id, TaskEventType.CONFIRMATION_REQUESTED)
        assert len(events) == 1
        finished = store.get(first.task_id)
        assert finished is not None
        assert finished.status is TaskStatus.COMPLETED
        assert finished.approval_state is None
        assert record == [("fake_echo", {"text": "hi"})]
    finally:
        saver.conn.close()
        store.close()


def test_batch_confirmation_records_the_first_eligible_step(tmp_path: Path) -> None:
    registry = registry_with(make_spec("fake_echo", base_tier=1, record=[]))
    store = _store(tmp_path, registry)
    saver = open_sqlite_checkpointer(":memory:")
    ctx = _ctx(
        store,
        registry,
        llm=FakeLLM(
            [
                brain_steps(
                    [
                        ActionIntent(
                            tool="fake_echo",
                            args={"text": "one"},
                            rationale="first echo",
                        ),
                        ActionIntent(
                            tool="fake_echo",
                            args={"text": "two"},
                            rationale="second echo",
                        ),
                    ]
                )
            ]
        ),
    )
    try:
        outcome = run_task(ctx, saver, "echo one then two")
        assert outcome.interrupted is True
        assert outcome.confirmation is not None
        assert outcome.confirmation["type"] == "plan_approval"
        events = _events(store, outcome.task_id, TaskEventType.CONFIRMATION_REQUESTED)
        assert len(events) == 1
        assert events[0].detail == {
            "approval_state": "pending",
            "step_id": "s1",
            "tool": "fake_echo",
        }
    finally:
        saver.conn.close()
        store.close()


def test_clarification_event_has_no_question_or_answer(tmp_path: Path) -> None:
    registry = registry_with(make_spec("fake_echo", base_tier=0, record=[]))
    store = _store(tmp_path, registry)
    saver = open_sqlite_checkpointer(":memory:")
    ctx = _ctx(
        store,
        registry,
        llm=FakeLLM(
            [
                brain_clarify("Which private file did you mean?"),
                brain_action("fake_echo", {"text": "hi"}),
            ]
        ),
    )
    try:
        first = run_task(ctx, saver, "open the file")
        assert first.interrupt_kind == "clarification"

        events = _events(store, first.task_id, TaskEventType.CLARIFICATION_REQUESTED)
        assert len(events) == 1
        assert events[0].detail == {"node": "clarify"}
        assert "private file" not in repr(events[0].detail)
        paused = store.get(first.task_id)
        assert paused is not None
        assert paused.status is TaskStatus.AWAITING_CLARIFICATION
        assert paused.active_step_id is None
        assert paused.active_tool is None

        done = resume_task(ctx, saver, first.task_id, "the report")
        assert done.interrupted is False
        events = _events(store, first.task_id, TaskEventType.CLARIFICATION_REQUESTED)
        assert len(events) == 1
    finally:
        saver.conn.close()
        store.close()


def test_event_write_failure_does_not_break_the_pause(tmp_path: Path) -> None:
    class _AppendFailingStore:
        available = True
        writable = True

        def __init__(self, real: TaskStore) -> None:
            self._real = real

        def __getattr__(self, name: str) -> Any:
            return getattr(self._real, name)

        def append_event(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("event store is broken")

    registry = registry_with(make_spec("fake_echo", base_tier=1, record=[]))
    real = _store(tmp_path, registry)
    store = _AppendFailingStore(real)
    saver = open_sqlite_checkpointer(":memory:")
    ctx = _ctx(
        store,
        registry,
        llm=FakeLLM([brain_action("fake_echo", {"text": "hi"})]),
    )
    try:
        outcome = run_task(ctx, saver, "echo hi")
        assert outcome.interrupted is True
        assert outcome.confirmation is not None
        snapshot = real.get(outcome.task_id)
        assert snapshot is not None
        assert snapshot.status is TaskStatus.AWAITING_CONFIRMATION
        assert _events(real, outcome.task_id, TaskEventType.CONFIRMATION_REQUESTED) == []
    finally:
        saver.conn.close()
        real.close()
