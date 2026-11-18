"""P0.3: cancellation persistence is centralized in the runner and fail-soft."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from jarvis.agent.context import make_app_context
from jarvis.agent.graph import open_sqlite_checkpointer
from jarvis.agent.runner import resume_task, run_task
from jarvis.config import Settings
from jarvis.llm.client import FakeLLM
from jarvis.taskstore import TaskEventType, TaskStatus, TaskStore, open_task_store
from jarvis.taskstore.runtime import build_task_vocabulary
from jarvis.tools.base import Cancelled, CancelToken
from support import approve, brain_action, make_spec, registry_with


def _store(tmp_path: Path, registry: Any) -> TaskStore:
    settings = Settings()
    return open_task_store(
        tmp_path / "task_events.db",
        create=True,
        vocabulary=build_task_vocabulary(settings, registry),
    )


def _ctx(store: TaskStore, registry: Any, *, llm: Any) -> Any:
    return make_app_context(Settings(), llm=llm, registry=registry, tasks=store)


def _finished_events(store: TaskStore, task_id: str) -> list[Any]:
    return [e for e in store.events(task_id) if e.event_type is TaskEventType.FINISHED]


def test_cancel_during_initial_execution_persists_terminal_cancelled(tmp_path: Path) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    registry = registry_with(make_spec("fake_echo", base_tier=0, record=record))
    store = _store(tmp_path, registry)
    saver = open_sqlite_checkpointer(":memory:")
    ctx = _ctx(store, registry, llm=FakeLLM([brain_action("fake_echo", {"text": "hi"})]))
    cancel = CancelToken()
    cancel.cancel()
    task_id = "00000000-0000-4000-8000-00000000c001"
    try:
        with pytest.raises(Cancelled):
            run_task(ctx, saver, "echo hi", thread_id=task_id, cancel=cancel)

        snapshot = store.get(task_id)
        assert snapshot is not None
        assert snapshot.status is TaskStatus.CANCELLED
        assert snapshot.stage.value == "done"
        assert snapshot.failure_category == "cancelled"
        assert snapshot.failure_code == "cancelled"
        finished = _finished_events(store, task_id)
        assert len(finished) == 1
        assert finished[0].detail.get("outcome") == "cancelled"
        assert record == []
    finally:
        saver.conn.close()
        store.close()


def test_cancel_during_resumed_execution_persists_terminal_cancelled(tmp_path: Path) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    registry = registry_with(make_spec("fake_echo", base_tier=1, record=record))
    store = _store(tmp_path, registry)
    saver = open_sqlite_checkpointer(":memory:")
    ctx = _ctx(store, registry, llm=FakeLLM([brain_action("fake_echo", {"text": "hi"})]))
    try:
        first = run_task(ctx, saver, "echo hi")
        assert first.interrupted is True
        assert first.interrupt_kind == "confirm"
        paused = store.get(first.task_id)
        assert paused is not None
        assert paused.status is TaskStatus.AWAITING_CONFIRMATION

        cancel = CancelToken()
        cancel.cancel()
        with pytest.raises(Cancelled):
            resume_task(ctx, saver, first.task_id, approve(first.confirmation or {}), cancel=cancel)

        snapshot = store.get(first.task_id)
        assert snapshot is not None
        assert snapshot.status is TaskStatus.CANCELLED
        assert snapshot.stage.value == "done"
        assert snapshot.failure_category == "cancelled"
        assert snapshot.failure_code == "cancelled"
        finished = _finished_events(store, first.task_id)
        assert len(finished) == 1
        assert record == []
    finally:
        saver.conn.close()
        store.close()


def test_cancel_persistence_failure_does_not_prevent_cancellation(tmp_path: Path) -> None:
    class _FailFinishStore:
        available = True
        writable = True

        def __init__(self, real: TaskStore) -> None:
            self._real = real

        def __getattr__(self, name: str) -> Any:
            return getattr(self._real, name)

        def finish(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("store finish broken")

    record: list[tuple[str, dict[str, Any]]] = []
    registry = registry_with(make_spec("fake_echo", base_tier=0, record=record))
    real = _store(tmp_path, registry)
    store = _FailFinishStore(real)
    saver = open_sqlite_checkpointer(":memory:")
    ctx = _ctx(store, registry, llm=FakeLLM([brain_action("fake_echo", {"text": "hi"})]))
    cancel = CancelToken()
    cancel.cancel()
    task_id = "00000000-0000-4000-8000-00000000c003"
    try:
        with pytest.raises(Cancelled):
            run_task(ctx, saver, "echo hi", thread_id=task_id, cancel=cancel)

        # Persistence failed, but cancellation still propagated and no tool ran.
        assert record == []
        snapshot = real.get(task_id)
        assert snapshot is None or snapshot.status is not TaskStatus.COMPLETED
    finally:
        saver.conn.close()
        real.close()
