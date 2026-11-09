from __future__ import annotations

import logging
import types
from pathlib import Path
from uuid import UUID

from jarvis.agent.context import make_app_context
from jarvis.agent.graph import open_sqlite_checkpointer
from jarvis.agent.nodes import telemetry
from jarvis.agent.nodes.util import new_task_id
from jarvis.agent.runner import run_task
from jarvis.agent.state import StepResult
from jarvis.config import Settings
from jarvis.daemon.server import _task_id
from jarvis.llm.client import FakeLLM
from jarvis.taskstore import (
    TaskEventType,
    TaskStage,
    TaskStatus,
    TaskStore,
    open_task_store,
)
from jarvis.taskstore.runtime import build_task_vocabulary
from support import brain_action, make_spec, registry_with

TASK = "00000000-0000-4000-8000-0000000000aa"


def _store(tmp_path: Path, registry: object) -> TaskStore:
    settings = Settings()
    return open_task_store(
        tmp_path / "task_events.db",
        create=True,
        vocabulary=build_task_vocabulary(settings, registry),
    )


def _ctx(tasks: TaskStore, *, dry_run: bool = False, llm: object | None = None):
    registry = registry_with(make_spec("fake_echo", base_tier=0, record=[]))
    return make_app_context(
        Settings(), llm=llm, registry=registry, tasks=tasks, dry_run=dry_run
    )


def _failed() -> StepResult:
    return StepResult(step_id="s1", ok=False, error="nope", verified=False)


def test_new_task_id_is_a_canonical_uuid() -> None:
    value = new_task_id()

    assert str(UUID(value)) == value


def test_daemon_replaces_non_uuid_and_keeps_valid_ids() -> None:
    assert _task_id(TASK) == TASK
    generated = _task_id("t1")
    assert generated != "t1"
    assert str(UUID(generated)) == generated
    minted = _task_id()
    assert str(UUID(minted)) == minted


def test_start_creates_task_and_moves_it_to_running(tmp_path: Path) -> None:
    store = _store(tmp_path, registry_with())
    ctx = _ctx(store)
    try:
        telemetry.record_started(
            {"task_id": TASK, "source": "terminal", "user_input": "do it"}, ctx
        )
        snapshot = store.get(TASK)
        events = store.events(TASK)
    finally:
        store.close()

    assert snapshot is not None
    assert snapshot.status is TaskStatus.RUNNING
    assert [event.event_type for event in events] == [
        TaskEventType.TASK_CREATED,
        TaskEventType.TASK_STARTED,
    ]


def test_stage_is_recorded_from_a_node_name(tmp_path: Path) -> None:
    store = _store(tmp_path, registry_with())
    ctx = _ctx(store)
    try:
        telemetry.record_started({"task_id": TASK, "source": "terminal"}, ctx)
        telemetry.record_stage({"task_id": TASK, "source": "terminal"}, ctx, "brain")
        snapshot = store.get(TASK)
        events = store.events(TASK)
    finally:
        store.close()

    assert snapshot is not None
    assert snapshot.stage is TaskStage.PLANNING
    assert any(
        event.event_type is TaskEventType.STAGE_CHANGED
        and event.detail.get("stage") == TaskStage.PLANNING.value
        for event in events
    )


def test_finish_records_honest_status_and_failure_code(tmp_path: Path) -> None:
    store = _store(tmp_path, registry_with())
    ctx = _ctx(store)
    try:
        telemetry.record_started({"task_id": TASK, "source": "terminal"}, ctx)
        telemetry.record_finished(
            {
                "task_id": TASK,
                "source": "terminal",
                "results": [_failed()],
                "halted_reason": "gave up",
                "replan_count": 2,
            },
            ctx,
        )
        snapshot = store.get(TASK)
        events = store.events(TASK)
    finally:
        store.close()

    assert snapshot is not None
    assert snapshot.status is TaskStatus.FAILED
    assert snapshot.stage is TaskStage.DONE
    assert snapshot.retry_count == 2
    finished = events[-1]
    assert finished.event_type is TaskEventType.FINISHED
    assert finished.detail.get("failure_code") == "failed"


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    store = _store(tmp_path, registry_with())
    ctx = _ctx(store, dry_run=True)
    try:
        telemetry.record_started({"task_id": TASK, "source": "terminal"}, ctx)
        assert store.get(TASK) is None
    finally:
        store.close()


def test_a_broken_store_never_raises_into_the_graph() -> None:
    class _Broken:
        writable = True

        def ensure_task(self, *args, **kwargs):
            raise RuntimeError("store is broken")

        def update_status(self, *args, **kwargs):
            raise RuntimeError("store is broken")

        def update_stage(self, *args, **kwargs):
            raise RuntimeError("store is broken")

        def finish(self, *args, **kwargs):
            raise RuntimeError("store is broken")

    ctx = types.SimpleNamespace(
        tasks=_Broken(), dry_run=False, logger=logging.getLogger("test")
    )

    telemetry.record_started({"task_id": TASK, "source": "terminal"}, ctx)
    telemetry.record_stage({"task_id": TASK, "source": "terminal"}, ctx, "act")
    telemetry.record_finished({"task_id": TASK, "source": "terminal"}, ctx)


def test_completed_graph_task_has_stage_timeline_and_no_user_text(tmp_path: Path) -> None:
    record: list[tuple[str, dict[str, object]]] = []
    registry = registry_with(make_spec("fake_echo", base_tier=0, record=record))
    store = _store(tmp_path, registry)
    saver = open_sqlite_checkpointer(":memory:")
    ctx = _ctx(
        store,
        llm=FakeLLM([brain_action("fake_echo", {"text": "hi"})]),
    )
    try:
        outcome = run_task(ctx, saver, "echo hi")
        snapshot = store.get(outcome.task_id)
        events = store.events(outcome.task_id)
    finally:
        saver.conn.close()
        store.close()

    assert snapshot is not None
    assert snapshot.status is TaskStatus.COMPLETED
    assert snapshot.stage is TaskStage.DONE
    kinds = [event.event_type for event in events]
    assert kinds[0] is TaskEventType.TASK_CREATED
    assert TaskEventType.TASK_STARTED in kinds
    assert kinds[-1] is TaskEventType.FINISHED
    stages = {
        event.detail.get("stage")
        for event in events
        if event.event_type is TaskEventType.STAGE_CHANGED
    }
    assert {"planning", "executing", "verifying", "responding"} <= stages
    assert all("echo hi" not in repr(event.detail) for event in events)
