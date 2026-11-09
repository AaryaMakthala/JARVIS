from __future__ import annotations

import inspect
import multiprocessing
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest

from jarvis import config
from jarvis.taskstore import (
    InvalidTaskTransition,
    NullTaskEventStore,
    TaskEventType,
    TaskStage,
    TaskStatus,
    TaskStore,
    TaskStoreCorruption,
    TaskStoreUnavailable,
    TaskStoreVocabulary,
    open_task_store,
    schema,
)

TASK_1 = "00000000-0000-4000-8000-000000000001"
TASK_2 = "00000000-0000-4000-8000-000000000002"
TASK_3 = "00000000-0000-4000-8000-000000000003"
TASK_4 = "00000000-0000-4000-8000-000000000004"
TASK_OLD = "00000000-0000-4000-8000-000000000010"
TASK_NEW = "00000000-0000-4000-8000-000000000011"
TASK_ACTIVE = "00000000-0000-4000-8000-000000000012"
TASK_MISSING = "00000000-0000-4000-8000-000000000099"
STEP_1 = "00000000-0000-4000-8000-000000000101"
STEP_2 = "00000000-0000-4000-8000-000000000102"
STEP_3 = "00000000-0000-4000-8000-000000000103"

FAILURE_CATEGORIES = (
    "policy",
    "validation",
    "tool",
    "llm",
    "timeout",
    "cancelled",
    "verification",
    "internal",
    "unknown",
)


def _vocab_data() -> tuple:
    providers = ("groq", "openrouter", "nvidia", "gemini")
    models = tuple((provider, "openai/gpt-oss-120b") for provider in providers)
    return (
        providers,
        models,
        ("browser_goto", "whatsapp_read_messages"),
        ("planner",),
        ("E_TOOL", "E_OTHER", "E_HALT"),
        FAILURE_CATEGORIES,
    )


def _vocab_from_data(data: tuple) -> TaskStoreVocabulary:
    providers, models, tools, nodes, codes, categories = data
    return TaskStoreVocabulary(
        providers=frozenset(providers),
        models=frozenset(models),
        tools=frozenset(tools),
        nodes=frozenset(nodes),
        failure_codes=frozenset(codes),
        failure_categories=frozenset(categories),
    )


VOCAB = _vocab_from_data(_vocab_data())


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "task_events.db"


@pytest.fixture
def store(db_path: Path):
    opened = open_task_store(db_path, create=True, vocabulary=VOCAB)
    try:
        yield opened
    finally:
        opened.close()


def _at(day: int) -> datetime:
    return datetime(2026, 1, day, tzinfo=UTC)


def _value(result):
    assert result.persisted is True
    assert result.value is not None
    return result.value


class _CommitFailingConnection:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self.fail_next_commit = True

    def __getattr__(self, name: str):
        return getattr(self._conn, name)

    def commit(self) -> None:
        if self.fail_next_commit:
            self.fail_next_commit = False
            raise sqlite3.OperationalError("simulated commit failure")
        self._conn.commit()


class _RollbackFailingConnection:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def __getattr__(self, name: str):
        return getattr(self._conn, name)

    def rollback(self) -> None:
        raise sqlite3.OperationalError("simulated rollback failure")


def _process_ensurer(
    db_path: str,
    task_id: str,
    start_event,
    vocab_data: tuple,
) -> None:
    vocab = _vocab_from_data(vocab_data)
    store = open_task_store(db_path, vocabulary=vocab)
    try:
        start_event.wait(10)
        store.ensure_task(task_id, source="voice")
    finally:
        store.close()


def _process_writer(
    db_path: str,
    task_id: str,
    offset: int,
    start_event,
    vocab_data: tuple,
) -> None:
    vocab = _vocab_from_data(vocab_data)
    store = open_task_store(db_path, vocabulary=vocab)
    try:
        start_event.wait(10)
        store.ensure_task(task_id, source="terminal")
        for count in range(10):
            store.append_event(
                task_id,
                TaskEventType.NOTE,
                detail={"event_code": "progress", "count": offset + count},
            )
    finally:
        store.close()


def test_schema_is_created_and_versioned(store, db_path: Path) -> None:
    assert store.available is True
    assert store.writable is True
    assert store.status().schema_version == schema.SCHEMA_VERSION
    assert schema.REQUIRED_TABLES <= schema.table_names(store._conn)
    assert schema.REQUIRED_INDEXES <= schema.index_names(store._conn)
    assert schema.user_version(store._conn) == schema.SCHEMA_VERSION
    assert "active_step_id" in schema.columns(store._conn, "tasks")


def test_missing_store_is_not_created(db_path: Path) -> None:
    missing = db_path.parent / "absent" / "task_events.db"
    with pytest.raises(TaskStoreUnavailable):
        open_task_store(missing, vocabulary=VOCAB)
    assert missing.exists() is False
    assert missing.parent.exists() is False


def test_open_requires_an_explicit_vocabulary(db_path: Path) -> None:
    with pytest.raises(ValueError):
        open_task_store(db_path, create=True)
    assert db_path.exists() is False


def test_default_path_uses_data_dir_without_creating(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("JARVIS_DATA_DIR", str(tmp_path))
    path = config.task_events_db()
    assert path == tmp_path / "task_events.db"
    assert path.exists() is False


def test_ensure_task_is_idempotent_and_queued(store) -> None:
    first = _value(store.ensure_task(TASK_1, source="terminal", created_at=_at(1)))
    second = _value(store.ensure_task(TASK_1, source="voice", created_at=_at(2)))

    assert first == second
    assert first.status is TaskStatus.QUEUED
    assert first.stage is TaskStage.RECEIVED
    assert first.source == "terminal"
    assert first.started_at is None
    assert first.finished_at is None
    assert first.active_step_id is None
    assert first.version == 1

    events = store.events(TASK_1)
    assert len(events) == 1
    assert events[0].event_type is TaskEventType.TASK_CREATED
    assert events[0].seq == 1
    assert events[0].detail == {"source": "terminal"}


def test_append_event_cannot_create_a_task(store) -> None:
    store.ensure_task(TASK_1)
    with pytest.raises(InvalidTaskTransition):
        store.append_event(TASK_1, TaskEventType.TASK_CREATED, detail={"source": "voice"})

    events = store.events(TASK_1)
    assert [event.event_type for event in events] == [TaskEventType.TASK_CREATED]
    assert len(events) == 1


def test_concurrent_ensure_creates_exactly_one_task_created_event(db_path: Path) -> None:
    first = open_task_store(db_path, create=True, vocabulary=VOCAB)
    second = open_task_store(db_path, vocabulary=VOCAB)
    barrier = threading.Barrier(2)

    def ensure(opened) -> None:
        barrier.wait(timeout=5)
        opened.ensure_task(TASK_2, source="terminal")

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(ensure, first), pool.submit(ensure, second)]
            for future in futures:
                future.result(timeout=10)

        events = first.events(TASK_2)
        assert sum(1 for event in events if event.event_type is TaskEventType.TASK_CREATED) == 1
        snapshot = first.get(TASK_2)
        assert snapshot is not None
        assert snapshot.version == 1
        assert len(events) == 1
    finally:
        first.close()
        second.close()


def test_process_level_concurrent_ensure_creates_exactly_one_event(db_path: Path) -> None:
    seed = open_task_store(db_path, create=True, vocabulary=VOCAB)
    seed.close()

    context = multiprocessing.get_context("spawn")
    start_event = context.Event()
    processes = [
        context.Process(
            target=_process_ensurer,
            args=(str(db_path), TASK_2, start_event, _vocab_data()),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    start_event.set()
    for process in processes:
        process.join(30)
        assert process.exitcode == 0

    store = open_task_store(db_path, vocabulary=VOCAB)
    try:
        events = store.events(TASK_2)
        assert sum(1 for event in events if event.event_type is TaskEventType.TASK_CREATED) == 1
        assert len(events) == 1
        snapshot = store.get(TASK_2)
        assert snapshot is not None
        assert snapshot.version == 1
    finally:
        store.close()


def test_lifecycle_updates_projection_and_ordered_events(store) -> None:
    store.ensure_task(TASK_1, created_at=_at(1))
    running = _value(store.update_status(TASK_1, TaskStatus.RUNNING))
    assert running.started_at is not None
    store.update_stage(TASK_1, TaskStage.PLANNING)
    store.update_status(TASK_1, TaskStatus.AWAITING_CONFIRMATION)
    store.update_status(TASK_1, TaskStatus.RUNNING)
    finished = _value(store.finish(TASK_1, TaskStatus.COMPLETED, created_at=_at(2)))

    assert finished.status is TaskStatus.COMPLETED
    assert finished.stage is TaskStage.DONE
    assert finished.started_at is not None
    assert finished.finished_at == _at(2)

    events = store.events(TASK_1)
    assert [event.seq for event in events] == [1, 2, 3, 4, 5, 6]
    assert [event.event_type for event in events] == [
        TaskEventType.TASK_CREATED,
        TaskEventType.TASK_STARTED,
        TaskEventType.STAGE_CHANGED,
        TaskEventType.STATUS_CHANGED,
        TaskEventType.STATUS_CHANGED,
        TaskEventType.FINISHED,
    ]
    assert events[-1].stage is TaskStage.DONE
    assert events[-1].detail["outcome"] == "success"


def test_task_started_event_matches_running_projection(store) -> None:
    store.ensure_task(TASK_1, created_at=_at(1))
    started = _value(store.update_status(TASK_1, TaskStatus.RUNNING))

    assert started.status is TaskStatus.RUNNING
    assert started.stage is TaskStage.RECEIVED
    assert started.started_at is not None

    event = store.events(TASK_1)[-1]
    assert event.event_type is TaskEventType.TASK_STARTED
    assert event.status is started.status
    assert event.stage is started.stage
    assert event.created_at == started.started_at


def test_stage_and_status_events_match_the_projection(store) -> None:
    store.ensure_task(TASK_1)
    store.update_status(TASK_1, TaskStatus.RUNNING)
    staged = _value(store.update_stage(TASK_1, TaskStage.PLANNING))
    waiting = _value(store.update_status(TASK_1, TaskStatus.AWAITING_CONFIRMATION))

    events = store.events(TASK_1)
    stage_event = events[-2]
    status_event = events[-1]
    assert stage_event.event_type is TaskEventType.STAGE_CHANGED
    assert stage_event.stage is staged.stage is TaskStage.PLANNING
    assert stage_event.status is staged.status is TaskStatus.RUNNING
    assert stage_event.detail["stage"] == TaskStage.PLANNING.value
    assert status_event.event_type is TaskEventType.STATUS_CHANGED
    assert status_event.status is waiting.status is TaskStatus.AWAITING_CONFIRMATION
    assert status_event.stage is waiting.stage is TaskStage.PLANNING
    assert status_event.detail["status"] == TaskStatus.AWAITING_CONFIRMATION.value


def test_neutral_events_cannot_mutate_projection_headers(store) -> None:
    store.ensure_task(TASK_1)
    store.update_status(TASK_1, TaskStatus.RUNNING)
    store.update_stage(TASK_1, TaskStage.PLANNING)
    before = store.get(TASK_1)
    assert before is not None
    event_count = len(store.events(TASK_1))

    for forbidden in ({"stage": TaskStage.DONE.value}, {"status": TaskStatus.FAILED.value}):
        with pytest.raises(InvalidTaskTransition):
            store.append_event(
                TASK_1,
                TaskEventType.NOTE,
                detail={"event_code": "progress", **forbidden},
            )

    assert store.get(TASK_1) == before
    assert len(store.events(TASK_1)) == event_count


def test_finish_detail_cannot_contradict_the_terminal_status(store) -> None:
    store.ensure_task(TASK_1)
    store.update_status(TASK_1, TaskStatus.RUNNING)
    with pytest.raises(ValueError):
        store.finish(
            TASK_1,
            TaskStatus.FAILED,
            detail={"outcome": "success"},
        )
    assert store.get(TASK_1).status is TaskStatus.RUNNING


def test_invalid_transition_does_not_mutate_the_task(store) -> None:
    store.ensure_task(TASK_1)
    before = store.get(TASK_1)
    assert before is not None

    with pytest.raises(InvalidTaskTransition):
        store.update_status(TASK_1, TaskStatus.COMPLETED)

    after = store.get(TASK_1)
    assert after is not None
    assert after.status is TaskStatus.QUEUED
    assert after.version == before.version
    assert len(store.events(TASK_1)) == 1


def test_terminal_state_is_idempotent_and_never_overwritten(store) -> None:
    store.ensure_task(TASK_1)
    store.update_status(TASK_1, TaskStatus.RUNNING)
    first = _value(
        store.finish(
            TASK_1,
            TaskStatus.FAILED,
            failure_category="tool",
            failure_code="E_TOOL",
            created_at=_at(2),
        )
    )
    again = _value(
        store.finish(
            TASK_1,
            TaskStatus.FAILED,
            failure_category="tool",
            failure_code="E_TOOL",
            created_at=_at(3),
        )
    )

    assert again.status is TaskStatus.FAILED
    assert again.stage is TaskStage.DONE
    assert again.active_tool is None
    assert again.active_step_id is None
    assert again.finished_at == first.finished_at
    assert again.version == first.version
    assert again.failure_category == "tool"
    assert again.failure_code == "E_TOOL"

    with pytest.raises(InvalidTaskTransition):
        store.finish(
            TASK_1,
            TaskStatus.FAILED,
            failure_category="internal",
            failure_code="E_OTHER",
        )
    with pytest.raises(InvalidTaskTransition):
        store.finish(TASK_1, TaskStatus.COMPLETED)

    finished_events = [
        event for event in store.events(TASK_1) if event.event_type is TaskEventType.FINISHED
    ]
    assert len(finished_events) == 1


def test_finish_idempotency_normalises_metadata_before_comparing(store) -> None:
    store.ensure_task(TASK_1)
    store.update_status(TASK_1, TaskStatus.RUNNING)
    first = _value(
        store.finish(
            TASK_1,
            TaskStatus.FAILED,
            failure_category="tool",
            failure_code="E_TOOL",
            detail={"count": 1},
            created_at=_at(2),
        )
    )
    again = _value(
        store.finish(
            TASK_1,
            TaskStatus.FAILED,
            failure_category="TOOL",
            failure_code="E_TOOL",
            detail={"count": 1},
            created_at=_at(3),
        )
    )
    assert again == first


def test_finish_validates_arguments_before_idempotency(store) -> None:
    store.ensure_task(TASK_1)
    store.update_status(TASK_1, TaskStatus.RUNNING)
    first = _value(
        store.finish(
            TASK_1,
            TaskStatus.FAILED,
            failure_category="tool",
            failure_code="E_TOOL",
            created_at=_at(2),
        )
    )

    for kwargs in (
        {"failure_code": "NOT_CONFIGURED"},
        {"failure_category": "not_a_category"},
        {"detail": {"not_allowed": "value"}},
        {"detail": {"outcome": "success"}},
    ):
        with pytest.raises(ValueError):
            store.finish(TASK_1, TaskStatus.FAILED, **kwargs)

    assert store.get(TASK_1) == first
    finished_events = [
        event for event in store.events(TASK_1) if event.event_type is TaskEventType.FINISHED
    ]
    assert len(finished_events) == 1


def test_finish_clears_active_tool_and_step(store) -> None:
    store.ensure_task(TASK_1)
    store.update_status(TASK_1, TaskStatus.RUNNING)
    store.append_event(
        TASK_1,
        TaskEventType.TOOL_STARTED,
        detail={"tool": "browser_goto", "step_id": STEP_1},
    )
    active = store.get(TASK_1)
    assert active is not None
    assert active.active_tool == "browser_goto"
    assert active.active_step_id == STEP_1

    finished = _value(store.finish(TASK_1, TaskStatus.COMPLETED))
    assert finished.stage is TaskStage.DONE
    assert finished.active_tool is None
    assert finished.active_step_id is None


def test_done_stage_is_owned_by_finish(store) -> None:
    store.ensure_task(TASK_1)
    store.update_status(TASK_1, TaskStatus.RUNNING)
    before = store.get(TASK_1)
    assert before is not None
    event_count = len(store.events(TASK_1))

    with pytest.raises(InvalidTaskTransition):
        store.update_stage(TASK_1, TaskStage.DONE)
    with pytest.raises(InvalidTaskTransition):
        store.append_event(
            TASK_1,
            TaskEventType.STAGE_CHANGED,
            detail={"stage": TaskStage.DONE.value},
        )

    assert store.get(TASK_1) == before
    assert len(store.events(TASK_1)) == event_count

    finished = _value(store.finish(TASK_1, TaskStatus.COMPLETED))
    assert finished.stage is TaskStage.DONE
    assert finished.status is TaskStatus.COMPLETED


@pytest.mark.parametrize(
    "status",
    [
        TaskStatus.COMPLETED,
        TaskStatus.ANSWERED,
        TaskStatus.PARTIAL,
        TaskStatus.FAILED,
        TaskStatus.HALTED,
        TaskStatus.CANCELLED,
        TaskStatus.EXPIRED,
    ],
)
def test_every_terminal_status_uses_the_done_stage(store, status: TaskStatus) -> None:
    store.ensure_task(TASK_1)
    store.update_status(TASK_1, TaskStatus.RUNNING)
    finished = _value(store.finish(TASK_1, status))

    assert finished.status is status
    assert finished.stage is TaskStage.DONE
    assert finished.finished_at is not None
    assert finished.active_tool is None
    assert finished.active_step_id is None


def test_only_finish_can_set_a_terminal_status(store) -> None:
    store.ensure_task(TASK_1)
    store.update_status(TASK_1, TaskStatus.RUNNING)

    with pytest.raises(InvalidTaskTransition):
        store.update_status(TASK_1, TaskStatus.COMPLETED)
    with pytest.raises(InvalidTaskTransition):
        store.append_event(TASK_1, TaskEventType.FINISHED)
    with pytest.raises(InvalidTaskTransition):
        store.append_event(
            TASK_1,
            TaskEventType.NOTE,
            detail={"event_code": "progress", "status": TaskStatus.FAILED.value},
        )

    snapshot = store.get(TASK_1)
    assert snapshot is not None
    assert snapshot.status is TaskStatus.RUNNING
    assert not any(event.event_type is TaskEventType.FINISHED for event in store.events(TASK_1))


def test_terminal_task_rejects_append_status_change(store) -> None:
    store.ensure_task(TASK_1)
    store.update_status(TASK_1, TaskStatus.RUNNING)
    store.finish(TASK_1, TaskStatus.COMPLETED)
    before = store.get(TASK_1)
    assert before is not None
    event_count = len(store.events(TASK_1))

    with pytest.raises(InvalidTaskTransition):
        store.append_event(
            TASK_1,
            TaskEventType.NOTE,
            detail={"event_code": "progress", "status": TaskStatus.FAILED.value},
        )

    after = store.get(TASK_1)
    assert after == before
    assert len(store.events(TASK_1)) == event_count


def test_terminal_task_rejects_append_stage_change(store) -> None:
    store.ensure_task(TASK_1)
    store.update_status(TASK_1, TaskStatus.RUNNING)
    store.finish(TASK_1, TaskStatus.FAILED, failure_code="E_TOOL", failure_category="tool")
    before = store.get(TASK_1)
    assert before is not None
    event_count = len(store.events(TASK_1))

    with pytest.raises(InvalidTaskTransition):
        store.append_event(
            TASK_1,
            TaskEventType.NOTE,
            detail={"event_code": "progress", "stage": TaskStage.RECEIVED.value},
        )

    after = store.get(TASK_1)
    assert after == before
    assert len(store.events(TASK_1)) == event_count


def test_terminal_task_rejects_append_projection_detail(store) -> None:
    store.ensure_task(TASK_1)
    store.update_status(TASK_1, TaskStatus.RUNNING)
    store.finish(TASK_1, TaskStatus.HALTED, failure_code="E_HALT", failure_category="tool")
    before = store.get(TASK_1)
    assert before is not None
    event_count = len(store.events(TASK_1))

    with pytest.raises(InvalidTaskTransition):
        store.append_event(
            TASK_1,
            TaskEventType.TOOL_STARTED,
            detail={"tool": "browser_goto", "step_id": STEP_1},
        )

    after = store.get(TASK_1)
    assert after == before
    assert len(store.events(TASK_1)) == event_count


def test_concurrent_finish_and_append_leave_terminal_projection_stable(
    db_path: Path,
) -> None:
    seed = open_task_store(db_path, create=True, vocabulary=VOCAB)
    seed.ensure_task(TASK_1)
    seed.update_status(TASK_1, TaskStatus.RUNNING)
    seed.close()

    first = open_task_store(db_path, vocabulary=VOCAB)
    second = open_task_store(db_path, vocabulary=VOCAB)
    barrier = threading.Barrier(2)
    append_rejected = False

    def finish_task() -> None:
        barrier.wait(timeout=5)
        first.finish(
            TASK_1,
            TaskStatus.FAILED,
            failure_category="tool",
            failure_code="E_TOOL",
        )

    def append_task() -> None:
        nonlocal append_rejected
        barrier.wait(timeout=5)
        try:
            second.append_event(
                TASK_1,
                TaskEventType.NOTE,
                detail={"event_code": "progress", "stage": TaskStage.RECEIVED.value},
            )
        except InvalidTaskTransition:
            append_rejected = True

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(finish_task), pool.submit(append_task)]
            for future in futures:
                future.result(timeout=10)

        snapshot = first.get(TASK_1)
        assert snapshot is not None
        assert snapshot.status is TaskStatus.FAILED
        assert snapshot.stage is TaskStage.DONE
        assert snapshot.failure_category == "tool"
        assert snapshot.failure_code == "E_TOOL"
        assert snapshot.finished_at is not None
        assert snapshot.active_tool is None
        assert snapshot.active_step_id is None

        events = first.events(TASK_1)
        assert events[-1].event_type is TaskEventType.FINISHED
        assert snapshot.version == len(events)
        if append_rejected:
            assert not any(event.event_type is TaskEventType.NOTE for event in events)
    finally:
        first.close()
        second.close()


def test_process_level_concurrent_ensure_and_append(db_path: Path) -> None:
    seed = open_task_store(db_path, create=True, vocabulary=VOCAB)
    seed.ensure_task(TASK_1)
    seed.close()

    context = multiprocessing.get_context("spawn")
    start_event = context.Event()
    processes = [
        context.Process(
            target=_process_writer,
            args=(str(db_path), TASK_1, offset, start_event, _vocab_data()),
        )
        for offset in (0, 10)
    ]
    for process in processes:
        process.start()
    start_event.set()
    for process in processes:
        process.join(30)
        assert process.exitcode == 0

    store = open_task_store(db_path, vocabulary=VOCAB)
    try:
        events = store.events(TASK_1)
        assert [event.seq for event in events] == list(range(1, 22))
        assert sum(1 for event in events if event.event_type is TaskEventType.TASK_CREATED) == 1
        snapshot = store.get(TASK_1)
        assert snapshot is not None
        assert snapshot.version == len(events)
    finally:
        store.close()


def test_cancelled_queued_task_has_no_start_time(store) -> None:
    store.ensure_task(TASK_1, created_at=_at(1))
    cancelled = _value(store.finish(TASK_1, TaskStatus.CANCELLED, created_at=_at(2)))
    assert cancelled.started_at is None
    assert cancelled.finished_at == _at(2)


def test_events_support_after_seq_cursor(store) -> None:
    store.ensure_task(TASK_1)
    for count in range(3):
        store.append_event(
            TASK_1,
            TaskEventType.NOTE,
            detail={"event_code": "progress", "count": count},
        )

    events = store.events(TASK_1)
    assert [event.seq for event in events] == [1, 2, 3, 4]
    tail = store.events(TASK_1, after_seq=2)
    assert [event.seq for event in tail] == [3, 4]
    assert store.events(TASK_MISSING) == []


def test_failed_append_rolls_back_event_and_projection(store) -> None:
    store.ensure_task(TASK_1)
    store.update_status(TASK_1, TaskStatus.RUNNING)
    before = store.get(TASK_1)
    assert before is not None
    before_count = len(store.events(TASK_1))
    store._conn.execute(
        """
        CREATE TRIGGER fail_projection
        BEFORE UPDATE ON tasks
        WHEN NEW.version > OLD.version
        BEGIN
          SELECT RAISE(ABORT, 'boom');
        END
        """
    )

    with pytest.raises(sqlite3.IntegrityError):
        store.append_event(
            TASK_1,
            TaskEventType.NOTE,
            detail={"event_code": "progress", "count": 1},
        )

    after = store.get(TASK_1)
    assert after is not None
    assert after.version == before.version
    assert len(store.events(TASK_1)) == before_count


def test_rollback_failure_discards_the_connection(store) -> None:
    store.ensure_task(TASK_1)
    store.update_status(TASK_1, TaskStatus.RUNNING)
    before = store.get(TASK_1)
    assert before is not None
    store._conn.execute(
        """
        CREATE TRIGGER fail_projection
        BEFORE UPDATE ON tasks
        WHEN NEW.version > OLD.version
        BEGIN
          SELECT RAISE(ABORT, 'boom');
        END
        """
    )
    store._conn = _RollbackFailingConnection(store._conn)

    with pytest.raises(sqlite3.IntegrityError, match="boom"):
        store.append_event(
            TASK_1,
            TaskEventType.NOTE,
            detail={"event_code": "progress", "count": 1},
        )

    assert store.available is False
    with pytest.raises(TaskStoreUnavailable):
        store.get(TASK_1)
    with pytest.raises(TaskStoreUnavailable):
        store.recent(0)


def test_commit_failure_rolls_back_and_allows_later_write(store) -> None:
    store.ensure_task(TASK_1)
    store.update_status(TASK_1, TaskStatus.RUNNING)
    before = store.get(TASK_1)
    assert before is not None
    event_count = len(store.events(TASK_1))
    store._conn = _CommitFailingConnection(store._conn)

    with pytest.raises(sqlite3.OperationalError, match="simulated commit failure"):
        store.append_event(
            TASK_1,
            TaskEventType.NOTE,
            detail={"event_code": "progress", "count": 1},
        )

    after_failure = store.get(TASK_1)
    assert after_failure == before
    assert len(store.events(TASK_1)) == event_count

    store.append_event(
        TASK_1,
        TaskEventType.NOTE,
        detail={"event_code": "progress", "count": 2},
    )
    after_success = store.get(TASK_1)
    assert after_success is not None
    assert after_success.version == before.version + 1
    assert len(store.events(TASK_1)) == event_count + 1


def test_concurrent_writers_get_unique_sequences(db_path: Path) -> None:
    first = open_task_store(db_path, create=True, vocabulary=VOCAB)
    second = open_task_store(db_path, vocabulary=VOCAB)
    try:
        first.ensure_task(TASK_1)
        barrier = threading.Barrier(2)

        def writer(opened, offset: int) -> None:
            barrier.wait(timeout=5)
            for count in range(10):
                opened.append_event(
                    TASK_1,
                    TaskEventType.NOTE,
                    detail={"event_code": "progress", "count": offset + count},
                )

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(writer, first, 0), pool.submit(writer, second, 10)]
            for future in futures:
                future.result(timeout=10)

        events = first.events(TASK_1)
        assert [event.seq for event in events] == list(range(1, 22))
        assert len({event.seq for event in events}) == len(events)
    finally:
        first.close()
        second.close()


def test_recent_summary_and_read_only_access(db_path: Path) -> None:
    writer = open_task_store(db_path, create=True, vocabulary=VOCAB)
    try:
        writer.ensure_task(TASK_1, created_at=_at(1))
        writer.update_status(TASK_1, TaskStatus.RUNNING)
        writer.finish(TASK_1, TaskStatus.COMPLETED, created_at=_at(2))
        writer.ensure_task(TASK_2, created_at=_at(3))
        writer.update_status(TASK_2, TaskStatus.RUNNING)
        writer.ensure_task(TASK_3, created_at=_at(4))
    finally:
        writer.close()

    reader = open_task_store(db_path, read_only=True, vocabulary=VOCAB)
    try:
        recent = reader.recent(limit=2)
        assert len(recent) == 2
        assert all(item.event_count >= 1 for item in recent)

        summary = reader.summary()
        assert summary.total == 3
        assert summary.active == 2
        assert summary.terminal == 1
        assert summary.by_status["completed"] == 1
        assert reader.get(TASK_1).status is TaskStatus.COMPLETED

        with pytest.raises(TaskStoreUnavailable):
            reader.ensure_task(TASK_4)
        with pytest.raises(TaskStoreUnavailable):
            reader.finish(TASK_2, TaskStatus.CANCELLED)
    finally:
        reader.close()


def test_read_only_missing_store_is_not_created(db_path: Path) -> None:
    with pytest.raises(TaskStoreUnavailable):
        open_task_store(db_path, read_only=True, vocabulary=VOCAB)
    assert db_path.exists() is False


def test_reopen_with_create_preserves_existing_data(db_path: Path) -> None:
    first = open_task_store(db_path, create=True, vocabulary=VOCAB)
    first.ensure_task(TASK_1)
    first.close()

    again = open_task_store(db_path, create=True, vocabulary=VOCAB)
    try:
        assert again.get(TASK_1) is not None
        assert again.status().schema_version == schema.SCHEMA_VERSION
    finally:
        again.close()


def test_prune_preserves_active_tasks_and_caps_terminal_events(db_path: Path) -> None:
    store = open_task_store(db_path, create=True, vocabulary=VOCAB)
    try:
        store.ensure_task(TASK_OLD, created_at=_at(1))
        store.update_status(TASK_OLD, TaskStatus.RUNNING)
        store.finish(TASK_OLD, TaskStatus.COMPLETED, created_at=_at(2))

        store.ensure_task(TASK_NEW, created_at=_at(3))
        store.update_status(TASK_NEW, TaskStatus.RUNNING)
        for count in range(5):
            store.append_event(
                TASK_NEW,
                TaskEventType.NOTE,
                detail={"event_code": "progress", "count": count},
            )
        store.finish(TASK_NEW, TaskStatus.COMPLETED, created_at=_at(4))
        store.ensure_task(TASK_ACTIVE, created_at=_at(5))

        result = store.prune(max_tasks=2, max_events_per_task=2)
        assert result.tasks_deleted == 1
        assert result.events_deleted == 9
        assert store.get(TASK_OLD) is None
        assert store.get(TASK_ACTIVE) is not None
        assert store.get(TASK_NEW) is not None
        assert store.events(TASK_OLD) == []
        assert [event.seq for event in store.events(TASK_NEW)] == [7, 8]
    finally:
        store.close()


def test_corrupt_database_is_not_replaced(db_path: Path) -> None:
    db_path.write_bytes(b"this is not a sqlite database")
    before = db_path.read_bytes()
    with pytest.raises(TaskStoreUnavailable):
        open_task_store(db_path, create=True, vocabulary=VOCAB)
    assert db_path.read_bytes() == before


def test_newer_schema_version_is_rejected(db_path: Path) -> None:
    store = open_task_store(db_path, create=True, vocabulary=VOCAB)
    store.close()

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA user_version = 999")
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(TaskStoreUnavailable):
        open_task_store(db_path, create=True, vocabulary=VOCAB)
    assert db_path.exists() is True


def test_older_schema_is_not_migrated(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(
            """
            CREATE TABLE tasks (
              task_id TEXT PRIMARY KEY,
              source TEXT NOT NULL,
              stage TEXT NOT NULL,
              status TEXT NOT NULL,
              created_at TEXT NOT NULL,
              started_at TEXT,
              finished_at TEXT,
              updated_at TEXT NOT NULL,
              version INTEGER NOT NULL DEFAULT 0,
              retry_count INTEGER NOT NULL DEFAULT 0,
              active_tool TEXT,
              approval_state TEXT,
              failure_category TEXT,
              failure_code TEXT
            );
            CREATE TABLE task_events (
              event_id INTEGER PRIMARY KEY AUTOINCREMENT,
              task_id TEXT NOT NULL,
              seq INTEGER NOT NULL,
              event_type TEXT NOT NULL,
              stage TEXT,
              status TEXT,
              detail_json TEXT NOT NULL DEFAULT '{}',
              created_at TEXT NOT NULL
            );
            PRAGMA user_version = 1;
            """
        )
    finally:
        conn.close()

    with pytest.raises(TaskStoreUnavailable):
        open_task_store(db_path, create=True, vocabulary=VOCAB)


def test_corrupt_event_detail_is_not_presented_as_history(store) -> None:
    store.ensure_task(TASK_1)
    store._conn.execute(
        "UPDATE task_events SET detail_json = 'not-json' WHERE task_id = ?",
        (TASK_1,),
    )

    with pytest.raises(TaskStoreCorruption):
        store.events(TASK_1)


def test_corrupt_event_type_is_rejected(store) -> None:
    store.ensure_task(TASK_1)
    store._conn.execute(
        "UPDATE task_events SET event_type = 'not_an_event' WHERE task_id = ?",
        (TASK_1,),
    )
    with pytest.raises(TaskStoreCorruption):
        store.events(TASK_1)


def test_corrupt_optional_event_label_is_rejected(store) -> None:
    store.ensure_task(TASK_1)
    store.append_event(TASK_1, TaskEventType.NOTE, detail={"event_code": "progress"})
    store._conn.execute(
        """
        UPDATE task_events SET detail_json = ?
        WHERE task_id = ? AND event_type = ?
        """,
        ('{"event_code":"progress","node":"sk-live-secret"}', TASK_1, TaskEventType.NOTE.value),
    )
    with pytest.raises(TaskStoreCorruption):
        store.events(TASK_1)


def test_corrupt_required_event_label_is_rejected(store) -> None:
    store.ensure_task(TASK_1)
    store._conn.execute(
        "UPDATE task_events SET detail_json = '{}' WHERE task_id = ? AND event_type = ?",
        (TASK_1, TaskEventType.TASK_CREATED.value),
    )
    with pytest.raises(TaskStoreCorruption):
        store.events(TASK_1)


def test_corrupt_event_payload_is_rejected(store) -> None:
    store.ensure_task(TASK_1)
    store.update_stage(TASK_1, TaskStage.PLANNING)
    store._conn.execute(
        """
        UPDATE task_events SET detail_json = ?
        WHERE task_id = ? AND event_type = ?
        """,
        ('{"stage":"not_a_stage"}', TASK_1, TaskEventType.STAGE_CHANGED.value),
    )
    with pytest.raises(TaskStoreCorruption):
        store.events(TASK_1)


def test_corrupt_projection_is_rejected_by_every_read(store) -> None:
    store.ensure_task(TASK_1)
    store._conn.execute(
        "UPDATE tasks SET status = 'not_a_status' WHERE task_id = ?",
        (TASK_1,),
    )
    with pytest.raises(TaskStoreCorruption):
        store.get(TASK_1)
    with pytest.raises(TaskStoreCorruption):
        store.recent(1)
    with pytest.raises(TaskStoreCorruption):
        store.summary()


def test_corrupt_projection_optional_label_is_rejected(store) -> None:
    store.ensure_task(TASK_1)
    store.update_status(TASK_1, TaskStatus.RUNNING)
    store._conn.execute(
        "UPDATE tasks SET active_tool = 'sk-live-secret' WHERE task_id = ?",
        (TASK_1,),
    )
    with pytest.raises(TaskStoreCorruption) as caught:
        store.get(TASK_1)
    assert "sk-live-secret" not in str(caught.value)


def test_corrupt_projection_required_label_is_rejected(store) -> None:
    store.ensure_task(TASK_1)
    store._conn.execute(
        "UPDATE tasks SET source = 'untrusted-source' WHERE task_id = ?",
        (TASK_1,),
    )
    with pytest.raises(TaskStoreCorruption) as caught:
        store.get(TASK_1)
    assert "untrusted-source" not in str(caught.value)


def test_non_terminal_task_with_done_stage_is_corruption(store) -> None:
    store.ensure_task(TASK_1)
    store._conn.execute(
        "UPDATE tasks SET stage = 'done' WHERE task_id = ?",
        (TASK_1,),
    )
    with pytest.raises(TaskStoreCorruption):
        store.get(TASK_1)


def test_terminal_task_without_done_stage_is_corruption(store) -> None:
    store.ensure_task(TASK_1)
    store.update_status(TASK_1, TaskStatus.RUNNING)
    store.finish(TASK_1, TaskStatus.COMPLETED)
    store._conn.execute(
        "UPDATE tasks SET stage = 'responding' WHERE task_id = ?",
        (TASK_1,),
    )
    with pytest.raises(TaskStoreCorruption):
        store.get(TASK_1)


def test_locked_database_does_not_look_like_success(db_path: Path) -> None:
    store = open_task_store(db_path, create=True, vocabulary=VOCAB)
    locker = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        store.ensure_task(TASK_1)
        before = store.get(TASK_1)
        assert before is not None
        store._conn.execute("PRAGMA busy_timeout = 10")
        locker.execute("BEGIN EXCLUSIVE")

        with pytest.raises(sqlite3.OperationalError):
            store.append_event(
                TASK_1,
                TaskEventType.NOTE,
                detail={"event_code": "progress", "count": 1},
            )
    finally:
        locker.rollback()
        locker.close()

    after = store.get(TASK_1)
    assert after is not None
    assert after.version == before.version
    assert len(store.events(TASK_1)) == 1
    store.close()


def test_null_store_is_fail_safe_for_writes_and_explicit_for_reads() -> None:
    null = NullTaskEventStore("disabled_for_test", vocabulary=VOCAB)
    assert isinstance(null, TaskStore)
    assert null.available is False
    assert null.writable is False
    assert null.status().available is False

    ensure = null.ensure_task(TASK_1)
    assert ensure.persisted is False
    assert ensure.value is None
    assert ensure.reason == "disabled_for_test"
    assert (
        null.append_event(TASK_1, TaskEventType.NOTE, detail={"event_code": "progress"}).persisted
        is False
    )
    assert null.update_stage(TASK_1, TaskStage.PLANNING).persisted is False
    assert null.update_status(TASK_1, TaskStatus.RUNNING).persisted is False
    assert null.finish(TASK_1, TaskStatus.COMPLETED).persisted is False

    with pytest.raises(TaskStoreUnavailable):
        null.get(TASK_1)
    with pytest.raises(TaskStoreUnavailable):
        null.events(TASK_1)
    with pytest.raises(TaskStoreUnavailable):
        null.recent(0)
    with pytest.raises(TaskStoreUnavailable):
        null.summary()
    with pytest.raises(TaskStoreUnavailable):
        null.prune()


def test_null_store_keeps_static_validation() -> None:
    null = NullTaskEventStore(vocabulary=VOCAB)
    with pytest.raises(ValueError):
        null.ensure_task("not-a-uuid")
    with pytest.raises(ValueError):
        null.ensure_task(TASK_1, source="password=hunter2")
    with pytest.raises(ValueError):
        null.append_event(
            TASK_1,
            TaskEventType.TOOL_STARTED,
            detail={"tool": "unregistered_tool", "step_id": STEP_1},
        )
    with pytest.raises(InvalidTaskTransition):
        null.update_status(TASK_1, TaskStatus.COMPLETED)
    with pytest.raises(InvalidTaskTransition):
        null.append_event(TASK_1, TaskEventType.FINISHED)
    with pytest.raises(InvalidTaskTransition):
        null.append_event(
            TASK_1,
            TaskEventType.NOTE,
            detail={"event_code": "progress", "status": TaskStatus.FAILED.value},
        )


def test_closed_store_reads_consistently_raise(store) -> None:
    store.ensure_task(TASK_1)
    store.close()

    with pytest.raises(TaskStoreUnavailable):
        store.get(TASK_1)
    with pytest.raises(TaskStoreUnavailable):
        store.events(TASK_1)
    with pytest.raises(TaskStoreUnavailable):
        store.recent(0)
    with pytest.raises(TaskStoreUnavailable):
        store.summary()
    with pytest.raises(TaskStoreUnavailable):
        store.prune()


def test_real_and_null_stores_share_the_public_write_contract(store) -> None:
    assert isinstance(store, TaskStore)
    assert _value(store.ensure_task(TASK_1)).task_id == TASK_1
    assert (
        _value(
            store.append_event(
                TASK_1,
                TaskEventType.NOTE,
                detail={"event_code": "progress"},
            )
        ).event_type
        is TaskEventType.NOTE
    )
    assert _value(store.update_stage(TASK_1, TaskStage.PLANNING)).stage is TaskStage.PLANNING
    assert _value(store.update_status(TASK_1, TaskStatus.RUNNING)).status is TaskStatus.RUNNING
    assert _value(store.finish(TASK_1, TaskStatus.COMPLETED)).status is TaskStatus.COMPLETED

    for method in (
        "ensure_task",
        "append_event",
        "update_stage",
        "update_status",
        "finish",
    ):
        real_parameters = inspect.signature(getattr(type(store), method)).parameters
        null_parameters = inspect.signature(getattr(NullTaskEventStore, method)).parameters
        assert list(real_parameters) == list(null_parameters)


def test_direct_connection_constructor_is_private() -> None:
    import jarvis.taskstore.store as store_module

    assert not hasattr(store_module, "TaskEventStore")
    assert not hasattr(store_module, "open_task_store_with_connection")
