"""B3 recovery tests: ownership, stale expiry, schema migration, resume guards."""

from __future__ import annotations

import logging
import multiprocessing
import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import psutil
import pytest

from jarvis.agent.runner import resume_task
from jarvis.daemon.recovery import (
    current_process_create_time,
    new_task_owner,
    owner_is_alive,
    reconcile_task_store,
)
from jarvis.daemon.server import DaemonServer
from jarvis.taskstore import (
    TaskOwner,
    TaskSnapshot,
    TaskStatus,
    TaskStoreVocabulary,
    open_task_store,
    schema,
)

TASK_1 = "00000000-0000-4000-8000-000000000201"
TASK_2 = "00000000-0000-4000-8000-000000000202"
TASK_3 = "00000000-0000-4000-8000-000000000203"
TASK_4 = "00000000-0000-4000-8000-000000000204"
TASK_5 = "00000000-0000-4000-8000-000000000205"
TASK_6 = "00000000-0000-4000-8000-000000000206"
TASK_7 = "00000000-0000-4000-8000-000000000207"

VOCAB = TaskStoreVocabulary(
    providers=frozenset({"groq"}),
    models=frozenset({("groq", "openai/gpt-oss-120b")}),
    tools=frozenset({"fake_echo"}),
    nodes=frozenset({"planner", "respond"}),
    failure_codes=frozenset({"expired", "E_TOOL"}),
    failure_categories=frozenset(
        {
            "policy",
            "validation",
            "tool",
            "llm",
            "timeout",
            "cancelled",
            "verification",
            "internal",
            "unknown",
        }
    ),
)


def _owner(instance: str, pid: int, create_time: float | None = 100.0) -> TaskOwner:
    return TaskOwner(instance_id=instance, pid=pid, process_create_time=create_time)


_DEAD_PID: int | None = None


def _exit_now() -> None:
    os._exit(0)


def _dead_pid() -> int:
    global _DEAD_PID
    if _DEAD_PID is None:
        context = multiprocessing.get_context("spawn")
        proc = context.Process(target=_exit_now)
        proc.start()
        pid = proc.pid
        proc.join(timeout=10)
        assert pid is not None
        assert not psutil.pid_exists(pid)
        _DEAD_PID = pid
    assert not psutil.pid_exists(_DEAD_PID)
    return _DEAD_PID


def _crash_process_writer(db_path: str, task_id: str) -> None:
    from jarvis.daemon.recovery import new_task_owner
    from jarvis.taskstore import TaskStoreVocabulary, open_task_store

    vocab = TaskStoreVocabulary(
        providers=frozenset({"groq"}),
        models=frozenset({("groq", "openai/gpt-oss-120b")}),
        tools=frozenset({"fake_echo"}),
        nodes=frozenset({"planner", "respond"}),
        failure_codes=frozenset({"expired", "E_TOOL"}),
        failure_categories=frozenset({"timeout"}),
    )
    store = open_task_store(db_path, vocabulary=vocab, owner=new_task_owner())
    store.ensure_task(task_id, source="voice")
    store.update_status(task_id, TaskStatus.RUNNING)
    store.close()


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "task_events.db"


def _create_task(db_path: Path, task_id: str, owner: TaskOwner | None) -> None:
    store = open_task_store(db_path, create=True, vocabulary=VOCAB, owner=owner)
    try:
        store.ensure_task(task_id, source="voice")
    finally:
        store.close()


def _set_status(db_path: Path, task_id: str, status: TaskStatus) -> None:
    store = open_task_store(db_path, vocabulary=VOCAB)
    try:
        current = store.get(task_id)
        assert current is not None
        if current.status is TaskStatus.QUEUED and status is not TaskStatus.QUEUED:
            store.update_status(task_id, TaskStatus.RUNNING)
        if status not in (TaskStatus.QUEUED, TaskStatus.RUNNING):
            store.update_status(task_id, status)
    finally:
        store.close()


@pytest.mark.parametrize(
    ("task_id", "status"),
    [
        (TASK_1, TaskStatus.RUNNING),
        (TASK_2, TaskStatus.AWAITING_CONFIRMATION),
        (TASK_3, TaskStatus.AWAITING_CLARIFICATION),
        (TASK_4, TaskStatus.QUEUED),
    ],
)
def test_interrupted_nonterminal_tasks_expire(
    db_path: Path, task_id: str, status: TaskStatus
) -> None:
    dead = _owner("a" * 32, _dead_pid(), 100.0)
    _create_task(db_path, task_id, dead)
    if status is not TaskStatus.QUEUED:
        _set_status(db_path, task_id, status)

    store = open_task_store(db_path, vocabulary=VOCAB)
    try:
        result = store.reconcile_stale_tasks(_owner("b" * 32, 999999, 200.0), lambda _: False)
        snapshot = store.get(task_id)
        assert snapshot is not None
        assert snapshot.status is TaskStatus.EXPIRED
        assert snapshot.failure_category == "timeout"
        assert snapshot.failure_code == "expired"
        assert snapshot.approval_state is None
        assert result.expired_task_ids == (task_id,)
        assert sum(e.event_type.value == "task_finished" for e in store.events(task_id)) == 1
    finally:
        store.close()


def test_queued_task_without_start_is_expired(db_path: Path) -> None:
    dead = _owner("a" * 32, _dead_pid(), 100.0)
    _create_task(db_path, TASK_1, dead)
    store = open_task_store(db_path, vocabulary=VOCAB)
    try:
        store.reconcile_stale_tasks(_owner("b" * 32, 999999, 200.0), lambda _: False)
        snapshot = store.get(TASK_1)
        assert snapshot is not None and snapshot.status is TaskStatus.EXPIRED
    finally:
        store.close()


def test_completed_task_is_never_changed(db_path: Path) -> None:
    dead = _owner("a" * 32, _dead_pid(), 100.0)
    _create_task(db_path, TASK_1, dead)
    store = open_task_store(db_path, vocabulary=VOCAB)
    try:
        store.update_status(TASK_1, TaskStatus.RUNNING)
        store.finish(TASK_1, TaskStatus.COMPLETED)
        result = store.reconcile_stale_tasks(_owner("b" * 32, 999999, 200.0), lambda _: False)
        snapshot = store.get(TASK_1)
        assert snapshot is not None and snapshot.status is TaskStatus.COMPLETED
        assert result.expired_count == 0
    finally:
        store.close()


def test_reconciliation_is_idempotent(db_path: Path) -> None:
    dead = _owner("a" * 32, _dead_pid(), 100.0)
    _create_task(db_path, TASK_1, dead)
    _set_status(db_path, TASK_1, TaskStatus.RUNNING)
    store = open_task_store(db_path, vocabulary=VOCAB)
    try:
        first = store.reconcile_stale_tasks(_owner("b" * 32, 999999, 200.0), lambda _: False)
        second = store.reconcile_stale_tasks(_owner("b" * 32, 999999, 200.0), lambda _: False)
        assert first.expired_task_ids == (TASK_1,)
        assert second.expired_task_ids == ()
        assert sum(e.event_type.value == "task_finished" for e in store.events(TASK_1)) == 1
    finally:
        store.close()


def test_live_owner_is_skipped(db_path: Path) -> None:
    live = TaskOwner(
        instance_id="a" * 32,
        pid=psutil.Process().pid,
        process_create_time=current_process_create_time(),
    )
    _create_task(db_path, TASK_1, live)
    _set_status(db_path, TASK_1, TaskStatus.RUNNING)
    store = open_task_store(db_path, vocabulary=VOCAB)
    try:
        result = store.reconcile_stale_tasks(_owner("b" * 32, 999999, 200.0), owner_is_alive)
        snapshot = store.get(TASK_1)
        assert snapshot is not None and snapshot.status is TaskStatus.RUNNING
        assert result.skipped_active == 1
    finally:
        store.close()


def test_unknown_liveness_and_legacy_owner_are_not_expired(db_path: Path) -> None:
    _create_task(db_path, TASK_1, _owner("a" * 32, _dead_pid(), 100.0))
    _set_status(db_path, TASK_1, TaskStatus.RUNNING)
    _create_task(db_path, TASK_2, None)
    _set_status(db_path, TASK_2, TaskStatus.RUNNING)
    store = open_task_store(db_path, vocabulary=VOCAB)
    try:
        result = store.reconcile_stale_tasks(_owner("b" * 32, 999999, 200.0), lambda _: None)
        first = store.get(TASK_1)
        second = store.get(TASK_2)
        assert first is not None and first.status is TaskStatus.RUNNING
        assert second is not None and second.status is TaskStatus.RUNNING
        assert result.skipped_unknown == 1
        assert result.skipped_legacy == 1
    finally:
        store.close()


def test_owner_is_alive_rejects_pid_reuse_and_missing_age() -> None:
    current = psutil.Process()
    assert (
        owner_is_alive(
            TaskOwner(
                instance_id="a" * 32,
                pid=current.pid,
                process_create_time=current.create_time(),
            )
        )
        is True
    )
    assert (
        owner_is_alive(
            TaskOwner(
                instance_id="a" * 32,
                pid=current.pid,
                process_create_time=current.create_time() - 20.0,
            )
        )
        is False
    )
    assert (
        owner_is_alive(TaskOwner(instance_id="a" * 32, pid=current.pid, process_create_time=None))
        is None
    )


def test_owner_fields_round_trip(db_path: Path) -> None:
    owner = _owner("a" * 32, 1234, 42.5)
    _create_task(db_path, TASK_1, owner)
    store = open_task_store(db_path, vocabulary=VOCAB)
    try:
        snapshot = store.get(TASK_1)
        assert snapshot is not None
        assert snapshot.owner_identity == owner
    finally:
        store.close()


def test_v2_schema_migrates_without_losing_rows(db_path: Path) -> None:
    v2 = schema.SCHEMA.replace(
        """  failure_code TEXT,
  owner_instance TEXT,
  owner_pid INTEGER,
  owner_process_create_time REAL
);""",
        """  failure_code TEXT
);""",
    ).replace(
        "CREATE INDEX IF NOT EXISTS idx_tasks_owner_status\n  ON tasks (owner_instance, status);\n",
        "",
    )
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(v2)
        conn.execute(
            """INSERT INTO tasks (
                task_id, source, stage, status, created_at, updated_at
            ) VALUES (?, 'voice', 'received', 'queued', ?, ?)""",
            (TASK_1, "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"),
        )
        conn.execute("PRAGMA user_version = 2")
        conn.commit()
    finally:
        conn.close()

    store = open_task_store(db_path, vocabulary=VOCAB)
    try:
        assert store.status().schema_version == 3
        snapshot = store.get(TASK_1)
        assert snapshot is not None and snapshot.status is TaskStatus.QUEUED
        assert snapshot.owner_identity is None
    finally:
        store.close()


def test_runtime_vocabulary_contains_the_recovery_code() -> None:
    """The bug: reconciliation used a code the production vocabulary rejected."""
    from jarvis.taskstore.runtime import build_task_vocabulary

    assert "expired" in build_task_vocabulary().failure_codes


def test_runtime_store_expires_a_dead_row(db_path: Path) -> None:
    """Reproduce the live startup failure end to end with the real vocabulary."""
    from jarvis.taskstore.runtime import open_runtime_task_store

    dead = _owner("a" * 32, _dead_pid(), 100.0)
    store = open_runtime_task_store(path=db_path, owner=dead)
    try:
        store.ensure_task(TASK_1, source="voice")
        store.update_status(TASK_1, TaskStatus.RUNNING)
    finally:
        store.close()

    store = open_runtime_task_store(path=db_path)
    try:
        result = store.reconcile_stale_tasks(_owner("b" * 32, 999999, 200.0), lambda _: False)
        snapshot = store.get(TASK_1)
        assert snapshot is not None and snapshot.status is TaskStatus.EXPIRED
        assert snapshot.failure_code == "expired"
        assert result.expired_task_ids == (TASK_1,)
    finally:
        store.close()


def test_one_unreadable_row_does_not_block_the_rest(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dead = _owner("a" * 32, _dead_pid(), 100.0)
    for task_id in (TASK_1, TASK_2):
        _create_task(db_path, task_id, dead)
        _set_status(db_path, task_id, TaskStatus.RUNNING)
    store = open_task_store(db_path, vocabulary=VOCAB)
    original = store.finish

    def broken(task_id: str, *args: Any, **kwargs: Any) -> Any:
        if task_id == TASK_1:
            raise ValueError("failure_code is not configured")
        return original(task_id, *args, **kwargs)

    monkeypatch.setattr(store, "finish", broken)
    try:
        result = store.reconcile_stale_tasks(_owner("b" * 32, 999999, 200.0), lambda _: False)
        first = store.get(TASK_1)
        second = store.get(TASK_2)
        assert first is not None and first.status is TaskStatus.RUNNING
        assert second is not None and second.status is TaskStatus.EXPIRED
        assert result.expired_task_ids == (TASK_2,)
        assert result.skipped_unknown == 1
    finally:
        store.close()


def test_process_crash_reconciles_real_owner(db_path: Path) -> None:
    open_task_store(db_path, create=True, vocabulary=VOCAB).close()
    context = multiprocessing.get_context("spawn")
    proc = context.Process(target=_crash_process_writer, args=(str(db_path), TASK_1))
    proc.start()
    proc.join(timeout=30)
    assert proc.exitcode == 0

    store = open_task_store(db_path, vocabulary=VOCAB)
    try:
        result = reconcile_task_store(store, new_task_owner(), logger=logging.getLogger("test"))
        snapshot = store.get(TASK_1)
        assert snapshot is not None and snapshot.status is TaskStatus.EXPIRED
        assert result.expired_task_ids == (TASK_1,)
    finally:
        store.close()


def test_daemon_prepares_recovery_before_serving(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dead = _owner("a" * 32, _dead_pid(), 100.0)
    _create_task(db_path, TASK_1, dead)
    _set_status(db_path, TASK_1, TaskStatus.RUNNING)
    store = open_task_store(db_path, vocabulary=VOCAB)
    server = DaemonServer(settings=None, store=SimpleNamespace(get=lambda _: "t"))
    monkeypatch.setattr(server, "_open_tasks", lambda: store)
    try:
        server._prepare_task_lifecycle()
        snapshot = store.get(TASK_1)
        assert snapshot is not None and snapshot.status is TaskStatus.EXPIRED
        assert server._task_owner is not None
    finally:
        store.close()


class _NoCheckpoint:
    def get_tuple(self, config: dict[str, Any]) -> None:
        del config


def _terminal_task(db_path: Path) -> TaskSnapshot:
    _create_task(db_path, TASK_1, None)
    store = open_task_store(db_path, vocabulary=VOCAB)
    try:
        store.update_status(TASK_1, TaskStatus.RUNNING)
        result = store.finish(TASK_1, TaskStatus.COMPLETED)
        assert result.value is not None
        return result.value
    finally:
        store.close()


def test_resume_refuses_missing_task_record(db_path: Path) -> None:
    store = open_task_store(db_path, create=True, vocabulary=VOCAB)
    try:
        outcome = resume_task(SimpleNamespace(tasks=store), None, TASK_7, "yes")
        assert outcome.error is not None and "no durable task record" in outcome.error
        assert outcome.final_answer is None
    finally:
        store.close()


def test_resume_refuses_terminal_task(db_path: Path) -> None:
    _terminal_task(db_path)
    store = open_task_store(db_path, vocabulary=VOCAB)
    try:
        outcome = resume_task(SimpleNamespace(tasks=store), None, TASK_1, "yes")
        assert outcome.error is not None and "already completed" in outcome.error
    finally:
        store.close()


def test_resume_refuses_expired_task(db_path: Path) -> None:
    _create_task(db_path, TASK_1, None)
    store = open_task_store(db_path, vocabulary=VOCAB)
    try:
        store.update_status(TASK_1, TaskStatus.RUNNING)
        store.finish(
            TASK_1,
            TaskStatus.EXPIRED,
            failure_category="timeout",
            failure_code="expired",
        )
        outcome = resume_task(SimpleNamespace(tasks=store), None, TASK_1, "yes")
        assert outcome.error is not None and "already expired" in outcome.error
    finally:
        store.close()


def test_resume_refuses_missing_or_unreadable_checkpoint(db_path: Path) -> None:
    _create_task(db_path, TASK_1, None)
    store = open_task_store(db_path, vocabulary=VOCAB)
    try:
        absent = resume_task(SimpleNamespace(tasks=store), None, TASK_1, "yes")
        assert absent.error is not None and "not available" in absent.error

        missing = resume_task(SimpleNamespace(tasks=store), _NoCheckpoint(), TASK_1, "yes")
        assert missing.error is not None and "checkpoint is missing" in missing.error

        class _Broken:
            def get_tuple(self, config: dict[str, Any]) -> None:
                del config
                raise sqlite3.OperationalError("closed")

        broken = resume_task(SimpleNamespace(tasks=store), _Broken(), TASK_1, "yes")
        assert broken.error is not None and "cannot be read" in broken.error
    finally:
        store.close()
