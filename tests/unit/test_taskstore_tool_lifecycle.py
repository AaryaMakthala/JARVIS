"""P0.2: tool lifecycle events around actual tool execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from jarvis.agent.context import make_app_context
from jarvis.agent.graph import open_sqlite_checkpointer
from jarvis.agent.nodes import telemetry
from jarvis.agent.runner import run_task
from jarvis.config import Settings
from jarvis.llm.client import FakeLLM
from jarvis.taskstore import TaskEventType, TaskStatus, TaskStore, open_task_store
from jarvis.taskstore.runtime import build_task_vocabulary
from jarvis.tools.base import ToolContext, ToolResult, ToolSpec
from support import brain_action, make_spec, registry_with

TASK = "00000000-0000-4000-8000-0000000000aa"


def _store(tmp_path: Path, registry: Any) -> TaskStore:
    settings = Settings()
    return open_task_store(
        tmp_path / "task_events.db",
        create=True,
        vocabulary=build_task_vocabulary(settings, registry),
    )


def _events(store: TaskStore, task_id: str, event_type: TaskEventType) -> list[Any]:
    return [event for event in store.events(task_id) if event.event_type is event_type]


def _assert_no_sensitive_detail(store: TaskStore, task_id: str, *markers: str) -> None:
    raw = json.dumps([event.model_dump(mode="json") for event in store.events(task_id)])
    for marker in markers:
        assert marker not in raw


def test_successful_tool_execution_records_start_finish_and_projections(tmp_path: Path) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    registry = registry_with(make_spec("fake_echo", base_tier=0, record=record))
    store = _store(tmp_path, registry)
    saver = open_sqlite_checkpointer(":memory:")
    ctx = make_app_context(
        Settings(),
        llm=FakeLLM([brain_action("fake_echo", {"text": "hi"})]),
        registry=registry,
        tasks=store,
    )
    try:
        outcome = run_task(ctx, saver, "echo hi")
        assert outcome.interrupted is False
        started = _events(store, outcome.task_id, TaskEventType.TOOL_STARTED)
        finished = _events(store, outcome.task_id, TaskEventType.TOOL_FINISHED)
        assert [event.detail for event in started] == [{"tool": "fake_echo", "step_id": "s1"}]
        assert len(finished) == 1
        detail = finished[0].detail
        assert detail["tool"] == "fake_echo"
        assert detail["step_id"] == "s1"
        assert detail["ok"] is True
        assert "verified" in detail
        assert detail["verified"] in (None, True, False)
        assert isinstance(detail["duration_ms"], int)
        assert detail["duration_ms"] >= 0
        assert started[0].seq < finished[0].seq
        snapshot = store.get(outcome.task_id)
        assert snapshot is not None
        assert snapshot.status is TaskStatus.COMPLETED
        assert snapshot.active_tool is None
        assert snapshot.active_step_id is None
        assert record == [("fake_echo", {"text": "hi"})]
        _assert_no_sensitive_detail(store, outcome.task_id, "echo hi", "password", "secret")
    finally:
        saver.conn.close()
        store.close()


def test_failed_tool_execution_records_failure_and_clears_active_tool(tmp_path: Path) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    registry = registry_with(
        make_spec("fake_echo", base_tier=0, record=record, run_ok=False, verify_ok=False)
    )
    store = _store(tmp_path, registry)
    saver = open_sqlite_checkpointer(":memory:")
    ctx = make_app_context(
        Settings(),
        llm=FakeLLM([brain_action("fake_echo", {"text": "boom"})]),
        registry=registry,
        tasks=store,
    )
    ctx.settings.agent.max_retries_per_step = 0
    ctx.settings.agent.max_replans = 0
    try:
        outcome = run_task(ctx, saver, "echo boom")
        started = _events(store, outcome.task_id, TaskEventType.TOOL_STARTED)
        finished = _events(store, outcome.task_id, TaskEventType.TOOL_FINISHED)
        assert len(started) == 1
        assert len(finished) == 1
        assert started[0].detail == {"tool": "fake_echo", "step_id": "s1"}
        assert finished[0].detail["tool"] == "fake_echo"
        assert finished[0].detail["step_id"] == "s1"
        assert finished[0].detail["ok"] is False
        assert finished[0].detail["verified"] in (None, True, False)
        assert isinstance(finished[0].detail["duration_ms"], int)
        assert finished[0].detail["duration_ms"] >= 0
        assert started[0].seq < finished[0].seq
        snapshot = store.get(outcome.task_id)
        assert snapshot is not None
        assert snapshot.active_tool is None
        assert snapshot.active_step_id is None
        _assert_no_sensitive_detail(
            store, outcome.task_id, "echo boom", "could not run", "RuntimeError"
        )
    finally:
        saver.conn.close()
        store.close()


def test_dry_run_emits_no_tool_lifecycle_events(tmp_path: Path) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    registry = registry_with(make_spec("fake_echo", base_tier=0, record=record))
    store = _store(tmp_path, registry)
    saver = open_sqlite_checkpointer(":memory:")
    ctx = make_app_context(
        Settings(),
        llm=FakeLLM([brain_action("fake_echo", {"text": "hi"})]),
        registry=registry,
        tasks=store,
        dry_run=True,
    )
    try:
        outcome = run_task(ctx, saver, "echo hi")
        assert outcome.interrupted is False
        assert store.get(outcome.task_id) is None
        assert _events(store, outcome.task_id, TaskEventType.TOOL_STARTED) == []
        assert _events(store, outcome.task_id, TaskEventType.TOOL_FINISHED) == []
        assert record == []
    finally:
        saver.conn.close()
        store.close()


def test_refusal_before_execution_emits_no_tool_lifecycle_events(tmp_path: Path) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    registry = registry_with(make_spec("fake_echo", base_tier=0, record=record))
    store = _store(tmp_path, registry)
    saver = open_sqlite_checkpointer(":memory:")
    ctx = make_app_context(
        Settings(),
        llm=FakeLLM([brain_action("missing_tool", {"text": "hi"})]),
        registry=registry,
        tasks=store,
    )
    try:
        outcome = run_task(ctx, saver, "do something")
        assert outcome.interrupted is False
        assert _events(store, outcome.task_id, TaskEventType.TOOL_STARTED) == []
        assert _events(store, outcome.task_id, TaskEventType.TOOL_FINISHED) == []
        snapshot = store.get(outcome.task_id)
        assert snapshot is not None
        assert snapshot.active_tool is None
        assert snapshot.active_step_id is None
        assert record == []
    finally:
        saver.conn.close()
        store.close()


def test_tool_finished_records_boolean_verification_from_execution_result(tmp_path: Path) -> None:
    class _Args(BaseModel):
        text: str = Field(min_length=1)

    def _run(args: _Args, ctx: ToolContext) -> ToolResult:
        return ToolResult(ok=True, output="done", data={"text": args.text}, verified=False)

    spec = ToolSpec(
        name="fake_verify",
        description="returns explicit verified flag",
        args_model=_Args,
        base_tier=0,
        run=_run,
        describe=lambda args: f"fake_verify {args.text}",
    )
    registry = registry_with(spec)
    store = _store(tmp_path, registry)
    saver = open_sqlite_checkpointer(":memory:")
    ctx = make_app_context(
        Settings(),
        llm=FakeLLM([brain_action("fake_verify", {"text": "hi"})]),
        registry=registry,
        tasks=store,
    )
    ctx.settings.agent.max_retries_per_step = 0
    ctx.settings.agent.max_replans = 0
    try:
        outcome = run_task(ctx, saver, "verify this")
        started = _events(store, outcome.task_id, TaskEventType.TOOL_STARTED)
        finished = _events(store, outcome.task_id, TaskEventType.TOOL_FINISHED)
        assert len(started) == 1
        assert len(finished) == 1
        assert finished[0].detail["ok"] is True
        assert finished[0].detail["verified"] is False
        assert started[0].seq < finished[0].seq
    finally:
        saver.conn.close()
        store.close()


def test_tool_lifecycle_writer_failure_does_not_break_execution(tmp_path: Path) -> None:
    class _BrokenAppendStore:
        available = True
        writable = True

        def __init__(self, real: TaskStore) -> None:
            self._real = real

        def __getattr__(self, name: str) -> Any:
            return getattr(self._real, name)

        def append_event(self, task_id: str, event_type: Any, **kwargs: Any) -> Any:
            kind = getattr(event_type, "value", str(event_type))
            if kind in {"tool_started", "tool_finished"}:
                raise RuntimeError("broken event append")
            return self._real.append_event(task_id, event_type, **kwargs)

    record: list[tuple[str, dict[str, Any]]] = []
    registry = registry_with(make_spec("fake_echo", base_tier=0, record=record))
    real = _store(tmp_path, registry)
    store = _BrokenAppendStore(real)
    saver = open_sqlite_checkpointer(":memory:")
    ctx = make_app_context(
        Settings(),
        llm=FakeLLM([brain_action("fake_echo", {"text": "hi"})]),
        registry=registry,
        tasks=store,
    )
    try:
        outcome = run_task(ctx, saver, "echo hi")
        assert outcome.interrupted is False
        assert record == [("fake_echo", {"text": "hi"})]
        assert _events(real, outcome.task_id, TaskEventType.TOOL_STARTED) == []
        assert _events(real, outcome.task_id, TaskEventType.TOOL_FINISHED) == []
    finally:
        saver.conn.close()
        real.close()


def test_tool_lifecycle_helper_writes_only_allowed_keys(tmp_path: Path) -> None:
    registry = registry_with(make_spec("fake_echo", base_tier=0, record=[]))
    store = _store(tmp_path, registry)
    ctx = make_app_context(Settings(), llm=None, registry=registry, tasks=store)
    telemetry.record_started({"task_id": TASK, "source": "terminal"}, ctx)
    telemetry.record_tool_finished(
        {"task_id": TASK, "source": "terminal"},
        ctx,
        tool="fake_echo",
        step_id="s1",
        ok=False,
        verified=False,
        duration_ms=1,
    )
    finished = _events(store, TASK, TaskEventType.TOOL_FINISHED)[0]
    assert set(finished.detail.keys()) == {"tool", "step_id", "ok", "verified", "duration_ms"}
    store.close()
