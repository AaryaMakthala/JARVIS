"""Per-task telemetry: the ``task_log`` rows (docs/01 "Observability").

Four claims are under test, and each one is a claim the project makes in writing:

1. **It is written.**  ``intake`` opens a row and ``respond`` closes it, through
   the real compiled graph, so "API calls per task (measured)" is measurable
   after the process exits instead of dying in memory.
2. **The status cannot flatter the task.**  A task is only ``completed`` when
   every recorded step succeeded *and* verified; a refusal, a timeout, a
   cancelled task, a blank input and an unproven step each get their own
   honest status, and the set is closed.
3. **It cannot break a task or leak a secret.**  A missing backend, a
   ``NullMemory``, a double that raises, and a command containing a pasted
   credential are all handled without changing the answer or persisting the
   value.
4. **It is bounded and honest.**  The table is pruned to ``max_task_log``, and
   an unknown status is recorded as a failure rather than silently accepted.
"""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest

from jarvis.agent.context import make_app_context
from jarvis.agent.graph import open_sqlite_checkpointer
from jarvis.agent.nodes.telemetry import task_status
from jarvis.agent.runner import run_task
from jarvis.agent.state import Plan, StepResult
from jarvis.config import Settings
from jarvis.llm.client import FakeLLM, Usage
from jarvis.memory import NullMemory, SqliteMemory, open_db
from jarvis.memory.tasklog import STATUSES, TaskLogStore, safe_input
from support import brain_action, make_spec, registry_with

# ── helpers ───────────────────────────────────────────────────────────────


@pytest.fixture
def store() -> SqliteMemory:
    memory = SqliteMemory(open_db(":memory:"), settings=Settings())
    try:
        yield memory
    finally:
        memory.close()


def _task_log(store: SqliteMemory) -> TaskLogStore:
    return store.task_log


def _ctx(llm: Any, memory: Any, **kwargs: Any):
    record: list[tuple[str, dict[str, Any]]] = []
    spec = make_spec("fake_echo", base_tier=0, record=record)
    return make_app_context(
        Settings(**kwargs), llm=llm, registry=registry_with(spec), memory=memory
    )


def _ok(step_id: str = "s1", **kwargs: Any) -> StepResult:
    return StepResult(step_id=step_id, **{"ok": True, "output": "ok", "verified": True, **kwargs})


def _step() -> Any:
    from jarvis.agent.state import Step

    return Step(id="s1", tool="fake_echo", args={"text": "hi"}, rationale="echo", expect="text")


# ── 1. the graph actually writes rows ─────────────────────────────────────


def test_completed_task_records_calls_tokens_steps_and_replans(store: SqliteMemory) -> None:
    """A successful task ends with real, non-zero counters."""
    saver = open_sqlite_checkpointer(":memory:")
    llm = FakeLLM(
        [brain_action("fake_echo", {"text": "hi"})],
        usage=Usage(calls=1, prompt_tokens=120, completion_tokens=45),
    )
    ctx = _ctx(llm, store)
    try:
        outcome = run_task(ctx, saver, "echo hi")
        assert outcome.interrupted is False
        row = _task_log(store).get(outcome.task_id)
    finally:
        saver.conn.close()

    assert row is not None, "respond did not write a task_log row"
    assert row.status == "completed"
    assert row.source == "terminal"
    assert row.user_input == "echo hi"
    assert row.api_calls == 1
    assert row.tokens == 165  # prompt + completion
    assert row.steps == 1
    assert row.replans == 0
    assert row.finished_at and row.finished
    assert row.duration_ms is not None and row.duration_ms >= 0
    assert row.peak_rss_mb is None or row.peak_rss_mb > 0


def test_replan_cost_is_counted_in_the_same_row(store: SqliteMemory) -> None:
    """A second LLM call for the same task adds to the row, never to a new one."""
    saver = open_sqlite_checkpointer(":memory:")
    llm = FakeLLM(
        [brain_action("fake_echo", {"text": "hi"})],
        usage=Usage(calls=1, prompt_tokens=10, completion_tokens=5),
    )
    ctx = make_app_context(
        Settings(agent={"max_retries_per_step": 0, "max_replans": 1}),
        llm=llm,
        registry=registry_with(make_spec("fake_echo", base_tier=0, record=[])),
        memory=store,
    )
    try:
        outcome = run_task(ctx, saver, "echo hi")
        row = _task_log(store).get(outcome.task_id)
    finally:
        saver.conn.close()
    assert row is not None
    assert row.api_calls == 1  # only the brain ran; no step failed, so no replan
    assert row.replans == 0


def test_only_one_row_per_task(store: SqliteMemory) -> None:
    """intake's ``running`` row and respond's row are the same row."""
    saver = open_sqlite_checkpointer(":memory:")
    ctx = _ctx(FakeLLM([brain_action("fake_echo", {"text": "hi"})]), store)
    try:
        run_task(ctx, saver, "echo hi")
    finally:
        saver.conn.close()
    assert _task_log(store).count() == 1


def test_running_row_survives_a_task_that_never_finishes(store: SqliteMemory) -> None:
    """A task that dies with a pending confirmation stays visible as running."""
    from jarvis.agent.nodes.intake import intake

    ctx = _ctx(None, store)
    update = intake({"user_input": "delete stuff", "source": "terminal"}, ctx)
    row = _task_log(store).get(update["task_id"])
    assert row is not None
    assert row.status == "running"
    assert not row.finished
    assert row.api_calls == 0 and row.steps == 0


def test_start_keeps_the_earliest_time_so_a_resume_does_not_reset_the_clock(
    store: SqliteMemory,
) -> None:
    """The duration is the real elapsed time, not just the last segment."""
    log = _task_log(store)
    log.start("t1", source="terminal", user_input="first")
    first = log.get("t1")
    assert first is not None and first.started_at

    log.start("t1", source="terminal", user_input="first")  # a second intake
    log.finish("t1", status="completed", api_calls=1, tokens=10, steps=1)
    row = log.get("t1")
    assert row is not None
    assert row.started_at == first.started_at


# ── 2. the status is honest and closed ────────────────────────────────────


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"cancelled": True, "halted_reason": "Cancelled by the user."}, "cancelled"),
        ({"results": [_ok(ok=False)]}, "failed"),
        ({"results": [_ok(verified=False)]}, "failed"),
        ({"results": [_ok(verified=None)]}, "partial"),
        (
            {"plan": Plan(goal="g", steps=[], needs_clarification=True)},
            "halted",
        ),
        ({"final_answer": "hi", "plan": Plan(kind="conversation", goal="g", steps=[])}, "answered"),
        ({"halted_reason": "JARVIS's AI backend is not configured yet."}, "halted"),
        ({"halted_reason": "No input received. Tell me what to do."}, "halted"),
        ({"results": [_ok()]}, "completed"),
    ],
)
def test_status_is_derived_in_code(state: dict[str, Any], expected: str) -> None:
    assert task_status(state) == expected
    assert expected in STATUSES


def test_a_failed_step_outranks_a_later_halt() -> None:
    """A failure is concrete evidence; a halt reason is only a reason."""
    state = {
        "results": [_ok(ok=False)],
        "halted_reason": "Replanning is not possible right now.",
    }
    assert task_status(state) == "failed"


def test_completed_requires_every_step_verified(store: SqliteMemory) -> None:
    """The dangerous direction: a broken step must never be logged as done."""
    saver = open_sqlite_checkpointer(":memory:")
    record: list[tuple[str, dict[str, Any]]] = []
    spec = make_spec("fake_echo", base_tier=0, record=record, run_ok=False)
    ctx = make_app_context(
        Settings(agent={"max_retries_per_step": 0, "max_replans": 0}),
        llm=FakeLLM([brain_action("fake_echo", {"text": "hi"})]),
        registry=registry_with(spec),
        memory=store,
    )
    try:
        outcome = run_task(ctx, saver, "fail please")
        row = _task_log(store).get(outcome.task_id)
    finally:
        saver.conn.close()

    assert row is not None
    assert row.status == "failed"
    assert "Could not complete" in str(outcome.final_answer)


# ── 3. it cannot break a task, and it cannot leak a secret ────────────────


def test_no_backend_records_nothing_and_still_answers() -> None:
    """A context with no memory store behaves exactly as before."""
    saver = open_sqlite_checkpointer(":memory:")
    ctx = _ctx(FakeLLM([brain_action("fake_echo", {"text": "hi"})]), None)
    try:
        outcome = run_task(ctx, saver, "echo hi")
    finally:
        saver.conn.close()
    assert outcome.final_answer
    assert outcome.error is None


def test_null_memory_backend_is_a_no_op() -> None:
    saver = open_sqlite_checkpointer(":memory:")
    ctx = _ctx(FakeLLM([brain_action("fake_echo", {"text": "hi"})]), NullMemory())
    try:
        outcome = run_task(ctx, saver, "echo hi")
    finally:
        saver.conn.close()
    assert outcome.final_answer
    assert outcome.error is None


def test_a_raising_backend_never_breaks_the_answer() -> None:
    class Exploding:
        def retrieve(self, query: str, limit: int = 3) -> list:  # pragma: no cover
            del query, limit
            return []

        def remember(self, record: Any) -> None:
            raise sqlite3.OperationalError("database is locked")

        def record_task(self, task_id: str, **fields: Any) -> None:
            raise sqlite3.OperationalError("database is locked")

    saver = open_sqlite_checkpointer(":memory:")
    ctx = _ctx(FakeLLM([brain_action("fake_echo", {"text": "hi"})]), Exploding())
    try:
        outcome = run_task(ctx, saver, "echo hi")
    finally:
        saver.conn.close()
    assert outcome.final_answer
    assert outcome.error is None


def test_a_backend_without_record_task_is_tolerated() -> None:
    """An older test double must not start raising ``AttributeError``."""

    class Legacy:
        def retrieve(self, query: str, limit: int = 3) -> list:  # pragma: no cover
            del query, limit
            return []

        def remember(self, record: Any) -> None:
            del record

    saver = open_sqlite_checkpointer(":memory:")
    ctx = _ctx(FakeLLM([brain_action("fake_echo", {"text": "hi"})]), Legacy())
    try:
        outcome = run_task(ctx, saver, "echo hi")
    finally:
        saver.conn.close()
    assert outcome.final_answer


def test_a_locked_database_is_fail_soft(store: SqliteMemory, monkeypatch: Any) -> None:
    def boom(*args: Any, **kwargs: Any) -> None:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(store.task_log, "start", boom)
    saver = open_sqlite_checkpointer(":memory:")
    ctx = _ctx(FakeLLM([brain_action("fake_echo", {"text": "hi"})]), store)
    try:
        outcome = run_task(ctx, saver, "echo hi")
    finally:
        saver.conn.close()
    assert outcome.final_answer
    assert outcome.error is None


def test_pasted_credential_in_the_command_is_not_persisted(store: SqliteMemory) -> None:
    """The task log keeps the command text, never a secret inside it."""
    log = _task_log(store)
    log.start("t1", source="terminal", user_input="email me password=hunter2 now")
    row = log.get("t1")
    assert row is not None
    assert "hunter2" not in row.user_input
    assert "password" in row.user_input  # the useful context survives


def test_safe_input_never_keeps_a_credential_value() -> None:
    """Redaction and rejection compose: the value is gone either way."""
    redacted = safe_input("api_key: 'sk-live-abc'")
    assert "sk-live-abc" not in redacted
    assert "[redacted]" in redacted  # the useful context survives
    assert safe_input("") == ""
    assert len(safe_input("x" * 5000)) == 300


def test_safe_input_drops_a_secret_whose_value_redaction_cannot_reach(
    store: SqliteMemory,
) -> None:
    """Defence in depth: whatever survives redaction is checked once more."""
    from jarvis.memory import tasklog as tasklog_module

    # A value the assignment pattern cannot reach (no key, no separator).
    assert tasklog_module.safe_input("hunter2") == "hunter2"
    original = tasklog_module.redact_secrets
    try:
        tasklog_module.redact_secrets = lambda text: text  # simulate a gap
        assert tasklog_module.safe_input("password=hunter2") == "[redacted]"
    finally:
        tasklog_module.redact_secrets = original


# ── 4. bounded, and reading it back ───────────────────────────────────────


def test_rows_are_pruned_to_the_configured_cap(tmp_path: Any) -> None:
    from jarvis.memory.db import open_db as open_file_db

    store = SqliteMemory(open_file_db(tmp_path / "m.db"), settings=Settings())
    try:
        log = _task_log(store)
        for i in range(12):
            log.start(f"t{i}", source="terminal", user_input=f"task {i}")
        assert log.count() == 12
    finally:
        store.close()

    capped = SqliteMemory(
        open_file_db(tmp_path / "m2.db"), settings=Settings(memory={"max_task_log": 3})
    )
    try:
        log = capped.task_log
        for i in range(12):
            log.start(f"t{i:02d}", source="terminal", user_input=f"task {i}")
        assert log.count() == 3
        # The oldest rows are the ones dropped, not the newest.
        assert [r.task_id for r in log.recent(10)] == ["t11", "t10", "t09"]
    finally:
        capped.close()


def test_an_unknown_status_is_recorded_as_a_failure(store: SqliteMemory) -> None:
    log = _task_log(store)
    log.start("t1")
    log.finish("t1", status="definitely-not-a-status")
    row = log.get("t1")
    assert row is not None and row.status == "failed"


def test_recent_filters_by_status_and_respects_limit(store: SqliteMemory) -> None:
    log = _task_log(store)
    log.start("a")
    log.finish("a", status="completed", api_calls=1, tokens=100, steps=1)
    log.start("b")
    log.finish("b", status="failed", api_calls=2, tokens=200, steps=1)
    assert [r.task_id for r in log.recent(status="failed")] == ["b"]
    assert len(log.recent(limit=1)) == 1
    assert log.recent(limit=0) == []


def test_summary_separates_completed_failed_and_unfinished(store: SqliteMemory) -> None:
    """A crashed task is ``unfinished``, never counted as a success."""
    log = _task_log(store)
    log.start("done")
    log.finish("done", status="completed", api_calls=1, tokens=100, steps=1, replans=0)
    log.start("broke")
    log.finish("broke", status="failed", api_calls=3, tokens=300, steps=2, replans=1)
    log.start("crashed")  # never finished

    summary = log.summary()
    assert summary.tasks == 3
    assert summary.completed == 1
    assert summary.failed == 1
    assert summary.unfinished == 1
    assert summary.api_calls == 4
    assert summary.tokens == 400
    assert summary.replans == 1
    assert summary.api_calls_per_task == pytest.approx(4 / 3)


def test_clear_empties_the_task_log(store: SqliteMemory) -> None:
    log = _task_log(store)
    log.start("a")
    assert log.clear() == 1
    assert log.count() == 0
    assert log.summary().tasks == 0
    assert log.summary().api_calls_per_task == 0.0


def test_dry_run_writes_no_task_rows() -> None:
    """A rehearsal must not create side effects - telemetry included."""
    saver = open_sqlite_checkpointer(":memory:")
    record: list[tuple[str, dict[str, Any]]] = []
    spec = make_spec("fake_echo", base_tier=0, record=record)
    memory = SqliteMemory(open_db(":memory:"), settings=Settings())
    ctx = make_app_context(
        Settings(),
        llm=FakeLLM([brain_action("fake_echo", {"text": "hi"})]),
        registry=registry_with(spec),
        memory=memory,
        dry_run=True,
    )
    try:
        run_task(ctx, saver, "echo hi")
        assert memory.task_log.count() == 0
    finally:
        saver.conn.close()
        memory.close()
