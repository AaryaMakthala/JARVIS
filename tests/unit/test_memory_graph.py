"""End-to-end Phase 8 memory through the real compiled graph.

These are the acceptance criteria from ``docs/05_BUILD_PLAN.md`` Phase 8, run
against the *actual* graph (``build_graph`` + ``run_task``), not a hand-wired
set of node calls:

* "Repeating a command retrieves the earlier skill."
* A task that is not fully verified never becomes a skill.
* Memory survives a restart (the SQLite file is the store).
* ``--dry-run`` writes nothing.
"""

from __future__ import annotations

from typing import Any

import pytest

from jarvis.agent.context import make_app_context
from jarvis.agent.graph import open_sqlite_checkpointer
from jarvis.agent.runner import run_task
from jarvis.config import Settings
from jarvis.llm.client import FakeLLM
from jarvis.memory import NullMemory, SqliteMemory, open_db
from support import brain_action, make_spec, registry_with


def _close(saver: Any) -> None:
    """Close a checkpointer connection if the installed saver supports it."""
    closer = getattr(saver, "conn", None)
    if closer is not None:
        closer.close()


def _ctx(llm: Any, memory: Any, *, dry_run: bool = False, **kwargs: Any):
    record: list[tuple[str, dict[str, Any]]] = []
    spec = make_spec("fake_echo", base_tier=0, record=record)
    return make_app_context(
        Settings(**kwargs),
        llm=llm,
        registry=registry_with(spec),
        memory=memory,
        dry_run=dry_run,
    )


@pytest.fixture
def store() -> SqliteMemory:
    memory = SqliteMemory(open_db(":memory:"), settings=Settings())
    try:
        yield memory
    finally:
        memory.close()


def test_repeating_a_command_retrieves_the_earlier_skill(store: SqliteMemory) -> None:
    goal = "echo hello"
    saver = open_sqlite_checkpointer(":memory:")
    ctx = _ctx(FakeLLM([brain_action("fake_echo", {"text": "hello"})]), store)
    try:
        first = run_task(ctx, saver, goal)
        assert first.state.get("memory_saved") is True
        assert store.counts()["skills"] == 1
    finally:
        _close(saver)

    # A *new* process would see the same row here; a new task certainly does.
    second = _ctx(FakeLLM([brain_action("fake_echo", {"text": "hello"})]), store)
    saver2 = open_sqlite_checkpointer(":memory:")
    try:
        outcome = run_task(second, saver2, goal)
        assert outcome.state.get("memory_saved") is True
    finally:
        _close(saver2)

    # The second run retrieved the first one instead of minting a duplicate.
    assert store.counts()["skills"] == 1
    assert store.skills.all()[0].success_count == 2
    assert store.skills.all()[0].tools_used == ["fake_echo"]


def test_memory_context_reaches_the_planner(store: SqliteMemory) -> None:
    """The retrieved example is actually handed to ``brain`` on the next run."""
    store.skills.save("echo hello", plan={"kind": "tool", "goal": "echo hello", "steps": []})

    prompts: list[str] = []

    class SpyLLM(FakeLLM):
        def structured(self, *, system: str, user: str, **kwargs: Any) -> tuple[Any, Any]:
            prompts.append(user)
            return super().structured(system=system, user=user, **kwargs)

    saver = open_sqlite_checkpointer(":memory:")
    ctx = _ctx(SpyLLM([brain_action("fake_echo", {"text": "hello"})]), store)
    try:
        run_task(ctx, saver, "echo hello")
    finally:
        _close(saver)
    assert prompts, "the brain LLM was never called"
    assert any("Earlier successful approach" in prompt for prompt in prompts)
    assert any("echo hello" in prompt for prompt in prompts)


def test_an_unverified_task_is_never_saved_as_a_skill(store: SqliteMemory) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    spec = make_spec("fake_echo", base_tier=0, record=record, verify_ok=False)
    ctx = make_app_context(
        Settings(),
        llm=FakeLLM([brain_action("fake_echo", {"text": "hello"})]),
        registry=registry_with(spec),
        memory=store,
    )
    saver = open_sqlite_checkpointer(":memory:")
    try:
        outcome = run_task(ctx, saver, "echo hello")
    finally:
        _close(saver)
    assert outcome.state.get("memory_saved") is False
    assert store.counts()["skills"] == 0
    assert store.counts()["failures"] == 1


def test_a_conversation_is_never_saved(store: SqliteMemory) -> None:
    from support import brain_conversation

    saver = open_sqlite_checkpointer(":memory:")
    ctx = _ctx(FakeLLM([brain_conversation("hi there")]), store)
    try:
        outcome = run_task(ctx, saver, "hello there")
    finally:
        _close(saver)
    assert outcome.state.get("memory_saved") is False
    assert store.counts() == {"skills": 0, "failures": 0, "preferences": 0}


def test_dry_run_writes_nothing(store: SqliteMemory) -> None:
    saver = open_sqlite_checkpointer(":memory:")
    ctx = _ctx(FakeLLM([brain_action("fake_echo", {"text": "hello"})]), store, dry_run=True)
    try:
        outcome = run_task(ctx, saver, "echo hello")
    finally:
        _close(saver)
    assert outcome.state.get("memory_saved") is False
    assert store.counts()["skills"] == 0


def test_without_a_backend_nothing_breaks() -> None:
    saver = open_sqlite_checkpointer(":memory:")
    ctx = _ctx(FakeLLM([brain_action("fake_echo", {"text": "hello"})]), None)
    try:
        outcome = run_task(ctx, saver, "echo hello")
    finally:
        _close(saver)
    assert outcome.state.get("memory_saved") is False
    assert outcome.state["memory_saved_reason"] == "no memory backend"
    assert outcome.final_answer


def test_null_memory_backend_is_transparent() -> None:
    saver = open_sqlite_checkpointer(":memory:")
    ctx = _ctx(FakeLLM([brain_action("fake_echo", {"text": "hello"})]), NullMemory())
    try:
        outcome = run_task(ctx, saver, "echo hello")
    finally:
        _close(saver)
    assert outcome.final_answer


def test_skills_survive_a_restart(tmp_path) -> None:
    """The SQLite file, not the process, is the memory."""
    from jarvis.memory import open_memory

    first = open_memory(tmp_path / "memory.db", settings=Settings())
    assert isinstance(first, SqliteMemory)
    first.skills.save("echo hello", plan={"kind": "tool", "goal": "echo hello", "steps": []})
    first.close()

    second = open_memory(tmp_path / "memory.db", settings=Settings())
    try:
        assert second.counts()["skills"] == 1
        assert [s.goal_text for s in second.skills.find("echo hello")] == ["echo hello"]
    finally:
        second.close()
