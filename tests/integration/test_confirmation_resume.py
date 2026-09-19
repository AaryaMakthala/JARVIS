"""End-to-end confirmation -> resume -> act -> verify flows.

These run the real compiled LangGraph on a file-backed SQLite checkpointer with
a fake tool and a ``FakeLLM``.  They deliberately carry no ``slow`` /
``windows_only`` / ``voice`` markers so the default CI set (check.ps1) runs them.
"""

from __future__ import annotations

import logging
from typing import Any

from jarvis.agent.context import make_app_context
from jarvis.agent.graph import build_graph, open_sqlite_checkpointer
from jarvis.agent.runner import resume_task, run_task
from jarvis.agent.state import Decision, Plan, Step, StepResult
from jarvis.config import AgentSettings, Settings
from jarvis.llm.client import FakeLLM
from support import approve, deny, echo_plan, make_spec, registry_with


def _ctx(record: list[tuple[str, dict[str, Any]]], settings: Settings | None = None) -> Any:
    return make_app_context(
        settings or Settings(),
        llm=FakeLLM([echo_plan()]),
        registry=registry_with(make_spec("fake_echo", base_tier=1, record=record)),
    )


def test_approve_runs_verifies_and_reports(tmp_path: Any) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    ctx = _ctx(record)
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        first = run_task(ctx, saver, "echo hi")
        assert first.interrupted is True
        assert first.confirmation == {
            "type": "confirm",
            "step_id": "s1",
            "tier": 1,
            "summary": "fake_echo text='hi'",
            "needs_unlock": False,
            "typed_confirmation": None,
            "action_hash": first.confirmation["action_hash"],
            "untrusted": False,
        }

        result = resume_task(ctx, saver, first.task_id, approve(first.confirmation))

        assert result.interrupted is False
        assert result.confirmation is None
        assert result.final_answer == "fake_echo ran hi"
        assert record == [("fake_echo", {"text": "hi"})]

        # The persisted state is restored as real models, not plain dicts.
        assert isinstance(result.state["plan"], Plan)
        assert isinstance(result.state["decisions"]["s1"], Decision)
        assert isinstance(result.state["results"][0], StepResult)
        assert result.state["results"][0].verified is True
        assert result.halted_reason is None
        assert result.error is None
    finally:
        saver.conn.close()


def test_deny_never_executes(tmp_path: Any) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    ctx = _ctx(record)
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        first = run_task(ctx, saver, "echo hi")
        result = resume_task(ctx, saver, first.task_id, deny(first.confirmation))
        assert record == []
        assert (
            result.final_answer
            == "Refused by you (confirmation answered with 'no' or a mismatched action)."
        )
        assert result.halted_reason is not None
    finally:
        saver.conn.close()


def test_tampered_hash_never_executes(tmp_path: Any) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    ctx = _ctx(record)
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        first = run_task(ctx, saver, "echo hi")
        forged = {"approved": True, "action_hash": "0" * 64}
        result = resume_task(ctx, saver, first.task_id, forged)
        assert record == []
        assert (
            result.final_answer
            == "Refused by you (confirmation answered with 'no' or a mismatched action)."
        )
    finally:
        saver.conn.close()


def test_modify_plan_after_approval_never_executes(tmp_path: Any) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    ctx = _ctx(record)
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        first = run_task(ctx, saver, "echo hi")
        graph = build_graph(ctx, saver)
        graph.update_state(
            {"configurable": {"thread_id": first.task_id}},
            {"plan": echo_plan(text="byee")},
        )
        result = resume_task(ctx, saver, first.task_id, approve(first.confirmation))
        assert record == []
        assert (
            result.final_answer
            == "Refused by you (confirmation answered with 'no' or a mismatched action)."
        )
    finally:
        saver.conn.close()


def test_unknown_tool_plan_is_rejected_end_to_end(tmp_path: Any) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    bad_plan = Plan(
        goal="do the thing",
        steps=[Step(id="s1", tool="no_such_tool", args={}, rationale="r")],
    )
    ctx = make_app_context(
        Settings(),
        llm=FakeLLM([bad_plan, bad_plan]),
        registry=registry_with(make_spec("fake_echo", base_tier=1, record=record)),
    )
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        result = run_task(ctx, saver, "do the thing")
        assert result.interrupted is False
        assert record == []
        assert result.final_answer == (
            "Could not produce a valid plan (after 2 attempts): "
            "step s1: tool 'no_such_tool' is not available"
        )
    finally:
        saver.conn.close()


def test_tier3_hard_block_reaches_respond_end_to_end(tmp_path: Any) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    defender = make_spec("disable_defender_x", base_tier=0, record=record)
    plan = Plan(
        goal="disable the thing",
        steps=[Step(id="s1", tool="disable_defender_x", args={"text": "on"}, rationale="r")],
    )
    ctx = make_app_context(
        Settings(),
        llm=FakeLLM([plan]),
        registry=registry_with(defender),
    )
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        result = run_task(ctx, saver, "disable the thing")
        assert record == []
        assert result.interrupted is False
        assert result.final_answer is not None
        assert result.final_answer.startswith("Refused: ")
        assert "hard-blocked by policy" in result.final_answer
    finally:
        saver.conn.close()


def test_tier0_step_skips_confirmation_and_is_verified(tmp_path: Any) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    auto = make_spec("fake_auto", base_tier=0, record=record)
    plan = Plan(
        goal="run auto",
        steps=[Step(id="s1", tool="fake_auto", args={"text": "x"}, rationale="r")],
    )
    ctx = make_app_context(Settings(), llm=FakeLLM([plan]), registry=registry_with(auto))
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        result = run_task(ctx, saver, "run auto")
        assert result.interrupted is False
        assert record == [("fake_auto", {"text": "x"})]
        assert result.final_answer == "fake_auto ran x"
    finally:
        saver.conn.close()


def test_verification_failure_is_reported_honestly_end_to_end(tmp_path: Any) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    auto = make_spec("fake_auto", base_tier=0, record=record, verify_ok=False)
    plan = Plan(
        goal="run auto",
        steps=[Step(id="s1", tool="fake_auto", args={"text": "x"}, rationale="r")],
    )
    settings = Settings(agent=AgentSettings(max_retries_per_step=1))
    ctx = make_app_context(settings, llm=FakeLLM([plan]), registry=registry_with(auto))
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        result = run_task(ctx, saver, "run auto")
        assert result.interrupted is False
        assert len(record) == 2  # 1 attempt + 1 retry (max_retries_per_step=1)
        answer = result.final_answer or ""
        assert answer.startswith("Could not complete the task")
        assert "could not be verified" in answer
    finally:
        saver.conn.close()


def test_simulated_daemon_restart_resume(tmp_path: Any) -> None:
    """A fresh connection on the same DB file can resume the same thread."""
    record: list[tuple[str, dict[str, Any]]] = []
    ctx = _ctx(record)
    db = tmp_path / "c.db"
    saver1 = open_sqlite_checkpointer(str(db))
    try:
        first = run_task(ctx, saver1, "echo hi")
    finally:
        saver1.conn.close()

    saver2 = open_sqlite_checkpointer(str(db))
    try:
        result = resume_task(ctx, saver2, first.task_id, approve(first.confirmation))
        assert result.final_answer == "fake_echo ran hi"
        assert record == [("fake_echo", {"text": "hi"})]
    finally:
        saver2.conn.close()


def test_no_blocked_serde_warnings_across_full_flow(tmp_path: Any, caplog: Any) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    ctx = _ctx(record)
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        with caplog.at_level(logging.WARNING, logger="langgraph.checkpoint.serde.jsonplus"):
            first = run_task(ctx, saver, "echo hi")
            resume_task(ctx, saver, first.task_id, approve(first.confirmation))
        assert "Blocked deserialization" not in caplog.text
        assert "Deserializing unregistered" not in caplog.text
    finally:
        saver.conn.close()
