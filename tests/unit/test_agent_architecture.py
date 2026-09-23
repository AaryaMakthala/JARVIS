"""Phase 2 architecture tests: the whole compiled graph, task to response.

These drive the graph through the public runner (``run_task`` / ``resume_task``)
with a scripted FakeLLM and a real SQLite checkpointer, covering the new
conversation vs. tool split, replanning, eager/policy-gated execution and
worker resilience.  Each test declares the *exact* FakeLLM script it consumes.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from jarvis.agent.context import make_app_context
from jarvis.agent.graph import open_sqlite_checkpointer
from jarvis.agent.runner import resume_task, run_task
from jarvis.agent.state import Plan, Step
from jarvis.config import AgentSettings, Settings
from jarvis.llm.client import FakeLLM
from support import (
    approve,
    conversation_plan,
    deny,
    echo_plan,
    make_spec,
    registry_with,
    replan_continue,
    replan_stop,
    tamper,
)


@pytest.fixture
def saver(tmp_path: Path) -> Any:
    conn = sqlite3.connect(tmp_path / "c.db", check_same_thread=False)
    yield open_sqlite_checkpointer(str(tmp_path / "c.db"))
    conn.close()


def _eager_spec(name: str, record: list[tuple[str, dict[str, Any]]]) -> Any:
    return make_spec(name, base_tier=0, record=record)


def _confirmed_spec(name: str, record: list[tuple[str, dict[str, Any]]]) -> Any:
    return make_spec(name, base_tier=1, record=record)


def _echo_plan_tool0() -> Plan:
    return Plan(
        goal="echo hi",
        steps=[Step(id="s1", tool="fake_echo", args={"text": "hi"}, rationale="echo it")],
    )


def _failing_echo_spec(record: list[tuple[str, dict[str, Any]]]) -> Any:
    """A ``fake_echo`` that always fails at run time (drives the replan path)."""
    return make_spec("fake_echo", base_tier=0, record=record, run_ok=False)


def _two_step_plan() -> Plan:
    return Plan(
        goal="ab",
        steps=[
            Step(id="s1", tool="fake_a", args={"text": "A"}, rationale="first"),
            Step(id="s2", tool="fake_b", args={"text": "B"}, rationale="second"),
        ],
    )


def _replacement_plan() -> Plan:
    return Plan(
        goal="repair",
        steps=[Step(id="s2", tool="fake_fix", args={"text": "fixed"}, rationale="repair path")],
    )


def _clarification_plan() -> Plan:
    return Plan(
        goal="clarify",
        steps=[],
        needs_clarification=True,
        clarification_question="Which file did you mean?",
    )


def _conv_no_answer() -> Plan:
    return Plan(kind="conversation", goal="answer", steps=[], dialog_answer=None)


# ── conversation vs. tool routing ──────────────────────────────────────────


def test_conversation_answer_written_by_planner_skips_llm_and_runs_no_tools(
    saver: Any,
) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    ctx = make_app_context(
        Settings(),
        llm=FakeLLM([conversation_plan("Hi there!")]),
        registry=registry_with(_eager_spec("fake_echo", record)),
    )
    out = run_task(ctx, saver, "hello jarvis")
    assert out.interrupted is False
    assert out.final_answer == "Hi there!"
    assert record == []  # nothing was ever executed
    assert out.state["request_kind"] == "conversation"
    assert out.agent_response.kind == "conversation"
    assert out.agent_response.interrupted is False


def test_conversation_answer_via_fast_model_when_planner_wrote_none(saver: Any) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    llm = FakeLLM([_conv_no_answer(), "Sure, I can help!"])
    ctx = make_app_context(
        Settings(),
        llm=llm,
        registry=registry_with(_eager_spec("fake_echo", record)),
    )
    out = run_task(ctx, saver, "tell me a joke")
    assert out.final_answer == "Sure, I can help!"
    assert record == []
    assert ("text", "fast", "str") in llm.calls


def test_no_llm_backend_halts_cleanly(saver: Any) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    ctx = make_app_context(
        Settings(),
        llm=None,
        registry=registry_with(_eager_spec("fake_echo", record)),
    )
    out = run_task(ctx, saver, "do something")
    assert out.final_answer == "No LLM backend configured (run `jarvis init`)."
    assert record == []


# ── tool execution paths ───────────────────────────────────────────────────


def test_single_eager_tool_executes(saver: Any) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    ctx = make_app_context(
        Settings(),
        llm=FakeLLM([_echo_plan_tool0()]),
        registry=registry_with(_eager_spec("fake_echo", record)),
    )
    out = run_task(ctx, saver, "echo hi")
    assert out.interrupted is False
    assert record == [("fake_echo", {"text": "hi"})]
    assert out.final_answer == "fake_echo ran hi"
    assert out.state["request_kind"] == "tool"


def test_multi_step_eager_plan_runs_in_order(saver: Any) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    ctx = make_app_context(
        Settings(),
        llm=FakeLLM([_two_step_plan()]),
        registry=registry_with(_eager_spec("fake_a", record), _eager_spec("fake_b", record)),
    )
    out = run_task(ctx, saver, "a then b")
    assert record == [("fake_a", {"text": "A"}), ("fake_b", {"text": "B"})]
    assert out.final_answer.count("fake_a ran A") == 1
    assert out.final_answer.count("fake_b ran B") == 1


def test_confirmation_required_suspends_then_approval_executes(saver: Any) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    ctx = make_app_context(
        Settings(),
        llm=FakeLLM([echo_plan()]),
        registry=registry_with(_confirmed_spec("fake_echo", record)),
    )
    first = run_task(ctx, saver, "echo hi")
    assert first.interrupted is True
    assert first.confirmation is not None and first.confirmation["step_id"] == "s1"
    assert record == []  # nothing happens before the approval

    resumed = resume_task(ctx, saver, first.task_id, approve(first.confirmation))
    assert resumed.interrupted is False
    assert record == [("fake_echo", {"text": "hi"})]
    assert resumed.final_answer == "fake_echo ran hi"


def test_confirmation_rejected_never_runs(saver: Any) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    ctx = make_app_context(
        Settings(),
        llm=FakeLLM([echo_plan()]),
        registry=registry_with(_confirmed_spec("fake_echo", record)),
    )
    first = run_task(ctx, saver, "echo hi")
    resumed = resume_task(ctx, saver, first.task_id, deny(first.confirmation))
    assert record == []
    assert "Refused by you" in (resumed.final_answer or "")


def test_tampered_confirmation_hash_is_refused(saver: Any) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    ctx = make_app_context(
        Settings(),
        llm=FakeLLM([echo_plan()]),
        registry=registry_with(_confirmed_spec("fake_echo", record)),
    )
    first = run_task(ctx, saver, "echo hi")
    resumed = resume_task(ctx, saver, first.task_id, tamper(first.confirmation))
    assert record == []
    assert "Refused" in (resumed.final_answer or "")


def test_clarification_plan_does_not_route_to_tools(saver: Any) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    ctx = make_app_context(
        Settings(),
        llm=FakeLLM([_clarification_plan()]),
        registry=registry_with(_eager_spec("fake_echo", record)),
    )
    out = run_task(ctx, saver, "open the file")
    assert record == []
    assert "Clarification needed: Which file did you mean?" in (out.final_answer or "")


# ── planner repair and failure handling ────────────────────────────────────


def test_unknown_tool_plan_is_rejected_then_repaired(saver: Any) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    bad = Plan(
        goal="x",
        steps=[Step(id="s1", tool="no_such_tool", args={}, rationale="nope")],
    )
    ctx = make_app_context(
        Settings(),
        llm=FakeLLM([bad, echo_plan()]),
        registry=registry_with(_eager_spec("fake_echo", record)),
    )
    out = run_task(ctx, saver, "echo hi")
    assert record == [("fake_echo", {"text": "hi"})]
    assert out.final_answer == "fake_echo ran hi"


def test_invalid_args_plan_is_rejected_then_repaired(saver: Any) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    bad = Plan(
        goal="x",
        steps=[Step(id="s1", tool="fake_echo", args={"text": ""}, rationale="empty")],
    )
    ctx = make_app_context(
        Settings(),
        llm=FakeLLM([bad, echo_plan()]),
        registry=registry_with(_eager_spec("fake_echo", record)),
    )
    out = run_task(ctx, saver, "echo hi")
    assert record == [("fake_echo", {"text": "hi"})]
    assert out.final_answer == "fake_echo ran hi"


def test_malformed_planner_output_halts_honestly(saver: Any) -> None:
    ctx = make_app_context(
        Settings(),
        llm=FakeLLM(["not a plan at all"]),
        registry=registry_with(_eager_spec("fake_echo", [])),
    )
    out = run_task(ctx, saver, "echo hi")
    assert "Planner error" in (out.final_answer or "")
    assert out.error is None  # a halt, not a crash


# ── replanning ─────────────────────────────────────────────────────────────


def test_tool_failure_replans_then_stops_with_honest_answer(saver: Any) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    ctx = make_app_context(
        Settings(),
        llm=FakeLLM([_echo_plan_tool0(), replan_stop("that approach will not work")]),
        registry=registry_with(_failing_echo_spec(record)),
    )
    out = run_task(ctx, saver, "echo hi")
    # fail -> retry(1) -> retry(2) -> replan -> stop
    assert record == [("fake_echo", {"text": "hi"})] * 3
    assert "Could not complete the task: that approach will not work." in (out.final_answer or "")
    assert out.state["replan_count"] == 0  # stop does not consume the budget


def test_replan_continue_executes_replacement(saver: Any) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    fix = _eager_spec("fake_fix", record)
    ctx = make_app_context(
        Settings(),
        llm=FakeLLM([_echo_plan_tool0(), replan_continue(_replacement_plan())]),
        registry=registry_with(_failing_echo_spec(record), fix),
    )
    out = run_task(ctx, saver, "echo hi")
    assert record == [("fake_echo", {"text": "hi"})] * 3 + [("fake_fix", {"text": "fixed"})]
    assert out.final_answer == "fake_fix ran fixed"
    assert out.state["replan_count"] == 1


def test_replan_budget_exhaustion_stops_fail_closed(saver: Any) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    settings = Settings(agent=AgentSettings(max_retries_per_step=1, max_replans=1))
    # The replacement plan is invalid (unknown tool), so validate rejects it and
    # routes back to replan, which is now over budget - enforced in code, and
    # the LLM is never called again for the second replan.
    bad_replacement = Plan(
        goal="bad repair",
        steps=[Step(id="s2", tool="no_such_tool", args={}, rationale="wrong")],
    )
    ctx = make_app_context(
        settings,
        llm=FakeLLM([_echo_plan_tool0(), replan_continue(bad_replacement)]),
        registry=registry_with(_failing_echo_spec(record)),
    )
    out = run_task(ctx, saver, "echo hi")
    assert "replan budget is exhausted" in (out.final_answer or "")
    # Only the initial plan's 2 retries ran; the invalid replacement never
    # reached the gate (validate refused it before any side effect).
    assert record == [("fake_echo", {"text": "hi"})] * 2


def test_replan_keeps_completed_successes_in_final_state(saver: Any) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    ctx = make_app_context(
        Settings(),
        llm=FakeLLM([_two_step_plan(), replan_stop("giving up")]),
        registry=registry_with(
            _eager_spec("fake_a", record),
            make_spec("fake_b", base_tier=0, record=record, run_ok=False),
        ),
    )
    # s1 (fake_a) succeeds and advances step_index; s2 (fake_b) fails -> replan -> stop.
    out = run_task(ctx, saver, "a then b")
    assert record[:1] == [("fake_a", {"text": "A"})]
    assert ("fake_b", {"text": "B"}) in record
    assert "Could not complete" in (out.final_answer or "")
    completed = [r for r in out.results if r.ok and r.verified is not False]
    assert [r.step_id for r in completed] == ["s1"]


# ── worker resilience / voice source ───────────────────────────────────────


def test_failure_does_not_kill_the_worker(saver: Any) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    ctx = make_app_context(
        Settings(),
        llm=FakeLLM([_echo_plan_tool0(), replan_stop("nope")]),
        registry=registry_with(_failing_echo_spec(record)),
    )
    first = run_task(ctx, saver, "echo hi")
    assert "Could not complete" in (first.final_answer or "")
    # The same checkpointer/context serves a brand-new task afterwards: a halt
    # is a normal terminal state, not a worker crash.
    ctx2 = make_app_context(
        Settings(),
        llm=FakeLLM([_echo_plan_tool0()]),
        registry=registry_with(_eager_spec("fake_echo", record)),
    )
    happy = run_task(ctx2, saver, "echo hi", thread_id="t2")
    assert happy.final_answer == "fake_echo ran hi"


def test_voice_source_flows_through_conversation_path(saver: Any) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    ctx = make_app_context(
        Settings(),
        llm=FakeLLM([conversation_plan("hi back")]),
        registry=registry_with(_eager_spec("fake_echo", record)),
    )
    out = run_task(ctx, saver, "hello", source="voice")
    assert out.state["source"] == "voice"
    assert out.final_answer == "hi back"
    assert out.agent_response.kind == "conversation"
    assert record == []
