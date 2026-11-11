"""C4: one canonical outcome classifier for wording, telemetry and storage.

The live defect: compose classified a successful date/time read as success (and
said "Done."), while telemetry logged every observation as ``partial`` because
``verified is None``.  A read-only observation has no post-condition to verify,
so its own successful result *is* the verification.
"""

from __future__ import annotations

from typing import Any

import pytest

from jarvis.agent.compose_rules import (
    STATUS_BY_OUTCOME,
    classify_outcome,
    spoken_answer,
    task_status_for,
)
from jarvis.agent.context import make_app_context
from jarvis.agent.nodes.telemetry import record_finished, record_started, task_status
from jarvis.agent.state import Plan, Step, StepResult
from jarvis.config import Settings
from jarvis.memory import SqliteMemory, open_db
from jarvis.memory.tasklog import STATUSES
from jarvis.tools.registry import ToolRegistry


def _plan(*tools: str, kind: str = "tool") -> Plan:
    steps = [Step(id=f"s{i}", tool=tool, args={}, rationale="r") for i, tool in enumerate(tools, 1)]
    return Plan(kind=kind, goal="g", steps=steps)


def _result(tool: str, **kwargs: Any) -> StepResult:
    fields: dict[str, Any] = {"step_id": "s1", "ok": True, "output": "ok"}
    fields.update(kwargs)
    return StepResult(**fields)


def _states() -> list[dict[str, Any]]:
    return [
        {"cancelled": True, "halted_reason": "Cancelled by the user."},
        {"plan": _plan("fake_echo"), "results": [_result("fake_echo", ok=False, error="boom")]},
        {"plan": _plan("fake_echo"), "results": [_result("fake_echo", verified=False)]},
        {"plan": _plan("type_text"), "results": [_result("type_text", verified=None)]},
        {"plan": _plan("get_time"), "results": [_result("get_time", verified=None)]},
        {"plan": Plan(goal="g", steps=[], needs_clarification=True)},
        {
            "plan": Plan(kind="conversation", goal="g", steps=[]),
            "final_answer": "A programming language.",
        },
        {"halted_reason": "JARVIS's AI backend is not configured yet."},
        {"halted_reason": "I did not perform the action: it is not allowed."},
        {"plan": _plan("fake_echo"), "results": [_result("fake_echo", verified=True)]},
        {"request_kind": "unsupported", "halted_reason": "unsupported request"},
        {"plan": _plan(), "results": []},
    ]


@pytest.mark.parametrize("state", _states())
def test_compose_and_telemetry_cannot_disagree(state: dict[str, Any]) -> None:
    outcome = classify_outcome(state)
    assert task_status(state) == task_status_for(state) == STATUS_BY_OUTCOME[outcome]
    assert task_status(state) in STATUSES


def test_a_date_observation_is_completed_not_partial() -> None:
    state = {
        "task_id": "t1",
        "plan": _plan("get_time"),
        "results": [_result("get_time", output="2026-11-10", verified=None)],
        "final_answer": "Today is Tuesday, 10 November 2026.",
    }
    assert classify_outcome(state) == "success"
    assert task_status(state) == "completed"
    assert "could not verify" not in spoken_answer(state, "success").casefold()


def test_an_unverified_side_effect_stays_partial() -> None:
    state = {"plan": _plan("type_text"), "results": [_result("type_text", verified=None)]}
    assert task_status(state) == "partial"


def test_the_stored_row_uses_the_same_status() -> None:
    memory = SqliteMemory(open_db(":memory:"), settings=Settings())
    try:
        ctx = make_app_context(Settings(), registry=ToolRegistry(), memory=memory)
        state = {
            "task_id": "00000000-0000-4000-8000-000000000401",
            "source": "terminal",
            "user_input": "what is the time",
            "plan": _plan("get_time"),
            "results": [_result("get_time", output="10:00", verified=None)],
            "final_answer": "It is 10:00.",
        }
        record_started(state, ctx)
        record_finished(state, ctx)

        row = memory.task_log.get(state["task_id"])
        assert row is not None
        assert row.status == task_status(state) == "completed"
    finally:
        memory.close()
