"""Phase 2 boundary schemas: defaults, validation, (de)serialisation safety.

Every boundary model lives in ``jarvis.agent.schemas`` and must stay plain
JSON-able (no secrets, no arbitrary objects) so it can round-trip through the
checkpoint serializer and the daemon IPC.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from jarvis.agent.schemas import (
    AgentResponse,
    ConfirmationRequest,
    ReplanDecision,
    ToolCall,
    UserRequest,
)
from jarvis.agent.state import Plan, Step


def test_user_request_defaults() -> None:
    r = UserRequest(text="hello")
    assert r.text == "hello"
    assert r.source in ("terminal", "voice", "benchmark")
    # no undocumented fields sneak in
    assert set(r.model_dump()) == {"text", "source"}


def test_user_request_rejects_empty_text() -> None:
    with pytest.raises(ValidationError):
        UserRequest(text="   ")


def test_user_request_rejects_unknown_source_and_extra_fields() -> None:
    with pytest.raises(ValidationError):
        UserRequest(text="x", source="shell")
    with pytest.raises(ValidationError):
        UserRequest(text="x", password="hunter2")  # type: ignore[call-arg]


def test_tool_call_rejects_secret_field_with_extra_forbid() -> None:
    with pytest.raises(ValidationError):
        ToolCall(tool="app_launch", args={}, status="succeeded", password="x")  # type: ignore[call-arg]


def test_replan_decision_stop_allows_message_without_plan() -> None:
    d = ReplanDecision(action="stop", message="give up")
    assert d.plan is None
    assert d.action == "stop"


def test_replan_decision_continue_requires_plan() -> None:
    with pytest.raises(ValidationError):
        ReplanDecision(action="continue", message="no plan given")


def test_replan_decision_unknown_action_rejected() -> None:
    with pytest.raises(ValidationError):
        ReplanDecision(action="halt")  # type: ignore[arg-type]


def test_confirmation_request_is_fully_json_serialisable() -> None:
    c = ConfirmationRequest(
        step_id="s1",
        tier=2,
        summary="delete x",
        action_hash="a" * 64,
        resolved_paths=["C:/tmp/x"],
        needs_unlock=True,
        typed_confirmation=None,
    )
    data: dict[str, Any] = c.model_dump(mode="json")
    assert data["action_hash"] == "a" * 64
    assert data["typed_confirmation"] is None
    # Round-trips through plain JSON without losing anything.
    assert ConfirmationRequest.model_validate(data) == c


def test_agent_response_kind_matches_plan_kind() -> None:
    r = AgentResponse(task_id="t", kind="tool", text="done")
    assert r.kind == "tool"
    with pytest.raises(ValidationError):
        AgentResponse(task_id="t", kind="shell", text="nope")  # type: ignore[arg-type]


def test_plan_kind_defaults_to_tool() -> None:
    p = Plan(goal="g", steps=[Step(id="s1", tool="x", args={}, rationale="r")])
    assert p.kind == "tool"
    assert p.dialog_answer is None


def test_plan_allows_conversation_without_steps() -> None:
    p = Plan(kind="conversation", goal="answer directly", steps=[], dialog_answer="hi")
    assert p.kind == "conversation"


def test_plan_rejects_unknown_kind() -> None:
    with pytest.raises(ValidationError):
        Plan(kind="monologue", goal="g")  # type: ignore[arg-type]
