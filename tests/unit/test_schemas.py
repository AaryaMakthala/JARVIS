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
    ActionIntent,
    AgentResponse,
    BrainDecision,
    ClarificationRequest,
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


def test_action_intent_defaults_and_extra_forbid() -> None:
    a = ActionIntent(tool="app_launch", args={"name": "notepad"})
    assert a.rationale == ""
    assert a.expect == ""
    assert a.depends_on_untrusted is False
    assert set(a.model_dump()) == {
        "tool",
        "args",
        "rationale",
        "expect",
        "depends_on_untrusted",
    }
    with pytest.raises(ValidationError):
        ActionIntent(tool="app_launch", args={}, action_hash="x")  # type: ignore[call-arg]


def test_brain_decision_conversation_shape() -> None:
    d = BrainDecision(request_type="conversation", response_text="hi", goal="answer")
    assert d.actions == []
    assert d.clarification_question is None
    # round-trips through plain JSON
    assert BrainDecision.model_validate(d.model_dump(mode="json")) == d


def test_brain_decision_action_requires_actions() -> None:
    with pytest.raises(ValidationError):
        BrainDecision(request_type="action", goal="do something")
    d = BrainDecision(
        request_type="action",
        goal="open notepad",
        actions=[ActionIntent(tool="app_launch", args={"name": "notepad"}, rationale="r")],
    )
    assert d.actions[0].tool == "app_launch"


def test_brain_decision_clarification_requires_question() -> None:
    with pytest.raises(ValidationError):
        BrainDecision(request_type="clarification", goal="?")

    with pytest.raises(ValidationError):
        BrainDecision(request_type="clarification", clarification_question="   ", goal="?")

    d = BrainDecision(request_type="clarification", clarification_question="which file?", goal="?")
    assert d.clarification_question == "which file?"


def test_brain_decision_unknown_request_type_and_extra_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        BrainDecision(request_type="shell")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        BrainDecision(request_type="conversation", response_text="hi", tier=3)  # type: ignore[call-arg]


def test_brain_decision_and_action_intent_are_transient_by_contract() -> None:
    # Serialization-side contract: the checkpoint serde must NOT revive a
    # BrainDecision/ActionIntent as a typed object.  The brain node projects
    # into the allowlisted Plan/Step shape before anything touches AgentState,
    # so these LLM-only types round-trip as plain data (never resurrected).
    from jarvis.agent.graph import build_secure_serde

    serde = build_secure_serde()

    brain = BrainDecision(
        request_type="action",
        actions=[ActionIntent(tool="app_launch", args={"name": "notepad"}, rationale="r")],
    )
    out = serde.loads_typed(serde.dumps_typed(brain))
    assert type(out) is dict
    assert out["request_type"] == "action"

    intent = ActionIntent(tool="app_launch", args={"name": "notepad"})
    out2 = serde.loads_typed(serde.dumps_typed(intent))
    assert type(out2) is dict
    assert out2["tool"] == "app_launch"

    # the checkpointed representation is the Plan/Step pair, revived as types
    plan = Plan(
        goal="g",
        steps=[Step(id="s1", tool="app_launch", args={"name": "notepad"}, rationale="r")],
    )
    out3 = serde.loads_typed(serde.dumps_typed(plan))
    assert type(out3) is Plan


def test_clarification_request_json_round_trip() -> None:
    c = ClarificationRequest(question="which file should I edit?")
    assert c.type == "clarification"
    data = c.model_dump(mode="json")
    assert data["question"] == "which file should I edit?"
    assert ClarificationRequest.model_validate(data) == c
    with pytest.raises(ValidationError):
        ClarificationRequest(question="   ")
    with pytest.raises(ValidationError):
        ClarificationRequest(question="x", confirmed=True)  # type: ignore[call-arg]
