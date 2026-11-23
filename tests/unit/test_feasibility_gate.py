"""Phase 8 acceptance tests for pre-planning feasibility and honest retries."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from pydantic import BaseModel

from jarvis.agent import feasibility
from jarvis.agent.compose_rules import classify_outcome, spoken_answer, task_status_for
from jarvis.agent.context import make_app_context
from jarvis.agent.nodes.brain import brain
from jarvis.agent.nodes.memory_save import _failure_record, _skill_gate
from jarvis.agent.nodes.replan import _completed_results
from jarvis.agent.nodes.respond import base_answer
from jarvis.agent.nodes.validate import validate
from jarvis.agent.results import effective_results
from jarvis.agent.schemas import BrainDecision
from jarvis.agent.state import Plan, Step, StepResult
from jarvis.config import Settings
from jarvis.llm.client import Usage
from support import brain_action, make_spec, registry_with


class ScriptedLLM:
    """Structured-output double that records prompts and call count."""

    def __init__(self, responses: list[BrainDecision]) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []
        self.prompts: list[tuple[str, str]] = []

    def structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[BaseModel],
        model_role: str = "planner",
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> tuple[BrainDecision, Usage]:
        self.prompts.append((system, user))
        self.calls.append(model_role)
        if not self.responses:
            raise AssertionError("scripted LLM exhausted")
        return self.responses.pop(0), Usage(calls=1, prompt_tokens=5, completion_tokens=3)


def _state(text: str, **extra: Any) -> dict[str, Any]:
    return {"task_id": "t-p8", "source": "terminal", "user_input": text, **extra}


def test_conversation_passes_the_feasibility_gate() -> None:
    ctx = make_app_context(Settings())
    report = feasibility.assess_request(_state("Explain Python decorators."), ctx)
    assert report.status == feasibility.SUPPORTED
    assert report.available_tools == ()


def test_unsupported_travel_request_stops_before_any_llm_call() -> None:
    llm = ScriptedLLM([])
    ctx = make_app_context(Settings(), llm=llm)
    out = brain(_state("Please book an airline ticket to Delhi."), ctx)
    assert out["request_kind"] == "unsupported"
    assert "book a flight" in str(out["halted_reason"])
    assert llm.calls == []


def test_whatsapp_without_the_isolated_profile_is_reported_unavailable() -> None:
    ctx = make_app_context(Settings())
    report = feasibility.assess_request(_state("Send a WhatsApp message to mom."), ctx)
    assert report.status == feasibility.UNSUPPORTED
    assert "whatsapp" in report.reason.casefold()


def test_missing_referent_becomes_a_clarification_before_planning() -> None:
    ctx = make_app_context(Settings())
    report = feasibility.assess_request(_state("Send this message on WhatsApp."), ctx)
    assert report.status == feasibility.NEEDS_CLARIFICATION


def test_partial_request_adds_a_bounded_prompt_constraint_and_plan_note() -> None:
    llm = ScriptedLLM([brain_action("open_app", {"name": "notepad"})])
    ctx = make_app_context(Settings(), llm=llm)
    out = brain(_state("Open Notepad and book a flight."), ctx)
    assert llm.calls == ["planner"]
    assert "Feasibility check" in llm.prompts[0][0]
    assert "book a flight" in llm.prompts[0][0]
    note = out["plan"].dialog_answer or ""
    assert "book a flight" in note
    assert "can't do this part" in note
    assert "I did the rest" not in note


def test_validate_rejects_a_windows_only_tool_on_this_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    spec = replace(make_spec("fake_windows", base_tier=0, record=record), windows_only=True)
    ctx = make_app_context(Settings(), registry=registry_with(spec))
    monkeypatch.setattr("jarvis.agent.feasibility.is_windows", lambda: False)
    plan = Plan(
        goal="run it", steps=[Step(id="s1", tool="fake_windows", args={"text": "x"}, rationale="r")]
    )
    out = validate({"plan": plan}, ctx)
    assert "only available on Windows" in (out.get("error") or "")
    assert record == []


def test_recovered_retry_is_not_reported_as_a_failure() -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    ctx = make_app_context(
        Settings(), registry=registry_with(make_spec("fake_echo", base_tier=0, record=record))
    )
    plan = Plan(
        goal="echo", steps=[Step(id="s1", tool="fake_echo", args={"text": "hi"}, rationale="r")]
    )
    state: dict[str, Any] = {
        "task_id": "t-p8",
        "plan": plan,
        "step_index": 1,
        "results": [
            StepResult(step_id="s1", ok=False, error="temporary"),
            StepResult(step_id="s1", ok=True, output="fake_echo ran hi", verified=True),
        ],
    }
    effective = effective_results(state)
    assert len(effective) == 1 and effective[0].ok is True
    assert classify_outcome(state) == "success"
    assert "Could not complete" not in base_answer(state)
    assert _failure_record(state, ctx) is None
    assert _skill_gate(state, ctx) == "saved"
    assert _completed_results(state) == effective


def test_partial_limitation_drives_status_screen_and_speech() -> None:
    state: dict[str, Any] = {
        "task_id": "t-p8",
        "plan": Plan(
            goal="partial",
            steps=[Step(id="s1", tool="fake_echo", args={}, rationale="r")],
            dialog_answer="I can't book a flight here. I did the rest.",
        ),
        "results": [StepResult(step_id="s1", ok=True, output="done", verified=True)],
    }
    assert classify_outcome(state) == "partial"
    assert task_status_for(state) == "partial"
    assert "Limitation:" in base_answer(state)
    assert "can't book a flight" in spoken_answer(state, "partial")


def test_partial_limitation_never_claims_success_on_failure() -> None:
    state: dict[str, Any] = {
        "task_id": "t-p8",
        "plan": Plan(
            goal="partial",
            steps=[Step(id="s1", tool="fake_echo", args={}, rationale="r")],
            dialog_answer="I can't book a flight here.",
        ),
        "results": [StepResult(step_id="s1", ok=False, error="boom")],
    }
    answer = base_answer(state)
    assert "Could not complete" in answer
    assert "Limitation:" in answer
    assert "I did the rest" not in answer
