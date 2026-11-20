"""P5: evidence-based step verification and evidence-aware recovery."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field, ValidationError

from jarvis.agent.compose_rules import classify_outcome
from jarvis.agent.context import make_app_context
from jarvis.agent.evidence import EVIDENCE_KINDS, evaluate_evidence
from jarvis.agent.graph import open_sqlite_checkpointer
from jarvis.agent.nodes import brain
from jarvis.agent.nodes.replan import _remaining_steps, _toolcalls
from jarvis.agent.nodes.respond import respond
from jarvis.agent.nodes.verify import verify
from jarvis.agent.prompts import brain_system_prompt
from jarvis.agent.runner import run_task
from jarvis.agent.schemas import ActionIntent, BrainDecision
from jarvis.agent.state import Plan, Step, StepResult
from jarvis.config import AgentSettings, Settings
from jarvis.llm.client import FakeLLM
from jarvis.llm.prompts import replan_system_prompt, replan_user_prompt
from jarvis.tools.base import ToolContext, ToolResult, ToolSpec
from support import registry_with, replan_continue


class _Args(BaseModel):
    text: str = Field(min_length=1)


def _spec(
    result: ToolResult,
    *,
    name: str = "fake_echo",
    verify_result: ToolResult | None = None,
    no_verify: bool = False,
    record: list[tuple[str, dict[str, Any]]] | None = None,
) -> ToolSpec:
    def run(args: _Args, ctx: ToolContext) -> ToolResult:
        if record is not None:
            record.append((name, args.model_dump()))
        return result

    def check(args: _Args, observed: ToolResult, ctx: ToolContext) -> ToolResult:
        return verify_result or observed

    return ToolSpec(
        name=name,
        description="fake tool for evidence tests",
        args_model=_Args,
        base_tier=0,
        run=run,
        verify=None if no_verify else check,
    )


def _result(
    *,
    output: str = "tool finished",
    data: dict[str, Any] | None = None,
    verified: bool | None = None,
) -> ToolResult:
    return ToolResult(ok=True, output=output, data=data or {}, verified=verified)


def _state(
    evidence: str | None,
    result: ToolResult,
    *,
    expect: str = "the expected outcome",
    next_step: bool = False,
) -> dict[str, Any]:
    steps = [
        Step(
            id="s1",
            tool="fake_echo",
            args={"text": "x"},
            rationale="do it",
            expect=expect,
            evidence=evidence,  # type: ignore[arg-type]
        )
    ]
    if next_step:
        steps.append(
            Step(
                id="s2",
                tool="fake_echo",
                args={"text": "secret argument"},
                rationale="then finish",
                expect="finish the goal",
                evidence="information",
            )
        )
    return {
        "plan": Plan(goal="reach the goal", steps=steps),
        "step_index": 0,
        "results": [
            StepResult(
                step_id="s1",
                ok=result.ok,
                output=result.output,
                data=result.data,
                error=result.error,
                verified=result.verified,
                verify_note=result.verify_note,
            )
        ],
        "retry_count": 0,
        "replan_count": 0,
    }


def _ctx(
    spec: ToolSpec,
    *,
    max_retries: int = 1,
    max_replans: int = 1,
) -> Any:
    return make_app_context(
        Settings(agent=AgentSettings(max_retries_per_step=max_retries, max_replans=max_replans)),
        llm=None,
        registry=registry_with(spec),
    )


def test_closed_vocabulary_is_validated_and_defaults_preserve_old_state() -> None:
    step = Step(id="s1", tool="fake_echo", args={"text": "x"}, rationale="r")
    assert step.evidence is None
    assert StepResult(step_id="s1", ok=True).evidence_status is None
    with pytest.raises(ValidationError):
        Step(id="s1", tool="fake_echo", args={}, rationale="r", evidence="guess")  # type: ignore[arg-type]
    assert evaluate_evidence("guess", _result()).status == "inconclusive"


def test_evidence_fields_round_trip_through_the_checkpoint_serde() -> None:
    from jarvis.agent.graph import build_secure_serde

    serde = build_secure_serde()
    plan = Plan(
        goal="g",
        steps=[
            Step(
                id="s1",
                tool="fake_echo",
                args={"text": "x"},
                rationale="r",
                expect="resource exists",
                evidence="resource",
            )
        ],
    )
    result = StepResult(
        step_id="s1", ok=True, evidence="resource", evidence_status="confirmed", verified=True
    )
    restored_plan = serde.loads_typed(serde.dumps_typed(plan))
    restored_result = serde.loads_typed(serde.dumps_typed(result))
    assert isinstance(restored_plan, Plan) and restored_plan.steps[0].evidence == "resource"
    assert isinstance(restored_result, StepResult)
    assert restored_result.evidence_status == "confirmed"

    legacy = Plan.model_validate(
        {"goal": "g", "steps": [{"id": "s1", "tool": "fake_echo", "args": {}, "rationale": "r"}]}
    )
    assert legacy.steps[0].evidence is None


def test_action_evidence_confirms_execution_only() -> None:
    out = verify(_state("action", _result()), _ctx(_spec(_result(), no_verify=True)))
    last = out["results"][-1]
    assert last.verified is True
    assert last.evidence == "action" and last.evidence_status == "confirmed"
    assert "execution itself" in last.verify_note
    assert out["step_index"] == 1 and out["retry_now"] is False


def test_resource_evidence_is_confirmed_by_structured_data() -> None:
    result = _result(data={"results": [{"title": "Installation"}]})
    out = verify(_state("resource", result), _ctx(_spec(result, no_verify=True)))
    last = out["results"][-1]
    assert last.verified is True and last.evidence_status == "confirmed"
    assert "resource" in last.verify_note
    assert out["step_index"] == 1


def test_tool_success_without_expected_resource_is_not_reported_complete() -> None:
    result = _result(data={"results": []})
    state = _state("resource", result)
    out = verify(state, _ctx(_spec(result, no_verify=True)))
    last = out["results"][-1]
    assert last.ok is True and last.verified is False
    assert last.evidence_status == "contradicted"
    assert out["retry_now"] is False and out["replan_now"] is True
    assert "step_index" not in out

    failed_state = {**state, "results": out["results"]}
    assert classify_outcome(failed_state) == "failed"
    answer = respond(failed_state, None)["final_answer"]
    assert "could not be verified" in answer
    assert "Done" not in answer


def test_information_evidence_without_observable_content_stays_inconclusive() -> None:
    result = _result(output="", data={})
    state = _state("information", result)
    out = verify(state, _ctx(_spec(result, no_verify=True)))
    last = out["results"][-1]
    assert last.verified is None and last.evidence_status == "inconclusive"
    assert last.verify_note == "The expected information could not be independently observed."
    assert out["step_index"] == 1

    honest_state = {**state, "results": out["results"]}
    assert classify_outcome(honest_state) == "partial"
    assert last.verify_note in respond(honest_state, None)["final_answer"]


def test_state_evidence_follows_the_tools_own_verify() -> None:
    observed = _result()
    confirmed = verify(
        _state("state", observed),
        _ctx(_spec(observed, verify_result=_result(verified=True))),
    )
    assert confirmed["results"][-1].verified is True
    assert confirmed["results"][-1].evidence_status == "confirmed"

    unchecked = verify(
        _state("state", observed),
        _ctx(_spec(observed, verify_result=_result(verified=None))),
    )
    assert unchecked["results"][-1].verified is None
    assert unchecked["results"][-1].evidence_status == "inconclusive"

    failed = verify(
        _state("state", observed),
        _ctx(_spec(observed, verify_result=_result(verified=False))),
    )
    assert failed["results"][-1].verified is False
    assert failed["results"][-1].evidence_status == "contradicted"
    assert failed["retry_now"] is False and failed["replan_now"] is True


def test_unverifiable_evidence_keeps_a_single_honest_note() -> None:
    result = _result(output="workstation locked")
    state = _state("unverifiable", result)
    out = verify(state, _ctx(_spec(result, no_verify=True)))
    last = out["results"][-1]
    assert last.verified is None and last.evidence_status == "inconclusive"
    assert last.verify_note == "The outcome could not be independently verified."
    assert "not independently checked" not in last.verify_note
    assert classify_outcome({**state, "results": out["results"]}) == "partial"


def test_legacy_plan_without_evidence_keeps_existing_verify_behaviour() -> None:
    observed = _result()
    checked = verify(
        _state(None, observed), _ctx(_spec(observed, verify_result=_result(verified=True)))
    )
    last = checked["results"][-1]
    assert last.verified is True and last.evidence_status is None and last.evidence is None
    assert last.verify_note == ""

    prose = _state(None, observed, expect="secret expectation text")
    unchecked = verify(prose, _ctx(_spec(observed, no_verify=True)))
    note = unchecked["results"][-1].verify_note
    assert "not independently checked" in note
    assert "secret expectation text" not in note
    assert unchecked["results"][-1].evidence_status is None
    assert unchecked["step_index"] == 1


def test_contradicted_evidence_fails_closed_when_replan_budget_is_gone() -> None:
    result = _result(data={"paths": []})
    out = verify(
        _state("resource", result),
        _ctx(_spec(result, no_verify=True), max_retries=0, max_replans=0),
    )
    assert out["retry_now"] is False and out["replan_now"] is False
    assert out["results"][-1].verified is False


def test_replan_context_carries_evidence_without_leaking_argument_values() -> None:
    state = _state("resource", _result(data={"results": []}), next_step=True)
    state["results"][0] = state["results"][0].model_copy(
        update={"verified": False, "evidence": "resource", "evidence_status": "contradicted"}
    )
    calls = _toolcalls(state)
    assert calls[0].evidence == "resource" and calls[0].evidence_status == "contradicted"
    assert calls[0].status == "unverified"
    state["step_index"] = 1
    remaining = _remaining_steps(state)
    assert remaining[0]["evidence"] == "information"
    assert "secret argument" not in str(remaining)

    prompt = replan_user_prompt(
        user_input="reach the goal",
        goal="reach the goal",
        toolcalls=[call.model_dump() for call in calls],
        remaining_steps=remaining,
    )
    assert '"evidence_status": "contradicted"' in prompt
    assert "secret argument" not in prompt


def test_brain_projects_declared_evidence_into_the_plan() -> None:
    decision = BrainDecision(
        request_type="action",
        goal="find the installation guide",
        actions=[
            ActionIntent(
                tool="fake_echo",
                args={"text": "docs"},
                rationale="search",
                expect="installation steps were found",
                evidence="information",
            )
        ],
    )
    ctx = make_app_context(Settings(), llm=FakeLLM([decision]))
    out = brain({"user_input": "find the installation guide"}, ctx)
    assert out["plan"].steps[0].evidence == "information"
    assert out["plan"].steps[0].expect == "installation steps were found"


def test_prompts_advertise_the_closed_evidence_vocabulary() -> None:
    for prompt in (brain_system_prompt([]), replan_system_prompt([])):
        assert all(kind in prompt for kind in EVIDENCE_KINDS)
        assert "evidence" in prompt


def _saver(tmp_path: Path) -> Any:
    return open_sqlite_checkpointer(str(tmp_path / "checkpoints.db"))


def test_graph_reports_failure_when_a_successful_tool_has_no_expected_resource(
    tmp_path: Path,
) -> None:
    result = _result(data={"results": []})
    decision = BrainDecision(
        request_type="action",
        goal="find the installation guide",
        actions=[
            ActionIntent(
                tool="fake_echo",
                args={"text": "docs"},
                rationale="search",
                expect="installation steps were found",
                evidence="resource",
            )
        ],
    )
    ctx = make_app_context(
        Settings(agent=AgentSettings(max_retries_per_step=0, max_replans=0)),
        llm=FakeLLM([decision]),
        registry=registry_with(_spec(result, no_verify=True)),
    )
    out = run_task(ctx, _saver(tmp_path), "find the installation guide")
    assert out.interrupted is False
    assert "could not be verified" in (out.final_answer or "")
    assert "Done" not in (out.final_answer or "")
    assert classify_outcome(out.state) == "failed"
    assert len(out.results) == 1
    assert out.results[0].evidence_status == "contradicted"


def test_graph_replans_once_instead_of_repeating_a_contradicted_action(
    tmp_path: Path,
) -> None:
    result = _result(data={"results": []})
    decision = BrainDecision(
        request_type="action",
        goal="find the installation guide",
        actions=[
            ActionIntent(
                tool="fake_echo",
                args={"text": "docs"},
                rationale="search",
                expect="installation steps were found",
                evidence="resource",
            )
        ],
    )
    record: list[tuple[str, dict[str, Any]]] = []
    replacement = Plan(
        goal="find the installation guide another way",
        steps=[
            Step(
                id="s2",
                tool="fake_fix",
                args={"text": "fixed"},
                rationale="try another source",
            )
        ],
    )
    ctx = make_app_context(
        Settings(agent=AgentSettings(max_retries_per_step=2, max_replans=1)),
        llm=FakeLLM([decision, replan_continue(replacement)]),
        registry=registry_with(
            _spec(result, no_verify=True, record=record),
            _spec(
                _result(output="fake_fix ran fixed"),
                name="fake_fix",
                no_verify=True,
                record=record,
            ),
        ),
    )
    out = run_task(ctx, _saver(tmp_path), "find the installation guide")
    assert record == [
        ("fake_echo", {"text": "docs"}),
        ("fake_fix", {"text": "fixed"}),
    ]
    assert [r.step_id for r in out.results] == ["s2"]
    assert out.state["replan_count"] == 1
    assert out.state["retry_count"] == 0
    assert out.final_answer == "fake_fix ran fixed"


def test_evidence_verification_does_not_change_authorization_state() -> None:
    result = _result(data={"results": []})
    state = _state("resource", result)
    state["approved_hashes"] = ["already-approved"]
    state["decisions"] = {"s1": object()}
    out = verify(state, _ctx(_spec(result, no_verify=True)))
    assert "approved_hashes" not in out and "decisions" not in out
    assert set(out) <= {
        "results",
        "step_index",
        "retry_count",
        "retry_now",
        "replan_now",
    }
