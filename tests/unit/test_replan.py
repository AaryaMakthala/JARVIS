"""Phase 2 replan/understand/converse node and router unit tests."""

from __future__ import annotations

from typing import Any

from jarvis.agent.context import make_app_context
from jarvis.agent.nodes import (
    converse,
    replan,
    route_after_replan,
    route_after_understand,
    route_after_validate,
    route_after_verify,
    understand_request,
)
from jarvis.agent.nodes.verify import verify
from jarvis.agent.schemas import ReplanDecision
from jarvis.agent.state import Plan, Step, StepResult
from jarvis.config import AgentSettings, Settings
from jarvis.llm.client import FakeLLM
from support import (
    conversation_plan,
    echo_plan,
    make_spec,
    registry_with,
    replan_continue,
    replan_stop,
)


def _ctx(script: list[Any] | None = None, max_replans: int = 2) -> Any:
    settings = Settings(agent=AgentSettings(max_replans=max_replans))
    return make_app_context(settings, llm=FakeLLM(script))


def _failing_state() -> dict[str, Any]:
    return {
        "user_input": "echo hi and fail",
        "plan": echo_plan(),
        "step_index": 0,
        "results": [StepResult(step_id="s1", ok=False, error="boom", output="", verified=True)],
        "replan_count": 0,
        "pending_replan": False,
        "api_calls": 0,
        "tokens": 0,
    }


def test_replan_stop_halts_fail_closed_without_consuming_budget() -> None:
    ctx = _ctx([replan_stop("this is a dead end")])
    out = replan(_failing_state(), ctx)
    assert out["halted_reason"] == "Could not complete the task: this is a dead end."
    assert out["replan_count"] == 0
    assert isinstance(out["replan_decision"], ReplanDecision)
    assert out["replan_decision"].action == "stop"


def test_replan_continue_resets_tool_bookkeeping_for_replacement() -> None:
    replacement = Plan(
        goal="retry",
        steps=[Step(id="s2", tool="fake_fix", args={"text": "x"}, rationale="repair")],
    )
    ctx = _ctx([replan_continue(replacement)])
    out = replan(_failing_state(), ctx)
    assert out["plan"] is replacement
    assert out["pending_replan"] is True
    assert out["replan_count"] == 1
    assert out["replan_now"] is False and out["retry_now"] is False
    assert out["step_index"] == 0 and out["error"] is None and out["halted_reason"] is None
    assert out["results"] == []  # the failed attempt is dropped, no successes yet


def test_replan_keeps_completed_prefix_only() -> None:
    state = _failing_state()
    state["step_index"] = 1
    state["results"] = [
        StepResult(step_id="s1", ok=True, output="ok", verified=True),
        StepResult(step_id="s2", ok=False, error="boom", verified=True),
    ]
    ctx = _ctx([replan_continue(echo_plan())])
    out = replan(state, ctx)
    assert [r.step_id for r in out["results"]] == ["s1"]


def test_replan_repair_call_preserves_already_trimmed_results() -> None:
    state = _failing_state()
    state["pending_replan"] = True
    kept = StepResult(step_id="s1", ok=True, output="ok", verified=True)
    state["results"] = [kept]
    ctx = _ctx([replan_continue(echo_plan())])
    out = replan(state, ctx)
    assert out["results"] == [kept]


def test_replan_budget_exhausted_never_calls_the_llm() -> None:
    ctx = _ctx([], max_replans=0)  # empty script: any LLM call would raise
    out = replan(_failing_state(), ctx)
    assert out["halted_reason"] and "replan budget is exhausted" in out["halted_reason"]
    assert out["replan_decision"].action == "stop"


def test_replan_without_llm_halts_cleanly() -> None:
    ctx = make_app_context(Settings())
    out = replan(_failing_state(), ctx)
    assert out["halted_reason"] == "No LLM backend configured (run `jarvis init`)."


def test_replan_llm_failure_halts_not_crashes() -> None:
    ctx = _ctx(["not a ReplanDecision"])  # structured() will refuse the str
    out = replan(_failing_state(), ctx)
    assert "Replan error" in (out["halted_reason"] or "")


def test_replan_continue_without_plan_is_rejected() -> None:
    # The schema itself refuses a 'continue' without a plan (docs: struct is a
    # boundary), so the node's defensive guard is belt-and-braces.
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ReplanDecision(action="continue", plan=None)


def test_verify_sets_replan_now_when_retries_exhausted() -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    ctx = make_app_context(
        Settings(agent=AgentSettings(max_retries_per_step=1, max_replans=2)),
        llm=None,
        registry=registry_with(make_spec("fake_echo", base_tier=0, record=record, run_ok=False)),
    )
    state = {
        "plan": echo_plan(),
        "step_index": 0,
        "results": [StepResult(step_id="s1", ok=False, error="boom", verified=True)],
        "retry_count": 1,
        "replan_count": 0,
    }
    out = verify(state, ctx)
    assert out["replan_now"] is True


def test_verify_fails_closed_when_replan_budget_gone() -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    ctx = make_app_context(
        Settings(agent=AgentSettings(max_retries_per_step=1, max_replans=2)),
        llm=None,
        registry=registry_with(make_spec("fake_echo", base_tier=0, record=record, run_ok=False)),
    )
    state = {
        "plan": echo_plan(),
        "step_index": 0,
        "results": [StepResult(step_id="s1", ok=False, error="boom", verified=True)],
        "retry_count": 1,
        "replan_count": 2,
    }
    out = verify(state, ctx)
    assert out["replan_now"] is False and out["retry_now"] is False


# ── understand_request + converse ─────────────────────────────────────────


def test_understand_records_conversation_kind() -> None:
    out = understand_request({"plan": conversation_plan()}, _ctx([]))
    assert out["request_kind"] == "conversation"


def test_understand_records_tool_kind() -> None:
    out = understand_request({"plan": echo_plan()}, _ctx([]))
    assert out["request_kind"] == "tool"


def test_understand_halts_without_plan() -> None:
    out = understand_request({"plan": None}, _ctx([]))
    assert out["request_kind"] == "tool"
    assert "No plan" in out["halted_reason"]


def test_understand_preserves_existing_halt() -> None:
    out = understand_request({"plan": None, "halted_reason": "Planner error: boom"}, _ctx([]))
    assert out["request_kind"] == "tool"
    assert "halted_reason" not in out  # the original message is kept


def test_converse_honours_planner_written_answer_without_llm() -> None:
    state = {"plan": conversation_plan("bear")}
    out = converse(state, _ctx([]))
    assert out["final_answer"] == "bear"


def test_converse_asks_fast_model_when_no_planner_answer() -> None:
    llm = FakeLLM(["Hi!"])
    ctx = make_app_context(Settings(), llm=llm)
    state = {"plan": Plan(kind="conversation", goal="g", steps=[], dialog_answer=None)}
    out = converse(state, ctx)
    assert out["final_answer"] == "Hi!"
    assert ("text", "fast", "str") in llm.calls


def test_converse_halts_without_llm() -> None:
    state = {"plan": Plan(kind="conversation", goal="g", steps=[])}
    out = converse(state, make_app_context(Settings()))
    assert "No LLM backend configured" in out["halted_reason"]


# ── routers ────────────────────────────────────────────────────────────────


def test_route_after_understand_branches() -> None:
    assert route_after_understand({"plan": conversation_plan()}) == "converse"
    assert route_after_understand({"plan": echo_plan(), "kind": "x"}) == "validate"
    assert route_after_understand({"halted_reason": "x"}) == "respond"
    assert route_after_understand({"plan": None}) == "respond"


def test_route_after_validate_branches() -> None:
    assert route_after_validate({"plan": echo_plan(), "error": "bad"}) == "plan"
    assert (
        route_after_validate({"plan": echo_plan(), "error": "bad", "pending_replan": True})
        == "replan"
    )
    assert route_after_validate({"plan": echo_plan(), "error": None}) == "policy_gate"
    assert route_after_validate({"halted_reason": "x"}) == "respond"


def test_route_after_verify_branches() -> None:
    ok = {
        "plan": echo_plan(),
        "results": [StepResult(step_id="s1", ok=True, verified=True)],
        "step_index": 0,
    }
    assert route_after_verify({"plan": echo_plan(), "replan_now": True}) == "replan"
    assert route_after_verify({"plan": echo_plan(), "retry_now": True}) == "act"
    assert route_after_verify(ok) == "policy_gate"
    assert (
        route_after_verify({"plan": echo_plan(), "results": [ok["results"][0]], "step_index": 1})
        == "respond"
    )
    assert route_after_verify({"plan": echo_plan(), "results": []}) == "respond"


def test_route_after_replan_branches() -> None:
    assert route_after_replan({"plan": echo_plan()}) == "validate"
    assert route_after_replan({"halted_reason": "x"}) == "respond"
