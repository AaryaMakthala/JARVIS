"""Phase 2 replan/brain/clarify node and router unit tests."""

from __future__ import annotations

from typing import Any

from jarvis.agent.context import make_app_context
from jarvis.agent.nodes import (
    brain,
    clarify,
    replan,
    route_after_brain,
    route_after_clarify,
    route_after_replan,
    route_after_validate,
    route_after_verify,
)
from jarvis.agent.nodes.verify import verify
from jarvis.agent.schemas import ReplanDecision
from jarvis.agent.state import Plan, Step, StepResult
from jarvis.config import AgentSettings, Settings
from jarvis.llm.client import FakeLLM
from support import (
    brain_action,
    brain_clarify,
    brain_conversation,
    brain_unsupported,
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


# ── routers ────────────────────────────────────────────────────────────────


def test_route_after_validate_branches() -> None:
    # a validation failure starts a fresh brain call (repair) unless a
    # replacement plan is already waiting from the replanner
    assert route_after_validate({"plan": echo_plan(), "error": "bad"}) == "brain"
    assert (
        route_after_validate({"plan": echo_plan(), "error": "bad", "pending_replan": True})
        == "replan"
    )
    assert route_after_validate({"plan": echo_plan(), "error": None}) == "policy_gate"
    assert route_after_validate({"halted_reason": "x"}) == "respond"
    assert (
        route_after_validate(
            {"plan": Plan(kind="tool", goal="g", steps=[], needs_clarification=True)}
        )
        == "respond"
    )


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


# ── brain node: the single LLM classification/projection stage ──────────────


def test_brain_conversation_projects_answer_and_plan() -> None:
    out = brain({"user_input": "hi"}, _ctx([brain_conversation("bear")]))
    assert out["request_kind"] == "conversation"
    assert out["final_answer"] == "bear"
    assert out["plan"].kind == "conversation"
    assert out["plan"].dialog_answer == "bear"
    assert out["plan"].steps == []
    assert out.get("halted_reason") is None


def test_brain_conversation_blank_answer_halts_fail_closed() -> None:
    out = brain({"user_input": "hi"}, _ctx([brain_conversation("  ")]))
    assert out["halted_reason"] == "I couldn't produce an answer."
    assert "final_answer" not in out


def test_brain_action_projects_multi_step_tool_plan() -> None:
    out = brain({"user_input": "echo hi"}, _ctx([brain_action("fake_echo", {"text": "hi"})]))
    assert out["request_kind"] == "tool"
    assert out["plan"].kind == "tool"
    assert [(s.id, s.tool, s.args) for s in out["plan"].steps] == [
        ("s1", "fake_echo", {"text": "hi"})
    ]
    assert out.get("error") is None  # a success is not misread as a repair cue


def test_brain_clarification_projects_needs_clarification() -> None:
    out = brain({"user_input": "open it"}, _ctx([brain_clarify("which file?")]))
    assert out["request_kind"] == "clarification"
    assert out["plan"].needs_clarification is True
    assert out["clarification_question"] == "which file?"
    assert out["plan"].steps == []


def test_brain_unsupported_halts_with_honest_message() -> None:
    out = brain(
        {"user_input": "pay someone"}, _ctx([brain_unsupported("payments are out of scope")])
    )
    assert out["request_kind"] == "unsupported"
    assert out["halted_reason"] == "payments are out of scope"


def test_brain_without_llm_halts_cleanly() -> None:
    ctx = make_app_context(Settings())
    out = brain({"user_input": "hi"}, ctx)
    assert (
        out["halted_reason"] == "JARVIS's AI backend is not configured yet. "
        "Configure an LLM provider before asking me to reason about tasks."
    )
    assert "final_answer" not in out


def test_brain_llm_failure_maps_to_safe_messages() -> None:
    # wrong model type (e.g. an old Plan) is a malformed output
    out = brain({"user_input": "hi"}, _ctx([conversation_plan()]))
    assert out["halted_reason"] == "I couldn't safely understand that request."


class _ExplodingLLM:
    """A fake llm whose ``structured`` raises a known LLMError message."""

    def __init__(self, message: str) -> None:
        self._message = message

    def structured(self, **kwargs: Any) -> Any:
        from jarvis.llm.client import LLMError

        raise LLMError(self._message)


def _exploding_ctx(message: str) -> Any:
    return make_app_context(Settings(), llm=_ExplodingLLM(message))


def test_brain_provider_error_maps_to_safe_message() -> None:
    out = brain({"user_input": "hi"}, _exploding_ctx("connection refused"))
    assert out["halted_reason"] == "I couldn't reach the AI service."
    assert "connection refused" not in out  # the raw error never reaches the user


def test_brain_timeout_maps_to_timeout_message() -> None:
    out = brain({"user_input": "hi"}, _exploding_ctx("request timed out after 30s"))
    assert out["halted_reason"] == "JARVIS couldn't get a response from the AI backend in time."


def test_brain_rate_limit_maps_to_safe_message() -> None:
    out = brain({"user_input": "hi"}, _exploding_ctx("rate limit exceeded"))
    assert out["halted_reason"] == "I couldn't reach the AI service."


def test_brain_repair_call_sees_previous_validation_error() -> None:
    llm = FakeLLM([brain_action("fake_echo", {"text": "fixed"})])
    ctx = make_app_context(Settings(), llm=llm)
    out = brain(
        {"user_input": "echo", "validate_attempts": 1, "error": "unknown tool: nope"},
        ctx,
    )
    assert out["request_kind"] == "tool"
    assert out.get("error") is None  # the repair cue is consumed on success
    assert llm.calls == [("structured", "planner", "BrainDecision")]


# ── clarify node: bounded, fail-closed questions ─────────────────────────────


def test_clarify_fails_closed_when_max_questions_reached() -> None:
    # the limit branch runs before any interrupt(), so it is direct-testable
    out = clarify({"clarification_count": 2, "clarification_question": "which?"}, _ctx([]))
    assert "I still need clarification" in out["halted_reason"]
    assert "2" in out["halted_reason"]


def test_clarify_limit_message_mentions_exactly_how_many_were_asked() -> None:
    from jarvis.agent.nodes.clarify import MAX_CLARIFICATIONS, _limit_message

    assert MAX_CLARIFICATIONS == 2
    assert str(MAX_CLARIFICATIONS) in _limit_message(MAX_CLARIFICATIONS)


def test_route_after_brain_branches() -> None:
    assert route_after_brain({"plan": echo_plan()}) == "validate"
    assert route_after_brain({"plan": conversation_plan()}) == "respond"
    assert (
        route_after_brain({"plan": Plan(kind="tool", goal="g", steps=[], needs_clarification=True)})
        == "clarify"
    )
    assert route_after_brain({"plan": None}) == "respond"
    assert route_after_brain({"halted_reason": "x"}) == "respond"


def test_route_after_clarify_branches() -> None:
    assert route_after_clarify({"clarification_answer": "left"}) == "brain"
    assert route_after_clarify({"cancelled": True}) == "respond"
    assert route_after_clarify({"halted_reason": "Cancelled by the user."}) == "respond"
