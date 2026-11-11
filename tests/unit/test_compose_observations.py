"""Compose must not describe a successful read-only observation as unverified."""

from jarvis.agent.compose_rules import classify_outcome, spoken_answer
from jarvis.agent.state import Plan, Step, StepResult


def _state(tool: str, *, verified: bool | None, final_answer: str | None = None) -> dict:
    return {
        "task_id": "t1",
        "plan": Plan(goal="x", steps=[Step(id="s1", tool=tool, args={}, rationale="r")]),
        "results": [
            StepResult(step_id="s1", ok=True, output="It is 10:00.", verified=verified)
        ],
        "final_answer": final_answer,
    }


def test_clock_read_is_success_and_has_no_unverified_wording():
    state = _state("get_time", verified=None)
    assert classify_outcome(state) == "success"
    speech = spoken_answer(state, "success")
    assert "nothing confirmed" not in speech
    assert "could not verify" not in speech


def test_unverified_side_effect_still_reports_partial():
    state = _state("type_text", verified=None)
    assert classify_outcome(state) == "partial"
    speech = spoken_answer(state, "partial")
    assert "verify" in speech or "confirmed" in speech


def test_verified_observation_is_success():
    assert classify_outcome(_state("get_time", verified=True)) == "success"


def test_failed_result_is_failed_even_with_a_brain_final_answer():
    from jarvis.agent.nodes.telemetry import task_status

    state = {
        "task_id": "t1",
        "plan": Plan(goal="x", steps=[Step(id="s1", tool="fake", args={}, rationale="r")]),
        "results": [StepResult(step_id="s1", ok=False, error="boom")],
        "final_answer": "Done.",
    }
    assert classify_outcome(state) == "failed"
    assert task_status(state) == "failed"
