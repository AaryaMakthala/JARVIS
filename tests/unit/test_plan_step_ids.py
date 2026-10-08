"""4.20: validate rejects unsafe replan step IDs, fail-closed.

``Step.id`` keys the decision map, the TOCTOU snapshot, batch approval's
eligible list, and act's exactly-once guard, so blank, duplicated, or
completed-result-colliding IDs must never leave the ``validate`` node.  The
rejection rides the existing validate -> repair/replan -> second-rejection
halt machinery; no new route or state field exists.
"""

from __future__ import annotations

from typing import Any

from jarvis.agent.context import make_app_context
from jarvis.agent.nodes.validate import validate
from jarvis.agent.state import Plan, Step, StepResult
from jarvis.config import Settings
from support import make_spec, registry_with


def _ctx() -> Any:
    record: list[tuple[str, dict[str, Any]]] = []
    spec = make_spec("fake_echo", base_tier=1, record=record)
    return make_app_context(Settings(), llm=None, registry=registry_with(spec))


def _plan(*ids: str) -> Plan:
    return Plan(
        goal="g",
        steps=[Step(id=sid, tool="fake_echo", args={"text": "hi"}, rationale="r") for sid in ids],
    )


def _rejects(out: dict[str, Any]) -> None:
    assert out["halted_reason"] is None, "first rejection must arm the repair path, not halt"
    assert out["validate_attempts"] == 1
    assert out["error"], "a rejected plan must carry an error for the repair call"


# ── unsafe shapes are rejected ──────────────────────────────────────────


def test_blank_step_id_is_rejected() -> None:
    for blank in ("", "   "):
        out = validate({"plan": _plan("s1", blank)}, _ctx())
        _rejects(out)
        assert "step 2 has a blank id" in (out["error"] or "")


def test_duplicate_step_ids_are_rejected() -> None:
    out = validate({"plan": _plan("s1", "s2", "s1")}, _ctx())
    _rejects(out)
    assert "duplicate step id 's1'" in (out["error"] or "")


def test_completed_result_id_collision_is_rejected_during_replan() -> None:
    state = {
        "plan": _plan("s1", "s2"),
        "pending_replan": True,
        "results": [StepResult(step_id="s1", ok=True, verified=True)],
    }
    out = validate(state, _ctx())
    _rejects(out)
    assert "reuses completed step id 's1'" in (out["error"] or "")


# ── valid plans pass unchanged ──────────────────────────────────────────


def test_valid_s1_s2_s3_plan_is_accepted() -> None:
    out = validate({"plan": _plan("s1", "s2", "s3")}, _ctx())
    assert out["error"] is None
    assert out["halted_reason"] is None
    assert out["validate_attempts"] == 0
    assert set(out["gated_resolved_paths"]) == {"s1", "s2", "s3"}


def test_needs_clarification_path_is_unaffected() -> None:
    """The clarify early-return happens before any step-ID checking."""
    plan = Plan(
        goal="g",
        steps=[Step(id="", tool="fake_echo", args={"text": "hi"}, rationale="r")],
        needs_clarification=True,
        clarification_question="Which folder should I use?",
    )
    out = validate({"plan": plan}, _ctx())
    assert out["plan"].needs_clarification is True
    assert out["error"] is None
    assert out["validate_attempts"] == 0
    assert out["pending_replan"] is False


# ── the existing second-rejection halt still fires ──────────────────────


def test_second_invalid_attempt_reaches_the_final_halt() -> None:
    state: dict[str, Any] = {"plan": _plan("")}
    first = validate(state, _ctx())
    _rejects(first)

    second = validate({**state, **first}, _ctx())
    assert second["validate_attempts"] == 2
    assert second["halted_reason"] is not None
    assert second["halted_reason"].startswith("Could not produce a valid plan (after 2 attempts)")
    assert "blank id" in second["halted_reason"]
