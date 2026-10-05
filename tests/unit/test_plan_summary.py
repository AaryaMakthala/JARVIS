"""Tests for :mod:`jarvis.agent.plan_summary` (Stage 3, pure module)."""

from __future__ import annotations

from jarvis.agent.plan_summary import summarise
from jarvis.agent.state import Decision, Plan, Step


def _decision(step_id: str, summary: str) -> Decision:
    return Decision(
        step_id=step_id,
        tier=1,
        allowed=True,
        needs_confirm=True,
        needs_unlock=False,
        summary=summary,
        action_hash="a" * 64,
    )


def _plan(count: int) -> Plan:
    return Plan(
        kind="tool",
        goal="do things",
        steps=[
            Step(id=f"s{i + 1}", tool="open_app", args={}, rationale="LLM RATIONALE TEXT")
            for i in range(count)
        ],
    )


def test_summary_is_deterministic_and_step_ordered() -> None:
    plan = _plan(2)
    decisions = {"s1": _decision("s1", "open Spotify"), "s2": _decision("s2", "search Believer")}

    out = summarise(plan, decisions)

    assert out == "I will open Spotify, then search Believer."
    # Same inputs in any mapping order give the same text (order comes from the plan).
    reversed_map = {"s2": decisions["s2"], "s1": decisions["s1"]}
    assert summarise(plan, reversed_map) == out
    # Never the planner's rationale.
    assert "LLM RATIONALE TEXT" not in out


def test_summary_is_bounded_and_never_mid_step() -> None:
    step_a, step_b, step_c = "A" * 40, "B" * 40, "C" * 40
    plan = _plan(3)
    decisions = {
        "s1": _decision("s1", step_a),
        "s2": _decision("s2", step_b),
        "s3": _decision("s3", step_c),
    }

    out = summarise(plan, decisions, budget=100)

    assert len(out) <= 100
    assert step_a in out  # the included step is whole
    assert step_b not in out  # dropped steps are not partially shown
    assert step_c not in out
    assert out.endswith("and 2 more steps.")


def test_summary_redacts_secret_like_text() -> None:
    plan = _plan(1)
    decisions = {"s1": _decision("s1", "create file with password=hunter2secret")}

    out = summarise(plan, decisions)

    assert "hunter2secret" not in out
    assert "<redacted>" in out
