"""Phase 1 agent unit tests: schema validation, package structure, runner
state access, and node-level honesty of ``act``/``respond``.
"""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import ValidationError

from jarvis.agent import nodes
from jarvis.agent.context import make_app_context
from jarvis.agent.graph import build_secure_serde
from jarvis.agent.nodes.act import act
from jarvis.agent.nodes.respond import respond
from jarvis.agent.runner import resume_task, run_task
from jarvis.agent.state import Decision, Plan, Step, StepResult
from jarvis.config import Settings
from jarvis.llm.client import FakeLLM
from support import approve, brain_action, make_spec, registry_with

# ---------------------------------------------------------------------------
# Decision schema (typed confirmation must stay a str | None)
# ---------------------------------------------------------------------------


def _decision_kwargs() -> dict[str, Any]:
    return {
        "step_id": "s1",
        "tier": 1,
        "allowed": True,
        "needs_confirm": True,
        "needs_unlock": False,
        "summary": "sum",
        "action_hash": "a" * 64,
    }


def test_decision_rejects_boolean_typed_confirmation() -> None:
    with pytest.raises(ValidationError):
        Decision(**_decision_kwargs(), needs_typed_confirmation=False)


def test_decision_accepts_typed_confirmation_string() -> None:
    d = Decision(**_decision_kwargs(), needs_typed_confirmation="Reports")
    assert d.needs_typed_confirmation == "Reports"
    assert d.model_dump()["needs_typed_confirmation"] == "Reports"


# ---------------------------------------------------------------------------
# nodes package structure: flat imports, no self-import
# ---------------------------------------------------------------------------


def test_nodes_package_is_a_flat_import_bundle() -> None:
    # ``from jarvis.agent.nodes import act`` returns the already-exported
    # *function*, so the submodules must be fetched with importlib to verify the
    # package re-exports the functions defined in the concrete modules (i.e. no
    # self-import / redefinition inside __init__.py).
    import importlib

    act_mod = importlib.import_module("jarvis.agent.nodes.act")
    brain_mod = importlib.import_module("jarvis.agent.nodes.brain")
    clarify_mod = importlib.import_module("jarvis.agent.nodes.clarify")
    validate_mod = importlib.import_module("jarvis.agent.nodes.validate")
    verify_mod = importlib.import_module("jarvis.agent.nodes.verify")

    assert nodes.act is act_mod.act
    assert nodes.brain is brain_mod.brain
    assert nodes.clarify is clarify_mod.clarify
    assert nodes.validate is validate_mod.validate
    assert nodes.verify is verify_mod.verify
    for name in ("wrap", "route_after_policy_gate", "route_after_verify"):
        assert name in nodes.__all__


# ---------------------------------------------------------------------------
# Runner must inspect state via the compiled graph, never the checkpointer
# ---------------------------------------------------------------------------


class NoGetStateSaver(SqliteSaver):
    """A checkpointer whose ``get_state`` always raises.

    ``SqliteSaver`` has no ``get_state`` at all; this subclass makes the
    regression explicit: if the runner ever calls ``checkpointer.get_state``
    instead of ``graph.get_state`` the test fails with a loud message.
    """

    def get_state(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("checkpointer.get_state must never be used; use graph.get_state")


def test_runner_reads_state_via_graph_not_checkpointer(tmp_path: Any) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    ctx = make_app_context(
        Settings(),
        llm=FakeLLM([brain_action("fake_echo", {"text": "hi"})]),
        registry=registry_with(make_spec("fake_echo", base_tier=1, record=record)),
    )
    conn = sqlite3.connect(tmp_path / "c.db", check_same_thread=False)
    saver: NoGetStateSaver = NoGetStateSaver(conn, serde=build_secure_serde())
    try:
        outcome = run_task(ctx, saver, "echo hi")
        assert outcome.interrupted is True
        assert outcome.confirmation is not None
        assert outcome.confirmation["step_id"] == "s1"
        # state was read through graph.get_state, and models came back as real
        # objects (not dicts), which is exactly what the state-access bug was.
        assert isinstance(outcome.state["plan"], Plan)
        # decisions are written only after the gate returns (post-interrupt),
        # so before resume the map must be empty - but never contain a dead dict.
        assert outcome.state["decisions"] == {}
        assert record == []  # nothing executed before approval

        outcome2 = resume_task(ctx, saver, outcome.task_id, approve(outcome.confirmation))
        assert isinstance(outcome2.state["decisions"]["s1"], Decision)
        assert isinstance(outcome2.state["results"][0], StepResult)
    finally:
        conn.close()


def test_runner_state_has_no_password_residue(tmp_path: Any) -> None:
    """A secret smuggled through the resume payload never reaches outcome.state."""
    record: list[tuple[str, dict[str, Any]]] = []
    ctx = make_app_context(
        Settings(),
        llm=FakeLLM([brain_action("fake_echo", {"text": "hi"})]),
        registry=registry_with(make_spec("fake_echo", base_tier=1, record=record)),
    )
    saver = NoGetStateSaver(
        sqlite3.connect(tmp_path / "c.db", check_same_thread=False),
        serde=build_secure_serde(),
    )
    try:
        first = run_task(ctx, saver, "echo hi")
        resumed = _resume_with_secret(ctx, saver, first, "hunter2secret")
        text = repr(resumed.state)
        assert "hunter2secret" not in text
        assert "hunter2" not in text
    finally:
        saver.conn.close()


def _resume_with_secret(ctx: Any, saver: Any, first: Any, secret: str) -> Any:
    from jarvis.agent.runner import resume_task

    return resume_task(
        ctx,
        saver,
        first.task_id,
        {"approved": True, "action_hash": first.confirmation["action_hash"], "password": secret},
    )


# ---------------------------------------------------------------------------
# act node: approvals bind to the exact action; hard blocks never run
# ---------------------------------------------------------------------------


@pytest.fixture
def act_ctx() -> tuple[list[tuple[str, dict[str, Any]]], Any]:
    record: list[tuple[str, dict[str, Any]]] = []
    ctx = make_app_context(
        Settings(),
        llm=None,
        registry=registry_with(make_spec("fake_echo", base_tier=1, record=record)),
    )
    return record, ctx


def _hash_for(ctx: Any, text: str) -> str:
    step = Step(id="s1", tool="fake_echo", args={"text": text}, rationale="r")
    return ctx.engine.decide(step, ctx.policy_ctx).action_hash


def test_act_runs_when_the_exact_action_was_approved(act_ctx: Any) -> None:
    record, ctx = act_ctx
    state = {
        "plan": Plan(
            goal="x", steps=[Step(id="s1", tool="fake_echo", args={"text": "A"}, rationale="r")]
        ),
        "step_index": 0,
        "approved_hashes": [_hash_for(ctx, "A")],
    }
    out = act(state, ctx)
    assert record == [("fake_echo", {"text": "A"})]
    assert out["results"][0].ok is True


def test_act_refuses_step_mutated_after_approval(act_ctx: Any) -> None:
    record, ctx = act_ctx
    state = {
        "plan": Plan(
            goal="x", steps=[Step(id="s1", tool="fake_echo", args={"text": "B"}, rationale="r")]
        ),
        "step_index": 0,
        "approved_hashes": [_hash_for(ctx, "A")],
    }
    out = act(state, ctx)
    assert record == []
    assert "approved action no longer matches" in out["halted_reason"]
    assert out["results"][0].ok is False
    assert out["results"][0].verified is False


def test_act_refuses_unknown_tool(act_ctx: Any) -> None:
    record, ctx = act_ctx
    state = {
        "plan": Plan(goal="x", steps=[Step(id="s1", tool="no_such_tool", args={}, rationale="r")]),
        "step_index": 0,
        "approved_hashes": [],
    }
    out = act(state, ctx)
    assert record == []
    assert "unknown tool" in out["halted_reason"]


def test_act_refuses_hard_blocked_tool_even_with_approved_hash() -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    defender = make_spec("disable_defender_x", base_tier=0, record=record)
    ctx = make_app_context(Settings(), llm=None, registry=registry_with(defender))
    step = Step(id="s1", tool="disable_defender_x", args={"text": "on"}, rationale="r")
    state = {
        "plan": Plan(goal="x", steps=[step]),
        "step_index": 0,
        "approved_hashes": [ctx.engine.decide(step, ctx.policy_ctx).action_hash],
    }
    out = act(state, ctx)
    assert record == []
    assert "blocked by policy" in out["halted_reason"]


# ---------------------------------------------------------------------------
# respond node: honest failure reporting
# ---------------------------------------------------------------------------


def test_respond_never_claims_success_when_verification_failed() -> None:
    results = [StepResult(step_id="s1", ok=True, output="I did it", verified=False)]
    out = respond({"results": results}, None)
    assert "could not be verified" in out["final_answer"]
    assert "Done" not in out["final_answer"]
    assert "I did it" not in out["final_answer"]


def test_respond_marks_unverified_success_honestly() -> None:
    results = [StepResult(step_id="s1", ok=True, output="I did it", verified=None)]
    out = respond({"results": results}, None)
    assert out["final_answer"].startswith("I did it")
    assert "could not be verified" in out["final_answer"]
