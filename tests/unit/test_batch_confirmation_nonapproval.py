"""Batch confirmations that do not approve must halt honestly (Stage 4.18).

The batch (``plan_approval``) resume path used to swallow both a confirmation
timeout and an explicit ``approved=False`` as a *silent* empty result, so the
graph continued into ``act``, which re-decided the step and reported the
misleading "the approved action no longer matches — the plan was likely
altered after confirmation." refusal.

These tests pin the fixed behaviour:

* timeout  -> the gate halts with the single-step timeout message;
* rejection -> the gate halts with the existing refusal wording;
* in both cases ``act`` is unreachable, no tool runs, and no hash enters
  ``approved_hashes``;
* an explicit approval still approves and executes the batch exactly as before.

The graph tests drive the real compiled LangGraph over a file-backed SQLite
checkpointer with a ``FakeLLM`` and fake tools: no Windows, no voice, no real
LLM calls.  The ``recompute_*`` tests pin ``batch_approval`` as a pure function.
"""

from __future__ import annotations

from typing import Any

from jarvis.agent.batch_approval import (
    TYPE_PLAN_APPROVAL,
    recompute_and_resolve_batch,
)
from jarvis.agent.context import make_app_context
from jarvis.agent.graph import open_sqlite_checkpointer
from jarvis.agent.nodes import route_after_policy_gate
from jarvis.agent.runner import TaskOutcome, resume_task, run_task
from jarvis.agent.schemas import ActionIntent
from jarvis.agent.state import Decision, Plan, Step
from jarvis.config import Settings
from jarvis.daemon.confirmations import plan_hash as _plan_hash
from jarvis.daemon.task_runtime import timeout_answer
from jarvis.llm.client import FakeLLM
from jarvis.policy.refusal import DEFAULT_REFUSAL_TEXT, REASON_USER_DECLINED
from support import approve, brain_steps, deny, make_spec, registry_with

#: Exactly the single-step path's wording (policy_gate), pinned here so both
#: confirmation styles can never drift apart.
TIMEOUT_TEXT = "Confirmation timed out. I did not perform the action."


# ── graph harness: two literal Tier 1 steps -> one plan_approval interrupt ──


def _ctx(record: list[tuple[str, dict[str, Any]]]) -> Any:
    return make_app_context(
        Settings(),
        llm=FakeLLM(
            [
                brain_steps(
                    [
                        ActionIntent(
                            tool="fake_echo",
                            args={"text": "one"},
                            rationale="first echo",
                        ),
                        ActionIntent(
                            tool="fake_echo",
                            args={"text": "two"},
                            rationale="second echo",
                        ),
                    ]
                )
            ]
        ),
        registry=registry_with(make_spec("fake_echo", base_tier=1, record=record)),
    )


def _first_batch_interrupt(ctx: Any, saver: Any) -> tuple[TaskOutcome, dict[str, Any]]:
    """Run the task up to the single batch confirmation for s1 + s2."""
    outcome = run_task(ctx, saver, "echo one then two")
    assert outcome.interrupted is True, f"expected a batch interrupt, got {outcome.final_answer!r}"
    payload = outcome.confirmation or {}
    assert payload.get("type") == TYPE_PLAN_APPROVAL, f"expected plan_approval, got {payload!r}"
    assert [step_id for step_id, _ in payload.get("eligible") or []] == ["s1", "s2"], (
        f"the batch must cover exactly s1+s2, got {payload.get('eligible')!r}"
    )
    return outcome, payload


def _assert_act_unreachable(result: TaskOutcome) -> None:
    """The halted gate state must route to respond, never to act."""
    assert result.halted_reason, "the task must be halted"
    assert route_after_policy_gate({"halted_reason": result.halted_reason}) == "respond"
    assert route_after_policy_gate({}) == "act"  # the router itself still works
    assert list(result.results) == [], "act never ran: no StepResult may exist"
    assert not (result.state.get("approved_hashes") or []), "no hash may be approved"


# ── 1. batch confirmation timeout ─────────────────────────────────────────


def test_batch_timeout_halts_honestly_and_no_tool_runs(tmp_path: Any) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    ctx = _ctx(record)
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        outcome, payload = _first_batch_interrupt(ctx, saver)
        # The real daemon shape, exactly what wait_for_response returns at +30 s.
        answer = timeout_answer(payload)

        result = resume_task(ctx, saver, outcome.task_id, answer)

        assert result.error is None, "a timeout is a normal halt, not a crash"
        assert result.interrupted is False
        assert result.halted_reason == TIMEOUT_TEXT
        assert result.final_answer == TIMEOUT_TEXT
        assert record == [], "no tool may execute after a batch timeout"
        _assert_act_unreachable(result)
    finally:
        saver.conn.close()


# ── 2. batch explicit rejection ───────────────────────────────────────────


def test_batch_rejection_halts_honestly_and_no_tool_runs(tmp_path: Any) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    ctx = _ctx(record)
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        outcome, payload = _first_batch_interrupt(ctx, saver)
        answer = deny(payload)  # {"approved": False, "action_hash": ...}

        result = resume_task(ctx, saver, outcome.task_id, answer)

        assert result.error is None
        assert result.interrupted is False
        assert result.halted_reason == DEFAULT_REFUSAL_TEXT
        assert result.final_answer == DEFAULT_REFUSAL_TEXT
        assert record == [], "no tool may execute after a batch rejection"
        _assert_act_unreachable(result)
    finally:
        saver.conn.close()


# ── 3. explicit approval still behaves exactly as before ──────────────────


def test_batch_approval_still_approves_and_executes(tmp_path: Any) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    ctx = _ctx(record)
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        outcome, payload = _first_batch_interrupt(ctx, saver)

        result = resume_task(ctx, saver, outcome.task_id, approve(payload))

        assert result.error is None
        assert result.interrupted is False
        assert result.halted_reason is None
        assert record == [
            ("fake_echo", {"text": "one"}),
            ("fake_echo", {"text": "two"}),
        ], "an approved batch must execute every eligible step"
        eligible_hashes = [str(h) for _, h in payload.get("eligible") or []]
        approved = [str(h) for h in (result.state.get("approved_hashes") or [])]
        assert sorted(approved) == sorted(eligible_hashes)
        assert result.final_answer is not None
    finally:
        saver.conn.close()


# ── pure-function pins on recompute_and_resolve_batch ─────────────────────


def _plan_and_decisions() -> tuple[Plan, dict[str, Decision]]:
    plan = Plan(
        goal="x",
        steps=[
            Step(id="s1", tool="fake_a", args={}, rationale="a"),
            Step(id="s2", tool="fake_b", args={}, rationale="b"),
        ],
    )
    decisions = {
        "s1": Decision(
            step_id="s1",
            tier=1,
            allowed=True,
            needs_confirm=True,
            needs_unlock=False,
            summary="do a",
            action_hash="a" * 64,
            warn_untrusted=False,
        ),
        "s2": Decision(
            step_id="s2",
            tier=1,
            allowed=True,
            needs_confirm=True,
            needs_unlock=False,
            summary="do b",
            action_hash="b" * 64,
            warn_untrusted=False,
        ),
    }
    return plan, decisions


def test_recompute_timeout_halts_with_timeout_message() -> None:
    plan, decisions = _plan_and_decisions()
    answer = {"approved": False, "action_hash": "a" * 64, "timed_out": True}
    updates, approved = recompute_and_resolve_batch(plan, decisions, {}, answer)
    assert approved == []
    assert updates.get("halted_reason") == TIMEOUT_TEXT


def test_recompute_rejection_halts_with_refusal_text() -> None:
    plan, decisions = _plan_and_decisions()
    answer = {"approved": False, "action_hash": "a" * 64}
    updates, approved = recompute_and_resolve_batch(plan, decisions, {}, answer)
    assert approved == []
    assert updates.get("halted_reason") == DEFAULT_REFUSAL_TEXT


def test_recompute_rejection_reason_maps_to_its_sentence() -> None:
    plan, decisions = _plan_and_decisions()
    answer = {"approved": False, "action_hash": "a" * 64, "reason": REASON_USER_DECLINED}
    updates, approved = recompute_and_resolve_batch(plan, decisions, {}, answer)
    assert approved == []
    assert updates.get("halted_reason") == "Cancelled."


def test_recompute_non_dict_answer_halts_like_single_step_path() -> None:
    plan, decisions = _plan_and_decisions()
    updates, approved = recompute_and_resolve_batch(plan, decisions, {}, "")
    assert approved == []
    assert updates.get("halted_reason") == DEFAULT_REFUSAL_TEXT


def test_recompute_explicit_approval_still_returns_hashes() -> None:
    plan, decisions = _plan_and_decisions()
    answer = {"approved": True, "action_hash": _plan_hash(["a" * 64, "b" * 64])}
    updates, approved = recompute_and_resolve_batch(plan, decisions, {}, answer)
    assert "halted_reason" not in updates
    assert approved == ["a" * 64, "b" * 64]
