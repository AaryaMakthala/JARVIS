"""Tests for plan-level (batch) Tier 1 approval in policy_gate.

The gate raises ONE interrupt when a plan contains 2+ batch-eligible Tier 1
steps; a single eligible step leaves behaviour unchanged; Tier 2 / runtime /
Tier 3 steps are gated individually.
"""

from __future__ import annotations

from jarvis.agent.batch_approval import (
    TYPE_PLAN_APPROVAL,
    _batch_eligible,
    _eligible_steps,
    _plan_approval_payload,
    _single_step_payload,
    decide_and_build_payload,
    recompute_and_resolve_batch,
)
from jarvis.agent.state import Decision, Plan, Step
from jarvis.daemon.confirmations import plan_hash as _plan_hash

# ── tiny helpers ──────────────────────────────────────────────────────────


def _step(id: str, tool: str, args: dict, *, runtime: bool = False) -> Step:
    return Step(
        id=id,
        tool=tool,
        args=args,
        rationale=f"do {id}",
        depends_on_untrusted=False,
        resolved_from_runtime=runtime,
    )


def _tier1(text: str, *, action_hash: str | None = None) -> Decision:
    h = action_hash or ("h1" * 6 + "a" * (64 - 12))
    return Decision(
        step_id="",
        tier=1,
        allowed=True,
        needs_confirm=True,
        needs_unlock=False,
        summary=text,
        action_hash=h,
        warn_untrusted=False,
    )


def _tier2(text: str) -> Decision:
    return Decision(
        step_id="",
        tier=2,
        allowed=True,
        needs_confirm=True,
        needs_unlock=True,
        summary=text,
        action_hash="t2" * 32,
        warn_untrusted=False,
    )


def _tier3() -> Decision:
    return Decision(
        step_id="",
        tier=3,
        allowed=False,
        needs_confirm=False,
        needs_unlock=False,
        summary="blocked",
        action_hash="t3" * 32,
        reasons=["Tier 3 is blocked in code"],
    )


def _plan(*steps: Step) -> Plan:
    return Plan(goal="x", steps=list(steps))


# ── eligibility (unit) ────────────────────────────────────────────────────


def test_eligible_only_tier1_confirm_no_unlock_not_runtime() -> None:
    d = _tier1("ok")
    step = _step("s1", "fake", {})
    assert _batch_eligible(d, step) is True


def test_eligible_false_when_runtime() -> None:
    d = _tier1("ok")
    step = _step("s1", "fake", {}, runtime=True)
    assert _batch_eligible(d, step) is False


def test_eligible_false_when_tier2() -> None:
    d = _tier2("delete")
    step = _step("s1", "fake", {})
    assert _batch_eligible(d, step) is False


def test_eligible_false_when_tier3() -> None:
    d = _tier3()
    step = _step("s1", "fake", {})
    assert _batch_eligible(d, step) is False


def test_eligible_false_when_blocked() -> None:
    d = Decision(
        step_id="s1",
        tier=1,
        allowed=False,
        needs_confirm=False,
        needs_unlock=False,
        summary="blocked",
        action_hash="x" * 64,
        reasons=["nope"],
    )
    step = _step("s1", "fake", {})
    assert _batch_eligible(d, step) is False


# ── payloads ──────────────────────────────────────────────────────────────


def test_single_step_payload_is_confirm_type() -> None:
    d = _tier1("open notepad")
    p = _single_step_payload(d)
    assert p["type"] == "confirm"
    assert p["step_id"] == ""
    assert p["action_hash"] == d.action_hash


def test_plan_approval_payload_names_eligible_steps() -> None:
    d1 = _tier1("open notepad")
    d2 = _tier1("search believer")
    d1.step_id = "s1"
    d2.step_id = "s2"
    p = _plan_approval_payload(
        _plan(),
        {"s1": d1, "s2": d2},
        [("s1", d1.action_hash), ("s2", d2.action_hash)],
        [d1.action_hash, d2.action_hash],
        summary_budget=600,
    )
    assert p["type"] == TYPE_PLAN_APPROVAL
    assert p["tier"] == 1
    assert p["eligible"] == [("s1", d1.action_hash), ("s2", d2.action_hash)]
    assert p["plan_hash"] == p["action_hash"]


# ── decide_and_build_payload (gate integration) ──────────────────────────


def test_two_tier1_steps_one_interrupt() -> None:
    d1, d2 = _tier1("open notepad"), _tier1("search believer")
    d1.step_id, d2.step_id = "s1", "s2"
    plan = _plan(_step("s1", "fake_a", {}), _step("s2", "fake_b", {}))
    p = decide_and_build_payload(plan, {"s1": d1, "s2": d2})
    assert p["type"] == TYPE_PLAN_APPROVAL
    assert p["eligible"] == [("s1", d1.action_hash), ("s2", d2.action_hash)]


def test_single_tier1_step_unchanged() -> None:
    d = _tier1("open notepad")
    d.step_id = "s1"
    plan = _plan(_step("s1", "fake_a", {}))
    p = decide_and_build_payload(plan, {"s1": d})
    assert p["type"] == "confirm"
    assert p["step_id"] == "s1"
    assert p["action_hash"] == d.action_hash


def test_batch_never_covers_tier2() -> None:
    d1 = _tier1("open notepad")
    d2 = _tier2("delete stuff")
    d1.step_id, d2.step_id = "s1", "s2"
    plan = _plan(_step("s1", "fake_a", {}), _step("s2", "fake_b", {}))
    p = decide_and_build_payload(plan, {"s1": d1, "s2": d2})
    # Only s1 is batch-eligible; with <2 eligible we fall back to single-step.
    assert p["type"] == "confirm"
    assert p["step_id"] == "s1"


def test_runtime_derived_step_regated_individually() -> None:
    d1 = _tier1("open notepad")
    d2 = _tier1("use runtime value")
    d1.step_id, d2.step_id = "s1", "s2"
    plan = _plan(
        _step("s1", "fake_a", {}),
        _step("s2", "fake_b", {}, runtime=True),
    )
    p = decide_and_build_payload(plan, {"s1": d1, "s2": d2})
    # Only one eligible -> single-step payload for s1.
    assert p["type"] == "confirm"
    assert p["step_id"] == "s1"


def test_tier3_still_refused_in_batch() -> None:
    d1 = _tier1("open notepad")
    d3 = _tier3()
    d1.step_id, d3.step_id = "s1", "s2"
    plan = _plan(_step("s1", "fake_a", {}), _step("s2", "fake_b", {}))
    p = decide_and_build_payload(plan, {"s1": d1, "s3": d3, "s2": d3})
    # s1 is alone eligible -> single step.
    assert p["type"] == "confirm"
    assert p["step_id"] == "s1"


# ── resume: recompute + resolve ──────────────────────────────────────────


def test_batch_approved_on_resume() -> None:
    d1, d2 = _tier1("open notepad"), _tier1("search believer")
    d1.step_id, d2.step_id = "s1", "s2"
    plan = _plan(_step("s1", "fake_a", {}), _step("s2", "fake_b", {}))
    decisions = {"s1": d1, "s2": d2}
    eligible = _eligible_steps(plan, decisions)
    ordered = [h for _, h in eligible]
    answer = {"approved": True, "action_hash": _plan_hash(ordered)}
    _, approved = recompute_and_resolve_batch(plan, decisions, {}, answer)
    assert approved == [d1.action_hash, d2.action_hash]


def test_batch_hash_mismatch_refused() -> None:
    d1, d2 = _tier1("open notepad"), _tier1("search believer")
    d1.step_id, d2.step_id = "s1", "s2"
    plan = _plan(_step("s1", "fake_a", {}), _step("s2", "fake_b", {}))
    decisions = {"s1": d1, "s2": d2}
    answer = {"approved": True, "action_hash": "0" * 64}
    updates, approved = recompute_and_resolve_batch(plan, decisions, {}, answer)
    assert approved == []
    assert updates.get("halted_reason") is not None
    assert "action hash does not match" in updates["halted_reason"]


def test_toctou_path_change_refused_in_batch() -> None:
    d1, d2 = _tier1("open notepad"), _tier1("search believer")
    d1.step_id, d2.step_id = "s1", "s2"
    d1.resolved_paths = ["p1"]
    d2.resolved_paths = ["p2"]
    plan = _plan(_step("s1", "fake_a", {}), _step("s2", "fake_b", {}))
    decisions = {"s1": d1, "s2": d2}
    eligible = _eligible_steps(plan, decisions)
    ordered = [h for _, h in eligible]
    answer = {"approved": True, "action_hash": _plan_hash(ordered)}
    # Pretend the stashed paths differ from the recomputed ones -> TOCTOU refusal.
    state = {"gated_resolved_paths": {"s1": ["different"]}}
    updates, approved = recompute_and_resolve_batch(plan, decisions, state, answer)
    assert approved == []
    assert updates.get("halted_reason") is not None
    assert "path changed after you confirmed" in updates["halted_reason"]


def test_late_proceed_for_a_leaves_b_unapproved() -> None:
    """Batch A built with hash set A; answer carries batch A's hash but we ask
    about a different (stale) batch B — the answer is refused because the
    batch plan_hash does not match."""
    d1, d2 = _tier1("open notepad"), _tier1("search believer")
    d1.step_id, d2.step_id = "s1", "s2"
    plan_a = _plan(_step("s1", "fake_a", {}), _step("s2", "fake_b", {}))
    decisions_a = {"s1": d1, "s2": d2}
    eligible_a = _eligible_steps(plan_a, decisions_a)
    hash_a = [h for _, h in eligible_a]
    hash_a_value = _plan_hash(hash_a)

    # Now pretend the *current* plan is a different batch B (different hashes).
    d1b, d2b = _tier1("other a"), _tier1("other b")
    d1b.step_id, d2b.step_id = "s1", "s2"
    d1b.action_hash = "b1" + "x" * 62
    d2b.action_hash = "b2" + "y" * 62
    plan_b = _plan(_step("s1", "fake_a", {}), _step("s2", "fake_b", {}))
    decisions_b = {"s1": d1b, "s2": d2b}

    # An answer that carries batch A's hash against batch B's plan must be refused.
    answer = {"approved": True, "action_hash": hash_a_value}
    updates, approved = recompute_and_resolve_batch(plan_b, decisions_b, {}, answer)
    assert approved == []
    assert updates.get("halted_reason") is not None


def test_act_refuses_unapproved_step_after_partial_batch() -> None:
    """If only some steps in a batch are approved (here: one is runtime-derived
    on resume so it is excluded), the remaining approved hashes are still
    returned; a step not in approved_hashes is refused by act."""
    d1 = _tier1("open notepad", action_hash="d1" + "1" * 62)
    d2 = _tier1("use runtime value", action_hash="d2" + "2" * 62)
    d1.step_id, d2.step_id = "s1", "s2"
    plan = _plan(_step("s1", "fake_a", {}), _step("s2", "fake_b", {}, runtime=True))
    decisions = {"s1": d1, "s2": d2}
    eligible = _eligible_steps(plan, decisions)
    ordered = [h for _, h in eligible]
    answer = {"approved": True, "action_hash": _plan_hash(ordered)}
    _, approved = recompute_and_resolve_batch(plan, decisions, {}, answer)
    # Only s1 is eligible; s2 is runtime-derived, excluded from batch.
    assert approved == [d1.action_hash]
    assert d2.action_hash not in approved


# ── eligible_steps ordering ───────────────────────────────────────────────


def test_eligible_steps_are_in_plan_order() -> None:
    d1, d2, d3 = _tier1("a"), _tier1("b"), _tier1("c")
    for i, d in enumerate((d1, d2, d3)):
        d.step_id = f"s{i + 1}"
    plan = _plan(
        _step("s1", "t1", {}),
        _step("s2", "t2", {}),
        _step("s3", "t3", {}),
    )
    decisions = {"s1": d1, "s2": d2, "s3": d3}
    eligible = _eligible_steps(plan, decisions)
    assert [sid for sid, _ in eligible] == ["s1", "s2", "s3"]


# ── degenerate plans ──────────────────────────────────────────────────────


def test_empty_plan_returns_confirm_payload_for_first_step() -> None:
    plan = _plan()
    p = decide_and_build_payload(plan, {})
    # With no steps we fall back to the degenerate single-step path, which
    # produces a ``confirm`` payload whose step_id is empty.
    assert p["type"] == "confirm"
    assert p["step_id"] == ""


def test_one_eligible_step_is_single_confirm() -> None:
    d = _tier1("open notepad")
    d.step_id = "s1"
    plan = _plan(_step("s1", "fake_a", {}))
    p = decide_and_build_payload(plan, {"s1": d})
    assert p["type"] == "confirm"
    assert p["step_id"] == "s1"
