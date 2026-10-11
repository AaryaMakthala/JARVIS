from __future__ import annotations

import threading

import pytest

from jarvis.policy.scope import ScopeCheck, ScopeError, ScopeRegistry


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _issue(reg: ScopeRegistry, task="t1", plan="p1", cid="c1", **kw) -> None:
    reg.issue(task, plan_hash=plan, confirmation_id=cid, ttl_s=kw.pop("ttl_s", 120.0), **kw)


def test_issue_and_validate_ok() -> None:
    reg = ScopeRegistry()
    grant = reg.issue("t1", plan_hash="p1", confirmation_id="c1", allowed_domains=["Example.com"])
    assert grant.max_actions == 8
    assert grant.allowed_domains == frozenset({"example.com"})
    assert grant.contains_domain("example.com")
    assert not grant.contains_domain("evil.com")
    check = reg.validate("t1", plan_hash="p1", confirmation_id="c1")
    assert check.ok and check.remaining == 8 and check.grant is grant


@pytest.mark.parametrize("bad", ["", "   "])
def test_issue_requires_ids(bad: str) -> None:
    reg = ScopeRegistry()
    with pytest.raises(ScopeError):
        reg.issue(bad, plan_hash="p1", confirmation_id="c1")
    with pytest.raises(ScopeError):
        reg.issue("t1", plan_hash=bad, confirmation_id="c1")
    with pytest.raises(ScopeError):
        reg.issue("t1", plan_hash="p1", confirmation_id=bad)


def test_duplicate_confirmation_rejected() -> None:
    reg = ScopeRegistry()
    reg.issue("t1", plan_hash="p1", confirmation_id="c1")
    with pytest.raises(ScopeError):
        reg.issue("t1", plan_hash="p1", confirmation_id="c1")
    with pytest.raises(ScopeError):
        reg.issue("t2", plan_hash="p9", confirmation_id="c1")


def test_no_scope_after_restart_is_fail_closed() -> None:
    reg = ScopeRegistry()
    reg.issue("t1", plan_hash="p1", confirmation_id="c1")
    fresh = ScopeRegistry()  # simulates daemon restart: state lost
    assert fresh.validate("t1", plan_hash="p1", confirmation_id="c1") == ScopeCheck(
        False, "no_scope"
    )


def test_unknown_task() -> None:
    reg = ScopeRegistry()
    reg.issue("t1", plan_hash="p1", confirmation_id="c1")
    assert reg.validate("other").reason == "no_scope"


def test_wrong_plan_and_wrong_confirmation() -> None:
    reg = ScopeRegistry()
    reg.issue("t1", plan_hash="p1", confirmation_id="c1")
    assert reg.validate("t1", plan_hash="pX").reason == "plan_mismatch"
    assert reg.validate("t1", confirmation_id="cX").reason == "confirmation_mismatch"


def test_expiry_blocks_validation_and_budget() -> None:
    clock = FakeClock()
    reg = ScopeRegistry(clock=clock)
    reg.issue("t1", plan_hash="p1", confirmation_id="c1", ttl_s=30.0)
    clock.advance(29.9)
    assert reg.validate("t1").ok
    clock.advance(0.1)
    assert reg.validate("t1").reason == "expired"
    assert reg.consume("t1", plan_hash="p1", confirmation_id="c1").reason == "expired"
    assert reg.remaining("t1") == 0


def test_budget_exhaustion_counts_and_refuses_ninth() -> None:
    reg = ScopeRegistry()
    reg.issue("t1", plan_hash="p1", confirmation_id="c1")
    for i in range(8):
        check = reg.consume("t1", plan_hash="p1", confirmation_id="c1")
        assert check.ok, check.reason
        assert check.remaining == 7 - i
    ninth = reg.consume("t1", plan_hash="p1", confirmation_id="c1")
    assert not ninth.ok and ninth.reason == "budget_exhausted"
    assert reg.consumed("t1") == 8 and reg.remaining("t1") == 0


def test_counter_inherited_across_replan_and_new_id_required() -> None:
    reg = ScopeRegistry()
    reg.issue("t1", plan_hash="p1", confirmation_id="c1")
    for _ in range(3):
        assert reg.consume("t1", plan_hash="p1", confirmation_id="c1").ok
    # Replan: changed plan_hash, fresh daemon confirmation id, same task.
    reg.issue("t1", plan_hash="p2", confirmation_id="c2")
    assert reg.consumed("t1") == 3  # inherited, not reset
    assert reg.remaining("t1") == 5
    # Old confirmation can no longer authorize anything.
    stale = reg.consume("t1", plan_hash="p2", confirmation_id="c1")
    assert not stale.ok and stale.reason == "superseded"
    # New confirmation works against the new plan.
    assert reg.consume("t1", plan_hash="p2", confirmation_id="c2").ok


def test_consume_requires_exact_plan_match() -> None:
    reg = ScopeRegistry()
    reg.issue("t1", plan_hash="p1", confirmation_id="c1")
    assert reg.consume("t1", plan_hash="p2", confirmation_id="c1").reason == "plan_mismatch"
    assert reg.consumed("t1") == 0


def test_revoke_supersedes() -> None:
    reg = ScopeRegistry()
    reg.issue("t1", plan_hash="p1", confirmation_id="c1")
    reg.revoke("t1")
    assert reg.validate("t1").reason == "no_scope"
    assert reg.consume("t1", plan_hash="p1", confirmation_id="c1").reason == "no_scope"


def test_concurrent_consumption_never_exceeds_budget() -> None:
    reg = ScopeRegistry()
    reg.issue("t1", plan_hash="p1", confirmation_id="c1", max_actions=8)
    workers, start = 40, threading.Barrier(40)
    results: list[bool] = []
    results_lock = threading.Lock()

    def worker() -> None:
        start.wait()
        ok = reg.consume("t1", plan_hash="p1", confirmation_id="c1").ok
        with results_lock:
            results.append(ok)

    threads = [threading.Thread(target=worker) for _ in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(results) == 8
    assert reg.consumed("t1") == 8


def test_custom_budget_and_validation_of_budget() -> None:
    reg = ScopeRegistry(default_max_actions=3)
    reg.issue("t1", plan_hash="p1", confirmation_id="c1")
    assert reg.active_grant("t1").max_actions == 3
    with pytest.raises(ScopeError):
        reg.issue("t2", plan_hash="p1", confirmation_id="c2", max_actions=0)


def test_override_above_hard_max_is_clamped_to_eight() -> None:
    reg = ScopeRegistry()
    grant = reg.issue("t1", plan_hash="p1", confirmation_id="c1", max_actions=99)
    assert grant.max_actions == 8
    assert reg.remaining("t1") == 8
    for _ in range(8):
        assert reg.consume("t1", plan_hash="p1", confirmation_id="c1").ok
    assert reg.consume("t1", plan_hash="p1", confirmation_id="c1").reason == "budget_exhausted"


def test_default_above_hard_max_is_clamped() -> None:
    reg = ScopeRegistry(default_max_actions=100)
    assert reg.issue("t1", plan_hash="p1", confirmation_id="c1").max_actions == 8


def test_config_can_lower_default_cap() -> None:
    reg = ScopeRegistry(default_max_actions=2)
    assert reg.issue("t1", plan_hash="p1", confirmation_id="c1").max_actions == 2


def test_configured_default_is_ceiling_over_larger_override() -> None:
    # Config cap 4 must win over an explicit, larger max_actions=8 override.
    reg = ScopeRegistry(default_max_actions=4)
    grant = reg.issue("t1", plan_hash="p1", confirmation_id="c1", max_actions=8)
    assert grant.max_actions == 4
    assert reg.remaining("t1") == 4
    for _ in range(4):
        assert reg.consume("t1", plan_hash="p1", confirmation_id="c1").ok
    assert reg.consume("t1", plan_hash="p1", confirmation_id="c1").reason == "budget_exhausted"


def test_reauthorization_cannot_raise_lifetime_budget() -> None:
    reg = ScopeRegistry()
    reg.issue("t1", plan_hash="p1", confirmation_id="c1", max_actions=3)
    assert reg.consume("t1", plan_hash="p1", confirmation_id="c1").ok  # 1 spent, 2 left
    assert reg.remaining("t1") == 2

    # Reauthorization requests a larger cap; the task's lifetime cap cannot grow.
    reg.issue("t1", plan_hash="p2", confirmation_id="c2", max_actions=8)
    assert reg.active_grant("t1").max_actions == 3
    assert reg.remaining("t1") == 2

    # Only two more actions are ever available, proving the cap did not grow.
    assert reg.consume("t1", plan_hash="p2", confirmation_id="c2").ok
    assert reg.consume("t1", plan_hash="p2", confirmation_id="c2").ok
    assert reg.consume("t1", plan_hash="p2", confirmation_id="c2").reason == "budget_exhausted"


def test_cap_can_shrink_on_reauthorization() -> None:
    reg = ScopeRegistry()
    reg.issue("t1", plan_hash="p1", confirmation_id="c1")  # cap 8
    reg.issue("t1", plan_hash="p2", confirmation_id="c2", max_actions=2)  # lower it
    assert reg.active_grant("t1").max_actions == 2
