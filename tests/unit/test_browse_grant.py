"""Unit tests for the task-scoped browsing grant (pure rules + registry)."""

from __future__ import annotations

import time

from jarvis.policy.browse_grant import (
    BROWSE_TOOLS,
    BrowseGrant,
    BrowseGrantRegistry,
    coverage,
    risky_label,
)
from jarvis.policy.scope import ScopeRegistry

TASK = "task-1"
HOST = "example.com"


class _Clock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def _registry(clock: _Clock | None = None) -> BrowseGrantRegistry:
    return BrowseGrantRegistry(clock=clock or _Clock())


def _covered(grant: BrowseGrant, **kwargs: object) -> tuple[bool, str]:
    base: dict[str, object] = {
        "task_id": TASK,
        "tool": "browser_goto",
        "tier": 1,
        "host": HOST,
        "now": grant.issued_at + 1,
    }
    base.update(kwargs)
    return coverage(grant, **base)  # type: ignore[arg-type]


def test_tool_outside_the_allowlist_is_not_covered() -> None:
    registry = _registry()
    grant = registry.issue(TASK, host=HOST)
    assert _covered(grant, tool="delete_path") == (False, "tool_not_allowed")
    # A browser tool that is not part of the closed allowlist is refused too.
    assert _covered(grant, tool="browser_screenshot") == (False, "tool_not_allowed")


def test_wrong_domain_is_not_covered() -> None:
    registry = _registry()
    grant = registry.issue(TASK, host=HOST)
    assert _covered(grant, tool="browser_click", host="evil.example", label="Read") == (
        False,
        "cross_domain",
    )
    assert _covered(grant, host="") == (False, "host_missing")


def test_expired_grant_is_not_covered() -> None:
    clock = _Clock()
    registry = _registry(clock)
    grant = registry.issue(TASK, host=HOST, ttl_s=10)
    assert registry.validate(TASK).ok is True
    clock.value = grant.expires_at
    assert registry.validate(TASK).reason == "expired"
    assert _covered(grant, now=grant.expires_at)[0] is False


def test_exhausted_budget_is_not_covered() -> None:
    clock = _Clock()
    registry = _registry(clock)
    grant = registry.issue(TASK, host=HOST, max_actions=3)
    assert registry.charge(TASK).ok
    assert registry.charge(TASK).ok
    assert registry.charge(TASK).ok
    exhausted = registry.charge(TASK)
    assert exhausted.ok is False and exhausted.reason == "budget_exhausted"
    assert _covered(grant, actions_used=3) == (False, "budget_exhausted")


def test_other_task_id_is_not_covered() -> None:
    registry = _registry()
    grant = registry.issue(TASK, host=HOST)
    assert _covered(grant, task_id="task-2") == (False, "task_mismatch")


def test_risky_click_labels_are_not_covered() -> None:
    registry = _registry()
    grant = registry.issue(TASK, host=HOST)
    for label in ("Buy now", "Place order", "Pay", "Check out", "Sign in", "Download file"):
        covered, reason = _covered(
            grant, tool="browser_click", label=label, role="button"
        )
        assert covered is False, label
        assert reason.startswith("risky_label:"), (label, reason)
    # Whole-word matching: "buyer" is not "buy".
    assert risky_label("buyer reviews") == ""
    assert _covered(grant, tool="browser_click", label="Read more", role="link") == (
        True,
        "",
    )


def test_risky_open_result_target_is_not_covered() -> None:
    registry = _registry()
    grant = registry.issue(TASK, host=HOST)
    for label in ("Delete note", "Sign in", "Download file"):
        covered, reason = _covered(
            grant, tool="browser_open_result", label=label, role="link"
        )
        assert covered is False, label
        assert reason.startswith("risky_label:"), (label, reason)
    assert _covered(
        grant, tool="browser_open_result", label="Continue", role="submit"
    ) == (False, "target_not_allowed")
    assert _covered(grant, tool="browser_open_result", role="link") == (
        False,
        "unobserved_target",
    )
    assert _covered(
        grant, tool="browser_open_result", label="Read result", role="link"
    ) == (True, "")


def test_password_field_is_not_covered() -> None:
    registry = _registry()
    grant = registry.issue(TASK, host=HOST)
    assert _covered(
        grant, tool="browser_type", label="Password", role="textbox", is_password=True
    ) == (False, "password_field")
    assert _covered(grant, tool="browser_type", label="Password", role="password") == (
        False,
        "password_field",
    )
    assert _covered(grant, tool="browser_type", label="Email", role="textbox") == (True, "")


def test_enter_after_non_search_typing_is_not_covered() -> None:
    registry = _registry()
    grant = registry.issue(TASK, host=HOST)
    assert _covered(
        grant,
        tool="browser_press_key",
        key="Enter",
        previous_tool="browser_type",
        previous_role="textbox",
    ) == (False, "enter_after_non_search_type")
    assert _covered(
        grant, tool="browser_press_key", key="Enter", previous_tool="browser_click"
    ) == (False, "enter_after_non_search_type")


def test_enter_after_search_typing_is_covered() -> None:
    registry = _registry()
    grant = registry.issue(TASK, host=HOST)
    assert _covered(
        grant,
        tool="browser_press_key",
        key="Enter",
        previous_tool="browser_type",
        previous_role="searchbox",
    ) == (True, "")
    # Other non-submitting keys are always fine.
    assert _covered(grant, tool="browser_press_key", key="Escape") == (True, "")


def test_only_tier_one_is_coverable() -> None:
    registry = _registry()
    grant = registry.issue(TASK, host=HOST)
    assert _covered(grant, tier=0) == (False, "tier_not_coverable")
    assert _covered(grant, tier=2, needs_unlock=True) == (False, "tier_not_coverable")
    assert _covered(grant, tier=3) == (False, "tier_not_coverable")
    assert _covered(grant, tier=2, needs_typed_confirmation="Folder") == (
        False,
        "tier_not_coverable",
    )


def test_tier_is_untouched_and_no_plan_hash_is_bound() -> None:
    """The grant carries no tier and no plan hash: it cannot lower or bind one."""
    fields = BrowseGrant.__dataclass_fields__
    assert "tier" not in fields
    assert "plan_hash" not in fields
    registry = _registry()
    grant = registry.issue(TASK, host=HOST)
    result = _covered(grant)
    assert isinstance(result, tuple) and result == (True, "")  # only (bool, reason)


def test_a_plan_change_does_not_invalidate_the_grant() -> None:
    """There is no plan hash to invalidate: a new one-step plan stays covered."""
    registry = _registry()
    grant = registry.issue(TASK, host=HOST)
    assert _covered(
        grant, tool="browser_open_result", label="Read result", role="link"
    ) == (True, "")
    # The specialist builds a brand-new one-step plan next iteration; nothing
    # about the grant changed, so the same host is still covered.
    assert _covered(grant, tool="browser_goto", host=HOST) == (True, "")
    assert registry.active_for_task(TASK) is grant


def test_revoked_grant_is_not_covered() -> None:
    registry = _registry()
    registry.issue(TASK, host=HOST)
    registry.revoke(TASK)
    assert registry.active_for_task(TASK) is None
    assert registry.validate(TASK).reason == "revoked"
    assert registry.charge(TASK).ok is False
    assert coverage(None, task_id=TASK, tool="browser_goto", tier=1, host=HOST) == (
        False,
        "no_grant",
    )


def test_registry_caps_and_disabled_switch() -> None:
    registry = BrowseGrantRegistry(max_actions=500, ttl_s=10_000)
    grant = registry.issue(TASK, host=HOST)
    assert grant.max_actions == BrowseGrantRegistry.HARD_MAX_ACTIONS == 30
    assert grant.expires_at - grant.issued_at <= BrowseGrantRegistry.HARD_MAX_TTL_S
    assert grant.tools == BROWSE_TOOLS
    disabled = BrowseGrantRegistry(enabled=False)
    assert disabled.validate(TASK).reason == "disabled"


def test_scope_registry_still_enforces_its_own_eight_action_cap() -> None:
    """The untouched ScopeRegistry keeps its hard 8-action ceiling."""
    now = time.monotonic()
    scope = ScopeRegistry(clock=lambda: now)
    grant = scope.issue(
        TASK,
        plan_hash="plan-hash",
        confirmation_id="cid-1",
        allowed_domains=[HOST],
        max_actions=30,
    )
    assert grant.max_actions == ScopeRegistry.HARD_MAX_ACTIONS == 8
    for _ in range(8):
        assert scope.consume(TASK, plan_hash="plan-hash", confirmation_id="cid-1").ok
    final = scope.consume(TASK, plan_hash="plan-hash", confirmation_id="cid-1")
    assert final.ok is False and final.reason == "budget_exhausted"
