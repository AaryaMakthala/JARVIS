"""Stage 4.7: "open notepad and write buy milk" end to end as policy/approval.

Proves the sequence open_app("notepad") -> type_text("buy milk") under the
existing safety rules with no new tool: decide() tiers, validate() on the
two-step plan, allowlist refusal ordering, and that ONE proceed covers both
steps. Nothing here touches a real window or the clipboard.
"""

from __future__ import annotations

from typing import Any

import pytest

from jarvis.agent.batch_approval import decide_and_build_payload
from jarvis.agent.context import make_app_context
from jarvis.agent.nodes.validate import validate as validate_fn
from jarvis.agent.state import Decision, Plan, Step
from jarvis.config import Settings
from jarvis.policy import engine as engine_module
from jarvis.policy import tiers
from jarvis.policy.engine import PolicyContext, PolicyEngine
from jarvis.tools.apps import make_open_app_spec
from jarvis.tools.base import ToolContext
from jarvis.tools.keyboard import TypeTextArgs, make_type_text_spec
from jarvis.tools.registry import build_default_registry
from support import registry_with


def _decide(tool: str, args: dict[str, Any]) -> Decision:
    settings = Settings()
    pctx = PolicyContext(registry=build_default_registry(settings), settings=settings)
    step = Step(id="s1", tool=tool, args=args, rationale="r")
    return PolicyEngine().decide(step, pctx)


def _plan_ctx(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Plan]:
    """App context with the two real specs + the two-step notepad plan."""
    monkeypatch.setattr(engine_module, "is_windows", lambda: True)
    registry = registry_with(make_open_app_spec(), make_type_text_spec())
    ctx = make_app_context(Settings(), registry=registry)
    plan = Plan(
        goal="open notepad and write buy milk",
        steps=[
            Step(id="s1", tool="open_app", args={"name": "notepad"}, rationale="open"),
            Step(
                id="s2",
                tool="type_text",
                args={"text": "buy milk", "target_app": "notepad"},
                rationale="type",
            ),
        ],
    )
    return ctx, plan


def test_open_notepad_is_tier0_allowed_without_confirm() -> None:
    decision = _decide("open_app", {"name": "notepad"})

    assert decision.allowed is True
    assert decision.tier == tiers.TIER_SAFE
    assert decision.needs_confirm is False


def test_type_text_notepad_is_tier1_needing_confirm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine_module, "is_windows", lambda: True)

    decision = _decide("type_text", {"text": "buy milk", "target_app": "notepad"})

    assert decision.allowed is True
    assert decision.tier == tiers.TIER_CONFIRM
    assert decision.needs_confirm is True


def test_type_text_non_allowlisted_target_refused_before_platform_guard() -> None:
    """The allowlist check runs first: on this non-Windows host the error
    must be the allowlist one, never the platform-guard one (ordering proof)."""
    ctx = ToolContext(settings=Settings(), dry_run=False)

    result = make_type_text_spec().run(TypeTextArgs(text="hi", target_app="netscape"), ctx)

    assert result.ok is False
    assert "not in the apps allowlist" in (result.error or "")
    assert "only supported on Windows" not in (result.error or "")


def test_two_step_plan_validates_and_decides_expected_tiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx, plan = _plan_ctx(monkeypatch)

    out = validate_fn({"plan": plan}, ctx)
    d1 = ctx.engine.decide(out["plan"].steps[0], ctx.policy_ctx)
    d2 = ctx.engine.decide(out["plan"].steps[1], ctx.policy_ctx)

    assert (d1.allowed, d1.tier, d1.needs_confirm) == (True, tiers.TIER_SAFE, False)
    assert (d2.allowed, d2.tier, d2.needs_confirm) == (True, tiers.TIER_CONFIRM, True)


def test_one_proceed_covers_open_and_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tier 0 needs no approval; the single Tier 1 step gets ONE confirm."""
    ctx, plan = _plan_ctx(monkeypatch)

    out = validate_fn({"plan": plan}, ctx)
    steps = out["plan"].steps
    d1 = ctx.engine.decide(steps[0], ctx.policy_ctx)
    d2 = ctx.engine.decide(steps[1], ctx.policy_ctx)
    d1.step_id, d2.step_id = "s1", "s2"

    # Mirrors policy_gate.py:44,91-95: the gate only interrupts on the step it
    # is ON, and s1 (Tier 0) never interrupts — so the first prompt is built
    # with step_index=1, where the single eligible Tier 1 step lives.
    payload = decide_and_build_payload(out["plan"], {"s1": d1, "s2": d2}, step_index=1)

    assert d1.needs_confirm is False
    assert payload["type"] == "confirm"
    assert payload["step_id"] == "s2"
