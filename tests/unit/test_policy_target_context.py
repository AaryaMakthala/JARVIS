"""Slice 4: policy-engine integration of Stage-7 browser/target context.

These tests drive the deterministic engine directly (no LLM, no network, no
browser/UIA runtime) and prove that the optional origin/scope/target context can
only *raise* a restriction.  With the context absent every decision is exactly
as it was before, so existing behaviour is preserved.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from jarvis.agent.context import make_app_context
from jarvis.agent.state import Step
from jarvis.config import Settings
from jarvis.policy import tiers
from jarvis.policy.engine import PolicyContext
from jarvis.policy.targets import ResolvedTarget
from support import make_spec, registry_with

WEB = "https://web.whatsapp.com"
SCOPE = ("web.whatsapp.com",)


def _app(*, base_tier: int = 0, name: str = "fake_act") -> tuple[Any, str]:
    registry = registry_with(make_spec(name, base_tier=base_tier, record=[]))
    return make_app_context(Settings(), registry=registry), name


def _step(name: str) -> Step:
    return Step(id="s1", tool=name, args={"text": "hi"}, rationale="r")


def _decide(app: Any, name: str, **ctx_fields: Any) -> Any:
    pctx = app.policy_ctx
    if ctx_fields:
        pctx = dataclasses.replace(pctx, **ctx_fields)
    return app.engine.decide(_step(name), pctx)


def _browser(**over: Any) -> ResolvedTarget:
    base: dict[str, Any] = {
        "kind": "browser",
        "ref": "node-1",
        "name": "Send message",
        "role": "button",
        "origin": WEB,
    }
    base.update(over)
    return ResolvedTarget(**base)


# ── context defaults / no-op ────────────────────────────────────────────────


def test_new_context_fields_default_to_absent() -> None:
    ctx = PolicyContext(
        registry=registry_with(make_spec("t", base_tier=0, record=[])), settings=Settings()
    )
    assert ctx.origin is None
    assert ctx.allowed_domains == ()
    assert ctx.target is None
    assert ctx.scope_approved is False


def test_absent_context_leaves_decision_unchanged() -> None:
    app, name = _app()
    base = app.engine.decide(_step(name), app.policy_ctx)
    explicit = _decide(
        app, name, origin=None, allowed_domains=(), target=None, scope_approved=False
    )
    assert base == explicit
    assert base.allowed and base.tier == tiers.TIER_SAFE


def test_valid_in_scope_browser_action_is_allowed() -> None:
    app, name = _app()
    decision = _decide(app, name, origin=WEB, allowed_domains=SCOPE, scope_approved=True)
    assert decision.allowed and decision.tier == tiers.TIER_SAFE


def test_origin_is_canonicalised_before_scope_check() -> None:
    app, name = _app()
    decision = _decide(
        app,
        name,
        origin="https://WEB.WhatsApp.com:443",
        allowed_domains=("web.whatsapp.com",),
        scope_approved=True,
    )
    assert decision.allowed


# ── origins must be valid and in scope ──────────────────────────────────────


@pytest.mark.parametrize(
    "bad_origin",
    [
        "",
        "   ",
        "ftp://web.whatsapp.com",
        "not-a-url",
        "web.whatsapp.com",
        "https://user@web.whatsapp.com",
    ],
)
def test_invalid_origin_fails_closed(bad_origin: str) -> None:
    app, name = _app()
    decision = _decide(app, name, origin=bad_origin, allowed_domains=SCOPE, scope_approved=True)
    assert decision.tier == tiers.TIER_BLOCKED and not decision.allowed


def test_out_of_scope_origin_fails_closed() -> None:
    app, name = _app()
    decision = _decide(
        app,
        name,
        origin="https://evil.example",
        allowed_domains=SCOPE,
        scope_approved=True,
    )
    assert decision.tier == tiers.TIER_BLOCKED and not decision.allowed


def test_absent_domain_scope_fails_closed() -> None:
    app, name = _app()
    decision = _decide(app, name, origin=WEB, allowed_domains=(), scope_approved=True)
    assert decision.tier == tiers.TIER_BLOCKED and not decision.allowed


def test_unapproved_task_scope_fails_closed() -> None:
    app, name = _app()
    decision = _decide(app, name, origin=WEB, allowed_domains=SCOPE, scope_approved=False)
    assert decision.tier == tiers.TIER_BLOCKED and not decision.allowed


# ── target risk: payment, fresh-confirm, password, identity ─────────────────


def _target_decision(app: Any, name: str, target: ResolvedTarget) -> Any:
    return _decide(
        app,
        name,
        origin=WEB,
        allowed_domains=SCOPE,
        target=target,
        scope_approved=True,
    )


def test_payment_target_is_blocked() -> None:
    app, name = _app()
    decision = _target_decision(app, name, _browser(name="Buy now"))
    assert decision.tier == tiers.TIER_BLOCKED and not decision.allowed


def test_risky_control_requires_fresh_confirmation_even_in_scope() -> None:
    app, name = _app()
    decision = _target_decision(app, name, _browser(name="Send"))
    assert decision.allowed  # not blocked, but never silently auto-approved
    assert decision.tier >= tiers.TIER_CONFIRM
    assert decision.needs_confirm
    assert any("fresh confirmation" in reason for reason in decision.reasons)


def test_password_target_is_refused_browser_and_uia() -> None:
    app, name = _app()
    browser_pw = _browser(name="Password", role="textbox", input_type="password", is_password=True)
    assert not _target_decision(app, name, browser_pw).allowed

    uia_pw = ResolvedTarget(
        kind="uia", ref="edit-1", name="Password", role="edit", is_password=True
    )
    decision = _decide(app, name, target=uia_pw)
    assert decision.tier == tiers.TIER_BLOCKED and not decision.allowed


def test_ambiguous_target_identity_is_blocked() -> None:
    app, name = _app()
    decision = _decide(
        app, name, target=ResolvedTarget(kind="uia", ref="", name="Save", role="button")
    )
    assert decision.tier == tiers.TIER_BLOCKED and not decision.allowed


def test_uia_target_without_origin_is_allowed() -> None:
    app, name = _app()
    decision = _decide(
        app, name, target=ResolvedTarget(kind="uia", ref="r1", name="Save", role="button")
    )
    assert decision.allowed and decision.tier == tiers.TIER_SAFE


# ── context can only raise, never lower ─────────────────────────────────────


def test_context_does_not_lower_an_existing_tier() -> None:
    app, name = _app(base_tier=tiers.TIER_CONFIRM_UNLOCK)
    base = app.engine.decide(_step(name), app.policy_ctx)
    assert base.tier == tiers.TIER_CONFIRM_UNLOCK
    with_ctx = _decide(app, name, origin=WEB, allowed_domains=SCOPE, scope_approved=True)
    assert with_ctx.tier >= base.tier


def test_hard_blocked_tool_stays_blocked_with_context() -> None:
    app, name = _app(name="fake_delete_now")
    base = app.engine.decide(_step(name), app.policy_ctx)
    assert base.tier == tiers.TIER_BLOCKED and not base.allowed
    with_ctx = _decide(app, name, origin=WEB, allowed_domains=SCOPE, scope_approved=True)
    assert with_ctx.tier == tiers.TIER_BLOCKED and not with_ctx.allowed


def test_action_hash_ignores_transient_target_context() -> None:
    app, name = _app()
    base = app.engine.decide(_step(name), app.policy_ctx)
    with_ctx = _target_decision(app, name, _browser())
    assert with_ctx.action_hash == base.action_hash
