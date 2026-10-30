"""W1: the generic browser can never reach a WhatsApp host.

These tests pin one security rule: ``whatsapp.com`` and every true subdomain of
it are permanently denied to the *generic* managed browser, at every boundary -
host matching, task-scope grant issuance, browsing-grant issuance and coverage,
policy-engine origin approval (static **or** dynamic allowed domains), and
navigation.  The dedicated WhatsApp integration is unaffected: it drives its own
isolated profile and pins its own exact origin.

No browser, network or WhatsApp session is started here; every collaboration
object is a fake.
"""

from __future__ import annotations

import dataclasses

import pytest

from jarvis.agent.context import make_app_context
from jarvis.agent.state import Step
from jarvis.browser.tools import (
    BrowserGotoArgs,
    _resolve_browser_goto,
    _run_browser_goto,
)
from jarvis.browser.whatsapp_web import WHATSAPP_ORIGIN, WhatsAppDriver, WhatsAppError
from jarvis.config import Settings
from jarvis.policy import tiers
from jarvis.policy.browse_grant import (
    BrowseGrant,
    BrowseGrantError,
    BrowseGrantRegistry,
    coverage,
)
from jarvis.policy.denylist import (
    DENIED_BROWSER_HOSTS,
    canonical_host,
    denied_host_reason,
    host_is_denied,
)
from jarvis.policy.scope import ScopeError, ScopeRegistry
from jarvis.tools.base import ToolContext
from support import make_spec, registry_with

# ── host matching ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "value",
    [
        "whatsapp.com",
        "web.whatsapp.com",
        "foo.whatsapp.com",
        "a.b.whatsapp.com",
        "WEB.WhatsApp.COM",
        "web.whatsapp.com.",
        "https://web.whatsapp.com",
        "https://WEB.WhatsApp.com:443/send?phone=919876543210",
    ],
)
def test_whatsapp_hosts_and_subdomains_are_denied(value: str) -> None:
    assert host_is_denied(value) is True
    assert denied_host_reason(value)


@pytest.mark.parametrize(
    "value",
    [
        "example.com",
        "notwhatsapp.com",
        "whatsappx.com",
        "whatsapp.com.attacker.invalid",
        "my.whatsapp.com.evil",
        "https://example.com/path",
    ],
)
def test_lookalike_hosts_are_not_mistaken_for_whatsapp(value: str) -> None:
    assert host_is_denied(value) is False
    assert denied_host_reason(value) == ""


@pytest.mark.parametrize("value", ["", "   ", "http://user@web.whatsapp.com", "web.whatsapp.com/x"])
def test_an_unreadable_host_fails_closed(value: str) -> None:
    assert host_is_denied(value) is True


def test_only_whatsapp_com_is_denied_by_default() -> None:
    assert DENIED_BROWSER_HOSTS == ("whatsapp.com",)


def test_hosts_are_canonicalised_before_matching() -> None:
    assert canonical_host("WEB.WhatsApp.com.") == "web.whatsapp.com"
    assert canonical_host("https://web.whatsapp.com:443/send") == "web.whatsapp.com"
    assert canonical_host("https://user@web.whatsapp.com") is None
    assert canonical_host("") is None


# ── task-scope grant issuance ──────────────────────────────────────────────


@pytest.mark.parametrize("host", ["whatsapp.com", "web.whatsapp.com", "a.b.whatsapp.com"])
def test_scope_grant_cannot_be_issued_for_a_whatsapp_host(host: str) -> None:
    registry = ScopeRegistry()
    with pytest.raises(ScopeError, match="dedicated WhatsApp"):
        registry.issue(
            "t1", plan_hash="p1", confirmation_id=f"cid-{host}", allowed_domains=[host]
        )
    assert registry.active_grant("t1") is None


def test_a_mixed_scope_is_refused_whole() -> None:
    registry = ScopeRegistry()
    with pytest.raises(ScopeError, match="dedicated WhatsApp"):
        registry.issue(
            "t1",
            plan_hash="p1",
            confirmation_id="cid-mixed",
            allowed_domains=["example.com", "web.whatsapp.com"],
        )
    assert registry.active_grant("t1") is None


def test_an_ordinary_host_is_still_scopeable() -> None:
    registry = ScopeRegistry()
    grant = registry.issue(
        "t1", plan_hash="p1", confirmation_id="cid-ok", allowed_domains=["Example.com"]
    )
    assert grant.contains_domain("example.com")


# ── browsing-grant issuance and coverage ───────────────────────────────────


@pytest.mark.parametrize("host", ["whatsapp.com", "web.whatsapp.com", "x.whatsapp.com"])
def test_browsing_grant_cannot_be_issued_for_a_whatsapp_host(host: str) -> None:
    registry = BrowseGrantRegistry()
    with pytest.raises(BrowseGrantError, match="dedicated WhatsApp"):
        registry.issue("t1", host=host)
    assert registry.active_for_task("t1") is None


def _hand_built_grant(host: str) -> BrowseGrant:
    """A grant built directly, i.e. as if one had survived the new check."""
    return BrowseGrant(
        task_id="t1",
        grant_id="g1",
        host=host,
        max_actions=30,
        issued_at=0.0,
        expires_at=1_000.0,
    )


@pytest.mark.parametrize("host", ["web.whatsapp.com", "whatsapp.com", "a.b.whatsapp.com"])
def test_an_existing_grant_cannot_cover_a_whatsapp_host(host: str) -> None:
    ok, reason = coverage(
        _hand_built_grant(host),
        task_id="t1",
        tool="browser_goto",
        tier=1,
        host=host,
        now=1.0,
    )
    assert ok is False
    assert reason == "host_denied"


def test_an_ordinary_host_is_still_covered() -> None:
    registry = BrowseGrantRegistry()
    grant = registry.issue("t1", host="example.com")
    ok, reason = coverage(
        grant, task_id="t1", tool="browser_goto", tier=1, host="example.com", now=1.0
    )
    assert (ok, reason) == (True, "")


# ── policy-engine origin approval ──────────────────────────────────────────


def _decide(
    origin: str, allowed_domains: tuple[str, ...], *, target: object | None = None
) -> object:
    registry = registry_with(make_spec("fake_act", base_tier=0, record=[]))
    app = make_app_context(Settings(), registry=registry)
    context = dataclasses.replace(
        app.policy_ctx,
        origin=origin,
        allowed_domains=allowed_domains,
        scope_approved=True,
        target=target,
    )
    step = Step(id="s1", tool="fake_act", args={"text": "hi"}, rationale="r")
    return app.engine.decide(step, context)


@pytest.mark.parametrize(
    "origin",
    [
        WHATSAPP_ORIGIN,
        "https://whatsapp.com",
        "https://a.b.whatsapp.com",
        "https://WEB.WhatsApp.com:443",
    ],
)
def test_engine_blocks_whatsapp_origins_even_when_the_scope_names_them(
    origin: str,
) -> None:
    decision = _decide(origin, ("whatsapp.com", "web.whatsapp.com"))
    assert decision.allowed is False
    assert decision.tier == tiers.TIER_BLOCKED
    assert any("dedicated WhatsApp" in reason for reason in decision.reasons)


def test_a_dynamic_allowed_domain_never_overrides_the_denylist() -> None:
    """The bypass H3: a granted host must not re-open WhatsApp to the browser."""
    decision = _decide(WHATSAPP_ORIGIN, ("web.whatsapp.com",))
    assert decision.allowed is False and decision.tier == tiers.TIER_BLOCKED


def test_an_ordinary_origin_is_still_allowed_in_scope() -> None:
    decision = _decide("https://example.com", ("example.com",))
    assert decision.allowed is True and decision.tier == tiers.TIER_SAFE


# ── navigation ─────────────────────────────────────────────────────────────


def test_browser_goto_refuses_a_whatsapp_url_before_touching_the_session() -> None:
    context = ToolContext(settings=Settings(), browser=None)
    result = _run_browser_goto(BrowserGotoArgs(url="https://web.whatsapp.com/"), context)
    assert result.ok is False
    assert "dedicated WhatsApp" in (result.error or "")
    # The refusal happens before the session is even requested.
    assert "managed browser" not in (result.error or "")


def test_browser_goto_target_resolution_refuses_a_whatsapp_url() -> None:
    context = ToolContext(settings=Settings())
    assert _resolve_browser_goto(BrowserGotoArgs(url=WHATSAPP_ORIGIN), context) is None


def test_static_config_cannot_pre_approve_whatsapp() -> None:
    assert "web.whatsapp.com" not in Settings().browser.allowed_domains


# ── the dedicated WhatsApp integration is untouched ────────────────────────


class _WhatsAppPage:
    """The minimum page surface ``open_chat_by_phone`` uses."""

    def __init__(self, url: str = "", header: str = "Mom", redirect_to: str = "") -> None:
        self._url = url or f"{WHATSAPP_ORIGIN}/"
        self._header = header
        self._redirect_to = redirect_to

    def url(self) -> str:
        return self._url

    def goto(self, url: str) -> None:
        # A redirect is the interesting case: the page ends up somewhere else.
        self._url = self._redirect_to or url

    def chat_header(self) -> str:
        return self._header


def test_the_dedicated_driver_still_opens_a_chat_on_its_own_origin() -> None:
    driver = WhatsAppDriver(_WhatsAppPage())
    assert driver.open_chat_by_phone("919876543210", "Mom") == "Mom"


def test_the_dedicated_driver_still_refuses_a_foreign_origin() -> None:
    page = _WhatsAppPage(redirect_to="https://evil.example/send")
    with pytest.raises(WhatsAppError, match="not on"):
        WhatsAppDriver(page).open_chat_by_phone("919876543210", "Mom")


def test_the_dedicated_tools_keep_their_tiers() -> None:
    from jarvis.tools.whatsapp_web import (
        make_whatsapp_open_chat_spec,
        make_whatsapp_send_file_spec,
    )

    assert make_whatsapp_open_chat_spec().base_tier == 1
    assert make_whatsapp_send_file_spec().base_tier == 2
