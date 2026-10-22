"""Site directory + ``browser_open_site`` tests (Stage 8.6, slice 8.6.1).

No Playwright, no Chrome, no network.  These tests pin the contract that makes
the directory safe:

* only owner-configured names resolve (no fallback, no invented URL);
* a configured URL that is not absolute http(s) without userinfo is dropped,
  so a typo in ``config.toml`` can never become a navigation;
* the directory grants **no** permission - ``browser_open_site`` produces the
  same browser target as ``browser_goto`` and is decided by the same engine;
* the brain prompt lists the site *names* (never URLs to copy).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from jarvis.browser.context import BrowserError
from jarvis.browser.sites import (
    Site,
    SiteDirectory,
    render_site_catalogue,
    sites_from_settings,
)
from jarvis.browser.tools import (
    BrowserOpenSiteArgs,
    make_browser_open_site_spec,
)
from jarvis.config import Settings, SiteSettings
from jarvis.policy.targets import ResolvedTarget
from jarvis.tools.base import ToolContext, resolve_target


class FakeSession:
    """A minimal managed-browser fake that records the navigations."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.goto_error: str | None = None

    def goto(self, url: str) -> dict[str, object]:
        self.calls.append(("goto", url))
        if self.goto_error:
            raise BrowserError(self.goto_error)
        return {"url": url, "origin": "https://www.youtube.com", "title": "YouTube"}


def _settings(*sites: SiteSettings) -> Settings:
    return Settings(sites=list(sites))


def _ctx(settings: Settings, session: FakeSession | None = None) -> ToolContext:
    return ToolContext(settings=settings, browser=session or FakeSession())


# ── the directory ──────────────────────────────────────────────────────────


def test_a_configured_site_resolves_by_name() -> None:
    directory = sites_from_settings(
        _settings(SiteSettings(name="youtube", url="https://www.youtube.com"))
    )
    site = directory.resolve("youtube")
    assert site is not None
    assert site.url == "https://www.youtube.com"
    assert site.host == "www.youtube.com"


def test_a_site_resolves_case_insensitively_and_by_alias() -> None:
    directory = sites_from_settings(
        _settings(SiteSettings(name="YouTube", url="https://www.youtube.com", aliases=["yt", "YT"]))
    )
    assert directory.resolve("youtube") is not None
    assert directory.resolve("  YOUTUBE  ") is not None
    assert directory.resolve("yt") is not None


def test_an_unknown_name_never_resolves() -> None:
    directory = sites_from_settings(
        _settings(SiteSettings(name="youtube", url="https://www.youtube.com"))
    )
    assert directory.resolve("facebook") is None
    assert directory.resolve("") is None
    assert directory.resolve("https://evil.example") is None


def test_an_unsafe_configured_url_is_dropped() -> None:
    directory = sites_from_settings(
        _settings(
            SiteSettings(name="good", url="https://good.example"),
            SiteSettings(name="file", url="file:///etc/passwd"),
            SiteSettings(name="creds", url="https://user:pw@host.example"),
            SiteSettings(name="js", url="javascript:alert(1)"),
        )
    )
    assert directory.names() == ["good"]
    assert directory.resolve("file") is None


def test_a_blank_name_is_dropped() -> None:
    directory = sites_from_settings(
        _settings(
            SiteSettings(name="   ", url="https://good.example"),
            SiteSettings(name="ok", url="https://ok.example"),
        )
    )
    assert directory.names() == ["ok"]


def test_the_first_entry_wins_a_duplicated_key() -> None:
    directory = SiteDirectory(
        [Site("yt", "https://first.example"), Site("yt", "https://second.example")]
    )
    site = directory.resolve("yt")
    assert site is not None
    assert site.url == "https://first.example"


def test_the_catalogue_lists_names_and_hosts_only() -> None:
    directory = sites_from_settings(
        _settings(
            SiteSettings(name="youtube", url="https://www.youtube.com", aliases=["yt"]),
            SiteSettings(name="news", url="https://news.example"),
        )
    )
    text = render_site_catalogue(directory)
    assert "youtube" in text and "www.youtube.com" in text
    assert "news" in text
    # no full URL is shown: the model picks a name, the code supplies the URL
    assert "https://" not in text


def test_an_empty_directory_renders_nothing() -> None:
    assert render_site_catalogue(sites_from_settings(Settings())) == ""


# ── the tool ───────────────────────────────────────────────────────────────


def test_the_tool_navigates_to_the_configured_url() -> None:
    session = FakeSession()
    spec = make_browser_open_site_spec()
    settings = _settings(SiteSettings(name="youtube", url="https://www.youtube.com"))
    args = BrowserOpenSiteArgs(site="youtube")
    result = spec.execute(args, _ctx(settings, session))
    assert result.ok is True
    assert session.calls == [("goto", "https://www.youtube.com")]
    assert result.data["site"] == "youtube"


def test_the_tool_refuses_an_unknown_site_and_lists_the_known_ones() -> None:
    session = FakeSession()
    spec = make_browser_open_site_spec()
    settings = _settings(SiteSettings(name="youtube", url="https://www.youtube.com"))
    args = BrowserOpenSiteArgs(site="facebook")
    result = spec.execute(args, _ctx(settings, session))
    assert result.ok is False
    assert "unknown site" in (result.error or "")
    assert "youtube" in (result.error or "")
    assert session.calls == []  # nothing was navigated


def test_the_tool_reports_a_browser_failure_without_raising() -> None:
    session = FakeSession()
    session.goto_error = "the managed browser session is not started"
    spec = make_browser_open_site_spec()
    settings = _settings(SiteSettings(name="youtube", url="https://www.youtube.com"))
    result = spec.execute(BrowserOpenSiteArgs(site="youtube"), _ctx(settings, session))
    assert result.ok is False
    assert "browser_open_site failed" in (result.error or "")


def test_the_tool_marks_the_page_as_tainted() -> None:
    spec = make_browser_open_site_spec()
    settings = _settings(SiteSettings(name="youtube", url="https://www.youtube.com"))
    result = spec.execute(BrowserOpenSiteArgs(site="youtube"), _ctx(settings))
    assert result.tainted is True


def test_extra_arguments_are_refused() -> None:
    with pytest.raises(ValidationError):
        BrowserOpenSiteArgs(site="youtube", url="https://evil.example")


# ── the resolver (the policy gate's view of the action) ───────────────────


def test_the_resolver_reports_the_documents_origin() -> None:
    spec = make_browser_open_site_spec()
    settings = _settings(SiteSettings(name="youtube", url="https://www.youtube.com"))
    target = resolve_target(spec, BrowserOpenSiteArgs(site="youtube"), _ctx(settings))
    assert isinstance(target, ResolvedTarget)
    assert target.kind == "browser"
    assert target.origin == "https://www.youtube.com"
    assert target.role == "document"


def test_the_resolver_refuses_an_unknown_site() -> None:
    spec = make_browser_open_site_spec()
    settings = _settings(SiteSettings(name="youtube", url="https://www.youtube.com"))
    assert resolve_target(spec, BrowserOpenSiteArgs(site="facebook"), _ctx(settings)) is None


def test_the_tool_costs_one_scope_action_unit() -> None:
    spec = make_browser_open_site_spec()
    assert spec.base_tier == 1
    assert spec.budget_cost == 1


# ── the registry ──────────────────────────────────────────────────────────


def test_the_tool_is_registered_by_default() -> None:
    from jarvis.tools.registry import build_default_registry

    registry = build_default_registry()
    assert registry.get_optional("browser_open_site") is not None


# ── the config file and the brain prompt ──────────────────────────────────


def test_the_directory_loads_from_a_config_file(tmp_path: Path) -> None:
    from jarvis.config import load_settings

    config = tmp_path / "config.toml"
    config.write_text(
        '[[sites]]\nname = "youtube"\nurl = "https://www.youtube.com"\naliases = ["yt"]\n',
        encoding="utf-8",
    )
    directory = sites_from_settings(load_settings(config))
    assert directory.names() == ["youtube"]
    assert directory.resolve("yt") is not None


def test_the_brain_prompt_offers_the_site_names_not_urls() -> None:
    from jarvis.agent.context import make_app_context
    from jarvis.agent.nodes.brain import brain
    from jarvis.agent.schemas import BrainDecision
    from jarvis.llm.client import Usage
    from support import brain_conversation, make_spec, registry_with

    class _RecorderLLM:
        def __init__(self) -> None:
            self.user: str | None = None

        def structured(
            self,
            *,
            system: str,
            user: str,
            schema: type[object],
            model_role: str = "planner",
            temperature: float = 0.0,
            max_retries: int | None = None,
            timeout_s: float | None = None,
        ) -> tuple[BrainDecision, Usage]:
            self.user = user
            return brain_conversation("hi"), Usage(calls=1, prompt_tokens=1, completion_tokens=1)

    recorder = _RecorderLLM()
    settings = _settings(SiteSettings(name="youtube", url="https://www.youtube.com"))
    ctx = make_app_context(
        settings,
        llm=recorder,
        registry=registry_with(make_spec("fake_echo", base_tier=0, record=[])),
    )
    brain({"user_input": "open youtube", "source": "terminal", "task_id": "t1"}, ctx)
    assert recorder.user is not None
    assert "youtube" in recorder.user
    assert "browser_open_site" in recorder.user
    # the model is never handed a URL to copy
    assert "https://www.youtube.com" not in recorder.user
