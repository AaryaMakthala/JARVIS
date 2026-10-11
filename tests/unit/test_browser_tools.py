"""Browser tool tests (Stage 7, Slice 6).

No Playwright, no Chrome, no network: each test injects a fake
:class:`~jarvis.browser.context.BrowserSession` into a plain ``ToolContext``
and asserts the tool contract - validation, taint marking, transient
references, resolver compatibility, credential refusal and that typed text
never leaks into output, data, describe or error strings.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.browser.context import BrowserError, ElementRef, Observation
from jarvis.browser.tools import (
    BrowserClickArgs,
    BrowserGotoArgs,
    BrowserReadArgs,
    BrowserTypeArgs,
    make_browser_click_spec,
    make_browser_goto_spec,
    make_browser_read_spec,
    make_browser_type_spec,
)
from jarvis.config import Settings
from jarvis.policy.targets import ResolvedTarget
from jarvis.tools.base import ToolContext, ToolResult, resolve_target

_ORIGIN = "https://example.com"


def _element(
    ref: str,
    name: str,
    *,
    role: str = "button",
    input_type: str = "",
    is_password: bool = False,
) -> ElementRef:
    return ElementRef(
        ref=ref,
        role=role,
        name=name,
        origin=_ORIGIN,
        input_type=input_type,
        is_password=is_password,
    )


class FakeSession:
    """Records every call and returns canned managed-browser results."""

    def __init__(self, elements: dict[str, ElementRef] | None = None) -> None:
        self.resolved = elements if elements is not None else {"e1": _element("e1", "Send")}
        self.calls: list[tuple[object, ...]] = []
        self.goto_error: str | None = None
        self.read_error: str | None = None
        self.click_error: str | None = None
        self.type_error: str | None = None
        self.resolve_raises = False
        self.observation = Observation(
            url=f"{_ORIGIN}/page",
            origin=_ORIGIN,
            title="Example",
            elements=(self.resolved.get("e1", _element("e1", "Send")),),
        )
        self.read_max: int | None = None

    def goto(self, url: str) -> dict[str, object]:
        self.calls.append(("goto", url))
        if self.goto_error:
            raise BrowserError(self.goto_error)
        return {"url": url, "origin": _ORIGIN, "title": "Example"}

    def read(self, max_elements: int = 80) -> Observation:
        self.calls.append(("read", max_elements))
        self.read_max = max_elements
        if self.read_error:
            raise BrowserError(self.read_error)
        return self.observation

    def resolve(self, ref: str) -> ElementRef | None:
        self.calls.append(("resolve", ref))
        if self.resolve_raises:
            raise RuntimeError("resolver exploded")
        return self.resolved.get(ref)

    def click(self, ref: str) -> dict[str, object]:
        self.calls.append(("click", ref))
        if self.click_error:
            raise BrowserError(self.click_error)
        el = self.resolved[ref]
        return {"ref": ref, "role": el.role, "name": el.name, "origin": el.origin}

    def type_text(self, ref: str, text: str) -> dict[str, object]:
        self.calls.append(("type_text", ref, text))
        if self.type_error:
            raise BrowserError(self.type_error)
        el = self.resolved[ref]
        return {
            "ref": ref,
            "role": el.role,
            "name": el.name,
            "origin": el.origin,
            "characters": len(text),
        }

    def close(self) -> None:  # pragma: no cover - protocol completeness
        self.calls.append(("close",))


def _ctx(browser: object | None = None) -> ToolContext:
    return ToolContext(settings=Settings(), browser=browser)


# ── spec contract ──────────────────────────────────────────────────────────


def test_spec_contract() -> None:
    goto = make_browser_goto_spec()
    read = make_browser_read_spec()
    click = make_browser_click_spec()
    type_ = make_browser_type_spec()
    assert (goto.name, goto.base_tier) == ("browser_goto", 1)
    assert (read.name, read.base_tier) == ("browser_read", 0)
    assert (click.name, click.base_tier) == ("browser_click", 1)
    assert (type_.name, type_.base_tier) == ("browser_type", 1)
    # Registration must not require Windows or a live browser.
    for spec in (goto, read, click, type_):
        assert spec.windows_only is False
        assert spec.args_model is not None
    # Resolve hooks exist exactly where a target is involved.
    assert goto.resolve is not None
    assert click.resolve is not None
    assert type_.resolve is not None
    assert read.resolve is None


def test_spec_budget_costs() -> None:
    """State-changing tools cost one budget unit; observations cost nothing."""
    assert make_browser_goto_spec().budget_cost == 1
    assert make_browser_click_spec().budget_cost == 1
    assert make_browser_type_spec().budget_cost == 1
    assert make_browser_read_spec().budget_cost == 0


def test_args_models_are_strict() -> None:
    with pytest.raises(ValidationError):
        BrowserReadArgs(max_elements=81)
    with pytest.raises(ValidationError):
        BrowserTypeArgs(ref="e1")  # text is required
    with pytest.raises(ValidationError):
        BrowserGotoArgs(url="https://x/", extra="nope")


# ── missing session ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "maker,args",
    [
        (make_browser_goto_spec, BrowserGotoArgs(url="https://example.com/")),
        (make_browser_read_spec, BrowserReadArgs()),
        (make_browser_click_spec, BrowserClickArgs(ref="e1")),
        (make_browser_type_spec, BrowserTypeArgs(ref="e1", text="hi")),
    ],
)
def test_missing_session_fails_cleanly(maker, args) -> None:
    spec = maker()
    result = spec.execute(args, _ctx(browser=None))
    assert result.ok is False
    assert "not available" in (result.error or "")


# ── goto ───────────────────────────────────────────────────────────────────


def test_goto_ok() -> None:
    session = FakeSession()
    result = make_browser_goto_spec().execute(
        BrowserGotoArgs(url="https://example.com/x"), _ctx(session)
    )
    assert result.ok is True
    assert result.tainted is True
    assert result.data["origin"] == _ORIGIN
    assert session.calls == [("goto", "https://example.com/x")]


@pytest.mark.parametrize(
    "url", ["javascript:alert(1)", "file:///c:/secret", "https://u:p@example.com/", "nonsense"]
)
def test_goto_rejects_unsafe_urls(url: str) -> None:
    session = FakeSession()
    result = make_browser_goto_spec().execute(BrowserGotoArgs(url=url), _ctx(session))
    assert result.ok is False
    assert session.calls == []  # never reached the session


def test_goto_redirect_error_is_safe() -> None:
    session = FakeSession()
    session.goto_error = "navigation redirected to unauthorized origin 'https://evil.example'"
    result = make_browser_goto_spec().execute(
        BrowserGotoArgs(url="https://example.com/"), _ctx(session)
    )
    assert result.ok is False
    assert (result.error or "").startswith("browser_goto failed:")


def test_goto_verify_sets_verified() -> None:
    spec = make_browser_goto_spec()
    args = BrowserGotoArgs(url="https://example.com/")
    ok = ToolResult(ok=True, data={"origin": _ORIGIN})
    assert spec.verify is not None
    assert spec.verify(args, ok, _ctx()).verified is True
    assert spec.verify(args, ToolResult(ok=True, data={}), _ctx()).verified is False


# ── read ───────────────────────────────────────────────────────────────────


def test_read_is_bounded_and_tainted() -> None:
    session = FakeSession()
    result = make_browser_read_spec().execute(BrowserReadArgs(max_elements=3), _ctx(session))
    assert result.ok is True
    assert result.tainted is True  # page-derived content is untrusted
    assert session.read_max == 3
    assert "[e1] button" in result.output
    assert result.data["origin"] == _ORIGIN
    assert len(result.data["elements"]) == 1


def test_read_error_is_safe() -> None:
    session = FakeSession()
    session.read_error = "managed browser session is not started"
    result = make_browser_read_spec().execute(BrowserReadArgs(), _ctx(session))
    assert result.ok is False
    assert (result.error or "").startswith("browser_read failed:")


# ── resolver hooks ─────────────────────────────────────────────────────────


def test_click_resolver_builds_target() -> None:
    session = FakeSession()
    spec = make_browser_click_spec()
    args = BrowserClickArgs(ref="e1")
    target = spec.resolve(args, _ctx(session))
    assert isinstance(target, ResolvedTarget)
    assert target.kind == "browser"
    assert (target.ref, target.role, target.name) == ("e1", "button", "Send")
    assert target.origin == _ORIGIN
    assert target.fingerprint() is not None
    assert resolve_target(spec, args, _ctx(session)) is not None


def test_goto_resolver_builds_document_target() -> None:
    session = FakeSession()
    spec = make_browser_goto_spec()
    target = spec.resolve(BrowserGotoArgs(url="https://example.com/a/b?q=1"), _ctx(session))
    assert isinstance(target, ResolvedTarget)
    assert target.kind == "browser"
    assert target.role == "document"
    assert target.origin == _ORIGIN
    assert target.name == "example.com"
    assert spec.resolve(BrowserGotoArgs(url="javascript:alert(1)"), _ctx(session)) is None


def test_resolver_fails_closed() -> None:
    spec = make_browser_click_spec()
    args = BrowserClickArgs(ref="e1")
    # Unknown ref -> None (never authorizes an unresolved target).
    assert spec.resolve(args, _ctx(FakeSession(elements={}))) is None
    # A broken resolver must never crash: resolve_target returns None.
    broken = FakeSession()
    broken.resolve_raises = True
    assert spec.resolve(args, _ctx(broken)) is None
    assert resolve_target(spec, args, _ctx(broken)) is None
    # No session at all -> None.
    assert resolve_target(spec, args, _ctx(browser=None)) is None


# ── click ──────────────────────────────────────────────────────────────────


def test_click_ok() -> None:
    session = FakeSession()
    result = make_browser_click_spec().execute(BrowserClickArgs(ref="e1"), _ctx(session))
    assert result.ok is True
    assert result.tainted is True
    assert result.data["ref"] == "e1"
    assert ("click", "e1") in session.calls


def test_click_stale_ref_error_is_safe() -> None:
    session = FakeSession()
    session.click_error = "unknown or stale element reference 'e1'"
    result = make_browser_click_spec().execute(BrowserClickArgs(ref="e1"), _ctx(session))
    assert result.ok is False
    assert (result.error or "").startswith("browser_click failed:")


# ── type ───────────────────────────────────────────────────────────────────


_SECRET = "sup3r-secret-token"


def test_type_never_leaks_text() -> None:
    session = FakeSession(elements={"e1": _element("e1", "Search", role="textbox")})
    spec = make_browser_type_spec()
    args = BrowserTypeArgs(ref="e1", text=_SECRET)
    result = spec.execute(args, _ctx(session))
    assert result.ok is True
    assert result.data["characters"] == len(_SECRET)
    assert "text" not in result.data
    blob = f"{result.output}\n{result.data}\n{result.error}\n{spec.describe(args)}"
    assert _SECRET not in blob
    assert ("type_text", "e1", _SECRET) in session.calls  # the session did receive it


def test_type_refuses_password_field() -> None:
    session = FakeSession(
        elements={
            "p1": _element(
                "p1", "Password", role="textbox", input_type="password", is_password=True
            )
        }
    )
    spec = make_browser_type_spec()
    result = spec.execute(BrowserTypeArgs(ref="p1", text=_SECRET), _ctx(session))
    assert result.ok is False
    assert "credential" in (result.error or "")
    assert _SECRET not in (result.error or "")
    assert ("type_text", "p1", _SECRET) not in session.calls  # never typed


def test_type_unknown_ref_refused() -> None:
    session = FakeSession(elements={})
    result = make_browser_type_spec().execute(
        BrowserTypeArgs(ref="e1", text=_SECRET), _ctx(session)
    )
    assert result.ok is False
    assert _SECRET not in (result.error or "")
    assert ("type_text", "e1", _SECRET) not in session.calls


def test_type_driver_error_does_not_include_text() -> None:
    session = FakeSession(elements={"e1": _element("e1", "Search", role="textbox")})
    session.type_error = "target is not a text input; trying to type into it is refused"
    result = make_browser_type_spec().execute(
        BrowserTypeArgs(ref="e1", text=_SECRET), _ctx(session)
    )
    assert result.ok is False
    assert _SECRET not in (result.error or "")


# ── DI plumbing ────────────────────────────────────────────────────────────


def test_app_context_threads_browser() -> None:
    from jarvis.agent.context import make_app_context
    from jarvis.tools.registry import build_default_registry

    session = FakeSession()
    app = make_app_context(
        settings=Settings(),
        registry=build_default_registry(Settings()),
        browser=session,
    )
    assert app.tool_context().browser is session
    assert make_app_context(settings=Settings()).tool_context().browser is None
