"""Browser tools (Stage 7, Slice 6): ``browser_goto``, ``browser_read``,
``browser_click``, ``browser_type``.

Everything here drives the JARVIS-owned managed Chrome context
(:mod:`jarvis.browser.context`); no real browser, Chrome profile or Playwright
process is ever touched at import/registration time.

Security posture
----------------
* ``browser_goto`` only accepts canonical http(s) URLs without userinfo and
  fails closed when a redirect ends up on an origin that is neither the
  requested one nor in ``allowed_domains`` (docs/03 B4/D18).  The tool never
  substitutes its own checks for the PolicyEngine's authorization - it only
  refuses structurally unsafe navigation and stops *after* an unauthorized
  redirect so no subsequent interaction runs against the wrong page.
* ``browser_read`` returns a bounded accessibility observation; every element
  is a transient, page-scoped handle (``e1..eN``) with role/name/origin, and
  the whole result is flagged ``tainted`` (untrusted page-derived text).
* ``browser_click``/``browser_type`` re-validate the observed identity
  immediately before acting (stale/ambiguous/changed refs are refused; no
  coordinate fallback).  ``browser_type`` refuses credential fields and never
  logs, returns or fingerprints the typed text.
* Each spec exposes a ``resolve`` hook (Slice 5) so the policy gate can
  observe the target *before* authorization.  Missing resolution is a refusal.

State-changing tools are ``browser_goto``/``browser_click``/``browser_type``
(one budget unit each); ``browser_read`` is a Tier-0 observation and costs
nothing (docs/02 Stage 7; config ``[browser]``).
"""

from __future__ import annotations

import urllib.parse
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from jarvis.browser.context import BrowserError, BrowserSession, ElementRef
from jarvis.browser.find import match_identity
from jarvis.browser.sites import sites_from_settings
from jarvis.policy.rules import url_origin
from jarvis.policy.targets import ResolvedTarget
from jarvis.tools.base import ToolContext, ToolResult, ToolSpec
from jarvis.tools.web import is_allowed_http_url

__all__ = [
    "BrowserClickArgs",
    "BrowserFindArgs",
    "BrowserGotoArgs",
    "BrowserOpenResultArgs",
    "BrowserOpenSiteArgs",
    "BrowserReadArgs",
    "BrowserTypeArgs",
    "make_browser_click_spec",
    "make_browser_find_spec",
    "make_browser_goto_spec",
    "make_browser_open_result_spec",
    "make_browser_open_site_spec",
    "make_browser_read_spec",
    "make_browser_type_spec",
]

_URL_MAX = 2083
_TEXT_MAX = 4096
_READ_MAX = 80
_REF_MAX = 64
_SITE_MAX = 64
_QUERY_MAX = 200
_ROLE_MAX = 32
_FIND_MAX = 20
_INDEX_MAX = 50


class BrowserGotoArgs(BaseModel):
    """Arguments for ``browser_goto``."""

    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=1, max_length=_URL_MAX, description="Absolute http(s) URL.")


class BrowserReadArgs(BaseModel):
    """Arguments for ``browser_read``."""

    model_config = ConfigDict(extra="forbid")

    max_elements: int = Field(
        default=_READ_MAX,
        ge=1,
        le=_READ_MAX,
        description=f"Maximum number of elements to observe (1-{_READ_MAX}).",
    )


class BrowserClickArgs(BaseModel):
    """Arguments for ``browser_click``."""

    model_config = ConfigDict(extra="forbid")

    ref: str = Field(min_length=1, max_length=_REF_MAX, description="Element reference.")


class BrowserTypeArgs(BaseModel):
    """Arguments for ``browser_type`` (the typed text is never logged)."""

    model_config = ConfigDict(extra="forbid")

    ref: str = Field(min_length=1, max_length=_REF_MAX, description="Element reference.")
    text: str = Field(min_length=1, max_length=_TEXT_MAX, description="Text to type.")


class BrowserOpenSiteArgs(BaseModel):
    """Arguments for ``browser_open_site`` (Stage 8.6)."""

    model_config = ConfigDict(extra="forbid")

    site: str = Field(
        min_length=1,
        max_length=_SITE_MAX,
        description="Name of a site from the configured site directory.",
    )


class BrowserFindArgs(BaseModel):
    """Arguments for ``browser_find`` (Stage 8.6)."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        max_length=_QUERY_MAX,
        description="Text to look for in the visible names of the page's elements.",
    )
    role: str = Field(
        default="",
        max_length=_ROLE_MAX,
        description="Optional role filter, e.g. 'link' or 'button'.",
    )
    limit: int = Field(
        default=_FIND_MAX,
        ge=1,
        le=_FIND_MAX,
        description=f"Maximum number of matches to report (1-{_FIND_MAX}).",
    )


class BrowserOpenResultArgs(BaseModel):
    """Arguments for ``browser_open_result`` (Stage 8.6)."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        max_length=_QUERY_MAX,
        description="Text to look for in the visible names of the page's elements.",
    )
    role: str = Field(
        default="",
        max_length=_ROLE_MAX,
        description="Optional role filter, e.g. 'link' or 'button'.",
    )
    index: int | None = Field(
        default=None,
        ge=1,
        le=_INDEX_MAX,
        description=(
            "Which match to open, counted from 1. Required when several elements match the query."
        ),
    )


# ── session helpers ────────────────────────────────────────────────────────


def _require_session(ctx: ToolContext) -> BrowserSession | None:
    """Return the managed session, or ``None`` when it is not wired up."""
    session = ctx.browser
    if session is None:
        return None
    return session  # type: ignore[return-value]


def _safe_goto(url: str) -> str | None:
    """Return the canonical origin of a structurally safe URL, else ``None``.

    Only absolute http(s) URLs without userinfo are accepted; the same check
    backs the ``resolve`` hook so the tool never hands an unsafe navigation to
    the driver (defence in depth - the driver re-checks).
    """
    if not is_allowed_http_url(url):
        return None
    return url_origin(url)


def _current_page_url(session: Any) -> str:
    """The URL of the page the managed browser is on (best effort).

    A session that cannot report it (an older fake, a browser that is not
    started) simply has no page to remember.
    """
    getter = getattr(session, "current_url", None)
    if getter is None:
        return ""
    try:
        return str(getter() or "").strip()
    except Exception:  # noqa: BLE001 - a session that cannot report it has none
        return ""


def _with_page_url(data: dict[str, Any], session: Any, url: str = "") -> dict[str, Any]:
    """Add the current page URL so a later turn can refer to this page.

    Stage 8.6, slice 8.6.3.  The URL is a *location*, not an authorization:
    the next navigation to it still goes through the ordinary domain-scope
    rules, and it is rendered as untrusted data in the prompt.
    """
    page_url = str(url or "").strip() or _current_page_url(session)
    if page_url:
        data["page_url"] = page_url
    return data


def _run_browser_goto(args: BrowserGotoArgs, ctx: ToolContext) -> ToolResult:
    if _safe_goto(args.url) is None:
        return ToolResult(
            ok=False,
            error="refusing a URL that is not absolute http(s) without userinfo",
        )
    session = _require_session(ctx)
    if session is None:
        return ToolResult(ok=False, error="the managed browser is not available in this context")
    try:
        data = session.goto(args.url)
    except BrowserError as exc:
        return ToolResult(ok=False, error=f"browser_goto failed: {exc}")
    return ToolResult(
        ok=True,
        output=f"navigated to {data.get('url', '')}",
        data=_with_page_url(dict(data), session, str(data.get("url", ""))),
        verified=True,
        tainted=True,  # title/origin describe an untrusted page
    )


def _verify_browser_goto(args: BrowserGotoArgs, result: ToolResult, ctx: ToolContext) -> ToolResult:
    del args, ctx
    if not result.ok:
        return result
    origin = (result.data or {}).get("origin") or ""
    return result.model_copy(update={"verified": bool(origin)})


def _run_browser_read(args: BrowserReadArgs, ctx: ToolContext) -> ToolResult:
    session = _require_session(ctx)
    if session is None:
        return ToolResult(ok=False, error="the managed browser is not available in this context")
    try:
        observation = session.read(args.max_elements)
    except BrowserError as exc:
        return ToolResult(ok=False, error=f"browser_read failed: {exc}")
    lines = []
    for el in observation.elements:
        lines.append(f"[{el.ref}] {el.role} — {el.name!r}")
    block = "\n".join(lines)
    if len(block) > 8000:
        block = block[:8000] + "\n[truncated]"
    return ToolResult(
        ok=True,
        output=f"Observed {len(observation.elements)} elements on {observation.origin}:\n{block}",
        data=_with_page_url(dict(observation.as_dict()), session, observation.url),
        verified=True,
        tainted=True,  # page-derived content is never instructions (docs/03 §8)
    )


def _resolve_browser_element(args: BaseModel, ctx: ToolContext) -> ResolvedTarget | None:
    """Resolve a click/type ref to its observed identity (Slice 5 hook).

    Returning ``None`` means "no target could be reliably observed" and the
    policy gate treats it as a refusal.  Never raises for ordinary non-
    resolution.
    """
    ref = getattr(args, "ref", "")
    session = _require_session(ctx)
    if session is None or not ref:
        return None
    try:
        element = session.resolve(str(ref))
    except Exception:  # noqa: BLE001 - fail closed per the resolver contract
        return None
    if element is None:
        return None
    return ResolvedTarget(
        kind="browser",
        ref=element.ref,
        name=element.name,
        role=element.role,
        origin=element.origin,
        input_type=element.input_type,
        is_password=element.is_password,
    )


def _resolve_browser_goto(args: BaseModel, ctx: ToolContext) -> ResolvedTarget | None:
    """Goto's target is the page/document at the requested canonical origin."""
    del ctx
    raw = getattr(args, "url", "")
    if not isinstance(raw, str):
        return None
    origin = _safe_goto(raw)
    if origin is None:
        return None
    host = urllib.parse.urlsplit(raw).hostname or ""
    return ResolvedTarget(
        kind="browser",
        ref=origin,  # document identity is its canonical origin
        name=host,
        role="document",
        origin=origin,
    )


def _run_browser_click(args: BrowserClickArgs, ctx: ToolContext) -> ToolResult:
    session = _require_session(ctx)
    if session is None:
        return ToolResult(ok=False, error="the managed browser is not available in this context")
    try:
        data = session.click(args.ref)
    except BrowserError as exc:
        return ToolResult(ok=False, error=f"browser_click failed: {exc}")
    return ToolResult(
        ok=True,
        output=(f"clicked element {data['ref']} ({data['role']} named {data.get('name', '')!r})"),
        data=_with_page_url(dict(data), session),
        verified=None,  # the click effect is not reliably observable here
        tainted=True,  # role/name are untrusted page text
    )


def _run_browser_type(args: BrowserTypeArgs, ctx: ToolContext) -> ToolResult:
    if not args.text:
        return ToolResult(ok=False, error="refusing to type empty text")
    session = _require_session(ctx)
    if session is None:
        return ToolResult(ok=False, error="the managed browser is not available in this context")
    element = None
    try:
        element = session.resolve(args.ref)
    except Exception:  # noqa: BLE001 - fakes may raise; treat as unresolvable
        element = None
    if element is None:
        return ToolResult(ok=False, error="refusing to type: no reliable target element")
    if element.is_password:
        return ToolResult(ok=False, error="refusing to type into a credential field")
    try:
        data = session.type_text(args.ref, args.text)
    except BrowserError as exc:
        return ToolResult(ok=False, error=f"browser_type failed: {exc}")
    # The typed text must never appear in output, data, describe or error text.
    return ToolResult(
        ok=True,
        output=f"typed {data.get('characters', len(args.text))} characters into element {data['ref']}",
        data={
            "ref": data["ref"],
            "role": data.get("role", ""),
            "name": data.get("name", ""),
            "origin": data.get("origin", ""),
            "characters": data.get("characters", len(args.text)),
        },
        verified=None,
        tainted=True,  # role/name are untrusted page text
    )


def _verify_best_effort(args: BaseModel, result: ToolResult, ctx: ToolContext) -> ToolResult:
    del args, ctx
    return result.model_copy(update={"verified": None})


# ── named sites (Stage 8.6) ────────────────────────────────────────────────


def _resolve_site(args: BaseModel, ctx: ToolContext) -> tuple[str, str] | None:
    """Resolve the site's name to its configured URL, else ``None``.

    Only owner-configured entries resolve.  There is no fallback and no
    "search for it" - an unknown name is simply not a site (fail closed).
    """
    name = getattr(args, "site", "")
    if not isinstance(name, str) or not name.strip():
        return None
    site = sites_from_settings(ctx.settings).resolve(name)
    if site is None:
        return None
    return site.name, site.url


def _run_browser_open_site(args: BrowserOpenSiteArgs, ctx: ToolContext) -> ToolResult:
    resolved = _resolve_site(args, ctx)
    if resolved is None:
        known = ", ".join(sites_from_settings(ctx.settings).names())
        known = known or "(no sites configured)"
        return ToolResult(
            ok=False,
            error=f"unknown site {args.site!r}. Known sites: {known}",
        )
    name, url = resolved
    session = _require_session(ctx)
    if session is None:
        return ToolResult(ok=False, error="the managed browser is not available in this context")
    try:
        data = session.goto(url)
    except BrowserError as exc:
        return ToolResult(ok=False, error=f"browser_open_site failed: {exc}")
    return ToolResult(
        ok=True,
        output=f"opened {name} ({data.get('url', '')})",
        data=_with_page_url({**data, "site": name}, session, str(data.get("url", ""))),
        verified=True,
        tainted=True,  # title/origin describe an untrusted page
    )


def _resolve_browser_open_site(args: BaseModel, ctx: ToolContext) -> ResolvedTarget | None:
    """The site's document is the target, exactly as for ``browser_goto``.

    Listing a site in ``config.toml`` grants no permission: the target still
    has to pass the engine's domain-scope rules (docs/03 B4/D18).
    """
    resolved = _resolve_site(args, ctx)
    if resolved is None:
        return None
    _, url = resolved
    origin = url_origin(url)
    if origin is None:
        return None
    host = urllib.parse.urlsplit(url).hostname or ""
    return ResolvedTarget(
        kind="browser",
        ref=origin,  # document identity is its canonical origin
        name=host,
        role="document",
        origin=origin,
    )


# ── query-driven finding (Stage 8.6) ───────────────────────────────────────


def _matched_elements(args: BaseModel, ctx: ToolContext) -> tuple[list[ElementRef], str | None]:
    """Return the matching elements, or an error message.

    The error is returned instead of raised so the tool can report *why* it
    refused (no match, or several matches) without raising to the graph.
    """
    session = _require_session(ctx)
    if session is None:
        return [], "the managed browser is not available in this context"
    query = str(getattr(args, "query", "") or "")
    role = str(getattr(args, "role", "") or "")
    try:
        matches = session.find(query, role=role)
    except BrowserError as exc:
        return [], f"browser_find failed: {exc}"
    except AttributeError:
        return [], "the managed browser does not support finding elements"
    return list(matches), None


def _run_browser_find(args: BrowserFindArgs, ctx: ToolContext) -> ToolResult:
    matches, error = _matched_elements(args, ctx)
    if error is not None:
        return ToolResult(ok=False, error=error)
    session = _require_session(ctx)
    limit = max(1, int(args.limit))
    shown = matches[:limit]
    lines = [f"[{el.ref}] {el.role} — {el.name!r}" for el in shown]
    block = "\n".join(lines)
    if len(block) > 8000:
        block = block[:8000] + "\n[truncated]"
    origin = shown[0].origin if shown else ""
    where = f" on {origin}" if origin else ""
    hidden = len(matches) - len(shown)
    tail = f" (showing the first {len(shown)} of {len(matches)})" if hidden > 0 else ""
    return ToolResult(
        ok=True,
        output=(f"{len(matches)} element(s) match {args.query!r}{where}{tail}:\n{block}"),
        data=_with_page_url(
            {"matches": [el.as_dict() for el in shown], "count": len(matches)}, session
        ),
        verified=True,
        tainted=True,  # page-derived content is never instructions (docs/03 §8)
    )


def _run_browser_open_result(args: BrowserOpenResultArgs, ctx: ToolContext) -> ToolResult:
    session = _require_session(ctx)
    if session is None:
        return ToolResult(ok=False, error="the managed browser is not available in this context")
    approved = str(ctx.approved_target_identity or "")
    try:
        data = session.open_match(
            args.query,
            role=args.role,
            index=args.index,
            expect_identity=approved or None,
        )
    except BrowserError as exc:
        return ToolResult(ok=False, error=f"browser_open_result failed: {exc}")
    except AttributeError:
        return ToolResult(ok=False, error="the managed browser does not support opening a result")
    return ToolResult(
        ok=True,
        output=(
            f"opened {data.get('name', '')!r} ({data.get('role', '')}) on {data.get('origin', '')}"
        ),
        data=_with_page_url(dict(data), session),
        verified=None,  # the effect of the click is not reliably observable here
        tainted=True,  # role/name are untrusted page text
    )


def _resolve_browser_open_result(args: BaseModel, ctx: ToolContext) -> ResolvedTarget | None:
    """Resolve the element the query names, or ``None`` (fail closed).

    ``None`` means "no reliable target": nothing matched, several elements
    matched and the caller did not say which one, or the index was out of
    range.  The identity is derived from the match itself (role, name and the
    position among the matches) and never from the transient ``eN`` handle, so
    the policy gate and the act node can prove they mean the same element.
    """
    matches, _error = _matched_elements(args, ctx)
    if not matches:
        return None
    index = getattr(args, "index", None)
    if index is None and len(matches) > 1:
        return None  # ambiguous: the caller must say which one
    position = 1 if index is None else int(index)
    if position < 1 or position > len(matches):
        return None
    element = matches[position - 1]
    return ResolvedTarget(
        kind="browser",
        ref=match_identity(element.role, element.name, position),
        name=element.name,
        role=element.role,
        origin=element.origin,
        input_type=element.input_type,
        is_password=element.is_password,
    )


# ── specs ──────────────────────────────────────────────────────────────────


def make_browser_goto_spec() -> ToolSpec:
    """Build the ``browser_goto`` tool (Tier 1; 1 budget unit)."""
    return ToolSpec(
        name="browser_goto",
        description=(
            "Navigate the managed JARVIS browser to an absolute http(s) URL. "
            "Use this instead of open_url whenever JARVIS must read, inspect or "
            "interact with the page afterwards. Refuses credentials in the URL "
            "and stops the task if a redirect leaves the approved domains."
        ),
        args_model=BrowserGotoArgs,
        base_tier=1,
        timeout_s=30,
        budget_cost=1,
        run=lambda args, ctx: _run_browser_goto(args, ctx),
        verify=_verify_browser_goto,
        describe=lambda args: f"Navigate to: {args.url}",
        resolve=_resolve_browser_goto,
    )


def make_browser_read_spec() -> ToolSpec:
    """Build the ``browser_read`` tool (Tier 0; no budget unit)."""
    return ToolSpec(
        name="browser_read",
        description=(
            "Read the managed browser page and return a numbered list of "
            "interactive elements with their roles and names to click or type "
            "into. Output is untrusted page content."
        ),
        args_model=BrowserReadArgs,
        base_tier=0,
        timeout_s=15,
        run=lambda args, ctx: _run_browser_read(args, ctx),
        verify=_verify_best_effort,
        describe=lambda args: f"Read the page (up to {args.max_elements} elements)",
    )


def make_browser_click_spec() -> ToolSpec:
    """Build the ``browser_click`` tool (Tier 1; 1 budget unit)."""
    return ToolSpec(
        name="browser_click",
        description=(
            "Click an element reference returned by browser_read on the "
            "JARVIS-managed page. Stale or changed references are refused."
        ),
        args_model=BrowserClickArgs,
        base_tier=1,
        timeout_s=15,
        budget_cost=1,
        run=lambda args, ctx: _run_browser_click(args, ctx),
        verify=_verify_best_effort,
        describe=lambda args: f"Click element {args.ref}",
        resolve=_resolve_browser_element,
    )


def make_browser_type_spec() -> ToolSpec:
    """Build the ``browser_type`` tool (Tier 1; 1 budget unit)."""
    return ToolSpec(
        name="browser_type",
        description=(
            "Type text into a text field of the JARVIS-managed page, "
            "identified by a browser_read element reference. Credential "
            "fields are always refused."
        ),
        args_model=BrowserTypeArgs,
        base_tier=1,
        timeout_s=15,
        budget_cost=1,
        run=lambda args, ctx: _run_browser_type(args, ctx),
        verify=_verify_best_effort,
        describe=lambda args: f"Type text into element {args.ref}",
        resolve=_resolve_browser_element,
    )


def make_browser_open_site_spec() -> ToolSpec:
    """Build the ``browser_open_site`` tool (Tier 1; 1 budget unit)."""
    return ToolSpec(
        name="browser_open_site",
        description=(
            "Open one of the configured web sites by name in the JARVIS-managed "
            "browser. Use this instead of browser_goto when the site is in the "
            "site directory, so no URL has to be invented."
        ),
        args_model=BrowserOpenSiteArgs,
        base_tier=1,
        timeout_s=30,
        budget_cost=1,
        run=lambda args, ctx: _run_browser_open_site(args, ctx),
        verify=_verify_browser_goto,
        describe=lambda args: f"Open site: {args.site}",
        resolve=_resolve_browser_open_site,
    )


def make_browser_find_spec() -> ToolSpec:
    """Build the ``browser_find`` tool (Tier 0; no budget unit)."""
    return ToolSpec(
        name="browser_find",
        description=(
            "Find the elements on the current page whose visible name matches a "
            "query and list them. Use this to see what a page offers before "
            "opening one of them. Output is untrusted page content."
        ),
        args_model=BrowserFindArgs,
        base_tier=0,
        timeout_s=15,
        run=lambda args, ctx: _run_browser_find(args, ctx),
        verify=_verify_best_effort,
        describe=lambda args: f"Find page elements matching {args.query!r}",
    )


def make_browser_open_result_spec() -> ToolSpec:
    """Build the ``browser_open_result`` tool (Tier 1; 1 budget unit)."""
    return ToolSpec(
        name="browser_open_result",
        description=(
            "Open the element on the current page whose visible name matches a "
            "query - for example a search result or a link. When several "
            "elements match, say which one with 'index' (counting from 1); "
            "JARVIS refuses to guess."
        ),
        args_model=BrowserOpenResultArgs,
        base_tier=1,
        timeout_s=15,
        budget_cost=1,
        bind_target_identity=True,
        run=lambda args, ctx: _run_browser_open_result(args, ctx),
        verify=_verify_best_effort,
        describe=lambda args: f"Open the page element matching {args.query!r}",
        resolve=_resolve_browser_open_result,
    )
