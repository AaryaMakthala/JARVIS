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
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from jarvis.browser.context import BrowserError, BrowserSession, ElementRef
from jarvis.browser.find import match_identity
from jarvis.browser.results import (
    BrowserResultSet,
    ResultItem,
    Selection,
    parse_ordinal,
)
from jarvis.browser.sites import sites_from_settings
from jarvis.policy.rules import url_origin
from jarvis.policy.targets import ResolvedTarget
from jarvis.tools.base import ToolContext, ToolResult, ToolSpec
from jarvis.tools.web import is_allowed_http_url

__all__ = [
    "BrowserClickArgs",
    "BrowserFinalClickArgs",
    "BrowserFindArgs",
    "BrowserGotoArgs",
    "BrowserOpenResultArgs",
    "BrowserOpenSiteArgs",
    "BrowserReadArgs",
    "BrowserTypeArgs",
    "make_browser_click_spec",
    "make_browser_final_click_spec",
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

    query: str = Field(default="")
    role: str = ""
    index: int | None = Field(default=None, ge=1)
    position: int | None = Field(default=None, ge=1)


class BrowserSearchArgs(BaseModel):
    site: str = Field(min_length=1, max_length=64)
    query: str = Field(min_length=1, max_length=200)


class BrowserPressKeyArgs(BaseModel):
    key: Literal["Enter", "Tab", "Escape", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "PageUp", "PageDown", "Backspace"]


class BrowserScrollArgs(BaseModel):
    direction: Literal["up", "down"]
    amount: int = Field(default=1, ge=1, le=10)


class BrowserWaitArgs(BaseModel):
    seconds: int = Field(default=1, ge=1, le=10)


class BrowserBackArgs(BaseModel):
    pass


class BrowserForwardArgs(BaseModel):
    pass


class BrowserTabsArgs(BaseModel):
    action: Literal["list", "switch", "close_extra"]
    index: int | None = None


class BrowserScreenshotArgs(BaseModel):
    path: str | None = None



# ── session helpers ────────────────────────────────────────────────────────


def _require_session(ctx: ToolContext) -> BrowserSession | None:
    """Return the managed session, or ``None`` when it is not wired up."""
    session = ctx.browser
    if session is None:
        return None
    return session  # type: ignore[return-value]



def _note_navigation(ctx: ToolContext, session: Any, url: str = ""):
    try:
        page_url = url or _current_page_url(session)
        if not page_url:
            return
        # get conversation id
        cid = getattr(getattr(ctx, "run_id", None), "__str__", lambda: None)()
        if cid is None:
            # try from state
            pass
        # store in referents if possible
        try:
            from jarvis.agent.referents import origin_from_url
            from jarvis.agent import referents as _ref
            origin = origin_from_url(page_url)
            REFERENT_STORE.set_page(str(getattr(ctx, "session_context", None) and getattr(ctx.session_context, "_id", None) or id(ctx)), PageRef(url=page_url, title=None, origin=origin, created_at=_ref.now_ts()))
        except Exception:
            pass
        try:
            if hasattr(ctx, "session_context") and ctx.session_context is not None:
                from jarvis.agent.session_context import SESSION_THREAD
                ctx.session_context.note_page(SESSION_THREAD, page_url)
        except Exception:
            pass
    except Exception:
        pass

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


# ── result sets (Stage 8, P2) ──────────────────────────────────────────────


def _session_scope(session: Any) -> str:
    """The scope key of a browser session's current result set.

    A real session reports a stable id (``PlaywrightBrowser``/
    ``_LazyBrowserSession``); anything else - a fake, an older adapter - falls
    back to the object identity, so two different sessions never share results.
    """
    getter = getattr(session, "session_id", None)
    value = ""
    if callable(getter):
        try:
            value = str(getter() or "")
        except Exception:  # noqa: BLE001 - a session that cannot report it has none
            value = ""
    elif isinstance(getter, str):
        value = getter
    scope = value.strip()
    return scope[:128] if scope else f"object:{id(session)}"


def _items_from_elements(elements: Any) -> list[ResultItem]:
    """Map observed elements to bounded result-set entries (metadata only)."""
    items: list[ResultItem] = []
    for position, element in enumerate(list(elements), start=1):
        items.append(
            ResultItem(
                position=position,
                title=str(getattr(element, "name", "") or ""),
                url=str(getattr(element, "url", "") or ""),
                role=str(getattr(element, "role", "") or ""),
            )
        )
    return items


def _record_results(ctx: ToolContext, session: Any, elements: Any) -> BrowserResultSet | None:
    """Store the listed elements as this session's current result set."""
    store = ctx.browser_results
    if store is None:
        return None
    return store.record(
        _session_scope(session),
        _items_from_elements(elements),
        page_url=_current_page_url(session),
    )


def _invalidate_results(ctx: ToolContext, session: Any) -> None:
    """Drop the current result set (a navigation makes every position stale)."""
    store = ctx.browser_results
    if store is not None:
        store.invalidate(_session_scope(session))


def _positional_selection(args: BaseModel, ctx: ToolContext, session: Any) -> Selection | None:
    """The result-set entry a position/ordinal names, or ``None``.

    ``None`` means the caller did not ask for a position at all; it never means
    "resolved".  An ordinal is only ever read inside the *current* set, so an
    older list can never be used by accident.
    """
    position = getattr(args, "position", None)
    query = str(getattr(args, "query", "") or "")
    if position is None:
        position = parse_ordinal(query)
        if position is None:
            return None
    store = ctx.browser_results
    if store is None:
        return Selection(
            kind="no_result_set",
            message="there is no current result list here; name the result to open instead",
        )
    return store.select(
        _session_scope(session),
        position=int(position),
        page_url=_current_page_url(session),
    )


def _title_selection(args: BaseModel, ctx: ToolContext, session: Any) -> Selection | None:
    """The result-set entry a title query names, or ``None`` (no set/empty)."""
    store = ctx.browser_results
    query = str(getattr(args, "query", "") or "").strip()
    if store is None or not query:
        return None
    return store.select(
        _session_scope(session),
        query=query,
        role=str(getattr(args, "role", "") or ""),
        page_url=_current_page_url(session),
    )


def _item_target(session: Any, item: ResultItem) -> tuple[str, Any] | None:
    """Re-observe one selected entry and return its stable identity.

    The entry must still be on the current page exactly once, and a link must
    still carry the same validated destination.  Anything else is ``None`` -
    the caller refuses instead of opening a different element.
    """
    try:
        matches = session.find(item.title, role=item.role)
    except Exception:  # noqa: BLE001 - fail closed per the resolver contract
        return None
    same = [element for element in matches if getattr(element, "name", "") == item.title]
    if len(same) != 1:
        return None
    element = same[0]
    if item.is_link:
        live_url = str(getattr(element, "url", "") or "")
        if not item.url or not is_allowed_http_url(item.url) or live_url != item.url:
            return None
    return match_identity(str(element.role), str(element.name), 1), element


def _resolved_target(element: ElementRef, identity: str) -> ResolvedTarget:
    """Build the policy target for a re-observed result entry."""
    return ResolvedTarget(
        kind="browser",
        ref=identity,
        name=element.name,
        role=element.role,
        origin=element.origin,
        input_type=element.input_type,
        is_password=element.is_password,
    )


def _open_item(session: Any, item: ResultItem, ctx: ToolContext) -> ToolResult:
    """Open one selected result entry, re-validating it first (fail closed)."""
    target = _item_target(session, item)
    if target is None:
        return ToolResult(
            ok=False,
            error=(
                "browser_open_result failed: that result is no longer on the page "
                "(it may have changed); nothing was opened"
            ),
        )
    identity, element = target
    approved = str(ctx.approved_target_identity or "")
    if approved and approved != identity:
        return ToolResult(
            ok=False,
            error=(
                "browser_open_result failed: the result changed after you confirmed "
                "it; nothing was opened"
            ),
        )
    try:
        data = session.open_match(
            str(element.name),
            role=str(element.role),
            index=None,
            expect_identity=identity,
        )
    except BrowserError as exc:
        return ToolResult(ok=False, error=f"browser_open_result failed: {exc}")
    data = dict(data)
    data["result_position"] = item.position
    data["result_url"] = item.url
    return ToolResult(
        ok=True,
        output=(
            f"opened result {item.position}: {data.get('name', '')!r} "
            f"({data.get('role', '')}) on {data.get('origin', '')}"
        ),
        data=_with_page_url(data, session),
        verified=None,  # the effect of the click is not reliably observable here
        tainted=True,  # title/name are untrusted page text
    )


def _run_browser_goto(args: BrowserGotoArgs, ctx: ToolContext) -> ToolResult:
    if _safe_goto(args.url) is None:
        return ToolResult(
            ok=False,
            error="refusing a URL that is not absolute http(s) without userinfo",
        )
    session = _require_session(ctx)
    if session is None:
        return ToolResult(ok=False, error="the managed browser is not available in this context")
    url = args.url
    try:
        data = session.goto(url)
    except BrowserError as exc:
        return ToolResult(ok=False, error=f"browser_goto failed: {exc}")
    _invalidate_results(ctx, session)  # every position from the old page is stale
    try:
        _note_navigation(ctx, session, str(data.get("url", url)))
    except Exception:
        pass
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
    _invalidate_results(ctx, session)  # a click may navigate; positions cannot be trusted
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
        data=_with_page_url(
            {
                "ref": data["ref"],
                "role": data.get("role", ""),
                "name": data.get("name", ""),
                "origin": data.get("origin", ""),
                "characters": data.get("characters", len(args.text)),
            },
            session,
        ),
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
        return ToolResult(ok=False, error=f"browser_goto failed: {exc}")
    _invalidate_results(ctx, session)  # every position from the old page is stale
    try:
        _note_navigation(ctx, session, str(data.get("url", url)))
    except Exception:
        pass
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
    # The listed order is the current result list (Stage 8, P2): a later
    # "open the first link" resolves inside exactly this list, on this page.
    result_set = _record_results(ctx, session, shown)
    data: dict[str, Any] = {"matches": [el.as_dict() for el in shown], "count": len(matches)}
    hint = ""
    if result_set is not None:
        # Only the generation id and size enter the checkpoint; the entries
        # themselves are page content and stay in memory.
        data["result_set"] = {"id": result_set.id, "count": len(result_set.items)}
        hint = (
            f"\nThis is the current result list (1-{len(result_set.items)}); "
            "browser_open_result can open one by position (1 = the first)."
        )
    return ToolResult(
        ok=True,
        output=(f"{len(matches)} element(s) match {args.query!r}{where}{tail}:\n{block}{hint}"),
        data=_with_page_url(data, session),
        verified=True,
        tainted=True,  # page-derived content is never instructions (docs/03 §8)
    )


def _run_browser_open_result(args: BrowserOpenResultArgs, ctx: ToolContext) -> ToolResult:
    session = _require_session(ctx)
    if session is None:
        return ToolResult(ok=False, error="the managed browser is not available in this context")
    # 1. A position (or an ordinal query such as "first") selects one entry of
    #    the *current* result list and nothing else.  A missing, expired,
    #    superseded or navigated-away list is a refusal - never a fallback.
    positional = _positional_selection(args, ctx, session)
    if positional is not None:
        if not positional.ok or positional.item is None:
            return ToolResult(ok=False, error=f"browser_open_result failed: {positional.message}")
        result = _open_item(session, positional.item, ctx)
        if result.ok:
            _invalidate_results(ctx, session)
        return result
    if not str(args.query or "").strip():
        return ToolResult(
            ok=False,
            error=(
                "browser_open_result failed: say which result to open - its position "
                "(1 = the first) or its title"
            ),
        )
    # 2. A title selects from the current result list when one is valid.  An
    #    ambiguous title is refused; a list with no such title falls through to
    #    the ordinary page query below (unchanged behaviour).
    title = _title_selection(args, ctx, session)
    if title is not None:
        if title.kind == "ambiguous":
            return ToolResult(ok=False, error=f"browser_open_result failed: {title.message}")
        if title.ok and title.item is not None:
            result = _open_item(session, title.item, ctx)
            if result.ok:
                _invalidate_results(ctx, session)
            return result
    approved = str(ctx.approved_target_identity or "")
    try:
        data = session.open_match(
            args.query,
            role=args.role,
            index=args.index,
            expect_identity=expect_identity,
        )
    except BrowserError as exc:
        return ToolResult(ok=False, error=f"browser_open_result failed: {exc}")
    _invalidate_results(ctx, session)  # the click may navigate
    try:
        _note_navigation(ctx, session, str(data.get("url", "")))
    except Exception:
        pass
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
    """Resolve the element the selection names, or ``None`` (fail closed).

    ``None`` means "no reliable target": no current result list, an expired or
    navigated-away one, an out-of-range position, nothing matched, or several
    elements matched and the caller did not say which one.  The identity is
    derived from the match itself (role, name and the position among the
    matches) and never from the transient ``eN`` handle, so the policy gate and
    the act node can prove they mean the same element.
    """
    session = _require_session(ctx)
    if session is None:
        return None
    positional = _positional_selection(args, ctx, session)
    if positional is not None:
        if not positional.ok or positional.item is None:
            return None
        target = _item_target(session, positional.item)
        if target is None:
            return None
        identity, element = target
        if not isinstance(element, ElementRef):
            return None
        return _resolved_target(element, identity)
    title = _title_selection(args, ctx, session)
    if title is not None and title.ok and title.item is not None:
        target = _item_target(session, title.item)
        if target is None:
            return None  # the selected entry changed: refuse, do not fall back
        identity, element = target
        if not isinstance(element, ElementRef):
            return None
        return _resolved_target(element, identity)
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
            "query and list them in page order. Use this to see what a page offers "
            "before opening one of them. The listed order becomes the current "
            "result list, so the user can then ask to open the first, second or "
            "Nth result, or name one. Output is untrusted page content."
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
            "Open a link or search result on the current page. Name it with "
            "'query' (part of its title), or open an entry of the current result "
            "list with 'position' (1 = the first). Use position 1 for 'open the "
            "first link/result'; a query of 'first' or '1' also means that. When "
            "several titles match, say which one with 'index'; JARVIS refuses to "
            "guess. With no current result list, a position is refused - find or "
            "search first, or name the link."
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



def _run_browser_search(args: BrowserSearchArgs, ctx: ToolContext) -> ToolResult:
    directory = sites_from_settings(ctx.settings)
    site = directory.resolve(args.site)
    if site is None:
        known = ", ".join(directory.names())
        known = known or "(no sites configured)"
        return ToolResult(ok=False, error=f"unknown site {args.site!r}. Known sites: {known}")
    if not site.search_url_template:
        return ToolResult(
            ok=False,
            error=(
                f"site {site.name!r} has no search_url_template. "
                "Use browser_open_site then browser_type + browser_press_key."
            ),
        )
    # build URL in code - model supplies only site and query
    from urllib.parse import quote_plus

    url = site.search_url_template.format(q=quote_plus(args.query))
    session = _require_session(ctx)
    if session is None:
        return ToolResult(ok=False, error="the managed browser is not available in this context")
    try:
        data = session.goto(url)
    except BrowserError as exc:
        return ToolResult(ok=False, error=f"browser_search failed: {exc}")
    _invalidate_results(ctx, session)
    try:
        _note_navigation(ctx, session, str(data.get("url", url)))
    except Exception:
        pass
    # Try to find results on the page like browser_find does
    shown: list[ElementRef] = []
    try:
        if hasattr(session, "find"):
            shown = session.find(args.query, limit=50)
    except Exception:
        shown = []
    result_set = None
    if shown:
        try:
            result_set = _record_results(ctx, session, shown)
        except Exception:
            result_set = None
    items = []
    for el in shown[:5]:
        items.append({"position": getattr(el, "ref", "") and shown.index(el) + 1 if False else None, "title": getattr(el, "name", "")})
    # compute positions
    items_out = []
    for i, el in enumerate(shown[:5], start=1):
        items_out.append({"position": i, "title": getattr(el, "name", ""), "role": getattr(el, "role", "")})
    out = {
        "site": site.name,
        "query": args.query,
        "url": data.get("url", url),
        "results": items_out,
        "count": len(items_out),
    }
    if result_set is not None:
        out["result_set_id"] = getattr(result_set, "id", "")
    return ToolResult(
        ok=True,
        output=f"Searched {site.name} for {args.query!r}; {len(items_out)} top results",
        data=_with_page_url(out, session, str(data.get("url", url))),
        verified=True,
        tainted=True,
    )


def _run_browser_press_key(args: BrowserPressKeyArgs, ctx: ToolContext) -> ToolResult:
    session = _require_session(ctx)
    if session is None:
        return ToolResult(ok=False, error="the managed browser is not available in this context")
    try:
        data = session.press_key(args.key)
    except BrowserError as exc:
        return ToolResult(ok=False, error=f"browser_press_key failed: {exc}")
    try:
        _note_navigation(ctx, session, _current_page_url(session))
    except Exception:
        pass
    return ToolResult(ok=True, output=f"pressed {args.key}", data=_with_page_url(dict(data), session), verified=None, tainted=True)


def _run_browser_scroll(args: BrowserScrollArgs, ctx: ToolContext) -> ToolResult:
    session = _require_session(ctx)
    if session is None:
        return ToolResult(ok=False, error="the managed browser is not available in this context")
    try:
        data = session.scroll(args.direction, args.amount)
    except BrowserError as exc:
        return ToolResult(ok=False, error=f"browser_scroll failed: {exc}")
    return ToolResult(ok=True, output=f"scrolled {args.direction} {args.amount}", data=_with_page_url(dict(data), session), verified=None, tainted=True)


def _run_browser_wait(args: BrowserWaitArgs, ctx: ToolContext) -> ToolResult:
    session = _require_session(ctx)
    if session is None:
        return ToolResult(ok=False, error="the managed browser is not available in this context")
    try:
        data = session.wait(float(args.seconds))
    except BrowserError as exc:
        return ToolResult(ok=False, error=f"browser_wait failed: {exc}")
    return ToolResult(ok=True, output=f"waited {args.seconds}s", data=_with_page_url(dict(data), session), verified=None, tainted=True)


def _run_browser_back(args: BrowserBackArgs, ctx: ToolContext) -> ToolResult:
    session = _require_session(ctx)
    if session is None:
        return ToolResult(ok=False, error="the managed browser is not available in this context")
    try:
        data = session.back()
    except BrowserError as exc:
        return ToolResult(ok=False, error=f"browser_back failed: {exc}")
    _invalidate_results(ctx, session)
    try:
        _note_navigation(ctx, session, str(data.get("url", "")))
    except Exception:
        pass
    return ToolResult(ok=True, output="went back", data=_with_page_url(dict(data), session), verified=True, tainted=True)


def _run_browser_forward(args: BrowserForwardArgs, ctx: ToolContext) -> ToolResult:
    session = _require_session(ctx)
    if session is None:
        return ToolResult(ok=False, error="the managed browser is not available in this context")
    try:
        data = session.forward()
    except BrowserError as exc:
        return ToolResult(ok=False, error=f"browser_forward failed: {exc}")
    _invalidate_results(ctx, session)
    try:
        _note_navigation(ctx, session, str(data.get("url", "")))
    except Exception:
        pass
    return ToolResult(ok=True, output="went forward", data=_with_page_url(dict(data), session), verified=True, tainted=True)


def _run_browser_tabs(args: BrowserTabsArgs, ctx: ToolContext) -> ToolResult:
    session = _require_session(ctx)
    if session is None:
        return ToolResult(ok=False, error="the managed browser is not available in this context")
    try:
        if args.action == "list":
            data = session.list_tabs()
            return ToolResult(ok=True, output=f"{len(data)} tabs", data={"tabs": data}, verified=None, tainted=True)
        elif args.action == "switch":
            if args.index is None:
                return ToolResult(ok=False, error="index required for switch")
            data = session.switch_tab(int(args.index))
            _invalidate_results(ctx, session)
            try:
                _note_navigation(ctx, session, str(data.get("url", "")))
            except Exception:
                pass
            return ToolResult(ok=True, output=f"switched to tab {args.index}", data=_with_page_url(dict(data), session), verified=True, tainted=True)
        elif args.action == "close_extra":
            data = session.close_extra_tab()
            _invalidate_results(ctx, session)
            return ToolResult(ok=True, output=f"closed {data.get('closed', 0)} extra tabs", data=data, verified=None, tainted=True)
        else:
            return ToolResult(ok=False, error="invalid action")
    except BrowserError as exc:
        return ToolResult(ok=False, error=f"browser_tabs failed: {exc}")


def _run_browser_screenshot(args: BrowserScreenshotArgs, ctx: ToolContext) -> ToolResult:
    session = _require_session(ctx)
    if session is None:
        return ToolResult(ok=False, error="the managed browser is not available in this context")
    try:
        data = session.screenshot(args.path)
    except BrowserError as exc:
        return ToolResult(ok=False, error=f"browser_screenshot failed: {exc}")
    path = data.get("path", "")
    return ToolResult(
        ok=True,
        output=f"screenshot saved to {path}",
        data={"path": path, "artifact": {"path": path, "type": "screenshot"}},
        verified=None,
        tainted=False,
    )




def make_browser_search_spec() -> ToolSpec:
    return ToolSpec(
        name="browser_search",
        description="Search a named site via its search template and list top results.",
        args_model=BrowserSearchArgs,
        base_tier=1,
        timeout_s=30,
        budget_cost=1,
        run=lambda args, ctx: _run_browser_search(args, ctx),
        verify=_verify_best_effort,
        describe=lambda args: f"Search {args.site} for {args.query!r}",
    )


def make_browser_press_key_spec() -> ToolSpec:
    return ToolSpec(
        name="browser_press_key",
        description="Press an allowed key in the managed browser.",
        args_model=BrowserPressKeyArgs,
        base_tier=1,
        timeout_s=10,
        run=lambda args, ctx: _run_browser_press_key(args, ctx),
        verify=_verify_best_effort,
        describe=lambda args: f"Press {args.key}",
    )


def make_browser_scroll_spec() -> ToolSpec:
    return ToolSpec(
        name="browser_scroll",
        description="Scroll the page up or down.",
        args_model=BrowserScrollArgs,
        base_tier=0,
        timeout_s=10,
        run=lambda args, ctx: _run_browser_scroll(args, ctx),
        verify=_verify_best_effort,
        describe=lambda args: f"Scroll {args.direction}",
    )


def make_browser_wait_spec() -> ToolSpec:
    return ToolSpec(
        name="browser_wait",
        description="Wait briefly or for load state.",
        args_model=BrowserWaitArgs,
        base_tier=0,
        timeout_s=15,
        run=lambda args, ctx: _run_browser_wait(args, ctx),
        verify=_verify_best_effort,
        describe=lambda args: f"Wait {args.seconds}s",
    )


def make_browser_back_spec() -> ToolSpec:
    return ToolSpec(
        name="browser_back",
        description="Go back in browser history.",
        args_model=BrowserBackArgs,
        base_tier=1,
        timeout_s=15,
        run=lambda args, ctx: _run_browser_back(args, ctx),
        verify=_verify_best_effort,
        describe=lambda args: "Go back",
    )


def make_browser_forward_spec() -> ToolSpec:
    return ToolSpec(
        name="browser_forward",
        description="Go forward in browser history.",
        args_model=BrowserForwardArgs,
        base_tier=1,
        timeout_s=15,
        run=lambda args, ctx: _run_browser_forward(args, ctx),
        verify=_verify_best_effort,
        describe=lambda args: "Go forward",
    )


def make_browser_tabs_spec() -> ToolSpec:
    return ToolSpec(
        name="browser_tabs",
        description="List, switch, or close extra tabs.",
        args_model=BrowserTabsArgs,
        base_tier=1,
        timeout_s=10,
        run=lambda args, ctx: _run_browser_tabs(args, ctx),
        verify=_verify_best_effort,
        describe=lambda args: f"Tabs: {args.action}",
    )


def make_browser_screenshot_spec() -> ToolSpec:
    return ToolSpec(
        name="browser_screenshot",
        description="Take a screenshot of the current tab.",
        args_model=BrowserScreenshotArgs,
        base_tier=1,
        timeout_s=15,
        run=lambda args, ctx: _run_browser_screenshot(args, ctx),
        verify=_verify_best_effort,
        describe=lambda args: "Take screenshot",
    )


class BrowserFinalClickArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ref: str = Field(min_length=1, max_length=_REF_MAX, description="Element reference.")
    action: Literal["send", "submit", "post", "publish", "subscribe", "book"] = Field(
        description="Declared intent checked against page."
    )
    expect: str = Field(default="", max_length=120, description="One line describing the effect.")


def _resolve_browser_element(session: Any, ref: str) -> Any | None:
    """Read-only re-resolution of one transient ref (a failed read is ``None``).

    The single place the guarded final-click path asks the session for an
    element, so the gate's observation and the tool's immediate pre-click
    re-observation can never diverge in *how* they look the element up.  It
    resolves, never clicks/types/navigates.
    """
    if session is None or not ref:
        return None
    try:
        return session.resolve(str(ref))
    except Exception:  # noqa: BLE001 - a failed observation is a refusal, not a guess
        return None


def _final_click_limits(ctx: ToolContext) -> tuple[int, float | None, str, float, bool]:
    """``(max_total, spending_cap, currency, binding_ttl_s, voice_opt_in)``."""
    from jarvis.browser.final_click import final_click_limits

    return final_click_limits(getattr(ctx, "settings", None))


def _resolve_browser_final_click(args: BaseModel, ctx: ToolContext) -> ResolvedTarget | None:
    """Resolve the real target read-only, so the gate can observe it pre-approval.

    Returns ``None`` (never a fabricated target) when the element cannot be
    resolved, when the declared action is not an allowed one, when the control
    is a password or payment/purchase control, or when the configured spending
    cap cannot be checked against an observed amount and currency.
    """
    from jarvis.browser.final_click import build_observed_readback
    from jarvis.browser.find import match_identity

    ref = str(getattr(args, "ref", "") or "")
    action = str(getattr(args, "action", "") or "")
    session = _require_session(ctx)
    element = _resolve_browser_element(session, ref)
    if element is None:
        return None
    _max_total, spending_cap, currency, _ttl, _voice = _final_click_limits(ctx)
    observation = build_observed_readback(
        element, action, spending_cap=spending_cap, spending_currency=currency
    )
    if not observation.ok:
        return None
    try:
        identity = match_identity(str(element.role), str(element.name), 1)
    except Exception:  # noqa: BLE001 - an unidentifiable target is refused upstream
        return None
    return ResolvedTarget(
        kind="browser",
        # The *stable* identity (role|name|position), exactly as the other
        # composite browser tools bind it - never the transient ``e1..eN``
        # handle, which changes on every observation.
        ref=identity,
        name=element.name,
        role=element.role,
        origin=element.origin,
        input_type=element.input_type,
        is_password=element.is_password,
    )


def _run_browser_final_click(args: BrowserFinalClickArgs, ctx: ToolContext) -> ToolResult:
    from jarvis.browser.final_click import (
        FinalClickJournalRow,
        build_observed_readback,
        digests_equal,
        get_journal,
        is_hex64,
        readback_digest,
        sha256_hex,
        _now_iso,
    )
    import uuid

    session = _require_session(ctx)
    if session is None:
        return ToolResult(ok=False, error="the managed browser is not available in this context")

    max_total, spending_cap, spending_currency, _ttl, _voice = _final_click_limits(ctx)

    # Immediate pre-click re-observation.  The element, its origin, its password
    # status, its payment marker and (when a spending limit is configured) the
    # amount and currency are all re-read from the live page *here*, and every
    # check below runs against this observation - never against the plan, the
    # confirmation text or the model's own words.  The model's ``expect`` is
    # carried only as an explicitly unverified proposal.
    element = _resolve_browser_element(session, args.ref)
    if element is None:
        return ToolResult(ok=False, error="refusing to click: no reliable target element")
    observation = build_observed_readback(
        element,
        args.action,
        proposed={"expect": args.expect},
        spending_cap=spending_cap,
        spending_currency=spending_currency,
    )
    if not observation.ok:
        return ToolResult(ok=False, error=f"refused: {observation.reason}")

    journal = get_journal()
    # If max_total == 0 -> disabled, refuse every call (fail closed)
    if max_total == 0:
        return ToolResult(ok=False, error="browser_final_click is disabled")

    # Check total count
    total = journal.total_count_all()
    if total >= max_total and max_total > 0:
        return ToolResult(ok=False, error="final click limit reached")

    # Get task id if possible
    task_id = ""
    try:
        task_id = getattr(ctx, "task_id", "") or ""
    except Exception:
        task_id = ""

    # Check for indeterminate/dispatched in same task - cannot retry
    if task_id and journal.has_indeterminate_or_dispatched(task_id):
        return ToolResult(
            ok=False,
            error="I could not confirm whether that was sent - check the site before trying again.",
        )

    # Binding identity check
    approved_ident = str(getattr(ctx, "approved_target_identity", "") or "")
    from jarvis.browser.find import match_identity

    try:
        fingerprint = match_identity(str(element.role), str(element.name), 1)
    except Exception:
        fingerprint = f"{element.role}:{element.name}:{element.origin}"
    if approved_ident and approved_ident != fingerprint:
        return ToolResult(ok=False, error="the target changed after you confirmed it; nothing was done")

    # Readback binding.  ``approved_final_click_readback_hash`` is minted by the
    # policy gate from the readback the *user* approved, and is verified by
    # recomputing the digest of this fresh observation with the same canonical
    # helper.  Fail closed: a missing, malformed or stale digest refuses, and
    # neither ``confirmed=True`` nor an arbitrary non-empty string is ever
    # authorization on its own.
    approved_readback_hash = str(getattr(ctx, "approved_final_click_readback_hash", "") or "")
    if not is_hex64(approved_readback_hash):
        return ToolResult(
            ok=False, error="missing approved readback binding for this consequential action"
        )
    if not digests_equal(approved_readback_hash, readback_digest(observation.details or {})):
        return ToolResult(
            ok=False,
            error="the action details changed after you confirmed them; nothing was done",
        )

    # Host/origin binding
    host = getattr(element, "origin", "") or ""

    # Confirmation binding hash
    confirmation_sha = str(getattr(ctx, "approved_typed_confirmation_sha256", "") or "")
    if not confirmation_sha:
        confirmation_sha = sha256_hex("")

    row_id = uuid.uuid4().hex
    now = _now_iso()
    row = FinalClickJournalRow(
        id=row_id,
        task_id=str(task_id),
        host=str(host),
        action=str(args.action),
        target_fingerprint=str(fingerprint),
        confirmation_sha256=str(confirmation_sha),
        state="reserved",
        outcome_code="reserved",
        created_at=now,
        updated_at=now,
    )
    if not journal.reserve(row):
        return ToolResult(ok=False, error="failed to reserve final click")

    try:
        data = session.click(args.ref)
    except Exception as exc:
        journal.update_state(row_id, "indeterminate", "click_exception")
        return ToolResult(ok=False, error=f"browser_final_click failed: {exc}")

    journal.update_state(row_id, "dispatched", "dispatched")
    _invalidate_results(ctx, session)
    journal.update_state(row_id, "indeterminate", "no_evidence")
    return ToolResult(
        ok=True,
        output=f"final click dispatched ({args.action}) on {element.name!r}",
        data=_with_page_url(dict(data), session),
        verified=None,
        tainted=True,
    )


def make_browser_final_click_spec() -> ToolSpec:
    return ToolSpec(
        name="browser_final_click",
        description=(
            "Perform a guarded final click (send/submit/post/publish/subscribe/book). "
            "Requires exact target binding, fresh readback, typed confirmation, unlocked session. "
            "Payments/purchases are blocked. Browsing grants never cover this."
        ),
        args_model=BrowserFinalClickArgs,
        base_tier=2,
        timeout_s=20,
        budget_cost=1,
        bind_target_identity=True,
        run=lambda args, ctx: _run_browser_final_click(args, ctx),
        verify=_verify_best_effort,
        describe=lambda args: f"Final click {args.action} on element {args.ref}",
        resolve=_resolve_browser_final_click,
    )
