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

from pydantic import BaseModel, ConfigDict, Field

from jarvis.browser.context import BrowserError, BrowserSession
from jarvis.policy.rules import url_origin
from jarvis.policy.targets import ResolvedTarget
from jarvis.tools.base import ToolContext, ToolResult, ToolSpec
from jarvis.tools.web import is_allowed_http_url

__all__ = [
    "BrowserClickArgs",
    "BrowserGotoArgs",
    "BrowserReadArgs",
    "BrowserTypeArgs",
    "make_browser_click_spec",
    "make_browser_goto_spec",
    "make_browser_read_spec",
    "make_browser_type_spec",
]

_URL_MAX = 2083
_TEXT_MAX = 4096
_READ_MAX = 80
_REF_MAX = 64


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
        data=data,
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
        data=observation.as_dict(),
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
        data=data,
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
