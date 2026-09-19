"""Web tools: open_url and google_search (Tier 0).

Opening a URL is a navigate/read action with no lasting change, hence Tier 0.
Only ``http``/``https`` URLs are allowed; everything else (``file:``,
``javascript:``, ``data:``, ``ms-*:``, custom protocols, credentials in the
URL) is refused.  No page content is fetched or scraped in this phase - that
is ``web_answer`` (Phase 4) and the source of taint-flagged untrusted text.
"""

from __future__ import annotations

import urllib.parse
import webbrowser
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from jarvis.tools.base import ToolContext, ToolResult, ToolSpec

_URL_MAX = 2083
_QUERY_MAX = 300

__all__ = [
    "GoogleSearchArgs",
    "OpenUrlArgs",
    "is_allowed_http_url",
    "make_google_search_spec",
    "make_open_url_spec",
    "open_url_in_browser",
]


class OpenUrlArgs(BaseModel):
    """Arguments for ``open_url``."""

    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=1, max_length=_URL_MAX, description="Absolute http(s) URL.")


class GoogleSearchArgs(BaseModel):
    """Arguments for ``google_search``."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=_QUERY_MAX, description="Search text.")


def is_allowed_http_url(raw: str) -> bool:
    """Return ``True`` for a safe, openable http/https URL."""
    try:
        parts = urllib.parse.urlsplit(raw)
    except ValueError:
        return False
    if parts.scheme not in ("http", "https"):
        return False
    if "@" in parts.netloc:  # credentials in the URL are always refused
        return False
    return bool(parts.netloc)


def open_url_in_browser(url: str, ctx: ToolContext) -> ToolResult:
    """Open ``url`` in the default browser with scheme enforcement."""
    if not is_allowed_http_url(url):
        return ToolResult(ok=False, error=f"refusing to open non-http(s) URL: {url}")
    if ctx.dry_run:
        return ToolResult(ok=True, output=f"[dry-run] would open {url}", data={"url": url})
    opened = webbrowser.open(url, new=2)
    if not opened:
        return ToolResult(ok=False, error="no default browser available", data={"url": url})
    return ToolResult(ok=True, output=f"opened {url}", data={"url": url})


def _verify_url(args: Any, result: ToolResult, ctx: ToolContext) -> ToolResult:
    """Best-effort: browsers cannot be introspected deterministically."""
    del args, ctx
    return result.model_copy(update={"verified": None})


def _run_open_url(args: OpenUrlArgs, ctx: ToolContext) -> ToolResult:
    return open_url_in_browser(args.url, ctx)


def _describe_open_url(args: OpenUrlArgs) -> str:
    return f"Open URL: {args.url}"


def _run_google_search(args: GoogleSearchArgs, ctx: ToolContext) -> ToolResult:
    url = "https://www.google.com/search?q=" + urllib.parse.quote(args.query)
    return open_url_in_browser(url, ctx)


def _describe_google_search(args: GoogleSearchArgs) -> str:
    return f"Google search: {args.query!r}"


def make_open_url_spec() -> ToolSpec:
    """Build the ``open_url`` tool (Tier 0)."""
    return ToolSpec(
        name="open_url",
        description="Open a web address in the default browser (http/https only).",
        args_model=OpenUrlArgs,
        base_tier=0,
        timeout_s=15,
        run=lambda args, ctx: _run_open_url(args, ctx),
        verify=_verify_url,
        describe=_describe_open_url,
    )


def make_google_search_spec() -> ToolSpec:
    """Build the ``google_search`` tool (Tier 0)."""
    return ToolSpec(
        name="google_search",
        description="Open Google search results for a query in the default browser.",
        args_model=GoogleSearchArgs,
        base_tier=0,
        timeout_s=15,
        run=lambda args, ctx: _run_google_search(args, ctx),
        verify=_verify_url,
        describe=_describe_google_search,
    )
