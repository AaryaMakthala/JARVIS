"""Web tools: open_url, google_search (Tier 0) and web_answer (Tier 0).

Opening a URL is a navigate/read action with no lasting change, hence Tier 0.
Only ``http``/``https`` URLs are allowed; everything else (``file:``,
``javascript:``, ``data:``, ``ms-*:``, custom protocols, credentials in the
URL) is refused.

``web_answer`` (Phase 4) searches the web via ``ddgs``, feeds the results to
the LLM for a cited answer, and returns the output as tainted (untrusted)
data.  Citation verification is independent of the LLM: we check that at
least one cited URL actually appears in the retrieved results.
"""

from __future__ import annotations

import urllib.parse
import webbrowser
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from jarvis.tools.base import ToolContext, ToolResult, ToolSpec

_URL_MAX = 2083
_QUERY_MAX = 300

logger = __import__("logging").getLogger("jarvis.tools.web")

_MAX_RESULTS = 5
_ANSWER_MAX_CHARS = 4096
_CITATION_RE = __import__("re").compile(r"\[(\d+)\]")

__all__ = [
    "GoogleSearchArgs",
    "OpenUrlArgs",
    "WebAnswerArgs",
    "is_allowed_http_url",
    "make_google_search_spec",
    "make_open_url_spec",
    "make_web_answer_spec",
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


# ── web_answer (Phase 4) ────────────────────────────────────────────────


class WebAnswerArgs(BaseModel):
    """Arguments for ``web_answer``."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        min_length=1, max_length=_QUERY_MAX, description="Question to search for and answer."
    )


class SearchResult(BaseModel):
    """One search result returned by ddgs."""

    title: str
    url: str
    snippet: str


def _search_ddgs(query: str) -> list[SearchResult]:
    """Search via ``ddgs``; raise on failure so the caller can handle it."""
    try:
        from ddgs import DDGS
    except ImportError as exc:
        raise RuntimeError("ddgs package is not installed") from exc
    raw = DDGS().text(query, max_results=_MAX_RESULTS)
    results: list[SearchResult] = []
    for item in raw:
        results.append(
            SearchResult(
                title=str(item.get("title", "")),
                url=str(item.get("href", "")),
                snippet=str(item.get("body", "")),
            )
        )
    return results


def _format_results_for_llm(results: list[SearchResult]) -> str:
    """Render search results as numbered entries for the LLM prompt."""
    parts: list[str] = []
    for i, r in enumerate(results, 1):
        parts.append(f"[{i}] {r.title}\n    URL: {r.url}\n    {r.snippet}")
    return "\n\n".join(parts)


def _verify_citations(answer: str, results: list[SearchResult]) -> tuple[bool, list[str]]:
    """Check that cited URLs in the answer actually exist in the results.

    Returns ``(all_valid, cited_urls)`` where ``all_valid`` is ``True`` when
    every ``[n]`` citation in the answer maps to a real result URL (or when
    there are no citations at all — we treat no-citation as unverified).
    """
    result_urls = {r.url for r in results if r.url}
    cited_nums = _CITATION_RE.findall(answer)
    if not cited_nums:
        return False, []  # no citations → unverified
    cited_urls: list[str] = []
    for num_str in cited_nums:
        idx = int(num_str) - 1
        if idx < 0 or idx >= len(results):
            return False, []  # out-of-range citation → fabricated
        cited_urls.append(results[idx].url)
    if not cited_urls:
        return False, []
    # All cited URLs must be in the results set
    return all(url in result_urls for url in cited_urls), cited_urls


def _run_web_answer(args: WebAnswerArgs, ctx: ToolContext) -> ToolResult:
    """Search the web and return a cited answer (always tainted)."""
    if ctx.llm is None:
        return ToolResult(ok=False, error="No LLM backend configured for web_answer.")
    try:
        results = _search_ddgs(args.question)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ddgs search failed: %s", exc)
        return ToolResult(ok=False, error=f"search failed: {exc}")
    if not results:
        return ToolResult(ok=False, error="no search results found")

    numbered = _format_results_for_llm(results)
    system = (
        "You are a helpful assistant. Answer the user's question using ONLY the "
        "search results below. Cite sources with [n] where n is the result number. "
        'Wrap your answer in <untrusted_data source="web"> tags.\n\n'
        f"Search results:\n{numbered}"
    )
    try:
        answer, _usage = ctx.llm.text(
            system=system,
            user=args.question,
            model_role="fast",
        )
    except Exception as exc:  # noqa: BLE001
        return ToolResult(ok=False, error=f"LLM answer generation failed: {exc}")

    # Truncate to prevent oversized output
    if len(answer) > _ANSWER_MAX_CHARS:
        answer = answer[:_ANSWER_MAX_CHARS] + "\n[truncated]"

    sources = []
    for i, r in enumerate(results, 1):
        sources.append(f"[{i}] {r.title} — {r.url}")
    sources_block = "\n".join(sources)

    return ToolResult(
        ok=True,
        output=f"{answer}\n\nSources:\n{sources_block}",
        data={"results": [r.model_dump() for r in results], "question": args.question},
        tainted=True,  # all web content is untrusted (docs/03 §8)
    )


def _verify_web_answer(args: WebAnswerArgs, result: ToolResult, ctx: ToolContext) -> ToolResult:
    """Post-condition: at least one citation must correspond to a real result."""
    del args, ctx
    if not result.ok or not result.data:
        return result
    raw_results = result.data.get("results", [])
    results = [SearchResult(**r) for r in raw_results if isinstance(r, dict)]
    if not results:
        return result.model_copy(update={"verified": False})
    # Extract just the answer part (before Sources:)
    output = result.output
    sources_idx = output.rfind("\nSources:")
    answer_text = output[:sources_idx] if sources_idx > 0 else output
    all_valid, _cited = _verify_citations(answer_text, results)
    return result.model_copy(update={"verified": all_valid})


def _describe_web_answer(args: WebAnswerArgs) -> str:
    return f"Web answer: {args.question!r}"


def make_web_answer_spec() -> ToolSpec:
    """Build the ``web_answer`` tool (Tier 0)."""
    return ToolSpec(
        name="web_answer",
        description=(
            "Search the web for a question and return a cited answer with sources. "
            "Output is untrusted data."
        ),
        args_model=WebAnswerArgs,
        base_tier=0,
        timeout_s=30,
        run=lambda args, ctx: _run_web_answer(args, ctx),
        verify=_verify_web_answer,
        describe=_describe_web_answer,
    )
