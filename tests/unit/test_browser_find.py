"""Query-driven finding and opening (Stage 8.6, slice 8.6.2).

No Playwright, no Chrome, no network: the matching rules are pure functions
and the tools are driven through a fake session.

These tests pin the properties that make the slice safe:

* matching is deterministic and most-specific-first (exact, then prefix, then
  contains), in document order;
* an ambiguous match is **refused**, never guessed;
* the identity an approval is bound to does not contain the transient ``eN``
  handle, so the policy gate and the act node can prove they mean the same
  element across an ``interrupt()`` boundary;
* finding is read-only (Tier 0, no scope action unit); opening costs one.
"""

from __future__ import annotations

from jarvis.browser.context import BrowserError, ElementRef
from jarvis.browser.find import match_identity, match_indexes, normalise
from jarvis.browser.tools import (
    BrowserFindArgs,
    BrowserOpenResultArgs,
    make_browser_find_spec,
    make_browser_open_result_spec,
)
from jarvis.config import Settings
from jarvis.policy.targets import ResolvedTarget
from jarvis.tools.base import ToolContext, resolve_target

_ORIGIN = "https://example.com"


def _element(ref: str, name: str, *, role: str = "link", origin: str = _ORIGIN) -> ElementRef:
    return ElementRef(ref=ref, role=role, name=name, origin=origin)


class FakeSession:
    """A fake managed browser that answers queries from a fixed page."""

    def __init__(self, elements: list[ElementRef] | None = None) -> None:
        self.elements = (
            elements
            if elements is not None
            else [
                _element("e1", "Python Tutorial - Part 1"),
                _element("e2", "Python Tutorial - Part 2"),
                _element("e3", "Java Basics"),
            ]
        )
        self.calls: list[tuple[object, ...]] = []
        self.open_error: str | None = None

    def find(self, query: str, *, role: str = "", limit: int = 300) -> list[ElementRef]:
        self.calls.append(("find", query, role, limit))
        pairs = [(el.role, el.name) for el in self.elements]
        indexes = match_indexes(pairs, query, role=role)
        return [self.elements[i] for i in indexes][:limit]

    def open_match(
        self,
        query: str,
        *,
        role: str = "",
        index: int | None = None,
        expect_identity: str | None = None,
    ) -> dict[str, object]:
        self.calls.append(("open_match", query, role, index))
        if self.open_error:
            raise BrowserError(self.open_error)
        matches = self.find(query, role=role)
        if not matches:
            raise BrowserError(f"no element on the page matches {query!r}")
        if index is None and len(matches) > 1:
            raise BrowserError(f"{len(matches)} elements match {query!r}")
        position = 1 if index is None else index
        if position > len(matches):
            raise BrowserError(f"index {position} is out of range")
        element = matches[position - 1]
        if expect_identity:
            identity = match_identity(element.role, element.name, position)
            if identity != expect_identity:
                raise BrowserError(
                    "the page element changed after you confirmed it; nothing was opened"
                )
        return {
            "ref": element.ref,
            "role": element.role,
            "name": element.name,
            "origin": element.origin,
            "position": position,
            "matches": len(matches),
        }


def _ctx(session: FakeSession | None = None) -> ToolContext:
    return ToolContext(settings=Settings(), browser=session or FakeSession())


# ── the pure matcher ──────────────────────────────────────────────────────


def test_normalise_folds_case_and_whitespace() -> None:
    assert normalise("  Sign   In ") == "sign in"
    assert normalise("") == ""


def test_an_exact_match_wins_over_a_partial_one() -> None:
    candidates = [("link", "Python Basics"), ("link", "Python Tutorial")]
    assert match_indexes(candidates, "Python Basics") == [0]


def test_a_prefix_match_wins_over_a_contains_match() -> None:
    candidates = [("link", "Something else Python"), ("link", "Python Tutorial")]
    assert match_indexes(candidates, "Python") == [1]


def test_matches_keep_document_order() -> None:
    candidates = [("link", "a python b"), ("link", "c python d")]
    assert match_indexes(candidates, "python") == [0, 1]


def test_the_role_filter_narrows_the_pool() -> None:
    candidates = [("button", "Search"), ("link", "Search")]
    assert match_indexes(candidates, "search", role="button") == [0]
    assert match_indexes(candidates, "search", role="link") == [1]
    assert match_indexes(candidates, "search") == [0, 1]


def test_an_empty_query_matches_nothing() -> None:
    assert match_indexes([("link", "anything")], "") == []
    assert match_indexes([("link", "anything")], "   ") == []


def test_the_identity_does_not_contain_the_transient_handle() -> None:
    identity = match_identity("link", "Python Tutorial", 2)
    assert "e1" not in identity
    assert "2" in identity and "python tutorial" in identity


# ── browser_find ──────────────────────────────────────────────────────────


def test_find_lists_the_matching_elements() -> None:
    session = FakeSession()
    spec = make_browser_find_spec()
    result = spec.execute(BrowserFindArgs(query="Python"), _ctx(session))
    assert result.ok is True
    assert "2 element(s) match" in result.output
    assert result.tainted is True
    assert result.data["count"] == 2


def test_find_is_read_only_and_costs_no_scope_action_unit() -> None:
    spec = make_browser_find_spec()
    assert spec.base_tier == 0
    assert spec.budget_cost == 0


def test_find_reports_a_browser_failure_without_raising() -> None:
    class BrokenSession(FakeSession):
        def find(self, query: str, *, role: str = "", limit: int = 300) -> list[ElementRef]:
            raise BrowserError("the managed browser session is not started")

    spec = make_browser_find_spec()
    result = spec.execute(BrowserFindArgs(query="Python"), _ctx(BrokenSession()))
    assert result.ok is False
    assert "browser_find failed" in (result.error or "")


def test_find_shows_only_the_first_matches_but_reports_the_total() -> None:
    session = FakeSession([_element(f"e{n}", f"Result {n}") for n in range(1, 8)])
    spec = make_browser_find_spec()
    result = spec.execute(BrowserFindArgs(query="Result", limit=3), _ctx(session))
    assert result.ok is True
    assert "7 element(s) match" in result.output
    assert "showing the first 3 of 7" in result.output
    assert result.data["count"] == 7
    assert len(result.data["matches"]) == 3


# ── browser_open_result ───────────────────────────────────────────────────


def test_open_result_clicks_the_matching_element() -> None:
    session = FakeSession()
    spec = make_browser_open_result_spec()
    result = spec.execute(BrowserOpenResultArgs(query="Java Basics"), _ctx(session))
    assert result.ok is True
    assert session.calls[0][0] == "open_match"
    assert result.data["name"] == "Java Basics"
    assert result.tainted is True


def test_open_result_refuses_an_ambiguous_match() -> None:
    session = FakeSession()
    spec = make_browser_open_result_spec()
    result = spec.execute(BrowserOpenResultArgs(query="Python"), _ctx(session))
    assert result.ok is False
    assert "elements match" in (result.error or "")


def test_open_result_opens_the_chosen_match() -> None:
    session = FakeSession()
    spec = make_browser_open_result_spec()
    result = spec.execute(BrowserOpenResultArgs(query="Python", index=2), _ctx(session))
    assert result.ok is True
    assert result.data["name"] == "Python Tutorial - Part 2"
    assert result.data["position"] == 2


def test_open_result_costs_one_scope_action_unit() -> None:
    spec = make_browser_open_result_spec()
    assert spec.base_tier == 1
    assert spec.budget_cost == 1


def test_open_result_refuses_when_the_approved_element_changed() -> None:
    """The element the user confirmed is gone: nothing is opened."""
    session = FakeSession()
    spec = make_browser_open_result_spec()
    ctx = _ctx(session)
    approved = match_identity("link", "Java Basics", 1)
    ctx.approved_target_identity = approved
    result = spec.execute(BrowserOpenResultArgs(query="Java Basics"), ctx)
    assert result.ok is True

    changed = FakeSession([_element("e1", "Java Basics 2024")])
    ctx2 = _ctx(changed)
    ctx2.approved_target_identity = approved
    result2 = spec.execute(BrowserOpenResultArgs(query="Java Basics"), ctx2)
    assert result2.ok is False
    assert "changed after you confirmed" in (result2.error or "")


def test_open_result_is_bound_to_the_approved_identity() -> None:
    """The tool opts into the identity binding, so act hands it the binding."""
    spec = make_browser_open_result_spec()
    assert spec.bind_target_identity is True


# ── the resolver (what an approval is bound to) ───────────────────────────


def test_the_resolver_reports_the_matched_element() -> None:
    spec = make_browser_open_result_spec()
    target = resolve_target(spec, BrowserOpenResultArgs(query="Java Basics"), _ctx())
    assert isinstance(target, ResolvedTarget)
    assert target.kind == "browser"
    assert target.name == "Java Basics"
    assert target.role == "link"
    assert target.origin == _ORIGIN


def test_the_resolver_refuses_an_ambiguous_match() -> None:
    spec = make_browser_open_result_spec()
    assert resolve_target(spec, BrowserOpenResultArgs(query="Python"), _ctx()) is None


def test_the_resolver_refuses_a_missing_match() -> None:
    spec = make_browser_open_result_spec()
    assert resolve_target(spec, BrowserOpenResultArgs(query="nothing here"), _ctx()) is None


def test_the_resolver_refuses_an_out_of_range_index() -> None:
    spec = make_browser_open_result_spec()
    args = BrowserOpenResultArgs(query="Python", index=9)
    assert resolve_target(spec, args, _ctx()) is None


def test_the_resolved_identity_is_the_same_for_gate_and_act() -> None:
    """The approval binding must survive the interrupt boundary.

    The gate and the act node both re-observe the page, and every observation
    hands out fresh ``eN`` handles.  The identity therefore may not contain
    the handle - only the match itself.
    """
    spec = make_browser_open_result_spec()
    args = BrowserOpenResultArgs(query="Python Tutorial - Part 1")
    first = resolve_target(spec, args, _ctx())
    second = resolve_target(spec, args, _ctx())
    assert first is not None and second is not None
    assert first.fingerprint() == second.fingerprint()
    assert first.fingerprint() is not None


def test_a_changed_page_changes_the_identity() -> None:
    """A different match must not reuse the earlier approval."""
    spec = make_browser_open_result_spec()
    args = BrowserOpenResultArgs(query="Python")
    first_page = FakeSession([_element("e1", "Python Tutorial")])
    first = resolve_target(spec, args, _ctx(first_page))
    other = FakeSession(
        [
            _element("e9", "Java Basics"),
            _element("e10", "Python Advanced"),
        ]
    )
    second = resolve_target(spec, args, _ctx(other))
    assert first is not None and second is not None
    assert first.fingerprint() != second.fingerprint()


def test_a_page_that_no_longer_matches_is_refused() -> None:
    """The match disappeared: there is no reliable target any more."""
    spec = make_browser_open_result_spec()
    args = BrowserOpenResultArgs(query="Python")
    other = FakeSession([_element("e9", "Java Basics")])
    assert resolve_target(spec, args, _ctx(other)) is None


# ── the registry ──────────────────────────────────────────────────────────


def test_both_tools_are_registered_by_default() -> None:
    from jarvis.tools.registry import build_default_registry

    registry = build_default_registry()
    assert registry.get_optional("browser_find") is not None
    assert registry.get_optional("browser_open_result") is not None


def test_the_new_tools_never_expose_a_press_or_hotkey_name() -> None:
    from jarvis.tools.registry import build_default_registry

    for name in build_default_registry().names():
        assert "press" not in name.lower()
        assert "hotkey" not in name.lower()
