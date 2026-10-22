"""Stage 8.6, slice 8.6.2: the acted-on element is the confirmed element.

``browser_open_result`` names its target by *query*, not by a handle the
backend issued, so the approval is bound to a stable identity (role, name and
the position among the matches) instead of the transient ``e1..eN`` handle.

These tests drive the **real** act node with the real ``browser_open_result``
spec and a fake managed browser (no Playwright, no Chrome, no network) and
assert the two guarantees:

* the gate's binding and the act node's re-observation agree on the same
  element, and the tool clicks that element;
* when the page changes between the confirmation and the action, the action is
  refused and **nothing** is clicked - the query is never resolved to a
  different element than the user confirmed.
"""

from __future__ import annotations

from typing import Any

from tests.support import registry_with

from jarvis.agent.context import make_app_context
from jarvis.agent.nodes.act import act
from jarvis.agent.state import Plan, Step
from jarvis.browser.context import ElementRef
from jarvis.browser.find import match_identity, match_indexes
from jarvis.browser.tools import make_browser_open_result_spec
from jarvis.config import BrowserSettings, Settings
from jarvis.policy import rules
from jarvis.policy.scope import ScopeRegistry
from jarvis.tools.base import resolve_target

_ORIGIN = "https://example.com"
_TASK = "t1"


def _element(ref: str, name: str, *, role: str = "link") -> ElementRef:
    return ElementRef(ref=ref, role=role, name=name, origin=_ORIGIN)


class FakeSession:
    """A fake managed Chrome session that answers queries from a page."""

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
        self.clicked: list[str] = []

    def _matches(self, query: str, role: str) -> list[ElementRef]:
        pairs = [(el.role, el.name) for el in self.elements]
        indexes = match_indexes(pairs, query, role=role)
        return [self.elements[i] for i in indexes]

    def find(self, query: str, *, role: str = "", limit: int = 300) -> list[ElementRef]:
        return self._matches(query, role)[:limit]

    def open_match(
        self,
        query: str,
        *,
        role: str = "",
        index: int | None = None,
        expect_identity: str | None = None,
    ) -> dict[str, Any]:
        matches = self._matches(query, role)
        if not matches:
            raise RuntimeError("no match")
        if index is None and len(matches) > 1:
            raise RuntimeError("ambiguous")
        position = 1 if index is None else index
        element = matches[position - 1]
        if expect_identity:
            identity = match_identity(element.role, element.name, position)
            if identity != expect_identity:
                raise RuntimeError("the page element changed after you confirmed it")
        self.clicked.append(element.name)
        return {
            "ref": element.ref,
            "role": element.role,
            "name": element.name,
            "origin": element.origin,
            "position": position,
            "matches": len(matches),
        }

    def close(self) -> None:  # pragma: no cover - protocol completeness
        return None


def _ctx(session: FakeSession, *, scope_registry: ScopeRegistry | None = None) -> Any:
    settings = Settings(browser=BrowserSettings(allowed_domains=["example.com"]))
    return make_app_context(
        settings,
        registry=registry_with(make_browser_open_result_spec()),
        browser=session,
        scope_registry=scope_registry,
    )


def _step(args: dict[str, Any], sid: str = "s1") -> Step:
    return Step(id=sid, tool="browser_open_result", args=args, rationale="open the result")


def _hash(ctx: Any, step: Step) -> str:
    spec = ctx.registry.get(step.tool)
    args = spec.args_model.model_validate(step.args)
    return str(rules.action_hash(spec, args))


def _state(step: Step, *, approved: list[str]) -> dict[str, Any]:
    return {
        "plan": Plan(goal="open the result", steps=[step]),
        "step_index": 0,
        "task_id": _TASK,
        "approved_hashes": list(approved),
        "gated_resolved_paths": {},
        "results": [],
        "user_input": "open the result",
    }


def _record_binding(ctx: Any, step: Step) -> None:
    """Record the binding the policy gate would have put for this step."""
    spec = ctx.registry.get(step.tool)
    args = spec.args_model.model_validate(step.args)
    target = resolve_target(spec, args, ctx.tool_context())
    assert target is not None
    ctx.target_bindings.put(
        _TASK,
        step.id,
        action_hash=_hash(ctx, step),
        fingerprint=str(target.fingerprint()),
        origin=str(target.origin or ""),
        scope_confirmation_id="",
    )


def _result(update: dict[str, Any]) -> Any:
    results = list(update.get("results") or [])
    assert results, "act must have recorded a result"
    return results[-1]


def test_the_confirmed_element_is_the_one_that_is_opened() -> None:
    session = FakeSession()
    ctx = _ctx(session)
    step = _step({"query": "Java Basics"})
    _record_binding(ctx, step)

    update = act(_state(step, approved=[_hash(ctx, step)]), ctx)

    result = _result(update)
    assert result.ok, result.error
    assert session.clicked == ["Java Basics"]
    assert ctx.target_bindings.get(_TASK, "s1") is None, "the binding was consumed"


def test_a_page_that_changed_after_the_confirmation_is_refused() -> None:
    """The confirmed element is gone: nothing is opened."""
    session = FakeSession()
    ctx = _ctx(session)
    step = _step({"query": "Java Basics"})
    _record_binding(ctx, step)

    # The page changed between the confirmation and the action.
    session.elements = [_element("e9", "Java Basics 2024")]

    update = act(_state(step, approved=[_hash(ctx, step)]), ctx)

    result = _result(update)
    assert not result.ok
    assert session.clicked == [], "nothing may be opened on a changed page"
    assert ctx.scope_registry.consumed(_TASK) == 0


def test_an_ambiguous_query_is_refused_before_anything_is_opened() -> None:
    """The gate refuses an ambiguous target, so no binding is ever recorded."""
    session = FakeSession()
    ctx = _ctx(session)
    step = _step({"query": "Python"})

    update = act(_state(step, approved=[_hash(ctx, step)]), ctx)

    result = _result(update)
    assert not result.ok
    assert session.clicked == []
    assert ctx.target_bindings.get(_TASK, "s1") is None


def test_an_ambiguous_query_is_refused_by_the_gate_too() -> None:
    """The resolver itself refuses an ambiguous query (fail closed)."""
    session = FakeSession()
    ctx = _ctx(session)
    step = _step({"query": "Python"})
    spec = ctx.registry.get(step.tool)
    args = spec.args_model.model_validate(step.args)
    assert resolve_target(spec, args, ctx.tool_context()) is None
