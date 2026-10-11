"""Stage 7 Slice 9: act re-observer, policy/execution parity, atomic budget.

These tests drive the **real** act node (`jarvis.agent.nodes.act.act`) with the
real browser specs and a fake managed-browser session — exactly what the graph
runs after the policy gate — plus one full graph round-trip to pin the batch
binding path.  They assert:

* act re-observes the interaction target and proves the gate's one-shot binding
  byte-identical immediately before dispatch, consuming it in the same step;
* a missing, changed or superseded binding fails closed with no side effect;
* the atomic task budget is reserved just before dispatch, counts exactly the
  approved, executed actions, and is never refunded;
* the exactly-once guard prevents a double charge when act is re-entered;
* generic ``type_text`` is hard-blocked once a managed browser is wired up.

This is the deterministic half of the acceptance check: "policy/execution
parity".  No network, no Playwright, no Windows GUI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from tests.support import approve, brain_steps, registry_with

from jarvis.agent.context import make_app_context
from jarvis.agent.graph import open_sqlite_checkpointer
from jarvis.agent.nodes.act import act
from jarvis.agent.runner import resume_task, run_task
from jarvis.agent.schemas import ActionIntent
from jarvis.agent.state import Plan, Step
from jarvis.browser.context import ElementRef, Observation
from jarvis.browser.tools import (
    make_browser_click_spec,
    make_browser_goto_spec,
    make_browser_read_spec,
)
from jarvis.config import BrowserSettings, Settings
from jarvis.llm.client import FakeLLM
from jarvis.policy import rules
from jarvis.policy.engine import PolicyContext, PolicyEngine
from jarvis.policy.scope import ScopeRegistry
from jarvis.policy.targets import ResolvedTarget
from jarvis.tools.base import ToolContext, ToolResult, ToolSpec, resolve_target
from jarvis.tools.keyboard import make_type_text_spec

_ORIGIN = "https://example.com"
_TASK = "t1"


def _element(ref: str, name: str, *, role: str = "button") -> ElementRef:
    return ElementRef(ref=ref, role=role, name=name, origin=_ORIGIN)


class FakeSession:
    """A fake managed Chrome session (no Playwright, no network)."""

    def __init__(self, elements: dict[str, ElementRef] | None = None) -> None:
        self.resolved = elements if elements is not None else {"e1": _element("e1", "Details")}
        self.calls: list[tuple[object, ...]] = []
        self.observation = Observation(
            url=f"{_ORIGIN}/page",
            origin=_ORIGIN,
            title="Example",
            elements=tuple(self.resolved.values()),
        )

    def goto(self, url: str) -> dict[str, object]:
        self.calls.append(("goto", url))
        return {"url": url, "origin": _ORIGIN, "title": "Example"}

    def read(self, max_elements: int = 80) -> Observation:
        self.calls.append(("read", max_elements))
        return self.observation

    def resolve(self, ref: str) -> ElementRef | None:
        self.calls.append(("resolve", ref))
        return self.resolved.get(ref)

    def click(self, ref: str) -> dict[str, object]:
        self.calls.append(("click", ref))
        el = self.resolved[ref]
        return {"ref": ref, "role": el.role, "name": el.name, "origin": el.origin}

    def close(self) -> None:  # pragma: no cover - protocol completeness
        self.calls.append(("close",))


def _registry() -> Any:
    return registry_with(
        make_browser_goto_spec(),
        make_browser_click_spec(),
        make_browser_read_spec(),
    )


def _ctx(
    session: FakeSession,
    *,
    scope_registry: ScopeRegistry | None = None,
    dry_run: bool = False,
) -> Any:
    settings = Settings(browser=BrowserSettings(allowed_domains=["example.com"]))
    return make_app_context(
        settings,
        registry=_registry(),
        browser=session,
        scope_registry=scope_registry,
        dry_run=dry_run,
    )


def _step(tool: str, args: dict[str, Any], sid: str = "s1") -> Step:
    return Step(id=sid, tool=tool, args=args, rationale="do it")


def _state(step: Step, *, approved: list[str]) -> dict[str, Any]:
    return {
        "plan": Plan(goal="perform the step", steps=[step]),
        "step_index": 0,
        "task_id": _TASK,
        "approved_hashes": list(approved),
        "gated_resolved_paths": {},
        "results": [],
        "user_input": "perform the step",
    }


def _hash(ctx: Any, step: Step) -> str:
    """The action_hash the gate would approve for this step."""
    spec = ctx.registry.get(step.tool)
    args = spec.args_model.model_validate(step.args)
    return str(rules.action_hash(spec, args))


def _record_binding(ctx: Any, session: FakeSession, step: Step, *, scope_cid: str = "") -> None:
    """Record the binding the policy gate would have put (Slice 8 path)."""
    spec = ctx.registry.get(step.tool)
    args = spec.args_model.model_validate(step.args)
    target = resolve_target(spec, args, ctx.tool_context())
    ctx.target_bindings.put(
        _TASK,
        step.id,
        action_hash=_hash(ctx, step),
        fingerprint=str(target.fingerprint()),
        origin=str(getattr(target, "origin", "") or ""),
        scope_confirmation_id=scope_cid,
    )


def _result(update: dict[str, Any]) -> Any:
    results = list(update.get("results") or [])
    assert results, "act must have recorded a result"
    return results[-1]


# ── parity: act re-observes, proves the one-shot binding, consumes it ────────


def test_approved_target_step_executes_and_consumes_binding_and_budget() -> None:
    session = FakeSession()
    ctx = _ctx(session)
    step = _step("browser_click", {"ref": "e1"})
    _record_binding(ctx, session, step)
    update = act(_state(step, approved=[_hash(ctx, step)]), ctx)

    result = _result(update)
    assert result.ok, result.error
    assert ("click", "e1") in session.calls
    assert ctx.scope_registry.consumed(_TASK) == 1, "the click charged one budget unit"
    assert ctx.target_bindings.get(_TASK, "s1") is None, "the one-shot binding was consumed"


def test_act_reobserves_immediately_before_dispatch() -> None:
    session = FakeSession(elements={"e1": _element("e1", "Details", role="link")})
    ctx = _ctx(session)
    step = _step("browser_click", {"ref": "e1"})
    _record_binding(ctx, session, step)
    session.calls.clear()

    act(_state(step, approved=[_hash(ctx, step)]), ctx)

    resolve_index = [i for i, c in enumerate(session.calls) if c and c[0] == "resolve"]
    click_index = [i for i, c in enumerate(session.calls) if c and c[0] == "click"]
    assert resolve_index and click_index, "act must re-observe the element then click it"
    assert click_index[0] > resolve_index[-1], "no click may happen before a fresh resolve"


def test_missing_binding_is_refused_no_side_effect() -> None:
    session = FakeSession()
    ctx = _ctx(session)
    step = _step("browser_click", {"ref": "e1"})

    update = act(_state(step, approved=[_hash(ctx, step)]), ctx)

    result = _result(update)
    assert not result.ok and result.error == "target binding no_binding"
    assert ("click", "e1") not in session.calls
    assert ctx.scope_registry.consumed(_TASK) == 0
    assert update.get("halted_reason")


def test_changed_target_element_is_refused_and_binding_invalidated() -> None:
    session = FakeSession(elements={"e1": _element("e1", "Details")})
    ctx = _ctx(session)
    step = _step("browser_click", {"ref": "e1"})
    _record_binding(ctx, session, step)
    session.resolved["e1"] = _element("e1", "Support")  # name changed after approval

    update = act(_state(step, approved=[_hash(ctx, step)]), ctx)

    result = _result(update)
    assert not result.ok and result.error == "target binding mismatch"
    assert ("click", "e1") not in session.calls
    assert ctx.target_bindings.get(_TASK, "s1") is None, "a mismatch destroys the binding"
    assert ctx.scope_registry.consumed(_TASK) == 0


def test_superseded_scope_confirmation_is_refused() -> None:
    session = FakeSession()
    ctx = _ctx(session)
    step = _step("browser_click", {"ref": "e1"})
    _record_binding(ctx, session, step, scope_cid="old-cid")  # gate authorized under grant A
    ctx.scope_registry.issue(_TASK, plan_hash="p2", confirmation_id="new-cid")  # re-scoped

    update = act(_state(step, approved=[_hash(ctx, step)]), ctx)

    result = _result(update)
    assert not result.ok and result.error == "target binding mismatch"
    assert ("click", "e1") not in session.calls


def test_unresolvable_target_is_refused() -> None:
    session = FakeSession(elements={"e1": _element("e1", "Details")})
    ctx = _ctx(session)
    step = _step("browser_click", {"ref": "e1"})
    _record_binding(ctx, session, step)
    session.resolved.pop("e1")  # target gone before dispatch

    update = act(_state(step, approved=[_hash(ctx, step)]), ctx)

    result = _result(update)
    assert not result.ok and result.error.startswith("interaction target")
    assert ("click", "e1") not in session.calls


class _UiaArgs(BaseModel):
    ref: str = Field(min_length=1)


def test_uia_style_target_needs_binding_but_no_browser_origin() -> None:
    """A non-browser target follows the UIA path: binding yes, origin no."""
    calls: list[str] = []

    def run(args: _UiaArgs, tool_ctx: ToolContext) -> ToolResult:
        calls.append(args.ref)
        return ToolResult(ok=True, output="uia_save ran", data={"ref": args.ref})

    def resolver(args: _UiaArgs, tool_ctx: ToolContext) -> ResolvedTarget:
        return ResolvedTarget(kind="uia", ref="save-btn", name="Save", role="button")

    registry = registry_with(
        ToolSpec(
            name="uia_save",
            description="fake UIA save",
            args_model=_UiaArgs,
            base_tier=1,
            budget_cost=1,
            run=run,
            resolve=resolver,
        )
    )
    settings = Settings(browser=BrowserSettings(allowed_domains=["example.com"]))
    ctx = make_app_context(settings, registry=registry)
    step = _step("uia_save", {"ref": "x"})
    target = resolve_target(ctx.registry.get("uia_save"), _UiaArgs(ref="x"), ctx.tool_context())
    ctx.target_bindings.put(
        _TASK,
        "s1",
        action_hash=_hash(ctx, step),
        fingerprint=str(target.fingerprint()),
        origin="",
        scope_confirmation_id="",
    )

    update = act(_state(step, approved=[_hash(ctx, step)]), ctx)

    result = _result(update)
    assert result.ok, result.error
    assert calls == ["x"]
    assert ctx.scope_registry.consumed(_TASK) == 1


# ── atomic task budget ───────────────────────────────────────────────────────


def test_budget_charges_exactly_the_executed_actions_and_stops_at_eight() -> None:
    session = FakeSession(elements={"e1": _element("e1", "A"), "e2": _element("e2", "B")})
    ctx = _ctx(session)
    step1 = _step("browser_click", {"ref": "e1"}, sid="s1")
    step2 = _step("browser_click", {"ref": "e2"}, sid="s2")
    _record_binding(ctx, session, step1)
    _record_binding(ctx, session, step2)
    ctx.scope_registry.reserve(_TASK, cost=7)  # 7 actions used earlier in the task

    first = act(_state(step1, approved=[_hash(ctx, step1)]), ctx)
    assert _result(first).ok  # the 8th action goes through
    assert ctx.scope_registry.consumed(_TASK) == 8

    second = act(_state(step2, approved=[_hash(ctx, step2)]), ctx)
    result = _result(second)
    assert not result.ok and result.error == "task action budget: budget_exhausted"
    assert ("click", "e2") not in session.calls
    assert second.get("halted_reason")


def test_read_observation_charges_no_budget_and_needs_no_binding() -> None:
    session = FakeSession()
    ctx = _ctx(session)
    step = _step("browser_read", {"max_elements": 50})

    update = act(_state(step, approved=[_hash(ctx, step)]), ctx)

    assert _result(update).ok
    assert ("read", 50) in session.calls
    assert ctx.scope_registry.consumed(_TASK) == 0


def test_dry_run_skips_binding_consume_and_budget() -> None:
    session = FakeSession()
    ctx = _ctx(session, dry_run=True)
    step = _step("browser_click", {"ref": "e1"})
    _record_binding(ctx, session, step)

    update = act(_state(step, approved=[_hash(ctx, step)]), ctx)

    assert _result(update).ok, _result(update).error
    assert ctx.scope_registry.consumed(_TASK) == 0, "dry runs must not spend the budget"
    assert ctx.target_bindings.get(_TASK, "s1") is not None, "dry runs keep the one-shot binding"


def test_exactly_once_guard_prevents_double_charge_on_reentry() -> None:
    session = FakeSession()
    ctx = _ctx(session)
    step = _step("browser_click", {"ref": "e1"})
    _record_binding(ctx, session, step)
    state = _state(step, approved=[_hash(ctx, step)])

    first = act(state, ctx)
    assert _result(first).ok
    assert ctx.scope_registry.consumed(_TASK) == 1

    reentered = act(  # the same step is presented again (replay/retry)
        {**state, "results": list(first["results"])}, ctx
    )
    assert reentered.get("step_index") == 1, "the step is advanced, not re-executed"
    assert ctx.scope_registry.consumed(_TASK) == 1, "no second charge on re-entry"
    assert len([c for c in session.calls if c and c[0] == "click"]) == 1


# ── managed-browser typing bypass (engine gate) ──────────────────────────────


def test_type_text_into_managed_browser_is_hard_blocked() -> None:
    registry = registry_with(make_type_text_spec())
    settings = Settings()
    engine = PolicyEngine()
    step = _step("type_text", {"target_app": "chrome", "text": "hello"})

    with_managed = engine.decide(
        step, PolicyContext(registry=registry, settings=settings, managed_browser_app="chrome")
    )
    assert not with_managed.allowed and with_managed.tier == 3
    assert any("must use browser_type" in r for r in with_managed.reasons)
    assert not with_managed.needs_confirm, "a hard block is never reachable by confirmation"

    without = engine.decide(step, PolicyContext(registry=registry, settings=settings))
    assert all("must use browser_type" not in r for r in without.reasons)


def test_type_text_into_other_apps_is_untouched() -> None:
    registry = registry_with(make_type_text_spec())
    settings = Settings()
    engine = PolicyEngine()
    step = _step("type_text", {"target_app": "notepad", "text": "hello"})

    decision = engine.decide(
        step, PolicyContext(registry=registry, settings=settings, managed_browser_app="chrome")
    )
    assert all("must use browser_type" not in r for r in decision.reasons)


# ── batch path records bindings for every target-bearing step ────────────────


def test_batch_approval_records_bindings_for_target_steps(tmp_path: Path) -> None:
    """Two config-allowed browser steps batch; bindings drive both dispatches."""
    session = FakeSession()
    settings = Settings(browser=BrowserSettings(allowed_domains=["example.com"]))
    ctx = make_app_context(
        settings,
        llm=FakeLLM([brain_steps([_goto("x"), _goto("y")])]),
        registry=_registry(),
        browser=session,
    )
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    first = run_task(ctx, saver, "open two pages")
    payload = first.confirmation or {}
    assert payload.get("type") == "plan_approval", payload
    assert first.interrupted is True
    assert session.calls == [], "nothing runs before the batch approval"

    second = resume_task(ctx, saver, first.task_id, approve(payload))
    assert second.interrupted is False, f"batch ran both pages, got {second.final_answer!r}"
    assert ("goto", f"{_ORIGIN}/x") in session.calls
    assert ("goto", f"{_ORIGIN}/y") in session.calls
    assert ctx.scope_registry.consumed(first.task_id) == 2
    binding_consumed = all(ctx.target_bindings.get(first.task_id, s) is None for s in ("s1", "s2"))
    assert binding_consumed, "act consumed the one-shot binding of every batch step"


def _goto(path: str) -> ActionIntent:
    return ActionIntent(
        tool="browser_goto",
        args={"url": f"{_ORIGIN}/{path}"},
        rationale=f"open page {path}",
    )
