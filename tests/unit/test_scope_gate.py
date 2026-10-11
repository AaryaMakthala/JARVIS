"""Stage 7 Slice 8: the policy gate's target-resolution + task-scope front end.

These tests drive the **real compiled graph** (``run_task`` / ``resume_task``
over a file-backed SQLite checkpointer) with the real browser tool specs and a
fake managed-browser session, so they assert the interrupt sequence a user
actually experiences:

* a fresh-domain browser step raises one *scope* confirmation, and approval
  issues a transient grant and lets the first action proceed (Option A);
* a risky control inside that freshly-approved scope still raises its own
  action confirmation;
* a statically allowed domain (config) needs no scope, only the normal action
  confirmation;
* a second action inside an existing grant needs only an action confirmation;
* payment controls, password fields and unresolvable targets are hard refusals
  with no confirmation.

Plus pure unit tests for the resolver helper, which never touches the graph.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jarvis.agent.context import make_app_context
from jarvis.agent.graph import open_sqlite_checkpointer
from jarvis.agent.nodes.util import resolve_step_target
from jarvis.agent.runner import resume_task, run_task
from jarvis.agent.schemas import ActionIntent
from jarvis.agent.state import Step
from jarvis.browser.context import ElementRef, Observation
from jarvis.browser.tools import (
    BrowserClickArgs,
    make_browser_click_spec,
    make_browser_goto_spec,
    make_browser_read_spec,
    make_browser_type_spec,
)
from jarvis.config import BrowserSettings, Settings
from jarvis.llm.client import FakeLLM
from jarvis.policy.engine import PolicyContext
from jarvis.policy.scope import ScopeRegistry
from support import approve, brain_steps, make_spec, registry_with

_ORIGIN = "https://example.com"
_FRESH = "https://news.example"
_FRESH_HOST = "news.example"


def _element(
    ref: str,
    name: str,
    *,
    role: str = "button",
    input_type: str = "",
    is_password: bool = False,
    origin: str = _ORIGIN,
) -> ElementRef:
    return ElementRef(
        ref=ref,
        role=role,
        name=name,
        origin=origin,
        input_type=input_type,
        is_password=is_password,
    )


class FakeSession:
    """A fake managed Chrome session (no Playwright, no network)."""

    def __init__(self, elements: dict[str, ElementRef] | None = None) -> None:
        self.resolved = (
            elements if elements is not None else {"e1": _element("e1", "Details", role="link")}
        )
        self.calls: list[tuple[object, ...]] = []
        self.observation = Observation(
            url=f"{_ORIGIN}/page",
            origin=_ORIGIN,
            title="Example",
            elements=tuple(self.resolved.values()) or (_element("e1", "Details", role="link"),),
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

    def type_text(self, ref: str, text: str) -> dict[str, object]:
        self.calls.append(("type_text", ref, text))
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


def _browser_registry() -> Any:
    return registry_with(
        make_browser_goto_spec(),
        make_browser_click_spec(),
        make_browser_type_spec(),
        make_browser_read_spec(),
    )


def _ctx(
    tmp_path: Path,
    decision: Any,
    *,
    session: FakeSession,
    allowed_domains: list[str] | None = None,
    scope_registry: ScopeRegistry | None = None,
) -> Any:
    settings = Settings(browser=BrowserSettings(allowed_domains=list(allowed_domains or [])))
    return make_app_context(
        settings,
        llm=FakeLLM([decision]),
        registry=_browser_registry(),
        browser=session,
        scope_registry=scope_registry,
    )


def _action(tool: str, args: dict[str, Any]) -> Any:
    return brain_steps([ActionIntent(tool=tool, args=args, rationale="do it")])


def _is_scope(payload: dict[str, Any]) -> bool:
    return payload.get("scope_confirm") is True


# ── fresh domain: one scope prompt, then it proceeds (Option A) ────────────


def test_fresh_domain_goto_scope_confirm_then_proceeds(tmp_path: Path) -> None:
    session = FakeSession()
    ctx = _ctx(tmp_path, _action("browser_goto", {"url": f"{_FRESH}/story"}), session=session)
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        first = run_task(ctx, saver, "read the news")
        assert first.interrupted is True, "a fresh-domain browse must stop for a scope approval"
        payload = first.confirmation or {}
        assert payload.get("type") == "confirm"
        assert _is_scope(payload) is True
        assert payload.get("scope_domain") == _FRESH_HOST
        assert "news.example" in str(payload.get("summary"))
        assert payload.get("tier") == 1
        assert session.calls == [], "nothing may run before approval"

        second = resume_task(ctx, saver, first.task_id, approve(payload))
        assert second.interrupted is False, (
            f"Option A: the approved scope lets the first action proceed, got {second.final_answer!r}"
        )
        assert ("goto", f"{_FRESH}/story") in session.calls

        grant = ctx.scope_registry.active_grant(first.task_id)
        assert grant is not None
        assert _FRESH_HOST in grant.allowed_domains
        assert grant.confirmation_id

        # Slice 9: act re-observes, verifies the gate's one-shot binding and
        # consumes it, so the action running at all proves a binding existed.
        assert ctx.target_bindings.get(first.task_id, "s1") is None
    finally:
        saver.conn.close()


def test_scope_confirmation_rejection_is_a_refusal(tmp_path: Path) -> None:
    session = FakeSession()
    ctx = _ctx(tmp_path, _action("browser_goto", {"url": f"{_FRESH}/story"}), session=session)
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        first = run_task(ctx, saver, "read the news")
        payload = first.confirmation or {}
        assert _is_scope(payload) is True
        second = resume_task(ctx, saver, first.task_id, {"approved": False})
        assert second.interrupted is False
        assert second.halted_reason
        assert session.calls == [], "a rejected scope must not run the action"
        assert ctx.scope_registry.active_grant(first.task_id) is None
    finally:
        saver.conn.close()


# ── risky control: scope prompt, then its own action prompt ────────────────


def test_risky_control_scopes_then_reconfirms(tmp_path: Path) -> None:
    session = FakeSession(elements={"e1": _element("e1", "Send")})
    ctx = _ctx(tmp_path, _action("browser_click", {"ref": "e1"}), session=session)
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        first = run_task(ctx, saver, "click send")
        payload = first.confirmation or {}
        assert _is_scope(payload) is True, f"expected a scope prompt, got {payload!r}"

        second = resume_task(ctx, saver, first.task_id, approve(payload))
        assert second.interrupted is True, "a risky control needs its own fresh confirmation"
        action = second.confirmation or {}
        assert _is_scope(action) is False
        assert action.get("type") == "confirm"
        assert action.get("step_id") == "s1"
        assert len(str(action.get("action_hash"))) == 64
        assert ("click", "e1") not in session.calls, "nothing may click before the action approval"

        third = resume_task(ctx, saver, first.task_id, approve(action))
        assert third.interrupted is False
        assert ("click", "e1") in session.calls
        assert str(third.decisions["s1"].action_hash) == str(action["action_hash"])
        grant = ctx.scope_registry.active_grant(first.task_id)
        assert grant is not None
        # Slice 9: act consumed the one-shot binding while executing the click.
        assert ctx.target_bindings.get(first.task_id, "s1") is None
    finally:
        saver.conn.close()


# ── statically allowed domain: no scope prompt ────────────────────────────


def test_config_allowed_domain_needs_only_action_confirm(tmp_path: Path) -> None:
    session = FakeSession()
    ctx = _ctx(
        tmp_path,
        _action("browser_goto", {"url": f"{_ORIGIN}/page"}),
        session=session,
        allowed_domains=["example.com"],
    )
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        first = run_task(ctx, saver, "open the page")
        payload = first.confirmation or {}
        assert _is_scope(payload) is False, "a config-allowed domain needs no scope approval"
        assert payload.get("type") == "confirm"

        second = resume_task(ctx, saver, first.task_id, approve(payload))
        assert second.interrupted is False
        assert ("goto", f"{_ORIGIN}/page") in session.calls
        assert ctx.scope_registry.active_grant(first.task_id) is None
    finally:
        saver.conn.close()


# ── second action inside an existing grant: action confirm only ────────────


def test_second_action_inside_grant_needs_only_action_confirm(tmp_path: Path) -> None:
    session = FakeSession(elements={"e1": _element("e1", "Details", role="link", origin=_FRESH)})
    decision = brain_steps(
        [
            ActionIntent(tool="browser_goto", args={"url": f"{_FRESH}/story"}, rationale="go"),
            ActionIntent(tool="browser_click", args={"ref": "e1"}, rationale="open details"),
        ]
    )
    ctx = _ctx(tmp_path, decision, session=session)
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        first = run_task(ctx, saver, "open the story and its details")
        scope = first.confirmation or {}
        assert _is_scope(scope) is True, f"expected the goto's scope prompt, got {scope!r}"

        second = resume_task(ctx, saver, first.task_id, approve(scope))
        assert second.interrupted is True, "the click inside the grant still confirms"
        click = second.confirmation or {}
        assert _is_scope(click) is False
        assert click.get("step_id") == "s2"
        assert ("goto", f"{_FRESH}/story") in session.calls

        third = resume_task(ctx, saver, first.task_id, approve(click))
        assert third.interrupted is False
        assert ("click", "e1") in session.calls
        grants = [ctx.scope_registry.active_grant(first.task_id)]
        assert len([g for g in grants if g is not None]) == 1, "only one scope grant is needed"
    finally:
        saver.conn.close()


# ── hard refusals: no confirmation, no action ──────────────────────────────


def test_payment_control_is_hard_refused(tmp_path: Path) -> None:
    session = FakeSession(elements={"e1": _element("e1", "Buy now")})
    ctx = _ctx(tmp_path, _action("browser_click", {"ref": "e1"}), session=session)
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        outcome = run_task(ctx, saver, "buy it")
        assert outcome.interrupted is False
        assert "payment" in (outcome.halted_reason or "").lower()
        assert ("click", "e1") not in session.calls
        assert ctx.scope_registry.active_grant(outcome.task_id) is None
    finally:
        saver.conn.close()


def test_password_field_is_hard_refused(tmp_path: Path) -> None:
    session = FakeSession(
        elements={
            "p1": _element(
                "p1", "Password", role="textbox", input_type="password", is_password=True
            )
        }
    )
    ctx = _ctx(
        tmp_path,
        _action("browser_type", {"ref": "p1", "text": "hunter2"}),
        session=session,
    )
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        outcome = run_task(ctx, saver, "log in")
        assert outcome.interrupted is False
        assert "password" in (outcome.halted_reason or "").lower()
        assert all(call[0] != "type_text" for call in session.calls)
    finally:
        saver.conn.close()


def test_unresolvable_target_is_hard_refused(tmp_path: Path) -> None:
    session = FakeSession(elements={})
    ctx = _ctx(tmp_path, _action("browser_click", {"ref": "missing"}), session=session)
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        outcome = run_task(ctx, saver, "click the missing thing")
        assert outcome.interrupted is False
        assert "reliably observed" in (outcome.halted_reason or "")
        assert all(call[0] != "click" for call in session.calls)
    finally:
        saver.conn.close()


# ── pure resolver helper ───────────────────────────────────────────────────


def _base_pctx() -> PolicyContext:
    return PolicyContext(registry=_browser_registry(), settings=Settings())


def test_resolve_passthrough_for_resolverless_step() -> None:
    spec = make_spec("fake_echo", base_tier=1, record=[])
    pctx = _base_pctx()
    step = Step(id="s1", tool="fake_echo", args={"text": "hi"}, rationale="x")
    resolved = resolve_step_target(
        pctx,
        step=step,
        spec=spec,
        settings=Settings(),
        tool_ctx=None,
        scope_registry=None,
        task_id="t1",
        plan_hash_v="h",
    )
    assert resolved.context is pctx
    assert resolved.target is None
    assert resolved.refused is False


def test_resolve_browser_goto_without_scope(tmp_path: Path) -> None:
    session = FakeSession()
    pctx = _base_pctx()
    step = Step(id="s1", tool="browser_goto", args={"url": f"{_FRESH}/x"}, rationale="x")
    resolved = resolve_step_target(
        pctx,
        step=step,
        spec=make_browser_goto_spec(),
        settings=Settings(),
        tool_ctx=type("C", (), {"browser": session})(),
        scope_registry=ScopeRegistry(),
        task_id="t1",
        plan_hash_v="h",
    )
    assert resolved.refused is False
    assert resolved.host == _FRESH_HOST
    assert resolved.context.origin == _FRESH
    assert resolved.context.allowed_domains == ()
    assert resolved.context.scope_approved is False


def test_resolve_browser_goto_with_config_domain() -> None:
    session = FakeSession()
    settings = Settings(browser=BrowserSettings(allowed_domains=["news.example"]))
    step = Step(id="s1", tool="browser_goto", args={"url": f"{_FRESH}/x"}, rationale="x")
    resolved = resolve_step_target(
        _base_pctx(),
        step=step,
        spec=make_browser_goto_spec(),
        settings=settings,
        tool_ctx=type("C", (), {"browser": session})(),
        scope_registry=ScopeRegistry(),
        task_id="t1",
        plan_hash_v="h",
    )
    assert resolved.context.scope_approved is True
    assert _FRESH_HOST in resolved.context.allowed_domains


def test_resolve_browser_goto_with_valid_grant() -> None:
    session = FakeSession()
    registry = ScopeRegistry()
    registry.issue(
        "t1",
        plan_hash="h",
        confirmation_id="cid-1",
        allowed_domains=[_FRESH_HOST],
        ttl_s=120.0,
    )
    step = Step(id="s1", tool="browser_goto", args={"url": f"{_FRESH}/x"}, rationale="x")
    resolved = resolve_step_target(
        _base_pctx(),
        step=step,
        spec=make_browser_goto_spec(),
        settings=Settings(),
        tool_ctx=type("C", (), {"browser": session})(),
        scope_registry=registry,
        task_id="t1",
        plan_hash_v="h",
    )
    assert resolved.context.scope_approved is True
    assert _FRESH_HOST in resolved.context.allowed_domains


def test_resolve_expired_grant_is_not_effective() -> None:
    session = FakeSession()
    clock = {"t": 1000.0}
    registry = ScopeRegistry(clock=lambda: clock["t"])
    registry.issue(
        "t1", plan_hash="h", confirmation_id="cid-1", allowed_domains=[_FRESH_HOST], ttl_s=10.0
    )
    clock["t"] = 2000.0  # long past expiry
    step = Step(id="s1", tool="browser_goto", args={"url": f"{_FRESH}/x"}, rationale="x")
    resolved = resolve_step_target(
        _base_pctx(),
        step=step,
        spec=make_browser_goto_spec(),
        settings=Settings(),
        tool_ctx=type("C", (), {"browser": session})(),
        scope_registry=registry,
        task_id="t1",
        plan_hash_v="h",
    )
    assert resolved.context.scope_approved is False
    assert resolved.context.allowed_domains == ()


def test_resolve_click_missing_ref_is_refused() -> None:
    session = FakeSession(elements={})
    args = BrowserClickArgs(ref="missing")
    resolved = resolve_step_target(
        _base_pctx(),
        step=Step(id="s1", tool="browser_click", args=args.model_dump(), rationale="x"),
        spec=make_browser_click_spec(),
        settings=Settings(),
        tool_ctx=type("C", (), {"browser": session})(),
        scope_registry=ScopeRegistry(),
        task_id="t1",
        plan_hash_v="h",
    )
    assert resolved.refused is True
    assert resolved.target is None
