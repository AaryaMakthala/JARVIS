"""Wiring tests: browsing grant in policy_gate / act / daemon revocation."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

if sys.version_info >= (3, 13):
    pytest.skip("skip: python too new for this fixture set", allow_module_level=True)

from tests.support import approve, deny, registry_with
from tests.unit.test_specialist_loop_e2e import _FakeSession

from jarvis.agent.context import make_app_context
from jarvis.agent.graph import open_sqlite_checkpointer
from jarvis.agent.nodes.act import act
from jarvis.agent.nodes.util import browse_grant_check
from jarvis.agent.runner import resume_task, run_task
from jarvis.agent.schemas import BrainDecision, SpecialistDecision
from jarvis.agent.state import Decision, Plan, Step
from jarvis.browser.context import ElementRef, Observation
from jarvis.browser.tools import make_browser_press_key_spec
from jarvis.config import AgentSettings, BrowserSettings, Settings
from jarvis.daemon.server import DaemonServer
from jarvis.llm.client import FakeLLM
from jarvis.policy.browse_grant import BrowseGrantRegistry
from jarvis.tools.registry import build_default_registry

BROWSE_PROMPT = "Allow JARVIS to browse example.com for this task?"


class _RiskySession(_FakeSession):
    """A fake page whose first element is a risky (non-payment) button."""

    def __init__(self) -> None:
        super().__init__()
        self.resolved = {
            "e2": ElementRef(
                ref="e2",
                role="button",
                name="Delete note",
                origin="https://example.com/page",
                input_type="",
                is_password=False,
            )
        }
        self.observation = Observation(
            url="https://example.com/page",
            origin="https://example.com",
            title="Shop",
            elements=tuple(self.resolved.values()),
        )


def _ctx(session: Any, *, grant: bool = True, max_calls: int = 10) -> Any:
    settings = Settings(
        agent=AgentSettings(use_specialists=True, specialist_max_calls=max_calls),
        browser=BrowserSettings(allowed_domains=["example.com"], task_grant_enabled=grant),
    )
    return make_app_context(
        settings,
        llm=FakeLLM([]),
        registry=build_default_registry(settings),
        browser=session,
    )


def _browse(goal: str = "browse example") -> BrainDecision:
    return BrainDecision(request_type="browse", goal=goal, actions=[])


def _goto(url: str = "https://example.com/page") -> SpecialistDecision:
    return SpecialistDecision(kind="act", tool="browser_goto", args={"url": url}, rationale="go")


def _done(text: str = "Done.") -> SpecialistDecision:
    return SpecialistDecision(kind="done", summary=text)


def _run(
    ctx: Any, saver: Any, decisions: list[Any], text: str = "browse example"
) -> tuple[Any, list[dict[str, Any]]]:
    """Run a task, approving every confirmation; return the outcome + payloads."""
    ctx.llm = FakeLLM(decisions)
    payloads: list[dict[str, Any]] = []
    out = run_task(ctx, saver, text, thread_id="t1")
    rounds = 0
    while out.interrupted and out.confirmation:
        payloads.append(dict(out.confirmation))
        rounds += 1
        assert rounds <= 10, "too many confirmation rounds"
        out = resume_task(ctx, saver, out.task_id, approve(out.confirmation))
    return out, payloads


def test_first_prompt_text_and_grant_covers_the_next_step(tmp_path: Path) -> None:
    ctx = _ctx(_FakeSession())
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        out, payloads = _run(
            ctx,
            saver,
            [_browse(), _goto(), _goto(), _done("Browsed.")],
        )
        assert payloads, "expected the task-scope prompt"
        assert payloads[0]["summary"] == BROWSE_PROMPT
        assert payloads[0]["browse_grant"] is True
        # The second goto on the same host is covered: no second interrupt.
        assert len(payloads) == 1, payloads
        assert out.halted_reason is None
        assert out.final_answer == "Browsed."
        assert ctx.browse_grants.consumed("t1") == 2
    finally:
        saver.conn.close()


def test_no_issues_no_grant(tmp_path: Path) -> None:
    ctx = _ctx(_FakeSession())
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        ctx.llm = FakeLLM([_browse(), _goto(), _done()])
        out = run_task(ctx, saver, "browse example", thread_id="t1")
        assert out.interrupted is True and out.confirmation is not None
        assert out.confirmation["summary"] == BROWSE_PROMPT
        out = resume_task(ctx, saver, out.task_id, deny(out.confirmation))
        assert out.interrupted is False
        assert out.halted_reason
        assert ctx.browse_grants.active_for_task("t1") is None
        assert ctx.browse_grants.validate("t1").reason == "no_grant"
    finally:
        saver.conn.close()


def test_risky_click_on_the_approved_host_still_interrupts(tmp_path: Path) -> None:
    ctx = _ctx(_RiskySession())
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        click = SpecialistDecision(
            kind="act", tool="browser_click", args={"ref": "e2"}, rationale="buy"
        )
        out, payloads = _run(ctx, saver, [_browse(), _goto(), click, _done()])
        assert len(payloads) == 2, payloads
        assert payloads[0]["summary"] == BROWSE_PROMPT
        assert payloads[1]["summary"] != BROWSE_PROMPT
        # A normal per-action confirmation, not a grant-covered skip.
        assert not payloads[1].get("browse_grant")
        assert payloads[1]["step_id"] == "s1"
        assert out.halted_reason is None
    finally:
        saver.conn.close()


def test_first_risky_click_gets_no_browse_prompt(tmp_path: Path) -> None:
    """An ineligible first step keeps its ordinary confirmation, no grant."""
    session = _RiskySession()
    ctx = _ctx(session)
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        click = SpecialistDecision(
            kind="act", tool="browser_click", args={"ref": "e2"}, rationale="delete"
        )
        out, payloads = _run(ctx, saver, [_browse(), click, _done("Clicked.")])
        assert payloads, "expected the ordinary action confirmation"
        assert payloads[0]["summary"] != BROWSE_PROMPT
        assert "browse_grant" not in payloads[0]
        assert ctx.browse_grants.active_for_task("t1") is None
        assert ("click", "e2") in session.calls
        assert out.final_answer == "Clicked."
    finally:
        saver.conn.close()


def test_step_on_another_host_interrupts(tmp_path: Path) -> None:
    """Host B gets its own authorization, and approving it lets the step run.

    A grant for host A never covers host B: the gate raises host B's own
    task-scope confirmation, names B, and approving it authorizes exactly that
    step.  Gate and act must agree on the authorization id, so the *result* is
    the contract here - if they disagreed, act would refuse and the step would
    not run.
    """
    session = _FakeSession()
    ctx = _ctx(session)
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        out, payloads = _run(
            ctx, saver, [_browse(), _goto(), _goto("https://other.example/page"), _done("B.")]
        )
        assert len(payloads) == 2, payloads
        assert payloads[0]["summary"] == BROWSE_PROMPT
        # Host B's own authorization: a scope confirmation naming B, never a
        # reuse of host A's browsing grant.
        assert "other.example" in payloads[1]["summary"]
        assert payloads[1].get("scope_confirm") is True
        assert "browse_grant" not in payloads[1]
        assert out.halted_reason is None, out.halted_reason
        # The step ran on host B under that authorization.
        assert ("goto", "https://other.example/page") in session.calls
        results = {r.step_id: r for r in out.state.get("results") or []}
        assert results["s1"].ok is True and results["s1"].error is None
        # Host A's grant was neither reused nor charged for B's step.
        grant_a = ctx.browse_grants.active_for_task("t1")
        assert grant_a is not None and grant_a.host == "example.com"
        assert ctx.browse_grants.consumed("t1") == 1
    finally:
        saver.conn.close()


def test_refusing_the_second_host_runs_nothing_there(tmp_path: Path) -> None:
    """Refusing host B leaves B untouched and host A's grant in force."""
    session = _FakeSession()
    ctx = _ctx(session)
    ctx.llm = FakeLLM([_browse(), _goto(), _goto("https://other.example/page"), _done()])
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        out = run_task(ctx, saver, "browse example", thread_id="t1")
        assert out.interrupted is True and out.confirmation is not None
        first = dict(out.confirmation)
        assert first["summary"] == BROWSE_PROMPT
        out = resume_task(ctx, saver, out.task_id, approve(first))
        assert out.interrupted is True and out.confirmation is not None
        second = dict(out.confirmation)
        assert "other.example" in second["summary"]
        out = resume_task(ctx, saver, out.task_id, deny(second))
        assert out.interrupted is False
        assert out.halted_reason
        # Nothing reached host B, and A's authorization is unaffected.
        assert ("goto", "https://other.example/page") not in session.calls
        assert all(not (r.step_id == "s1" and r.ok) for r in out.state.get("results") or [])
        assert ctx.browse_grants.consumed("t1") == 1
    finally:
        saver.conn.close()


class _VanishingGrantRegistry(BrowseGrantRegistry):
    """A registry whose grant is revoked the instant it is issued.

    Models the gate -> act window closing: the gate approves and records the
    target binding under the grant id, then the coverage is gone before act
    re-checks it.
    """

    def issue(self, *args: Any, **kwargs: Any) -> Any:
        grant = super().issue(*args, **kwargs)
        self.revoke(grant.task_id)
        return grant


def test_act_refuses_when_coverage_vanishes_after_the_gate(tmp_path: Path) -> None:
    """Act re-checks coverage itself: losing it after the gate refuses the step."""
    session = _FakeSession()
    ctx = _ctx(session)
    ctx.browse_grants = _VanishingGrantRegistry()
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        out, payloads = _run(ctx, saver, [_browse(), _goto(), _done()])
        assert payloads and payloads[0]["summary"] == BROWSE_PROMPT
        results = {r.step_id: r for r in out.state.get("results") or []}
        assert "s0" in results and results["s0"].ok is False
        assert ("goto", "https://example.com/page") not in session.calls
    finally:
        saver.conn.close()


def test_browser_search_with_unknown_site_is_not_covered(tmp_path: Path) -> None:
    ctx = _ctx(_FakeSession())
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        search = SpecialistDecision(
            kind="act",
            tool="browser_search",
            args={"site": "nosuchsite", "query": "test"},
            rationale="search",
        )
        out, payloads = _run(ctx, saver, [_browse(), search, _done()])
        assert payloads, payloads
        assert payloads[0]["summary"] != BROWSE_PROMPT
        assert "browse_grant" not in payloads[0]
        assert out.state.get("halted_reason") is None
    finally:
        saver.conn.close()


def test_grant_disabled_keeps_the_normal_prompt(tmp_path: Path) -> None:
    ctx = _ctx(_FakeSession(), grant=False)
    assert ctx.browse_grants.enabled is False
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        out, payloads = _run(ctx, saver, [_browse(), _goto(), _done()])
        assert payloads and payloads[0]["summary"] != BROWSE_PROMPT
        assert "browse_grant" not in payloads[0]
        assert ctx.browse_grants.active_for_task("t1") is None
        assert out.final_answer
    finally:
        saver.conn.close()


def test_covered_step_survives_a_simultaneous_scope_grant(tmp_path: Path) -> None:
    """A browse-covered step must still pass act while a ScopeGrant is active.

    Regression for a suspected binding mismatch: the gate records a browse-grant
    target binding with an empty scope id, while act consumes it with the active
    ScopeGrant's confirmation id.  Both grants are real registries and the real
    gate/act nodes run - only the browser is faked.
    """
    ctx = _ctx(_FakeSession())
    # A task-scope grant for the same task, as if an earlier scope prompt had
    # been approved (its plan hash may be stale; only its id matters here).
    ctx.scope_registry.issue(
        "t1",
        plan_hash="scope-plan-hash",
        confirmation_id="scope-cid-1",
        allowed_domains=["example.com"],
        max_actions=1,
    )
    assert ctx.scope_registry.active_grant("t1") is not None
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        out, payloads = _run(ctx, saver, [_browse(), _goto(), _done("Browsed.")])
        assert len(payloads) == 1, payloads
        assert payloads[0]["summary"] == BROWSE_PROMPT
        assert out.halted_reason is None, out.halted_reason
        assert out.final_answer == "Browsed."
        assert ctx.browse_grants.consumed("t1") == 1
    finally:
        saver.conn.close()




class _PressKeyDecisionEngine:
    """Deterministic engine double: one allowed, confirming Tier 1 decision.

    The engine's own tier logic is covered by the policy tests; this test is
    about the grant/host boundary, so the decision is fixed and needs_confirm
    is set exactly as it is for a browser Tier 1 step outside a grant.
    """

    def decide(self, step: Any, pctx: Any) -> Decision:
        return Decision(
            step_id=str(getattr(step, "id", "") or ""),
            tier=1,
            allowed=True,
            needs_confirm=True,
            needs_unlock=False,
            summary="Press Enter",
            action_hash="a" * 64,
        )


class _LiveHostSession:
    """A fake page that reports where the browser is *now* and records presses."""

    def __init__(self, live_url: str) -> None:
        self.live_url = live_url
        self.pressed: list[str] = []

    def current_url(self) -> str:
        return self.live_url

    def press_key(self, key: str) -> dict[str, Any]:
        self.pressed.append(str(key))
        return {"key": str(key)}

    def press(self, key: str) -> dict[str, Any]:
        return self.press_key(key)


def _press_key_state() -> dict[str, Any]:
    """State for a press-key step whose previous typing happened on example.com."""
    step = Step(
        id="s1", tool="browser_press_key", args={"key": "Enter"}, rationale="press enter"
    )
    return {
        "plan": Plan(goal="press enter", steps=[step]),
        "step_index": 0,
        "task_id": "t1",
        "specialist_phase": "decide",
        "specialist_history": [
            {
                "tool": "browser_type",
                "args-summary": "type into searchbox",
                "ok": True,
                "role": "searchbox",
                "host": "example.com",
            }
        ],
        "approved_hashes": [],
        "gated_resolved_paths": {},
        "gated_content_hashes": {},
        "gated_browser_readback_hashes": {},
        "results": [],
        "user_input": "search then press enter",
        "source": "text",
    }


def test_enter_after_navigation_is_not_covered_by_a_grant_for_the_previous_host() -> None:
    """A grant for host A cannot authorize Enter while the live page is on host B.

    Unit level on purpose: no graph, no FakeLLM, no checkpointer.  The grant was
    issued for the host the search text was typed into, and the live page has
    since moved to another host, so coverage must be withheld and act must
    dispatch zero key presses.
    """
    session = _LiveHostSession("https://other.example/page")
    settings = Settings(
        agent=AgentSettings(use_specialists=True, specialist_max_calls=10),
        browser=BrowserSettings(
            allowed_domains=["example.com", "other.example"], task_grant_enabled=True
        ),
    )
    ctx = make_app_context(
        settings,
        registry=registry_with(make_browser_press_key_spec()),
        browser=session,
    )
    ctx.engine = _PressKeyDecisionEngine()
    grant = ctx.browse_grants.issue("t1", host="example.com")
    assert grant.covers_host("example.com") is True

    state = _press_key_state()
    step = state["plan"].steps[0]
    decision = ctx.engine.decide(step, None)

    # The grant must not cover the action: the live page is on another host.
    covered, host, reason = browse_grant_check(
        state,
        ctx,
        step=step,
        decision=decision,
        resolved=SimpleNamespace(host="other.example", target=None),
    )
    assert covered is None
    assert host == "other.example"
    assert reason == "cross_domain"

    update = act(state, ctx)
    results = update.get("results") or []
    assert results and results[-1].ok is False
    assert results[-1].step_id == step.id
    # The security boundary: nothing was pressed on the unapproved host.
    assert session.pressed == []
    assert ctx.browse_grants.consumed("t1") == 0
    assert ctx.browse_grants.active_for_task("t1") is not None

def test_daemon_end_task_runtime_revokes_the_grant() -> None:
    ctx = _ctx(_FakeSession())
    ctx.browse_grants.issue("t1", host="example.com")
    assert ctx.browse_grants.active_for_task("t1") is not None
    DaemonServer._end_task_runtime(SimpleNamespace(_ctx=ctx), "t1")  # type: ignore[arg-type]
    assert ctx.browse_grants.active_for_task("t1") is None
    assert ctx.browse_grants.validate("t1").reason == "revoked"
