"""Wiring tests: browsing grant in policy_gate / act / daemon revocation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from tests.support import approve, deny
from tests.unit.test_specialist_loop_e2e import _FakeSession

from jarvis.agent.context import make_app_context
from jarvis.agent.graph import open_sqlite_checkpointer
from jarvis.agent.runner import resume_task, run_task
from jarvis.agent.schemas import BrainDecision, SpecialistDecision
from jarvis.browser.context import ElementRef, Observation
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


def test_daemon_end_task_runtime_revokes_the_grant() -> None:
    ctx = _ctx(_FakeSession())
    ctx.browse_grants.issue("t1", host="example.com")
    assert ctx.browse_grants.active_for_task("t1") is not None
    DaemonServer._end_task_runtime(SimpleNamespace(_ctx=ctx), "t1")  # type: ignore[arg-type]
    assert ctx.browse_grants.active_for_task("t1") is None
    assert ctx.browse_grants.validate("t1").reason == "revoked"
