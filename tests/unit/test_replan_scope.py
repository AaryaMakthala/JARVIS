"""Stage 7 re-plan interactions with the task-scope front end.

These are the graph-level cases named by the V2.2 roadmap that were not yet
covered end to end.  They drive the **real compiled graph** (``run_task`` /
``resume_task`` over a file-backed SQLite checkpointer) with the real browser
tool specs and a fake managed-browser session:

* an **in-scope** re-plan (the replacement stays on the approved domain) is
  re-authorized with a fresh confirmation and *continues*; the action counter is
  inherited, not reset;
* an **out-of-scope** re-plan (a new domain) needs its own confirmation and the
  old grant never covers the new domain;
* a popup ``Allow`` control still gets its own fresh action confirmation
  *after* the scope approval (it never rides on the scope alone);
* an unauthorized **cross-origin redirect** encountered while running a
  re-planned step stops the task before any interaction with the new origin;
* the action **counter survives** a re-plan (a replacement action over the
  lifetime cap is refused);
* a **stale approval hash** from the pre-re-plan scope prompt cannot approve the
  post-re-plan scope prompt.

The plan hash is refreshed on every re-plan (docs/03:236; T4), so a changed plan
can never be approved by a stale answer; these tests assert exactly that while
proving the task still *continues* when the replacement is inside scope.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jarvis.agent.context import make_app_context
from jarvis.agent.graph import open_sqlite_checkpointer
from jarvis.agent.runner import resume_task, run_task
from jarvis.agent.schemas import ActionIntent
from jarvis.agent.state import Plan, Step
from jarvis.browser.context import BrowserError, ElementRef, Observation
from jarvis.browser.tools import (
    make_browser_click_spec,
    make_browser_goto_spec,
    make_browser_read_spec,
)
from jarvis.config import AgentSettings, BrowserSettings, Settings
from jarvis.llm.client import FakeLLM
from support import approve, brain_steps, make_spec, registry_with, replan_continue

_ORIGIN = "https://example.com"
_FRESH = "https://news.example"
_FRESH_HOST = "news.example"
_OTHER = "https://other.example"
_OTHER_HOST = "other.example"


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


class RedirectSession(FakeSession):
    """A session that blocks all interaction after an unauthorized redirect.

    Mirrors the real driver: a navigation that lands on an origin outside the
    approved scope raises :class:`BrowserError` and sets an internal block so no
    later read/click can run against the wrong page (docs/03 B4/D18; I5).
    """

    def __init__(self) -> None:
        super().__init__()
        self.blocked = False

    def _guard(self) -> None:
        if self.blocked:
            raise BrowserError("interactions are blocked after an unauthorized redirect")

    def goto(self, url: str) -> dict[str, object]:
        self.calls.append(("goto", url))
        if url.endswith("/redir"):
            self.blocked = True
            raise BrowserError(
                "navigation redirected to unauthorized origin 'https://evil.example'"
            )
        self._guard()
        return {"url": url, "origin": _ORIGIN, "title": "Example"}

    def read(self, max_elements: int = 80) -> Observation:
        self._guard()
        return super().read(max_elements)

    def click(self, ref: str) -> dict[str, object]:
        self._guard()
        return super().click(ref)


def _element(ref: str, name: str, *, role: str = "button", origin: str = _ORIGIN) -> ElementRef:
    return ElementRef(ref=ref, role=role, name=name, origin=origin)


def _action(tool: str, args: dict[str, Any]) -> ActionIntent:
    return ActionIntent(tool=tool, args=args, rationale="do it")


def _fail_spec(name: str = "fake_fail") -> Any:
    return make_spec(name, base_tier=0, record=[], run_ok=False)


def _is_scope(payload: dict[str, Any]) -> bool:
    return payload.get("scope_confirm") is True


def _failing_browse_plan(first_url: str = f"{_FRESH}/story") -> Any:
    """The initial plan: browse a page, then fail (forces a re-plan)."""
    return brain_steps(
        [
            _action("browser_goto", {"url": first_url}),
            _action("fake_fail", {"text": "boom"}),
        ]
    )


def _ctx(settings: Settings, session: FakeSession, script: list[Any], *extra: Any) -> Any:
    """Build an AppContext that drives the real graph with a scripted FakeLLM."""
    return make_app_context(
        settings,
        llm=FakeLLM(script),
        registry=registry_with(
            make_browser_goto_spec(),
            make_browser_click_spec(),
            make_browser_read_spec(),
            *extra,
        ),
        browser=session,
    )


# ── in-scope re-plan: re-authorized, then continues; counter inherited ─────


def test_in_scope_replan_continues(tmp_path: Path) -> None:
    session = FakeSession()
    settings = Settings(agent=AgentSettings(max_retries_per_step=0, max_replans=1))
    replacement = Plan(
        goal="try another page on the same site",
        steps=[Step(id="r1", tool="browser_goto", args={"url": f"{_FRESH}/other"}, rationale="x")],
    )
    ctx = _ctx(
        settings, session, [_failing_browse_plan(), replan_continue(replacement)], _fail_spec()
    )
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        first = run_task(ctx, saver, "read the news then fail")
        scope1 = first.confirmation or {}
        assert _is_scope(scope1) is True and scope1.get("scope_domain") == _FRESH_HOST
        hash1 = str(scope1["action_hash"])

        second = resume_task(ctx, saver, first.task_id, approve(scope1))
        grant1 = ctx.scope_registry.active_grant(first.task_id)
        scope2 = second.confirmation or {}
        assert second.interrupted is True, "the re-planned step must be re-authorized"
        assert _is_scope(scope2) is True, f"expected a same-domain scope prompt, got {scope2!r}"
        assert scope2.get("scope_domain") == _FRESH_HOST
        assert scope2.get("step_id") == "r1"
        assert str(scope2["action_hash"]) != hash1, "a re-plan refreshes the plan hash"
        assert ("goto", f"{_FRESH}/other") not in session.calls, "nothing runs before approval"
        assert ctx.scope_registry.consumed(first.task_id) == 1, "counter inherited across replan"

        third = resume_task(ctx, saver, first.task_id, approve(scope2))
        assert third.interrupted is False, third.final_answer
        assert ("goto", f"{_FRESH}/other") in session.calls
        assert ctx.scope_registry.consumed(first.task_id) == 2, "the re-planned action counts"
        grant2 = ctx.scope_registry.active_grant(first.task_id)
        assert grant1 is not None and _FRESH_HOST in grant1.allowed_domains
        assert grant2 is not None and _FRESH_HOST in grant2.allowed_domains
        assert grant2.confirmation_id != grant1.confirmation_id, "a new scope needs a new id"
    finally:
        saver.conn.close()


# ── out-of-scope re-plan: a new domain needs its own confirmation ──────────


def test_out_of_scope_replan_needs_new_confirmation_id(tmp_path: Path) -> None:
    session = FakeSession()
    settings = Settings(agent=AgentSettings(max_retries_per_step=0, max_replans=1))
    replacement = Plan(
        goal="go to a different site",
        steps=[Step(id="r1", tool="browser_goto", args={"url": f"{_OTHER}/x"}, rationale="x")],
    )
    ctx = _ctx(
        settings, session, [_failing_browse_plan(), replan_continue(replacement)], _fail_spec()
    )
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        first = run_task(ctx, saver, "read the news then fail")
        scope1 = first.confirmation or {}
        assert _is_scope(scope1) is True and scope1.get("scope_domain") == _FRESH_HOST
        hash1 = str(scope1["action_hash"])

        second = resume_task(ctx, saver, first.task_id, approve(scope1))
        grant1 = ctx.scope_registry.active_grant(first.task_id)
        scope2 = second.confirmation or {}
        assert second.interrupted is True
        assert _is_scope(scope2) is True
        assert scope2.get("scope_domain") == _OTHER_HOST, "the new domain needs its own scope"
        assert str(scope2["action_hash"]) != hash1
        assert grant1 is not None and grant1.contains_domain(_OTHER_HOST) is False
        assert ("goto", f"{_OTHER}/x") not in session.calls

        third = resume_task(ctx, saver, first.task_id, approve(scope2))
        assert third.interrupted is False, third.final_answer
        assert ("goto", f"{_OTHER}/x") in session.calls
        grant2 = ctx.scope_registry.active_grant(first.task_id)
        assert grant2 is not None
        assert grant2.confirmation_id != grant1.confirmation_id
        assert grant2.contains_domain(_OTHER_HOST) is True
    finally:
        saver.conn.close()


# ── popup "Allow" still re-gates after the scope approval ──────────────────


def test_popup_allow_regates(tmp_path: Path) -> None:
    session = FakeSession(elements={"e1": _element("e1", "Allow", origin=_FRESH)})
    click = brain_steps([_action("browser_click", {"ref": "e1"})])
    ctx = _ctx(Settings(), session, [click])
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        first = run_task(ctx, saver, "allow it")
        scope = first.confirmation or {}
        assert _is_scope(scope) is True, f"expected a scope prompt, got {scope!r}"

        second = resume_task(ctx, saver, first.task_id, approve(scope))
        action = second.confirmation or {}
        assert second.interrupted is True, "an Allow control needs its own fresh confirmation"
        assert _is_scope(action) is False
        assert action.get("type") == "confirm"
        assert action.get("step_id") == "s1"
        assert ("click", "e1") not in session.calls, "nothing may click before the action approval"

        third = resume_task(ctx, saver, first.task_id, approve(action))
        assert third.interrupted is False, third.final_answer
        assert ("click", "e1") in session.calls
    finally:
        saver.conn.close()


# ── cross-origin redirect during a re-plan stops the task ──────────────────


def test_cross_origin_redirect_during_replan_stops(tmp_path: Path) -> None:
    session = RedirectSession()
    settings = Settings(
        browser=BrowserSettings(allowed_domains=["example.com"]),
        agent=AgentSettings(max_retries_per_step=0, max_replans=1),
    )
    replacement = Plan(
        goal="retry via a redirected page",
        steps=[
            Step(id="r1", tool="browser_goto", args={"url": f"{_ORIGIN}/redir"}, rationale="x"),
            Step(id="r2", tool="browser_read", args={"max_elements": 5}, rationale="observe"),
        ],
    )
    ctx = _ctx(
        settings,
        session,
        [_failing_browse_plan(f"{_ORIGIN}/page"), replan_continue(replacement)],
        _fail_spec(),
    )
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        first = run_task(ctx, saver, "open the page then fail")
        assert first.interrupted is True
        assert _is_scope(first.confirmation or {}) is False, "a config domain needs no scope"

        second = resume_task(ctx, saver, first.task_id, approve(first.confirmation or {}))
        goto_confirm = second.confirmation or {}
        assert second.interrupted is True, "the re-planned navigation still confirms"
        assert _is_scope(goto_confirm) is False
        assert goto_confirm.get("step_id") == "r1"

        third = resume_task(ctx, saver, first.task_id, approve(goto_confirm))
        assert third.interrupted is False, "the redirect must stop the task"
        assert "could not complete" in third.final_answer.lower()
        assert session.blocked is True
        assert ("goto", f"{_ORIGIN}/redir") in session.calls
        # Fail closed: no read/click ever ran after the unauthorized redirect.
        assert all(c[0] not in ("read", "click") for c in session.calls)
    finally:
        saver.conn.close()


# ── the action counter survives a re-plan ──────────────────────────────────


def test_counter_survives_replan(tmp_path: Path) -> None:
    # A config-allowed domain isolates the shared task budget (checked in ``act``
    # by ``reserve``) from the scope-grant path.
    session = FakeSession()
    settings = Settings(
        browser=BrowserSettings(allowed_domains=[_FRESH_HOST], task_scope_max_actions=2),
        agent=AgentSettings(max_retries_per_step=0, max_replans=1),
    )
    replacement = Plan(
        goal="two more pages on the same site",
        steps=[
            Step(id="r1", tool="browser_goto", args={"url": f"{_FRESH}/a"}, rationale="x"),
            Step(id="r2", tool="browser_read", args={"max_elements": 5}, rationale="observe"),
            Step(id="r3", tool="browser_goto", args={"url": f"{_FRESH}/b"}, rationale="y"),
        ],
    )
    ctx = _ctx(
        settings, session, [_failing_browse_plan(), replan_continue(replacement)], _fail_spec()
    )
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        first = run_task(ctx, saver, "read the news then fail")
        confirm1 = first.confirmation or {}
        assert _is_scope(confirm1) is False and confirm1.get("step_id") == "s1"

        second = resume_task(ctx, saver, first.task_id, approve(confirm1))
        confirm2 = second.confirmation or {}
        assert second.interrupted is True
        assert _is_scope(confirm2) is False and confirm2.get("step_id") == "r1"
        assert ("goto", f"{_FRESH}/story") in session.calls
        assert ctx.scope_registry.consumed(first.task_id) == 1, "the counter is charged per action"

        third = resume_task(ctx, saver, first.task_id, approve(confirm2))
        confirm3 = third.confirmation or {}
        assert third.interrupted is True
        assert _is_scope(confirm3) is False and confirm3.get("step_id") == "r3"
        assert ("goto", f"{_FRESH}/a") in session.calls
        assert ctx.scope_registry.consumed(first.task_id) == 2, "counter inherited across replans"

        fourth = resume_task(ctx, saver, first.task_id, approve(confirm3))
        assert fourth.interrupted is False
        assert "action limit" in (fourth.halted_reason or "").lower()
        assert ctx.scope_registry.consumed(first.task_id) == 2, "the counter never resets"
        assert ("goto", f"{_FRESH}/b") not in session.calls, "the capped action must not run"
    finally:
        saver.conn.close()


# ── a stale approval hash is rejected across a re-plan ─────────────────────


def test_stale_hash_rejected_after_replan(tmp_path: Path) -> None:
    session = FakeSession()
    settings = Settings(agent=AgentSettings(max_retries_per_step=0, max_replans=1))
    replacement = Plan(
        goal="try another page",
        steps=[Step(id="r1", tool="browser_goto", args={"url": f"{_FRESH}/other"}, rationale="x")],
    )
    ctx = _ctx(
        settings, session, [_failing_browse_plan(), replan_continue(replacement)], _fail_spec()
    )
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        first = run_task(ctx, saver, "read the news then fail")
        scope1 = first.confirmation or {}
        assert _is_scope(scope1) is True
        stale = approve(scope1)

        second = resume_task(ctx, saver, first.task_id, approve(scope1))
        grant_before = ctx.scope_registry.active_grant(first.task_id)
        scope2 = second.confirmation or {}
        assert second.interrupted is True
        assert _is_scope(scope2) is True
        assert str(scope2["action_hash"]) != str(scope1["action_hash"])

        # Answer the *new* scope prompt with the *old* plan's approval hash.
        third = resume_task(ctx, saver, first.task_id, stale)
        grant_after = ctx.scope_registry.active_grant(first.task_id)
        assert third.interrupted is False
        assert third.halted_reason, "a stale hash must never authorize the new plan"
        assert ("goto", f"{_FRESH}/other") not in session.calls
        assert grant_before is not None and grant_after is not None
        assert grant_after.confirmation_id == grant_before.confirmation_id, "no new grant issued"
    finally:
        saver.conn.close()
