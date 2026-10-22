"""Stage 8.6, slice 8.6.3: the page JARVIS is on survives into the next turn.

A browser step publishes the page it is on; the session context keeps it for a
short while, and a later request that names a page ("open that page again")
gets that URL as *untrusted data* in the brain prompt.

These tests use the real act node and the real browser tools with a fake
managed browser (no Playwright, no Chrome, no network) and assert:

* a successful navigation publishes the page;
* a failed navigation and a dry run publish nothing;
* the next turn's prompt carries the URL, wrapped as data;
* the page never leaks into the artifact store ("delete it" must not resolve to
  a URL) and never reaches the checkpoint state.
"""

from __future__ import annotations

from typing import Any

from tests.support import brain_conversation, registry_with

from jarvis.agent.context import make_app_context
from jarvis.agent.graph import _ALLOWED_MSGPACK_TYPES
from jarvis.agent.nodes.act import act
from jarvis.agent.nodes.brain import brain
from jarvis.agent.schemas import BrainDecision
from jarvis.agent.state import AgentState, Plan, Step
from jarvis.browser.tools import make_browser_goto_spec
from jarvis.config import BrowserSettings, Settings
from jarvis.llm.client import Usage
from jarvis.policy import rules
from jarvis.tools.base import resolve_target

_ORIGIN = "https://example.com"
_TASK = "t1"


class FakeSession:
    """A fake managed browser that only knows its current URL."""

    def __init__(self, url: str = f"{_ORIGIN}/start") -> None:
        self.url = url
        self.calls: list[tuple[object, ...]] = []
        self.goto_error: str | None = None

    def goto(self, url: str) -> dict[str, object]:
        self.calls.append(("goto", url))
        if self.goto_error:
            raise RuntimeError(self.goto_error)
        self.url = url
        return {"url": url, "origin": _ORIGIN, "title": "Example"}

    def current_url(self) -> str:
        return self.url

    def close(self) -> None:  # pragma: no cover - protocol completeness
        return None


def _ctx(session: FakeSession, *, dry_run: bool = False) -> Any:
    settings = Settings(browser=BrowserSettings(allowed_domains=["example.com"]))
    return make_app_context(
        settings,
        registry=registry_with(make_browser_goto_spec()),
        browser=session,
        dry_run=dry_run,
    )


def _step(url: str = f"{_ORIGIN}/page") -> Step:
    return Step(id="s1", tool="browser_goto", args={"url": url}, rationale="go there")


def _hash(ctx: Any, step: Step) -> str:
    spec = ctx.registry.get(step.tool)
    args = spec.args_model.model_validate(step.args)
    return str(rules.action_hash(spec, args))


def _state(step: Step, *, approved: list[str]) -> dict[str, Any]:
    return {
        "plan": Plan(goal="go there", steps=[step]),
        "step_index": 0,
        "task_id": _TASK,
        "approved_hashes": list(approved),
        "gated_resolved_paths": {},
        "results": [],
        "user_input": "go there",
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


def _navigate(session: FakeSession, *, dry_run: bool = False) -> tuple[Any, Step]:
    ctx = _ctx(session, dry_run=dry_run)
    step = _step()
    if not dry_run:
        _record_binding(ctx, step)
    act(_state(step, approved=[_hash(ctx, step)]), ctx)
    return ctx, step


def test_a_successful_navigation_publishes_the_page() -> None:
    session = FakeSession()
    ctx, _ = _navigate(session)
    resolved = ctx.session_context.resolve_page("local")
    assert resolved.kind == "resolved"
    assert resolved.artifact is not None
    assert resolved.artifact.url == f"{_ORIGIN}/page"


def test_a_failed_navigation_publishes_nothing() -> None:
    session = FakeSession()
    session.goto_error = "the managed browser session is not started"
    ctx, _ = _navigate(session)
    assert ctx.session_context.resolve_page("local").kind == "none"


def test_a_dry_run_publishes_nothing() -> None:
    session = FakeSession()
    ctx, _ = _navigate(session, dry_run=True)
    assert ctx.session_context.resolve_page("local").kind == "none"


def test_the_page_never_becomes_a_file_artifact() -> None:
    """'delete it' must not resolve to a URL."""
    session = FakeSession()
    ctx, _ = _navigate(session)
    assert ctx.session_context.resolve("local").kind == "none"


def test_the_page_never_reaches_the_checkpoint_state() -> None:
    assert "session_context" not in AgentState.__annotations__
    allowed = {name for _, name in _ALLOWED_MSGPACK_TYPES}
    assert not any("session" in name.lower() for name in allowed)


class _RecorderLLM:
    def __init__(self, decision: BrainDecision) -> None:
        self._decision = decision
        self.user: str | None = None

    def structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[Any],
        model_role: str = "planner",
        temperature: float = 0.0,
        max_retries: int | None = None,
        timeout_s: float | None = None,
    ) -> tuple[BrainDecision, Usage]:
        self.user = user
        return self._decision, Usage(calls=1, prompt_tokens=1, completion_tokens=1)


def _brain_ctx(recorder: _RecorderLLM, session: FakeSession, session_context: Any = None) -> Any:
    return make_app_context(
        Settings(browser=BrowserSettings(allowed_domains=["example.com"])),
        llm=recorder,
        registry=registry_with(make_browser_goto_spec()),
        browser=session,
        session_context=session_context,
    )


def test_the_next_turn_gets_the_page_as_untrusted_data() -> None:
    session = FakeSession()
    ctx, _ = _navigate(session)

    recorder = _RecorderLLM(brain_conversation("ok"))
    brain_ctx = _brain_ctx(recorder, session, session_context=ctx.session_context)
    brain({"user_input": "open that page again", "source": "terminal", "task_id": "t2"}, brain_ctx)

    assert recorder.user is not None
    assert f"{_ORIGIN}/page" in recorder.user
    assert "<untrusted_data>" in recorder.user
    assert ctx.session_context.resolve_page("local").kind == "resolved"


def test_a_request_without_a_page_reference_gets_no_block() -> None:
    session = FakeSession()
    recorder = _RecorderLLM(brain_conversation("ok"))
    brain_ctx = _brain_ctx(recorder, session)
    brain_ctx.session_context.note_page("local", f"{_ORIGIN}/page")
    brain({"user_input": "say hello", "source": "terminal", "task_id": "t2"}, brain_ctx)

    assert recorder.user is not None
    assert f"{_ORIGIN}/page" not in recorder.user
