from __future__ import annotations

from pathlib import Path
from typing import Any

from tests.support import approve, deny

from jarvis.agent.context import make_app_context
from jarvis.agent.graph import open_sqlite_checkpointer
from jarvis.agent.nodes.telemetry import task_status
from jarvis.agent.runner import resume_task, run_task
from jarvis.agent.schemas import BrainDecision, SpecialistDecision
from jarvis.browser.context import ElementRef, Observation
from jarvis.config import AgentSettings, BrowserSettings, Settings
from jarvis.llm.client import FakeLLM
from jarvis.tools.registry import build_default_registry


class _FakeSession:
    def __init__(self) -> None:
        self.resolved = {
            "e1": ElementRef(
                ref="e1",
                role="link",
                name="First",
                origin="https://example.com/page",
                input_type="",
                is_password=False,
            )
        }
        self.calls: list[tuple[object, ...]] = []
        self.observation = Observation(
            url="https://example.com/page",
            origin="https://example.com",
            title="Example",
            elements=tuple(self.resolved.values()),
        )

    def goto(self, url: str) -> dict[str, object]:
        self.calls.append(("goto", url))
        self.observation = Observation(
            url=url,
            origin="https://example.com",
            title="Example",
            elements=tuple(self.resolved.values()),
        )
        return {"url": url, "origin": "https://example.com", "title": "Example"}

    def read(self, max_elements: int = 80) -> Observation:
        self.calls.append(("read", max_elements))
        return self.observation

    def resolve(self, ref: str) -> ElementRef | None:
        self.calls.append(("resolve", ref))
        return self.resolved.get(ref)

    def find(self, query: str, *, role: str = "", limit: int = 300) -> list[ElementRef]:
        self.calls.append(("find", query, role, limit))
        return list(self.resolved.values())

    def open_match(
        self,
        query: str,
        *,
        role: str = "",
        index: int | None = None,
        expect_identity: str | None = None,
    ) -> dict[str, object]:
        self.calls.append(("open_match", query, role, index, expect_identity))
        element = next(iter(self.resolved.values()))
        return {
            "ref": element.ref,
            "role": element.role,
            "name": element.name,
            "origin": element.origin,
        }

    def current_url(self) -> str:
        self.calls.append(("current_url",))
        return self.observation.url

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

    def close(self) -> None:
        self.calls.append(("close",))


def _browser_registry() -> Any:
    return build_default_registry(
        Settings(browser=BrowserSettings(allowed_domains=["example.com"]))
    )


def _run_to_completion(
    ctx: Any, saver: Any, text: str, *, max_rounds: int = 12
) -> Any:
    """Run a task, approving every confirmation, and return the final outcome."""
    out = run_task(ctx, saver, text, thread_id="t1")
    rounds = 0
    while out.interrupted and out.confirmation:
        rounds += 1
        assert rounds <= max_rounds, "too many confirmation rounds"
        out = resume_task(ctx, saver, out.task_id, approve(out.confirmation))
    return out


def _settings(**agent: Any) -> Settings:
    return Settings(
        agent=AgentSettings(use_specialists=True, **agent),
        browser=BrowserSettings(allowed_domains=["example.com"]),
    )


def _search(query: str, *, site: str = "Google") -> SpecialistDecision:
    return SpecialistDecision(
        kind="act",
        tool="browser_search",
        args={"site": site, "query": query},
        rationale="search",
    )


def test_specialist_loop_e2e_three_steps_once(tmp_path: Path) -> None:
    """Three decision steps run exactly once, accumulate results and finish."""
    session = _FakeSession()
    decisions = [
        BrainDecision(request_type="browse", goal="search example for test", actions=[]),
        _search("one"),
        _search("two"),
        SpecialistDecision(kind="act", tool="browser_read", args={}, rationale="read"),
        SpecialistDecision(kind="done", summary="Done searching."),
    ]
    llm = FakeLLM(decisions)
    ctx = make_app_context(
        _settings(specialist_max_calls=10),
        llm=llm,
        registry=_browser_registry(),
        browser=session,
    )
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        out = _run_to_completion(ctx, saver, "search example for test")
        assert out.interrupted is False
        res = out.state.get("results") or []
        ids = [getattr(r, "step_id", None) for r in res]
        assert ids == ["s0", "s1", "s2"], ids
        assert len(ids) == len(set(ids))
        assert all(getattr(r, "ok", False) for r in res)
        assert out.final_answer == "Done searching."
        assert out.state.get("specialist_calls") == 3
        # One structured call per decision plus the brain classification.
        assert len(llm.calls) == 5
    finally:
        saver.conn.close()


def test_specialist_loop_adapts_after_a_failed_step(tmp_path: Path) -> None:
    """A failed step goes back to decide with the failure in the history."""
    session = _FakeSession()
    decisions = [
        BrainDecision(request_type="browse", goal="search example for test", actions=[]),
        _search("test", site="example"),
        _search("test"),
        SpecialistDecision(kind="done", summary="Recovered."),
    ]
    llm = FakeLLM(decisions)
    ctx = make_app_context(
        _settings(specialist_max_calls=10),
        llm=llm,
        registry=_browser_registry(),
        browser=session,
    )
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        out = _run_to_completion(ctx, saver, "search example for test")
        assert out.interrupted is False
        assert out.halted_reason is None
        res = out.state.get("results") or []
        ids = [getattr(r, "step_id", None) for r in res]
        assert ids == ["s0", "s1"], ids
        assert getattr(res[0], "ok", True) is False
        assert getattr(res[1], "ok", False) is True
        history = out.state.get("specialist_history") or []
        assert [entry.get("ok") for entry in history] == [False, True], history
        assert "unknown site" in str(history[0].get("note", ""))
        assert out.final_answer == "Recovered."
    finally:
        saver.conn.close()


def test_specialist_loop_recovers_from_a_gate_refusal(tmp_path: Path) -> None:
    """An unobservable target is recorded in history, not a task halt."""
    session = _FakeSession()
    decisions = [
        BrainDecision(request_type="browse", goal="open the first result", actions=[]),
        # No result set exists yet, so the gate cannot resolve the target.
        SpecialistDecision(
            kind="act", tool="browser_open_result", args={"position": 1}, rationale="open"
        ),
        _search("test"),
        SpecialistDecision(kind="done", summary="Recovered from refusal."),
    ]
    ctx = make_app_context(
        _settings(specialist_max_calls=10),
        llm=FakeLLM(decisions),
        registry=_browser_registry(),
        browser=session,
    )
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        out = _run_to_completion(ctx, saver, "open the first result")
        assert out.interrupted is False
        assert out.halted_reason is None
        history = out.state.get("specialist_history") or []
        assert len(history) == 2, history
        assert history[0]["tool"] == "browser_open_result"
        assert history[0]["ok"] is False
        assert "could not be reliably observed" in history[0]["note"]
        ids = [getattr(r, "step_id", None) for r in (out.state.get("results") or [])]
        assert ids == ["s1"], ids
        assert out.final_answer == "Recovered from refusal."
        # The refusal carries the synthetic TIER_BLOCKED resolution marker, yet
        # telemetry reports the task as completed: it derives the status from
        # results/halted_reason and never reads decision tiers.
        assert task_status(out.state) in ("completed", "partial")
    finally:
        saver.conn.close()


def test_specialist_loop_recovers_from_a_validate_rejection(tmp_path: Path) -> None:
    """A bad-args plan goes back to decide, capped and recorded."""
    session = _FakeSession()
    decisions = [
        BrainDecision(request_type="browse", goal="search broken then fixed", actions=[]),
        SpecialistDecision(
            kind="act", tool="browser_search", args={"site": "Google"}, rationale="broken"
        ),
        _search("test"),
        SpecialistDecision(kind="done", summary="Fixed the plan."),
    ]
    ctx = make_app_context(
        _settings(specialist_max_calls=10),
        llm=FakeLLM(decisions),
        registry=_browser_registry(),
        browser=session,
    )
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        out = _run_to_completion(ctx, saver, "search broken then fixed")
        assert out.interrupted is False
        assert out.halted_reason is None
        history = out.state.get("specialist_history") or []
        assert len(history) == 2, history
        assert history[0]["tool"] == "validate"
        assert history[0]["ok"] is False
        assert "invalid args" in history[0]["note"], history[0]
        ids = [getattr(r, "step_id", None) for r in (out.state.get("results") or [])]
        assert ids == ["s1"], ids
        assert out.final_answer == "Fixed the plan."
    finally:
        saver.conn.close()


def test_specialist_user_no_still_halts(tmp_path: Path) -> None:
    """A refused confirmation must stop the task, not feed the loop."""
    session = _FakeSession()
    decisions = [
        BrainDecision(request_type="browse", goal="search example", actions=[]),
        _search("test"),
        SpecialistDecision(kind="done", summary="Should never run."),
    ]
    ctx = make_app_context(
        _settings(specialist_max_calls=10),
        llm=FakeLLM(decisions),
        registry=_browser_registry(),
        browser=session,
    )
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        out = run_task(ctx, saver, "search example", thread_id="t1")
        assert out.interrupted is True and out.confirmation is not None
        out = resume_task(ctx, saver, out.task_id, deny(out.confirmation))
        assert out.interrupted is False
        assert out.halted_reason
        assert out.halted_reason.startswith("Refused"), out.halted_reason
        assert not (out.state.get("results") or [])
        assert out.state.get("specialist_phase") == "observe"
        assert not (out.state.get("specialist_history") or [])
    finally:
        saver.conn.close()


def test_specialist_loop_halts_at_the_call_cap(tmp_path: Path) -> None:
    """A script that never says done stops at specialist_max_calls."""
    session = _FakeSession()
    decisions = [
        BrainDecision(request_type="browse", goal="loop forever", actions=[]),
        _search("q1"),
        _search("q2"),
        _search("q3"),
        _search("q4"),  # must never be requested
    ]
    llm = FakeLLM(decisions)
    ctx = make_app_context(
        _settings(specialist_max_calls=3),
        llm=llm,
        registry=_browser_registry(),
        browser=session,
    )
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        out = _run_to_completion(ctx, saver, "loop forever")
        assert out.interrupted is False
        assert out.halted_reason == "Specialist call limit reached."
        res = out.state.get("results") or []
        ids = [getattr(r, "step_id", None) for r in res]
        assert ids == ["s0", "s1", "s2"], ids
        assert out.state.get("specialist_phase") == "idle"
        # brain + three decisions; the fourth act decision is never requested.
        assert len(llm.calls) == 4
    finally:
        saver.conn.close()
