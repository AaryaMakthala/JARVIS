"""Stage 7 Slice 10: daemon wiring of the managed-browser runtime services.

These tests prove the *daemon layer* (``jarvis.daemon.server``) owns and
threads the Stage 7 runtime services the graph already consumes:

* the managed browser arrives as a lazy, daemon-lived, single session adapter
  (``_LazyBrowserSession``) that starts only on the first call that needs a
  page, fails closed when Chrome cannot start, and never lets a browser
  failure take down non-browser tasks or the daemon itself;
* scope grants and target bindings live on the shared daemon context keyed by
  ``task_id``, survive a run -> confirm -> resume across graph/IPC boundaries,
  are atomic per-task (fresh budget per task, One Session reused), and are
  cleaned up when a task ends;
* the checkpoint carries none of these runtime services; and no client message
  can inject one.

Deterministic: fake sessions, an in-memory scope registry, and ``FakeLLM``
only - no network, no Playwright, no Windows GUI, no live desktop automation.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from tests.support import approve, brain_conversation, brain_steps, registry_with

from jarvis.agent.context import make_app_context
from jarvis.agent.graph import open_sqlite_checkpointer
from jarvis.agent.runner import resume_task, run_task
from jarvis.agent.schemas import ActionIntent
from jarvis.agent.state import AgentState, Step
from jarvis.browser.context import BrowserError, ElementRef, Observation
from jarvis.browser.tools import (
    make_browser_click_spec,
    make_browser_goto_spec,
    make_browser_read_spec,
    make_browser_type_spec,
)
from jarvis.config import BrowserSettings, Settings, VoiceSettings
from jarvis.daemon.confirmations import plan_hash
from jarvis.daemon.protocol import ConfirmResponse
from jarvis.daemon.server import (
    ClientConnection,
    DaemonServer,
    TaskSlot,
    _LazyBrowserSession,
    _parse_message,
)
from jarvis.llm.client import FakeLLM
from jarvis.policy.engine import PolicyContext, PolicyEngine
from jarvis.tools.registry import build_default_registry

_ORIGIN = "https://example.com"
_FRESH = "https://news.example"
_FRESH_HOST = "news.example"


# ── fakes ──────────────────────────────────────────────────────────────────


class _FakeSession:
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


def _browser_registry() -> Any:
    return registry_with(
        make_browser_goto_spec(),
        make_browser_click_spec(),
        make_browser_type_spec(),
        make_browser_read_spec(),
    )


def _daemon_ctx(decisions: list[Any], acquire: Any) -> Any:
    """A context shaped like ``DaemonServer._build_default_context``.

    The browser arrives through the daemon's lazy adapter (the same wiring the
    real ``_build_default_context`` installs) so the tests prove the adapter's
    laziness and fail-closed behaviour, not just the session itself.
    """
    settings = Settings(browser=BrowserSettings(allowed_domains=[]))
    return make_app_context(
        settings,
        llm=FakeLLM(decisions),
        registry=_browser_registry(),
        browser=_LazyBrowserSession(acquire),
    )


def _active_acquire(session: _FakeSession, started: list[str]) -> Any:
    def acquire() -> Any:
        started.append("acquire")
        return session

    return acquire


class _FakeStore:
    def __init__(self) -> None:
        self._kv: dict[str, str] = {"ipc_token": "test-token-123"}

    def get(self, name: str) -> str | None:
        return self._kv.get(name)

    def set(self, name: str, value: str) -> None:
        self._kv[name] = value

    def has(self, name: str) -> bool:
        return name in self._kv

    def check_store_access(self) -> str:
        return "FakeStore"


def _server() -> DaemonServer:
    return DaemonServer(settings=Settings(voice=VoiceSettings(enabled=False)), store=_FakeStore())


# ── the lazy adapter itself ────────────────────────────────────────────────


class TestLazyBrowserSession:
    def test_delegates_to_the_acquired_session_without_starting_one(self) -> None:
        session = _FakeSession()
        lazy = _LazyBrowserSession(lambda: session)
        assert session.calls == [], "building the adapter must not touch a browser"
        assert lazy.resolve("e1").ref == "e1"
        assert lazy.read().title == "Example"
        assert lazy.goto(_FRESH)["url"] == _FRESH
        assert lazy.type_text("e1", "hi")["characters"] == 2
        assert [call[0] for call in session.calls] == ["resolve", "read", "goto", "type_text"]

    def test_fails_closed_when_no_session_is_available(self) -> None:
        lazy = _LazyBrowserSession(lambda: None)
        for call in (
            lambda: lazy.goto(_FRESH),
            lambda: lazy.read(),
            lambda: lazy.resolve("e1"),
            lambda: lazy.click("e1"),
            lambda: lazy.type_text("e1", "x"),
        ):
            with pytest.raises(BrowserError, match="not available"):
                call()

    def test_close_without_a_started_browser_is_a_noop(self) -> None:
        lazy = _LazyBrowserSession(lambda: None)
        lazy.close()  # must not raise


# ── the daemon's browser lifecycle ─────────────────────────────────────────


class TestDaemonBrowserLifecycle:
    def test_browser_start_is_lazy_and_reuses_one_session(self, monkeypatch: Any) -> None:
        made: list[Any] = []

        class _Stub:
            def __init__(self) -> None:
                self.started = False
                self.closed: list[str] = []

            def start(self) -> None:
                self.started = True

            def close(self) -> None:
                self.closed.append("close")

        def _factory(settings: Any = None, profile: str = "general") -> _Stub:
            stub = _Stub()
            made.append(stub)
            return stub

        monkeypatch.setattr("jarvis.daemon.server.PlaywrightBrowser", _factory)
        server = _server()
        assert server._browser is None, "construction must not start a browser"

        first = server._browser_start()
        assert first is not None and first.started is True
        assert len(made) == 1, "the browser is launched only when first needed"

        second = server._browser_start()
        assert second is first, "the daemon reuses its single session"
        assert len(made) == 1

        server._close_browser()
        assert server._browser is None
        assert first.closed == ["close"]
        server._close_browser()  # idempotent, no error

    def test_browser_start_failure_fails_closed(self, monkeypatch: Any) -> None:
        class _Bad:
            started = False

            def start(self) -> None:
                raise RuntimeError("no chrome here")

        def _factory(settings: Any = None, profile: str = "general") -> _Bad:
            return _Bad()

        monkeypatch.setattr("jarvis.daemon.server.PlaywrightBrowser", _factory)
        server = _server()
        assert server._browser_start() is None, "a failed start returns None (fail closed)"
        assert server._browser is None, "the failed session is dropped, not kept"
        assert server._browser_start() is None, "a retry fails the same way, never crashes"

        # Tools reached through the daemon context refuse cleanly, not loudly.
        lazy = _LazyBrowserSession(server._browser_start)
        with pytest.raises(BrowserError, match="not available"):
            lazy.goto(_FRESH)

    def test_status_request_never_starts_a_browser(self, monkeypatch: Any) -> None:
        starts: list[Any] = []

        def _factory(settings: Any = None, profile: str = "general") -> Any:
            stub = type("S", (), {"started": False, "start": lambda self: None})()
            starts.append(stub)
            return stub

        monkeypatch.setattr("jarvis.daemon.server.PlaywrightBrowser", _factory)
        server = _server()
        conn = ClientConnection(writer=AsyncMock(), reader=AsyncMock())
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(server._handle_status(conn))
        finally:
            loop.close()
        assert starts == [], "a health/status request must never launch the browser"
        assert server._browser is None

    def test_end_task_runtime_cleans_only_that_task(self) -> None:
        ctx = make_app_context(
            Settings(browser=BrowserSettings(allowed_domains=[])),
            registry=_browser_registry(),
            llm=FakeLLM([]),
        )
        ctx.scope_registry.issue(
            "t1", plan_hash="p1", confirmation_id="c1", allowed_domains=[_FRESH_HOST]
        )
        ctx.scope_registry.issue("t2", plan_hash="p1", confirmation_id="c2")
        ctx.target_bindings.put(
            "t1", "s1", action_hash="a", fingerprint="f", origin=_FRESH, scope_confirmation_id="c1"
        )

        server = _server()
        server._ctx = ctx
        server._end_task_runtime("t1")

        assert ctx.scope_registry.active_grant("t1") is None, "the finished task's grant is gone"
        assert ctx.target_bindings.get("t1", "s1") is None, "its bindings are gone"
        assert ctx.scope_registry.active_grant("t2") is not None, "other tasks are untouched"

        # Blank task ids and a missing context must never raise.
        server._ctx = None
        server._end_task_runtime("t1")
        server._end_task_runtime("")


# ── full graph round trips on the daemon-shaped context ────────────────────


class TestDaemonCtxGraphWiring:
    def test_scope_grant_survives_resume_but_not_task_boundary(self, tmp_path: Path) -> None:
        session = _FakeSession()
        started: list[str] = []
        decisions = [
            brain_steps(
                [ActionIntent(tool="browser_goto", args={"url": f"{_FRESH}/story"}, rationale="go")]
            ),
            brain_steps(
                [ActionIntent(tool="browser_goto", args={"url": f"{_FRESH}/more"}, rationale="go")]
            ),
        ]
        ctx = _daemon_ctx(decisions, _active_acquire(session, started))
        saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
        try:
            # Task 1: a fresh domain must earn a scope approval first.
            first = run_task(ctx, saver, "read the news", thread_id="task1")
            assert first.interrupted is True
            p1 = first.confirmation or {}
            assert p1.get("scope_confirm") is True
            assert p1.get("scope_domain") == _FRESH_HOST
            assert session.calls == [], "nothing runs before approval"
            assert started == [], "the daemon browser stays untouched before the first action"

            # Approval survives the run -> confirm -> resume boundary, then the
            # action runs on the daemon's lazy session and charges its budget.
            second = resume_task(ctx, saver, first.task_id, approve(p1))
            assert second.interrupted is False
            assert ("goto", f"{_FRESH}/story") in session.calls
            assert len(started) == 1, "the action launched the one daemon browser"
            grant = ctx.scope_registry.active_grant("task1")
            assert grant is not None and _FRESH_HOST in grant.allowed_domains
            assert ctx.scope_registry.consumed("task1") == 1, "one state-changing action charged"
            assert not ctx.settings.browser.allowed_domains, "grants never mutate static config"
            assert ctx.target_bindings.get("task1", "s1") is None, (
                "act consumed the one-shot binding"
            )

            # Task 2, same shared daemon context: a *fresh* task grant, a fresh
            # budget, and the same session reused - task 1's grant is untouched.
            first2 = run_task(ctx, saver, "read more news", thread_id="task2")
            assert ctx.scope_registry.active_grant("task2") is None, "no inherited authorization"
            p2 = first2.confirmation or {}
            assert p2.get("scope_confirm") is True
            assert ctx.scope_registry.active_grant("task1") is not None, "task1's grant lives on"
            prev = len(started)
            second2 = resume_task(ctx, saver, first2.task_id, approve(p2))
            assert second2.interrupted is False
            assert ("goto", f"{_FRESH}/more") in session.calls
            assert len(started) == prev + 1, "the same daemon session served both tasks"
            assert ctx.scope_registry.consumed("task2") == 1
            assert ctx.scope_registry.consumed("task1") == 1, "per-task counters never mix"
            assert ctx.scope_registry.active_grant("task2") is not None
            assert ctx.scope_registry.remaining("task2") == 7, "a fresh task gets a fresh budget"
        finally:
            saver.conn.close()

    def test_checkpoint_state_contains_no_runtime_services(self, tmp_path: Path) -> None:
        session = _FakeSession()
        started: list[str] = []
        ctx = _daemon_ctx(
            [
                brain_steps(
                    [
                        ActionIntent(
                            tool="browser_goto", args={"url": f"{_FRESH}/story"}, rationale="go"
                        )
                    ]
                )
            ],
            _active_acquire(session, started),
        )
        saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
        try:
            first = run_task(ctx, saver, "read the news", thread_id="task1")
            second = resume_task(ctx, saver, first.task_id, approve(first.confirmation or {}))
            allowed_keys = set(AgentState.__annotations__)
            assert set(second.state) <= allowed_keys, (
                "every checkpointed key is a declared AgentState key"
            )
            marshalled = json.dumps(second.state, default=str, sort_keys=True)
            for banned in ('"scope_registry"', '"target_bindings"', '"scope_confirmation_id"'):
                assert banned not in marshalled, f"runtime service {banned} reached the checkpoint"
        finally:
            saver.conn.close()

    def test_non_browser_task_never_starts_the_browser(self, tmp_path: Path) -> None:
        session = _FakeSession()
        started: list[str] = []
        ctx = _daemon_ctx([brain_conversation()], _active_acquire(session, started))
        saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
        try:
            outcome = run_task(ctx, saver, "say hello", thread_id="task1")
            assert outcome.interrupted is False
            assert outcome.final_answer
            assert started == [], "a conversation task must never launch the browser"
            assert session.calls == []
        finally:
            saver.conn.close()


# ── a client can never inject a grant or binding ───────────────────────────


class TestClientCannotInject:
    def test_parse_confirm_drops_unknown_injection_fields(self) -> None:
        raw = json.dumps(
            {
                "type": "confirm_response",
                "task_id": "t1",
                "approved": True,
                "action_hash": "h",
                "scope_confirmation_id": "injected-grant-cid",
                "approved_domains": ["evil.example"],
                "max_actions": 999,
                "fingerprint": "injected",
            }
        )
        msg = _parse_message(raw)
        assert isinstance(msg, ConfirmResponse)
        assert msg.approved is True and msg.action_hash == "h"
        for field in ("scope_confirmation_id", "approved_domains", "max_actions", "fingerprint"):
            assert not hasattr(msg, field), f"client-supplied {field!r} must be dropped"

    def test_handle_confirm_resume_answer_never_injects_grants(self) -> None:
        server = _server()
        server._ctx = make_app_context(
            Settings(browser=BrowserSettings(allowed_domains=[])),
            registry=_browser_registry(),
            llm=FakeLLM([]),
        )
        action_hash = "a" * 64
        pending = server._confirmations.issue("t1", "s1", 1, action_hash, plan_hash([action_hash]))
        slot = TaskSlot(task_id="t1", text="x", source="terminal")
        server._active = slot
        conn = ClientConnection(writer=AsyncMock(), reader=AsyncMock())
        msg = ConfirmResponse(
            task_id="t1",
            approved=True,
            action_hash=action_hash,
            confirmation_id=pending.confirmation_id,
            plan_hash=pending.plan_hash,
        )
        # The wire answer may try to smuggle extra fields; the daemon answers
        # with only what the gate itself built.
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(server._handle_confirm(msg, conn))
        finally:
            loop.close()

        answer = slot.resume_answer or {}
        assert answer.get("approved") is True
        assert answer.get("action_hash") == action_hash
        for field in ("scope_confirmation_id", "approved_domains", "max_actions", "fingerprint"):
            assert field not in answer, "the resume answer must not spread client-supplied fields"
        # The daemon handler itself must never create scope/binding state.
        assert server._ctx.scope_registry.active_grant("t1") is None
        assert server._ctx.target_bindings.get("t1", "s1") is None


# ── the default registry carries the Stage 7 tools ─────────────────────────


class TestRegistryCarriesStage7Tools:
    def test_default_registry_contains_browser_and_uia_specs(self) -> None:
        registry = build_default_registry(Settings())
        expected = {
            "browser_goto": (1, 1, False),
            "browser_read": (0, 0, False),
            "browser_click": (1, 1, False),
            "browser_type": (1, 1, False),
            "ui_list_controls": (0, 0, True),
            "ui_click": (1, 1, True),
        }
        for name, (tier, cost, windows_only) in expected.items():
            spec = registry.get(name)
            assert spec.base_tier == tier, name
            assert spec.budget_cost == cost, name
            assert spec.windows_only is windows_only, name

    def test_engine_never_decides_below_base_tier_for_new_tools(self) -> None:
        registry = build_default_registry(Settings())
        engine = PolicyEngine()
        pctx = PolicyContext(registry=registry, settings=Settings())
        for name in (
            "browser_goto",
            "browser_read",
            "browser_click",
            "browser_type",
            "ui_list_controls",
            "ui_click",
        ):
            decision = engine.decide(Step(id="s1", tool=name, args={}, rationale="x"), pctx)
            assert decision.tier >= registry.get(name).base_tier, name
            assert decision.allowed == (decision.tier < 3), name
