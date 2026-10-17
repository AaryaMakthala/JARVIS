"""Stage 8 Slice 4a: the lazy, profile-scoped managed-browser holder.

Two things are proven here:

* :class:`~jarvis.browser.lazy.LazyProfileBrowser` starts a session only on the
  first :meth:`acquire`, reuses it, fails closed (returns ``None``) when a start
  is impossible, and closes exactly once; and
* the daemon wires a **second**, ``whatsapp``-profile holder - independent from
  the general profile - into the shared app context and closes it at shutdown,
  before any WhatsApp tool exists to use it.

Deterministic: fake sessions and fakes only - no Playwright, no network, no
Windows GUI, no live desktop automation.
"""

from __future__ import annotations

from typing import Any

import pytest

from jarvis.agent.context import make_app_context
from jarvis.browser.lazy import LazyProfileBrowser
from jarvis.config import Settings, VoiceSettings
from jarvis.daemon.server import DaemonServer


class _FakeSession:
    """A fake managed-browser session recording lifecycle calls."""

    def __init__(self) -> None:
        self.start_calls = 0
        self.close_calls = 0
        self.started = False

    def start(self) -> None:
        self.start_calls += 1
        self.started = True

    def close(self) -> None:
        self.close_calls += 1
        self.started = False


class _DeadSession:
    started = False

    def start(self) -> None:
        raise RuntimeError("no chrome here")

    def close(self) -> None:  # pragma: no cover - never reached
        raise AssertionError("a failed session must never be closed")


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


# ── the holder itself ─────────────────────────────────────────────────────


class TestLazyProfileBrowser:
    def test_unknown_profile_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown browser profile"):
            LazyProfileBrowser("nope")

    def test_constructing_never_starts_a_session(self) -> None:
        made: list[_FakeSession] = []

        def _factory() -> _FakeSession:
            made.append(_FakeSession())
            return made[-1]

        holder = LazyProfileBrowser("whatsapp", factory=_factory)
        assert made == [], "building the holder must not touch a browser"
        assert holder.started is False
        assert holder.profile == "whatsapp"

    def test_acquire_starts_once_and_reuses_the_session(self) -> None:
        session = _FakeSession()
        holder = LazyProfileBrowser("whatsapp", factory=lambda: session)

        first = holder.acquire()
        assert first is session
        assert session.start_calls == 1
        assert holder.started is True

        second = holder.acquire()
        assert second is session, "the holder reuses its one session"
        assert session.start_calls == 1, "a live session is never restarted"

    def test_acquire_fails_closed_when_start_raises(self) -> None:
        made: list[_DeadSession] = []

        def _factory() -> _DeadSession:
            made.append(_DeadSession())
            return made[-1]

        holder = LazyProfileBrowser("whatsapp", factory=_factory)
        assert holder.acquire() is None, "a failed start returns None (fail closed)"
        assert holder.started is False
        assert holder.acquire() is None, "a retry fails the same clean way, never crashes"
        assert len(made) == 2, "the failed session is dropped, not cached"

    def test_acquire_fails_closed_when_the_factory_raises(self) -> None:
        def _factory() -> _FakeSession:
            raise RuntimeError("cannot even build one")

        holder = LazyProfileBrowser("whatsapp", factory=_factory)
        assert holder.acquire() is None

    def test_close_closes_a_started_session_and_is_idempotent(self) -> None:
        session = _FakeSession()
        holder = LazyProfileBrowser("whatsapp", factory=lambda: session)
        holder.acquire()

        holder.close()
        assert session.close_calls == 1
        assert holder.started is False
        holder.close()  # must not raise or double-close
        assert session.close_calls == 1

    def test_close_without_a_started_session_is_a_noop(self) -> None:
        holder = LazyProfileBrowser("whatsapp", factory=lambda: _FakeSession())
        holder.close()  # must not raise

    def test_acquire_after_close_starts_a_fresh_session(self) -> None:
        sessions: list[_FakeSession] = []

        def _factory() -> _FakeSession:
            sessions.append(_FakeSession())
            return sessions[-1]

        holder = LazyProfileBrowser("whatsapp", factory=_factory)
        first = holder.acquire()
        holder.close()
        second = holder.acquire()
        assert second is not first
        assert len(sessions) == 2

    def test_default_factory_builds_a_session_for_this_profile(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}

        class _StubSession:
            started = False

            def start(self) -> None:
                self.started = True

            def close(self) -> None:  # pragma: no cover - protocol completeness
                self.started = False

        def _factory(settings: Any = None, profile: str = "general") -> _StubSession:
            seen["profile"] = profile
            return _StubSession()

        monkeypatch.setattr("jarvis.browser.lazy.PlaywrightBrowser", _factory)
        holder = LazyProfileBrowser("whatsapp", Settings())
        assert holder.acquire() is not None
        assert seen["profile"] == "whatsapp", "the holder binds the real session to its profile"


# ── daemon / context wiring (no WhatsApp tool exists yet) ──────────────────


def _server() -> DaemonServer:
    return DaemonServer(settings=Settings(voice=VoiceSettings(enabled=False)), store=_FakeStore())


class TestWhatsAppProfileWiring:
    def test_daemon_owns_one_unstarted_whatsapp_holder(self) -> None:
        server = _server()
        holder = server._whatsapp_browser
        assert isinstance(holder, LazyProfileBrowser)
        assert holder.profile == "whatsapp"
        assert holder.started is False, "constructing the daemon never starts Chrome"
        assert server._browser is None, "the general browser is still lazy too"

    def test_build_default_context_threads_the_whatsapp_holder(self) -> None:
        server = _server()
        server._open_memory = lambda: None  # type: ignore[method-assign]
        ctx = server._build_default_context()

        assert ctx.whatsapp_browser is server._whatsapp_browser
        assert ctx.whatsapp_browser.profile == "whatsapp"
        assert ctx.browser is not ctx.whatsapp_browser, (
            "the WhatsApp profile must never be the general browser session"
        )
        assert ctx.tool_context().whatsapp_browser is server._whatsapp_browser

    def test_make_app_context_threads_and_defaults_it(self) -> None:
        settings = Settings()
        assert make_app_context(settings).tool_context().whatsapp_browser is None
        holder = LazyProfileBrowser("whatsapp", settings)
        ctx = make_app_context(settings, whatsapp_browser=holder)
        assert ctx.whatsapp_browser is holder
        assert ctx.tool_context().whatsapp_browser is holder

    def test_close_whatsapp_browser_is_idempotent(self) -> None:
        session = _FakeSession()
        server = _server()
        server._whatsapp_browser = LazyProfileBrowser("whatsapp", factory=lambda: session)
        server._whatsapp_browser.acquire()

        server._close_whatsapp_browser()
        assert session.close_calls == 1
        server._close_whatsapp_browser()  # must not raise or double-close
        assert session.close_calls == 1
