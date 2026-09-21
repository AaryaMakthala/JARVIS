"""Daemon voice tests (F7, F8, F9, F12).

Covers:
* ``voice_toggle`` protocol message → daemon starts/stops the VoiceService
  (``jarvis on/off`` no longer spawns a private CLI voice loop).
* Voice task submission (_voice_submit) runs the worker, resolves Tier-1
  confirmations on the worker thread, and returns the final answer straight
  to the blocked voice-loop thread — no IPC socket involved.
* Tier 2+ actions can never be approved from a ``voice``-sourced
  ConfirmResponse (fail closed in the daemon, not the prompt).
"""

from __future__ import annotations

import asyncio
import json
import json as _json
import threading
import time
from typing import Any

from jarvis.config import Settings, VoiceSettings
from jarvis.daemon.protocol import (
    AuthMessage,
    ChatMessage,
    ConfirmResponse,
    ErrorMessage,
    VoiceToggleMessage,
)
from jarvis.daemon.server import DaemonServer, TaskSlot, _parse_message
from jarvis.voice.fakes import FakeSTT, FakeTTS, FakeWakeWord
from jarvis.voice.service import VoiceService

TEST_TOKEN = "daemon-voice-token-0123456789"


class _FakeStore:
    def __init__(self) -> None:
        self._kv: dict[str, str] = {"ipc_token": TEST_TOKEN}

    def get(self, name: str) -> str | None:
        return self._kv.get(name)

    def set(self, name: str, value: str) -> None:
        self._kv[name] = value

    def has(self, name: str) -> bool:
        return name in self._kv

    def check_store_access(self) -> str:
        return "FakeStore"


class _FakeVoiceService:
    """Records start/stop calls like the real VoiceService would."""

    def __init__(self) -> None:
        self.start_calls = 0
        self.stop_calls = 0
        self.active = False
        self._state = "off"
        self._error_code: str | None = None

    def start(self) -> str:
        if self.active:
            return "voice is already active"
        self.start_calls += 1
        self.active = True
        self._state = "on"
        return "voice activated"

    def stop(self) -> str:
        self.stop_calls += 1
        self.active = False
        self._state = "off"
        return "voice deactivated"

    def is_active(self) -> bool:
        return self.active

    @property
    def state(self) -> str:
        return self._state

    @property
    def error_code(self) -> str | None:
        return self._error_code

    @property
    def loop(self) -> Any:
        return None


class _ToggleServer(DaemonServer):
    """Real server with a fake VoiceService plugged in."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings=settings, store=_FakeStore())
        self.fake_voice = _FakeVoiceService()

    def _build_voice_service(self) -> Any:
        return self.fake_voice


class _ServerHarness:
    """Runs DaemonServer._handle_client + _dispatch_loop on a TCP socket."""

    def __init__(self, server: DaemonServer) -> None:
        self.server = server
        self.port = 0
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        async def _serve() -> None:
            srv = await asyncio.start_server(self.server._handle_client, "127.0.0.1", 0)
            self.port = srv.sockets[0].getsockname()[1]
            self.server._loop = asyncio.get_running_loop()
            dispatch_task = asyncio.create_task(self.server._dispatch_loop())
            async with srv:
                await self.server._shutdown_event.wait()
            dispatch_task.cancel()

        def _run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_serve())

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        for _ in range(50):
            if self.port:
                return
            time.sleep(0.1)
        raise RuntimeError("server did not start")

    def stop(self) -> None:
        loop = self.server._loop
        if loop is not None:
            loop.call_soon_threadsafe(self.server._shutdown_event.set)
        if self._thread:
            self._thread.join(timeout=3)


class _Client:
    def __init__(self, port: int) -> None:
        self.loop = asyncio.new_event_loop()
        self.reader, self.writer = self.loop.run_until_complete(
            asyncio.open_connection("127.0.0.1", port)
        )

    def send(self, msg: Any) -> None:
        self.writer.write((msg.model_dump_json() + "\n").encode("utf-8"))
        self.loop.run_until_complete(self.writer.drain())

    def recv_json(self, timeout: float = 2.0) -> dict[str, Any] | None:
        try:
            raw = self.loop.run_until_complete(
                asyncio.wait_for(self.reader.readline(), timeout=timeout)
            )
        except TimeoutError:
            return None
        if not raw:
            return None
        return json.loads(raw.decode("utf-8"))

    def close(self) -> None:
        self.writer.close()
        self.loop.run_until_complete(asyncio.sleep(0.05))
        self.loop.close()


# ── F7/F12: protocol + toggle ───────────────────────────────────────────


class TestVoiceToggleProtocol:
    def test_parse_voice_toggle_message(self) -> None:
        msg = _parse_message('{"type": "voice_toggle", "state": true}')
        assert isinstance(msg, VoiceToggleMessage)
        assert msg.state is True

    def test_parse_malformed_voice_toggle_returns_none(self) -> None:
        assert _parse_message('{"type": "voice_toggle"}') is None  # missing state
        assert _parse_message('{"type": "not_a_toggle", "state": true}') is None

    def test_confirm_response_defaults_to_terminal_source(self) -> None:
        msg = ConfirmResponse(task_id="t", approved=True, action_hash="h" * 64)
        assert msg.source == "terminal"
        assert msg.model_dump()["source"] == "terminal"


class TestToggleEndToEnd:
    def _start(self, settings: Settings) -> tuple[_ToggleServer, _ServerHarness]:
        server = _ToggleServer(settings)
        server._bootstrap_voice()  # mirrors the boot wiring the real _serve() does
        harness = _ServerHarness(server)
        harness.start()
        return server, harness

    def test_toggle_on_starts_daemon_voice(self) -> None:
        server, harness = self._start(Settings(voice=VoiceSettings(enabled=True)))
        try:
            assert server.fake_voice.start_calls == 1  # auto-started at boot
            c = _Client(harness.port)
            try:
                c.send(AuthMessage(token=TEST_TOKEN))
                assert c.recv_json(timeout=2) and c.recv_json(timeout=2) is None
                c.send(VoiceToggleMessage(state=True))
                event = c.recv_json(timeout=2)
                assert event and event["type"] == "event"
                assert "already active" in event["data"]["message"]
                assert server.fake_voice.start_calls == 1  # no second start

                c.send(VoiceToggleMessage(state=False))
                off = c.recv_json(timeout=2)
                assert off and off["type"] == "event"
                assert "deactivated" in off["data"]["message"]
                assert server.fake_voice.stop_calls == 1
            finally:
                c.close()
        finally:
            harness.stop()

    def test_toggle_off_when_never_started_is_idempotent(self) -> None:
        server, harness = self._start(Settings(voice=VoiceSettings(enabled=False)))
        try:
            c = _Client(harness.port)
            try:
                c.send(AuthMessage(token=TEST_TOKEN))
                c.recv_json(timeout=2)
                c.send(VoiceToggleMessage(state=False))
                event = c.recv_json(timeout=2)
                assert event and event["type"] == "event"
                assert server.fake_voice.stop_calls == 0  # no service built → no-op
            finally:
                c.close()
        finally:
            harness.stop()

    def test_toggle_on_rejected_when_disabled_in_config(self) -> None:
        _server, harness = self._start(Settings(voice=VoiceSettings(enabled=False)))
        try:
            c = _Client(harness.port)
            try:
                c.send(AuthMessage(token=TEST_TOKEN))
                c.recv_json(timeout=2)
                c.send(VoiceToggleMessage(state=True))
                err = c.recv_json(timeout=2)
                assert err and err["type"] == "error"
                assert err["code"] == "voice_disabled"
            finally:
                c.close()
        finally:
            harness.stop()


# ── F12: voice cannot approve Tier 2+ ───────────────────────────────────


class _FakeConn:
    def __init__(self) -> None:
        self.sent: list[Any] = []

    async def send(self, msg: Any) -> None:
        self.sent.append(msg)


class TestVoiceSourceGuard:
    def _server(self) -> DaemonServer:
        return DaemonServer(
            settings=Settings(voice=VoiceSettings(enabled=False)), store=_FakeStore()
        )

    def test_tier2_voice_source_rejected(self) -> None:
        server = self._server()
        slot = TaskSlot(task_id="t1", text="x", source="chat", owner_id="c1", confirm_tier=2)
        server._active = slot
        conn = _FakeConn()
        msg = ConfirmResponse(task_id="t1", approved=True, action_hash="h" * 64, source="voice")
        asyncio.run(server._handle_confirm(msg, conn))
        err = conn.sent[-1]
        assert isinstance(err, ErrorMessage)
        assert err.code == "tier_requires_terminal"
        assert not slot.done  # the worker keeps waiting; nothing was approved

    def test_tier3_voice_source_rejected(self) -> None:
        server = self._server()
        slot = TaskSlot(task_id="t3", text="x", source="chat", owner_id="c1", confirm_tier=3)
        server._active = slot
        conn = _FakeConn()
        asyncio.run(
            server._handle_confirm(
                ConfirmResponse(task_id="t3", approved=True, action_hash="h" * 64, source="voice"),
                conn,
            )
        )
        assert conn.sent[-1].code == "tier_requires_terminal"

    def test_tier1_voice_source_not_user_error(self) -> None:
        """A Tier-1 voice answer must reach the worker, not bounce as an error."""
        server = self._server()
        slot = TaskSlot(task_id="t1", text="x", source="chat", owner_id="c1", confirm_tier=1)
        server._active = slot
        conn = _FakeConn()
        asyncio.run(
            server._handle_confirm(
                ConfirmResponse(task_id="t1", approved=True, action_hash="h" * 64, source="voice"),
                conn,
            )
        )
        assert not any(isinstance(m, ErrorMessage) for m in conn.sent)
        assert slot.resume_answer == {"approved": True, "action_hash": "h" * 64}


# ── F8: voice task bridge (worker-side confirmation) ────────────────────


class _VoiceLoopStub:
    """Minimal loop stub; approve() behaviour is switchable per test."""

    def __init__(self, can_approve: bool = True) -> None:
        self.can_approve = can_approve
        self.confirmation_payloads: list[dict[str, Any]] = []

    def is_active(self) -> bool:
        return True

    def can_confirm_by_voice(self, payload: dict[str, Any]) -> bool:
        return self.can_approve

    def confirm_by_voice(self, payload: dict[str, Any], *, on_confirmation: Any = None) -> str:
        payload = dict(payload)
        self.confirmation_payloads.append(payload)
        if on_confirmation is not None:
            on_confirmation(
                {"approved": self.can_approve, "action_hash": payload.get("action_hash", "")}
            )
        return "Approved." if self.can_approve else "Cancelled."


class _VoiceBridgeServer(DaemonServer):
    """Server whose worker mirrors the real voice branch exactly.

    ``_worker_run`` is a faithful copy of the real voice-confirmation branch
    of DaemonServer._worker_run: publish → _run_voice_confirmation → resume →
    final → done → notify the (blocked) voice thread.
    """

    def __init__(self, *, approve: bool = True, voice_service: Any = None) -> None:
        super().__init__(settings=Settings(voice=VoiceSettings(enabled=False)), store=_FakeStore())
        self._voice_service = voice_service
        self.loop_stub = _VoiceLoopStub(can_approve=approve)
        self.worker_events: list[str] = []

    def _ctx_voice_loop(self) -> Any | None:
        return self.loop_stub

    def _worker_run(self, slot: TaskSlot) -> None:  # type: ignore[override]
        self.worker_events.append("started")
        payload = {
            "tier": 1,
            "summary": "create file",
            "action_hash": "h" * 64,
            "typed_confirmation": None,
        }
        self._run_voice_confirmation(slot, payload)
        slot.event.clear()
        answer = slot.resume_answer
        slot.resume_answer = None
        if answer is None:
            answer = {"approved": False, "action_hash": payload["action_hash"]}
        slot.result_text = f"approved={bool(answer.get('approved'))}"
        slot.done = True
        slot.event.set()
        self.worker_events.append("done")


class TestVoiceWorkerBridge:
    def test_voice_submit_runs_worker_and_returns_final(self) -> None:
        server = _VoiceBridgeServer(approve=True)
        outcome = server._voice_submit("create file", "voice")
        assert outcome.error is None
        assert outcome.final_answer == "approved=True"
        assert server.worker_events == ["started", "done"]
        # The confirmation dialogue saw the hash so approval is bound to it.
        assert server.loop_stub.confirmation_payloads[0]["action_hash"] == "h" * 64
        # Completed slot is done and answered; a later dispatch-loop pass
        # would promote the next queued task (owner is VOICE_OWNER).
        slot = server._active
        assert slot is not None and slot.done
        server._active = None  # cleanup, as the dispatch loop would do

    def test_voice_submit_refuses_when_confirmation_blocked(self) -> None:
        server = _VoiceBridgeServer(approve=False)
        outcome = server._voice_submit("create file", "voice")
        assert outcome.final_answer == "approved=False"

    def test_voice_submit_without_live_loop_fails_closed(self) -> None:
        server = _VoiceBridgeServer(approve=True)

        def _no_loop() -> Any | None:
            return None

        server._ctx_voice_loop = _no_loop  # type: ignore[method-assign]
        outcome = server._voice_submit("create file", "voice")
        assert outcome.final_answer == "approved=False"

    def test_voice_submit_queue_full_returns_error(self) -> None:
        from jarvis.daemon.server import MAX_QUEUE_SIZE

        server = _VoiceBridgeServer(approve=True)
        busy = TaskSlot(task_id="busy", text="x", source="chat", owner_id="c1")
        busy.done = False
        server._active = busy
        server._queue = [
            TaskSlot(task_id=f"q{i}", text="x", source="chat", owner_id=None)
            for i in range(MAX_QUEUE_SIZE)
        ]
        outcome = server._voice_submit("hello", "voice")
        assert outcome.error == "queue is full — try again later"
        server._active = None


# ── F7: chat still works while voice queue helpers exist ────────────────


# ── F7: chat still works while voice queue helpers exist ────────────────


class TestVoiceDoesNotBreakChat:
    def test_chat_message_parses_unchanged(self) -> None:
        msg = _parse_message(_json.dumps({"type": "chat", "id": "1", "text": "hi"}))
        assert isinstance(msg, ChatMessage)
        assert msg.text == "hi"


# ── Stage 2: voice error states surface through the daemon ──────────────


class _FlakyMic:
    """AudioInput that fails the first open (mic busy) then works."""

    def __init__(self) -> None:
        self._fail = True
        self.close_calls = 0
        self.open_calls = 0

    def open(self, sample_rate: int = 16_000, channels: int = 1) -> None:
        self.open_calls += 1
        if self._fail:
            self._fail = False
            raise RuntimeError("device busy")

    def read(self, num_frames: int) -> Any:
        from jarvis.voice.interfaces import AudioSegment

        return AudioSegment(samples=[0.0] * num_frames, sample_rate=16_000)

    def close(self) -> None:
        self.close_calls += 1

    def is_open(self) -> bool:
        return not self._fail


class TestVoiceErrorStates:
    def _server(self, service: Any) -> DaemonServer:
        server = DaemonServer(
            settings=Settings(voice=VoiceSettings(enabled=True)), store=_FakeStore()
        )
        server._voice_service = service
        return server

    def test_toggle_on_sends_error_message_with_fixed_code(self) -> None:
        svc = VoiceService()  # audio None → no-audio-library
        server = self._server(svc)
        conn = _FakeConn()
        asyncio.run(server._handle_voice_toggle(VoiceToggleMessage(state=True), conn))
        err = conn.sent[-1]
        assert isinstance(err, ErrorMessage)
        assert err.code == "no-audio-library"
        assert "no-audio-library" in err.message
        assert svc.state == "error"

    def test_toggle_on_after_error_retries_and_activates(self) -> None:
        mic = _FlakyMic()
        svc = VoiceService(
            audio=mic,
            wake_detector=FakeWakeWord(results=[]),
            stt=FakeSTT(),
            tts=FakeTTS(),
            idle_timeout_s=600.0,
        )
        server = self._server(svc)
        conn = _FakeConn()
        asyncio.run(server._handle_voice_toggle(VoiceToggleMessage(state=True), conn))
        assert conn.sent[-1].code == "mic-open-failed"
        assert svc.state == "error"
        conn.sent.clear()
        asyncio.run(server._handle_voice_toggle(VoiceToggleMessage(state=True), conn))
        ev = conn.sent[-1]
        assert ev.type == "event"
        assert "voice activated" in ev.data["message"]
        assert svc.state == "on"
        asyncio.run(server._handle_voice_toggle(VoiceToggleMessage(state=False), conn))
        assert svc.state == "off"  # loop stopped cleanly

    def test_status_reports_error_and_reason(self) -> None:
        svc = VoiceService()
        assert "no-audio-library" in svc.start()
        server = self._server(svc)
        conn = _FakeConn()
        asyncio.run(server._handle_status(conn))
        resp = conn.sent[-1]
        assert resp.voice == "error"
        assert resp.voice_reason == "no-audio-library"
        assert resp.daemon == "running"

    def test_boot_auto_start_failure_sets_error_and_keeps_daemon_alive(self) -> None:
        """Boot auto-start may fail (e.g. no voice deps) but the daemon lives on."""
        server = DaemonServer(
            settings=Settings(voice=VoiceSettings(enabled=True)), store=_FakeStore()
        )
        server._build_voice_service = lambda: VoiceService()  # type: ignore[method-assign]
        server._bootstrap_voice()  # must not raise
        assert server._voice_service is not None
        assert server._voice_service.state == "error"
        assert server._voice_service.error_code == "no-audio-library"
        # The daemon control plane still answers.
        conn = _FakeConn()
        asyncio.run(server._handle_status(conn))
        resp = conn.sent[-1]
        assert resp.daemon == "running"
        assert resp.voice == "error"
        assert resp.voice_reason == "no-audio-library"
