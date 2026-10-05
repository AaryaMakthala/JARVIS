"""Unit tests for daemon server internals.

Tests the message parsing, task slot state machine, and authentication logic
without requiring a real TCP connection (those are in the integration tests).
"""

from __future__ import annotations

import asyncio
import json
import threading
from unittest.mock import AsyncMock

import pytest

from jarvis.daemon.confirmations import plan_hash
from jarvis.daemon.protocol import (
    AuthMessage,
    StatusRequest,
)
from jarvis.daemon.server import (
    ClientConnection,
    DaemonServer,
    TaskSlot,
    _parse_message,
)

# ── Message parsing (delegated from protocol.py) ─────────────────────────


class TestServerMessageParsing:
    """Verify the server's _parse_message handles all client types."""

    def test_parse_auth(self) -> None:
        msg = _parse_message(json.dumps({"type": "auth", "token": "abc"}))
        assert isinstance(msg, AuthMessage)
        assert msg.token == "abc"

    def test_parse_chat(self) -> None:
        msg = _parse_message(json.dumps({"type": "chat", "text": "hello"}))
        from jarvis.daemon.protocol import ChatMessage

        assert isinstance(msg, ChatMessage)
        assert msg.text == "hello"

    def test_parse_confirm(self) -> None:
        msg = _parse_message(
            json.dumps(
                {
                    "type": "confirm_response",
                    "task_id": "t1",
                    "approved": True,
                    "action_hash": "h",
                }
            )
        )
        from jarvis.daemon.protocol import ConfirmResponse

        assert isinstance(msg, ConfirmResponse)
        assert msg.approved is True

    def test_parse_clarification(self) -> None:
        msg = _parse_message(
            json.dumps({"type": "clarification_response", "task_id": "t1", "answer": "the report"})
        )
        from jarvis.daemon.protocol import ClarificationResponse

        assert isinstance(msg, ClarificationResponse)
        assert msg.answer == "the report"

    def test_parse_status(self) -> None:
        msg = _parse_message(json.dumps({"type": "status"}))
        assert isinstance(msg, StatusRequest)

    def test_parse_shutdown(self) -> None:
        from jarvis.daemon.protocol import ShutdownMessage

        msg = _parse_message(json.dumps({"type": "shutdown"}))
        assert isinstance(msg, ShutdownMessage)

    def test_parse_garbage_returns_none(self) -> None:
        assert _parse_message("not json") is None
        assert _parse_message("") is None
        assert _parse_message("{}") is None  # missing type


# ── TaskSlot ─────────────────────────────────────────────────────────────


class TestTaskSlot:
    def test_slot_defaults(self) -> None:
        slot = TaskSlot(task_id="t1", text="hello", source="terminal")
        assert slot.done is False
        assert slot.error is None
        assert slot.confirm_payload is None
        assert slot.resume_answer is None
        assert slot.result_text is None
        assert slot.event.is_set() is False  # new event starts unset

    def test_slot_event_signalling(self) -> None:
        slot = TaskSlot(task_id="t1", text="hello", source="terminal")
        # Worker thread would wait on slot.event
        slot.event.set()
        assert slot.event.is_set() is True
        # After consuming the event
        slot.event.clear()
        assert slot.event.is_set() is False


# ── ClientConnection ─────────────────────────────────────────────────────


class TestClientConnection:
    """Test ClientConnection with mock reader/writer."""

    def test_closed_connection_send_is_noop(self) -> None:
        writer = AsyncMock()
        reader = AsyncMock()
        conn = ClientConnection(writer=writer, reader=reader, closed=True)
        # Should not raise
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(conn.send(AuthMessage(token="x")))
        finally:
            loop.close()
        writer.write.assert_not_called()

    def test_send_writes_ndjson(self) -> None:
        from jarvis.daemon.protocol import AuthOk

        writer = AsyncMock()
        reader = AsyncMock()
        conn = ClientConnection(writer=writer, reader=reader)
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(conn.send(AuthOk()))
        finally:
            loop.close()
        writer.write.assert_called_once()
        written = writer.write.call_args[0][0]
        assert written.endswith(b"\n")
        data = json.loads(written.decode("utf-8").strip())
        assert data["type"] == "auth_ok"

    def test_send_handles_connection_error(self) -> None:
        from jarvis.daemon.protocol import AuthOk

        writer = AsyncMock()
        writer.drain = AsyncMock(side_effect=ConnectionError("closed"))
        reader = AsyncMock()
        conn = ClientConnection(writer=writer, reader=reader)
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(conn.send(AuthOk()))
        finally:
            loop.close()
        assert conn.closed is True

    def test_recv_returns_none_on_eof(self) -> None:
        reader = AsyncMock()
        reader.readline = AsyncMock(return_value=b"")
        writer = AsyncMock()
        conn = ClientConnection(writer=writer, reader=reader)
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(conn.recv())
        finally:
            loop.close()
        assert result is None

    def test_recv_returns_none_on_oversized(self) -> None:
        reader = AsyncMock()
        reader.readline = AsyncMock(return_value=b"x" * 100_000)
        writer = AsyncMock()
        conn = ClientConnection(writer=writer, reader=reader)
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(conn.recv())
        finally:
            loop.close()
        assert result is None

    def test_recv_parses_valid_message(self) -> None:
        line = json.dumps({"type": "auth", "token": "test"}).encode() + b"\n"
        reader = AsyncMock()
        reader.readline = AsyncMock(return_value=line)
        writer = AsyncMock()
        conn = ClientConnection(writer=writer, reader=reader)
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(conn.recv())
        finally:
            loop.close()
        assert isinstance(result, AuthMessage)
        assert result.token == "test"

    def test_close_idempotent(self) -> None:
        writer = AsyncMock()
        reader = AsyncMock()
        conn = ClientConnection(writer=writer, reader=reader)
        conn.close()
        conn.close()  # second close should not raise
        assert conn.closed is True


# ── Clarification handling (ClarificationResponse → worker answer) ────────


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
    from jarvis.config import Settings, VoiceSettings

    return DaemonServer(settings=Settings(voice=VoiceSettings(enabled=False)), store=_FakeStore())


class TestHandleClarification:
    def test_delivers_answer_to_active_slot(self) -> None:
        from jarvis.daemon.protocol import ClarificationResponse

        server = _server()
        slot = TaskSlot(task_id="t1", text="setup", source="chat", owner_id="oid")
        server._active = slot
        conn = ClientConnection(writer=AsyncMock(), reader=AsyncMock())

        asyncio.run(
            server._handle_clarification(
                ClarificationResponse(task_id="t1", answer="the report"), conn
            )
        )
        assert slot.resume_answer == "the report"
        assert slot.event.is_set() is True
        assert slot.done is False

    def test_blank_answer_is_delivered_and_fails_closed_later(self) -> None:
        from jarvis.daemon.protocol import ClarificationResponse

        server = _server()
        slot = TaskSlot(task_id="t1", text="setup", source="chat", owner_id="oid")
        server._active = slot
        conn = ClientConnection(writer=AsyncMock(), reader=AsyncMock())

        asyncio.run(server._handle_clarification(ClarificationResponse(task_id="t1"), conn))
        # Valid on the wire; the clarify node decides it is "no clarification".
        assert slot.resume_answer == ""
        assert slot.event.is_set() is True

    def test_unknown_task_receives_error(self) -> None:
        from jarvis.daemon.protocol import ClarificationResponse

        server = _server()
        server._active = None
        writer = AsyncMock()
        conn = ClientConnection(writer=writer, reader=AsyncMock())

        asyncio.run(
            server._handle_clarification(ClarificationResponse(task_id="t9", answer="x"), conn)
        )
        written = writer.write.call_args[0][0].decode("utf-8")
        data = json.loads(written)
        assert data["type"] == "error"
        assert data["code"] == "no_such_task"

    def test_mismatched_task_id_receives_error(self) -> None:
        from jarvis.daemon.protocol import ClarificationResponse

        server = _server()
        slot = TaskSlot(task_id="t1", text="setup", source="chat", owner_id="oid")
        server._active = slot
        writer = AsyncMock()
        conn = ClientConnection(writer=writer, reader=AsyncMock())

        asyncio.run(
            server._handle_clarification(ClarificationResponse(task_id="t2", answer="x"), conn)
        )
        data = json.loads(writer.write.call_args[0][0].decode("utf-8"))
        assert data["code"] == "no_such_task"
        assert slot.resume_answer is None  # nothing was delivered

    def test_done_slot_receives_error(self) -> None:
        from jarvis.daemon.protocol import ClarificationResponse

        server = _server()
        slot = TaskSlot(task_id="t1", text="setup", source="chat", owner_id="oid")
        slot.done = True
        server._active = slot
        writer = AsyncMock()
        conn = ClientConnection(writer=writer, reader=AsyncMock())

        asyncio.run(
            server._handle_clarification(ClarificationResponse(task_id="t1", answer="x"), conn)
        )
        data = json.loads(writer.write.call_args[0][0].decode("utf-8"))
        assert data["code"] == "no_such_task"
        assert slot.resume_answer is None


class TestRequestAcks:
    """Regression (live bug): three handlers sent no reply on their success
    path.  The CLI's send helpers each block on the *next* server message, so
    with no ack they consumed downstream messages instead: ``send_chat`` ate
    the ConfirmRequest (prompt never displayed, answer window elapsed), and
    ``send_confirm``/``send_clarification`` would have eaten the final result.
    Every accepted client→server request must get exactly one reply.
    """

    def test_immediately_started_chat_sends_an_ack(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from jarvis.daemon.protocol import ChatMessage

        server = _server()
        # No real graph: the spawned worker thread must do nothing and exit.
        monkeypatch.setattr(server, "_worker_run", lambda slot: None)
        server._active = None  # no running task -> the task starts immediately
        writer = AsyncMock()
        conn = ClientConnection(writer=writer, reader=AsyncMock())

        asyncio.run(server._handle_chat(ChatMessage(id="t1", text="lock my computer"), conn))

        # The ack is sent synchronously, before any worker progress.
        written = writer.write.call_args[0][0].decode("utf-8")
        data = json.loads(written)
        assert data["type"] == "event"
        assert data["task_id"] == "t1"
        # A dedicated kind, not a progress "log": the client has to be able to
        # tell the submission receipt from a message about the same task.
        assert data["kind"] == "ack"
        assert data["data"]["message"] == "started"
        assert server._active is not None and server._active.task_id == "t1"

    def test_delivered_confirm_response_sends_an_ack(self) -> None:
        from jarvis.daemon.protocol import ConfirmResponse

        server = _server()
        slot = TaskSlot(task_id="t1", text="setup", source="chat", owner_id="oid")
        server._active = slot
        pending = server._confirmations.issue("t1", "s1", 1, "h", plan_hash(["h"]))
        writer = AsyncMock()
        conn = ClientConnection(writer=writer, reader=AsyncMock())

        asyncio.run(
            server._handle_confirm(
                ConfirmResponse(
                    task_id="t1",
                    approved=True,
                    action_hash="h",
                    confirmation_id=pending.confirmation_id,
                    plan_hash=pending.plan_hash,
                ),
                conn,
            )
        )
        assert slot.resume_answer == {"approved": True, "action_hash": "h"}
        assert slot.event.is_set() is True
        written = writer.write.call_args[0][0].decode("utf-8")
        data = json.loads(written)
        assert data["type"] == "event"
        assert data["task_id"] == "t1"
        # A dedicated kind: the client must be able to tell the ack from a
        # progress "log" for the same task while it waits for it.
        assert data["kind"] == "ack"
        assert data["data"]["message"] == "confirmation received"

    def test_delivered_clarification_sends_an_ack(self) -> None:
        from jarvis.daemon.protocol import ClarificationResponse

        server = _server()
        slot = TaskSlot(task_id="t1", text="setup", source="chat", owner_id="oid")
        server._active = slot
        writer = AsyncMock()
        conn = ClientConnection(writer=writer, reader=AsyncMock())

        asyncio.run(
            server._handle_clarification(ClarificationResponse(task_id="t1", answer="x"), conn)
        )
        assert slot.resume_answer == "x"
        written = writer.write.call_args[0][0].decode("utf-8")
        data = json.loads(written)
        assert data["type"] == "event"
        assert data["task_id"] == "t1"
        assert data["kind"] == "ack"
        assert data["data"]["message"] == "clarification received"

    def test_cancel_sends_an_ack(self) -> None:
        from jarvis.daemon.protocol import CancelMessage

        server = _server()
        slot = TaskSlot(task_id="t1", text="setup", source="chat", owner_id="oid")
        server._active = slot
        writer = AsyncMock()
        conn = ClientConnection(writer=writer, reader=AsyncMock())

        asyncio.run(server._handle_cancel(CancelMessage(task_id="t1"), conn))
        data = json.loads(writer.write.call_args[0][0].decode("utf-8"))
        assert data["type"] == "event"
        assert data["task_id"] == "t1"
        assert data["kind"] == "ack"
        assert data["data"]["message"] == "cancelled"


class TestAckPrecedesTheWorkerWakeup:
    """The ack must be on the wire *before* the worker thread is released.

    The client reads one message while waiting for its ack.  If the worker is
    woken first it can dispatch the next ``ConfirmRequest`` (or the
    ``FinalMessage``) on this same connection, and that request would be read as
    the ack - the approval prompt would vanish and the task would fail closed
    with "confirmation expired".
    """

    class _RecordingEvent:
        """A ``threading.Event`` that appends to a shared order list on ``set``."""

        def __init__(self, order: list[str]) -> None:
            self._order = order
            self._event = threading.Event()

        def set(self) -> None:
            self._order.append("worker-woken")
            self._event.set()

        def is_set(self) -> bool:
            return self._event.is_set()

        def wait(self, timeout: float | None = None) -> bool:
            return self._event.wait(timeout)

        def clear(self) -> None:
            self._event.clear()

    def _run_handler(self, server, slot, handler) -> list[str]:
        order: list[str] = []
        original_send = ClientConnection.send

        async def spy_send(self, message):
            order.append("ack")
            await original_send(self, message)

        slot.event = self._RecordingEvent(order)
        conn = ClientConnection(writer=AsyncMock(), reader=AsyncMock())
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(ClientConnection, "send", spy_send, raising=True)
            asyncio.run(handler(conn))
        return order

    def test_confirm_acks_before_waking_the_worker(self) -> None:
        from jarvis.daemon.protocol import ConfirmResponse

        server = _server()
        slot = TaskSlot(task_id="t1", text="lock", source="chat", owner_id="oid")
        server._active = slot
        pending = server._confirmations.issue("t1", "s1", 1, "h", plan_hash(["h"]))
        order = self._run_handler(
            server,
            slot,
            lambda conn: server._handle_confirm(
                ConfirmResponse(
                    task_id="t1",
                    approved=True,
                    action_hash="h",
                    confirmation_id=pending.confirmation_id,
                    plan_hash=pending.plan_hash,
                ),
                conn,
            ),
        )
        assert order == ["ack", "worker-woken"]

    def test_clarification_acks_before_waking_the_worker(self) -> None:
        from jarvis.daemon.protocol import ClarificationResponse

        server = _server()
        slot = TaskSlot(task_id="t1", text="lock", source="chat", owner_id="oid")
        server._active = slot
        order = self._run_handler(
            server,
            slot,
            lambda conn: server._handle_clarification(
                ClarificationResponse(task_id="t1", answer="y"), conn
            ),
        )
        assert order == ["ack", "worker-woken"]
