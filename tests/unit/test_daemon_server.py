"""Unit tests for daemon server internals.

Tests the message parsing, task slot state machine, and authentication logic
without requiring a real TCP connection (those are in the integration tests).
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

from jarvis.daemon.protocol import (
    AuthMessage,
    StatusRequest,
)
from jarvis.daemon.server import (
    ClientConnection,
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
