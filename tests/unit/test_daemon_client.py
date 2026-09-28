"""Unit tests for the daemon IPC client (daemon/client.py).

Covers ``_parse_server_message`` and ``wait_for_event`` — the latter must
surface :class:`ClarificationRequest` / :class:`ConfirmRequest` so the CLI can
answer interrupts inline.  No real TCP connection is used.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from jarvis.daemon.client import DaemonClient, DaemonError, _parse_server_message
from jarvis.daemon.protocol import ClarificationRequest, ConfirmRequest, FinalMessage


class _FakeReader:
    """Async reader that serves scripted lines, then blocks forever."""

    def __init__(self, lines: list[bytes]) -> None:
        self._lines = list(lines)

    async def readline(self) -> bytes:
        if self._lines:
            return self._lines.pop(0)
        await asyncio.Event().wait()  # block; lets the outer timeout fire
        return b""


class _FakeWriter:
    """Minimal async writer capturing frames for assertions."""

    def __init__(self) -> None:
        self.buffer: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.buffer.append(data)

    async def drain(self) -> None:
        return None

    def is_closing(self) -> bool:
        return False

    def close(self) -> None:
        return None


def _client(lines: list[bytes]) -> DaemonClient:
    client = DaemonClient(port=1, token="t")
    client._reader = _FakeReader(lines)
    return client


# ── _parse_server_message ────────────────────────────────────────────────


class TestParseServerMessage:
    def test_parses_clarification_request(self) -> None:
        msg = _parse_server_message(
            json.dumps({"type": "clarification_request", "task_id": "t1", "question": "Which?"})
        )
        assert isinstance(msg, ClarificationRequest)
        assert msg.question == "Which?"

    def test_parses_confirm_request(self) -> None:
        msg = _parse_server_message(
            json.dumps({"type": "confirm_request", "task_id": "t1", "tier": 2, "summary": "del"})
        )
        assert isinstance(msg, ConfirmRequest)
        assert msg.tier == 2

    def test_parses_ordinary_server_messages(self) -> None:
        msg = _parse_server_message(json.dumps({"type": "final", "task_id": "t1", "text": "done"}))
        assert isinstance(msg, FinalMessage)

    def test_unknown_type_returns_none(self) -> None:
        assert _parse_server_message(json.dumps({"type": "weird"})) is None

    def test_junk_returns_none(self) -> None:
        assert _parse_server_message("not json") is None
        assert _parse_server_message("") is None
        assert _parse_server_message("[1, 2]") is None

    def test_malformed_dispatch_message_returns_none(self) -> None:
        assert _parse_server_message(json.dumps({"type": "event"})) is None


# ── wait_for_event ───────────────────────────────────────────────────────


class TestWaitForEvent:
    def test_returns_clarification_request(self) -> None:
        line = (
            json.dumps(
                {"type": "clarification_request", "task_id": "t1", "question": "Which file?"}
            ).encode("utf-8")
            + b"\n"
        )
        msg = _client([line]).wait_for_event(timeout=5)
        assert isinstance(msg, ClarificationRequest)
        assert msg.task_id == "t1"
        assert msg.question == "Which file?"

    def test_returns_confirm_request(self) -> None:
        line = (
            json.dumps(
                {
                    "type": "confirm_request",
                    "task_id": "t1",
                    "tier": 2,
                    "summary": "Delete 3 files",
                    "action_hash": "abc",
                }
            ).encode("utf-8")
            + b"\n"
        )
        msg = _client([line]).wait_for_event(timeout=5)
        assert isinstance(msg, ConfirmRequest)
        assert msg.action_hash == "abc"

    def test_returns_final_message(self) -> None:
        line = json.dumps({"type": "final", "task_id": "t1", "text": "done"}).encode() + b"\n"
        msg = _client([line]).wait_for_event(timeout=5)
        assert isinstance(msg, FinalMessage)

    def test_eof_returns_none(self) -> None:
        assert _client([b""]).wait_for_event(timeout=5) is None

    def test_unknown_message_type_returns_none(self) -> None:
        assert _client([b'{"type": "nonsense"}\n']).wait_for_event(timeout=5) is None

    def test_timeout_returns_none(self) -> None:
        assert _client([]).wait_for_event(timeout=0.05) is None


# ── send_chat: only the ack may be consumed ──────────────────────────────


class TestSendChatConsumesOnlyTheAck:
    """Regression (live bug): the server used to send no ack for an
    immediately-started chat task, so ``send_chat`` consumed the first
    downstream message — the ConfirmRequest — and the confirmation prompt
    never reached the terminal; the answer window elapsed and the task was
    refused with "Confirmation timed out."
    """

    @staticmethod
    def _client_with_writer(lines: list[bytes]) -> tuple[DaemonClient, _FakeWriter]:
        client = DaemonClient(port=1, token="t")
        client._reader = _FakeReader(lines)
        writer = _FakeWriter()
        client._writer = writer
        return client, writer

    def test_ack_is_consumed_and_confirm_request_stays_queued(self) -> None:
        """The live message order for a Tier-1 task: the chat ack first, then
        the ConfirmRequest must still be delivered to ``wait_for_event``."""
        ack = json.dumps({"type": "event", "task_id": "t7", "kind": "log"}).encode() + b"\n"
        confirm = (
            json.dumps(
                {"type": "confirm_request", "task_id": "t7", "tier": 1, "summary": "Lock"}
            ).encode()
            + b"\n"
        )
        client, _writer = self._client_with_writer([ack, confirm])
        returned = client.send_chat("lock my computer")
        assert returned == "t7"
        msg = client.wait_for_event(timeout=5)
        assert isinstance(msg, ConfirmRequest)
        assert msg.tier == 1
        assert msg.summary == "Lock"

    def test_confirm_request_in_reply_to_submission_fails_loudly(self) -> None:
        """Belt-and-braces: if a ConfirmRequest ever arrives as the reply to a
        submission (stale daemon, protocol drift) the client must raise — not
        discard it, which is what made the prompt invisible."""
        confirm = (
            json.dumps({"type": "confirm_request", "task_id": "t7", "tier": 1}).encode() + b"\n"
        )
        client, _writer = self._client_with_writer([confirm])
        with pytest.raises(DaemonError) as excinfo:
            client.send_chat("lock my computer")
        assert "ConfirmRequest" in excinfo.value.message

    def test_error_reply_to_submission_still_raises(self) -> None:
        err = (
            json.dumps(
                {"type": "error", "task_id": "", "code": "queue_full", "message": "full"}
            ).encode()
            + b"\n"
        )
        client, _writer = self._client_with_writer([err])
        with pytest.raises(DaemonError) as excinfo:
            client.send_chat("hello")
        assert excinfo.value.code == "queue_full"

    def test_eof_on_submission_raises(self) -> None:
        """The scripted EOF line (``b""``) makes ``_recv`` return ``None``;
        the submission must surface that as a DaemonError, not hang."""
        client, _writer = self._client_with_writer([b""])
        with pytest.raises(DaemonError):
            client.send_chat("hello")
