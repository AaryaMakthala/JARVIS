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
from jarvis.daemon.protocol import (
    ClarificationRequest,
    ConfirmRequest,
    EventMessage,
    FinalMessage,
)


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
        ack = json.dumps({"type": "event", "task_id": "t7", "kind": "ack"}).encode() + b"\n"
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

    def test_a_confirm_request_overtaking_the_submission_ack_is_delivered(self) -> None:
        """The server starts the worker *before* it acks the submission, so for
        a fast Tier-1 action the ConfirmRequest can overtake the ack.  It must
        be replayed by ``wait_for_event``, not swallowed as the reply."""
        confirm = (
            json.dumps(
                {"type": "confirm_request", "task_id": "t7", "tier": 1, "summary": "Lock"}
            ).encode()
            + b"\n"
        )
        ack = json.dumps({"type": "event", "task_id": "t7", "kind": "ack"}).encode() + b"\n"
        client, _writer = self._client_with_writer([confirm, ack])
        assert client.send_chat("lock my computer") == "t7"
        assert isinstance(client.wait_for_event(timeout=5), ConfirmRequest)

    def test_a_reply_that_is_never_an_ack_is_kept_and_the_wait_times_out(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ConfirmRequest with no ack behind it (stale daemon, protocol drift)
        is kept for the caller and the wait ends in a truthful error - the old
        behaviour discarded it, which is what made the prompt invisible.  The
        wait is bounded, so a wedged daemon cannot hang the terminal either."""
        monkeypatch.setattr(DaemonClient, "ack_timeout", 0.05)
        confirm = (
            json.dumps({"type": "confirm_request", "task_id": "t7", "tier": 1}).encode() + b"\n"
        )
        client, _writer = self._client_with_writer([confirm])
        with pytest.raises(DaemonError) as excinfo:
            client.send_chat("lock my computer")
        assert excinfo.value.code == "ack_timeout"
        assert isinstance(client.wait_for_event(timeout=5), ConfirmRequest)

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

    def test_the_queued_ack_text_is_kept_for_the_terminal(self) -> None:
        """A request that is *queued* is why the next command appears to do
        nothing, and that used to be invisible: the ack was thrown away."""
        ack = (
            json.dumps(
                {"type": "event", "task_id": "t7", "kind": "ack", "data": {"message": "queued"}}
            ).encode()
            + b"\n"
        )
        client, _writer = self._client_with_writer([ack])
        assert client.send_chat("lock my computer") == "t7"
        assert client.last_ack_message == "queued"


# ── send_confirm / send_clarification: never lose a request ─────────────


class TestResponseSendsNeverSwallowARequest:
    """Regression (same failure mode as 663d559, one layer down).

    The server stores the answer, acks it, and *then* wakes the worker, which
    can dispatch the next ``ConfirmRequest`` or the ``FinalMessage`` on the same
    connection.  When that message arrived before the ack, the client used to
    consume it inside ``send_confirm`` and drop it - so the approval prompt
    never appeared and the task failed closed with "confirmation expired",
    which is exactly what was reported live.
    """

    ACK = (
        json.dumps(
            {
                "type": "event",
                "task_id": "t7",
                "kind": "ack",
                "data": {"message": "clarification received"},
            }
        ).encode()
        + b"\n"
    )
    #: A progress line for the *same* task.  It must not be mistaken for the
    #: ack: the real ack has to be found, or it resurfaces later as a stray
    #: event and the answer looks unanswered.
    PROGRESS = (
        json.dumps(
            {
                "type": "event",
                "task_id": "t7",
                "kind": "log",
                "data": {"message": "planner: unsupported, asking a question"},
            }
        ).encode()
        + b"\n"
    )
    CONFIRM = (
        json.dumps(
            {
                "type": "confirm_request",
                "task_id": "t7",
                "tier": 1,
                "summary": "Lock the computer",
                "action_hash": "abc",
            }
        ).encode()
        + b"\n"
    )
    FINAL = json.dumps({"type": "final", "task_id": "t7", "text": "locked"}).encode() + b"\n"

    @staticmethod
    def _client(lines: list[bytes]) -> DaemonClient:
        client = DaemonClient(port=1, token="t")
        client._reader = _FakeReader(lines)
        client._writer = _FakeWriter()
        return client

    def test_a_confirm_request_overtaking_the_ack_is_still_delivered(self) -> None:
        client = self._client([self.CONFIRM, self.ACK])
        client.send_clarification("t7", answer="y")
        msg = client.wait_for_event(timeout=5)
        assert isinstance(msg, ConfirmRequest)
        assert msg.tier == 1
        assert msg.action_hash == "abc"

    def test_a_final_message_overtaking_the_ack_is_still_delivered(self) -> None:
        client = self._client([self.FINAL, self.ACK])
        client.send_confirm("t7", approved=True, action_hash="abc")
        msg = client.wait_for_event(timeout=5)
        assert isinstance(msg, FinalMessage)
        assert msg.text == "locked"

    def test_buffered_messages_keep_their_order(self) -> None:
        client = self._client([self.FINAL, self.CONFIRM, self.ACK])
        client.send_confirm("t7", approved=True, action_hash="abc")
        first = client.wait_for_event(timeout=5)
        second = client.wait_for_event(timeout=5)
        assert isinstance(first, FinalMessage)
        assert isinstance(second, ConfirmRequest)

    def test_cancel_does_not_swallow_a_late_result_either(self) -> None:
        client = self._client([self.FINAL, self.ACK])
        client.send_cancel("t7")
        assert isinstance(client.wait_for_event(timeout=5), FinalMessage)

    def test_a_progress_line_is_not_mistaken_for_the_ack(self) -> None:
        """Same ``task_id``, different ``kind``.

        The ack has to be found *behind* the progress line, and the line is
        replayed in order afterwards - matching on ``task_id`` alone would
        return early and leave the real ack to resurface as a stray event.
        """
        client = self._client([self.PROGRESS, self.ACK, self.FINAL])
        client.send_clarification("t7", answer="the first one")
        first = client.wait_for_event(timeout=5)
        second = client.wait_for_event(timeout=5)
        assert isinstance(first, EventMessage)
        assert first.kind == "log"
        assert "asking a question" in str(first.data.get("message"))
        assert isinstance(second, FinalMessage)

    def test_an_error_reply_still_raises(self) -> None:
        err = (
            json.dumps(
                {
                    "type": "error",
                    "task_id": "t7",
                    "code": "no_such_task",
                    "message": "no active task",
                }
            ).encode()
            + b"\n"
        )
        with pytest.raises(DaemonError) as excinfo:
            self._client([err]).send_confirm("t7", approved=True, action_hash="abc")
        assert excinfo.value.code == "no_such_task"

    def test_eof_while_waiting_for_the_ack_raises(self) -> None:
        with pytest.raises(DaemonError) as excinfo:
            self._client([b""]).send_clarification("t7", answer="y")
        assert excinfo.value.code == "eof"

    def test_a_silent_daemon_times_out_instead_of_hanging(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A daemon that accepts the request and then says nothing must not
        freeze the CLI: the ack is sent before the worker is released, so a
        long silence means something is wrong, and saying so is the fix."""
        monkeypatch.setattr(DaemonClient, "ack_timeout", 0.05)
        with pytest.raises(DaemonError) as excinfo:
            self._client([]).send_clarification("t7", answer="y")
        assert excinfo.value.code == "ack_timeout"
        assert "not responding" in excinfo.value.message

    def test_a_daemon_that_never_acks_is_reported_not_guessed(self) -> None:
        client = self._client([self.CONFIRM] * (DaemonClient.max_pending + 1))
        with pytest.raises(DaemonError) as excinfo:
            client.send_confirm("t7", approved=True, action_hash="abc")
        assert excinfo.value.code == "no_ack"

    def test_close_discards_buffered_messages(self) -> None:
        client = self._client([self.CONFIRM, self.ACK])
        client.send_clarification("t7", answer="y")
        client.close()
        assert client._pending == []
