"""Unit tests for the daemon IPC client (daemon/client.py).

Covers ``_parse_server_message`` and ``wait_for_event`` — the latter must
surface :class:`ClarificationRequest` / :class:`ConfirmRequest` so the CLI can
answer interrupts inline.  No real TCP connection is used.
"""

from __future__ import annotations

import asyncio
import json

from jarvis.daemon.client import DaemonClient, _parse_server_message
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
