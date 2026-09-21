"""Tests for daemon IPC protocol message models.

Verifies serialization round-trips, validation, and max-message-size enforcement.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from jarvis.daemon.protocol import (
    MAX_MESSAGE_SIZE,
    PROTOCOL_VERSION,
    AuthMessage,
    AuthOk,
    CancelMessage,
    ChatMessage,
    ConfirmRequest,
    ConfirmResponse,
    ErrorMessage,
    EventMessage,
    FinalMessage,
    ShutdownMessage,
    StatusRequest,
    StatusResponse,
)
from jarvis.daemon.server import _parse_message

# ── Serialization round-trips ───────────────────────────────────────────


class TestSerializationRoundTrip:
    """Every message type should survive a JSON round-trip."""

    @pytest.mark.parametrize(
        "msg",
        [
            AuthMessage(token="test-token-123"),
            ChatMessage(id="t1", text="open notepad", source="terminal"),
            ChatMessage(text="hello"),  # defaults
            ConfirmResponse(task_id="t1", approved=True, action_hash="abc123"),
            ConfirmResponse(
                task_id="t1",
                approved=True,
                action_hash="abc",
                password="secret",
                typed_confirmation="MyFolder",
            ),
            CancelMessage(task_id="t1"),
            StatusRequest(),
            ShutdownMessage(),
        ],
    )
    def test_client_to_server_round_trip(self, msg: Any) -> None:
        json_str = msg.model_dump_json()
        parsed = _parse_message(json_str)
        assert parsed is not None
        assert type(parsed) is type(msg)
        assert parsed.model_dump() == msg.model_dump()

    @pytest.mark.parametrize(
        "msg",
        [
            AuthOk(version=PROTOCOL_VERSION),
            EventMessage(task_id="t1", kind="plan", data={"goal": "test"}),
            ConfirmRequest(
                task_id="t1",
                tier=2,
                summary="Delete 3 files",
                needs_password=True,
                action_hash="abc123",
            ),
            FinalMessage(task_id="t1", text="Done!"),
            ErrorMessage(code="queue_full", message="queue is full"),
            StatusResponse(daemon="running", voice="off", unlocked=False, queue=0),
        ],
    )
    def test_server_to_client_round_trip(self, msg: Any) -> None:
        json_str = msg.model_dump_json()
        data = json.loads(json_str)
        # Server messages are parsed via their respective constructors
        if msg.type == "confirm_request":
            parsed = ConfirmRequest.model_validate(data)
        else:
            parsed = type(msg).model_validate(data)
        assert parsed.model_dump() == msg.model_dump()


# ── Default values ───────────────────────────────────────────────────────


class TestDefaults:
    def test_chat_message_defaults(self) -> None:
        msg = ChatMessage(text="hello")
        assert msg.type == "chat"
        assert msg.id == ""
        assert msg.source == "terminal"

    def test_auth_ok_default_version(self) -> None:
        msg = AuthOk()
        assert msg.version == PROTOCOL_VERSION

    def test_event_message_default_kind(self) -> None:
        msg = EventMessage(task_id="t1")
        assert msg.kind == "log"
        assert msg.data == {}

    def test_status_response_defaults(self) -> None:
        msg = StatusResponse()
        assert msg.daemon == "running"
        assert msg.voice == "off"
        assert msg.unlocked is False
        assert msg.queue == 0
        assert msg.active_task is None


# ── Validation ───────────────────────────────────────────────────────────


class TestValidation:
    def test_chat_message_rejects_empty_text(self) -> None:
        with pytest.raises(ValidationError):
            ChatMessage(text="")

    def test_chat_message_rejects_long_text(self) -> None:
        with pytest.raises(ValidationError):
            ChatMessage(text="x" * 5000)

    def test_confirm_response_password_optional(self) -> None:
        msg = ConfirmResponse(task_id="t1", approved=True, action_hash="h")
        assert msg.password is None
        assert msg.typed_confirmation is None

    def test_event_message_invalid_kind_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EventMessage(task_id="t1", kind="invalid_kind")  # type: ignore[arg-type]


# ── Message parsing ──────────────────────────────────────────────────────


class TestParseMessage:
    def test_valid_json_wrong_type_returns_none(self) -> None:
        result = _parse_message(json.dumps({"type": "unknown_type"}))
        assert result is None

    def test_invalid_json_returns_none(self) -> None:
        result = _parse_message("not json at all")
        assert result is None

    def test_non_dict_json_returns_none(self) -> None:
        result = _parse_message('"just a string"')
        assert result is None

    def test_array_json_returns_none(self) -> None:
        result = _parse_message("[1, 2, 3]")
        assert result is None

    def test_empty_string_returns_none(self) -> None:
        result = _parse_message("")
        assert result is None

    def test_auth_message_parsed(self) -> None:
        result = _parse_message(json.dumps({"type": "auth", "token": "abc"}))
        assert isinstance(result, AuthMessage)
        assert result.token == "abc"

    def test_chat_message_parsed(self) -> None:
        result = _parse_message(json.dumps({"type": "chat", "text": "hello"}))
        assert isinstance(result, ChatMessage)
        assert result.text == "hello"

    def test_missing_required_field_returns_none(self) -> None:
        result = _parse_message(json.dumps({"type": "chat"}))
        assert result is None


# ── Max message size ─────────────────────────────────────────────────────


class TestMaxMessageSize:
    def test_max_size_constant_is_reasonable(self) -> None:
        assert MAX_MESSAGE_SIZE == 65_536
        assert MAX_MESSAGE_SIZE >= 1024

    def test_large_payload_still_serializes(self) -> None:
        """The Pydantic models don't enforce size; the server does."""
        large_text = "x" * 100_000
        msg = ChatMessage(text=large_text[:4096])  # Pydantic max_length
        json_str = msg.model_dump_json()
        assert len(json_str.encode("utf-8")) < MAX_MESSAGE_SIZE
