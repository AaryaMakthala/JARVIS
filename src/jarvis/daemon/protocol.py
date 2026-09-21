"""IPC wire protocol: typed Pydantic messages for client↔server communication.

Transport: TCP on ``127.0.0.1``, newline-delimited JSON (NDJSON), UTF-8.
Max message size: 64 KB.  First message must be ``auth`` with the token
from keyring (constant-time compare via ``hmac.compare_digest``).

Client → server: :class:`AuthMessage`, :class:`ChatMessage`,
:class:`ConfirmResponse`, :class:`CancelMessage`, :class:`StatusRequest`,
:class:`ShutdownMessage`, :class:`VoiceToggleMessage`.

Server → client: :class:`AuthOk`, :class:`EventMessage`,
:class:`ConfirmRequest`, :class:`FinalMessage`, :class:`ErrorMessage`,
:class:`StatusResponse`.

The daemon stores passwords only transiently (in ``confirm_response``) and
never forwards them to the graph (docs/03 invariant 8).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

__all__ = [
    "AuthMessage",
    "AuthOk",
    "CancelMessage",
    "ChatMessage",
    "ConfirmRequest",
    "ConfirmResponse",
    "DaemonMessage",
    "ErrorMessage",
    "EventMessage",
    "FinalMessage",
    "ShutdownMessage",
    "StatusRequest",
    "StatusResponse",
    "VoiceToggleMessage",
]

#: Maximum size of a single NDJSON line (bytes).  Server rejects anything larger.
MAX_MESSAGE_SIZE = 65_536

#: Protocol version reported in the ``auth_ok`` handshake.
PROTOCOL_VERSION = "0.1.0"


# ── Client → Server ──────────────────────────────────────────────────────


class AuthMessage(BaseModel):
    """First message on every connection.  Token is verified server-side."""

    type: Literal["auth"] = "auth"
    token: str


class ChatMessage(BaseModel):
    """Submit a new task to the daemon's queue."""

    type: Literal["chat"] = "chat"
    id: str = ""  # client-generated correlation id
    text: str = Field(min_length=1, max_length=4096)
    source: Literal["terminal", "voice", "benchmark"] = "terminal"


class ConfirmResponse(BaseModel):
    """User's answer to a confirmation request.

    The ``password`` field is consumed exactly once by the daemon (to verify
    the unlock) and is **never** included in the resume payload sent to the
    graph (docs/03 invariant 8).

    ``source`` records where the answer came from.  The daemon refuses answers
    claiming ``voice`` for Tier 2+ actions (docs/03 §7.8): those must be
    answered at a terminal.  Voice-driven Tier 1 answers never travel over
    IPC at all — they are resolved inside the daemon worker thread.
    """

    type: Literal["confirm_response"] = "confirm_response"
    task_id: str
    approved: bool
    action_hash: str
    password: str | None = None  # only for Tier 2 unlocks; discarded after use
    typed_confirmation: str | None = None  # folder-name confirmation
    source: Literal["terminal", "dialog", "voice"] = "terminal"


class CancelMessage(BaseModel):
    """Cancel a running or queued task."""

    type: Literal["cancel"] = "cancel"
    task_id: str


class StatusRequest(BaseModel):
    """Request daemon status."""

    type: Literal["status"] = "status"


class ShutdownMessage(BaseModel):
    """Gracefully stop the daemon."""

    type: Literal["shutdown"] = "shutdown"


class VoiceToggleMessage(BaseModel):
    """Turn the voice pipeline on/off via the daemon (``jarvis on`` / ``off``)."""

    type: Literal["voice_toggle"] = "voice_toggle"
    state: bool


# ── Server → Client ──────────────────────────────────────────────────────


class AuthOk(BaseModel):
    """Authentication succeeded."""

    type: Literal["auth_ok"] = "auth_ok"
    version: str = PROTOCOL_VERSION


class EventMessage(BaseModel):
    """Streaming event while a task is running (plan, step_start, step_result, log)."""

    type: Literal["event"] = "event"
    task_id: str
    kind: Literal["plan", "step_start", "step_result", "log"] = "log"
    data: dict[str, Any] = Field(default_factory=dict)


class ConfirmRequest(BaseModel):
    """Daemon asks the client to confirm a pending action.

    Forwarded from the graph's ``interrupt()`` payload by the daemon.
    """

    type: Literal["confirm_request"] = "confirm_request"
    task_id: str
    tier: int = 0
    summary: str = ""
    needs_password: bool = False
    typed_confirmation: str | None = None
    action_hash: str = ""
    untrusted: bool = False


class FinalMessage(BaseModel):
    """Task completed; includes the agent's final answer."""

    type: Literal["final"] = "final"
    task_id: str
    text: str = ""


class ErrorMessage(BaseModel):
    """Error communicating with or inside the daemon."""

    type: Literal["error"] = "error"
    code: str = "unknown"
    message: str = ""
    task_id: str | None = None


class StatusResponse(BaseModel):
    """Daemon status snapshot."""

    type: Literal["status_response"] = "status_response"
    daemon: str = "running"
    voice: str = "off"
    unlocked: bool = False
    queue: int = 0
    active_task: str | None = None
    version: str = PROTOCOL_VERSION


# ── Union type for parsing ───────────────────────────────────────────────

DaemonMessage = (
    AuthMessage
    | ChatMessage
    | ConfirmResponse
    | CancelMessage
    | StatusRequest
    | ShutdownMessage
    | VoiceToggleMessage
)
"""Union of all client→server message types (used for dispatch)."""
