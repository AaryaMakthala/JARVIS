"""IPC wire protocol: typed Pydantic messages for client↔server communication.

Transport: TCP on ``127.0.0.1``, newline-delimited JSON (NDJSON), UTF-8.
Max message size: 64 KB.  First message must be ``auth`` with the token
from keyring (constant-time compare via ``hmac.compare_digest``).

Client → server: :class:`AuthMessage`, :class:`ChatMessage`,
:class:`ConfirmResponse`, :class:`ClarificationResponse`,
:class:`CancelMessage`, :class:`StatusRequest`, :class:`ShutdownMessage`,
:class:`VoiceToggleMessage`.

Server → client: :class:`AuthOk`, :class:`EventMessage`,
:class:`ConfirmRequest`, :class:`ClarificationRequest`,
:class:`FinalMessage`, :class:`ErrorMessage`, :class:`StatusResponse`.

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
    "ClarificationRequest",
    "ClarificationResponse",
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


class ClarificationResponse(BaseModel):
    """User's answer to a clarification request.

    The answer is free text (never a password or confirmation flag).  An empty
    string is valid on the wire but fails closed in the agent's clarify node
    ("No clarification received.").
    """

    type: Literal["clarification_response"] = "clarification_response"
    task_id: str
    answer: str = ""


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


class ClarificationRequest(BaseModel):
    """Daemon asks the client for a clarification while a task is suspended.

    Forwarded from the clarify node's ``interrupt()`` payload by the daemon.
    The client replies with :class:`ClarificationResponse`; the answer is
    resumed into the clarify node (empty answer fails closed).
    """

    type: Literal["clarification_request"] = "clarification_request"
    task_id: str
    question: str = ""


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
    """Daemon status snapshot.

    ``voice`` is one of the state-machine states the daemon reports for
    the voice pipeline: ``off`` | ``starting`` | ``on`` | ``error``.
    ``voice_reason`` carries the fixed error code (see
    ``jarvis.voice.service.VOICE_ERROR_CODES``) exactly when
    ``voice == "error"``, so ``jarvis status`` can surface the same reason
    that ``jarvis on`` reported.
    """

    type: Literal["status_response"] = "status_response"
    daemon: str = "running"
    voice: str = "off"
    voice_reason: str | None = None
    unlocked: bool = False
    queue: int = 0
    active_task: str | None = None
    version: str = PROTOCOL_VERSION


# ── Union type for parsing ───────────────────────────────────────────────

DaemonMessage = (
    AuthMessage
    | ChatMessage
    | ConfirmResponse
    | ClarificationResponse
    | CancelMessage
    | StatusRequest
    | ShutdownMessage
    | VoiceToggleMessage
)
"""Union of all client→server message types (used for dispatch)."""
