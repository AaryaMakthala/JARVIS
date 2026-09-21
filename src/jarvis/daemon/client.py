"""Daemon client used by the CLI to communicate with a running daemon.

Connects over TCP to ``127.0.0.1:<port>`` (read from ``daemon.json``),
authenticates with the IPC token from keyring, and exchanges NDJSON messages.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from jarvis.config import daemon_runtime_file
from jarvis.daemon.protocol import (
    AuthMessage,
    AuthOk,
    CancelMessage,
    ChatMessage,
    ConfirmResponse,
    DaemonMessage,
    ErrorMessage,
    EventMessage,
    FinalMessage,
    ShutdownMessage,
    StatusRequest,
    StatusResponse,
)
from jarvis.logging_setup import get_logger
from jarvis.secrets import SecretStore

logger = get_logger("daemon.client")


class DaemonError(Exception):
    """Raised when the daemon returns an error or is unreachable."""

    def __init__(self, code: str = "unknown", message: str = "") -> None:
        self.code = code
        self.message = message
        super().__init__(message or code)


def _get_or_create_loop() -> asyncio.AbstractEventLoop:
    """Get or create an event loop that works on Python 3.10+.

    ``asyncio.get_event_loop()`` raises ``RuntimeError`` when no loop is
    set in the current thread (Python 3.10+).  This helper creates one
    lazily and stores it so repeated calls within the same thread reuse it.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None:
        return loop
    # No running loop; create or get a cached one for this thread
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("closed")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


class DaemonClient:
    """Synchronous client for the JARVIS daemon IPC socket.

    All methods use ``asyncio`` internally but provide a blocking public API
    so the CLI doesn't need ``async``/``await`` at every call site.
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int | None = None,
        token: str | None = None,
        store: SecretStore | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._token = token
        self._store = store or SecretStore()
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None

    def connect(self) -> None:
        """Connect to the daemon, authenticate, and wait for ``auth_ok``."""
        if self._port is None:
            self._port = self._read_port()
        if self._token is None:
            self._token = self._store.get("ipc_token") or ""
        if not self._token:
            raise DaemonError(code="no_token", message="no IPC token — run `jarvis init`")

        loop = _get_or_create_loop()
        loop.run_until_complete(self._async_connect())

    async def _async_connect(self) -> None:
        try:
            self._reader, self._writer = await asyncio.open_connection(self._host, self._port)
        except (ConnectionRefusedError, OSError) as exc:
            raise DaemonError(
                code="connection_refused",
                message=f"cannot connect to daemon on {self._host}:{self._port} — is it running?",
            ) from exc

        await self._send(AuthMessage(token=self._token))
        msg = await self._recv()
        if not isinstance(msg, AuthOk):
            if isinstance(msg, ErrorMessage):
                raise DaemonError(code=msg.code, message=msg.message)
            raise DaemonError(code="auth_failed", message="authentication failed")

    def close(self) -> None:
        """Close the connection."""
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                logger.debug("error closing client connection", exc_info=True)
            self._writer = None
            self._reader = None

    # ── send helpers ───────────────────────────────────────────────────

    def send_chat(self, text: str, *, source: str = "terminal", task_id: str = "") -> str:
        """Submit a chat message.  Returns the task_id assigned by the server."""
        msg = ChatMessage(id=task_id, text=text, source=source)
        resp = self._send_recv_sync(msg)
        if isinstance(resp, EventMessage):
            return resp.task_id
        if isinstance(resp, ErrorMessage):
            raise DaemonError(code=resp.code, message=resp.message)
        return task_id

    def send_confirm(
        self,
        task_id: str,
        *,
        approved: bool,
        action_hash: str,
        password: str | None = None,
        typed_confirmation: str | None = None,
    ) -> None:
        """Send a confirmation response."""
        msg = ConfirmResponse(
            task_id=task_id,
            approved=approved,
            action_hash=action_hash,
            password=password,
            typed_confirmation=typed_confirmation,
        )
        resp = self._send_recv_sync(msg)
        if isinstance(resp, ErrorMessage):
            raise DaemonError(code=resp.code, message=resp.message)

    def send_cancel(self, task_id: str) -> None:
        """Cancel a task."""
        msg = CancelMessage(task_id=task_id)
        resp = self._send_recv_sync(msg)
        if isinstance(resp, ErrorMessage):
            raise DaemonError(code=resp.code, message=resp.message)

    def send_shutdown(self) -> None:
        """Ask the daemon to shut down gracefully."""
        self._send_sync(ShutdownMessage())

    def get_status(self) -> StatusResponse:
        """Request daemon status."""
        resp = self._send_recv_sync(StatusRequest())
        if isinstance(resp, StatusResponse):
            return resp
        if isinstance(resp, ErrorMessage):
            raise DaemonError(code=resp.code, message=resp.message)
        raise DaemonError(code="unexpected", message=f"unexpected response: {type(resp)}")

    def wait_for_event(
        self, *, timeout: float = 300.0
    ) -> EventMessage | FinalMessage | ErrorMessage | None:
        """Block until the next event, final message, or error from the daemon."""
        loop = _get_or_create_loop()
        try:
            msg = loop.run_until_complete(asyncio.wait_for(self._recv(), timeout=timeout))
        except TimeoutError:
            return None
        if isinstance(msg, (EventMessage, FinalMessage, ErrorMessage)):
            return msg
        return None

    # ── low-level I/O ──────────────────────────────────────────────────

    def _send_sync(self, msg: Any) -> None:
        """Send a message synchronously (fire-and-forget)."""
        loop = _get_or_create_loop()
        loop.run_until_complete(self._send(msg))

    def _send_recv_sync(self, msg: Any) -> DaemonMessage:
        """Send a message and wait for the next response."""
        loop = _get_or_create_loop()
        loop.run_until_complete(self._send(msg))
        result = loop.run_until_complete(self._recv())
        if result is None:
            raise DaemonError(code="eof", message="daemon closed the connection")
        return result

    async def _send(self, msg: Any) -> None:
        if self._writer is None:
            raise DaemonError(code="not_connected", message="not connected to daemon")
        line = msg.model_dump_json() + "\n"
        self._writer.write(line.encode("utf-8"))
        await self._writer.drain()

    async def _recv(self) -> DaemonMessage | None:
        if self._reader is None:
            return None
        try:
            raw = await asyncio.wait_for(self._reader.readline(), timeout=300)
        except (TimeoutError, ConnectionError, OSError):
            return None
        if not raw:
            return None
        text = raw.decode("utf-8").strip()
        if not text:
            return None
        return _parse_server_message(text)

    # ── helpers ────────────────────────────────────────────────────────

    def _read_port(self) -> int:
        """Read the daemon port from ``daemon.json``."""
        pid_file = daemon_runtime_file()
        if not pid_file.is_file():
            raise DaemonError(
                code="no_daemon",
                message="daemon is not running (no daemon.json)",
            )
        try:
            data = json.loads(pid_file.read_text("utf-8"))
            return int(data["port"])
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
            raise DaemonError(
                code="bad_pid_file",
                message=f"corrupt daemon.json: {exc}",
            ) from exc


def _parse_server_message(text: str) -> DaemonMessage | None:
    """Parse a JSON line from the server."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    msg_type = data.get("type")
    if msg_type == "confirm_request":
        try:
            from jarvis.daemon.protocol import ConfirmRequest

            return ConfirmRequest.model_validate(data)
        except Exception:  # noqa: BLE001
            return None
    dispatch: dict[str, type] = {
        "event": EventMessage,
        "final": FinalMessage,
        "error": ErrorMessage,
        "status_response": StatusResponse,
        "auth_ok": AuthOk,
    }
    cls = dispatch.get(msg_type)
    if cls is None:
        return None
    try:
        return cls.model_validate(data)
    except Exception:  # noqa: BLE001
        return None
