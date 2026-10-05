"""Daemon client used by the CLI to communicate with a running daemon.

Connects over TCP to ``127.0.0.1:<port>`` (read from ``daemon.json``),
authenticates with the IPC token from keyring, and exchanges NDJSON messages.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, TypeGuard

from jarvis.config import daemon_runtime_file
from jarvis.daemon.protocol import (
    AuthMessage,
    AuthOk,
    CancelMessage,
    ChatMessage,
    ClarificationRequest,
    ClarificationResponse,
    ConfirmRequest,
    ConfirmResponse,
    DaemonMessage,
    ErrorMessage,
    EventMessage,
    FinalMessage,
    ShutdownMessage,
    StatusRequest,
    StatusResponse,
    VoiceToggleMessage,
)
from jarvis.logging_setup import get_logger

#: Everything the daemon can push at a client that is *not* a reply to the
#: request it just sent.  ``ack`` events are excluded on purpose: the client
#: consumes those itself while looking for the one it is waiting for.
ServerEvent = EventMessage | FinalMessage | ErrorMessage | ConfirmRequest | ClarificationRequest
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

    #: Upper bound on how many messages :meth:`_send_recv_ack` will set aside
    #: while looking for its acknowledgement.  A daemon that never acks must
    #: not turn into an unbounded memory growth; past this we report the
    #: mismatch instead.
    max_pending = 64

    #: Seconds to wait for an acknowledgement before giving up.  The server
    #: acks *before* it releases the worker, so this is generous: a longer
    #: silence means the daemon is wedged, and saying so beats hanging the
    #: terminal with no output.
    ack_timeout = 30.0

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
        #: Messages that arrived while a send was waiting for its ack.  They are
        #: replayed by :meth:`wait_for_event` rather than dropped.
        self._pending: list[DaemonMessage] = []
        #: Human-readable text of the last chat-submission ack ("started" /
        #: "queued"), for the CLI to show.  Empty until a chat is submitted.
        self.last_ack_message: str = ""

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
        self._pending.clear()
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                logger.debug("error closing client connection", exc_info=True)
            self._writer = None
            self._reader = None

    # ── send helpers ───────────────────────────────────────────────────

    def send_chat(self, text: str, *, source: str = "terminal", task_id: str = "") -> str:
        """Submit a chat message.  Returns the task_id assigned by the server.

        The ack is read with :meth:`_send_recv_ack`, so a ``ConfirmRequest``
        that the freshly started worker dispatches *before* the ack arrives is
        buffered and replayed by :meth:`wait_for_event` instead of being
        consumed and dropped here.  The ack's text ("started" / "queued") is
        kept in :attr:`last_ack_message` so the terminal can show it: a request
        that is *queued* is why a second command appears to do nothing, and that
        used to be invisible.
        """
        ack = self._send_recv_ack(ChatMessage(id=task_id, text=text, source=source))
        self.last_ack_message = str(ack.data.get("message") or "")
        return ack.task_id

    def send_confirm(
        self,
        task_id: str,
        *,
        approved: bool,
        action_hash: str,
        password: str | None = None,
        typed_confirmation: str | None = None,
        confirmation_id: str = "",
        plan_hash: str = "",
    ) -> None:
        """Send a confirmation response (echoing the I3 id/plan_hash)."""
        msg = ConfirmResponse(
            task_id=task_id,
            approved=approved,
            action_hash=action_hash,
            password=password,
            typed_confirmation=typed_confirmation,
            confirmation_id=confirmation_id,
            plan_hash=plan_hash,
        )
        self._send_recv_ack(msg)

    def send_clarification(self, task_id: str, *, answer: str) -> None:
        """Send the answered clarification text for a suspended task."""
        msg = ClarificationResponse(task_id=task_id, answer=answer or "")
        self._send_recv_ack(msg)

    def send_cancel(self, task_id: str, *, ack_timeout: float | None = None) -> None:
        """Cancel a task.

        ``ack_timeout`` defaults to :attr:`ack_timeout`.  A caller that is
        walking away (the CLI on Ctrl+C) passes a short one: the point is to
        release the daemon's slot, and the cancel is already on the wire before
        the receipt is read, so waiting a long time for it helps nobody.
        """
        self._send_recv_ack(CancelMessage(task_id=task_id), timeout=ack_timeout)

    def send_shutdown(self) -> None:
        """Ask the daemon to shut down gracefully."""
        self._send_sync(ShutdownMessage())

    def send_voice_toggle(self, state: bool) -> str:
        """Turn the daemon's voice pipeline on/off.  Returns the status text."""
        msg = VoiceToggleMessage(state=state)
        resp = self._send_recv_sync(msg)
        if isinstance(resp, EventMessage):
            return str(resp.data.get("message", "voice toggled"))
        if isinstance(resp, ErrorMessage):
            raise DaemonError(code=resp.code, message=resp.message)
        return "voice toggled"

    def get_status(self) -> StatusResponse:
        """Request daemon status."""
        resp = self._send_recv_sync(StatusRequest())
        if isinstance(resp, StatusResponse):
            return resp
        if isinstance(resp, ErrorMessage):
            raise DaemonError(code=resp.code, message=resp.message)
        raise DaemonError(code="unexpected", message=f"unexpected response: {type(resp)}")

    def wait_for_event(self, *, timeout: float = 300.0) -> ServerEvent | None:
        """Block until the next event, request, or error from the daemon.

        Requests (:class:`ConfirmRequest` / :class:`ClarificationRequest`)
        are returned to the caller so the CLI can answer them inline; they are
        matched by ``type`` here (not in the protocol union, which only covers
        client→server messages).

        Anything that arrived while a send was waiting for its ack is replayed
        from :attr:`_pending` first, in order, so no daemon message can be lost
        to the read that consumed the ack.
        """
        buffered = self._take_buffered()
        if buffered is not None:
            return buffered
        loop = _get_or_create_loop()
        try:
            msg = loop.run_until_complete(asyncio.wait_for(self._recv(), timeout=timeout))
        except TimeoutError:
            return None
        if isinstance(
            msg, (EventMessage, FinalMessage, ErrorMessage, ConfirmRequest, ClarificationRequest)
        ):
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

    def _send_recv_ack(self, msg: Any, *, timeout: float | None = None) -> EventMessage:
        """Send a request and return *its* ack, without ever losing a message.

        Every server reply the client waits for is an ``ack``: the chat
        submission ("started" / "queued") and the responses (confirmation /
        clarification / cancel).  The worker is released - and can dispatch the
        next ``ConfirmRequest``, a progress line, or the ``FinalMessage`` on the
        same connection - either before or after the ack, so the reply read here
        is not necessarily the ack: anything that is not the ack is buffered on
        :attr:`_pending` and handed back by :meth:`wait_for_event`, in order.

        Dropping it instead is the bug from ``663d559`` one layer down - the
        approval prompt silently disappears and the task then fails closed with
        "confirmation expired".  An ``ErrorMessage`` reply raises, as it always
        did; EOF, a missing ack (:attr:`ack_timeout` and :attr:`max_pending`)
        and a read that outlasts :attr:`ack_timeout` all raise too, so a caller
        can never mistake a silent daemon for a delivered request.
        """
        loop = _get_or_create_loop()
        loop.run_until_complete(self._send(msg))
        wait = self.ack_timeout if timeout is None else float(timeout)
        for _ in range(self.max_pending):
            try:
                result = loop.run_until_complete(asyncio.wait_for(self._recv(), timeout=wait))
            except TimeoutError as exc:
                # Anything already buffered is still the caller's: it is
                # replayed by wait_for_event, not thrown away with the error.
                raise DaemonError(
                    code="ack_timeout",
                    message=(
                        f"no acknowledgement for {type(msg).__name__} within"
                        f" {wait:g}s; daemon is not responding"
                    ),
                ) from exc
            if result is None:
                raise DaemonError(code="eof", message="daemon closed the connection")
            if isinstance(result, ErrorMessage):
                raise DaemonError(code=result.code, message=result.message)
            if _is_ack(result, msg):
                return result
            self._pending.append(result)
        raise DaemonError(
            code="no_ack",
            message=f"no acknowledgement for {type(msg).__name__}; daemon is not responding",
        )

    def _take_buffered(self) -> ServerEvent | None:
        """Pop the next messageable buffered message, skipping junk."""
        while self._pending:
            candidate = self._pending.pop(0)
            if isinstance(
                candidate,
                (EventMessage, FinalMessage, ErrorMessage, ConfirmRequest, ClarificationRequest),
            ):
                return candidate
        return None

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


def _is_ack(msg: DaemonMessage, sent: Any) -> TypeGuard[EventMessage]:
    """True when ``msg`` is the server's acknowledgement of ``sent``.

    The server answers every request the client waits for with an ``ack`` event.
    Matching on the dedicated ``ack`` kind and not on ``log`` matters: a
    progress line about the same task must not be mistaken for the
    acknowledgement, or the real ack would then arrive as a stray event on the
    next read.

    When the client supplied its own id the ack must match it.  When it did not
    (``ChatMessage.id`` empty, so the server assigns one) the first ``ack`` on
    the connection is necessarily ours: this client sends one request at a time
    and blocks until each is acknowledged.
    """
    if not isinstance(msg, EventMessage) or msg.kind != "ack":
        return False
    requested = str(getattr(sent, "task_id", "") or getattr(sent, "id", "") or "")
    return not requested or msg.task_id == requested


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
    if msg_type == "clarification_request":
        try:
            from jarvis.daemon.protocol import ClarificationRequest

            return ClarificationRequest.model_validate(data)
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
