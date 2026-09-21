"""Daemon server: asyncio TCP on 127.0.0.1 with token auth, task queue, and worker thread.

Runs under ``pythonw`` on Windows: stdout/stderr are ``None``.  All output goes
to the JSONL log file.  Single-instance enforced via ``daemon.json`` pid check.

Protocol: newline-delimited JSON (NDJSON), UTF-8, max 64 KB per line.
First message must be ``auth`` with the IPC token from keyring.

Architecture
------------
The LangGraph agent runs in a **worker thread** (it is synchronous).  The
asyncio event loop handles I/O.  Communication between the two is via
:class:`threading.Event` objects: the worker signals when it needs attention
(confirm request, completion) and the loop signals when a response arrives.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from jarvis.config import Settings, daemon_runtime_file, load_settings
from jarvis.daemon.protocol import (
    MAX_MESSAGE_SIZE,
    AuthMessage,
    AuthOk,
    CancelMessage,
    ChatMessage,
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
from jarvis.policy.unlock import UnlockManager
from jarvis.secrets import SecretStore
from jarvis.tools.base import CancelToken

try:
    from jarvis.voice.service import VoiceService
except ImportError:
    VoiceService = None  # type: ignore[assignment,misc]

logger = get_logger("daemon.server")

#: Maximum tasks waiting in the queue (not counting the active one).
MAX_QUEUE_SIZE = 5

#: Stale confirmation timeout (seconds).
CONFIRMATION_TIMEOUT_S = 60

#: How often to check for stale confirmations (seconds).
STALE_CHECK_INTERVAL_S = 10

#: Sentinel owner id for tasks submitted by the voice pipeline.  These have no
#: IPC client socket; results are read back by the blocked voice thread
#: directly (see :meth:`DaemonServer._voice_submit`).
VOICE_OWNER = "__voice__"


# ── NDJSON framing ──────────────────────────────────────────────────────


def _parse_message(text: str) -> DaemonMessage | None:
    """Parse a JSON line into a known client→server message."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    msg_type = data.get("type")
    dispatch: dict[str, type] = {
        "auth": AuthMessage,
        "chat": ChatMessage,
        "confirm_response": ConfirmResponse,
        "cancel": CancelMessage,
        "status": StatusRequest,
        "shutdown": ShutdownMessage,
        "voice_toggle": VoiceToggleMessage,
    }
    cls = dispatch.get(msg_type)
    if cls is None:
        return None
    try:
        return cls.model_validate(data)
    except Exception:  # noqa: BLE001
        return None


# ── Client connection wrapper ────────────────────────────────────────────


@dataclass
class ClientConnection:
    """A single authenticated client session."""

    writer: asyncio.StreamWriter
    reader: asyncio.StreamReader
    conn_id: str = ""  # stable identity used for task ownership
    authenticated: bool = False
    active_task: str | None = None  # task_id this client owns
    closed: bool = False

    async def send(self, msg: Any) -> None:
        """Write a Pydantic model as one NDJSON line."""
        if self.closed:
            return
        try:
            line = msg.model_dump_json() + "\n"
            self.writer.write(line.encode("utf-8"))
            await self.writer.drain()
        except (ConnectionError, OSError):
            self.closed = True

    async def recv(self) -> DaemonMessage | None:
        """Read one NDJSON line; return ``None`` on EOF or oversized."""
        try:
            raw = await asyncio.wait_for(self.reader.readline(), timeout=300)
        except (TimeoutError, ConnectionError, OSError):
            return None
        if not raw:
            return None
        if len(raw) > MAX_MESSAGE_SIZE:
            return None
        text = raw.decode("utf-8").strip()
        if not text:
            return None
        return _parse_message(text)

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            try:
                self.writer.close()
            except Exception:
                logger.debug("error closing writer", exc_info=True)


# ── Task slot ────────────────────────────────────────────────────────────


@dataclass
class TaskSlot:
    """Tracks one task's lifecycle: queued → active → (confirming) → done."""

    task_id: str
    text: str
    source: str
    # conn_id of the client that submitted the task (set at acceptance so it
    # survives queueing and promotion — the dispatch loop uses it to route
    # confirm requests and final results back to the right client).
    owner_id: str | None = None
    # Worker-thread signals
    event: threading.Event = field(default_factory=threading.Event)
    confirm_payload: dict[str, Any] | None = None
    confirm_tier: int = 0  # tier of the pending confirmation (for IPC guarding)
    resume_answer: dict[str, Any] | None = None
    result_text: str | None = None
    error: str | None = None
    done: bool = False
    # Timestamps
    started_at: float = 0.0
    confirm_sent_at: float = 0.0


@dataclass
class _VoiceOutcome:
    """Minimal outcome object returned to the voice loop by ``_voice_submit``.

    Confirmation is always resolved inside the worker before this is returned,
    which is why ``confirmation`` is fixed to ``None``.
    """

    final_answer: str | None = None
    error: str | None = None
    confirmation: None = None


# ── Server ───────────────────────────────────────────────────────────────


class DaemonServer:
    """The main daemon.  Construct with settings, then call :meth:`run`."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        store: SecretStore | None = None,
        ctx: Any = None,  # AppContext — built lazily if None
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self._settings = settings or load_settings()
        self._store = store or SecretStore()
        self._ctx = ctx
        self._host = host or self._settings.daemon.host
        self._port = port or self._settings.daemon.port
        self._token = self._store.get("ipc_token") or ""
        self._unlock = UnlockManager(self._store, settings=self._settings)
        self._clients: dict[str, ClientConnection] = {}
        self._queue: list[TaskSlot] = []
        self._active: TaskSlot | None = None
        self._active_cancel = CancelToken()
        self._lock = threading.Lock()
        self._shutdown_event = asyncio.Event()
        self._task_event = asyncio.Event()  # signals the loop: "check _active"
        # The loop is captured in _serve(); worker threads wake the dispatch
        # loop via _wake_dispatch(), which schedules on the loop thread.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._pid_file: Any = None
        self._voice_service: Any = None  # VoiceService | None

    # ── public API ──────────────────────────────────────────────────────

    def run(self) -> None:
        """Start the server and block until shutdown."""
        self._redirect_std()
        self._install_excepthooks()
        if not self._check_single_instance():
            return
        self._pid_file = daemon_runtime_file()
        self._pid_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            asyncio.run(self._serve())
        finally:
            self._cleanup_pid_file()

    # ── stdio redirect for pythonw ─────────────────────────────────────

    def _redirect_std(self) -> None:
        """Redirect stdout/stderr to the log file when running under pythonw."""
        if sys.stdout is None or sys.stderr is None:
            from jarvis.config import log_file

            log_path = log_file()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = open(log_path, "a", encoding="utf-8")  # noqa: SIM115
            sys.stdout = fh
            sys.stderr = fh

    def _install_excepthooks(self) -> None:
        """Log unhandled exceptions instead of silently dying."""

        def _hook(exc_type: type, exc_value: BaseException, tb: Any) -> None:
            logger.error("unhandled exception", exc_info=(exc_type, exc_value, tb))

        sys.excepthook = _hook

        def _thread_hook(args: threading.excepthook_args) -> None:
            logger.error(
                "unhandled exception in thread %s",
                args.thread.name,
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )

        threading.excepthook = _thread_hook

    # ── single instance ────────────────────────────────────────────────

    def _check_single_instance(self) -> bool:
        """Return ``True`` if we should run; ``False`` if another daemon is alive."""
        pid_file = daemon_runtime_file()
        if not pid_file.is_file():
            return True
        try:
            data = json.loads(pid_file.read_text("utf-8"))
            old_pid = int(data.get("pid", 0))
        except (json.JSONDecodeError, ValueError, KeyError, TypeError):
            return True
        if old_pid <= 0 or old_pid == os.getpid():
            return True
        try:
            os.kill(old_pid, 0)
            logger.error("another daemon is running (pid %d); exiting", old_pid)
            return False
        except OSError:
            return True

    def _write_pid_file(self, port: int) -> None:
        data = {"pid": os.getpid(), "port": port, "started_at": time.time()}
        self._pid_file.write_text(json.dumps(data), encoding="utf-8")

    def _cleanup_pid_file(self) -> None:
        try:
            if self._pid_file and self._pid_file.is_file():
                data = json.loads(self._pid_file.read_text("utf-8"))
                if int(data.get("pid", 0)) == os.getpid():
                    self._pid_file.unlink(missing_ok=True)
        except Exception:
            logger.debug("error cleaning pid file", exc_info=True)

    # ── main serve loop ────────────────────────────────────────────────

    async def _serve(self) -> None:
        """Accept connections, manage the task queue, clean up stale confirmations."""
        # Initialise the voice service first so the agent context can be wired
        # to its live loop (dictation tools need ``ctx.voice``).
        self._bootstrap_voice()

        if self._ctx is None:
            self._ctx = self._build_default_context()

        server = await asyncio.start_server(
            self._handle_client,
            self._host,
            self._port,
        )
        assigned_port = server.sockets[0].getsockname()[1]
        self._write_pid_file(assigned_port)
        logger.info("daemon listening on %s:%d (pid=%d)", self._host, assigned_port, os.getpid())

        # Background tasks
        stale_task = asyncio.create_task(self._stale_confirmation_loop())
        dispatch_task = asyncio.create_task(self._dispatch_loop())

        # Graceful shutdown on SIGTERM/SIGINT
        loop = asyncio.get_running_loop()
        self._loop = loop
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, self._shutdown_event.set)
            except NotImplementedError:
                pass

        await self._shutdown_event.wait()
        stale_task.cancel()
        dispatch_task.cancel()
        server.close()
        await server.wait_closed()
        for conn in list(self._clients.values()):
            conn.close()
        self._stop_voice()
        logger.info("daemon shut down cleanly")

    def _bootstrap_voice(self) -> None:
        """Initialise the VoiceService and honour ``[voice] enabled`` (boot).

        Idempotent and safe to call from tests/shareholders; building the
        service may be slow (backend discovery) so it is done once at startup.
        A failed auto-start moves the service into ``error`` state but never
        stops the daemon — text/CLI operation continues.
        """
        self._voice_service = None
        if self._settings.voice.enabled and VoiceService is not None:
            try:
                self._voice_service = self._build_voice_service()
            except Exception:
                logger.warning("could not initialise voice service", exc_info=True)

        if self._voice_service is not None:
            try:
                msg = self._voice_service.start()
                if self._service_state(self._voice_service) == "error":
                    code = self._service_error_code(self._voice_service) or "loop-crashed"
                    logger.warning("voice auto-start failed: %s (%s)", code, msg)
                else:
                    logger.info("voice auto-start: %s", msg)
            except Exception:
                logger.warning("failed to auto-start voice", exc_info=True)

    def _build_default_context(self) -> Any:
        """Build an AppContext with the current settings."""
        from jarvis.agent.context import VoiceToolsFacade, make_app_context

        groq_key = self._store.get("groq_api_key") or ""
        llm = None
        if groq_key:
            try:
                from jarvis.llm.client import GroqClient

                llm = GroqClient(groq_key, self._settings)
            except Exception:  # noqa: BLE001
                logger.warning("could not create GroqClient")
        return make_app_context(
            self._settings,
            llm=llm,
            unlock=self._unlock,
            voice=VoiceToolsFacade(service=self._voice_service),
        )

    def _build_voice_service(self) -> Any:
        """Build a VoiceService with lazy-imported backends."""
        from jarvis.voice.service import VoiceService

        audio = None
        wake = None
        stt = None
        tts = None
        focus = None

        try:
            from jarvis.voice.audio_input import create as create_audio_input

            audio = create_audio_input(
                sample_rate=16_000,
                channels=1,
                input_device=self._settings.voice.input_device,
            )
        except Exception:
            logger.debug("microphone audio unavailable", exc_info=True)

        try:
            from jarvis.voice.wake import create as create_wake

            wake = create_wake(model_name=self._settings.voice.wake_word)
        except Exception:
            logger.debug("wake-word unavailable", exc_info=True)

        try:
            from jarvis.voice.stt import create as create_stt

            stt = create_stt(model_size=self._settings.voice.stt_model)
        except Exception:
            logger.debug("STT unavailable", exc_info=True)

        try:
            from jarvis.voice.tts import create as create_tts

            tts = create_tts(backend=self._settings.voice.tts_backend)
        except Exception:
            logger.debug("TTS unavailable", exc_info=True)

        try:
            from jarvis.voice.focus import WindowFocusChecker

            focus = WindowFocusChecker()
        except Exception:
            logger.debug("focus checker unavailable", exc_info=True)

        return VoiceService(
            audio=audio,
            wake_detector=wake,
            stt=stt,
            tts=tts,
            focus_checker=focus,
            submit_task=self._voice_submit,
            wake_word=self._settings.voice.wake_word,
            listen_timeout_s=self._settings.voice.listen_timeout_s,
            idle_timeout_s=self._settings.voice.idle_timeout_s,
            max_session_s=self._settings.voice.max_session_s,
            max_dictation_chars=self._settings.voice.max_dictation_chars,
        )

    def _stop_voice(self) -> None:
        """Stop the voice pipeline on shutdown / toggle-off (idempotent)."""
        if self._voice_service is None:
            return
        try:
            self._voice_service.stop()
        except Exception:
            logger.warning("error stopping voice service", exc_info=True)

    @staticmethod
    def _service_state(service: Any) -> str:
        """Read the voice state machine (defaults to the legacy active flag)."""
        state = getattr(service, "state", None)
        if state is None:
            return "on" if service.is_active() else "off"
        return state

    @staticmethod
    def _service_error_code(service: Any) -> str | None:
        """Read the fixed voice error code, else ``None``."""
        if getattr(service, "state", None) != "error":
            return None
        return getattr(service, "error_code", None) or None

    def _refresh_ctx_voice_bridge(self) -> None:
        """Point ``ctx.voice`` at the current voice service.

        The context is built once, but a lazily-created toggle-on service must
        reach the agent's dictation tools through the same facade.
        """
        ctx = self._ctx
        if ctx is None:
            return
        voice = getattr(ctx, "voice", None)
        if voice is not None and hasattr(voice, "service"):
            voice.service = self._voice_service

    # ── dispatch loop: monitors the worker thread ──────────────────────

    async def _dispatch_loop(self) -> None:
        """Watch ``_active`` for state changes and forward to the right client."""
        while True:
            await self._task_event.wait()
            self._task_event.clear()

            slot = self._active
            if slot is None:
                continue

            # Find the client that owns this task
            owner = self._find_owner(slot.task_id)
            if owner is None:
                if slot.done:
                    # The submitting client is gone; the result can never be
                    # delivered.  Drop it (logged) instead of stalling the
                    # whole queue behind an undeliverable task.
                    logger.warning("result for %s dropped: client disconnected", slot.task_id)
                    self._promote_next()
                # Not done yet: wait for the worker/confirm flow to progress
                continue

            if slot.confirm_payload is not None and not slot.done:
                # Forward confirmation to the client
                req = ConfirmRequest(
                    task_id=slot.task_id,
                    tier=slot.confirm_payload.get("tier", 0),
                    summary=slot.confirm_payload.get("summary", ""),
                    needs_password=slot.confirm_payload.get("needs_unlock", False),
                    typed_confirmation=slot.confirm_payload.get("typed_confirmation"),
                    action_hash=slot.confirm_payload.get("action_hash", ""),
                    untrusted=slot.confirm_payload.get("untrusted", False),
                )
                await owner.send(req)
                slot.confirm_sent_at = time.time()
                slot.confirm_payload = None  # consumed

            elif slot.done:
                if slot.error:
                    await owner.send(
                        FinalMessage(task_id=slot.task_id, text=f"Error: {slot.error}")
                    )
                elif slot.result_text:
                    await owner.send(FinalMessage(task_id=slot.task_id, text=slot.result_text))
                else:
                    await owner.send(FinalMessage(task_id=slot.task_id, text="(no answer)"))
                # Promote next queued task
                self._promote_next()
                owner.active_task = None

    def _find_owner(self, task_id: str) -> ClientConnection | None:
        """Find the authenticated client that owns ``task_id``.

        Ownership is recorded on the :class:`TaskSlot` when the task is
        accepted — whether it started immediately or was queued — so it
        survives promotion to active.  The ``conn.active_task`` marker is
        kept as a fallback.
        """
        with self._lock:
            slot = self._active
            owner_id = slot.owner_id if slot is not None and slot.task_id == task_id else None
        if owner_id is not None:
            for conn in self._clients.values():
                if conn.authenticated and conn.conn_id == owner_id:
                    return conn
        for conn in self._clients.values():
            if conn.authenticated and conn.active_task == task_id:
                return conn
        return None

    def _promote_next(self) -> None:
        """Move the next queued task to active and start its worker thread."""
        if self._queue:
            next_slot = self._queue.pop(0)
            next_slot.started_at = time.time()
            self._active = next_slot
            threading.Thread(target=self._worker_run, args=(next_slot,), daemon=True).start()
        else:
            self._active = None

    def _wake_dispatch(self) -> None:
        """Wake the dispatch loop from a worker thread.

        ``asyncio.Event.set()`` only appends the waiters' wakeup to the loop's
        ready queue — it does not interrupt a ``select()`` sleep, so calling it
        directly from a non-loop thread leaves the loop waiting until some
        unrelated event happens to wake it.  ``loop.call_soon_threadsafe`` is
        the documented cross-thread interface: it writes to the loop's
        self-pipe/socketpair so ``select()`` returns immediately.

        When called from the loop thread itself (e.g. the stale-confirmation
        loop) there is no sleeping selector to break, so we set the event
        directly and skip the scheduling round-trip.
        """
        loop = self._loop
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None  # worker thread: no running loop in this thread
        if loop is not None and running is not loop:
            loop.call_soon_threadsafe(self._task_event.set)
        else:
            self._task_event.set()

    # ── client handler ─────────────────────────────────────────────────

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        addr = writer.get_extra_info("peername")
        conn_id = f"{addr[0]}:{addr[1]}" if addr else "unknown"
        conn = ClientConnection(writer=writer, reader=reader, conn_id=conn_id)
        self._clients[conn_id] = conn

        try:
            # Require auth as first message
            msg = await conn.recv()
            if not isinstance(msg, AuthMessage):
                conn.close()
                return
            if not hmac.compare_digest(msg.token, self._token):
                logger.warning("auth failed from %s", conn_id)
                await asyncio.sleep(0.5)
                conn.close()
                return
            conn.authenticated = True
            await conn.send(AuthOk())
            logger.info("client authenticated: %s", conn_id)

            # Command loop
            while not conn.closed:
                msg = await conn.recv()
                if msg is None:
                    break
                await self._dispatch(msg, conn)
        except (ConnectionError, OSError):
            pass
        finally:
            conn.close()
            self._clients.pop(conn_id, None)
            logger.info("client disconnected: %s", conn_id)

    async def _dispatch(self, msg: DaemonMessage, conn: ClientConnection) -> None:
        match msg:
            case ChatMessage() as chat:
                await self._handle_chat(chat, conn)
            case ConfirmResponse() as conf:
                await self._handle_confirm(conf, conn)
            case CancelMessage() as cancel:
                await self._handle_cancel(cancel, conn)
            case StatusRequest():
                await self._handle_status(conn)
            case ShutdownMessage():
                self._shutdown_event.set()
            case VoiceToggleMessage() as toggle:
                await self._handle_voice_toggle(toggle, conn)
            case _:
                await conn.send(
                    ErrorMessage(code="unknown_type", message="unrecognised message type")
                )

    # ── voice toggle (jarvis on / off) ─────────────────────────────────

    async def _handle_voice_toggle(self, msg: VoiceToggleMessage, conn: ClientConnection) -> None:
        """Turn voice listening on/off at the daemon level.

        The CLI no longer runs a private voice loop; it asks the daemon so
        `jarvis on`/`off` control the same pipeline that the agent's dictation
        tools talk to through ``ctx.voice``.
        """
        if not msg.state:
            self._stop_voice()
            await conn.send(
                EventMessage(task_id="", kind="log", data={"message": "voice deactivated"})
            )
            return

        if not self._settings.voice.enabled:
            await conn.send(
                ErrorMessage(
                    code="voice_disabled",
                    message="voice is disabled in config — set [voice] enabled = true",
                )
            )
            return

        if self._voice_service is None:
            try:
                self._voice_service = self._build_voice_service()
            except Exception:
                logger.warning("could not initialise voice service", exc_info=True)
            if self._voice_service is None:
                await conn.send(
                    ErrorMessage(
                        code="voice_unavailable",
                        message="voice dependencies are not installed",
                    )
                )
                return
            self._refresh_ctx_voice_bridge()

        text = self._voice_service.start()
        if self._service_state(self._voice_service) == "error":
            code = self._service_error_code(self._voice_service) or "loop-crashed"
            await conn.send(ErrorMessage(code=code, message=text))
        else:
            await conn.send(EventMessage(task_id="", kind="log", data={"message": text}))

    # ── chat ───────────────────────────────────────────────────────────

    async def _handle_chat(self, msg: ChatMessage, conn: ClientConnection) -> None:
        """Accept a new task or queue it."""
        task_id = msg.id or f"t{int(time.time() * 1000)}"
        slot = TaskSlot(
            task_id=task_id,
            text=msg.text,
            source=msg.source,
        )

        with self._lock:
            if self._active is not None and not self._active.done:
                if len(self._queue) >= MAX_QUEUE_SIZE:
                    await conn.send(
                        ErrorMessage(
                            code="queue_full",
                            message=f"queue is full (max {MAX_QUEUE_SIZE})",
                        )
                    )
                    return
                slot.owner_id = conn.conn_id
                self._queue.append(slot)
                await conn.send(
                    EventMessage(
                        task_id=task_id,
                        kind="log",
                        data={"message": "queued", "position": len(self._queue)},
                    )
                )
            else:
                # Start immediately
                slot.started_at = time.time()
                slot.owner_id = conn.conn_id
                self._active = slot
                conn.active_task = task_id
                threading.Thread(target=self._worker_run, args=(slot,), daemon=True).start()

    # ── voice task submission (runs in the voice loop thread) ──────────

    def _voice_submit(self, text: str, source: str = "voice") -> Any:
        """Submit a task from the voice-loop thread and block until it finishes.

        Voice tasks have ``owner_id == VOICE_OWNER`` because there is no IPC
        socket to route confirmations/results back to; the blocked voice thread
        reads the outcome straight off the :class:`TaskSlot`.  Confirmations
        for voice tasks are driven inside the worker thread by
        :meth:`_run_voice_confirmation` (Tier 1 only), never over IPC.
        """
        task_id = f"v{int(time.time() * 1000)}"
        slot = TaskSlot(task_id=task_id, text=text, source=source, owner_id=VOICE_OWNER)

        with self._lock:
            if self._active is not None and not self._active.done:
                if len(self._queue) >= MAX_QUEUE_SIZE:
                    return _VoiceOutcome(error="queue is full — try again later")
                self._queue.append(slot)
            else:
                slot.started_at = time.time()
                self._active = slot
                threading.Thread(target=self._worker_run, args=(slot,), daemon=True).start()

        while not slot.done:
            slot.event.wait(0.25)
            slot.event.clear()

        if slot.error:
            return _VoiceOutcome(error=slot.error)
        return _VoiceOutcome(final_answer=slot.result_text)

    def _ctx_voice_loop(self) -> Any | None:
        """The live voice loop driving confirmations, or ``None``."""
        svc = self._voice_service
        if svc is not None:
            try:
                if svc.is_active() and svc.loop is not None:
                    return svc.loop
            except Exception:
                logger.debug("voice service not reportable", exc_info=True)
        ctx = self._ctx
        voice = getattr(ctx, "voice", None)
        if voice is not None:
            for attr in ("active_loop", "loop"):
                value = getattr(voice, attr, None)
                if callable(value):
                    value = value()
                if value is not None:
                    return value
        return None

    def _run_voice_confirmation(self, slot: TaskSlot, payload: dict[str, Any]) -> None:
        """Drive a Tier-1 confirmation by voice from the worker thread.

        The speech dialogue runs on the audio device while the voice-loop
        thread is blocked inside :meth:`_voice_submit`, so the audio stream is
        not contended.  Fails closed: without a live loop, or when the action
        cannot be confirmed by voice (Tier 2+ / typed folder names), the
        answer is a refusal and the graph's own policy gate denies the step.
        """
        loop = self._ctx_voice_loop()
        if loop is None or not getattr(loop, "is_active", lambda: True)():
            logger.warning("voice confirmation dropped: no active voice loop")
            slot.resume_answer = {"approved": False, "action_hash": payload.get("action_hash", "")}
            slot.confirm_payload = None
            slot.event.set()
            return

        can_confirm = getattr(loop, "can_confirm_by_voice", lambda p: False)
        if not can_confirm(payload):
            slot.resume_answer = {"approved": False, "action_hash": payload.get("action_hash", "")}
            slot.confirm_payload = None
            slot.event.set()
            return

        def responder(answer: dict[str, Any]) -> Any:
            slot.resume_answer = answer
            slot.event.set()
            return None

        try:
            loop.confirm_by_voice(payload, on_confirmation=responder)
        except Exception:
            logger.exception("voice confirmation failed; refusing")
            slot.resume_answer = {"approved": False, "action_hash": payload.get("action_hash", "")}
        finally:
            slot.confirm_payload = None
            slot.event.set()

    # ── worker thread ──────────────────────────────────────────────────

    def _worker_run(self, slot: TaskSlot) -> None:
        """Run the LangGraph task in a background thread.

        The flow is:
        1. Call ``run_task`` → blocks until graph pauses at ``interrupt()`` or finishes.
        2. If paused → store the confirmation payload and signal the event loop.
        3. Wait for the event loop to deliver the user's answer.
        4. Call ``resume_task`` → blocks until the graph pauses again or finishes.
        5. Repeat until done.
        """
        from jarvis.agent import open_sqlite_checkpointer, resume_task, run_task
        from jarvis.config import checkpoints_db

        saver = open_sqlite_checkpointer(str(checkpoints_db()))
        task_id = slot.task_id

        try:
            outcome = run_task(
                self._ctx,
                saver,
                slot.text,
                source=slot.source,
                thread_id=task_id,
            )

            while True:
                if outcome.error:
                    slot.error = outcome.error
                    slot.done = True
                    self._wake_dispatch()
                    if slot.owner_id == VOICE_OWNER:
                        slot.event.set()
                    return

                if outcome.final_answer and not outcome.confirmation:
                    slot.result_text = outcome.final_answer
                    slot.done = True
                    self._wake_dispatch()
                    if slot.owner_id == VOICE_OWNER:
                        slot.event.set()
                    return

                if outcome.confirmation:
                    if slot.source == "voice":
                        # Voice confirmations are driven here, on the worker
                        # thread, while the voice-loop thread waits on
                        # ``slot.event`` inside _voice_submit().
                        self._run_voice_confirmation(slot, outcome.confirmation)
                        slot.event.clear()
                        answer = slot.resume_answer
                        slot.resume_answer = None
                        if answer is None:
                            answer = {
                                "approved": False,
                                "action_hash": outcome.confirmation.get("action_hash", ""),
                            }
                        outcome = resume_task(self._ctx, saver, task_id, answer)
                        continue

                    # Signal the loop: "here's a confirmation request"
                    slot.confirm_payload = outcome.confirmation
                    slot.confirm_tier = int(outcome.confirmation.get("tier", 0) or 0)
                    slot.event.clear()
                    self._wake_dispatch()

                    # Wait for the event loop to deliver the answer
                    slot.event.wait(timeout=CONFIRMATION_TIMEOUT_S)

                    if slot.event.is_set() and slot.resume_answer is not None:
                        answer = slot.resume_answer
                        slot.resume_answer = None
                        slot.confirm_payload = None
                        outcome = resume_task(self._ctx, saver, task_id, answer)
                    else:
                        # Timeout or cancelled
                        slot.error = "confirmation timed out"
                        slot.done = True
                        self._wake_dispatch()
                        if slot.owner_id == VOICE_OWNER:
                            slot.event.set()
                        return
                else:
                    # No confirmation and no final answer — shouldn't happen, but be safe
                    slot.done = True
                    self._wake_dispatch()
                    return

        except Exception as exc:  # noqa: BLE001
            slot.error = f"{type(exc).__name__}: {exc}"
            slot.done = True
            self._wake_dispatch()
            if slot.owner_id == VOICE_OWNER:
                slot.event.set()
            logger.error("worker for %s failed: %s", task_id, slot.error)

    # ── confirmation response ──────────────────────────────────────────

    async def _handle_confirm(self, msg: ConfirmResponse, conn: ClientConnection) -> None:
        """Deliver the user's answer to the waiting worker thread."""
        slot = self._active
        if slot is None or slot.task_id != msg.task_id or slot.done:
            await conn.send(
                ErrorMessage(
                    code="no_such_task",
                    message=f"no active task awaiting confirmation: {msg.task_id!r}",
                    task_id=msg.task_id,
                )
            )
            return

        # Tier 2+ can never be approved by voice (docs/03 §7.8): anything that
        # claims a voice origin for a Tier 2+ action is refused here, on top of
        # the refusal inside the worker's own voice-confirmation path.
        if msg.source == "voice" and slot.confirm_tier >= 2:
            await conn.send(
                ErrorMessage(
                    code="tier_requires_terminal",
                    message="this action requires terminal confirmation — it cannot be approved by voice",
                    task_id=msg.task_id,
                )
            )
            return

        # Password verification for Tier 2 (daemon layer, never the graph)
        if msg.password is not None and not self._unlock.unlock(msg.password):
            await conn.send(
                ErrorMessage(
                    code="wrong_password",
                    message="wrong password or session locked out",
                    task_id=msg.task_id,
                )
            )
            return

        # Build resume answer (password never enters the graph — invariant 8)
        answer: dict[str, Any] = {
            "approved": msg.approved,
            "action_hash": msg.action_hash,
        }
        if msg.typed_confirmation is not None:
            answer["typed_confirmation"] = msg.typed_confirmation

        slot.resume_answer = answer
        slot.event.set()  # wake the worker thread

    # ── cancel ─────────────────────────────────────────────────────────

    async def _handle_cancel(self, msg: CancelMessage, conn: ClientConnection) -> None:
        """Cancel a running or queued task."""
        with self._lock:
            if self._active and self._active.task_id == msg.task_id:
                self._active_cancel.cancel()
                self._active.error = "cancelled by user"
                self._active.done = True
                self._active.event.set()  # wake worker if waiting
                self._task_event.set()
                await conn.send(
                    EventMessage(task_id=msg.task_id, kind="log", data={"message": "cancelled"})
                )
                return
            for i, slot in enumerate(self._queue):
                if slot.task_id == msg.task_id:
                    self._queue.pop(i)
                    await conn.send(
                        EventMessage(
                            task_id=msg.task_id, kind="log", data={"message": "removed from queue"}
                        )
                    )
                    return
        await conn.send(
            ErrorMessage(
                code="no_such_task",
                message=f"no task with id {msg.task_id!r}",
                task_id=msg.task_id,
            )
        )

    # ── status ─────────────────────────────────────────────────────────

    async def _handle_status(self, conn: ClientConnection) -> None:
        with self._lock:
            queue_len = len(self._queue)
            active_id = self._active.task_id if self._active and not self._active.done else None
        voice_status = "off"
        voice_reason: str | None = None
        if self._voice_service is not None:
            state = self._service_state(self._voice_service)
            if state == "error":
                voice_status = "error"
                voice_reason = self._service_error_code(self._voice_service)
            elif state == "starting":
                voice_status = "starting"
            elif state == "on":
                voice_status = "on"
        await conn.send(
            StatusResponse(
                daemon="running",
                voice=voice_status,
                voice_reason=voice_reason,
                unlocked=self._unlock.is_unlocked(),
                queue=queue_len,
                active_task=active_id,
            )
        )

    # ── stale confirmation cleanup ─────────────────────────────────────

    async def _stale_confirmation_loop(self) -> None:
        """Periodically expire confirmations waiting too long."""
        while True:
            await asyncio.sleep(STALE_CHECK_INTERVAL_S)
            now = time.time()
            slot = self._active
            if (
                slot is not None
                and slot.confirm_payload is not None
                and not slot.done
                and slot.confirm_sent_at > 0
                and (now - slot.confirm_sent_at) > CONFIRMATION_TIMEOUT_S
            ):
                logger.warning("confirmation for %s expired", slot.task_id)
                slot.error = "confirmation timed out"
                slot.done = True
                slot.event.set()
                self._task_event.set()
