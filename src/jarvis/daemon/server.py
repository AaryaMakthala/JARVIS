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
from pathlib import Path
from typing import Any

import psutil

from jarvis import logging_setup
from jarvis.config import Settings, daemon_runtime_file, load_settings, log_file
from jarvis.daemon.confirmations import ConfirmationRegistry, plan_hash
from jarvis.daemon.protocol import (
    MAX_MESSAGE_SIZE,
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
from jarvis.daemon.task_runtime import TaskRuntime, TerminalSink, VoiceSink, timeout_answer
from jarvis.logging_setup import get_logger
from jarvis.memory import open_memory
from jarvis.policy.unlock import UnlockManager
from jarvis.secrets import SecretStore
from jarvis.tools.base import Cancelled, CancelToken
from jarvis.voice.tier2_voice import KEY_VOICE_TIER2
from jarvis.voice.tier2_voice import allowed as tier2_allowed

try:
    from jarvis.voice.service import VoiceService
except ImportError:
    VoiceService = None  # type: ignore[assignment,misc]

logger = get_logger("daemon.server")

#: Maximum tasks waiting in the queue (not counting the active one).
MAX_QUEUE_SIZE = 5

#: How often to check for stale confirmations (seconds).  The timeout value
#: itself is owned by ``jarvis.daemon.task_runtime`` (D1): the worker and this
#: sweeper both drain through ``self._task_runtime`` so there is exactly one
#: window.
STALE_CHECK_INTERVAL_S = 10

#: Sentinel owner id for tasks submitted by the voice pipeline.  These have no
#: IPC client socket; results are read back by the blocked voice thread
#: directly (see :meth:`DaemonServer._voice_submit`).
VOICE_OWNER = "__voice__"

#: Poll slice used by the voice loop while waiting for the worker, and the
#: cadence of the "still waiting" heartbeat (logged every 30 s).
_VOICE_POLL_INTERVAL_S = 0.25

#: Maximum wall time the voice-loop thread may block waiting for a worker to
#: finish a voice task.  Without this bound a hung worker would wedge the
#: voice loop on the first command and no further wake word would ever be
#: answered (the old 30 s "still waiting" heartbeat only logged; it never
#: returned).  A normal task finishes far sooner; on timeout the voice loop
#: speaks a failure and re-arms instead of going deaf.
VOICE_SUBMIT_TIMEOUT_S = 180.0


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
        "clarification_response": ClarificationResponse,
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
    #: This task's own cancellation token.  The graph checks it before entering
    #: every node, so a cancel stops the run at the next step instead of leaving
    #: the remaining plan to execute.  Per-slot on purpose: a cancelled run can
    #: still be unwinding when the next task is promoted, and a single shared
    #: token would let one task's cancel leak into another's.
    cancel: CancelToken = field(default_factory=CancelToken)
    #: True once a cancellation was requested (voice "stop" or IPC cancel).
    cancelled: bool = False
    # Worker-thread signals
    event: threading.Event = field(default_factory=threading.Event)
    confirm_payload: dict[str, Any] | None = None
    confirm_tier: int = 0  # tier of the pending confirmation (for IPC guarding)
    #: The pending confirmation's own payload, kept after ``confirm_payload`` is
    #: consumed for dispatch.  Stage 3 needs it to re-verify a voice Tier 2
    #: approval (``tool``/``allowed``/``readback_args``); the registry check
    #: still runs first and is what binds the id.
    confirm_evidence: dict[str, Any] | None = None
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
    which is why ``confirmation`` is fixed to ``None``.  ``cancelled`` is set
    when the task was stopped on request (a spoken "stop" or a shutdown) and the
    loop answered with its own fixed acknowledgement instead of an error.
    """

    final_answer: str | None = None
    error: str | None = None
    confirmation: None = None
    cancelled: bool = False


# ── Server ───────────────────────────────────────────────────────────────


def _looks_like_jarvis_daemon(name: str, cmdline: list[str]) -> bool:
    """Return True if a process name + cmdline match a JARVIS daemon.

    The daemon is launched as ``python -m jarvis daemon`` (or the ``jarvis``
    console script).  We look for a python interpreter name plus an explicit
    ``jarvis`` + ``daemon`` marker on the command line.  This guards the
    single-instance check against pid reuse: a recycled pid owned by an
    unrelated process must not be mistaken for a running daemon.
    """
    if "python" not in name and "jarvis" not in name:
        return False
    text = " ".join(cmdline)
    return "jarvis" in text and "daemon" in text


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
        self._task_runtime = TaskRuntime()
        #: I3 confirmation correlation.  In-memory, daemon-transient; never checkpointed.
        self._confirmations = ConfirmationRegistry()
        self._clients: dict[str, ClientConnection] = {}
        self._queue: list[TaskSlot] = []
        self._active: TaskSlot | None = None
        #: Set for the whole of a voice-pipeline shutdown (Ctrl+C, `jarvis off`).
        #: The voice-loop thread can be blocked waiting for an in-flight task, and
        #: that wait only unwinds when it can *see* that the pipeline is going
        #: down - without it ``VoiceLoop.stop()`` waits out its 10 s join and
        #: reports "loop thread did not stop within 10s".
        self._voice_stopping = threading.Event()
        self._lock = threading.Lock()
        self._shutdown_event = asyncio.Event()
        self._task_event = asyncio.Event()  # signals the loop: "check _active"
        # The loop is captured in _serve(); worker threads wake the dispatch
        # loop via _wake_dispatch(), which schedules on the loop thread.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._pid_file: Any = None
        self._voice_service: Any = None  # VoiceService | None
        self._memory: Any = None  # SqliteMemory | None (Phase 8, opened lazily)
        self._app_registry: Any = None  # ToolRegistry | None (built with the context)

    # ── public API ──────────────────────────────────────────────────────

    def run(self) -> None:
        """Start the server and block until shutdown."""
        # Redirect first, then report: under ``pythonw`` ``sys.stdout`` is
        # ``None``, and every console write below would otherwise raise.  This
        # ordering also means the "log file" line lands in the log too.
        self._redirect_std()
        # Attach the JSONL file handler unconditionally: the daemon is the long
        # lived process that must prove the voice pipeline stage-by-stage, and
        # the console script may bypass cli.main() (where logging is normally
        # configured). configure_logging() is idempotent per log file.
        logging_setup.configure_logging()
        logging_setup.console(f"[DAEMON] log file -> {log_file()}")
        self._install_excepthooks()
        if not self._check_single_instance():
            logger.info("daemon: another instance is already alive; exiting")
            return
        self._pid_file = daemon_runtime_file()
        self._pid_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            asyncio.run(self._serve())
        except KeyboardInterrupt:
            logger.info("daemon: interrupted by the user")
            raise
        except BaseException:
            logger.exception("daemon: unhandled failure while serving")
            raise
        finally:
            self._close_memory()
            self._cleanup_pid_file()
        logger.info("daemon: stopped cleanly")

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
            if not psutil.pid_exists(old_pid):
                return self._take_stale_pid_file(pid_file, old_pid)
            # pid is alive — but is it really a JARVIS daemon, or a recycled pid
            # now owned by an unrelated process after reboot?
            return self._check_foreign_pid(old_pid, pid_file)
        except OSError as exc:  # pragma: no cover - defensive; unlink may race
            logger.debug("could not clean stale daemon.json: %s", exc, exc_info=True)
            return True

    def _take_stale_pid_file(self, pid_file: Path, old_pid: int) -> bool:
        """Dead/reused pid: remove the stale runtime file and allow startup."""
        logger.warning(
            "stale daemon.json: pid %d is not alive; removing stale runtime file",
            old_pid,
        )
        try:
            pid_file.unlink(missing_ok=True)
        except OSError as exc:
            logger.debug("could not remove stale daemon.json: %s", exc, exc_info=True)
        return True

    def _check_foreign_pid(self, old_pid: int, pid_file: Path) -> bool:
        """Decide whether an alive pid is a live JARVIS daemon.

        Confirms the pid belongs to a JARVIS process (name + cmdline marker) to
        avoid mistaking pid reuse after reboot for a running daemon.

        - pid confirmed as a JARVIS daemon -> refuse (return ``False``)
        - pid died in the race between ``pid_exists`` and ``Process`` -> stale (``True``)
        - pid is alive but its identity cannot be read -> fail closed (``False``)
        - pid is alive but is *not* JARVIS (pid reuse) -> stale (``True``)
        """
        try:
            proc = psutil.Process(old_pid)
            name = proc.name().lower()
            cmdline = [c.lower() for c in proc.cmdline()]
        except psutil.NoSuchProcess:
            # Race: died between pid_exists() and Process().
            return self._take_stale_pid_file(pid_file, old_pid)
        except psutil.AccessDenied:
            # Elevated / other-user process: we cannot confirm it is a JARVIS
            # daemon, so do not assume it is safe to start a second one.
            logger.error(
                "pid %d is alive but its identity cannot be read; "
                "assuming another daemon is running",
                old_pid,
            )
            return False

        if _looks_like_jarvis_daemon(name, cmdline):
            logger.error("another daemon is running (pid %d); exiting", old_pid)
            return False

        # Alive but not a JARVIS daemon: pid reuse after reboot.
        logger.warning(
            "daemon.json pid %d is not a JARVIS daemon process; treating as stale",
            old_pid,
        )
        return self._take_stale_pid_file(pid_file, old_pid)

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
        self._initialize_runtime()

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
        logger.info("daemon ready — serving until shutdown is requested")

        try:
            await self._shutdown_event.wait()
        finally:
            # Guaranteed cleanup: a client disconnect or worker failure must
            # never stop the daemon.  The shutdown path is isolated so no
            # exception can escape ``asyncio.run`` and kill the process.
            for task in (stale_task, dispatch_task):
                task.cancel()
            try:
                await asyncio.gather(stale_task, dispatch_task, return_exceptions=True)
            except Exception:
                logger.debug("background tasks did not cancel cleanly", exc_info=True)
            server.close()
            await server.wait_closed()
            for conn in list(self._clients.values()):
                conn.close()
            self._stop_voice()
            logger.info("daemon shut down cleanly")

    def _initialize_runtime(self) -> None:
        """Build the agent context and wire the voice service safely."""
        if self._ctx is None:
            self._ctx = self._build_default_context()
        self._bootstrap_voice()
        self._refresh_ctx_voice_bridge()

    def _bootstrap_voice(self) -> None:
        """Initialise the VoiceService and honour ``[voice] enabled`` (boot).

        Idempotent and safe to call from tests/shareholders; building the
        service may be slow (backend discovery) so it is done once at startup.
        A failed auto-start moves the service into ``error`` state but never
        stops the daemon — text/CLI operation continues.
        """
        self._voice_service = None
        logger.info(
            "voice bootstrap: enabled=%s service_available=%s",
            self._settings.voice.enabled,
            bool(VoiceService),
        )
        if self._settings.voice.enabled and VoiceService is not None:
            try:
                self._voice_service = self._build_voice_service()
            except Exception:
                logger.warning("could not initialise voice service", exc_info=True)

        if self._voice_service is not None:
            self._refresh_ctx_voice_bridge()
            self._voice_stopping.clear()
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
        from jarvis.llm.provider import build_llm_client
        from jarvis.ml.risk import build_classifier

        selection = build_llm_client(self._settings, self._store, logger=logger)
        llm = selection.client
        if llm is None:
            logger.warning("no LLM provider available: %s", "; ".join(selection.reasons))
        elif selection.reasons:
            logger.info(
                "LLM provider %s selected (skipped: %s)",
                selection.info.name if selection.info else "unknown",
                "; ".join(selection.reasons),
            )
        return make_app_context(
            self._settings,
            llm=llm,
            registry=self._registry(),
            memory=self._open_memory(),
            unlock=self._unlock,
            classifier=build_classifier(self._settings, self._registry(), logger=logger),
            voice=VoiceToolsFacade(service=self._voice_service),
        )

    def _registry(self) -> Any:
        """The tool registry the daemon's context is built with.

        Built once and shared with the risk classifier, so the classifier's
        per-tool priors come from the *same* specs the engine will decide on
        (there is no second copy of the tier table to drift).
        """
        if self._app_registry is None:
            from jarvis.tools.registry import build_default_registry

            self._app_registry = build_default_registry(self._settings)
        return self._app_registry

    def _open_memory(self) -> Any:
        """Open the local memory store once, for the daemon's whole lifetime.

        Phase 8: the daemon is where memory actually pays off — the same
        verified task asked by voice twice now retrieves the earlier approach.
        One connection is shared by the worker threads (it is lock-protected in
        :class:`~jarvis.memory.store.SqliteMemory`), opened lazily so a daemon
        that fails to start never creates the file, and closed on shutdown.
        """
        if self._memory is None:
            try:
                self._memory = open_memory(settings=self._settings, logger=logger)
            except Exception:
                logger.warning("could not open the memory store", exc_info=True)
                self._memory = None
        return self._memory

    def _close_memory(self) -> None:
        """Close the memory connection if one was opened."""
        memory, self._memory = self._memory, None
        if memory is None:
            return
        try:
            close = getattr(memory, "close", None)
            if callable(close):
                close()
        except Exception:
            logger.debug("closing the memory store failed", exc_info=True)

    def _build_voice_service(self) -> Any:
        """Build a VoiceService with lazy-imported backends.

        Every backend is optional, so each factory is probed independently and
        a failure only disables that component (surfacing later as a fixed
        ``VOICE_ERROR_CODES`` entry from :class:`VoiceService`).  Diagnostics go
        to the log, never to ``print``: the daemon also runs under ``pythonw``,
        where ``sys.stdout`` is ``None`` and a print would raise.
        """
        from jarvis.voice.service import VoiceService

        audio = None
        wake = None
        stt = None
        tts = None
        focus = None
        logger.info(
            "voice backends: silence_timeout_s=%.2f max_segment_s=%.1f "
            "rearm_quiet_gate_s=%.2f confirm_window_s=%.1f max_spoken_chars=%d "
            "bare_stop_while_busy=%s",
            self._settings.voice.silence_timeout_s,
            self._settings.voice.max_segment_s,
            self._settings.voice.rearm_quiet_gate_s,
            self._settings.voice.confirm_window_s,
            self._settings.voice.max_spoken_chars,
            self._settings.voice.bare_stop_while_busy,
        )

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

            wake = create_wake(
                model_name=self._settings.voice.wake_word,
                threshold=self._settings.voice.wake_threshold,
            )
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
            auto_idle_timeout_s=self._settings.voice.auto_idle_timeout_s,
            max_session_s=self._settings.voice.max_session_s,
            max_dictation_chars=self._settings.voice.max_dictation_chars,
            silence_timeout_s=self._settings.voice.silence_timeout_s,
            max_segment_s=self._settings.voice.max_segment_s,
            silence_threshold=self._settings.voice.silence_threshold,
            rearm_quiet_gate_s=self._settings.voice.rearm_quiet_gate_s,
            confirm_window_s=self._settings.voice.confirm_window_s,
            bare_stop_while_busy=self._settings.voice.bare_stop_while_busy,
            max_spoken_chars=self._settings.voice.max_spoken_chars,
            report_status=self._settings.voice.status,
            echo_transcript=self._settings.voice.echo_transcript,
            voice_settings=self._settings.voice,
            routing_report=self._llm_routing_report,
        )

    def _llm_routing_report(self) -> Any:
        """Describe which provider/model served the most recent LLM call.

        Read only, and only safe labels: the provider name, the model id, a
        latency, and a failure-category string.  A missing or unconfigured LLM
        returns ``None`` so the voice console simply omits the ``[LLM]`` lines
        instead of guessing.  This never raises — the voice loop treats a
        failure here as "no routing metadata".
        """
        llm = getattr(self._ctx, "llm", None) if self._ctx is not None else None
        meta = getattr(llm, "last_call", None)
        if meta is None:
            return None
        try:
            from jarvis.voice.status import report_from_call_meta

            return report_from_call_meta(meta)
        except Exception:
            logger.debug("routing report could not be built", exc_info=True)
            return None

    def _stop_voice(self) -> None:
        """Stop the voice pipeline on shutdown / toggle-off (idempotent).

        ``_voice_stopping`` is set first: the voice-loop thread may be blocked
        waiting for an in-flight task (see :meth:`_voice_submit`), and that wait
        only unwinds once it can see the pipeline is going down.
        """
        self._voice_stopping.set()
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
        """Watch ``_active`` for state changes and forward to the right client.

        Each iteration is isolated: an unexpected exception must never kill
        the background task (that would silently stop all task delivery and,
        once ``_serve`` cancels the task at shutdown, could even surface as a
        daemon crash).  On failure the loop logs, recovers the queue, and
        keeps serving.
        """
        while True:
            await self._task_event.wait()
            self._task_event.clear()
            try:
                await self._dispatch_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # never let a bad iteration kill the loop
                logger.exception("dispatch loop iteration failed")
                self._recover_dispatch_after_error()

    async def _dispatch_once(self) -> None:
        """Handle one dispatch event; ``return`` where the loop used to ``continue``."""
        slot = self._active
        if slot is None:
            return

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
            return

        if slot.confirm_payload is not None and not slot.done:
            payload = slot.confirm_payload
            if payload.get("type") == "clarification":
                req: ConfirmRequest | ClarificationRequest = ClarificationRequest(
                    task_id=slot.task_id,
                    question=payload.get("question") or "",
                )
            else:
                req = ConfirmRequest(
                    task_id=slot.task_id,
                    tier=payload.get("tier", 0),
                    summary=payload.get("summary", ""),
                    needs_password=payload.get("needs_unlock", False),
                    typed_confirmation=payload.get("typed_confirmation"),
                    action_hash=payload.get("action_hash", ""),
                    confirmation_id=payload.get("confirmation_id", ""),
                    plan_hash=payload.get("plan_hash", ""),
                    untrusted=payload.get("untrusted", False),
                )
            await owner.send(req)
            slot.confirm_sent_at = time.time()
            slot.confirm_payload = None  # consumed

        elif slot.done:
            if slot.error:
                await owner.send(FinalMessage(task_id=slot.task_id, text=f"Error: {slot.error}"))
            elif slot.result_text:
                await owner.send(FinalMessage(task_id=slot.task_id, text=slot.result_text))
            else:
                await owner.send(FinalMessage(task_id=slot.task_id, text="(no answer)"))
            # Promote next queued task
            self._promote_next()
            owner.active_task = None

    def _recover_dispatch_after_error(self) -> None:
        """After a failed dispatch iteration, unblock the queue if possible."""
        if self._active is not None and self._active.done:
            self._promote_next()

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
                try:
                    await self._dispatch(msg, conn)
                except (ConnectionError, OSError):
                    break
                except Exception:  # one bad message never kills the session
                    logger.exception("error handling message from %s", conn_id)
                    try:
                        await conn.send(
                            ErrorMessage(code="internal_error", message="internal error")
                        )
                    except Exception:
                        logger.debug("error sending error reply", exc_info=True)
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
            case ClarificationResponse() as clar:
                await self._handle_clarification(clar, conn)
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
        self._voice_stopping.clear()
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
                        kind="ack",
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
                # Ack every accepted request (the queued branch above already
                # does).  ``send_chat`` blocks on the next server message to
                # learn the task_id; with no ack it would receive the first
                # *downstream* message instead - e.g. the ConfirmRequest -
                # and silently discard it, so the confirmation prompt never
                # reached the terminal and the answer window elapsed.
                await conn.send(
                    EventMessage(
                        task_id=task_id,
                        kind="ack",
                        data={"message": "started"},
                    )
                )

    # ── voice task submission (runs in the voice loop thread) ──────────

    def _voice_submit(self, text: str, source: str = "voice") -> Any:
        """Submit a task from the voice-loop thread and block until it finishes.

        Voice tasks have ``owner_id == VOICE_OWNER`` because there is no IPC
        socket to route confirmations/results back to; the blocked voice thread
        reads the outcome straight off the :class:`TaskSlot`.  Confirmations
        for voice tasks are driven inside the worker thread by
        :meth:`_run_voice_confirmation` (Tier 1 only), never over IPC.
        """
        if self._ctx is None:
            logger.warning("voice task rejected: runtime context is not initialised")
            return _VoiceOutcome(error="AI backend is starting — try again")
        task_id = f"v{int(time.time() * 1000)}"
        slot = TaskSlot(task_id=task_id, text=text, source=source, owner_id=VOICE_OWNER)

        with self._lock:
            if self._active is not None and not self._active.done:
                if len(self._queue) >= MAX_QUEUE_SIZE:
                    logger.warning("voice task rejected: queue full (task_id=%s)", task_id)
                    return _VoiceOutcome(error="queue is full — try again later")
                self._queue.append(slot)
                logger.info(
                    "voice task queued (task_id=%s, queue_len=%d)", task_id, len(self._queue)
                )
            else:
                slot.started_at = time.time()
                self._active = slot
                threading.Thread(target=self._worker_run, args=(slot,), daemon=True).start()
                logger.info("voice task started (task_id=%s)", task_id)

        started = time.monotonic()
        deadline = started + VOICE_SUBMIT_TIMEOUT_S
        waited = 0.0
        while not slot.done:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                # A hung worker must never wedge the voice loop: return a
                # failure so the interaction speaks an error and re-arms.
                logger.warning(
                    "voice task timed out after %.0fs (task_id=%s); un-wedging the voice loop",
                    VOICE_SUBMIT_TIMEOUT_S,
                    task_id,
                )
                return _VoiceOutcome(error="command timed out — try again")
            if self._voice_stopping.is_set():
                # The pipeline is going down (Ctrl+C / `jarvis off`).  Do not hold
                # the loop thread for the rest of the timeout: cancel the task so
                # the worker stops at its next node boundary, and let
                # VoiceLoop.stop() join the thread promptly.
                logger.info(
                    "voice task abandoned: the voice pipeline is stopping (task_id=%s)", task_id
                )
                self._cancel_voice_task(slot)
                return _VoiceOutcome(cancelled=True)
            watched, stop_requested = self._watch_for_stop(
                slot, min(_VOICE_POLL_INTERVAL_S, remaining)
            )
            if stop_requested:
                logger.info(
                    "voice task cancelled by a spoken stop request (task_id=%s, elapsed=%.1fs)",
                    task_id,
                    time.monotonic() - started,
                )
                self._cancel_voice_task(slot)
                return _VoiceOutcome(cancelled=True)
            if not watched:
                # Nothing to watch (no live loop, a confirmation owns the mic,
                # or the detector failed): wait on the slot instead.  One of the
                # two always bounds the iteration, so this wait loop can never
                # spin the voice-loop thread.
                slot.event.wait(min(_VOICE_POLL_INTERVAL_S, max(remaining, 0.001)))
                slot.event.clear()
            waited += _VOICE_POLL_INTERVAL_S
            if waited >= 30.0:
                logger.info(
                    "voice task still waiting (task_id=%s, elapsed=%.1fs)",
                    task_id,
                    time.monotonic() - started,
                )
                waited = 0.0

        if slot.error:
            logger.warning("voice task errored (task_id=%s, error=%s)", task_id, slot.error)
            return _VoiceOutcome(error=slot.error)
        logger.info(
            "voice task finished (task_id=%s, elapsed_s=%.1f final_answer_chars=%d)",
            task_id,
            time.monotonic() - started,
            len(slot.result_text or ""),
        )
        return _VoiceOutcome(final_answer=slot.result_text)

    def _cancel_voice_task(self, slot: TaskSlot) -> None:
        """Stop the in-flight task at its next node boundary (idempotent).

        Sets the slot's own token - the graph checks it before entering every
        node - so nothing further is planned, and no further tool runs.  The
        caller (the voice-loop thread) does **not** wait for the worker: waiting
        is what wedges the loop when a provider call is slow.  The worker
        finishes its run, sees the token, flags the slot and wakes dispatch.
        """
        slot.cancelled = True
        slot.cancel.cancel()
        slot.done = True
        slot.event.set()
        self._wake_dispatch()

    def _can_watch_for_stop(self, slot: TaskSlot) -> bool:
        """Whether the voice loop may listen for a "stop" right now.

        False when there is no live loop, and false while a confirmation or
        clarification is pending: that dialogue runs on the worker thread, and
        the microphone must have exactly one reader.
        """
        if slot.done or slot.confirm_payload is not None:
            return False
        loop = self._ctx_voice_loop()
        return callable(getattr(loop, "watch_for_stop", None))

    def _watch_for_stop(self, slot: TaskSlot, budget_s: float) -> tuple[bool, bool]:
        """Listen for a spoken cancel request for one bounded slice.

        The audio itself is read by the voice loop (it owns the microphone on
        this thread); this only lends it the slot's end condition, so a
        confirmation starting on the worker thread or a finished task stops the
        watch within one frame.  Returns ``(watched, stop_requested)`` exactly as
        :meth:`VoiceLoop.watch_for_stop` does, with ``False, False`` whenever the
        watch could not run - the caller then waits on the slot instead, which is
        what keeps both the wait loop and the shutdown path bounded.
        """
        if not self._can_watch_for_stop(slot):
            return False, False
        loop = self._ctx_voice_loop()
        if loop is None:
            return False, False
        try:
            watched, stop_requested = loop.watch_for_stop(
                budget_s,
                keep_watching=lambda: slot.confirm_payload is None and not slot.done,
            )
            return bool(watched), bool(stop_requested)
        except Exception:  # a failed watch must never fail the task
            logger.warning("stop watch failed; continuing the task", exc_info=True)
            return False, False

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

    def _tier2_voice_ok(self, slot: TaskSlot) -> tuple[bool, str]:
        """Whether this slot's pending confirmation may be approved by voice.

        Delegates every condition to
        :func:`jarvis.voice.tier2_voice.allowed` (flag, Tier exactly 2, named
        tool, not blocked, no typed folder name, exact readback) using the
        payload the graph's own gate produced.  Returns ``(ok, reason)``; the
        reason is for the log.  Pure delegation - no policy decision is made
        here, and nothing is unlocked.
        """
        payload = slot.confirm_evidence or {}
        return tier2_allowed(
            payload.get("tool"),
            payload,
            self._settings,
            args=payload.get("readback_args"),
        )

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

        confirmation_id = str(payload.get("confirmation_id") or "")
        plan_hash_value = str(payload.get("plan_hash") or "")

        def responder(answer: dict[str, Any]) -> Any:
            # I3: the daemon holds the pending id, so a spoken approval is
            # checked against the exact confirmation before it can resume.
            if isinstance(answer, dict) and answer.get("approved"):
                check = self._confirmations.check(
                    confirmation_id,
                    plan_hash_value,
                    str(answer.get("action_hash") or payload.get("action_hash") or ""),
                )
                if check.ok:
                    self._confirmations.resolve(confirmation_id)
                    # Stage 3: mark a voice-approved Tier 2 answer for the gate,
                    # which re-verifies it independently.  The registry check
                    # above has already run and passed.
                    if int(payload.get("tier", 0) or 0) >= 2:
                        ok, reason = tier2_allowed(
                            payload.get("tool"),
                            payload,
                            self._settings,
                            args=payload.get("readback_args"),
                        )
                        if ok:
                            answer[KEY_VOICE_TIER2] = True
                        else:
                            logger.info("voice tier 2 answer not marked: %s", reason)
                else:
                    logger.warning("voice confirmation rejected: %s", check.reason)
                    answer = {"approved": False, "action_hash": answer.get("action_hash")}
            slot.resume_answer = answer
            slot.event.set()
            return None

        try:
            loop.confirm_by_voice(
                payload, on_confirmation=responder, transcript=slot.text
            )
        except Exception:
            logger.exception("voice confirmation failed; refusing")
            slot.resume_answer = {"approved": False, "action_hash": payload.get("action_hash", "")}
        finally:
            slot.confirm_payload = None
            slot.event.set()

    def _run_voice_clarification(self, slot: TaskSlot, payload: dict[str, Any]) -> None:
        """Capture a clarification answer by voice from the worker thread.

        Mirror of :meth:`_run_voice_confirmation` for the clarify interrupt:
        the audio dialogue runs on the worker thread while the voice-loop
        thread waits inside ``_voice_submit``, and the captured free text is
        stashed on ``slot.resume_answer``.  Fails closed: an empty answer (no
        live loop, no utterance, or no ``capture_free_text`` support) is
        resumed into the clarify node, which reports a clean "No clarification
        received." refusal.
        """
        loop = self._ctx_voice_loop()
        if loop is None or not getattr(loop, "is_active", lambda: True)():
            logger.warning("voice clarification dropped: no active voice loop")
            slot.resume_answer = ""
            slot.confirm_payload = None
            slot.event.set()
            return

        capture = getattr(loop, "capture_free_text", None)
        if capture is None:
            logger.warning("voice clarification dropped: loop cannot capture free text")
            slot.resume_answer = ""
            slot.event.set()
            return

        try:
            answer = capture(payload.get("question") or "Please clarify.")
            slot.resume_answer = answer if isinstance(answer, str) else ""
        except Exception:
            logger.exception("voice clarification failed; answering empty (fail closed)")
            slot.resume_answer = ""
        finally:
            slot.confirm_payload = None
            slot.event.set()

    def _sink_for(self, slot: TaskSlot) -> TerminalSink | VoiceSink:
        """The response sink that offers this task's interrupts to the user.

        Voice-sourced tasks are answered on the audio device (Tier-1 voice
        only for confirmations); everything else goes over IPC to the owner
        connection and is answered by the CLI/dialog client.
        """
        if slot.source == "voice":
            return VoiceSink(
                runtime=self._task_runtime,
                confirm=self._run_voice_confirmation,
                clarify=self._run_voice_clarification,
            )
        return TerminalSink(runtime=self._task_runtime)

    # ── worker thread ──────────────────────────────────────────────────

    def _worker_run(self, slot: TaskSlot) -> None:
        """Run the LangGraph task in a background thread.

        The flow is:
        1. Call ``run_task`` → blocks until graph pauses at ``interrupt()`` or finishes.
        2. If paused → store the interrupt payload and signal the event loop.
        3. Offer it through the task's response sink (terminal IPC or voice)
           and wait (bounded by ``TaskRuntime``) for the user's answer.
        4. Call ``resume_task`` → blocks until the graph pauses again or finishes.
        5. Repeat until done.

        A timeout is **not** an abandoned task: the graph is resumed with a
        timed-out refusal (D1) and finishes with an honest message, instead of
        the worker dropping the slot and leaving the checkpoint orphaned.
        """
        from jarvis.agent import open_sqlite_checkpointer, resume_task, run_task
        from jarvis.config import checkpoints_db

        task_id = slot.task_id

        try:
            saver = open_sqlite_checkpointer(str(checkpoints_db()))
            outcome = run_task(
                self._ctx,
                saver,
                slot.text,
                source=slot.source,
                thread_id=task_id,
                cancel=slot.cancel,
            )

            while True:
                if slot.cancel.cancelled:
                    self._finish_cancelled(slot)
                    return
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
                    payload = dict(outcome.confirmation)
                    kind = outcome.interrupt_kind  # "confirm" | "clarification"
                    # I3: issue the correlation id before publishing, so the
                    # client answers against the exact confirmation it saw.
                    # issue() supersedes this task's older pending ids.
                    if kind == "confirm":
                        # Honour a batch plan_hash carried in the payload (Stage 3)
                        # rather than recomputing a single-step one: the agent may
                        # publish one interrupt covering several Tier 1 steps.
                        carry_plan_hash = str(payload.get("plan_hash") or "")
                        step_ids = payload.get("eligible") or []
                        if (
                            kind == "confirm"
                            and carry_plan_hash
                            and isinstance(step_ids, list)
                            and len(step_ids) >= 2
                        ):
                            pending = self._confirmations.issue(
                                slot.task_id,
                                str(payload.get("step_id") or ""),
                                int(payload.get("tier", 0) or 0),
                                str(payload.get("action_hash") or ""),
                                carry_plan_hash,
                            )
                        else:
                            pending = self._confirmations.issue(
                                slot.task_id,
                                str(payload.get("step_id") or ""),
                                int(payload.get("tier", 0) or 0),
                                str(payload.get("action_hash") or ""),
                                plan_hash([str(payload.get("action_hash") or "")]),
                            )
                        payload["confirmation_id"] = pending.confirmation_id
                        payload["plan_hash"] = pending.plan_hash
                        payload["expires_at"] = pending.expires_at
                    # Signal the loop: "here's an interrupt for the client".
                    slot.confirm_payload = payload
                    slot.confirm_tier = int(payload.get("tier", 0) or 0)
                    slot.confirm_evidence = payload
                    # Clear the answer event BEFORE waking dispatch so a reply
                    # that lands right after the signal can never be missed.
                    slot.event.clear()
                    self._wake_dispatch()

                    sink = self._sink_for(slot)
                    if kind == "clarification":
                        answer_value = sink.clarify(slot, payload)
                        slot.confirm_payload = None
                        if slot.done:
                            return  # cancelled while waiting (slot already finished)
                        # Fail closed: a timed-out / missing answer resumes as
                        # empty text and the clarify node reports a clean
                        # "No clarification received." refusal.
                        resume_value: Any = answer_value if isinstance(answer_value, str) else ""
                    else:
                        answer_value = sink.confirm(slot, payload)
                        slot.confirm_payload = None
                        if slot.done:
                            return  # cancelled while waiting (slot already finished)
                        resume_value = answer_value
                    outcome = resume_task(
                        self._ctx, saver, task_id, resume_value, cancel=slot.cancel
                    )
                    continue
                else:
                    # No confirmation and no final answer — shouldn't happen, but be safe
                    slot.done = True
                    self._wake_dispatch()
                    return

        except Cancelled:
            # The user cancelled while a node was running: the run stopped at a
            # node boundary, so no further step of the plan executed.  The
            # canceller owns the wording and the slot flags, so nothing is
            # rewritten here.
            self._finish_cancelled(slot)

        except Exception as exc:  # noqa: BLE001
            slot.error = f"{type(exc).__name__}: {exc}"
            slot.done = True
            self._wake_dispatch()
            if slot.owner_id == VOICE_OWNER:
                slot.event.set()
            logger.error("worker for %s failed: %s", task_id, slot.error)

    def _finish_cancelled(self, slot: TaskSlot) -> None:
        """Book-keep a task that stopped at a cancellation boundary.

        Deliberately does not touch ``slot.error`` / ``slot.result_text``: the
        canceller already decided the wording (the voice path answers with its
        own fixed acknowledgement, the IPC path with "cancelled by user").
        """
        slot.cancelled = True
        slot.done = True
        self._wake_dispatch()
        if slot.owner_id == VOICE_OWNER:
            slot.event.set()
        logger.info("worker for %s stopped at a cancellation boundary", slot.task_id)

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

        # Stage 3: voice is Tier 1 only *unless* the owner enabled the Tier 2
        # relaxation.  Anything that claims a voice origin for a Tier 2+ action
        # is still refused here unless tier2_voice re-verifies every condition
        # (flag on, Tier exactly 2, a named tool, not blocked, no typed folder
        # name, exact readback).  Default flag off => identical refusal to before.
        if msg.source == "voice" and slot.confirm_tier >= 2:
            ok, reason = self._tier2_voice_ok(slot)
            if not ok:
                await conn.send(
                    ErrorMessage(
                        code="tier_requires_terminal",
                        message="this action requires terminal confirmation — it cannot be approved by voice",
                        task_id=msg.task_id,
                    )
                )
                logger.info("voice tier 2 confirm refused: %s", reason)
                return

        # I3: the answer must correspond to this task's exact pending
        # confirmation.  Fail closed *before* the password check or any resume:
        # a missing/unknown/expired/superseded/used/mismatched id approves nothing.
        check = self._confirmations.check(msg.confirmation_id, msg.plan_hash, msg.action_hash)
        if not check.ok:
            await conn.send(
                ErrorMessage(
                    code="confirmation_stale",
                    message=f"this confirmation can no longer be used ({check.reason})",
                    task_id=msg.task_id,
                )
            )
            return
        self._confirmations.resolve(msg.confirmation_id)

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
        if msg.source == "voice" and slot.confirm_tier >= 2:
            # Advisory marker for the graph: the gate still re-verifies it from
            # its own decision.  Set only after the registry check above passed.
            ok, _ = self._tier2_voice_ok(slot)
            if ok:
                answer[KEY_VOICE_TIER2] = True

        slot.resume_answer = answer
        # Ack the response **before** waking the worker, not after: the worker
        # runs the graph on another thread and can dispatch the next
        # ConfirmRequest (or the FinalMessage) on this same connection the
        # instant the event is set.  The client is blocked reading one message
        # for the ack, so a request arriving first would be consumed as if it
        # were the ack - the approval prompt would vanish and the task would
        # then time out.  (Same failure mode as 663d559, one layer down.)
        await conn.send(
            EventMessage(
                task_id=slot.task_id, kind="ack", data={"message": "confirmation received"}
            )
        )
        slot.event.set()  # wake the worker thread

    async def _handle_clarification(
        self, msg: ClarificationResponse, conn: ClientConnection
    ) -> None:
        """Deliver the user's clarification answer to the waiting worker."""
        slot = self._active
        if slot is None or slot.task_id != msg.task_id or slot.done:
            await conn.send(
                ErrorMessage(
                    code="no_such_task",
                    message=f"no active task awaiting clarification: {msg.task_id!r}",
                    task_id=msg.task_id,
                )
            )
            return

        # Empty answers are legal on the wire; the clarify node turns a
        # non-string / empty resume into "No clarification received." (fail
        # closed).  Never the password, never a confirmation flag.
        slot.resume_answer = msg.answer if isinstance(msg.answer, str) else ""
        # Ack before waking the worker - see the note in _handle_confirm.
        await conn.send(
            EventMessage(
                task_id=slot.task_id, kind="ack", data={"message": "clarification received"}
            )
        )
        slot.event.set()  # wake the worker thread

    # ── cancel ─────────────────────────────────────────────────────────

    async def _handle_cancel(self, msg: CancelMessage, conn: ClientConnection) -> None:
        """Cancel a running or queued task."""
        with self._lock:
            if self._active and self._active.task_id == msg.task_id:
                # Cancel this task's own token: the graph stops at its next node
                # boundary, so the remaining steps of the plan never execute.
                self._active.cancel.cancel()
                self._active.cancelled = True
                self._active.error = "cancelled by user"
                self._active.done = True
                # Ack before waking the worker - see the note in _handle_confirm.
                await conn.send(
                    EventMessage(task_id=msg.task_id, kind="ack", data={"message": "cancelled"})
                )
                self._active.event.set()  # wake worker if waiting
                self._task_event.set()
                return
            for i, slot in enumerate(self._queue):
                if slot.task_id == msg.task_id:
                    self._queue.pop(i)
                    await conn.send(
                        EventMessage(
                            task_id=msg.task_id, kind="ack", data={"message": "removed from queue"}
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
        voice_mode = "NORMAL"
        if self._voice_service is not None:
            state = self._service_state(self._voice_service)
            if state == "error":
                voice_status = "error"
                voice_reason = self._service_error_code(self._voice_service)
            elif state == "starting":
                voice_status = "starting"
            elif state == "on":
                voice_status = "on"
            try:
                loop = getattr(self._voice_service, "_loop", None)
                if loop is not None:
                    mode = getattr(loop, "mode", None)
                    if mode is not None:
                        try:
                            voice_mode = mode.value
                        except Exception:  # noqa: BLE001 - best effort
                            voice_mode = str(mode)
            except Exception:  # noqa: BLE001, S110 - best effort
                pass
        await conn.send(
            StatusResponse(
                daemon="running",
                voice=voice_status,
                voice_reason=voice_reason,
                mode=voice_mode,
                unlocked=self._unlock.is_unlocked(),
                queue=queue_len,
                active_task=active_id,
            )
        )

    # ── stale confirmation cleanup ─────────────────────────────────────

    async def _stale_confirmation_loop(self) -> None:
        """Periodically expire confirmations waiting too long."""
        while True:
            try:
                await asyncio.sleep(STALE_CHECK_INTERVAL_S)
                now = time.time()
                slot = self._active
                if (
                    slot is not None
                    and slot.confirm_payload is not None
                    and not slot.done
                    and slot.confirm_sent_at > 0
                    and (now - slot.confirm_sent_at) > self._task_runtime.confirm_timeout_s
                ):
                    logger.warning("confirmation for %s expired", slot.task_id)
                    # I3: a timed-out confirmation's id must not stay usable.
                    self._confirmations.supersede(slot.task_id)
                    # Never abandon the task: resume the graph with a timed-out
                    # refusal so the checkpoint finishes cleanly (D1).
                    slot.resume_answer = timeout_answer(slot.confirm_payload)
                    slot.event.set()
                    self._task_event.set()
            except asyncio.CancelledError:
                raise
            except Exception:  # this housekeeping loop must never die
                logger.exception("stale-confirmation loop iteration failed")
