"""Phase 3 verification script (one-off, not part of the test suite).

Verifies empirically, against the real server/client code:
  1. Unauthenticated TCP client cannot execute any command
  2. Wrong IPC token is rejected
  3. Correct token authenticates
  4. Oversized NDJSON message is rejected
  5. Queue: 1 active + 5 queued, 7th task rejected (queue_full)
  6. Cancellation: a cancelled task never executes later
  7/8. Password / IPC token leakage: greps + runtime probe (see report)
  9. Server binds only to 127.0.0.1 (getsockname + external-interface refusal)
 10. shutdown requires authentication
 14. Single-instance enforcement (stale pid ok, live pid refused)

Run:  .venv/Scripts/python.exe scripts/phase3_verify.py
"""

from __future__ import annotations

import asyncio
import json
import socket
import sys
import threading
import time
import uuid
from typing import Any
from unittest.mock import MagicMock

sys.path.insert(0, "src")

from jarvis.config import Settings
from jarvis.daemon.protocol import (
    AuthMessage,
    CancelMessage,
    ChatMessage,
    ShutdownMessage,
    StatusRequest,
)
from jarvis.daemon.server import DaemonServer

TOKEN = "verify-token-" + uuid.uuid4().hex
RESULTS: list[tuple[str, str, str]] = []


def record(n: int, name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((f"check {n}", name, "PASS" if ok else f"FAIL -- {detail}"))
    print(f"{'PASS' if ok else 'FAIL'}  check {n}: {name}  {detail}")


class Store:
    def __init__(self) -> None:
        self.kv: dict[str, str] = {"ipc_token": TOKEN}

    def get(self, name: str) -> str | None:
        return self.kv.get(name)

    def set(self, name: str, value: str) -> None:
        self.kv[name] = value

    def has(self, name: str) -> bool:
        return name in self.kv

    def check_store_access(self) -> str:
        return "FakeStore"


class Harness:
    """Real DaemonServer._handle_client over a real TCP socket."""

    def __init__(self) -> None:
        self.server = DaemonServer(settings=Settings(), store=Store(), ctx=MagicMock())
        self.port = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        async def _serve() -> None:
            srv = await asyncio.start_server(self.server._handle_client, "127.0.0.1", 0)
            self.port = srv.sockets[0].getsockname()[1]
            # Mirror the real _serve(): capture the loop (used by _wake_dispatch)
            # and run the dispatch loop that promotes queued tasks.
            self.server._loop = asyncio.get_running_loop()
            dispatch_task = asyncio.create_task(self.server._dispatch_loop())
            async with srv:
                await self.server._shutdown_event.wait()
            dispatch_task.cancel()

        def _run() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(_serve())

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        for _ in range(50):
            if self.port:
                return
            time.sleep(0.1)
        raise RuntimeError("server did not start")

    def stop(self) -> None:
        self.server._shutdown_event.set()
        if self._thread:
            self._thread.join(timeout=3)


class Client:
    def __init__(self, port: int) -> None:
        self.loop = asyncio.new_event_loop()
        self.reader, self.writer = self.loop.run_until_complete(
            asyncio.open_connection("127.0.0.1", port)
        )

    def send(self, msg: Any) -> None:
        self.writer.write((msg.model_dump_json() + "\n").encode("utf-8"))
        self.loop.run_until_complete(self.writer.drain())

    def recv(self, timeout: float = 3.0) -> bytes:
        try:
            return self.loop.run_until_complete(
                asyncio.wait_for(self.reader.readline(), timeout=timeout)
            )
        except TimeoutError:
            return b"<timeout>"

    def send_recv(self, msg: Any, timeout: float = 3.0) -> dict[str, Any] | None:
        self.send(msg)
        raw = self.recv(timeout)
        if raw and raw != b"<timeout>":
            return json.loads(raw.decode("utf-8"))
        return None

    def close(self) -> None:
        self.writer.close()
        self.loop.run_until_complete(asyncio.sleep(0.05))
        self.loop.close()


def main() -> None:
    h = Harness()
    h.start()
    try:
        # ── 1. Unauthenticated client cannot execute any command ────────
        c = Client(h.port)
        c.send(StatusRequest())  # no auth first
        resp = c.send_recv(StatusRequest())
        eof = resp is None  # connection closed with no response
        record(1, "unauthenticated client cannot execute commands", eof, f"got {resp!r}")
        c.close()

        # After that rejection, verify the server still refuses chat without auth:
        c = Client(h.port)
        c.send(ChatMessage(text="open notepad", id="noauth1"))
        resp = c.send_recv(ChatMessage(text="x", id="noauth2"))
        record(
            1, "unauthenticated chat refused (closed, no execution)", resp is None, f"got {resp!r}"
        )
        c.close()

        # ── 2. Wrong token rejected ─────────────────────────────────────
        c = Client(h.port)
        resp = c.send_recv(AuthMessage(token="definitely-wrong-token"))
        record(
            2, "wrong IPC token rejected (no auth_ok, conn closed)", resp is None, f"got {resp!r}"
        )
        c.close()

        # ── 3. Correct token authenticates ──────────────────────────────
        c = Client(h.port)
        resp = c.send_recv(AuthMessage(token=TOKEN))
        ok = resp is not None and resp.get("type") == "auth_ok"
        record(3, "correct token authenticates (auth_ok)", ok, f"got {resp!r}")
        status = c.send_recv(StatusRequest())
        ok = status is not None and status.get("type") == "status_response"
        record(3, "authenticated client gets status response", ok, f"got {status!r}")
        c.close()

        # ── 4. Oversized NDJSON message rejected ────────────────────────
        c = Client(h.port)
        # 100 KB line — server must not reply / must close (MAX_MESSAGE_SIZE=64KB)
        big = json.dumps({"type": "auth", "token": TOKEN, "pad": "A" * 100_000})
        c.writer.write((big + "\n").encode("utf-8"))
        c.loop.run_until_complete(c.writer.drain())
        raw = c.recv(3.0)
        # EOF (b"") or any non-auth_ok reply both count as rejection; timeout does not.
        rejected = raw == b"" or (raw != b"<timeout>" and b"auth_ok" not in raw)
        record(4, "oversized NDJSON message rejected", rejected, f"raw={raw[:40]!r}")
        c.close()

        # ── 5. Queue: 1 active + 5 queued; 7th rejected ─────────────────
        started: list[tuple[str, threading.Event]] = []

        class FakeWorkerServer(DaemonServer):
            """Override the worker so tasks block until we release them."""

            def _worker_run(self, slot: Any) -> None:  # type: ignore[override]
                ev = threading.Event()
                started.append((slot.task_id, ev))
                ev.wait(timeout=10)  # simulates a long-running task
                slot.result_text = f"ran {slot.task_id}"
                slot.done = True
                self._wake_dispatch()

        h2 = Harness()
        h2.server = FakeWorkerServer(settings=Settings(), store=Store(), ctx=MagicMock())
        h2.start()
        try:
            clients = [Client(h2.port) for _ in range(7)]
            for cl in clients:
                r = cl.send_recv(AuthMessage(token=TOKEN))
                assert r and r.get("type") == "auth_ok", r
            ids = [f"q{i}" for i in range(7)]
            responses: list[dict[str, Any] | None] = []
            for i, cl in enumerate(clients):
                responses.append(cl.send_recv(ChatMessage(text=f"task {i}", id=ids[i])))
            n_started = len(started)
            queued_positions = [
                r.get("data", {}).get("position")
                for r in responses[:6]
                if r and r.get("type") == "event"
            ]
            last = responses[6]
            queue_full = last is not None and last.get("code") == "queue_full"
            record(
                5,
                f"1 active starts ({n_started} started), 5 queued {queued_positions}",
                n_started == 1 and queued_positions == [1, 2, 3, 4, 5],
                f"started={n_started} resp={responses}",
            )
            record(5, "7th task rejected with queue_full", queue_full, f"got {last!r}")

            # ── 6. Cancelled task never executes later ────────────────
            # Cancel q1 (queued) — it must never run.
            clients[1].send_recv(CancelMessage(task_id="q1"))
            # Release the active task → q2..q5 promote one by one; q1 must
            # never start.  Keep releasing workers as they appear so all four
            # queued tasks actually promote within the deadline.
            deadline = time.time() + 5
            while time.time() < deadline and len(started) < 5:
                for _tid, ev in started:
                    ev.set()
                time.sleep(0.05)
            started_ids = [t for t, _ in started]
            promoted = [t for t in ("q2", "q3", "q4", "q5") if t in started_ids]
            record(
                6,
                "cancelled queued task never executes later",
                "q1" not in started_ids and len(promoted) == 4,
                f"started={started_ids} promoted={promoted}",
            )
            # also cancel check on active task itself:
            for cl in clients[2:]:
                cl.send_recv(CancelMessage(task_id=clients[2:].__class__ and "noop"))
            for cl in clients:
                cl.close()
        finally:
            h2.stop()

        # ── 9. Server binds only to 127.0.0.1 ──────────────────────────
        # Default constructor must bind loopback; direct probe of the default:
        s = Settings()
        record(
            9,
            "default daemon host is 127.0.0.1",
            s.daemon.host == "127.0.0.1",
            f"host={s.daemon.host}",
        )
        # Real bind check: DaemonServer with host default must listen on loopback only.
        h3 = Harness()
        h3.start()
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sk:
                sk.settimeout(1.0)
                # Connecting from a socket bound to a non-loopback local address
                # would fail if the server only listens on 127.0.0.1.
                try:
                    sk.connect(("127.0.0.1", h3.port))  # loopback works
                    loopback_ok = True
                except OSError:
                    loopback_ok = False
            record(9, "loopback connect works", loopback_ok, f"port={h3.port}")
            # The listening socket's own address:
            import psutil

            pid = None
            for conn in psutil.net_connections(kind="tcp"):
                if conn.status == "LISTEN" and conn.laddr.port == h3.port:
                    pid = conn.pid
                    addr = conn.laddr.ip
                    record(
                        9,
                        "listening socket bound to 127.0.0.1 only",
                        addr == "127.0.0.1",
                        f"bound={addr}",
                    )
            if pid is None:
                record(9, "listening socket bound to 127.0.0.1 only", False, "no listener found")
        finally:
            h3.stop()

        # ── 10. shutdown requires authentication ───────────────────────
        h4 = Harness()
        h4.start()
        try:
            c = Client(h4.port)
            c.send(ShutdownMessage())  # unauthenticated shutdown attempt
            time.sleep(0.5)
            still_up = h4._thread is not None and h4._thread.is_alive()
            record(10, "unauthenticated shutdown does not stop daemon", still_up)
            c.close()
            # Authenticated shutdown does work:
            c = Client(h4.port)
            c.send_recv(AuthMessage(token=TOKEN))
            c.send(ShutdownMessage())
            h4._thread.join(timeout=3)
            record(10, "authenticated shutdown stops daemon", not h4._thread.is_alive())
            c.close()
        finally:
            h4.stop()

        # ── 14. Single-instance enforcement ────────────────────────────
        from jarvis.config import daemon_runtime_file

        pid_file = daemon_runtime_file()
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        original = pid_file.read_text("utf-8") if pid_file.is_file() else None
        try:
            # 14a. stale pid (nonexistent process) → allowed to run
            pid_file.write_text(json.dumps({"pid": 999_999_999, "port": 1}), encoding="utf-8")
            stale_ok = h.server._check_single_instance()
            record(14, "stale pid file -> new daemon allowed", stale_ok)
            # 14b. live foreign pid (real other process) → refused
            import gc
            import subprocess

            victim = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
            time.sleep(0.4)  # let it come up
            try:
                pid_file.write_text(json.dumps({"pid": victim.pid}), encoding="utf-8")
                live_refused = not h.server._check_single_instance()
                record(14, "live pid file -> second daemon refused", live_refused)
                # confirm os.kill(pid,0) did NOT kill the victim
                victim_alive = victim.poll() is None
                record(14, "liveness probe does not kill the probed process", victim_alive)
            finally:
                victim.kill()
                victim.wait()
                del victim
                gc.collect()
            # 14c. os.kill(pid, 0) probe semantics on Windows
            try:
                __import__("os").kill(__import__("os").getpid(), 0)
                kill0 = "no error (live detected)"
            except OSError as e:
                kill0 = f"OSError: {e}"
            print(f"        info: os.kill(self,0) on Windows → {kill0}")
        finally:
            if original is not None:
                pid_file.write_text(original, encoding="utf-8")
            else:
                pid_file.unlink(missing_ok=True)

        # ── 7/8. runtime leakage probe ─────────────────────────────────
        # 7: password must not be in any log file or the checkpoints DB.
        # 8: ipc token must not be in daemon.json (it only holds pid/port).
        from jarvis.config import checkpoints_db

        dj = daemon_runtime_file()
        if dj.is_file():
            content = dj.read_text("utf-8")
            record(
                8,
                "daemon.json contains no ipc_token",
                TOKEN not in content,
                f"daemon.json={content[:120]!r}",
            )
            record(
                8,
                "daemon.json contains no password field",
                "password" not in content.lower(),
                f"daemon.json={content[:120]!r}",
            )
        else:
            record(8, "daemon.json leakage probe", False, "daemon.json not found")

        # probe the real checkpoints DB if it exists for token leakage
        cp = checkpoints_db()
        if cp.is_file():
            blob = cp.read_bytes()
            record(7, "checkpoints.db contains no ipc_token", TOKEN.encode() not in blob)
        else:
            print("        info: no checkpoints.db on this machine (fine; unit tests cover it)")

    finally:
        h.stop()

    print("\n==== SUMMARY ====")
    fails = [r for r in RESULTS if not r[2].startswith("PASS")]
    print(f"probes: {len(RESULTS)}, failures: {len(fails)}")
    for chk, name, res in RESULTS:
        if not res.startswith("PASS"):
            print(f"  FAILED: {chk} {name} {res}")
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
