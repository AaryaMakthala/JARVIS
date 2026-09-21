"""Phase 3 check 11: interrupt -> confirmation -> resume THROUGH the daemon.

Uses the real DaemonServer dispatch loop, real TCP, real _handle_chat /
_handle_confirm / _dispatch_loop wiring.  Only the LangGraph worker itself is
stubbed (the graph-level interrupt/resume is proven separately by
tests/integration/test_confirmation_resume.py against the real compiled graph).

Run:  .venv/Scripts/python.exe scripts/phase3_verify_confirm.py
"""

from __future__ import annotations

import asyncio
import json
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
    ConfirmResponse,
)
from jarvis.daemon.server import DaemonServer

TOKEN = "confirm-token-" + uuid.uuid4().hex
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        FAILURES.append(name)


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


class ConfirmServer(DaemonServer):
    """Worker stub that pauses at a 'confirmation' like the graph interrupt does."""

    def _worker_run(self, slot: Any) -> None:  # type: ignore[override]
        # Phase 1: run → interrupt.  Capture the expected hash BEFORE waiting:
        # the dispatch loop consumes confirm_payload while we sleep.
        expected_hash = "a" * 64
        slot.confirm_payload = {
            "tier": 1,
            "summary": "fake_echo text='hi'",
            "needs_unlock": False,
            "typed_confirmation": None,
            "action_hash": expected_hash,
            "untrusted": False,
        }
        slot.event.clear()
        self._wake_dispatch()  # thread-safe wake (same mechanism as the real worker)

        # Wait for the event loop to deliver the answer (same as real worker)
        slot.event.wait(timeout=10)

        if slot.event.is_set() and slot.resume_answer is not None:
            answer = slot.resume_answer
            # Mirror the real act-node contract: a mismatched action_hash is
            # refused exactly like tests/integration/
            # test_confirmation_resume.py::test_tampered_hash_never_executes
            # proves for the real graph.
            if answer.get("action_hash") == expected_hash:
                slot.result_text = (
                    f"executed after approval (hash={answer['action_hash'][:8]})"
                    if answer.get("approved")
                    else "Refused by you (confirmation answered with 'no')"
                )
            else:
                slot.result_text = (
                    "Refused by you (confirmation answered with 'no' or a mismatched action)."
                )
            slot.resume_answer = None
            slot.confirm_payload = None
            slot.done = True
            self._wake_dispatch()
        else:
            slot.error = "confirmation timed out"
            slot.done = True
            self._wake_dispatch()


class Client:
    def __init__(self, port: int) -> None:
        self.loop = asyncio.new_event_loop()
        self.reader, self.writer = self.loop.run_until_complete(
            asyncio.open_connection("127.0.0.1", port)
        )

    def send(self, msg: Any) -> None:
        self.writer.write((msg.model_dump_json() + "\n").encode("utf-8"))
        self.loop.run_until_complete(self.writer.drain())

    def recv(self, timeout: float = 5.0) -> bytes:
        try:
            return self.loop.run_until_complete(
                asyncio.wait_for(self.reader.readline(), timeout=timeout)
            )
        except TimeoutError:
            return b"<timeout>"

    def recv_json(self, timeout: float = 5.0) -> dict[str, Any] | None:
        raw = self.recv(timeout)
        if raw and raw != b"<timeout>":
            return json.loads(raw.decode("utf-8"))
        return None

    def close(self) -> None:
        self.writer.close()
        self.loop.run_until_complete(asyncio.sleep(0.05))
        self.loop.close()


class Harness:
    def __init__(self) -> None:
        self.server = ConfirmServer(settings=Settings(), store=Store(), ctx=MagicMock())
        self.port = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        async def _serve() -> None:
            srv = await asyncio.start_server(self.server._handle_client, "127.0.0.1", 0)
            self.port = srv.sockets[0].getsockname()[1]
            # Mirror the real _serve(): capture the loop (used by _wake_dispatch)
            # and run the dispatch loop that forwards interrupt payloads from the
            # worker thread to the owning client.
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

    def stop(self) -> None:
        self.server._shutdown_event.set()
        if self._thread:
            self._thread.join(timeout=3)


def main() -> None:
    # ── Scenario A: approve → task executes ────────────────────────────
    h = Harness()
    h.start()
    try:
        c = Client(h.port)
        c.send(AuthMessage(token=TOKEN))
        hello = c.recv_json()
        assert hello and hello.get("type") == "auth_ok", hello
        c.send(ChatMessage(text="echo hi", id="tA"))
        req = c.recv_json()  # dispatch loop → ConfirmRequest
        check(
            "A: daemon forwards confirm_request from interrupt",
            req is not None and req["type"] == "confirm_request",
            f"got {req!r}",
        )
        check(
            "A: confirm_request carries tier/summary/hash",
            req is not None and req["tier"] == 1 and req["action_hash"] == "a" * 64,
            f"{req!r}",
        )
        c.send(
            ConfirmResponse(task_id=req["task_id"], approved=True, action_hash=req["action_hash"])
        )
        final = c.recv_json()
        check(
            "A: approved task resumes and finishes with final message",
            final is not None
            and final["type"] == "final"
            and "executed after approval" in final.get("text", ""),
            f"got {final!r}",
        )
        c.close()
    finally:
        h.stop()

    # ── Scenario B: deny → task never executes ─────────────────────────
    h = Harness()
    h.start()
    try:
        c = Client(h.port)
        c.send(AuthMessage(token=TOKEN))
        c.recv_json()
        c.send(ChatMessage(text="echo hi", id="tB"))
        req = c.recv_json()
        c.send(
            ConfirmResponse(task_id=req["task_id"], approved=False, action_hash=req["action_hash"])
        )
        final = c.recv_json()
        check(
            "B: denied task resumes but reports refusal, never executes",
            final is not None and final["type"] == "final" and "Refused" in final.get("text", ""),
            f"got {final!r}",
        )
        c.close()
    finally:
        h.stop()

    # ── Scenario C: tampered hash → refused ────────────────────────────
    h = Harness()
    h.start()
    try:
        c = Client(h.port)
        c.send(AuthMessage(token=TOKEN))
        c.recv_json()
        c.send(ChatMessage(text="echo hi", id="tC"))
        req = c.recv_json()
        c.send(ConfirmResponse(task_id=req["task_id"], approved=True, action_hash="0" * 64))
        final = c.recv_json()
        check(
            "C: tampered action_hash is refused by the graph gate (mirrored), never executed",
            final is not None and final["type"] == "final" and "Refused" in final.get("text", ""),
            f"got {final!r}",
        )
        c.close()
    finally:
        h.stop()

    # ── Scenario D: wrong task_id confirm → error, no execution ────────
    h = Harness()
    h.start()
    try:
        c = Client(h.port)
        c.send(AuthMessage(token=TOKEN))
        c.recv_json()
        c.send(ChatMessage(text="echo hi", id="tD"))
        c.recv_json()  # confirm_request
        c.send(ConfirmResponse(task_id="no-such-task", approved=True, action_hash="a" * 64))
        err = c.recv_json()
        check(
            "D: confirm_response for unknown task_id is an error, no side effect",
            err is not None and err["type"] == "error" and err["code"] == "no_such_task",
            f"got {err!r}",
        )
        c.close()
    finally:
        h.stop()

    # ── Scenario E: cancel while awaiting confirmation → task never executes ──
    h = Harness()
    h.start()
    try:
        c = Client(h.port)
        c.send(AuthMessage(token=TOKEN))
        c.recv_json()
        c.send(ChatMessage(text="echo hi", id="tE"))
        req = c.recv_json()
        c.send(CancelMessage(task_id=req["task_id"]))
        ev = c.recv_json()
        check(
            "E1: cancel acknowledged while awaiting confirmation",
            ev is not None and ev.get("data", {}).get("message") == "cancelled",
            f"got {ev!r}",
        )
        final = c.recv_json(timeout=5)
        check(
            "E2: cancelled task never executes (error/final, no execution)",
            final is not None and final["type"] == "final" and "Error" in final.get("text", ""),
            f"got {final!r}",
        )
        c.close()
    finally:
        h.stop()

    # ── Scenario F: password is never part of the resume answer ────────
    import inspect

    from jarvis.daemon.server import DaemonServer as DS

    src = inspect.getsource(DS._handle_confirm)
    check(
        "F: _handle_confirm builds the resume answer without the password field",
        "msg.password" in src and '"password"' not in src.split("answer:")[1].split("}")[0],
        "(code inspection)",
    )

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} -> {FAILURES}")
        sys.exit(1)
    print("ALL CHECK-11 SCENARIOS PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
