"""Regression tests for cross-thread wake-ups in the daemon server.

Bug (fixed): ``DaemonServer`` ran the LangGraph agent in a worker thread and
signalled the asyncio dispatch loop with ``asyncio.Event.set()`` called
directly from that thread.  A foreign-thread ``Event.set()`` only records the
flag — it never breaks the loop's ``select()`` sleep — so task results were
delivered only when an unrelated event-loop wakeup happened to fire (in
practice: the 10 s stale-confirmation timer), or never.

Fix: ``_wake_dispatch()`` uses ``loop.call_soon_threadsafe(...)`` from worker
threads, and falls back to a direct set only when already on the loop thread.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from jarvis.config import Settings
from jarvis.daemon.protocol import AuthMessage, ChatMessage, ConfirmResponse, StatusRequest
from jarvis.daemon.server import DaemonServer

TEST_TOKEN = "wake-regression-token-0123456789"


class _FakeStore:
    def __init__(self) -> None:
        self._kv: dict[str, str] = {"ipc_token": TEST_TOKEN}

    def get(self, name: str) -> str | None:
        return self._kv.get(name)

    def set(self, name: str, value: str) -> None:
        self._kv[name] = value

    def has(self, name: str) -> bool:
        return name in self._kv

    def check_store_access(self) -> str:
        return "FakeStore"


class _BlockingWorkerServer(DaemonServer):
    """Worker stub that pauses at a confirmation, like a real interrupt.

    Mirrors the real worker contract: it publishes ``confirm_payload``, waits
    for the loop to deliver the answer, then completes and signals the
    dispatch loop — from this *worker* thread, which is exactly the path the
    bug broke.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.worker_started = threading.Event()

    def _worker_run(self, slot: Any) -> None:  # type: ignore[override]
        self.worker_started.set()
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
        self._wake_dispatch()

        got_answer = slot.event.wait(timeout=5)
        if got_answer and slot.resume_answer is not None:
            answer = slot.resume_answer
            slot.resume_answer = None
            slot.confirm_payload = None
            # Same gate as the real act node: a mismatched hash never executes.
            if answer.get("action_hash") == expected_hash and answer.get("approved"):
                slot.result_text = "fake_echo ran hi"
            else:
                slot.result_text = (
                    "Refused by you (confirmation answered with 'no' or a mismatched action)."
                )
            slot.done = True
            self._wake_dispatch()  # ← the call that used to be a dead signal
        else:
            slot.error = "confirmation timed out"
            slot.done = True
            self._wake_dispatch()


class _ServerHarness:
    """Runs a real DaemonServer._handle_client + _dispatch_loop on a TCP socket."""

    def __init__(self, server: DaemonServer) -> None:
        self.server = server
        self.port = 0
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        async def _serve() -> None:
            srv = await asyncio.start_server(self.server._handle_client, "127.0.0.1", 0)
            self.port = srv.sockets[0].getsockname()[1]
            self.server._loop = asyncio.get_running_loop()
            dispatch_task = asyncio.create_task(self.server._dispatch_loop())
            async with srv:
                await self.server._shutdown_event.wait()
            dispatch_task.cancel()

        def _run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_serve())

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        for _ in range(50):
            if self.port:
                return
            time.sleep(0.1)
        raise RuntimeError("server did not start")

    def stop(self) -> None:
        loop = self.server._loop
        if loop is not None:
            loop.call_soon_threadsafe(self.server._shutdown_event.set)
        if self._thread:
            self._thread.join(timeout=3)


class _Client:
    def __init__(self, port: int) -> None:
        self.loop = asyncio.new_event_loop()
        self.reader, self.writer = self.loop.run_until_complete(
            asyncio.open_connection("127.0.0.1", port)
        )

    def send(self, msg: Any) -> None:
        self.writer.write((msg.model_dump_json() + "\n").encode("utf-8"))
        self.loop.run_until_complete(self.writer.drain())

    def recv_json(self, timeout: float = 2.0) -> dict[str, Any] | None:
        try:
            raw = self.loop.run_until_complete(
                asyncio.wait_for(self.reader.readline(), timeout=timeout)
            )
        except TimeoutError:
            return None
        if not raw:
            return None
        return json.loads(raw.decode("utf-8"))

    def close(self) -> None:
        self.writer.close()
        self.loop.run_until_complete(asyncio.sleep(0.05))
        self.loop.close()


class _DirectCompleteWorkerServer(DaemonServer):
    """Worker stub that completes immediately without confirmation (Tier 0).

    Mirrors the real worker path for auto-execute (Tier 0) tools: the graph
    finishes without ``interrupt()``, the worker sets ``slot.result_text``
    and signals the dispatch loop — all from the worker thread.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.worker_started = threading.Event()

    def _worker_run(self, slot: Any) -> None:  # type: ignore[override]
        self.worker_started.set()
        # Simulate a Tier 0 tool completing without needing confirmation.
        slot.result_text = "open_app ran notepad"
        slot.done = True
        self._wake_dispatch()


def _make_server() -> _BlockingWorkerServer:
    return _BlockingWorkerServer(settings=Settings(), store=_FakeStore(), ctx=MagicMock())


def _make_direct_server() -> _DirectCompleteWorkerServer:
    return _DirectCompleteWorkerServer(settings=Settings(), store=_FakeStore(), ctx=MagicMock())


# ── 1. Unit: the wake mechanism itself ───────────────────────────────────


def test_wake_dispatch_from_worker_thread_wakes_sleeping_loop() -> None:
    """A worker-thread _wake_dispatch() must wake a loop sleeping on the event.

    This is the exact failure mode of the bug: without call_soon_threadsafe,
    the wait below times out even though the event was "set" long before.
    """
    server = _make_server()
    loop = asyncio.new_event_loop()
    try:
        woken = threading.Event()

        async def _waiter() -> None:
            await asyncio.wait_for(server._task_event.wait(), timeout=2.0)
            woken.set()

        def _run_loop() -> None:
            asyncio.set_event_loop(loop)
            server._loop = loop  # simulate _serve() having captured the loop
            loop.create_task(_waiter())
            loop.run_forever()

        t = threading.Thread(target=_run_loop, daemon=True)
        t.start()
        time.sleep(0.2)  # let the loop start and block on the event

        server._wake_dispatch()  # from this (worker) thread
        assert woken.wait(timeout=2.0), (
            "worker-thread _wake_dispatch() did not wake the loop within 2s"
        )
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=2)
        loop.close()


def test_wake_dispatch_on_loop_thread_sets_event_directly() -> None:
    """On the loop thread the direct set path is used (no scheduling needed)."""
    server = _make_server()

    async def _main() -> None:
        server._loop = asyncio.get_running_loop()
        server._wake_dispatch()  # we ARE the loop thread here
        await asyncio.wait_for(server._task_event.wait(), timeout=1.0)

    asyncio.run(_main())


def test_wake_dispatch_without_captured_loop_still_sets_event() -> None:
    """Fail-safe: without a captured loop the event is set directly.

    (The real server always captures the loop in _serve(); this only proves
    the fallback cannot crash or no-op.)
    """
    server = _make_server()
    assert server._loop is None

    async def _main() -> None:
        await asyncio.wait_for(server._task_event.wait(), timeout=1.0)

    # Called from a plain thread (no running loop): must not raise.
    threading.Thread(target=server._wake_dispatch, daemon=True).start()
    asyncio.run(_main())


# ── 2. Regression: interrupt → confirm → final arrives promptly ──────────


@pytest.fixture
def harness() -> Any:
    h = _ServerHarness(_make_server())
    h.start()
    yield h
    h.stop()


def test_confirm_flow_final_delivered_within_2s(harness: Any) -> None:
    """chat → confirm_request → ConfirmResponse → final, all within 2 s.

    Deliberately does NOT rely on the stale-confirmation timer (10 s period)
    or any other unrelated event-loop wakeup: if the worker-thread wake-up is
    broken, the final cannot arrive inside the bound.
    """
    c = _Client(harness.port)
    try:
        c.send(AuthMessage(token=TEST_TOKEN))
        hello = c.recv_json(timeout=2)
        assert hello and hello["type"] == "auth_ok"

        t0 = time.monotonic()
        c.send(ChatMessage(text="echo hi", id="t1"))
        req = c.recv_json(timeout=2)
        assert req is not None and req["type"] == "confirm_request", req
        assert req["action_hash"] == "a" * 64

        c.send(
            ConfirmResponse(task_id=req["task_id"], approved=True, action_hash=req["action_hash"])
        )
        final = c.recv_json(timeout=2.0)
        elapsed = time.monotonic() - t0
        assert final is not None and final["type"] == "final", (
            f"final never arrived (after {elapsed:.2f}s) — dispatch loop was "
            f"not woken by the worker thread"
        )
        assert final["text"] == "fake_echo ran hi"
        assert elapsed < 2.0, f"final took {elapsed:.2f}s — wake-up is not prompt"
    finally:
        c.close()


def test_denied_confirmation_never_executes_and_arrives_promptly(harness: Any) -> None:
    c = _Client(harness.port)
    try:
        c.send(AuthMessage(token=TEST_TOKEN))
        c.recv_json(timeout=2)
        c.send(ChatMessage(text="echo hi", id="t2"))
        req = c.recv_json(timeout=2)
        assert req and req["type"] == "confirm_request"
        c.send(
            ConfirmResponse(task_id=req["task_id"], approved=False, action_hash=req["action_hash"])
        )
        final = c.recv_json(timeout=2.0)
        assert final is not None and final["type"] == "final"
        assert "Refused" in final["text"]
    finally:
        c.close()


def test_tampered_hash_cannot_drive_resume_to_execution(harness: Any) -> None:
    c = _Client(harness.port)
    try:
        c.send(AuthMessage(token=TEST_TOKEN))
        c.recv_json(timeout=2)
        c.send(ChatMessage(text="echo hi", id="t3"))
        req = c.recv_json(timeout=2)
        assert req and req["type"] == "confirm_request"
        c.send(ConfirmResponse(task_id=req["task_id"], approved=True, action_hash="0" * 64))
        final = c.recv_json(timeout=2.0)
        assert final is not None and final["type"] == "final"
        assert "Refused" in final["text"]  # mismatched hash → never executed
    finally:
        c.close()


def test_queued_task_result_delivered_after_promotion(harness: Any) -> None:
    """End-to-end queue flow: task 1 confirms and completes, queued task 2
    is promoted and its final is delivered through the same dispatch loop."""
    c1 = _Client(harness.port)
    c2 = _Client(harness.port)
    try:
        for c in (c1, c2):
            c.send(AuthMessage(token=TEST_TOKEN))
            hello = c.recv_json(timeout=2)
            assert hello and hello["type"] == "auth_ok"

        # c1's task occupies the active slot and pauses at confirmation.
        c1.send(ChatMessage(text="echo hi", id="t1"))
        req1 = c1.recv_json(timeout=2)
        assert req1 and req1["type"] == "confirm_request"

        # c2's task is queued while t1 is active.
        c2.send(ChatMessage(text="echo hi", id="t2"))
        queued = c2.recv_json(timeout=2)
        assert queued and queued["type"] == "event"
        assert queued["data"]["message"] == "queued"

        # Approve t1 → it completes → t2 is promoted → t2's confirm_request
        # must reach c2 (its owner), then c2 approves and gets its final.
        c1.send(
            ConfirmResponse(task_id=req1["task_id"], approved=True, action_hash=req1["action_hash"])
        )
        final1 = c1.recv_json(timeout=2.0)
        assert final1 and final1["type"] == "final" and final1["task_id"] == "t1"

        req2 = c2.recv_json(timeout=2.0)
        assert req2 and req2["type"] == "confirm_request" and req2["task_id"] == "t2", (
            "promoted task's confirm_request was not routed to its owner"
        )
        c2.send(
            ConfirmResponse(task_id=req2["task_id"], approved=True, action_hash=req2["action_hash"])
        )
        final2 = c2.recv_json(timeout=2.0)
        assert final2 and final2["type"] == "final" and final2["task_id"] == "t2"
    finally:
        c1.close()
        c2.close()


def test_status_while_worker_running_uses_no_worker_wake(harness: Any) -> None:
    """Sanity: status responses still work while a worker is mid-task (the
    loop's own socket handling must not depend on worker-thread wake-ups)."""
    c = _Client(harness.port)
    try:
        c.send(AuthMessage(token=TEST_TOKEN))
        c.recv_json(timeout=2)
        c.send(ChatMessage(text="echo hi", id="t9"))
        req = c.recv_json(timeout=2)
        assert req and req["type"] == "confirm_request"  # worker is paused now
        c.send(StatusRequest())
        resp = c.recv_json(timeout=2)
        assert resp and resp["type"] == "status_response"
        assert resp["active_task"] == "t9"
    finally:
        c.close()


# ── 3. Direct completion (Tier 0 — no confirmation needed) ──────────────


@pytest.fixture
def direct_harness() -> Any:
    h = _ServerHarness(_make_direct_server())
    h.start()
    yield h
    h.stop()


def test_direct_completion_final_delivered_promptly(direct_harness: Any) -> None:
    """chat → Tier-0 auto-execute → final, with no confirmation in between.

    Verifies that the dispatch loop delivers a result whose worker finishes
    without ever pausing at interrupt() — the Tier 0 fast path.
    """
    c = _Client(direct_harness.port)
    try:
        c.send(AuthMessage(token=TEST_TOKEN))
        hello = c.recv_json(timeout=2)
        assert hello and hello["type"] == "auth_ok"

        t0 = time.monotonic()
        c.send(ChatMessage(text="open notepad", id="t10"))
        final = c.recv_json(timeout=2.0)
        elapsed = time.monotonic() - t0
        assert final is not None and final["type"] == "final", (
            f"final never arrived (after {elapsed:.2f}s) for a Tier-0 task"
        )
        assert final["text"] == "open_app ran notepad"
        assert final["task_id"] == "t10"
        assert elapsed < 2.0, f"direct completion took {elapsed:.2f}s"
    finally:
        c.close()
