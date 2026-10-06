"""Daemon-level confirmation correlation tests (I3, step 3b-2)."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from jarvis.config import Settings, VoiceSettings
from jarvis.daemon.confirmations import ConfirmationRegistry, plan_hash
from jarvis.daemon.protocol import ConfirmResponse, ErrorMessage
from jarvis.daemon.server import DaemonServer, TaskSlot
from jarvis.daemon.task_runtime import CONFIRMATION_TIMEOUT_S

_HASH_A = "a" * 64
_HASH_B = "b" * 64


class _FakeStore:
    def __init__(self) -> None:
        self._kv: dict[str, str] = {"ipc_token": "tok-0123456789abcdef"}

    def get(self, name: str) -> str | None:
        return self._kv.get(name)

    def set(self, name: str, value: str) -> None:
        self._kv[name] = value

    def has(self, name: str) -> bool:
        return name in self._kv


class _FakeConn:
    def __init__(self) -> None:
        self.sent: list[Any] = []

    async def send(self, msg: Any) -> None:
        self.sent.append(msg)


def _server() -> DaemonServer:
    return DaemonServer(settings=Settings(voice=VoiceSettings(enabled=False)), store=_FakeStore())


def _slot() -> TaskSlot:
    return TaskSlot(task_id="t1", text="x", source="chat", owner_id="c1", confirm_tier=1)


def _answer(pending, action_hash: str = _HASH_A, approved: bool = True) -> ConfirmResponse:
    return ConfirmResponse(
        task_id="t1",
        approved=approved,
        action_hash=action_hash,
        confirmation_id=pending.confirmation_id,
        plan_hash=pending.plan_hash,
    )


def test_confirm_response_without_id_refused() -> None:
    server = _server()
    slot = _slot()
    server._active = slot
    server._confirmations.issue("t1", "s1", 1, _HASH_A, plan_hash([_HASH_A]))
    conn = _FakeConn()

    asyncio.run(
        server._handle_confirm(
            ConfirmResponse(task_id="t1", approved=True, action_hash=_HASH_A), conn
        )
    )

    assert isinstance(conn.sent[-1], ErrorMessage)
    assert conn.sent[-1].code == "confirmation_stale"
    assert slot.resume_answer is None
    assert slot.event.is_set() is False


def test_confirm_response_with_wrong_id_refused() -> None:
    server = _server()
    slot = _slot()
    server._active = slot
    pending = server._confirmations.issue("t1", "s1", 1, _HASH_A, plan_hash([_HASH_A]))
    conn = _FakeConn()

    asyncio.run(
        server._handle_confirm(
            ConfirmResponse(
                task_id="t1",
                approved=True,
                action_hash=_HASH_A,
                confirmation_id="0" * 32,
                plan_hash=pending.plan_hash,
            ),
            conn,
        )
    )

    assert conn.sent[-1].code == "confirmation_stale"
    assert slot.resume_answer is None


def test_late_proceed_for_a_leaves_b_unapproved() -> None:
    server = _server()
    slot = _slot()
    server._active = slot
    a = server._confirmations.issue("t1", "s1", 1, _HASH_A, plan_hash([_HASH_A]))
    b = server._confirmations.issue("t1", "s2", 1, _HASH_B, plan_hash([_HASH_B]))  # supersedes a
    conn = _FakeConn()

    asyncio.run(server._handle_confirm(_answer(a, _HASH_A), conn))

    assert conn.sent[-1].code == "confirmation_stale"
    assert slot.resume_answer is None  # the late A proceed approved nothing
    # B is still pending, not auto-approved.
    assert server._confirmations.check(b.confirmation_id, b.plan_hash, b.action_hash).ok is True


def test_expired_confirmation_refused() -> None:
    now = [1000.0]
    server = _server()
    server._confirmations = ConfirmationRegistry(clock=lambda: now[0])
    slot = _slot()
    server._active = slot
    pending = server._confirmations.issue("t1", "s1", 1, _HASH_A, plan_hash([_HASH_A]))
    now[0] += CONFIRMATION_TIMEOUT_S + 1.0
    conn = _FakeConn()

    asyncio.run(server._handle_confirm(_answer(pending), conn))

    assert conn.sent[-1].code == "confirmation_stale"
    assert "expired" in conn.sent[-1].message
    assert slot.resume_answer is None


def test_valid_confirmation_still_resumes() -> None:
    server = _server()
    slot = _slot()
    server._active = slot
    pending = server._confirmations.issue("t1", "s1", 1, _HASH_A, plan_hash([_HASH_A]))
    conn = _FakeConn()

    asyncio.run(server._handle_confirm(_answer(pending), conn))

    assert not any(isinstance(m, ErrorMessage) for m in conn.sent)
    assert slot.resume_answer == {"approved": True, "action_hash": _HASH_A}
    assert slot.event.is_set() is True


def test_second_response_with_same_id_refused() -> None:
    server = _server()
    slot = _slot()
    server._active = slot
    pending = server._confirmations.issue("t1", "s1", 1, _HASH_A, plan_hash([_HASH_A]))
    conn = _FakeConn()

    asyncio.run(server._handle_confirm(_answer(pending), conn))
    assert slot.resume_answer == {"approved": True, "action_hash": _HASH_A}

    slot.resume_answer = None
    slot.event.clear()
    asyncio.run(server._handle_confirm(_answer(pending), conn))

    assert conn.sent[-1].code == "confirmation_stale"
    assert "already_used" in conn.sent[-1].message
    assert slot.resume_answer is None


def test_timeout_sweep_clears_pending_id(monkeypatch) -> None:
    import jarvis.daemon.server as server_mod

    server = _server()
    slot = _slot()
    server._active = slot
    pending = server._confirmations.issue("t1", "s1", 1, _HASH_A, plan_hash([_HASH_A]))
    slot.confirm_payload = {"action_hash": _HASH_A}
    slot.confirm_sent_at = time.time() - 100.0
    server._task_runtime.confirm_timeout_s = 1.0
    monkeypatch.setattr(server_mod, "STALE_CHECK_INTERVAL_S", 0.01)

    async def run_one_sweep() -> None:
        task = asyncio.ensure_future(server._stale_confirmation_loop())
        await asyncio.sleep(0.06)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(run_one_sweep())

    assert slot.resume_answer is not None and slot.resume_answer.get("timed_out") is True
    # The timed-out id must no longer be usable.
    assert (
        server._confirmations.check(pending.confirmation_id, pending.plan_hash, _HASH_A).reason
        == "superseded"
    )
