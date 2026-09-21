"""Safety invariants that must hold for every phase (docs/03_SECURITY_AND_POLICY.md).

Phase 1 covers the machinery (docs/05_BUILD_PLAN.md Phase 1: confirmation and
resume) that the Phase 2+ tools plug into:

 * 1  unknown / rejected tool -> no execution
 * 2  no generic shell / powershell / cmd / exec / eval tool exists
 * 3  hard-blocked (Tier 3) steps are refused in code, never by prompt
 * 6  approvals bind to the exact action_hash
 * 7  secrets never enter graph state or the checkpointed bytes
 * 9  unverifiable / failed verification is reported, never claimed as success
 * 11 Tier-2 actions cannot slip past an unlocked-session gate
 * 12 the LLM cannot lower a tier (planner-only claims are ignored)
 * 13 no side effects before interrupt(); resume never re-runs the planner
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from jarvis.agent.context import make_app_context
from jarvis.agent.graph import build_graph, open_sqlite_checkpointer
from jarvis.agent.runner import resume_task, run_task
from jarvis.agent.state import Plan, Step
from jarvis.config import AgentSettings, PolicySettings, Settings
from jarvis.llm.client import FakeLLM
from jarvis.policy.unlock import UnlockManager
from jarvis.tools.base import ToolContext
from jarvis.tools.files import make_delete_path_spec
from support import (
    FakeDirTrash,
    approve,
    deny,
    echo_plan,
    make_spec,
    registry_with,
    tamper,
)

# ---------------------------------------------------------------------------
# Invariant 1: unknown tool -> rejected before any execution
# ---------------------------------------------------------------------------


def test_invariant_1_unknown_tool_is_rejected_without_execution(tmp_path: Any) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    ctx = make_app_context(
        Settings(),
        llm=FakeLLM([_plan("no_such_tool", {"x": 1}), _plan("no_such_tool", {"x": 1})]),
        registry=registry_with(make_spec("fake_echo", base_tier=1, record=record)),
    )
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        outcome = run_task(ctx, saver, "do the thing")
        assert outcome.interrupted is False
        assert record == []
        assert "not available" in (outcome.final_answer or "")
    finally:
        saver.conn.close()


# ---------------------------------------------------------------------------
# Invariant 2: no generic shell-ish tool can be registered
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["shell", "powershell", "cmd", "exec", "eval"])
def test_invariant_2_registry_forbids_shell_tools(name: str) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    with pytest.raises(ValueError):
        registry_with(make_spec(name, base_tier=0, record=record))


def test_invariant_2_registry_name_gap_is_closed_by_policy_rules() -> None:
    """Defense in depth: a name like ``run_powershell`` slips past the registry
    word-boundary regex, so the policy rules must still hard-block it."""
    from jarvis.policy import rules

    for name in ("run_powershell", "disable_defender", "manage_password", "browser_password"):
        record: list[tuple[str, dict[str, Any]]] = []
        spec = make_spec(name, base_tier=0, record=record)
        registry_with(spec)  # registry accepts the name...
        assert rules.matches_blocked(spec, spec.args_model(text="x"))  # ...but policy blocks it


# ---------------------------------------------------------------------------
# Invariant 3: hard-blocked (Tier 3) steps are refused in code
# ---------------------------------------------------------------------------


def test_invariant_3_hard_blocked_tool_never_runs(tmp_path: Any) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    defender = make_spec("disable_defender_x", base_tier=0, record=record)
    ctx = make_app_context(
        Settings(),
        llm=FakeLLM([_plan("disable_defender_x", {"text": "on"})]),
        registry=registry_with(defender),
    )
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        outcome = run_task(ctx, saver, "disable defender")
        assert record == []
        assert outcome.confirmation is None  # blocked in code before any confirmation
        assert "Refused" in (outcome.final_answer or "")
    finally:
        saver.conn.close()


# ---------------------------------------------------------------------------
# Invariant 6: approvals bind to the exact action_hash
# ---------------------------------------------------------------------------


def _echo_context(record: list[tuple[str, dict[str, Any]]]) -> Any:
    return make_app_context(
        Settings(),
        llm=FakeLLM([echo_plan()]),
        registry=registry_with(make_spec("fake_echo", base_tier=1, record=record)),
    )


def test_invariant_6_tampered_hash_never_executes(tmp_path: Any) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    ctx = _echo_context(record)
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        first = run_task(ctx, saver, "echo hi")
        outcome = resume_task(ctx, saver, first.task_id, tamper(first.confirmation))
        assert record == []
        assert "Refused by you" in (outcome.final_answer or "")
    finally:
        saver.conn.close()


def test_invariant_6_denied_never_executes(tmp_path: Any) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    ctx = _echo_context(record)
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        first = run_task(ctx, saver, "echo hi")
        outcome = resume_task(ctx, saver, first.task_id, deny(first.confirmation))
        assert record == []
        assert "Refused by you" in (outcome.final_answer or "")
    finally:
        saver.conn.close()


def test_invariant_6_mutated_plan_after_approval_never_executes(tmp_path: Any) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    ctx = _echo_context(record)
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        first = run_task(ctx, saver, "echo hi")
        graph = build_graph(ctx, saver)
        graph.update_state(
            {"configurable": {"thread_id": first.task_id}},
            {"plan": echo_plan(text="byee")},
        )
        outcome = resume_task(ctx, saver, first.task_id, approve(first.confirmation))
        assert record == []
        assert "Refused by you" in (outcome.final_answer or "")
    finally:
        saver.conn.close()


# ---------------------------------------------------------------------------
# Invariant 7: secrets never enter checkpoint bytes or graph state
# ---------------------------------------------------------------------------


def test_invariant_7_values_in_state_never_contain_secrets(tmp_path: Any) -> None:
    """Secrets never enter graph state or the persisted channel_values.

    LangGraph stores the raw resume *input* internally in the ``__resume__``
    write channel, so the protocol guarantee (docs/03 section 5, invariant 8) is
    that the system never *puts* a secret into a resume payload - password
    verification happens in the daemon layer before the graph is resumed.  This
    test asserts that boundary: a smuggled password field shows up nowhere but
    that one langgraph-internal write channel, and never in state.
    """
    record: list[tuple[str, dict[str, Any]]] = []
    ctx = _echo_context(record)
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        first = run_task(ctx, saver, "echo hi")
        outcome = resume_task(
            ctx,
            saver,
            first.task_id,
            {
                "approved": True,
                "action_hash": first.confirmation["action_hash"],
                "password": "hunter2secret",
            },
        )
        assert record  # the approved step did run; the secret is the only concern

        # (1) persisted channel_values (checkpoints.checkpoint) stay clean.
        for (blob,) in saver.conn.execute("SELECT checkpoint FROM checkpoints"):
            assert b"hunter2secret" not in (blob or b"")

        # (2) the visible graph state stays clean.
        assert "hunter2secret" not in repr(outcome.state)

        # (3) the secret appears only in the langgraph-internal __resume__ write
        # channel, never in any application channel.
        leaks: list[str] = []
        for channel, value in saver.conn.execute("SELECT channel, value FROM writes"):
            if channel != "__resume__" and value and b"hunter2secret" in value:
                leaks.append(channel)
        assert leaks == []

        # (4) our protocol's own resume payload never carries a secret field.
        payload = approve(first.confirmation)
        assert set(payload) == {"approved", "action_hash", "resolved_paths"}
    finally:
        saver.conn.close()


# ---------------------------------------------------------------------------
# Invariant 9: failed verification is never reported as success
# ---------------------------------------------------------------------------


def test_invariant_9_verification_failure_fails_closed(tmp_path: Any) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    auto = make_spec("fake_auto", base_tier=0, record=record, verify_ok=False)
    settings = Settings(agent=AgentSettings(max_retries_per_step=1))
    ctx = make_app_context(
        settings,
        llm=FakeLLM([_plan("fake_auto", {"text": "x"})]),
        registry=registry_with(auto),
    )
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        outcome = run_task(ctx, saver, "run auto")
        # base_tier 0 so it executed without a confirmation gate, then failed verify.
        assert record  # retries re-ran the tool (1 attempt + 1 retry)
        answer = outcome.final_answer or ""
        assert answer.startswith("Could not complete the task")
        assert "could not be verified" in answer
        assert "Done" not in answer
    finally:
        saver.conn.close()


# ---------------------------------------------------------------------------
# Invariant 11: Tier 2 needs an unlocked session even if the user says yes
# ---------------------------------------------------------------------------


def test_invariant_11_tier2_refuses_without_unlocked_session(tmp_path: Any) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    t2 = make_spec("fake_tier2", base_tier=2, record=record)
    ctx = make_app_context(
        Settings(),
        llm=FakeLLM([_plan("fake_tier2", {"text": "x"})]),
        registry=registry_with(t2),
        unlock=None,
    )
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        first = run_task(ctx, saver, "do secret thing")
        assert first.confirmation is not None
        assert first.confirmation["needs_unlock"] is True
        outcome = resume_task(ctx, saver, first.task_id, approve(first.confirmation))
        assert record == []
        assert "unlocked JARVIS session" in (outcome.final_answer or "")
    finally:
        saver.conn.close()


def test_invariant_11_tier2_approval_with_unlock_still_needs_confirm() -> None:
    """A Tier-2 step with an unlocked session still pauses for confirmation."""
    record: list[tuple[str, dict[str, Any]]] = []

    class Unlocked:
        def is_unlocked(self) -> bool:
            return True

    ctx = make_app_context(
        Settings(),
        llm=FakeLLM([_plan("fake_tier2", {"text": "x"})]),
        registry=registry_with(make_spec("fake_tier2", base_tier=2, record=record)),
        unlock=Unlocked(),
    )
    ck = open_sqlite_checkpointer(":memory:")
    try:
        first = run_task(ctx, ck, "do secret thing")
        assert first.interrupted is True
        assert first.confirmation["needs_unlock"] is True
        outcome = resume_task(ctx, ck, first.task_id, approve(first.confirmation))
        assert record == [("fake_tier2", {"text": "x"})]
        assert outcome.final_answer == "fake_tier2 ran x"
    finally:
        ck.conn.close()


# ---------------------------------------------------------------------------
# Invariant 12: planner claims can never lower a tier
# ---------------------------------------------------------------------------


def test_invariant_12_llm_cannot_lower_tier(tmp_path: Any) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    # The step args smuggle a "tier": 0 claim; the real base_tier is 1 and the
    # engine ignores the extra key entirely.
    claim_plan = Plan(
        goal="echo hi",
        steps=[Step(id="s1", tool="fake_echo", args={"text": "hi", "tier": 0}, rationale="r")],
    )
    ctx = make_app_context(
        Settings(),
        llm=FakeLLM([claim_plan]),
        registry=registry_with(make_spec("fake_echo", base_tier=1, record=record)),
    )
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        first = run_task(ctx, saver, "echo hi")
        assert first.confirmation is not None
        assert first.confirmation["tier"] == 1  # base_tier won, not the LLM's claim
    finally:
        saver.conn.close()


# ---------------------------------------------------------------------------
# Invariant 13: no side effects before interrupt; resume reuses the plan
# ---------------------------------------------------------------------------


def test_invariant_13_no_side_effects_before_interrupt(tmp_path: Any) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    llm = FakeLLM([echo_plan()])
    ctx = make_app_context(
        Settings(),
        llm=llm,
        registry=registry_with(make_spec("fake_echo", base_tier=1, record=record)),
    )
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        first = run_task(ctx, saver, "echo hi")
        assert record == []  # nothing executed before the confirmation gate
        assert len(llm.calls) == 1  # planner consulted exactly once pre-interrupt
        outcome = resume_task(ctx, saver, first.task_id, approve(first.confirmation))
        assert record == [("fake_echo", {"text": "hi"})]  # executed exactly once
        assert len(llm.calls) == 1  # resume does not re-run the planner
        assert outcome.final_answer == "fake_echo ran hi"
    finally:
        saver.conn.close()


def test_no_blocked_deserialization_warnings_through_full_flow(tmp_path: Any, caplog: Any) -> None:
    """The whole checkpoint path uses the allowlist; nothing is blocked silently."""
    record: list[tuple[str, dict[str, Any]]] = []
    llm = FakeLLM([echo_plan()])
    ctx = make_app_context(
        Settings(),
        llm=llm,
        registry=registry_with(make_spec("fake_echo", base_tier=1, record=record)),
    )
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        with caplog.at_level(logging.WARNING, logger="langgraph.checkpoint.serde.jsonplus"):
            first = run_task(ctx, saver, "echo hi")
            resume_task(ctx, saver, first.task_id, approve(first.confirmation))
        assert "Blocked deserialization" not in caplog.text
        assert "Deserializing unregistered" not in caplog.text
    finally:
        saver.conn.close()


# ---------------------------------------------------------------------------
# Invariant 4: protected/escaping paths can never be targeted
# ---------------------------------------------------------------------------


def test_invariant_4_delete_escape_and_protected_paths_never_even_decide(tmp_path: Any) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    ctx = make_app_context(
        Settings(policy=PolicySettings(allowed_roots=[str(ws)])),
        registry=registry_with(make_delete_path_spec()),
    )
    targets = [str(ws / ".." / ".." / ".." / "Windows" / "x"), str(tmp_path / "outside")]
    if os.name == "nt":
        targets.append(str(Path(os.environ["SystemRoot"]) / "System32"))
    for raw in targets:
        decision = ctx.engine.decide(_step("delete_path", {"paths": [raw]}), ctx.policy_ctx)
        assert decision.allowed is False, raw
        assert decision.tier == 3
    # the allowed root itself / an ancestor of it is also untouchable
    for raw in (str(ws), str(ws.parent)):
        decision = ctx.engine.decide(_step("delete_path", {"paths": [raw]}), ctx.policy_ctx)
        assert decision.allowed is False, raw


# ---------------------------------------------------------------------------
# Invariant 5: deletes go to the Recycle Bin (undo-logged), never permanent
# ---------------------------------------------------------------------------


def test_invariant_5_delete_uses_bin_undo_log_and_no_permanent_fallback(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    settings = Settings(policy=PolicySettings(allowed_roots=[str(ws)]))
    target = ws / "precious.txt"
    target.write_text("do not lose me", encoding="utf-8")

    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("permanent deletion API used")

    monkeypatch.setattr(os, "remove", _boom)
    monkeypatch.setattr(shutil, "rmtree", _boom)

    ctx = ToolContext(
        settings=settings,
        trash=FakeDirTrash(tmp_path / "trash"),
        undo_log=tmp_path / "undo.jsonl",
    )
    spec = make_delete_path_spec()
    result = spec.run(spec.args_model(paths=[str(target)]), ctx)
    assert result.ok is True, result.error
    assert target.exists() is False  # moved to the bin
    assert tuple((tmp_path / "trash").iterdir())  # something is in the bin
    log = (tmp_path / "undo.jsonl").read_text(encoding="utf-8")
    record = json.loads(log.strip().splitlines()[-1])
    assert record["op"] == "delete"
    assert record["succeeded"] is True


# ---------------------------------------------------------------------------
# Invariant 7 & 14: the JARVIS password is hash-only and outside the graph
# ---------------------------------------------------------------------------


class _PwStore:
    def __init__(self) -> None:
        self._values: dict[str, str] = {}

    def get(self, name: str) -> str | None:
        return self._values.get(name)

    def set(self, name: str, value: str) -> None:
        self._values[name] = value

    def has(self, name: str) -> bool:
        return name in self._values


def test_invariant_7_and_14_jarvis_password_lives_only_as_a_hash_outside_the_graph(
    tmp_path: Any, caplog: Any
) -> None:
    """The password enters through the manager, not the resume payload; only
    the Argon2id hash is stored; nothing reaches state, checkpoints or logs."""
    store = _PwStore()
    manager = UnlockManager(store)  # real argon2 hashing
    manager.set_password("hunter2-jarvis-1")

    record: list[tuple[str, dict[str, Any]]] = []
    ctx = make_app_context(
        Settings(),
        llm=FakeLLM([_plan("fake_tier2", {"text": "x"})]),
        registry=registry_with(make_spec("fake_tier2", base_tier=2, record=record)),
        unlock=manager,
    )
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        first = run_task(ctx, saver, "do secret thing")
        # the password is verified OUTSIDE the graph; resume only forwards
        # approved+action_hash (invariant 8).
        assert manager.verify("hunter2-jarvis-1") is True
        with caplog.at_level(logging.DEBUG):
            outcome = resume_task(ctx, saver, first.task_id, approve(first.confirmation))
        assert record == [("fake_tier2", {"text": "x"})]

        # only an Argon2id hash is stored; plaintext is nowhere
        stored = store.get("password_hash")
        assert stored is not None and stored.startswith("$argon2id$")
        assert "hunter2-jarvis-1" not in stored

        # no log line, checkpoint blob or state carries the password
        assert "hunter2-jarvis-1" not in caplog.text
        for (blob,) in saver.conn.execute("SELECT checkpoint FROM checkpoints"):
            assert b"hunter2-jarvis-1" not in (blob or b"")
        assert "hunter2-jarvis-1" not in repr(outcome.state)
    finally:
        saver.conn.close()


# ---------------------------------------------------------------------------
# Invariant 10: IPC requires token — unauthenticated connection is rejected
# ---------------------------------------------------------------------------


def test_invariant_10_ipc_requires_token() -> None:
    """Unauthenticated or wrong-token IPC connections are rejected.

    This test verifies the auth gate by attempting to connect to a daemon
    with a wrong token and verifying the connection is closed without
    executing any command.
    """
    import json
    import threading
    import time

    from jarvis.config import Settings
    from jarvis.daemon.protocol import AuthMessage, StatusRequest
    from jarvis.daemon.server import DaemonServer

    TEST_TOKEN = "invariant10-test-token"

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

    from unittest.mock import MagicMock

    settings = Settings()
    store = _FakeStore()
    ctx = MagicMock()
    server = DaemonServer(settings=settings, store=store, ctx=ctx)

    loop = asyncio.new_event_loop()
    actual_port = 0

    async def _start() -> None:
        nonlocal actual_port
        srv = await asyncio.start_server(server._handle_client, "127.0.0.1", 0)
        actual_port = srv.sockets[0].getsockname()[1]
        async with srv:
            await server._shutdown_event.wait()

    def _run() -> None:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_start())

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    time.sleep(0.3)

    try:
        import asyncio as _aio

        async def _test_auth() -> None:
            # Test 1: wrong token -> rejected
            reader, writer = await _aio.open_connection("127.0.0.1", actual_port)
            line = AuthMessage(token="wrong-token").model_dump_json() + "\n"
            writer.write(line.encode())
            await writer.drain()
            raw = await _aio.wait_for(reader.readline(), timeout=2)
            # Connection should be closed (EOF or error)
            assert raw == b"" or b"auth" in raw.lower()
            writer.close()

            # Test 2: correct token -> accepted
            reader, writer = await _aio.open_connection("127.0.0.1", actual_port)
            line = AuthMessage(token=TEST_TOKEN).model_dump_json() + "\n"
            writer.write(line.encode())
            await writer.drain()
            raw = await _aio.wait_for(reader.readline(), timeout=2)
            data = json.loads(raw.decode())
            assert data["type"] == "auth_ok"
            writer.close()

            # Test 3: no auth message -> rejected
            reader, writer = await _aio.open_connection("127.0.0.1", actual_port)
            line = StatusRequest().model_dump_json() + "\n"
            writer.write(line.encode())
            await writer.drain()
            raw = await _aio.wait_for(reader.readline(), timeout=2)
            assert raw == b""  # closed without response
            writer.close()

        _aio.run(_test_auth())
    finally:
        # The server loop is blocked on _shutdown_event.wait() in the
        # background thread; cancelling tasks does not wake it.  Use the
        # thread-safe path (same as _ServerHarness.stop()).
        loop.call_soon_threadsafe(server._shutdown_event.set)
        t.join(timeout=3)
        loop.close()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _plan(tool: str, args: dict[str, Any]) -> Plan:
    return Plan(goal=f"run {tool}", steps=[Step(id="s1", tool=tool, args=args, rationale="r")])


def _step(tool: str, args: dict[str, Any]) -> Step:
    return Step(id="s1", tool=tool, args=args, rationale="r")
