"""Stage 8.4d: an opt-in tool's approval is bound to the exact file content.

Two layers are pinned here:

* the policy gate records a sha256 per resolved path **only after** every
  approval check succeeds (a denied/timed-out answer records nothing), and
* the act node re-reads the same paths immediately before dispatch and refuses
  on any change, missing/over-cap file, malformed snapshot or missing binding.

The gate tests drive the real compiled graph over a SQLite checkpointer, so the
interrupt/resume sequence is exactly the one a user sees.  The act tests drive
the real act node directly (as ``test_act_target_parity.py`` does) because a
change made *after* the gate recorded the binding must be simulated between
gate and dispatch.  Everything is deterministic: no network, no Windows GUI.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel, Field
from tests.support import approve, brain_action, deny, registry_with

from jarvis.agent.context import make_app_context
from jarvis.agent.graph import open_sqlite_checkpointer
from jarvis.agent.nodes.act import act
from jarvis.agent.runner import resume_task, run_task
from jarvis.agent.state import Plan, Step
from jarvis.config import PolicySettings, Settings
from jarvis.llm.client import FakeLLM
from jarvis.policy import rules
from jarvis.policy.paths import CONTENT_HASH_MAX_BYTES, ContentHashError, content_hash
from jarvis.policy.unlock import UnlockManager
from jarvis.tools.base import ToolContext, ToolResult, ToolSpec

PW = "correct-horse-battery-staple"
TASK = "t1"
TOOL = "fake_send_file"


class FileArgs(BaseModel):
    path: str = Field(min_length=1, max_length=4096)


def _spec(record: list[tuple[str, tuple[str, ...] | None]], *, bind: bool = True) -> ToolSpec:
    """A fake Tier 2 file tool that records the digests act handed it."""

    def run(args: FileArgs, ctx: ToolContext) -> ToolResult:
        record.append((args.path, getattr(ctx, "approved_content_hashes", None)))
        return ToolResult(ok=True, output="sent")

    return ToolSpec(
        name=TOOL,
        description="fake content-bound file send",
        args_model=FileArgs,
        base_tier=2,
        path_args=("path",),
        bind_content_hash=bind,
        run=run,
        describe=lambda args: f"send {args.path}",
    )


class _PwStore:
    """In-memory password store (no keyring, no real secrets)."""

    def __init__(self) -> None:
        self._values: dict[str, str] = {}

    def get(self, name: str) -> str | None:
        return self._values.get(name)

    def set(self, name: str, value: str) -> None:
        self._values[name] = value

    def has(self, name: str) -> bool:
        return name in self._values


def _unlocked_manager() -> UnlockManager:
    manager = UnlockManager(_PwStore())
    manager.set_password(PW)
    assert manager.verify(PW) is True
    return manager


def _settings(ws: Path) -> Settings:
    return Settings(policy=PolicySettings(allowed_roots=[str(ws.resolve())]))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolved(path: Path) -> str:
    return str(path.resolve(strict=False))


def _grow_past_cap(path: Path) -> None:
    """Make ``path`` one byte larger than the content-binding cap (sparse/fast)."""
    with path.open("r+b") as handle:
        handle.truncate(CONTENT_HASH_MAX_BYTES + 1)


# ── ToolSpec opt-in ─────────────────────────────────────────────────────────


def test_bind_content_hash_is_opt_in_and_defaults_off() -> None:
    record: list[tuple[str, tuple[str, ...] | None]] = []
    assert _spec(record, bind=False).bind_content_hash is False
    assert _spec(record).bind_content_hash is True


# ── content_hash itself ────────────────────────────────────────────────────


def test_content_hash_matches_sha256_of_the_bytes(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    target.write_bytes(b"hello world")
    assert content_hash(target) == hashlib.sha256(b"hello world").hexdigest()


def test_content_hash_of_an_empty_file(tmp_path: Path) -> None:
    target = tmp_path / "empty.txt"
    target.write_bytes(b"")
    assert content_hash(target) == hashlib.sha256(b"").hexdigest()


def test_content_hash_refuses_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ContentHashError, match="cannot stat"):
        content_hash(tmp_path / "missing.txt")


def test_content_hash_refuses_a_directory(tmp_path: Path) -> None:
    with pytest.raises(ContentHashError, match="not a regular file"):
        content_hash(tmp_path)


def test_content_hash_refuses_a_file_over_the_cap(tmp_path: Path) -> None:
    target = tmp_path / "big.txt"
    target.write_bytes(b"12345")
    with pytest.raises(ContentHashError, match="cap"):
        content_hash(target, max_bytes=4)


def test_content_hash_enforces_the_cap_while_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stat that lies about the size must not let an over-cap read through."""
    target = tmp_path / "growing.txt"
    target.write_bytes(b"12345")
    real_stat = Path.stat

    def fake_stat(self: Path, *, follow_symlinks: bool = True) -> Any:
        stat = real_stat(self, follow_symlinks=follow_symlinks)
        return SimpleNamespace(st_mode=stat.st_mode, st_size=0)

    monkeypatch.setattr(Path, "stat", fake_stat)
    with pytest.raises(ContentHashError, match="cap"):
        content_hash(target, max_bytes=4)


def test_content_hash_refuses_an_unreadable_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "locked.txt"
    target.write_bytes(b"secret")

    def boom(self: Path, *args: Any, **kwargs: Any) -> Any:
        raise PermissionError("no access")

    monkeypatch.setattr(Path, "open", boom)
    with pytest.raises(ContentHashError, match="cannot read"):
        content_hash(target)


def test_default_cap_is_sixteen_mebibytes() -> None:
    assert CONTENT_HASH_MAX_BYTES == 16 * 1024 * 1024


# ── gate: bind only after approval succeeds ────────────────────────────────


def _graph_ctx(ws: Path, target: Path, record: list[tuple[str, Any]]) -> Any:
    return make_app_context(
        _settings(ws),
        llm=FakeLLM([brain_action(TOOL, {"path": str(target)})]),
        registry=registry_with(_spec(record)),
        unlock=_unlocked_manager(),
    )


def _run_to_interrupt(tmp_path: Path, ws: Path, target: Path, record: list[tuple[str, Any]]) -> Any:
    ctx = _graph_ctx(ws, target, record)
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        first = run_task(ctx, saver, "send the file")
        assert first.interrupted is True, (
            f"a Tier 2 send must interrupt, got {first.final_answer!r}"
        )
        assert (first.confirmation or {}).get("step_id") == "s1"
        return ctx, saver, first
    except Exception:
        saver.conn.close()
        raise


def test_approval_records_the_bound_digest_and_runs(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    target = ws / "report.txt"
    target.write_text("report", encoding="utf-8")
    record: list[tuple[str, tuple[str, ...] | None]] = []
    ctx, saver, first = _run_to_interrupt(tmp_path, ws, target, record)
    try:
        second = resume_task(ctx, saver, first.task_id, approve(first.confirmation or {}))
        assert record == [(str(target), (_digest(target),))], (
            "the tool must receive the approved digest only after act re-read it"
        )
        assert second.state["gated_content_hashes"] == {"s1": [_digest(target)]}
        assert str(first.confirmation["action_hash"]) in second.state["approved_hashes"]
    finally:
        saver.conn.close()


def test_a_denied_approval_records_no_binding_and_runs_nothing(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    target = ws / "report.txt"
    target.write_text("report", encoding="utf-8")
    record: list[tuple[str, Any]] = []
    ctx, saver, first = _run_to_interrupt(tmp_path, ws, target, record)
    try:
        second = resume_task(ctx, saver, first.task_id, deny(first.confirmation or {}))
        assert record == []
        assert "s1" not in (second.state.get("gated_content_hashes") or {})
        assert second.state.get("halted_reason")
    finally:
        saver.conn.close()


def test_a_timed_out_approval_records_no_binding_and_runs_nothing(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    target = ws / "report.txt"
    target.write_text("report", encoding="utf-8")
    record: list[tuple[str, Any]] = []
    ctx, saver, first = _run_to_interrupt(tmp_path, ws, target, record)
    try:
        second = resume_task(ctx, saver, first.task_id, {"timed_out": True})
        assert record == []
        assert "s1" not in (second.state.get("gated_content_hashes") or {})
        assert "timed out" in (second.state.get("halted_reason") or "")
    finally:
        saver.conn.close()


def test_a_file_removed_before_approval_is_refused_and_runs_nothing(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    target = ws / "report.txt"
    target.write_text("report", encoding="utf-8")
    record: list[tuple[str, Any]] = []
    ctx, saver, first = _run_to_interrupt(tmp_path, ws, target, record)
    try:
        target.unlink()
        second = resume_task(ctx, saver, first.task_id, approve(first.confirmation or {}))
        assert record == []
        assert "s1" not in (second.state.get("gated_content_hashes") or {})
        assert "could not bind approval" in (second.state.get("halted_reason") or "")
        assert str(first.confirmation["action_hash"]) not in (
            second.state.get("approved_hashes") or []
        )
    finally:
        saver.conn.close()


def test_a_file_replaced_by_a_directory_before_approval_is_refused(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    target = ws / "report.txt"
    target.write_text("report", encoding="utf-8")
    record: list[tuple[str, Any]] = []
    ctx, saver, first = _run_to_interrupt(tmp_path, ws, target, record)
    try:
        target.unlink()
        target.mkdir()
        second = resume_task(ctx, saver, first.task_id, approve(first.confirmation or {}))
        assert record == []
        assert "s1" not in (second.state.get("gated_content_hashes") or {})
        assert "not a regular file" in (second.state.get("halted_reason") or "")
    finally:
        saver.conn.close()


def test_an_over_cap_file_before_approval_is_refused(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    target = ws / "report.bin"
    target.write_bytes(b"small")
    record: list[tuple[str, Any]] = []
    ctx, saver, first = _run_to_interrupt(tmp_path, ws, target, record)
    try:
        _grow_past_cap(target)
        second = resume_task(ctx, saver, first.task_id, approve(first.confirmation or {}))
        assert record == []
        assert "s1" not in (second.state.get("gated_content_hashes") or {})
        assert "cap" in (second.state.get("halted_reason") or "")
    finally:
        saver.conn.close()


# ── act: verify immediately before dispatch ────────────────────────────────


def _step(target: Path) -> Step:
    return Step(id="s1", tool=TOOL, args={"path": str(target)}, rationale="send it")


def _action_hash(ctx: Any, step: Step) -> str:
    spec = ctx.registry.get(step.tool)
    args = spec.args_model.model_validate(step.args)
    return str(rules.action_hash(spec, args))


def _act_ctx(
    ws: Path, record: list[tuple[str, Any]], *, bind: bool = True, dry_run: bool = False
) -> Any:
    return make_app_context(
        _settings(ws),
        registry=registry_with(_spec(record, bind=bind)),
        dry_run=dry_run,
    )


def _direct_state(
    step: Step,
    *,
    action_hash: str,
    snapshot: Any = None,
    snapshot_key: str | None = None,
    include_paths: bool = True,
    approved: bool = True,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "plan": Plan(goal="send the file", steps=[step]),
        "step_index": 0,
        "task_id": TASK,
        "approved_hashes": [action_hash] if approved else [],
        "results": [],
        "user_input": "send the file",
    }
    if include_paths:
        state["gated_resolved_paths"] = {step.id: [_resolved(Path(step.args["path"]))]}
    if snapshot is not None:
        state["gated_content_hashes"] = {snapshot_key or step.id: snapshot}
    return state


def _result(update: dict[str, Any]) -> Any:
    results = list(update.get("results") or [])
    assert results, "act must record a result"
    return results[-1]


def test_act_dispatches_when_the_content_matches(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    target = ws / "report.txt"
    target.write_text("report", encoding="utf-8")
    record: list[tuple[str, Any]] = []
    ctx = _act_ctx(ws, record)
    step = _step(target)
    digest = _digest(target)

    update = act(_direct_state(step, action_hash=_action_hash(ctx, step), snapshot=[digest]), ctx)

    assert _result(update).ok, _result(update).error
    assert record == [(str(target), (digest,))]


def test_act_refuses_when_the_content_changed_after_approval(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    target = ws / "report.txt"
    target.write_text("report", encoding="utf-8")
    record: list[tuple[str, Any]] = []
    ctx = _act_ctx(ws, record)
    step = _step(target)
    old_digest = _digest(target)
    target.write_text("tampered", encoding="utf-8")

    update = act(
        _direct_state(step, action_hash=_action_hash(ctx, step), snapshot=[old_digest]), ctx
    )

    result = _result(update)
    assert not result.ok and result.error == "file content changed after confirmation"
    assert record == []
    assert update.get("halted_reason")


def test_act_refuses_when_the_file_is_missing(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    target = ws / "report.txt"
    target.write_text("report", encoding="utf-8")
    record: list[tuple[str, Any]] = []
    ctx = _act_ctx(ws, record)
    step = _step(target)
    digest = _digest(target)
    target.unlink()

    update = act(_direct_state(step, action_hash=_action_hash(ctx, step), snapshot=[digest]), ctx)

    result = _result(update)
    assert not result.ok and "could not be re-verified" in result.error
    assert record == []


def test_act_refuses_an_over_cap_file(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    target = ws / "report.bin"
    target.write_bytes(b"report")
    record: list[tuple[str, Any]] = []
    ctx = _act_ctx(ws, record)
    step = _step(target)
    digest = _digest(target)
    _grow_past_cap(target)

    update = act(_direct_state(step, action_hash=_action_hash(ctx, step), snapshot=[digest]), ctx)

    result = _result(update)
    assert not result.ok and "could not be re-verified" in result.error
    assert record == []


def test_act_refuses_a_missing_binding(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    target = ws / "report.txt"
    target.write_text("report", encoding="utf-8")
    record: list[tuple[str, Any]] = []
    ctx = _act_ctx(ws, record)
    step = _step(target)

    update = act(_direct_state(step, action_hash=_action_hash(ctx, step)), ctx)

    result = _result(update)
    assert not result.ok and "missing or malformed" in result.error
    assert record == []


def test_act_refuses_a_binding_recorded_for_another_step(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    target = ws / "report.txt"
    target.write_text("report", encoding="utf-8")
    record: list[tuple[str, Any]] = []
    ctx = _act_ctx(ws, record)
    step = _step(target)

    update = act(
        _direct_state(
            step,
            action_hash=_action_hash(ctx, step),
            snapshot=[_digest(target)],
            snapshot_key="s9",
        ),
        ctx,
    )

    result = _result(update)
    assert not result.ok and "missing or malformed" in result.error
    assert record == []


@pytest.mark.parametrize("snapshot", ["not-a-list", [], [123], [""]])
def test_act_refuses_a_malformed_binding(tmp_path: Path, snapshot: Any) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    target = ws / "report.txt"
    target.write_text("report", encoding="utf-8")
    record: list[tuple[str, Any]] = []
    ctx = _act_ctx(ws, record)
    step = _step(target)

    update = act(_direct_state(step, action_hash=_action_hash(ctx, step), snapshot=snapshot), ctx)

    result = _result(update)
    assert not result.ok and "missing or malformed" in result.error
    assert record == []


def test_act_refuses_when_the_resolved_paths_snapshot_is_missing(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    target = ws / "report.txt"
    target.write_text("report", encoding="utf-8")
    record: list[tuple[str, Any]] = []
    ctx = _act_ctx(ws, record)
    step = _step(target)

    update = act(
        _direct_state(
            step,
            action_hash=_action_hash(ctx, step),
            snapshot=[_digest(target)],
            include_paths=False,
        ),
        ctx,
    )

    result = _result(update)
    assert not result.ok and "not associated" in result.error
    assert record == []


def test_act_refuses_a_hash_of_some_other_file(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    target = ws / "report.txt"
    target.write_text("report", encoding="utf-8")
    other = ws / "other.txt"
    other.write_text("something else", encoding="utf-8")
    record: list[tuple[str, Any]] = []
    ctx = _act_ctx(ws, record)
    step = _step(target)

    update = act(
        _direct_state(step, action_hash=_action_hash(ctx, step), snapshot=[_digest(other)]),
        ctx,
    )

    result = _result(update)
    assert not result.ok and result.error == "file content changed after confirmation"
    assert record == []


def test_act_still_requires_the_approved_action_hash(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    target = ws / "report.txt"
    target.write_text("report", encoding="utf-8")
    record: list[tuple[str, Any]] = []
    ctx = _act_ctx(ws, record)
    step = _step(target)

    update = act(
        _direct_state(
            step,
            action_hash=_action_hash(ctx, step),
            snapshot=[_digest(target)],
            approved=False,
        ),
        ctx,
    )

    result = _result(update)
    assert not result.ok and result.error == "approval binding mismatch"
    assert record == []


def test_a_non_opt_in_tool_is_unaffected(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    target = ws / "report.txt"
    target.write_text("report", encoding="utf-8")
    record: list[tuple[str, Any]] = []
    ctx = _act_ctx(ws, record, bind=False)
    step = _step(target)

    update = act(_direct_state(step, action_hash=_action_hash(ctx, step)), ctx)

    assert _result(update).ok, _result(update).error
    assert record == [(str(target), None)]


def test_a_dry_run_needs_no_binding_and_executes_nothing(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    target = ws / "report.txt"
    target.write_text("report", encoding="utf-8")
    record: list[tuple[str, Any]] = []
    ctx = _act_ctx(ws, record, dry_run=True)
    step = _step(target)

    update = act(_direct_state(step, action_hash=_action_hash(ctx, step)), ctx)

    assert _result(update).ok, _result(update).error
    assert record == []
