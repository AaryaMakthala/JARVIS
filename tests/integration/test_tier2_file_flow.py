"""End-to-end Tier-2 delete flows over the real compiled graph.

Uses the real ``delete_path`` / ``undo_last_delete`` tools and the real
:class:`UnlockManager`, but a fake Recycle Bin (``FakeDirTrash``) and an
injected undo log, so no system paths are touched.  Deliberately carries no
``windows_only`` / ``slow`` markers: the whole workflow runs on the default CI
set (docs/05_BUILD_PLAN.md Phase 2 acceptance E2E-3..E2E-7).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from jarvis.agent.context import make_app_context
from jarvis.agent.graph import open_sqlite_checkpointer
from jarvis.agent.runner import resume_task, run_task
from jarvis.agent.state import Plan, Step
from jarvis.config import PolicySettings, Settings
from jarvis.llm.client import FakeLLM
from jarvis.policy.unlock import UnlockManager
from jarvis.tools.files import make_delete_path_spec, make_undo_last_delete_spec
from support import FakeDirTrash, approve, approve_typed, registry_with

PW = "correct-horse-battery-staple"


class _PwStore:
    def __init__(self) -> None:
        self._values: dict[str, str] = {}

    def get(self, name: str) -> str | None:
        return self._values.get(name)

    def set(self, name: str, value: str) -> None:
        self._values[name] = value

    def has(self, name: str) -> bool:
        return name in self._values


def _make_manager() -> UnlockManager:
    manager = UnlockManager(_PwStore())
    manager.set_password(PW)
    return manager


def _settings(ws: Path) -> Settings:
    return Settings(policy=PolicySettings(allowed_roots=[str(ws.resolve())]))


def _make_ctx(ws: Path, tmp_path: Path, plans: list[Any], manager: UnlockManager) -> Any:
    registry = registry_with(make_delete_path_spec(), make_undo_last_delete_spec())
    return make_app_context(
        _settings(ws),
        llm=FakeLLM(plans),
        registry=registry,
        unlock=manager,
        trash=FakeDirTrash(tmp_path / "trash"),
        undo_log=tmp_path / "undo.jsonl",
    )


def _delete_plan(targets: list[str]) -> Plan:
    return Plan(
        goal="delete files",
        steps=[
            Step(id="s1", tool="delete_path", args={"paths": targets}, rationale="delete them"),
        ],
    )


def _undo_plan() -> Plan:
    return Plan(
        goal="undo last delete",
        steps=[Step(id="u1", tool="undo_last_delete", args={}, rationale="restore it")],
    )


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    return root


def _write(ws: Path, name: str, content: str = "hello") -> Path:
    target = ws / name
    target.write_text(content, encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# full happy path: delete (unlock + confirm) then undo
# ---------------------------------------------------------------------------


def test_e2e_delete_needs_unlock_then_undo_restores(tmp_path: Path, ws: Path) -> None:
    a = _write(ws, "a.txt", "content-a")
    b = _write(ws, "b.txt", "content-b")
    manager = _make_manager()
    plan = _delete_plan([str(a), str(b)])
    ctx = _make_ctx(ws, tmp_path, [plan, plan], manager)
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        # task 1: approving while the session is locked must be refused (inv 11)
        first = run_task(ctx, saver, "delete a and b")
        assert first.interrupted is True
        req = first.confirmation
        assert req["tier"] == 2
        assert req["needs_unlock"] is True
        assert req["resolved_paths"] == [str(a.resolve(strict=False)), str(b.resolve(strict=False))]

        locked = resume_task(ctx, saver, first.task_id, approve(req))
        assert locked.halted_reason is not None
        assert "unlocked JARVIS session" in (locked.final_answer or "")
        assert a.exists() and b.exists()

        # task 2: unlock outside the graph, then approval runs it
        second = run_task(ctx, saver, "delete a and b again")
        assert manager.verify(PW) is True
        approved = resume_task(ctx, saver, second.task_id, approve(second.confirmation))
        assert approved.halted_reason is None
        assert a.exists() is False and b.exists() is False
        assert len(list((tmp_path / "trash").iterdir())) == 2
        undo_log = (tmp_path / "undo.jsonl").read_text(encoding="utf-8")
        assert undo_log.count('"op": "delete"') == 2

        # task 3: undo_last_delete restores the most recent (b)
        ctx2 = _make_ctx(ws, tmp_path, [_undo_plan()], _make_manager())
        third = run_task(ctx2, saver, "undo the last delete")
        assert third.interrupted is True
        assert third.confirmation["tier"] == 1
        undone = resume_task(ctx2, saver, third.task_id, approve(third.confirmation))
        assert (undone.final_answer or "").startswith("restored ")
        assert (undone.final_answer or "").endswith("from the Recycle Bin")
        assert b.exists() and b.read_text(encoding="utf-8") == "content-b"
        assert a.exists() is False
    finally:
        saver.conn.close()


# ---------------------------------------------------------------------------
# typed folder-name confirmation
# ---------------------------------------------------------------------------


def test_e2e_folder_delete_needs_typed_confirmation(tmp_path: Path, ws: Path) -> None:
    folder = ws / "Reports"
    folder.mkdir()
    (folder / "summary.txt").write_text("x", encoding="utf-8")
    manager = _make_manager()
    plan = _delete_plan([str(folder)])
    ctx = _make_ctx(ws, tmp_path, [plan, plan], manager)
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        # task 1: approve WITHOUT typing the folder name -> refused
        first = run_task(ctx, saver, "delete the reports folder")
        req = first.confirmation
        assert req["typed_confirmation"] == "Reports"
        refused = resume_task(ctx, saver, first.task_id, approve(req))
        assert "typed folder-name confirmation" in (refused.final_answer or "")
        assert folder.exists()

        # task 2: unlock, then approve WITH the typed name -> runs
        second = run_task(ctx, saver, "delete the reports folder again")
        assert manager.verify(PW) is True
        done = resume_task(
            ctx, saver, second.task_id, approve_typed(second.confirmation, "Reports")
        )
        assert done.halted_reason is None
        assert folder.exists() is False
    finally:
        saver.conn.close()


def test_e2e_folder_delete_wrong_typed_name_is_refused(tmp_path: Path, ws: Path) -> None:
    folder = ws / "Docs"
    folder.mkdir()
    manager = _make_manager()
    ctx = _make_ctx(ws, tmp_path, [_delete_plan([str(folder)])], manager)
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        first = run_task(ctx, saver, "delete the docs folder")
        assert manager.verify(PW) is True
        refused = resume_task(
            ctx, saver, first.task_id, approve_typed(first.confirmation, "wrong name")
        )
        assert "typed folder-name confirmation" in (refused.final_answer or "")
        assert folder.exists()
    finally:
        saver.conn.close()


# ---------------------------------------------------------------------------
# lock invalidates an already-approved Tier-2 action
# ---------------------------------------------------------------------------


def test_e2e_locking_after_interrupt_invalidates_approval(tmp_path: Path, ws: Path) -> None:
    target = _write(ws, "keepme.txt")
    manager = _make_manager()
    assert manager.verify(PW) is True  # unlocked
    ctx = _make_ctx(ws, tmp_path, [_delete_plan([str(target)])], manager)
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        first = run_task(ctx, saver, "delete keepme")
        manager.lock()
        outcome = resume_task(ctx, saver, first.task_id, approve(first.confirmation))
        assert "unlocked JARVIS session" in (outcome.final_answer or "")
        assert target.exists()
    finally:
        saver.conn.close()


# ---------------------------------------------------------------------------
# TOCTOU: a path swapped for a junction after confirmation is refused
# ---------------------------------------------------------------------------


def test_e2e_path_swapped_for_junction_after_confirmation_is_refused(
    tmp_path: Path, ws: Path
) -> None:
    if os.name != "nt":
        pytest.skip("junctions only exist on Windows")
    target = _write(ws, "a.txt")
    (ws / "b_dir").mkdir()
    manager = _make_manager()
    ctx = _make_ctx(ws, tmp_path, [_delete_plan([str(target)])], manager)
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        first = run_task(ctx, saver, "delete a.txt")
        # swap a.txt for a junction pointing at a folder we never approved
        os.remove(str(target))
        rc = os.system(f'mklink /J "{ws / "a.txt"}" "{ws / "b_dir"}"')
        if rc != 0:
            pytest.skip("junction creation unavailable in this environment")
        assert manager.verify(PW) is True
        outcome = resume_task(ctx, saver, first.task_id, approve(first.confirmation))
        assert (outcome.final_answer or "").startswith("Refused: a file path changed")
        assert (ws / "b_dir").exists()  # the junction target was never deleted
    finally:
        saver.conn.close()


# ---------------------------------------------------------------------------
# protected path never even reaches the confirmation gate
# ---------------------------------------------------------------------------


def test_e2e_delete_protected_path_blocks_before_confirmation(tmp_path: Path, ws: Path) -> None:
    if os.name != "nt":
        pytest.skip("protected OS paths only exist on Windows")
    protected = str(Path(os.environ["SystemRoot"]) / "System32")
    manager = _make_manager()
    ctx = _make_ctx(ws, tmp_path, [_delete_plan([protected])], manager)
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        outcome = run_task(ctx, saver, "delete system32")
        assert outcome.interrupted is False
        assert outcome.confirmation is None
        assert "Refused: " in (outcome.final_answer or "")
        assert len(list((tmp_path / "trash").iterdir())) == 0
        assert (tmp_path / "undo.jsonl").exists() is False
    finally:
        saver.conn.close()
