"""TOCTOU state-based security tests for the policy_gate node.

Verifies that the authoritative pre-interrupt resolved paths (stored in
``gated_resolved_paths`` by ``validate``) are compared against the fresh
filesystem resolution after resume, and that the comparison is engine-only
(answer fields are not consulted).
"""

from __future__ import annotations

import os
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
from jarvis.tools.files import make_delete_path_spec
from support import FakeDirTrash, approve, registry_with

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
    registry = registry_with(make_delete_path_spec())
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
            Step(
                id="s1",
                tool="delete_path",
                args={"paths": targets},
                rationale="delete them",
            ),
        ],
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


# -------------------------------------------------------------------
# 1. Normal unchanged path -> allowed to continue
# -------------------------------------------------------------------


def test_toctou_unchanged_path_passes(tmp_path: Path, ws: Path) -> None:
    """When the filesystem is unchanged between interrupt and resume,
    the TOCTOU guard passes and the action executes normally."""
    target = _write(ws, "keep.txt", "precious")
    manager = _make_manager()
    ctx = _make_ctx(ws, tmp_path, [_delete_plan([str(target)])], manager)
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        first = run_task(ctx, saver, "delete keep.txt")
        assert first.interrupted is True
        assert first.confirmation is not None
        assert manager.verify(PW) is True

        # No filesystem change -- approve
        outcome = resume_task(ctx, saver, first.task_id, approve(first.confirmation))
        assert outcome.halted_reason is None
        assert target.exists() is False  # deleted successfully
    finally:
        saver.conn.close()


# -------------------------------------------------------------------
# 2. File -> junction-to-directory -> refused
# -------------------------------------------------------------------


@pytest.mark.skipif(os.name != "nt", reason="junctions only on Windows")
def test_toctou_file_to_junction_dir_refused(tmp_path: Path, ws: Path) -> None:
    """A file swapped for a junction pointing at a directory must be refused
    because the resolved path changed after confirmation."""
    target = _write(ws, "a.txt")
    (ws / "b_dir").mkdir()
    manager = _make_manager()
    ctx = _make_ctx(ws, tmp_path, [_delete_plan([str(target)])], manager)
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        first = run_task(ctx, saver, "delete a.txt")
        assert first.interrupted is True
        assert manager.verify(PW) is True

        # Swap file for junction
        os.remove(str(target))
        rc = os.system(f'mklink /J "{ws / "a.txt"}" "{ws / "b_dir"}"')
        if rc != 0:
            pytest.skip("junction creation unavailable")

        outcome = resume_task(ctx, saver, first.task_id, approve(first.confirmation))
        assert "Refused: a file path changed" in (outcome.final_answer or "")
        assert (ws / "b_dir").exists()  # target never deleted
    finally:
        saver.conn.close()


# -------------------------------------------------------------------
# 3. File -> renamed/removed -> refused
# -------------------------------------------------------------------


def test_toctou_file_removed_before_resume_reported_honestly(tmp_path: Path, ws: Path) -> None:
    """A file deleted between interrupt and resume: the TOCTOU guard passes
    (the resolved path is the same, just absent), and the tool itself reports
    'no longer exists' at execution time.  This is correct — the TOCTOU guard
    detects path *substitution* (junctions, symlinks), not mere deletion.
    """
    target = _write(ws, "gone.txt", "vanish")
    manager = _make_manager()
    # max_replans=0 keeps the tool's honest 'no longer exists' report as the
    # final answer (the replan path is covered in test_agent_architecture.py).
    settings = Settings(
        policy=PolicySettings(allowed_roots=[str(ws.resolve())]),
        agent=AgentSettings(max_replans=0),
    )
    ctx = make_app_context(
        settings,
        llm=FakeLLM([_delete_plan([str(target)])]),
        registry=registry_with(make_delete_path_spec()),
        unlock=manager,
        trash=FakeDirTrash(tmp_path / "trash"),
        undo_log=tmp_path / "undo.jsonl",
    )
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        first = run_task(ctx, saver, "delete gone.txt")
        assert first.interrupted is True
        assert manager.verify(PW) is True

        # Delete the file before resume
        os.remove(str(target))

        outcome = resume_task(ctx, saver, first.task_id, approve(first.confirmation))
        # Path resolved the same (just absent), so TOCTOU passes;
        # the tool reports the error honestly.
        assert not target.exists()
        assert "no longer exists" in (outcome.final_answer or "")
    finally:
        saver.conn.close()


# -------------------------------------------------------------------
# 4. Checkpoint/resume preserves the original resolved paths
# -------------------------------------------------------------------


def test_toctou_checkpoint_preserves_original_paths(tmp_path: Path, ws: Path) -> None:
    """After validate runs, gated_resolved_paths is committed to the checkpoint
    and survives across resume."""
    target = _write(ws, "tracked.txt")
    manager = _make_manager()
    ctx = _make_ctx(ws, tmp_path, [_delete_plan([str(target)])], manager)
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        first = run_task(ctx, saver, "delete tracked.txt")
        assert first.interrupted is True

        # Read the gated_resolved_paths from the persisted graph state
        graph = build_graph(ctx, saver)
        config = {"configurable": {"thread_id": first.task_id}}
        snapshot = graph.get_state(config)
        gated = snapshot.values.get("gated_resolved_paths") or {}
        assert "s1" in gated
        assert gated["s1"] == [str(target.resolve(strict=False))]
    finally:
        saver.conn.close()


# -------------------------------------------------------------------
# 5. LLM/user-supplied resolved_paths cannot bypass the comparison
# -------------------------------------------------------------------


def test_toctou_tampered_answer_paths_cannot_bypass(tmp_path: Path, ws: Path) -> None:
    """Even if the user's answer contains manipulated resolved_paths,
    the TOCTOU guard uses the engine-computed paths from state, so the
    comparison cannot be bypassed by tampering with the answer."""
    target = _write(ws, "verify.txt")
    manager = _make_manager()
    ctx = _make_ctx(ws, tmp_path, [_delete_plan([str(target)])], manager)
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        first = run_task(ctx, saver, "delete verify.txt")
        assert first.interrupted is True
        assert manager.verify(PW) is True

        # Approve with correct hash but bogus resolved_paths
        tampered_answer = {
            "approved": True,
            "action_hash": first.confirmation["action_hash"],
            "resolved_paths": ["/nonexistent/fake/path"],
        }
        outcome = resume_task(ctx, saver, first.task_id, tampered_answer)
        # The action succeeds because the TOCTOU check compares
        # engine-computed paths (from state), not answer paths.
        assert outcome.halted_reason is None
        assert target.exists() is False  # deleted successfully
    finally:
        saver.conn.close()


def test_toctou_tampered_answer_paths_on_changed_fs_refused(tmp_path: Path, ws: Path) -> None:
    """When the filesystem changes AND the answer has tampered paths,
    the TOCTOU guard still catches the change because it compares
    engine-computed paths, not answer paths."""
    if os.name != "nt":
        pytest.skip("junctions only on Windows")
    target = _write(ws, "beware.txt")
    (ws / "other_dir").mkdir()
    manager = _make_manager()
    ctx = _make_ctx(ws, tmp_path, [_delete_plan([str(target)])], manager)
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        first = run_task(ctx, saver, "delete beware.txt")
        assert first.interrupted is True
        assert manager.verify(PW) is True

        # Swap file for junction
        os.remove(str(target))
        rc = os.system(f'mklink /J "{ws / "beware.txt"}" "{ws / "other_dir"}"')
        if rc != 0:
            pytest.skip("junction creation unavailable")

        # Approve with correct hash but claim the original path
        tampered_answer = {
            "approved": True,
            "action_hash": first.confirmation["action_hash"],
            "resolved_paths": [str(target.resolve(strict=False))],
        }
        outcome = resume_task(ctx, saver, first.task_id, tampered_answer)
        # Must be refused because the engine re-resolves to the junction target
        assert "Refused: a file path changed" in (outcome.final_answer or "")
    finally:
        saver.conn.close()
