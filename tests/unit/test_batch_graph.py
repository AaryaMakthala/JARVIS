"""Graph-level B1: one batch interrupt for the Tier 1 steps, then one interrupt
each for the Tier 2 and runtime-derived steps.

The unit tests in ``test_policy_gate_batch.py`` pin ``batch_approval`` as a pure
function.  These tests drive the **real compiled graph** over a file-backed
SQLite checkpointer and assert the *interrupt sequence* a user actually
experiences, because the batch decision is made from ``state["decisions"]``,
which the gate fills one step at a time.

Plan (one brain decision, four steps, in this order):

* ``s1`` ``create_file`` with literal args  -> Tier 1, batch-eligible
* ``s2`` ``create_file`` with literal args  -> Tier 1, batch-eligible
* ``s3`` ``delete_path``                    -> Tier 2, needs unlock (B1: never batched)
* ``s4`` ``create_file`` whose content is a runtime reference (``{{...}}``) ->
  ``resolved_from_runtime`` (B1: gated individually)

Everything is sandboxed under ``tmp_path``: real ``create_file`` /
``delete_path`` tools, a fake Recycle Bin, and an injected undo log.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jarvis.agent.batch_approval import TYPE_PLAN_APPROVAL
from jarvis.agent.context import make_app_context
from jarvis.agent.graph import open_sqlite_checkpointer
from jarvis.agent.runner import resume_task, run_task
from jarvis.agent.schemas import ActionIntent
from jarvis.config import PolicySettings, Settings
from jarvis.llm.client import FakeLLM
from jarvis.policy.unlock import UnlockManager
from jarvis.tools.files import make_create_file_spec, make_delete_path_spec
from support import FakeDirTrash, approve, brain_steps, registry_with

PW = "correct-horse-battery-staple"

#: A runtime reference in the args; ``runtime_args.mark_runtime_args`` turns this
#: into ``resolved_from_runtime=True`` inside ``validate``.
RUNTIME_CONTENT = "{{output of s1}}"


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
    """An already-unlocked session, so Tier 2 refusals are never about locking."""
    manager = UnlockManager(_PwStore())
    manager.set_password(PW)
    assert manager.verify(PW) is True
    return manager


def _settings(ws: Path) -> Settings:
    return Settings(policy=PolicySettings(allowed_roots=[str(ws.resolve())]))


def _mixed_plan(ws: Path) -> Any:
    """s1/s2 literal Tier 1, s3 Tier 2, s4 runtime-derived."""
    return brain_steps(
        [
            ActionIntent(
                tool="create_file",
                args={"path": str(ws / "a.txt"), "content": "A"},
                rationale="first literal file",
            ),
            ActionIntent(
                tool="create_file",
                args={"path": str(ws / "b.txt"), "content": "B"},
                rationale="second literal file",
            ),
            ActionIntent(
                tool="delete_path",
                args={"paths": [str(ws / "c.txt")]},
                rationale="tier 2 delete",
            ),
            ActionIntent(
                tool="create_file",
                args={"path": str(ws / "d.txt"), "content": RUNTIME_CONTENT},
                rationale="runtime-derived file",
            ),
        ]
    )


def _ctx(tmp_path: Path, ws: Path) -> Any:
    return make_app_context(
        _settings(ws),
        llm=FakeLLM([_mixed_plan(ws)]),
        registry=registry_with(make_create_file_spec(), make_delete_path_spec()),
        unlock=_unlocked_manager(),
        trash=FakeDirTrash(tmp_path / "trash"),
        undo_log=tmp_path / "undo.jsonl",
    )


def _approved(outcome: Any) -> list[str]:
    return [str(h) for h in (outcome.state.get("approved_hashes") or [])]


def _batch_answer(batch: dict[str, Any]) -> dict[str, Any]:
    """The batch approval, replayed verbatim as a resume answer."""
    return {"approved": True, "action_hash": batch["action_hash"], "resolved_paths": []}


# ── the interrupt sequence a user actually sees ───────────────────────────


def test_mixed_plan_batch_then_individual_interrupts(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "c.txt").write_text("C", encoding="utf-8")
    ctx = _ctx(tmp_path, ws)
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        # (a) exactly ONE plan_approval interrupt, covering only s1 + s2.
        first = run_task(ctx, saver, "create two files, delete one, create one more")
        assert first.interrupted is True, "the mixed plan must stop for a confirmation"
        batch = first.confirmation or {}
        assert batch.get("type") == TYPE_PLAN_APPROVAL, (
            f"expected one plan_approval interrupt for the batch, got {batch!r}"
        )
        assert [step_id for step_id, _ in batch.get("eligible") or []] == ["s1", "s2"], (
            f"the batch must name exactly s1+s2, got {batch.get('eligible')!r}"
        )
        batch_hash = str(batch["action_hash"])

        # (b) approving it runs s1 and s2, then (c) s3 interrupts on its own.
        second = resume_task(ctx, saver, first.task_id, approve(batch))
        assert (ws / "a.txt").read_text(encoding="utf-8") == "A"
        assert (ws / "b.txt").read_text(encoding="utf-8") == "B"
        assert second.interrupted is True, (
            f"s3 must raise its own interrupt, got {second.final_answer!r}"
        )
        s3 = second.confirmation or {}
        assert s3.get("type") == "confirm"
        assert s3.get("step_id") == "s3"
        assert s3.get("tier") == 2
        assert s3["action_hash"] != batch_hash, "s3 must not ride on the batch hash"
        assert s3["action_hash"] == second.decisions["s3"].action_hash
        assert (ws / "c.txt").exists(), "nothing may be deleted before s3 is confirmed"

        # (d) s4 is gated individually too, after s3 is properly confirmed.
        third = resume_task(ctx, saver, first.task_id, approve(s3))
        assert (ws / "c.txt").exists() is False, "the confirmed Tier 2 step must run"
        assert third.interrupted is True, (
            f"s4 must raise its own interrupt, got {third.final_answer!r}"
        )
        s4 = third.confirmation or {}
        assert s4.get("type") == "confirm"
        assert s4.get("step_id") == "s4"
        assert s4["action_hash"] != batch_hash, "s4 must not ride on the batch hash"
        assert (ws / "d.txt").exists() is False

        # (e) replaying the batch approval at s4's gate must not approve s4.
        fourth = resume_task(ctx, saver, first.task_id, _batch_answer(batch))
        assert fourth.interrupted is False
        assert (ws / "d.txt").exists() is False, "s4 ran on a stale batch approval"
        assert s4["action_hash"] not in _approved(fourth), (
            "the batch approval must never enter s4's action hash"
        )
    finally:
        saver.conn.close()


# ── B1: the batch never covers Tier 2 or a runtime-derived step ───────────


def test_batch_approval_does_not_approve_tier2_or_runtime_step(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "c.txt").write_text("C", encoding="utf-8")
    ctx = _ctx(tmp_path, ws)
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        first = run_task(ctx, saver, "create two files, delete one, create one more")
        batch = first.confirmation or {}
        assert batch.get("type") == TYPE_PLAN_APPROVAL, (
            f"expected a plan_approval interrupt, got {batch!r}"
        )
        eligible = list(batch.get("eligible") or [])
        assert [step_id for step_id, _ in eligible] == ["s1", "s2"], (
            f"only the two literal Tier 1 steps may be batched, got {eligible!r}"
        )
        batch_hash = str(batch["action_hash"])

        second = resume_task(ctx, saver, first.task_id, approve(batch))
        s3_hash = second.decisions["s3"].action_hash
        s4_hash = second.decisions["s4"].action_hash
        assert s3_hash != batch_hash
        assert s4_hash != batch_hash

        approved = _approved(second)
        assert sorted(approved) == sorted(action_hash for _, action_hash in eligible), (
            f"the batch must approve exactly its own two hashes, got {approved!r}"
        )
        assert s3_hash not in approved, "a Tier 2 hash must never come from the batch"
        assert s4_hash not in approved, "a runtime-derived hash must never come from the batch"
        assert (ws / "c.txt").exists(), "s3 must not have run"
        assert (ws / "d.txt").exists() is False, "s4 must not have run"

        # Whatever the next interrupt is, answering it with the batch hash must
        # not execute s3 or s4.
        assert second.interrupted is True, (
            f"expected another interrupt, got {second.final_answer!r}"
        )
        third = resume_task(ctx, saver, first.task_id, _batch_answer(batch))
        assert third.interrupted is False
        assert (ws / "c.txt").exists(), "s3 ran on the batch approval"
        assert (ws / "d.txt").exists() is False, "s4 ran on the batch approval"
        assert s3_hash not in _approved(third)
        assert s4_hash not in _approved(third)
    finally:
        saver.conn.close()


# ── contiguity: a Tier 2 step in the middle breaks the run ────────────────


def test_non_contiguous_eligible_not_batched(tmp_path: Path) -> None:
    """s1 Tier 1, s2 Tier 2, s3 Tier 1: only s1 is contiguous, so s1 gets its own
    single-step interrupt and s3 is never batched across the Tier 2 step."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "nc2.txt").write_text("X", encoding="utf-8")
    plan = brain_steps(
        [
            ActionIntent(
                tool="create_file",
                args={"path": str(ws / "nc1.txt"), "content": "1"},
                rationale="first literal file",
            ),
            ActionIntent(
                tool="delete_path",
                args={"paths": [str(ws / "nc2.txt")]},
                rationale="tier 2 in the middle",
            ),
            ActionIntent(
                tool="create_file",
                args={"path": str(ws / "nc3.txt"), "content": "3"},
                rationale="third literal file",
            ),
        ]
    )
    ctx = make_app_context(
        _settings(ws),
        llm=FakeLLM([plan]),
        registry=registry_with(make_create_file_spec(), make_delete_path_spec()),
        unlock=_unlocked_manager(),
        trash=FakeDirTrash(tmp_path / "trash"),
        undo_log=tmp_path / "undo.jsonl",
    )
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    try:
        first = run_task(ctx, saver, "create one, delete one, create one")
        payload = first.confirmation or {}
        assert payload.get("type") == "confirm", (
            f"a run of 1 eligible step must not be batched, got {payload!r}"
        )
        assert payload.get("step_id") == "s1"
        assert payload.get("tier") == 1
        assert "eligible" not in payload, "a single-step payload never names a batch"

        second = resume_task(ctx, saver, first.task_id, approve(payload))
        assert (ws / "nc1.txt").read_text(encoding="utf-8") == "1"
        s2_payload = second.confirmation or {}
        assert second.interrupted is True, f"expected s2's interrupt, got {second.final_answer!r}"
        assert s2_payload.get("type") == "confirm"
        assert s2_payload.get("step_id") == "s2", (
            f"s3 must never be batched across the Tier 2 step, got {s2_payload!r}"
        )
        assert s2_payload.get("tier") == 2

        third = resume_task(ctx, saver, first.task_id, approve(s2_payload))
        assert (ws / "nc2.txt").exists() is False
        s3_payload = third.confirmation or {}
        assert third.interrupted is True, f"expected s3's interrupt, got {third.final_answer!r}"
        assert s3_payload.get("type") == "confirm"
        assert s3_payload.get("step_id") == "s3"
    finally:
        saver.conn.close()
