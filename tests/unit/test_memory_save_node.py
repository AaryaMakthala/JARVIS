"""memory_save node: the gate that decides what may become a skill (Phase 8).

Every branch of :func:`jarvis.agent.nodes.memory_save._skill_gate` has a test
here, because this is the only place in the system that can turn a past task
into reusable context.  The invariant under test: **a skill is only ever a
fully verified, non-tainted, secret-free example** - it is never executed, and
it can never lower a tier.
"""

from __future__ import annotations

import pytest

from jarvis.agent.context import make_app_context
from jarvis.agent.nodes.memory_save import _skill_gate, memory_save
from jarvis.agent.state import Plan, Step, StepResult
from jarvis.config import Settings
from jarvis.memory import MemoryRecord, NullMemory, SqliteMemory, open_db


class SpyMemory:
    """Records what ``remember()`` was handed, without a database."""

    def __init__(self) -> None:
        self.records: list[MemoryRecord] = []

    def retrieve(self, query: str, limit: int = 3) -> list[MemoryRecord]:  # pragma: no cover
        del query, limit
        return []

    def remember(self, record: MemoryRecord) -> None:
        self.records.append(record)


class ExplodingMemory(SpyMemory):
    def remember(self, record: MemoryRecord) -> None:
        raise sqlite_error()


def sqlite_error() -> RuntimeError:
    return RuntimeError("database is locked")


@pytest.fixture
def backend() -> SqliteMemory:
    store = SqliteMemory(open_db(":memory:"), settings=Settings())
    try:
        yield store
    finally:
        store.close()


def _ctx(backend=None, **settings_kwargs):
    settings = Settings(**settings_kwargs)
    return make_app_context(settings, memory=backend)


def _step(tool: str = "open_app", untrusted: bool = False) -> Step:
    return Step(
        id="s1",
        tool=tool,
        args={"name": "notepad"},
        rationale="open it",
        expect="a window is open",
        depends_on_untrusted=untrusted,
    )


def _plan(*steps: Step, kind: str = "tool", goal: str = "open notepad") -> Plan:
    return Plan(kind=kind, goal=goal, steps=list(steps) or [_step()])


def _ok(step_id: str = "s1", **kwargs) -> StepResult:
    defaults = {"ok": True, "output": "opened", "verified": True}
    return StepResult(step_id=step_id, **{**defaults, **kwargs})


def _state(plan: Plan | None = None, results: list[StepResult] | None = None, **extra):
    plan = plan if plan is not None else _plan()
    results = results if results is not None else [_ok() for _ in plan.steps]
    return {"plan": plan, "results": results, **extra}


# ------------------------------------------------------------------ happy path


def test_verified_tool_plan_becomes_a_skill(backend: SqliteMemory) -> None:
    out = memory_save(_state(), _ctx(backend))
    assert out["memory_saved"] is True
    assert out["memory_saved_reason"] is None
    assert backend.counts()["skills"] == 1


def test_skill_records_the_goal_and_the_tools(backend: SqliteMemory) -> None:
    memory_save(_state(_plan(_step("open_app"))), _ctx(backend))
    skill = backend.skills.all()[0]
    assert skill.goal_text == "open notepad"
    assert skill.tools_used == ["open_app"]


def test_saving_the_same_goal_twice_reinforces_one_skill(backend: SqliteMemory) -> None:
    memory_save(_state(), _ctx(backend))
    memory_save(_state(), _ctx(backend))
    assert backend.counts()["skills"] == 1
    assert backend.skills.all()[0].success_count == 2


def test_multi_step_plan_is_saved_when_every_step_verifies(backend: SqliteMemory) -> None:
    plan = _plan(
        _step("open_app"),
        Step(id="s2", tool="type_text", args={"text": "hi"}, rationale="type", expect="typed"),
    )
    out = memory_save(_state(plan, [_ok("s1"), _ok("s2")]), _ctx(backend))
    assert out["memory_saved"] is True


# ------------------------------------------------------------------- the gate


def test_no_backend_writes_nothing() -> None:
    out = memory_save(_state(), _ctx(None))
    assert out["memory_saved"] is False
    assert out["memory_saved_reason"] == "no memory backend"


def test_null_memory_accepts_and_discards() -> None:
    out = memory_save(_state(), _ctx(NullMemory()))
    assert out["memory_saved"] is True  # the backend accepted and discarded it
    assert out["memory_failure_logged"] is False
    assert NullMemory().retrieve("anything") == []


def test_dry_run_never_writes(backend: SqliteMemory) -> None:
    ctx = _ctx(backend)
    ctx.dry_run = True
    out = memory_save(_state(), ctx)
    assert out["memory_saved"] is False
    assert "dry-run" in out["memory_saved_reason"]
    assert backend.counts()["skills"] == 0


def test_cancelled_task_is_not_saved(backend: SqliteMemory) -> None:
    out = memory_save(_state(cancelled=True), _ctx(backend))
    assert out["memory_saved"] is False
    assert "cancelled" in out["memory_saved_reason"]
    assert backend.counts()["skills"] == 0


def test_halted_task_is_not_saved(backend: SqliteMemory) -> None:
    out = memory_save(_state(halted_reason="Tier 3 is blocked in code."), _ctx(backend))
    assert "halted" in out["memory_saved_reason"]
    assert backend.counts()["skills"] == 0


def test_conversation_plan_is_not_saved(backend: SqliteMemory) -> None:
    state = _state(_plan(kind="conversation", goal="who won the final"), [])
    out = memory_save(state, _ctx(backend))
    assert "not a tool plan" in out["memory_saved_reason"]
    assert backend.counts()["skills"] == 0


def test_no_results_is_not_saved(backend: SqliteMemory) -> None:
    out = memory_save(_state(_plan(), []), _ctx(backend))
    assert "no step results" in out["memory_saved_reason"]


def test_a_failed_step_is_not_saved(backend: SqliteMemory) -> None:
    out = memory_save(_state(results=[_ok(ok=False, error="boom", verified=None)]), _ctx(backend))
    assert "at least one step failed" in out["memory_saved_reason"]
    assert backend.counts()["skills"] == 0
    assert backend.counts()["failures"] == 1


def test_an_unverifiable_step_is_not_saved_and_not_a_failure(backend: SqliteMemory) -> None:
    """ok + verified=None: succeeded, but no post-condition exists to check.

    Regression (live bug): a successful ``lock_computer`` (the workstation is
    locked, so nothing can be independently verified) was classified as a
    failure, written into the failure memory, and the next "lock my computer"
    was refused with "a previous attempt failed".  Unverifiable is not failed.
    """
    out = memory_save(_state(results=[_ok(verified=None)]), _ctx(backend))
    assert out["memory_saved"] is False  # still not a *skill* (no proof)
    assert out["memory_failure_logged"] is False  # and never a *failure*
    assert "could not be independently verified" in out["memory_saved_reason"]
    assert backend.counts() == {"skills": 0, "failures": 0, "preferences": 0}


def test_a_verification_failure_is_not_saved(backend: SqliteMemory) -> None:
    out = memory_save(_state(results=[_ok(verified=False)]), _ctx(backend))
    assert "failed verification" in out["memory_saved_reason"]


def test_unverified_success_never_blocks_the_same_task_later(
    backend: SqliteMemory,
) -> None:
    """The live scenario end-to-end: lock succeeds unverified; a later task
    with the same goal must not retrieve a "do not repeat that approach"
    hint from memory."""
    plan = _plan(_step("lock_computer"), goal="lock my computer")
    memory_save(_state(plan, [_ok(verified=None)]), _ctx(backend))

    # A second identical run (as the user actually did) is not blocked:
    again = memory_save(_state(plan, [_ok(verified=None)]), _ctx(backend))
    assert again["memory_failure_logged"] is False

    # And the planner is never shown a failure for the successful task.
    records = backend.retrieve("lock my computer", limit=5)
    assert all("failed" not in record.text.lower() for record in records)


def test_tainted_output_is_not_saved(backend: SqliteMemory) -> None:
    out = memory_save(_state(results=[_ok(tainted=True)]), _ctx(backend))
    assert "untrusted" in out["memory_saved_reason"]
    assert backend.counts()["skills"] == 0


def test_untrusted_arguments_are_not_saved(backend: SqliteMemory) -> None:
    out = memory_save(_state(_plan(_step(untrusted=True))), _ctx(backend))
    assert "untrusted" in out["memory_saved_reason"]
    assert backend.counts()["skills"] == 0


def test_partial_results_are_not_saved(backend: SqliteMemory) -> None:
    plan = _plan(_step(), Step(id="s2", tool="type_text", args={}, rationale="t", expect="t"))
    out = memory_save(_state(plan, [_ok("s1")]), _ctx(backend))
    assert "disagree on step count" in out["memory_saved_reason"]
    assert backend.counts()["skills"] == 0


def test_skill_saving_can_be_switched_off(backend: SqliteMemory) -> None:
    ctx = _ctx(backend, memory={"save_skills": False})
    out = memory_save(_state(), ctx)
    assert "disabled by settings" in out["memory_saved_reason"]
    assert backend.counts()["skills"] == 0


# ------------------------------------------------------------- failure memory


def test_a_failed_task_is_recorded_as_a_failure(backend: SqliteMemory) -> None:
    state = _state(results=[_ok(ok=False, error="Access denied: file in use", verified=None)])
    out = memory_save(state, _ctx(backend))
    # A failure is remembered, but it is NOT a skill and the reason says so.
    assert out["memory_saved"] is False
    assert out["memory_failure_logged"] is True
    assert "at least one step failed" in out["memory_saved_reason"]
    assert "recorded as a failure" in out["memory_saved_reason"]
    assert backend.counts() == {"skills": 0, "failures": 1, "preferences": 0}
    assert backend.failures.recent()[0].error == "Access denied: file in use"


def test_a_tainted_task_is_recorded_as_a_failure_but_never_a_skill(
    backend: SqliteMemory,
) -> None:
    state = _state(results=[_ok(tainted=True)])
    out = memory_save(state, _ctx(backend))
    assert out["memory_saved"] is False
    assert out["memory_failure_logged"] is True
    assert backend.counts()["skills"] == 0
    assert backend.counts()["failures"] == 1


def test_a_halted_task_records_nothing(backend: SqliteMemory) -> None:
    """A refusal has no failing *step*, so there is nothing to remember."""
    out = memory_save(_state(halted_reason="refused"), _ctx(backend))
    assert out["memory_saved"] is False
    assert out["memory_failure_logged"] is False
    assert backend.counts() == {"skills": 0, "failures": 0, "preferences": 0}


def test_the_failure_names_the_tool_that_failed(backend: SqliteMemory) -> None:
    state = _state(
        _plan(_step("delete_path")),
        [StepResult(step_id="s1", ok=False, error="denied", verified=None)],
    )
    memory_save(state, _ctx(backend))
    assert backend.failures.recent()[0].step["tool"] == "delete_path"
    assert backend.failures.repeated_tools("open notepad") == ["delete_path"]


def test_ok_false_still_enters_failure_memory(backend: SqliteMemory) -> None:
    """Fail-closed preserved: a genuine ok=False failure is still remembered
    (unchanged behaviour alongside the verified=None correction)."""
    state = _state(
        _plan(_step("lock_computer"), goal="lock my computer"),
        [StepResult(step_id="s1", ok=False, error="access denied", verified=None)],
    )
    out = memory_save(state, _ctx(backend))
    assert out["memory_failure_logged"] is True
    assert backend.counts()["failures"] == 1
    assert backend.failures.repeated_tools("lock my computer") == ["lock_computer"]


def test_verified_false_enters_failure_memory_but_none_does_not(backend: SqliteMemory) -> None:
    """The classification line, side by side: ``False`` is a real failure,
    ``None`` is only an unverifiable success."""
    bad = _state(
        _plan(_step("lock_computer"), goal="lock my computer"),
        [_ok(verified=False)],
    )
    memory_save(bad, _ctx(backend))
    assert backend.counts()["failures"] == 1

    good = _state(
        _plan(_step("lock_computer"), goal="lock my computer"),
        [_ok(verified=None)],
    )
    memory_save(good, _ctx(backend))
    assert backend.counts()["failures"] == 1  # unchanged by the second run


# --------------------------------------------------------------------- safety


def test_secret_argument_values_never_reach_the_store() -> None:
    spy = SpyMemory()
    plan = _plan(
        Step(id="s1", tool="unlock", args={"password": "hunter2"}, rationale="r", expect="e")
    )
    memory_save(_state(plan, [_ok()]), _ctx(spy))
    assert "hunter2" not in str(spy.records[0].model_dump())
    assert spy.records[0].meta["plan"]["steps"][0]["args"]["password"] == "[redacted]"


def test_a_real_credential_is_redacted_and_never_persisted(backend: SqliteMemory) -> None:
    """The node redacts before the store writes, so the secret never lands."""
    plan = _plan(
        Step(id="s1", tool="unlock", args={"password": "hunter2"}, rationale="r", expect="e")
    )
    out = memory_save(_state(plan, [_ok()]), _ctx(backend))
    assert out["memory_saved"] is True
    skill = backend.skills.all()[0]
    assert skill.plan["steps"][0]["args"]["password"] == "[redacted]"
    assert "hunter2" not in str(skill.plan) + skill.goal_text


def test_a_broken_backend_never_breaks_a_completed_task() -> None:
    out = memory_save(_state(), _ctx(ExplodingMemory()))
    assert out["memory_saved"] is False
    assert out["memory_failure_logged"] is False
    assert out["memory_saved_reason"] == "memory write failed"


def test_the_node_returns_only_its_three_state_keys() -> None:
    out = memory_save(_state(), _ctx(SpyMemory()))
    assert set(out) == {"memory_saved", "memory_saved_reason", "memory_failure_logged"}


# -------------------------------------------------------------- gate internals


@pytest.mark.parametrize(
    ("state", "expected_fragment"),
    [
        ({"cancelled": True}, "cancelled"),
        ({"halted_reason": "nope"}, "halted"),
        ({}, "not a tool plan"),
    ],
)
def test_gate_reasons_are_specific(state: dict, expected_fragment: str) -> None:
    assert expected_fragment in _skill_gate(state, _ctx(SpyMemory()))


def test_gate_passes_for_a_clean_plan() -> None:
    assert _skill_gate(_state(), _ctx(SpyMemory())) == ""


def test_gate_rejects_verified_none_for_the_skill_but_for_a_specific_reason() -> None:
    """The gate distinguishes "failed" from "unverifiable" in its reason text,
    even though neither may become a skill."""
    failed = _skill_gate(_state(results=[_ok(verified=False)]), _ctx(SpyMemory()))
    unverifiable = _skill_gate(_state(results=[_ok(verified=None)]), _ctx(SpyMemory()))
    assert "failed verification" in failed
    assert "could not be independently verified" in unverifiable
    assert failed != unverifiable
