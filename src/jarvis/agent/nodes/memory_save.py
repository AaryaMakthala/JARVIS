"""memory_save node: record what worked (and what did not) — Phase 8.

This is the **only** node that writes to memory, and it runs after ``respond``
so every terminal path (success, refusal, cancel, failure) passes through it
exactly once.  It is deliberately dumb and deterministic: no LLM, no policy
call, no tool execution.  It cannot approve anything — a stored skill is only
ever *text handed to the planner*, which must still produce a new plan that
goes through ``validate`` and ``policy_gate``.

The skill gate (docs/05_BUILD_PLAN.md, Phase 8: "only successful, fully
verified, non-tainted plans are saved") is :func:`_skill_gate` and every one of
its checks has a test in ``tests/unit/test_memory_save_node.py``:

============================  =============================================
Condition                     Why it blocks a save
============================  =============================================
no memory backend             nothing to write to (``NullMemory``)
``--dry-run``                 a rehearsal must not create side effects
``cancelled`` / halted        the human stopped; nothing was proven
not a tool plan               a conversation is not an executable skill
any result not ``ok``         the task did not succeed
any ``verified is False``     the tool's own check said it did not hold
any tainted result            output came from web/file/window text
any ``depends_on_untrusted``  the args were built from untrusted text
============================  =============================================

``verified is None`` ("succeeded, but there is no post-condition to check")
is a *third* state, not a failure: the step executed successfully and only the
independent proof is missing.  Recording it as a failure poisoned the failure
memory — a later "lock my computer" was refused with "a previous attempt
failed" although the lock had succeeded — so it is excluded here and in
:func:`_failure_record`.

Failures are still worth recording, and that is the *other* half of this node:
a failed or verification-failed step is appended to the failure log so a later
replan sees "this approach already failed here".  A tainted step is still
recorded as a failure (the failure text is redacted) but never becomes a skill.
"""

from __future__ import annotations

import logging
from typing import Any

from jarvis.agent.context import AppContext
from jarvis.agent.state import Plan, StepResult
from jarvis.memory.base import MemoryRecord
from jarvis.memory.db import safe_value

__all__ = ["memory_save"]

_LOG = logging.getLogger("memory")

#: ``meta`` key that marks a record as a *failure* rather than a skill, so the
#: backend routes it to the failure log (shared with
#: :class:`jarvis.memory.store.SqliteMemory`).
_FAILURE_META = "failure_id"


def memory_save(state: dict[str, Any], ctx: AppContext) -> dict[str, Any]:
    """Persist one bounded, secret-free memory record for this task.

    Returns three state keys so a caller can report honestly:

    * ``memory_saved`` - a **skill** was stored (a failure record does not
      count: learning from a failure is not the same as learning a skill);
    * ``memory_saved_reason`` - why not, in words, or ``None``;
    * ``memory_failure_logged`` - a failure was appended to the log instead.
    """
    outcome = _outcome(state, ctx)
    if outcome.reason:
        ctx.logger.debug("memory: %s", outcome.reason)
    return {
        "memory_saved": outcome.record is not None and not outcome.is_failure,
        "memory_saved_reason": outcome.reason or None,
        "memory_failure_logged": outcome.is_failure,
    }


# --------------------------------------------------------------------- gate


class _Outcome:
    """What :func:`memory_save` decided, and why."""

    __slots__ = ("is_failure", "reason", "record")

    def __init__(
        self, reason: str = "", record: MemoryRecord | None = None, *, is_failure: bool = False
    ) -> None:
        self.reason = reason
        self.record = record
        self.is_failure = is_failure


def _outcome(state: dict[str, Any], ctx: AppContext) -> _Outcome:
    """Decide the single record (if any) this task contributes to memory."""
    backend = ctx.memory
    if backend is None:
        return _Outcome("no memory backend")
    if ctx.dry_run:
        return _Outcome("dry-run: nothing is written")

    blocked = _skill_gate(state, ctx)
    if not blocked:
        return _write(backend, _skill_record(state), "")

    failure = _failure_record(state)
    if failure is None:
        return _Outcome(blocked)
    return _write(backend, failure, f"{blocked}; recorded as a failure", is_failure=True)


def _skill_gate(state: dict[str, Any], ctx: AppContext) -> str:
    """Return the reason this task must **not** become a skill ("" if it may)."""
    if state.get("cancelled"):
        return "cancelled by the user"
    if state.get("halted_reason"):
        return f"task halted: {state['halted_reason']}"

    plan: Plan | None = state.get("plan")
    if plan is None or plan.kind != "tool" or not plan.steps:
        return "not a tool plan"

    results: list[StepResult] = list(state.get("results") or [])
    if not results:
        return "no step results"
    if len(results) != len(plan.steps):
        return "plan and results disagree on step count"
    if any(not result.ok for result in results):
        return "at least one step failed"
    if any(result.verified is False for result in results):
        return "at least one step failed verification"
    if any(result.verified is None for result in results):
        return "a step's completion could not be independently verified"
    if any(result.tainted for result in results):
        return "a step returned untrusted external text"
    if any(step.depends_on_untrusted for step in plan.steps):
        return "a step's arguments came from untrusted text"

    if not bool(getattr(ctx.settings.memory, "save_skills", True)):
        return "skill saving disabled by settings"
    return ""


# ------------------------------------------------------------------ records


def _skill_record(state: dict[str, Any]) -> MemoryRecord:
    """The verified plan to remember (goal text + tool names, args redacted)."""
    plan: Plan = state["plan"]
    return MemoryRecord(
        kind="episodic",
        text=plan.goal.strip() or "an unnamed task",
        meta={
            "goal_text": plan.goal,
            "tools": [step.tool for step in plan.steps],
            "plan": _safe_plan(plan),
        },
    )


def _failure_record(state: dict[str, Any]) -> MemoryRecord | None:
    """The first bad step to remember, if the task did not fully succeed.

    "Bad" means failed, failed-verification **or tainted**: a step whose
    output was untrusted external text is exactly the approach a later run
    should not repeat, even though it technically succeeded.  A step with
    ``verified=None`` is *not* bad — "succeeded, unverifiable" is not
    "failed" — and must never enter the failure log (that poisoned it and
    blocked later successful runs of the same task).  A tainted result is
    recorded with its output text dropped, since that text is the untrusted
    part.
    """
    results: list[StepResult] = list(state.get("results") or [])
    bad = next(
        (r for r in results if not r.ok or r.verified is False or r.tainted),
        None,
    )
    if bad is None:
        return None
    plan: Plan | None = state.get("plan")
    step = next((s for s in (plan.steps if plan else []) if s.id == bad.step_id), None)
    goal = plan.goal if plan else ""
    return MemoryRecord(
        kind="episodic",
        text=(bad.error or f"step {bad.step_id} did not complete")[:300],
        meta={
            _FAILURE_META: True,  # routes this record to the failure log
            "goal_text": goal,
            "step": {
                "tool": step.tool if step else "",
                "id": bad.step_id,
                "tainted": bad.tainted,
            },
        },
    )


def _safe_plan(plan: Plan) -> dict:
    """Plan as a plain dict with credential-looking argument values removed.

    Uses the same :func:`jarvis.memory.db.safe_value` the store uses, so a
    secret nested inside a list argument (``{"paths": [{"password": ...}]}``)
    is stripped here too, not just at the top level of ``args``.
    """
    steps = []
    for step in plan.steps:
        args = {key: safe_value(key, value) for key, value in step.args.items()}
        steps.append({"id": step.id, "tool": step.tool, "args": args, "expect": step.expect})
    return {"goal": plan.goal, "kind": plan.kind, "steps": steps}


def _write(
    backend: Any,
    record: MemoryRecord | None,
    reason: str,
    *,
    is_failure: bool = False,
) -> _Outcome:
    """Hand the record to the backend; a memory error never fails the task."""
    if record is None:
        return _Outcome(reason or "nothing to record")
    try:
        backend.remember(record)
    except Exception as exc:  # noqa: BLE001 - memory must never break a task
        _LOG.warning("memory write failed: %s", exc)
        return _Outcome("memory write failed")
    return _Outcome(reason, record=record, is_failure=is_failure)
