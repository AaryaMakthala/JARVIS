"""Per-task telemetry for the graph (docs/01 "Observability").

The ``brain`` and ``replan`` nodes already accumulate ``api_calls`` and
``tokens`` in ``AgentState``; without this module those numbers die with the
process, and the two acceptance criteria that need them - Phase 8's "API calls
per task drop or stay equal (measured)" and the Phase 9 benchmark's
calls/task and tokens/task metrics - cannot be evaluated at all.

Two writes bracket a task:

* :func:`record_started` from ``intake`` - opens the row.  A task that then
  suspends on a confirmation, or dies with the daemon, is therefore visible as
  ``running`` rather than silently absent;
* :func:`record_finished` from ``respond`` - closes the same row.  ``respond``
  is the documented writer (``docs/02_ARCHITECTURE.md`` section 5) and is the
  only node every terminal path reaches, so the counters are final by then.

Two rules govern this module:

**The status is derived in code, never by the LLM.**  :func:`task_status`
reads the same state fields ``respond`` reads to build its own honest message,
so a task that is reported to the user as "Could not complete" can never be
logged as a success.  The vocabulary is closed
(:data:`~jarvis.memory.tasklog.STATUSES`).

**Nothing here may raise.**  Telemetry is not the product; a locked or corrupt
``memory.db`` must degrade to "no metrics" and leave the task's actual answer
untouched.  ``--dry-run`` writes nothing at all, exactly like the other nodes.
"""

from __future__ import annotations

from typing import Any

from jarvis.agent.context import AppContext
from jarvis.agent.state import Plan, StepResult
from jarvis.memory.tasklog import STATUSES

__all__ = ["record_finished", "record_started", "task_status"]


def record_started(state: dict[str, Any], ctx: AppContext) -> None:
    """Open the ``running`` row for this task (called from ``intake``)."""
    _write(
        state,
        ctx,
        status="running",
        api_calls=0,
        tokens=0,
        steps=0,
        replans=0,
    )


def record_finished(state: dict[str, Any], ctx: AppContext) -> None:
    """Close the row with the task's final counters and status (``respond``)."""
    plan: Plan | None = state.get("plan")
    _write(
        state,
        ctx,
        status=task_status(state),
        api_calls=int(state.get("api_calls") or 0),
        tokens=int(state.get("tokens") or 0),
        steps=len(plan.steps) if plan is not None else 0,
        replans=int(state.get("replan_count") or 0),
    )


def task_status(state: dict[str, Any]) -> str:
    """The closed, honest status for a finished task.

    A recorded ``StepResult`` is concrete evidence, so it is classified before
    any ``halted_reason``: a task that failed a step and then ran out of replan
    budget is logged as the failure it was, never as the halt that followed.
    That ordering is what makes "logged as completed" mean *every* step
    succeeded and verified.

    ==========================  ==========  ==================================
    Condition                   Status      Why
    ==========================  ==========  ==================================
    ``cancelled``               cancelled   the human stopped it
    a failed/unverified step    failed      the work did not happen
    any unproven step           partial     it worked but nothing proved it
    clarification requested     halted      asked, not answered
    a conversation answer       answered    an answer, not an action
    ``halted_reason``           halted      refused / unsupported / no LLM
    everything else             completed   the plan ran and verified
    ==========================  ==========  ==================================
    """
    if state.get("cancelled"):
        return "cancelled"

    results: list[StepResult] = [
        r for r in (state.get("results") or []) if isinstance(r, StepResult)
    ]
    if any(not r.ok or r.verified is False for r in results):
        return "failed"
    if any(r.verified is None for r in results):
        return "partial"

    plan: Plan | None = state.get("plan")
    if plan is not None and plan.needs_clarification:
        return "halted"
    if state.get("final_answer") or (
        plan is not None and getattr(plan, "kind", None) == "conversation" and not plan.steps
    ):
        return "answered"
    if state.get("halted_reason"):
        return "halted"
    return "completed"


def _write(
    state: dict[str, Any],
    ctx: AppContext,
    *,
    status: str,
    api_calls: int,
    tokens: int,
    steps: int,
    replans: int,
) -> None:
    """Hand the row to the memory backend, tolerating every backend shape.

    A missing backend, a ``NullMemory``, a read-only file, a locked database
    and a test double written before this method existed must all end the same
    way: no row, no error, same answer to the user.  ``--dry-run`` is the same
    case: a rehearsal creates no side effects at all, telemetry included.
    """
    backend = getattr(ctx, "memory", None)
    sink = getattr(backend, "record_task", None)
    if not callable(sink):
        return
    task_id = str(state.get("task_id") or "")
    if not task_id or bool(getattr(ctx, "dry_run", False)):
        return
    try:
        sink(
            task_id,
            status=status,
            source=str(state.get("source") or ""),
            user_input=str(state.get("user_input") or ""),
            api_calls=api_calls,
            tokens=tokens,
            steps=steps,
            replans=replans,
        )
    except Exception as exc:  # noqa: BLE001 - telemetry must never break a task
        ctx.logger.warning("task telemetry not recorded: %s", exc)
        ctx.logger.debug("unusable task status %r (valid: %s)", status, ", ".join(STATUSES))
