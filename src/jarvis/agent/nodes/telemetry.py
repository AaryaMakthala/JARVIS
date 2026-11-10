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

A third, cheap signal rides along: the **node path** of the run.  :func:`wrap`
appends the node's *name* to an in-memory trace keyed by ``task_id``; at
``respond`` the trace is folded into one compact string (``brain x3 > respond``)
and stored in the task-log row's free-form ``tools_used`` column, then dropped.
Only node names are recorded - never user text, tool arguments or results - so
the trace cannot leak anything the state did not already store, and the writer
is bounded twice (200 entries per task, a 600-character path).
"""

from __future__ import annotations

import re
import threading
from collections import OrderedDict
from typing import Any

from jarvis.agent.context import AppContext
from jarvis.agent.state import Plan, StepResult
from jarvis.memory.tasklog import STATUSES
from jarvis.taskstore import TaskEventType, TaskStage, TaskStatus

__all__ = [
    "MAX_PATH_CHARS",
    "MAX_TRACED_TASKS",
    "MAX_TRACE_ENTRIES",
    "NODE_CODES",
    "NODE_STAGES",
    "discard_trace",
    "format_trace",
    "get_trace",
    "node_code",
    "record_finished",
    "record_route",
    "record_stage",
    "record_started",
    "task_status",
    "trace_node",
]

#: Most tasks whose path is held in memory at once (oldest evicted first).
MAX_TRACED_TASKS = 200

#: Most node entries recorded for one task (oldest dropped first).
MAX_TRACE_ENTRIES = 200

#: Longest compact path string written to the task-log field.
MAX_PATH_CHARS = 600

#: A recorded name is a function's ``__name__``; anything else is normalised
#: down to this shape before it is kept.
_NAME_RE = re.compile(r"[^A-Za-z0-9_]")
_MAX_NAME_CHARS = 40

#: Short codes for the stored path.  The `task_log.tools_used` column is
#: truncated at 300 characters by the store, and a long specialist run in full
#: node names (``context_resolve > memory_retrieve > brain > validate >
#: policy_gate > act > verify``) reached that limit, losing the *end* of the
#: path - the part that says how the task terminated.  Codes keep a whole run
#: in ~3 characters per visit.  This is the ONE place the mapping lives; the
#: in-memory trace and :func:`get_trace` keep the full names, so debugging in a
#: running process is unaffected.  A node with no code is stored by its own
#: (sanitised) name rather than dropped.
NODE_CODES: dict[str, str] = {
    "intake": "in",
    "context_resolve": "cr",
    "memory_retrieve": "mr",
    "brain": "br",
    "validate": "va",
    "policy_gate": "pg",
    "act": "ac",
    "verify": "ve",
    "replan": "rp",
    "clarify": "cl",
    "compose": "co",
    "respond": "re",
    "memory_save": "ms",
    "specialist_decide": "sd",
    "specialist_observe": "so",
}

#: Graph node -> closed lifecycle stage written to the independent task store.
#: ``intake`` is absent on purpose: :func:`record_started` opens the task and
#: moves it to ``running`` itself, so the entry hook would write a redundant
#: event.  ``memory_save`` is reached after ``respond`` has already written the
#: terminal event, so its write is expected to be refused and ignored.
NODE_STAGES: dict[str, TaskStage] = {
    "context_resolve": TaskStage.RECALLING,
    "memory_retrieve": TaskStage.RECALLING,
    "brain": TaskStage.PLANNING,
    "validate": TaskStage.VALIDATING,
    "prepare_readback": TaskStage.AUTHORIZING,
    "policy_gate": TaskStage.AUTHORIZING,
    "act": TaskStage.EXECUTING,
    "verify": TaskStage.VERIFYING,
    "replan": TaskStage.REPLANNING,
    "clarify": TaskStage.CLARIFYING,
    "compose": TaskStage.RESPONDING,
    "respond": TaskStage.RESPONDING,
    "memory_save": TaskStage.SAVING,
    "specialist_decide": TaskStage.PLANNING,
    "specialist_observe": TaskStage.VERIFYING,
}

#: Failure code recorded with a terminal status (the stored vocabulary is the
#: closed :class:`~jarvis.agent.state.FailureCode` set).
_FAILURE_CODES: dict[TaskStatus, str] = {
    TaskStatus.FAILED: "failed",
    TaskStatus.PARTIAL: "partial",
    TaskStatus.CANCELLED: "cancelled",
    TaskStatus.HALTED: "halted",
}

_SOURCES = ("terminal", "voice", "benchmark")


class _NodeTraces:
    """Bounded, thread-safe map of ``task_id`` -> recorded node names.

    Held by this module rather than in ``AgentState`` on purpose: the path is
    observability, not graph state, so it needs no checkpoint key and no
    serializer allowlist entry.  It is process-local and dropped as soon as
    the task's row is written.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_task: OrderedDict[str, list[str]] = OrderedDict()

    def append(self, task_id: str, name: str) -> None:
        with self._lock:
            names = self._by_task.get(task_id)
            if names is None:
                names = []
                self._by_task[task_id] = names
                while len(self._by_task) > MAX_TRACED_TASKS:
                    self._by_task.popitem(last=False)
            names.append(name)
            if len(names) > MAX_TRACE_ENTRIES:
                del names[0 : len(names) - MAX_TRACE_ENTRIES]

    def names(self, task_id: str) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._by_task.get(task_id) or ())

    def discard(self, task_id: str) -> None:
        with self._lock:
            self._by_task.pop(task_id, None)


#: The process-wide trace store.  Private: callers use the functions below so
#: the bounded dict itself is never handed out.
_TRACES = _NodeTraces()


def trace_node(task_id: str, name: str) -> None:
    """Record that ``name`` is being entered for ``task_id`` (called by :func:`wrap`).

    Best effort by construction: an empty id or a name with no usable
    characters records nothing, and nothing here can raise into the graph.
    """
    task_id = (task_id or "").strip()
    clean = _NAME_RE.sub("", str(name or ""))[:_MAX_NAME_CHARS]
    if not task_id or not clean:
        return
    _TRACES.append(task_id, clean)


def get_trace(task_id: str) -> tuple[str, ...]:
    """The node names recorded so far for ``task_id`` (oldest first)."""
    return _TRACES.names((task_id or "").strip())


def discard_trace(task_id: str) -> None:
    """Forget ``task_id``'s path (called once its row is written)."""
    _TRACES.discard((task_id or "").strip())


def node_code(name: str) -> str:
    """The short, stored code for a node name (see :data:`NODE_CODES`).

    Unknown nodes keep their sanitised name: a new node must still show up in
    the path, just longer than three characters.
    """
    label = _NAME_RE.sub("", str(name or ""))[:_MAX_NAME_CHARS]
    return NODE_CODES.get(label, label)


def format_trace(names: tuple[str, ...] | list[str]) -> str:
    """One compact, bounded string for the recorded node names.

    Consecutive repeats collapse to ``ac x3`` (a routed loop such as
    ``policy_gate > act > verify`` reads as one pass, not as noise) and each
    name becomes its short :data:`NODE_CODES` code, joined by ``>`` without
    spaces, so the whole run fits the task-log column.  The result is still cut
    at a separator boundary so it never ends mid-code.
    """
    runs: list[tuple[str, int]] = []
    for name in names:
        label = str(name or "")
        if not label:
            continue
        if runs and runs[-1][0] == label:
            runs[-1] = (label, runs[-1][1] + 1)
        else:
            runs.append((label, 1))
    path = ">".join(
        f"{node_code(label)} x{count}" if count > 1 else node_code(label)
        for label, count in runs
    )
    if len(path) <= MAX_PATH_CHARS:
        return path
    cut = path.rfind(">", 0, MAX_PATH_CHARS - 4)
    if cut <= 0:
        return path[: MAX_PATH_CHARS - 1] + "…"
    return path[: cut + 1] + "…"


def _task_store(ctx: AppContext) -> Any | None:
    """The independent store, or ``None`` when observability is off."""
    if bool(getattr(ctx, "dry_run", False)):
        return None
    store = getattr(ctx, "tasks", None)
    if store is None or not getattr(store, "writable", False):
        return None
    return store


def _safe_source(state: dict[str, Any]) -> str:
    source = str(state.get("source") or "")
    return source if source in _SOURCES else "terminal"


def record_route(
    state: dict[str, Any],
    ctx: AppContext,
    *,
    task_class: str,
    model_role: str,
    reason_code: str,
) -> None:
    """Record the closed routing decision, never the user text behind it.

    The task store validates every value against its own closed vocabularies.
    Like the other telemetry writers, a store failure is swallowed: an
    unreadable history must not change the task's answer.
    """
    store = _task_store(ctx)
    task_id = str(state.get("task_id") or "")
    if store is None or not task_id:
        return
    try:
        store.ensure_task(task_id, source=_safe_source(state))
        store.append_event(
            task_id,
            TaskEventType.ROUTE,
            detail={
                "task_class": task_class,
                "model_role": model_role,
                "reason_code": reason_code,
            },
        )
    except Exception as exc:  # noqa: BLE001 - telemetry must never break a task
        ctx.logger.debug("task route not recorded: %s", exc)


def record_stage(state: dict[str, Any], ctx: AppContext, node_name: str) -> None:
    """Record that ``node_name`` was entered for this task.

    Called from :func:`jarvis.agent.nodes.wrap`, the single choke point every
    node passes through.  It writes only the closed stage vocabulary and the
    task's own id - no user text, arguments or results.  Every store error is
    swallowed: observability must never change the task's answer.
    """
    store = _task_store(ctx)
    task_id = str(state.get("task_id") or "")
    if store is None or not task_id:
        return
    stage = NODE_STAGES.get(node_name)
    if stage is None:
        return
    try:
        store.ensure_task(task_id, source=_safe_source(state))
        store.update_stage(task_id, stage)
    except Exception as exc:  # noqa: BLE001 - telemetry must never break a task
        ctx.logger.debug("task stage not recorded (%s): %s", node_name, exc)


def _task_begin(state: dict[str, Any], ctx: AppContext) -> None:
    """Open the task and move it to ``running`` (from ``intake``)."""
    store = _task_store(ctx)
    task_id = str(state.get("task_id") or "")
    if store is None or not task_id:
        return
    try:
        store.ensure_task(task_id, source=_safe_source(state))
        store.update_status(task_id, TaskStatus.RUNNING)
    except Exception as exc:  # noqa: BLE001 - telemetry must never break a task
        ctx.logger.debug("task start not recorded: %s", exc)


def _task_finish(state: dict[str, Any], ctx: AppContext, status: str) -> None:
    """Write the terminal event with the derived status and failure code."""
    store = _task_store(ctx)
    task_id = str(state.get("task_id") or "")
    if store is None or not task_id:
        return
    try:
        final = TaskStatus(status)
        code = _FAILURE_CODES.get(final)
        replans = int(state.get("replan_count") or 0)
        store.ensure_task(task_id, source=_safe_source(state))
        if replans > 0:
            store.append_event(
                task_id,
                TaskEventType.NOTE,
                detail={"event_code": "replan", "retry_count": replans},
            )
        store.finish(task_id, final, failure_code=code)
    except Exception as exc:  # noqa: BLE001 - telemetry must never break a task
        ctx.logger.debug("task finish not recorded: %s", exc)


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
    _task_begin(state, ctx)


def record_finished(state: dict[str, Any], ctx: AppContext) -> None:
    """Close the row with the task's final counters and status (``respond``).

    Also writes the run's node path (see :func:`format_trace`, short node
    codes so the whole run fits the column) into the row's free-form
    ``tools_used`` field and then frees the in-memory trace, whether or not the
    write succeeded - a finished task must not leak memory.
    """
    plan: Plan | None = state.get("plan")
    task_id = str(state.get("task_id") or "")
    path = format_trace(get_trace(task_id)) if task_id else ""
    status = task_status(state)
    try:
        _write(
            state,
            ctx,
            status=status,
            api_calls=int(state.get("api_calls") or 0),
            tokens=int(state.get("tokens") or 0),
            steps=len(plan.steps) if plan is not None else 0,
            replans=int(state.get("replan_count") or 0),
            tools_used=path or None,
        )
    finally:
        if task_id:
            discard_trace(task_id)
    _task_finish(state, ctx, status)


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
    tools_used: str | None = None,
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
    extra = {"tools_used": tools_used} if tools_used else {}
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
            **extra,
        )
    except Exception as exc:  # noqa: BLE001 - telemetry must never break a task
        ctx.logger.warning("task telemetry not recorded: %s", exc)
        ctx.logger.debug("unusable task status %r (valid: %s)", status, ", ".join(STATUSES))
