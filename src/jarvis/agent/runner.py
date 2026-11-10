"""Synchronous task runner (Phase 1).

Owns the run/resume lifecycle: build the graph on a shared SqliteSaver, run a
task in its own thread, capture the pending confirmation, and resume it with
the user's answer.  No daemon/IPC yet (Phase 3); this is what the CLI drives.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from langgraph.types import Command

from jarvis.agent.context import AppContext
from jarvis.agent.graph import build_graph
from jarvis.agent.schemas import AgentResponse
from jarvis.agent.state import Decision, StepResult
from jarvis.taskstore.models import TERMINAL_STATUSES, TaskStatus, TaskStoreError
from jarvis.tools.base import CancelToken

__all__ = ["TaskOutcome", "extract_interrupt", "resume_task", "run_task"]


@dataclass
class TaskOutcome:
    """The externally visible result of a ``run_task`` / ``resume_task`` call."""

    task_id: str
    interrupted: bool
    confirmation: dict[str, Any] | None
    final_answer: str | None
    error: str | None
    halted_reason: str | None
    decisions: dict[str, Decision]
    results: list[StepResult]
    #: Written by the ``compose`` node.  ``screen_answer`` is what the terminal
    #: shows, ``spoken_answer`` what the voice channel says; both are ``None``
    #: when compose is disabled or fell back, and every reader then uses
    #: :attr:`final_answer` as before.
    spoken_answer: str | None = None
    screen_answer: str | None = None
    state: dict[str, Any] = field(default_factory=dict)

    @property
    def interrupt_kind(self) -> str:
        """``"confirm"``, ``"clarification"`` or ``""`` for this outcome.

        Read from the pending interrupt payload's ``type`` field; no payload
        means no interruption happened.
        """
        if not self.confirmation:
            return ""
        kind = self.confirmation.get("type")
        return kind if isinstance(kind, str) and kind in ("confirm", "clarification") else ""

    @property
    def agent_response(self) -> AgentResponse:
        """The typed boundary response for this outcome.

        Kind is derived from the brain's classification (conversation vs.
        tool) and falls back to ``text`` being either the final answer or an
        honest pending-confirmation / pending-clarification / error message.
        """
        plan = self.state.get("plan")
        kind = "conversation" if getattr(plan, "kind", None) == "conversation" else "tool"
        if self.final_answer:
            text = self.final_answer
        elif self.interrupt_kind == "clarification":
            text = f"Clarification needed: {self.confirmation.get('question') or '...'}"
        elif self.confirmation:
            text = f"Pending confirmation: {self.confirmation.get('summary') or '...'}"
        else:
            text = self.error or self.halted_reason or "No answer."
        return AgentResponse(
            task_id=self.task_id,
            kind=kind,
            interrupted=self.interrupted,
            text=text,
        )


def run_task(
    ctx: AppContext,
    checkpointer: Any,
    user_input: str,
    *,
    source: str = "terminal",
    thread_id: str | None = None,
    cancel: CancelToken | None = None,
    conversation_id: str | None = None,
) -> TaskOutcome:
    """Start a new task; returns a TaskOutcome, possibly awaiting confirmation.

    ``cancel`` is the caller's per-task :class:`CancelToken`; the graph checks it
    before entering every node, so a cancelled run stops at the next step.
    ``conversation_id`` (B5) is the stable identity of the conversation this
    request belongs to; it is stored in state (and therefore the checkpoint)
    so a resumed task keeps it, while the *turns* themselves stay in the
    caller's in-memory :class:`~jarvis.agent.conversation.ConversationManager`.
    """
    task_id = thread_id or str(uuid4())
    graph = build_graph(ctx, checkpointer, cancel)
    config = {"configurable": {"thread_id": task_id}}
    # Seed ``task_id`` so the graph and the outcome agree on one identifier.
    # Left to itself ``intake`` would mint a second, unrelated id, and the
    # per-task ``task_log`` row (and every ``agent event=... task_id=`` log
    # line) would be keyed on an id the caller never sees.
    values: dict[str, Any] = {
        "user_input": user_input,
        "source": source,
        "task_id": task_id,
    }
    if conversation_id:
        values["conversation_id"] = conversation_id
    values = graph.invoke(values, config)
    outcome = _outcome(task_id, config, checkpointer, graph, values)
    _mark_paused(ctx, outcome)
    return outcome


def resume_task(
    ctx: AppContext,
    checkpointer: Any,
    thread_id: str,
    answer: Any,
    *,
    cancel: CancelToken | None = None,
) -> TaskOutcome:
    """Resume a task at its interruption point with the user's answer."""
    config = {"configurable": {"thread_id": thread_id}}
    refusal = _resume_refusal(ctx, checkpointer, config)
    if refusal is not None:
        return TaskOutcome(
            task_id=thread_id,
            interrupted=False,
            confirmation=None,
            final_answer=None,
            error=refusal,
            halted_reason=refusal,
            decisions={},
            results=[],
        )
    graph = build_graph(ctx, checkpointer, cancel)
    _mark_running(ctx, thread_id)
    values = graph.invoke(Command(resume=answer), config)
    outcome = _outcome(thread_id, config, checkpointer, graph, values)
    _mark_paused(ctx, outcome)
    return outcome


def _mark_running(ctx: AppContext, task_id: str) -> None:
    """Record that a paused task is running again (best-effort, B7)."""
    _set_status(ctx, task_id, TaskStatus.RUNNING)


def _mark_paused(ctx: AppContext, outcome: TaskOutcome) -> None:
    """Record an honest ``awaiting_*`` status while the graph waits on a human.

    Without this, the store (and therefore ``jarvis status``) called an
    interrupted task "running" while it was actually waiting for an approval
    or a clarification.  Best-effort: observability never changes the answer.
    """
    if not outcome.confirmation:
        return
    from jarvis.taskstore import TaskStatus as Status

    status = (
        Status.AWAITING_CLARIFICATION
        if outcome.interrupt_kind == "clarification"
        else Status.AWAITING_CONFIRMATION
    )
    _set_status(ctx, outcome.task_id, status)


def _set_status(ctx: AppContext, task_id: str, status: Any) -> None:
    tasks = getattr(ctx, "tasks", None)
    if tasks is None or not getattr(tasks, "available", False) or not task_id:
        return
    try:
        tasks.update_status(task_id, status)
    except Exception:  # observability must never change the task's answer
        ctx.logger.debug("task status %s not recorded", status, exc_info=True)


def _resume_refusal(
    ctx: AppContext,
    checkpointer: Any,
    config: dict[str, Any],
) -> str | None:
    """Refuse to resume when durable task state and the checkpoint disagree."""
    tasks = getattr(ctx, "tasks", None)
    if tasks is None or not getattr(tasks, "available", False):
        return None
    thread_id = config["configurable"]["thread_id"]
    try:
        snapshot = tasks.get(thread_id)
    except TaskStoreError:
        return "task lifecycle store is unavailable; refusing to resume"
    if snapshot is None:
        return "no durable task record matches this checkpoint; refusing to resume"
    if snapshot.status in TERMINAL_STATUSES:
        return f"task is already {snapshot.status.value}; refusing to resume"
    if checkpointer is None:
        return "task checkpoint is not available; refusing to resume"
    try:
        checkpoint = checkpointer.get_tuple(config)
    except (sqlite3.Error, KeyError, ValueError, RuntimeError):
        return "task checkpoint cannot be read; refusing to resume"
    if checkpoint is None:
        return "task checkpoint is missing; refusing to resume"
    return None


def extract_interrupt(values: dict[str, Any]) -> dict[str, Any] | None:
    """Pull the first pending confirmation payload from a raw invoke result."""
    interrupts = values.get("__interrupt__") or ()
    for it in interrupts:
        value = getattr(it, "value", None)
        if isinstance(value, dict):
            return value
    return None


def _outcome(
    task_id: str,
    config: dict[str, Any],
    checkpointer: Any,
    graph: Any,
    values: dict[str, Any],
) -> TaskOutcome:
    confirmation = extract_interrupt(values)
    if checkpointer is None:
        persisted: dict[str, Any] = values
    else:
        snapshot = graph.get_state(config)
        persisted = dict(snapshot.values)

    decision_map: dict[str, Decision] = {
        k: v for k, v in (persisted.get("decisions") or {}).items() if isinstance(v, Decision)
    }
    results: list[StepResult] = [
        r for r in (persisted.get("results") or []) if isinstance(r, StepResult)
    ]

    return TaskOutcome(
        task_id=task_id,
        interrupted=bool(confirmation),
        confirmation=confirmation,
        final_answer=persisted.get("final_answer"),
        error=persisted.get("error"),
        halted_reason=persisted.get("halted_reason"),
        decisions=decision_map,
        results=results,
        spoken_answer=persisted.get("spoken_answer"),
        screen_answer=persisted.get("screen_answer"),
        state=persisted,
    )
