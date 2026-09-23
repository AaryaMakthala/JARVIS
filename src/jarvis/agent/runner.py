"""Synchronous task runner (Phase 1).

Owns the run/resume lifecycle: build the graph on a shared SqliteSaver, run a
task in its own thread, capture the pending confirmation, and resume it with
the user's answer.  No daemon/IPC yet (Phase 3); this is what the CLI drives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from langgraph.types import Command

from jarvis.agent.context import AppContext
from jarvis.agent.graph import build_graph
from jarvis.agent.schemas import AgentResponse
from jarvis.agent.state import Decision, StepResult

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
) -> TaskOutcome:
    """Start a new task; returns a TaskOutcome, possibly awaiting confirmation."""
    task_id = thread_id or uuid4().hex[:12]
    graph = build_graph(ctx, checkpointer)
    config = {"configurable": {"thread_id": task_id}}
    values = graph.invoke({"user_input": user_input, "source": source}, config)
    return _outcome(task_id, config, checkpointer, graph, values)


def resume_task(
    ctx: AppContext,
    checkpointer: Any,
    thread_id: str,
    answer: Any,
) -> TaskOutcome:
    """Resume a task at its interruption point with the user's answer."""
    graph = build_graph(ctx, checkpointer)
    config = {"configurable": {"thread_id": thread_id}}
    values = graph.invoke(Command(resume=answer), config)
    return _outcome(thread_id, config, checkpointer, graph, values)


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
        state=persisted,
    )
