"""LangGraph agent package (implemented in Phase 1).

Public pieces: :class:`AppContext` (dependency container), ``build_graph``,
``run_task`` / ``resume_task``, and the state schema in :mod:`jarvis.agent.state`.
"""

from jarvis.agent.context import AppContext, make_app_context
from jarvis.agent.graph import build_graph, open_sqlite_checkpointer
from jarvis.agent.runner import TaskOutcome, resume_task, run_task

__all__ = [
    "AppContext",
    "TaskOutcome",
    "build_graph",
    "make_app_context",
    "open_sqlite_checkpointer",
    "resume_task",
    "run_task",
]
