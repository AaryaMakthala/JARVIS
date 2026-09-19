"""LangGraph assembly for the Phase 1 agent.

Topology (docs/02_ARCHITECTURE.md sec. 4):

    START -> intake -> memory_retrieve -> plan -> validate
                 |                                    |
                 v                                    v
              respond <--------- policy_gate <- policy_gate <-+
                 ^            |            |                  |
                 |            v            v                  |
                 +---- policy_gate -> act -> verify ----------+
                             |               |
                             v               v
                          respond         (act retry / next step)

Confirmations suspend inside ``policy_gate`` via ``interrupt``; resume
re-enters that node deterministically.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from jarvis.agent.context import AppContext
from jarvis.agent.nodes import (
    act,
    intake,
    memory_retrieve,
    plan,
    policy_gate,
    respond,
    route_after_act,
    route_after_intake,
    route_after_policy_gate,
    route_after_validate,
    route_after_verify,
    validate,
    verify,
    wrap,
)
from jarvis.agent.state import AgentState

__all__ = ["build_graph", "build_secure_serde", "open_sqlite_checkpointer"]

#: The ONLY first-party types the checkpoint serializer is allowed to revive.
#: Everything else in a checkpoint is refused at the msgpack layer (returned as
#: plain data, never imported/instantiated).  ``Step`` is listed because it is
#: nested inside ``Plan``; ``Profile`` is never stored in state.  See
#: docs/03_SECURITY_AND_POLICY.md section 5 (secrets) and the checkpoint
#: serializer notes in docs/02_ARCHITECTURE.md section 4.
_ALLOWED_MSGPACK_TYPES: tuple[tuple[str, str], ...] = (
    ("jarvis.agent.state", "Plan"),
    ("jarvis.agent.state", "Step"),
    ("jarvis.agent.state", "Decision"),
    ("jarvis.agent.state", "StepResult"),
)


def build_secure_serde() -> JsonPlusSerializer:
    """Build the strict, allowlisted checkpoint serializer.

    The SQLite checkpointer stores the whole graph state; on load we must never
    import or instantiate an arbitrary type encoded in those bytes.  This
    serializer uses strict msgpack, allowlists exactly the state models above,
    and deliberately does **not** enable pickle fallback or
    ``allowed_msgpack_modules=True`` (the permissive "warn but allow" mode).
    """
    return JsonPlusSerializer(
        pickle_fallback=False,
        allowed_json_modules=None,
        allowed_msgpack_modules=_ALLOWED_MSGPACK_TYPES,
    )


def open_sqlite_checkpointer(path: str | None) -> SqliteSaver:
    """Open a checkpointer bound to this process' connection.

    ``check_same_thread=False`` is required or the synchronous call raises a
    thread-affinity ProgrammingError on Windows.  The checkpoint bytes are
    read/written with :func:`build_secure_serde` so only our State models can
    be deserialized.
    """
    import os
    from pathlib import Path

    if path is None:
        raise ValueError("checkpoints_db path is None")
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    return SqliteSaver(conn, serde=build_secure_serde())


def build_graph(ctx: AppContext, checkpointer: Any = None) -> Any:
    """Compile the agent graph for this context (optionally persistable)."""
    g = StateGraph(AgentState)
    g.add_node("intake", wrap(intake, ctx))
    g.add_node("memory_retrieve", wrap(memory_retrieve, ctx))
    g.add_node("plan", wrap(plan, ctx))
    g.add_node("validate", wrap(validate, ctx))
    g.add_node("policy_gate", wrap(policy_gate, ctx))
    g.add_node("act", wrap(act, ctx))
    g.add_node("verify", wrap(verify, ctx))
    g.add_node("respond", wrap(respond, ctx))

    g.add_edge(START, "intake")
    g.add_conditional_edges(
        "intake", route_after_intake, {"memory_retrieve": "memory_retrieve", "respond": "respond"}
    )

    g.add_edge("memory_retrieve", "plan")
    g.add_edge("plan", "validate")
    g.add_conditional_edges(
        "validate",
        route_after_validate,
        {"plan": "plan", "policy_gate": "policy_gate", "respond": "respond"},
    )

    g.add_conditional_edges(
        "policy_gate", route_after_policy_gate, {"act": "act", "respond": "respond"}
    )
    g.add_conditional_edges("act", route_after_act, {"verify": "verify", "respond": "respond"})
    g.add_conditional_edges(
        "verify",
        route_after_verify,
        {"act": "act", "policy_gate": "policy_gate", "respond": "respond"},
    )
    g.add_edge("respond", END)

    if checkpointer is None:
        return g.compile()
    return g.compile(checkpointer=checkpointer)
