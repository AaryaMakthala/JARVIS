"""LangGraph assembly for the agent.

Topology (docs/02_ARCHITECTURE.md sec. 4):

    START -> intake -> memory_retrieve -> brain
                                             |
                 +---------------------------+--------------------------+------------------+
                 | (conversation)           | (action)                 | (clarification)  | (browse, flagged)
                 v                           v                         v                  v
              respond                    validate                   clarify        specialist_decide
                                             |  ^        (answer)        |                |
                                             v  +- replan<-...           +----> brain     (act)->validate
                  prepare_readback <- - - - +      (validate rejected)                |
                                                                                      observe<-verify
                                                                                         |
                                                                                      act/decide

                       prepare_readback -> policy_gate -> act -> verify -> ... | respond

                       compose -> respond -> memory_save -> END

Every router still returns the route key ``"respond"``; the conditional-edge
maps translate that key to the **compose** node, which writes the short spoken
answer and the screen answer and then hands over to ``respond`` (the single
sanitizer and task-log writer).  With ``compose.enabled = false`` compose writes
nothing and ``respond`` produces exactly its old text.

Every entry into ``policy_gate`` passes through ``prepare_readback`` first: it
observes the real target, read-only, and persists the readback a consequential
browser click is confirmed against (Phase 5C).  Confirmations suspend inside
``policy_gate`` via ``interrupt`` and
clarifications suspend inside ``clarify``; resume re-enters that node
deterministically.  ``brain`` is the only LLM classification step and projects
its transient :class:`BrainDecision` onto the checkpointed Plan/Step shape
before anything else sees the state.  ``memory_save`` is a deterministic,
LLM-free writer that only records a *verified* plan (or a failure); it can
neither execute a tool nor approve one.
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
    brain,
    clarify,
    compose,
    context_resolve,
    intake,
    memory_retrieve,
    memory_save,
    policy_gate,
    prepare_readback,
    replan,
    respond,
    route_after_act,
    route_after_brain,
    route_after_clarify,
    route_after_intake,
    route_after_policy_gate,
    route_after_replan,
    route_after_specialist_decide,
    route_after_specialist_observe,
    route_after_validate,
    route_after_verify,
    specialist,
    validate,
    verify,
    wrap,
)
from jarvis.agent.state import AgentState

__all__ = [
    "GRAPH_NODE_NAMES",
    "build_graph",
    "build_secure_serde",
    "open_sqlite_checkpointer",
]

#: Every node name the graph can contain.  Used to build the trusted task-store
#: vocabulary; the set is fixed even when the optional specialist nodes are
#: disabled, so a stored detail naming either specialist node stays valid.
GRAPH_NODE_NAMES: frozenset[str] = frozenset(
    {
        "intake",
        "context_resolve",
        "memory_retrieve",
        "brain",
        "clarify",
        "validate",
        "prepare_readback",
        "policy_gate",
        "act",
        "verify",
        "replan",
        "compose",
        "respond",
        "memory_save",
        "specialist_decide",
        "specialist_observe",
    }
)

#: The ONLY first-party types the checkpoint serializer is allowed to revive.
#: Everything else in a checkpoint is refused at the msgpack layer (returned as
#: plain data, never imported/instantiated).  ``Step`` is listed because it is
#: nested inside ``Plan``; ``ReplanDecision`` is stored by the replan node;
#: ``Profile`` is never stored in state.  See
#: docs/03_SECURITY_AND_POLICY.md section 5 (secrets) and the checkpoint
#: serializer notes in docs/02_ARCHITECTURE.md section 4.
_ALLOWED_MSGPACK_TYPES: tuple[tuple[str, str], ...] = (
    ("jarvis.agent.state", "Plan"),
    ("jarvis.agent.state", "Step"),
    ("jarvis.agent.state", "Decision"),
    ("jarvis.agent.state", "StepResult"),
    ("jarvis.agent.schemas", "ReplanDecision"),
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


def build_graph(ctx: AppContext, checkpointer: Any = None, cancel: Any = None) -> Any:
    """Compile the agent graph for this context (optionally persistable).

    ``cancel`` is the run's :class:`~jarvis.tools.base.CancelToken`; when it is
    set (the user cancelled), the next node raises instead of executing.  Every
    node is gated on it, so a cancelled run stops between steps and never
    reaches the next tool call; ``None`` keeps the plain, ungated graph.
    """
    g = StateGraph(AgentState)
    g.add_node("intake", wrap(intake, ctx, cancel))
    g.add_node("context_resolve", wrap(context_resolve, ctx, cancel))
    g.add_node("memory_retrieve", wrap(memory_retrieve, ctx, cancel))
    g.add_node("brain", wrap(brain, ctx, cancel))
    g.add_node("clarify", wrap(clarify, ctx, cancel))
    g.add_node("validate", wrap(validate, ctx, cancel))
    # Phase 5C: observe the real target ONE HOP BEFORE the node that interrupts.
    # An interrupted node re-runs from the top on resume and loses anything it
    # wrote first, so the readback has to be persisted by a node that already
    # completed - otherwise the resume value could be matched against a
    # re-observation of a target that changed while the user was reading.
    g.add_node("prepare_readback", wrap(prepare_readback, ctx, cancel))
    g.add_node("policy_gate", wrap(policy_gate, ctx, cancel))
    g.add_node("act", wrap(act, ctx, cancel))
    g.add_node("verify", wrap(verify, ctx, cancel))
    g.add_node("replan", wrap(replan, ctx, cancel))
    g.add_node("compose", wrap(compose, ctx, cancel))
    g.add_node("respond", wrap(respond, ctx, cancel))
    g.add_node("memory_save", wrap(memory_save, ctx, cancel))
    use_sp = getattr(getattr(ctx, "settings", None), "agent", None)
    if use_sp is not None and getattr(use_sp, "use_specialists", False):
        g.add_node("specialist_decide", wrap(specialist.decide, ctx, cancel))
        g.add_node("specialist_observe", wrap(specialist.observe, ctx, cancel))

    g.add_edge(START, "intake")
    g.add_conditional_edges(
        "intake", route_after_intake, {"memory_retrieve": "context_resolve", "respond": "compose"}
    )
    g.add_edge("context_resolve", "memory_retrieve")
    g.add_edge("memory_retrieve", "brain")
    use_sp = getattr(getattr(ctx, "settings", None), "agent", None)
    if use_sp is not None and getattr(use_sp, "use_specialists", False):
        g.add_conditional_edges(
            "brain",
            route_after_brain,
            {"clarify": "clarify", "validate": "validate", "respond": "compose", "specialist_decide": "specialist_decide"},
        )
        g.add_conditional_edges(
            "specialist_decide", route_after_specialist_decide, {"validate": "validate", "respond": "compose"}
        )
        g.add_conditional_edges(
            "specialist_observe", route_after_specialist_observe, {"act": "act", "decide": "specialist_decide"}
        )
    else:
        g.add_conditional_edges(
            "brain",
            route_after_brain,
            {"clarify": "clarify", "validate": "validate", "respond": "compose"},
        )
    g.add_conditional_edges(
        "clarify", route_after_clarify, {"brain": "brain", "respond": "compose"}
    )
    g.add_edge("prepare_readback", "policy_gate")

    if use_sp is not None and getattr(use_sp, "use_specialists", False):
        g.add_conditional_edges(
            "validate",
            route_after_validate,
            {
                "brain": "brain",
                "replan": "replan",
                "policy_gate": "prepare_readback",
                "respond": "compose",
                "specialist_decide": "specialist_decide",
            },
        )
        g.add_conditional_edges(
            "policy_gate",
            route_after_policy_gate,
            {"act": "act", "respond": "compose", "specialist_decide": "specialist_decide"},
        )
    else:
        g.add_conditional_edges(
            "validate",
            route_after_validate,
            {
                "brain": "brain",
                "replan": "replan",
                "policy_gate": "prepare_readback",
                "respond": "compose",
            },
        )
        g.add_conditional_edges(
            "policy_gate", route_after_policy_gate, {"act": "act", "respond": "compose"}
        )
    g.add_conditional_edges("act", route_after_act, {"verify": "verify", "respond": "compose"})
    if use_sp is not None and getattr(use_sp, "use_specialists", False):
        g.add_conditional_edges(
            "verify",
            route_after_verify,
            {"act": "act", "replan": "replan", "policy_gate": "prepare_readback", "respond": "compose", "specialist_observe": "specialist_observe"},
        )
    else:
        g.add_conditional_edges(
            "verify",
            route_after_verify,
            {"act": "act", "replan": "replan", "policy_gate": "prepare_readback", "respond": "compose"},
        )
    g.add_conditional_edges(
        "replan", route_after_replan, {"validate": "validate", "respond": "compose"}
    )
    g.add_edge("compose", "respond")
    g.add_edge("respond", "memory_save")
    g.add_edge("memory_save", END)

    if checkpointer is None:
        return g.compile()
    return g.compile(checkpointer=checkpointer)
