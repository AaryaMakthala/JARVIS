"""Graph node factories.

Each node is a pure function ``fn(state, ctx)``; :func:`wrap` closes over the
:class:`AppContext` so LangGraph only ever calls ``fn(state)``.  Tests call the
pure functions directly with a fake context.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from jarvis.agent.nodes.act import act
from jarvis.agent.nodes.intake import intake
from jarvis.agent.nodes.memory_retrieve import memory_retrieve
from jarvis.agent.nodes.plan import plan
from jarvis.agent.nodes.policy_gate import policy_gate
from jarvis.agent.nodes.respond import respond
from jarvis.agent.nodes.validate import validate
from jarvis.agent.nodes.verify import verify

__all__ = [
    "act",
    "intake",
    "memory_retrieve",
    "plan",
    "policy_gate",
    "respond",
    "route_after_act",
    "route_after_intake",
    "route_after_policy_gate",
    "route_after_validate",
    "route_after_verify",
    "validate",
    "verify",
    "wrap",
]


def wrap(
    node: Callable[..., dict[str, Any]], ctx: Any
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Bind a node function to its AppContext for use inside the graph."""
    return lambda state: node(state, ctx)


def route_after_intake(state: dict[str, Any]) -> str:
    """START -> intake -> (memory_retrieve | respond)."""
    if state.get("cancelled") or state.get("halted_reason"):
        return "respond"
    return "memory_retrieve"


def route_after_validate(state: dict[str, Any]) -> str:
    """validate -> (plan | policy_gate | respond)."""
    if state.get("halted_reason"):
        return "respond"
    plan_obj = state.get("plan")
    if plan_obj is not None and plan_obj.needs_clarification:
        return "respond"
    if state.get("error"):
        return "plan"
    return "policy_gate"


def route_after_policy_gate(state: dict[str, Any]) -> str:
    """policy_gate -> (act | respond)."""
    if state.get("halted_reason"):
        return "respond"
    return "act"


def route_after_act(state: dict[str, Any]) -> str:
    """act -> (verify | respond)."""
    if state.get("halted_reason"):
        return "respond"
    return "verify"


def route_after_verify(state: dict[str, Any]) -> str:
    """verify -> (act | policy_gate | respond)."""
    if state.get("retry_now"):
        return "act"
    results = state.get("results") or []
    plan_obj = state.get("plan")
    if not plan_obj or not plan_obj.steps:
        return "respond"
    if (not results) or (not results[-1].ok) or results[-1].verified is False:
        return "respond"  # fail closed inside verify
    idx = int(state.get("step_index") or 0)
    if idx >= len(plan_obj.steps):
        return "respond"
    return "policy_gate"
