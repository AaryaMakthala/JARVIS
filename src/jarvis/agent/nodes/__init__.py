"""Graph node factories.

Each node is a pure function ``fn(state, ctx)``; :func:`wrap` closes over the
:class:`AppContext` so LangGraph only ever calls ``fn(state)``.  Tests call the
pure functions directly with a fake context.

Routers (all deterministic, no LLM):

* ``route_after_intake``  -> memory_retrieve | respond
* ``route_after_brain``   -> clarify | validate | respond   (BrainDecision)
* ``route_after_clarify`` -> brain | respond               (answer or cancel)
* ``route_after_validate``-> brain | replan | policy_gate | respond
* ``route_after_policy_gate`` -> act | respond
* ``route_after_act``     -> verify | respond
* ``route_after_verify``  -> act | replan | policy_gate | respond
* ``route_after_replan``  -> validate | respond
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from jarvis.agent.nodes.act import act
from jarvis.agent.nodes.brain import brain
from jarvis.agent.nodes.clarify import clarify
from jarvis.agent.nodes.intake import intake
from jarvis.agent.nodes.memory_retrieve import memory_retrieve
from jarvis.agent.nodes.policy_gate import policy_gate
from jarvis.agent.nodes.replan import replan
from jarvis.agent.nodes.respond import respond
from jarvis.agent.nodes.validate import validate
from jarvis.agent.nodes.verify import verify

__all__ = [
    "act",
    "brain",
    "clarify",
    "intake",
    "memory_retrieve",
    "policy_gate",
    "replan",
    "respond",
    "route_after_act",
    "route_after_brain",
    "route_after_clarify",
    "route_after_intake",
    "route_after_policy_gate",
    "route_after_replan",
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


def route_after_brain(state: dict[str, Any]) -> str:
    """brain -> (clarify | validate | respond).

    The human-facing routes are decided here, in code, from the *projected*
    Plan produced by the brain node: clarification goes to the ``clarify``
    interrupt, conversation straight to ``respond``, anything else through the
    deterministic validate / policy_gate / act pipeline.
    """
    if state.get("halted_reason"):
        return "respond"
    plan_obj = state.get("plan")
    if plan_obj is None:
        return "respond"
    if plan_obj.needs_clarification:
        return "clarify"
    if plan_obj.kind == "conversation":
        return "respond"
    return "validate"


def route_after_clarify(state: dict[str, Any]) -> str:
    """clarify -> (brain | respond)."""
    if state.get("cancelled") or state.get("halted_reason"):
        return "respond"
    return "brain"


def route_after_validate(state: dict[str, Any]) -> str:
    """validate -> (brain | replan | policy_gate | respond)."""
    if state.get("halted_reason"):
        return "respond"
    plan_obj = state.get("plan")
    if plan_obj is not None and plan_obj.needs_clarification:
        return "respond"
    if state.get("error"):
        return "replan" if state.get("pending_replan") else "brain"
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
    """verify -> (act | replan | policy_gate | respond)."""
    if state.get("replan_now"):
        return "replan"
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


def route_after_replan(state: dict[str, Any]) -> str:
    """replan -> (validate | respond)."""
    if state.get("halted_reason"):
        return "respond"
    return "validate"
