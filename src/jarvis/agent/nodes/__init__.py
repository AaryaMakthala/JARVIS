"""Graph node factories.

Each node is a pure function ``fn(state, ctx)``; :func:`wrap` closes over the
:class:`AppContext` so LangGraph only ever calls ``fn(state)``.  Tests call the
pure functions directly with a fake context.

Routers (all deterministic, no LLM):

* ``route_after_intake``  -> memory_retrieve | respond
* ``route_after_brain``   -> clarify | validate | respond   (BrainDecision;
  the table lives in :mod:`jarvis.agent.routing`)
* ``route_after_clarify`` -> brain | respond               (answer or cancel)
* ``route_after_validate``-> brain | replan | policy_gate | respond
* ``route_after_policy_gate`` -> act | respond
* ``route_after_act``     -> verify | respond
* ``route_after_verify``  -> act | replan | policy_gate | respond
* ``route_after_replan``  -> validate | respond

``memory_save`` is not a routed node: ``respond`` is the only terminal node and
it always hands off to ``memory_save`` (docs/02 section 4), so every task gets
exactly one memory write attempt - including refusals and failures.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from jarvis.agent.nodes import specialist
from jarvis.agent.nodes.act import act
from jarvis.agent.nodes.brain import brain
from jarvis.agent.nodes.clarify import clarify
from jarvis.agent.nodes.compose import compose
from jarvis.agent.nodes.context_resolve import context_resolve
from jarvis.agent.nodes.intake import intake
from jarvis.agent.nodes.memory_retrieve import memory_retrieve
from jarvis.agent.nodes.memory_save import memory_save
from jarvis.agent.nodes.policy_gate import policy_gate
from jarvis.agent.nodes.replan import replan
from jarvis.agent.nodes.respond import respond
from jarvis.agent.nodes.telemetry import trace_node
from jarvis.agent.nodes.validate import validate
from jarvis.agent.nodes.verify import verify
from jarvis.agent.routing import route_for_brain_result
from jarvis.tools.base import CancelToken

__all__ = [
    "act",
    "brain",
    "clarify",
    "compose",
    "context_resolve",
    "intake",
    "memory_retrieve",
    "memory_save",
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
    "specialist",
    "validate",
    "verify",
    "wrap",
]


def wrap(
    node: Callable[..., dict[str, Any]],
    ctx: Any,
    cancel: CancelToken | None = None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Bind a node function to its AppContext for use inside the graph.

    ``cancel`` (optional) is the **per-task** token: when the user cancels a
    run, the next node to be entered raises :class:`~jarvis.tools.base.Cancelled`
    instead of executing, so the remaining steps of the plan are never planned
    or acted on.    The check belongs here because ``wrap`` is the single choke
    point every node passes through.  The token is bound per *graph* rather than
    on the shared :class:`~jarvis.agent.context.AppContext` on purpose: a
    cancelled run can still be unwinding while the next task starts, and a
    shared field would let the new task's token mask the old task's cancel.

    The same choke point records the run's **node path**: after the cancel
    check, the node's name is appended to the in-memory trace for
    ``state["task_id"]`` (:func:`jarvis.agent.nodes.telemetry.trace_node`).
    Node names only - never user text, tool arguments or results - and the
    trace is written once and freed by ``respond``.
    """
    name = getattr(node, "__name__", "node")

    if cancel is None:

        def _plain(state: dict[str, Any]) -> dict[str, Any]:
            trace_node(str(state.get("task_id") or ""), name)
            return node(state, ctx)

        return _plain

    def _gated(state: dict[str, Any]) -> dict[str, Any]:
        cancel.check()
        trace_node(str(state.get("task_id") or ""), name)
        return node(state, ctx)

    return _gated


def route_after_intake(state: dict[str, Any]) -> str:
    """START -> intake -> (memory_retrieve | respond)."""
    if state.get("cancelled") or state.get("halted_reason"):
        return "respond"
    return "memory_retrieve"


def route_after_brain(state: dict[str, Any]) -> str:
    """brain -> (clarify | validate | respond | specialist_decide).

    The route table itself lives in :mod:`jarvis.agent.routing`
    (:func:`~jarvis.agent.routing.route_for_brain_result`), so the graph's edges
    and the contract are defined in one place.  This function stays as the name
    ``graph.py`` and the tests already use.
    """
    return route_for_brain_result(state)


def route_after_clarify(state: dict[str, Any]) -> str:
    """clarify -> (brain | respond)."""
    if state.get("cancelled") or state.get("halted_reason"):
        return "respond"
    return "brain"


def route_after_validate(state: dict[str, Any]) -> str:
    """validate -> (brain | replan | specialist_decide | policy_gate | respond)."""
    if state.get("halted_reason"):
        return "respond"
    plan_obj = state.get("plan")
    if plan_obj is not None and plan_obj.needs_clarification:
        return "respond"
    if state.get("error"):
        # A rejected specialist plan is the specialist's to correct; every other
        # task keeps the original brain/replan repair path.
        if state.get("specialist_phase") == "decide":
            return "specialist_decide"
        return "replan" if state.get("pending_replan") else "brain"
    return "policy_gate"


def route_after_policy_gate(state: dict[str, Any]) -> str:
    """policy_gate -> (act | specialist_decide | respond)."""
    if state.get("halted_reason"):
        return "respond"
    # A recoverable specialist refusal records why and hands back to decide;
    # an ordinary step keeps the unchanged path to act.
    if state.get("specialist_phase") == "decide":
        return "specialist_decide"
    return "act"


def route_after_act(state: dict[str, Any]) -> str:
    """act -> (verify | respond)."""
    if state.get("halted_reason"):
        return "respond"
    return "verify"


def route_after_verify(state: dict[str, Any]) -> str:
    """verify -> (act | replan | policy_gate | respond)."""
    results = state.get("results") or []
    if state.get("specialist_phase") == "observe" and (
        not results or not results[-1].ok or results[-1].verified is False
    ):
        # The specialist loop owns its own recovery: observe records the failure
        # in specialist_history and the next decide adapts to it.  A failed step
        # never falls into the generic replan node, and a blind retry of the
        # same failing step is not taken either (the LLM decides what to do).
        # Non-specialist runs never see phase "observe", so they are unchanged.
        return "specialist_observe"
    if state.get("replan_now"):
        return "replan"
    if state.get("retry_now"):
        return "act"
    plan_obj = state.get("plan")
    if not plan_obj or not plan_obj.steps:
        # check if in specialist flow
        if state.get("specialist_phase") == "observe":
            return "specialist_observe"
        return "respond"
    if (not results) or (not results[-1].ok) or results[-1].verified is False:
        if state.get("specialist_phase") == "observe":
            return "specialist_observe"
        return "respond"  # fail closed inside verify
    idx = int(state.get("step_index") or 0)
    if idx >= len(plan_obj.steps):
        if state.get("specialist_phase") == "observe":
            return "specialist_observe"
        return "respond"
    # if specialist active and we just finished a step, go to observe
    if state.get("specialist_phase") == "observe":
        return "specialist_observe"
    return "policy_gate"


def route_after_replan(state: dict[str, Any]) -> str:
    """replan -> (validate | respond)."""
    if state.get("halted_reason"):
        return "respond"
    return "validate"


def route_after_specialist_decide(state: dict[str, Any]) -> str:
    """specialist_decide -> (validate | respond).

    ``done`` / ``give_up`` / the call catch all leave the previous iteration's
    plan in state; the terminal answer must win over that stale plan, or the
    run would validate (and reset the bookkeeping of) a plan that is finished.
    """
    if state.get("halted_reason") or state.get("final_answer") is not None:
        return "respond"
    plan_obj = state.get("plan")
    if plan_obj is not None and getattr(plan_obj, "steps", None):
        return "validate"
    return "respond"


def route_after_specialist_observe(state: dict[str, Any]) -> str:
    """specialist_observe -> (act | specialist_decide).

    ``observe`` sets the phase back to "observe" only when it built the page-
    refresh step act must execute; otherwise it hands straight back to decide.
    The previous plan is still in state, so the phase (not the plan) is what
    says what just happened.
    """
    if state.get("specialist_phase") == "observe":
        return "act"
    return "decide"
