"""understand_request node: deterministic request classification.

The LLM already classified the request during :mod:`plan` (``Plan.kind``); this
node records that classification in the state (``request_kind``) and is the
router that splits the graph onto the conversation vs. tool path.  It does no
LLM work - classification is LLM *proposed*, but the routing decision is made
here in code, and only ``tool`` kind may ever reach ``validate``/``act``.
"""

from __future__ import annotations

from typing import Any

from jarvis.agent.context import AppContext


def understand_request(state: dict[str, Any], ctx: AppContext) -> dict[str, Any]:
    """Normalise the planner's classification into ``request_kind``."""
    if state.get("halted_reason"):
        # A previous node (e.g. plan) already halted the task; never clobber it.
        return {"request_kind": "tool"}
    plan = state.get("plan")
    if plan is None:
        return {"request_kind": "tool", "halted_reason": "No plan was produced."}
    kind = plan.kind if getattr(plan, "kind", "tool") == "conversation" else "tool"
    ctx.logger.info(
        "agent event=understand_request task_id=%s kind=%s",
        state.get("task_id"),
        kind,
    )
    return {"request_kind": kind}
