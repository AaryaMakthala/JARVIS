"""intake node: normalise input, assign task_id, handle meta-commands.

Cheap and deterministic (no LLM).  Meta commands like "cancel" / "stop" mark
the task cancelled so the graph can respond without planning.  The raw input
is validated and normalised through :class:`UserRequest` so every downstream
node sees a trimmed ``text`` and a known ``source``.
"""

from __future__ import annotations

import re
from typing import Any

from jarvis.agent.context import AppContext
from jarvis.agent.nodes.util import new_task_id
from jarvis.agent.schemas import UserRequest

_CANCEL_RE = re.compile(r"^(cancel|stop|abort|never mind|no thanks|wait)$", re.IGNORECASE)


def intake(state: dict[str, Any], ctx: AppContext) -> dict[str, Any]:
    """Prepare the state for one task run."""
    del ctx
    user_input = (state.get("user_input") or "").strip()
    source = state.get("source", "terminal")
    request = UserRequest(
        text=user_input,
        source=source if source in ("terminal", "voice", "benchmark") else "terminal",
    )
    cancelled = bool(_CANCEL_RE.match(request.text))

    update: dict[str, Any] = {
        "task_id": state.get("task_id") or new_task_id(),
        "source": request.source,
        "user_input": request.text,
        "cancelled": cancelled,
        "request_kind": "",
        "clarification_question": None,
        "clarification_answer": None,
        "clarification_count": 0,
        "step_index": 0,
        "retry_count": 0,
        "retry_now": False,
        "replan_now": False,
        "replan_count": 0,
        "pending_replan": False,
        "replan_decision": None,
        "validate_attempts": 0,
        "api_calls": 0,
        "tokens": 0,
        "plan": None,
        "decisions": {},
        "approved_hashes": [],
        "results": [],
        "memory_context": [],
    }
    if not request.text:
        update["halted_reason"] = "No input received. Tell me what to do."
    elif cancelled:
        update["halted_reason"] = "Cancelled by the user."
    else:
        update["halted_reason"] = None
    return update
