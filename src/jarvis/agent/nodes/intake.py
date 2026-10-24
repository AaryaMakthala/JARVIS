"""intake node: normalise input, assign task_id, handle meta-commands.

Cheap and deterministic (no LLM).  Meta commands like "cancel" / "stop" mark
the task cancelled so the graph can respond without planning.  The raw input
is validated and normalised through :class:`UserRequest` so every downstream
node sees a trimmed ``text`` and a known ``source``.

This is also where the task's ``task_log`` row is *opened* (observability,
docs/01 section "Targets"): a task that then suspends on a confirmation, or
dies with the daemon, stays visible as ``running`` instead of leaving no trace
at all.  :func:`jarvis.agent.nodes.telemetry.record_started` cannot raise.
"""

from __future__ import annotations

import re
from typing import Any

from jarvis.agent.context import AppContext
from jarvis.agent.nodes.telemetry import record_started
from jarvis.agent.nodes.util import new_task_id
from jarvis.agent.schemas import UserRequest

_CANCEL_RE = re.compile(r"^(cancel|stop|abort|never mind|no thanks|wait)$", re.IGNORECASE)


def intake(state: dict[str, Any], ctx: AppContext) -> dict[str, Any]:
    """Prepare the state for one task run."""
    user_input = (state.get("user_input") or "").strip()
    source = state.get("source", "terminal")
    if not user_input:
        task_id = state.get("task_id") or new_task_id()
        update: dict[str, Any] = {
            "task_id": task_id,
            "conversation_id": state.get("conversation_id") or "local",
            "source": source if source in ("terminal", "voice", "benchmark") else "terminal",
            "user_input": "",
            "cancelled": False,
            "request_kind": "",
            "clarification_question": None,
            "clarification_answer": None,
            "clarification_count": 0,
            "clarification_history": [],
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
            "halted_reason": "No input received. Tell me what to do.",
        }
        record_started(update, ctx)
        return update
    try:
        request = UserRequest(
            text=user_input,
            source=source if source in ("terminal", "voice", "benchmark") else "terminal",
        )
    except Exception:
        task_id = state.get("task_id") or new_task_id()
        update = {
            "task_id": task_id,
            "conversation_id": state.get("conversation_id") or "local",
            "source": source if source in ("terminal", "voice", "benchmark") else "terminal",
            "user_input": user_input,
            "cancelled": False,
            "request_kind": "",
            "clarification_question": None,
            "clarification_answer": None,
            "clarification_count": 0,
            "clarification_history": [],
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
            "halted_reason": "No input received. Tell me what to do.",
        }
        record_started(update, ctx)
        return update
    cancelled = bool(_CANCEL_RE.match(request.text))

    update = {
        "task_id": state.get("task_id") or new_task_id(),
        "conversation_id": state.get("conversation_id") or "local",
        "source": request.source,
        "user_input": request.text,
        "cancelled": cancelled,
        "request_kind": "",
        "clarification_question": None,
        "clarification_answer": None,
        "clarification_count": 0,
        "clarification_history": [],
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
    if cancelled:
        update["halted_reason"] = "Cancelled by the user."
    else:
        update["halted_reason"] = None
    # Read from ``update`` (not ``state``): on a first run the task_id is
    # assigned here, so the caller would otherwise have no id to log under.
    record_started(update, ctx)
    return update
