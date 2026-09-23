"""converse node: plain-text answer for conversational (no-tool) requests.

Only reached when ``understand_request`` routed a conversation here.  Produces
the spoken answer via the LLM ``fast`` role and stops the graph at
``respond`` - no plan, no policy gate, no tool execution.
"""

from __future__ import annotations

from typing import Any

from jarvis.agent.context import AppContext
from jarvis.llm.prompts import converse_system_prompt, converse_user_prompt


def converse(state: dict[str, Any], ctx: AppContext) -> dict[str, Any]:
    """Ask the LLM for a concise conversational answer."""
    plan = state.get("plan")
    dialog = getattr(plan, "dialog_answer", None)
    if isinstance(dialog, str) and dialog.strip():
        # The planner already wrote a direct answer ("answer this directly");
        # honour it and skip a second, redundant LLM call.
        answer = dialog.strip()
        ctx.logger.info(
            "agent event=converse task_id=%s answer_chars=%d source=plan",
            state.get("task_id"),
            len(answer),
        )
        return {"final_answer": answer}

    if ctx.llm is None:
        return {"halted_reason": "No LLM backend configured (run `jarvis init`)."}

    memory_context = [x if isinstance(x, dict) else {} for x in (state.get("memory_context") or [])]
    try:
        answer, usage = ctx.llm.text(
            system=converse_system_prompt(),
            user=converse_user_prompt(state.get("user_input") or "", memory_context),
            model_role="fast",
            temperature=0.2,
            max_tokens=400,
        )
    except Exception as exc:  # noqa: BLE001 - LLM failures stop this task only
        ctx.logger.warning("conversation failed: %s", exc)
        return {"halted_reason": f"Answer error: {exc}"}

    answer = (answer or "").strip()
    if not answer:
        return {"halted_reason": "Could not produce an answer."}

    ctx.logger.info(
        "agent event=converse task_id=%s answer_chars=%d",
        state.get("task_id"),
        len(answer),
    )
    return {
        "final_answer": answer,
        "api_calls": int(state.get("api_calls") or 0) + usage.calls,
        "tokens": int(state.get("tokens") or 0) + usage.prompt_tokens + usage.completion_tokens,
    }
