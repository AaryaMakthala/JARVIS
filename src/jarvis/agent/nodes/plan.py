"""plan node: LLM structured-output call producing a Plan.

Uses the planner system prompt (with the registry catalogue, no tiers) and the
user input.  On a rejected previous plan (validate set ``error``), the rejection
text is appended and the planner is asked to repair it exactly once.
"""

from __future__ import annotations

from typing import Any

from jarvis.agent.context import AppContext
from jarvis.agent.state import Plan
from jarvis.llm.prompts import planner_system_prompt, planner_user_prompt


def plan(state: dict[str, Any], ctx: AppContext) -> dict[str, Any]:
    """Ask the LLM for a structured Plan; accumulate usage counters."""
    if ctx.llm is None:
        return {"halted_reason": "No LLM backend configured (run `jarvis init`)."}

    previous_error = state.get("error") if int(state.get("validate_attempts") or 0) > 0 else None
    catalogue = ctx.registry.catalogue_for_llm()
    system = planner_system_prompt(catalogue)
    user = planner_user_prompt(
        state.get("user_input") or "",
        memory_context=[
            x if isinstance(x, dict) else {} for x in (state.get("memory_context") or [])
        ],
        repair_error=previous_error,
    )

    try:
        model, usage = ctx.llm.structured(
            system=system, user=user, schema=Plan, model_role="planner", temperature=0.0
        )
    except Exception as exc:  # noqa: BLE001 - LLM failures stop this task only
        ctx.logger.warning("planning failed: %s", exc)
        return {"halted_reason": f"Planner error: {exc}"}
    if not isinstance(model, Plan):
        return {"halted_reason": "Planner returned an unexpected structure."}

    return {
        "plan": model,
        "error": None,  # consume the repair cue so a success isn't misread as a failure
        "api_calls": int(state.get("api_calls") or 0) + usage.calls,
        "tokens": int(state.get("tokens") or 0) + usage.prompt_tokens + usage.completion_tokens,
    }
