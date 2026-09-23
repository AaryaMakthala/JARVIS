"""respond node: deterministic final message (the only terminal node).

Composes an honest summary: cancelled, clarification requested, refused,
partial/hard failure, unverifiable success, or full success.  Never claims
verification that was not performed.
"""

from __future__ import annotations

from typing import Any

from jarvis.agent.context import AppContext
from jarvis.agent.state import StepResult


def respond(state: dict[str, Any], ctx: AppContext) -> dict[str, Any]:
    """Produce the final human-facing answer."""
    del ctx
    if state.get("cancelled"):
        return {"final_answer": "Cancelled."}

    plan = state.get("plan")
    if plan is not None and plan.needs_clarification:
        question = plan.clarification_question or "Could you rephrase that?"
        return {"final_answer": f"Clarification needed: {question}"}

    if state.get("halted_reason"):
        return {"final_answer": state["halted_reason"]}

    if state.get("final_answer"):
        # Set by the brain node for conversational (no-tool) requests.
        return {"final_answer": state["final_answer"]}

    results: list[StepResult] = list(state.get("results") or [])
    if not results:
        return {"final_answer": "No task performed."}

    # Verified-False is a hard failure: never claim success on an action whose
    # post-condition check reported failure (docs/02_ARCHITECTURE.md section 11).
    failures = [r for r in results if not r.ok or r.verified is False]
    if failures:
        last = failures[-1]
        if last.ok and last.verified is False:
            detail = f"step {last.step_id} could not be verified"
        else:
            detail = last.error or f"step {last.step_id} failed"
        return {"final_answer": f"Could not complete the task: {detail}."}

    unverified = [r for r in results if r.verified is None and r.ok]
    if unverified:
        hint = f" ({len(unverified)} step(s) could not be verified)"
        return {"final_answer": _success_text(results) + hint}

    return {"final_answer": _success_text(results)}


def _success_text(results: list[StepResult]) -> str:
    if len(results) == 1:
        return _single_text(results[0])
    lines = [r.output or f"done: {r.step_id}" for r in results]
    return "Done.\n" + "\n".join(f"- {line}" for line in lines)


def _single_text(result: StepResult) -> str:
    if result.output:
        return result.output
    return "Done."
