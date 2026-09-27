"""respond node: deterministic final message (the only terminal node).

Composes an honest summary: cancelled, clarification requested, refused,
partial/hard failure, unverifiable success, or full success.  Never claims
verification that was not performed, and never lets machine output through:
every string this node returns passes
:func:`jarvis.agent.answer.safe_final_answer`, so a leaked routing document
cannot reach the terminal, the IPC client, or TTS even if it somehow got into
state (an old checkpoint, a future node, a provider that ignored the schema).
"""

from __future__ import annotations

from typing import Any

from jarvis.agent.answer import safe_final_answer
from jarvis.agent.context import AppContext
from jarvis.agent.state import StepResult

#: Longest single tool output line shown as a final answer.  A tool's raw
#: output is evidence, not an answer: the logs showed a 921-character
#: three-source web-search dump handed straight to the user and then read
#: aloud for 79 seconds.  Anything longer is truncated *for display* and the
#: full text stays in ``StepResult`` for the audit log.
MAX_ANSWER_LINE_CHARS = 400


def respond(state: dict[str, Any], ctx: AppContext) -> dict[str, Any]:
    """Produce the final human-facing answer."""
    del ctx
    if state.get("cancelled"):
        return _answer("Cancelled.")

    plan = state.get("plan")
    if plan is not None and plan.needs_clarification:
        question = plan.clarification_question or "Could you rephrase that?"
        return _answer(f"Clarification needed: {question}")

    if state.get("halted_reason"):
        return _answer(str(state["halted_reason"]))

    if state.get("final_answer"):
        # Set by the brain node for conversational (no-tool) requests.
        return _answer(str(state["final_answer"]))

    results: list[StepResult] = list(state.get("results") or [])
    if not results:
        return _answer("No task performed.")

    # Verified-False is a hard failure: never claim success on an action whose
    # post-condition check reported failure (docs/02_ARCHITECTURE.md section 11).
    failures = [r for r in results if not r.ok or r.verified is False]
    if failures:
        last = failures[-1]
        if last.ok and last.verified is False:
            detail = f"step {last.step_id} could not be verified"
        else:
            detail = last.error or f"step {last.step_id} failed"
        return _answer(f"Could not complete the task: {detail}.")

    unverified = [r for r in results if r.verified is None and r.ok]
    if unverified:
        hint = f" ({len(unverified)} step(s) could not be verified)"
        return _answer(_success_text(results) + hint)

    return _answer(_success_text(results))


def _answer(text: str) -> dict[str, str]:
    """Wrap a final answer after stripping any leaked internal payload."""
    return {"final_answer": safe_final_answer(text)}


def _success_text(results: list[StepResult]) -> str:
    if len(results) == 1:
        return _single_text(results[0])
    lines = [_display_line(r) for r in results]
    return "Done.\n" + "\n".join(f"- {line}" for line in lines)


def _single_text(result: StepResult) -> str:
    if result.output:
        return _display_line(result)
    return "Done."


def _display_line(result: StepResult) -> str:
    """One bounded, human-facing line for a completed step.

    A tool's raw output is evidence, not an answer: the logs showed a
    921-character three-source web-search dump handed straight to the user and
    then read aloud for 79 seconds.  Long or multi-paragraph output is
    truncated *for display only* — the full text stays in ``StepResult`` and in
    the audit log.
    """
    output = (result.output or "").strip()
    if not output:
        return f"done: {result.step_id}"
    if len(output) <= MAX_ANSWER_LINE_CHARS and "\n\n" not in output:
        return output
    return output[:MAX_ANSWER_LINE_CHARS].rstrip() + "…"
