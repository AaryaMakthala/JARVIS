"""clarify node: ask the human a free-text question via an interrupt.

Reached only when ``route_after_brain`` saw ``Plan.needs_clarification``.  The
node interrupts with a :class:`ClarificationRequest` payload; the daemon / CLI /
voice layers forward the question and resume with the human's free-text answer.
The answer is validated here - a non-blank string that is not a cancel word -
and stored so the brain can re-run with the question+answer in context.

The number of questions is bounded by ``MAX_CLARIFICATIONS`` (enforced at
entry): beyond that the task fails closed to ``respond`` instead of looping.
Cancel words follow the same deterministic set as :mod:`intake`; no side
effects happen before ``interrupt()``.
"""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from jarvis.agent.context import AppContext
from jarvis.agent.nodes.intake import _CANCEL_RE
from jarvis.agent.schemas import ClarificationRequest
from jarvis.agent.state import clarification_rounds

#: Maximum number of clarification questions before the task fails closed.
MAX_CLARIFICATIONS = 2


def _limit_message(asked: int) -> str:
    return (
        f"I still need clarification, and I've already asked {asked} question(s). "
        "Please start over with more detail."
    )


def clarify(state: dict[str, Any], ctx: AppContext) -> dict[str, Any]:
    """Ask the clarifying question; store and route the validated answer."""
    asked = int(state.get("clarification_count") or 0)
    if asked >= MAX_CLARIFICATIONS:
        ctx.logger.warning(
            "agent event=clarify task_id=%s outcome=limit reached=%d",
            state.get("task_id"),
            asked,
        )
        return {"halted_reason": _limit_message(asked)}

    question = state.get("clarification_question") or "Could you rephrase that?"
    ctx.logger.info(
        "agent event=clarify task_id=%s question_chars=%d",
        state.get("task_id"),
        len(question),
    )
    answer = interrupt(ClarificationRequest(question=question).model_dump(mode="json"))

    if not isinstance(answer, str) or not answer.strip():
        return {"halted_reason": "No clarification received."}
    if _CANCEL_RE.match(answer.strip()):
        return {"cancelled": True, "halted_reason": "Cancelled by the user."}

    # Keep every answered round (Stage 8.4e).  The single
    # clarification_question/answer pair only holds the newest round, so a
    # second question would otherwise erase what the human already told us and
    # the brain would re-plan from scratch.
    history = [
        {"question": question_text, "answer": answer_text}
        for question_text, answer_text in clarification_rounds(state.get("clarification_history"))
    ]
    history.append({"question": question.strip(), "answer": answer.strip()})

    return {
        "clarification_answer": answer.strip(),
        "clarification_count": asked + 1,
        "clarification_history": history,
        "halted_reason": None,
    }
