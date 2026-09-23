"""brain node: the single LLM classification/planning stage.

One structured LLM call produces a transient :class:`BrainDecision` that splits
the graph deterministically:

* ``conversation`` - answered inline; projected into a conversation
  :class:`~jarvis.agent.state.Plan` (with ``dialog_answer``) plus the final
  answer so ``respond`` works unchanged;
* ``action`` - actions projected into a multi-step tool
  :class:`~jarvis.agent.state.Plan` and run through validate / policy_gate /
  act;
* ``clarification`` - routed to the ``clarify`` node, which interrupts and asks
  the human before this node re-runs with the answer in context;
* ``unsupported`` - halted with an honest message.

The :class:`BrainDecision` / :class:`ActionIntent` objects are **never written
to AgentState**: only the projected :class:`Plan` / :class:`Step` models (which
are on the checkpoint serializer allowlist) enter the state.
"""

from __future__ import annotations

from typing import Any

from jarvis.agent.context import AppContext
from jarvis.agent.prompts import brain_system_prompt, brain_user_prompt
from jarvis.agent.schemas import BrainDecision
from jarvis.agent.state import Plan, Step

#: Deterministic, user-facing halt reasons.  The LLM output never reaches the
#: user verbatim on failure; a code path decides which of these is shown.
LLM_NOT_CONFIGURED = (
    "JARVIS's AI backend is not configured yet. "
    "Configure an LLM provider before asking me to reason about tasks."
)
LLM_TIMEOUT = "JARVIS couldn't get a response from the AI backend in time."
LLM_RATE_LIMITED = "I couldn't reach the AI service."
LLM_PROVIDER_ERROR = "I couldn't reach the AI service."
LLM_INVALID_OUTPUT = "I couldn't safely understand that request."
LLM_NO_ANSWER = "I couldn't produce an answer."


def brain(state: dict[str, Any], ctx: AppContext) -> dict[str, Any]:
    """Classify the request and propose actions via one structured LLM call."""
    if ctx.llm is None:
        return {"halted_reason": LLM_NOT_CONFIGURED}

    previous_error = state.get("error") if int(state.get("validate_attempts") or 0) > 0 else None
    catalogue = ctx.registry.catalogue_for_llm()
    system = brain_system_prompt(catalogue)
    user = brain_user_prompt(
        state.get("user_input") or "",
        memory_context=[
            x if isinstance(x, dict) else {} for x in (state.get("memory_context") or [])
        ],
        clarification_question=state.get("clarification_question"),
        clarification_answer=state.get("clarification_answer"),
        repair_error=previous_error,
    )

    try:
        model, usage = ctx.llm.structured(
            system=system, user=user, schema=BrainDecision, model_role="planner", temperature=0.0
        )
    except Exception as exc:  # noqa: BLE001 - a brain failure stops this task only
        ctx.logger.warning("brain call failed: %s", exc)
        return {"halted_reason": _llm_halted_reason(exc)}
    if not isinstance(model, BrainDecision):
        ctx.logger.warning("brain returned a non-BrainDecision: %s", type(model).__name__)
        return {"halted_reason": LLM_INVALID_OUTPUT}

    ctx.logger.info(
        "agent event=brain task_id=%s request_type=%s actions=%d",
        state.get("task_id"),
        model.request_type,
        len(model.actions),
    )
    return _project(model, state, usage)


def _project(decision: BrainDecision, state: dict[str, Any], usage: Any) -> dict[str, Any]:
    """Map the transient BrainDecision onto checkpointable state updates."""
    update: dict[str, Any] = {
        "api_calls": int(state.get("api_calls") or 0) + usage.calls,
        "tokens": int(state.get("tokens") or 0) + usage.prompt_tokens + usage.completion_tokens,
        "clarification_question": None,
        "clarification_answer": None,
    }

    if decision.request_type == "conversation":
        answer = (decision.response_text or "").strip()
        if not answer:
            return {**update, "halted_reason": LLM_NO_ANSWER}
        return {
            **update,
            "plan": Plan(
                kind="conversation",
                goal=decision.goal or "answer directly",
                steps=[],
                dialog_answer=answer,
            ),
            "request_kind": "conversation",
            "final_answer": answer,
            "error": None,
        }

    if decision.request_type == "clarification":
        return {
            **update,
            "plan": Plan(
                kind="tool",
                goal=decision.goal or "clarify request",
                steps=[],
                needs_clarification=True,
                clarification_question=decision.clarification_question,
            ),
            "request_kind": "clarification",
            "clarification_question": decision.clarification_question,
            "error": None,
        }

    if decision.request_type == "unsupported":
        reason = (decision.response_text or "").strip()
        return {
            **update,
            "request_kind": "unsupported",
            "halted_reason": reason or "I can't do that.",
        }

    plan = Plan(
        kind="tool",
        goal=decision.goal or "",
        steps=[
            Step(
                id=f"s{i + 1}",
                tool=a.tool,
                args=a.args,
                rationale=a.rationale,
                expect=a.expect,
                depends_on_untrusted=a.depends_on_untrusted,
            )
            for i, a in enumerate(decision.actions)
        ],
    )
    return {
        **update,
        "plan": plan,
        "request_kind": "tool",
        "error": None,  # consume the repair cue so a success isn't misread as a failure
    }


def _llm_halted_reason(exc: Exception) -> str:
    """Deterministic mapping of an LLM failure to a safe, honest message.

    Decided purely from the exception message markers (the LLM client never
    embeds secrets or keys in these messages); anything unrecognised is treated
    as a provider error rather than leaking details to the user.
    """
    text = str(exc).lower()
    if "rate limit" in text:
        return LLM_RATE_LIMITED
    if "timed out" in text or "timeout" in text:
        return LLM_TIMEOUT
    if any(
        marker in text
        for marker in ("repaired", "expected a", "unexpected structure", "script exhausted")
    ):
        return LLM_INVALID_OUTPUT
    return LLM_PROVIDER_ERROR
