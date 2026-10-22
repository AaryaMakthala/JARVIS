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

from jarvis.agent import fastpath
from jarvis.agent.context import AppContext
from jarvis.agent.nodes.clarify import MAX_CLARIFICATIONS
from jarvis.agent.prompts import brain_system_prompt, brain_user_prompt
from jarvis.agent.schemas import BrainDecision
from jarvis.agent.session_context import (
    SESSION_THREAD,
    render_resolution,
)
from jarvis.agent.state import Plan, Step

#: Deterministic, user-facing halt reasons.  The LLM output never reaches the
#: user verbatim on failure; a code path decides which of these is shown.
LLM_NOT_CONFIGURED = (
    "JARVIS's AI backend is not configured yet. "
    "Configure an LLM provider before asking me to reason about tasks."
)

#: Voice variant of the not-configured halt.  The full terminal message is
#: read aloud through TTS, which costs several seconds of dead air after every
#: conversational command while no LLM provider is configured (the agent path
#: itself takes ~50 ms; the rest was SAPI speaking 108 characters).  The voice
#: channel gets the same honest refusal in a speech-sized form; terminal keeps
#: the actionable full text.
LLM_NOT_CONFIGURED_VOICE = "My AI backend is not configured yet."
LLM_TIMEOUT = "JARVIS couldn't get a response from the AI backend in time."
LLM_RATE_LIMITED = "I couldn't reach the AI service."
LLM_PROVIDER_ERROR = "I couldn't reach the AI service."
LLM_INVALID_OUTPUT = "I couldn't safely understand that request."
LLM_NO_ANSWER = "I couldn't produce an answer."


def brain(state: dict[str, Any], ctx: AppContext) -> dict[str, Any]:
    """Classify the request and propose actions via one structured LLM call."""
    task_id = state.get("task_id")
    fp_update = fastpath.brain_update(state, ctx)
    if fp_update is not None:
        return fp_update
    if ctx.llm is None:
        # Fail immediately: there is nothing to wait for, so no retry, no
        # timeout, no provider probing may happen on this path.
        if (state.get("source") or "") == "voice":
            ctx.logger.info(
                "agent event=llm_skip task_id=%s reason=no_provider variant=voice",
                task_id,
            )
            return {"halted_reason": LLM_NOT_CONFIGURED_VOICE}
        ctx.logger.info(
            "agent event=llm_skip task_id=%s reason=no_provider variant=terminal",
            task_id,
        )
        return {"halted_reason": LLM_NOT_CONFIGURED}

    ctx.logger.info("agent event=llm_start task_id=%s", task_id)

    previous_error = state.get("error") if int(state.get("validate_attempts") or 0) > 0 else None
    source = state.get("source") or "terminal"
    catalogue = ctx.registry.catalogue_for_llm()
    system = brain_system_prompt(catalogue)
    session_block = _session_reference_block(state, ctx)
    user = brain_user_prompt(
        state.get("user_input") or "",
        memory_context=[
            x if isinstance(x, dict) else {} for x in (state.get("memory_context") or [])
        ],
        clarification_question=state.get("clarification_question"),
        clarification_answer=state.get("clarification_answer"),
        repair_error=previous_error,
        source=source,
        clarification_count=int(state.get("clarification_count") or 0),
        session_context=session_block,
        clarification_history=state.get("clarification_history"),
    )

    llm_budget: dict[str, Any] = {}
    if source == "voice":
        llm_budget = {
            "max_retries": 0,
            "timeout_s": min(float(ctx.settings.llm.timeout_seconds), 10.0),
        }

    try:
        model, usage = ctx.llm.structured(
            system=system,
            user=user,
            schema=BrainDecision,
            model_role="planner",
            temperature=0.0,
            **llm_budget,
        )
    except Exception as exc:  # noqa: BLE001 - a brain failure stops this task only
        ctx.logger.warning("brain call failed: %s", exc)
        return {"halted_reason": _llm_halted_reason(exc)}
    finally:
        ctx.logger.info("agent event=llm_complete task_id=%s", task_id)
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


def _session_reference_block(state: dict[str, Any], ctx: AppContext) -> str | None:
    """Resolve a deictic reference ("it", "that photo") to a prompt block.

    The store holds artifacts only from trusted tool results (see
    :mod:`jarvis.agent.session_context`).  The code - not the LLM - decides
    whether the words reference a prior item and whether exactly one is
    resolvable; anything unresolvable (none, or more than one) becomes a plain
    instruction to ask one clarification.
    """
    text = state.get("user_input") or ""
    answer = state.get("clarification_answer")
    if answer:
        text = f"{text}\n{answer}"
    return render_resolution(ctx.session_context.reference_resolution(SESSION_THREAD, text))


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
        asked = int(state.get("clarification_count") or 0)
        if asked >= MAX_CLARIFICATIONS:
            # Round budget exhausted: the clarify-node limit check is the
            # backstop, but this converts *before* another interrupt so the
            # task fails closed deterministically with an honest stop message.
            listed = ", ".join(decision.missing[:5]) if decision.missing else "some details"
            return {
                **update,
                "plan": None,  # drop the leftover round's clarification plan so respond shows the halt
                "request_kind": "unsupported",
                "halted_reason": (
                    f"Still missing: {listed}. I have already asked {asked} "
                    "question(s); please start over with more detail."
                ),
            }
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

    if decision.missing and not (decision.assumption or "").strip():
        # Completeness contract (ROADMAP V2.2 Stage 5a): proceeding on a gap is
        # only lawful when the default used is disclosed.  Missing details with
        # no stated assumption means values were silently invented - fail closed.
        return {
            **update,
            "plan": None,
            "request_kind": "unsupported",
            "halted_reason": (
                "I cannot proceed safely: some details are still missing and no "
                "assumption was stated. Please repeat the request with the specifics."
            ),
        }

    plan = Plan(
        kind="tool",
        goal=decision.goal or "",
        assumption=(decision.assumption or "").strip(),
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
