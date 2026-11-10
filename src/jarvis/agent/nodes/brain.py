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
from jarvis.agent.nodes.telemetry import record_route
from jarvis.agent.prompts import brain_system_prompt, brain_user_prompt
from jarvis.agent.routing import UNSUPPORTED_ANSWER
from jarvis.agent.schemas import BrainDecision
from jarvis.agent.session_context import (
    SESSION_THREAD,
    render_resolution,
)
from jarvis.agent.state import Plan, RequestKind, Step
from jarvis.agent.tool_groups import select_catalogue, trim_site_catalogue
from jarvis.browser.sites import render_site_catalogue, sites_from_settings
from jarvis.llm.routing import route_request
from jarvis.memory.intent import detect_memory_intent
from jarvis.messaging.grounding import ground_brain_update

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
    memory_update = _memory_command(state, ctx)
    if memory_update is not None:
        return memory_update
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

    route = route_request(state.get("user_input") or "", has_image=bool(state.get("has_image")))
    ctx.logger.info(
        "agent event=llm_route task_id=%s class=%s role=%s reason=%s",
        task_id,
        route.task_class.value,
        route.model_role,
        route.reason_code.value,
    )
    record_route(
        state,
        ctx,
        task_class=route.task_class.value,
        model_role=route.model_role,
        reason_code=route.reason_code.value,
    )

    previous_error = state.get("error") if int(state.get("validate_attempts") or 0) > 0 else None
    source = state.get("source") or "terminal"
    agent_settings = getattr(getattr(ctx, "settings", None), "agent", None)
    # Prompt trimming is opt-in (agent.prompt_trim_enabled, default false).  It
    # only decides which tools/sites are *described*; the registry stays closed
    # and validate/policy_gate/act judge every step exactly as before.  With the
    # flag off the catalogue and site block are byte-identical to before.
    prompt_trim = bool(getattr(agent_settings, "prompt_trim_enabled", False))
    if prompt_trim:
        catalogue = select_catalogue(ctx.registry, state)
    else:
        catalogue = ctx.registry.catalogue_for_llm()
    system = brain_system_prompt(catalogue)
    if getattr(agent_settings, "use_specialists", False):
        system = (
            system
            + "\n12b. If the task needs several browser steps that depend on what the page shows (searching, opening a result, reading a page, playing a video), return request_type 'browse' with a goal and no actions. Simple single actions stay 'action'."
        )
    session_block = _session_reference_block(state, ctx)
    conversation_block = _conversation_history_block(state, ctx)
    site_directory = sites_from_settings(ctx.settings)
    sites_block = (
        trim_site_catalogue(site_directory, state)
        if prompt_trim
        else render_site_catalogue(site_directory)
    )
    # build referent summary
    referent_summary = None
    try:
        rr = state.get("resolved_refs") or {}
        if isinstance(rr, dict):
            parts = []
            if rr.get("has_result_set") and rr.get("result_set_positions"):
                parts.append(
                    "REFERENTS: result set positions: "
                    + ",".join(str(x) for x in rr.get("result_set_positions", []))
                )
            if rr.get("has_page") and rr.get("page_url"):
                parts.append(
                    f"REFERENTS: current page: <untrusted_data>{rr['page_url']}</untrusted_data>"
                )
            if rr.get("has_photo") and rr.get("photo_path"):
                parts.append(f"REFERENTS: last photo: {rr['photo_path']}")
            elif rr.get("has_file") and rr.get("file_path"):
                parts.append(f"REFERENTS: last file: {rr['file_path']}")
            if parts:
                referent_summary = "\n".join(parts)
                if len(referent_summary) > 1200:
                    referent_summary = referent_summary[:1200]
    except Exception:  # noqa: BLE001 - referent context must never block a task
        referent_summary = None
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
        sites=sites_block,
        referent_summary=referent_summary,
        conversation_history=conversation_block,
    )

    llm_budget: dict[str, Any] = {}
    if source == "voice":
        # No retry, and a per-call timeout the composite client recognises as
        # latency-sensitive: it then bounds the WHOLE walk by
        # llm.voice_deadline_seconds instead of the total deadline.
        voice_deadline = float(getattr(ctx.settings.llm, "voice_deadline_seconds", 12.0) or 12.0)
        llm_budget = {
            "max_retries": 0,
            "timeout_s": min(float(ctx.settings.llm.timeout_seconds), voice_deadline, 10.0),
        }

    try:
        model, usage = ctx.llm.structured(
            system=system,
            user=user,
            schema=BrainDecision,
            model_role=route.model_role,
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
    update = _project(model, state, usage, ctx)
    # Ground WhatsApp contact references *before* validation, the policy gate
    # and the W4 action hash, so the confirmation names the canonical saved
    # contact and the approval binds to it.  Deterministic resolution runs
    # first; only an unresolved reference may use the tool-free text model.
    return ground_brain_update(update, state, ctx)


def _memory_command(state: dict[str, Any], ctx: AppContext) -> dict[str, Any] | None:
    """Answer a clear remember/forget command without calling the LLM.

    The deterministic parser only recognises imperative memory commands.  The
    write itself happens in ``memory_save`` (the single memory-write node), so
    this function has no side effects and cannot change the policy path.
    """
    intent = detect_memory_intent(str(state.get("user_input") or ""))
    if intent is None:
        return None
    # Never hijack the answer to a pending clarification: that turn belongs to
    # the original task, and a memory-like phrase may be its requested detail.
    if state.get("clarification_question") or state.get("clarification_answer") is not None:
        return None
    task_id = state.get("task_id")
    if intent.error == "sensitive":
        ctx.logger.info(
            "agent event=memory_command task_id=%s action=%s reason=sensitive",
            task_id,
            intent.action,
        )
        return _memory_answer(
            "I won't remember that: it looks like a credential or other sensitive information.",
            "refuse a sensitive memory",
        )
    if intent.error == "too_long":
        ctx.logger.info(
            "agent event=memory_command task_id=%s action=%s reason=too_long",
            task_id,
            intent.action,
        )
        return _memory_answer(
            "I can only remember short facts (up to 400 characters), so I haven't stored that.",
            "refuse an oversized memory",
        )
    if ctx.dry_run:
        ctx.logger.info(
            "agent event=memory_command task_id=%s action=%s reason=dry_run",
            task_id,
            intent.action,
        )
        verb = "store" if intent.action == "remember" else "forget"
        return _memory_answer(f"This is a dry run, so I won't {verb} that.", "simulate memory")
    preferences = getattr(ctx.memory, "preferences", None)
    getter = getattr(preferences, "get", None)
    setter = getattr(preferences, "set", None)
    deleter = getattr(preferences, "delete", None)
    if intent.action == "remember":
        if not callable(setter):
            return _memory_answer(
                "I can't remember that because local memory isn't available.", "memory unavailable"
            )
        try:
            existing = getter(intent.key) if callable(getter) else None
        except Exception:  # noqa: BLE001 - reading memory must never break the turn
            existing = None
        answer = (
            "I already have that stored, sir."
            if existing == intent.content
            else "Noted, sir. I'll remember that."
        )
        ctx.logger.info(
            "agent event=memory_command task_id=%s action=remember reason=accepted",
            task_id,
        )
        return _memory_answer(answer, "remember an explicit fact")
    if not callable(getter) or not callable(deleter):
        return _memory_answer(
            "I can't forget that because local memory isn't available.", "memory unavailable"
        )
    try:
        existing = getter(intent.key)
    except Exception as exc:  # noqa: BLE001 - report honestly, never raise
        ctx.logger.warning("memory lookup failed: %s", exc)
        return _memory_answer(
            "I couldn't check local memory, so I haven't forgotten anything.",
            "memory unavailable",
        )
    answer = "I'll forget that, sir." if existing else "I don't have that stored, sir."
    ctx.logger.info(
        "agent event=memory_command task_id=%s action=forget reason=accepted",
        task_id,
    )
    return _memory_answer(answer, "forget an explicit fact")


def _memory_answer(answer: str, goal: str) -> dict[str, Any]:
    """A conversation update with no tool steps and no LLM usage."""
    return {
        "plan": Plan(kind="conversation", goal=goal, steps=[], dialog_answer=answer),
        "request_kind": "conversation",
        "final_answer": answer,
        "clarification_question": None,
        "clarification_answer": None,
        "error": None,
    }


def _conversation_history_block(state: dict[str, Any], ctx: AppContext) -> str | None:
    """The bounded recent-turn block for this conversation (B5), or ``None``.

    History is read-only prompt context from the caller's in-memory
    :class:`~jarvis.agent.conversation.ConversationManager`; a missing manager,
    an unknown conversation or a rendering failure must never break the task,
    so every failure degrades to "no history".  The block itself is already
    wrapped in ``<untrusted_data>`` by the manager.
    """
    manager = getattr(ctx, "conversations", None)
    render = getattr(manager, "render_history", None)
    if not callable(render):
        return None
    conversation_id = state.get("conversation_id")
    if not isinstance(conversation_id, str) or not conversation_id:
        return None
    try:
        return render(conversation_id)
    except Exception:  # history must never break the brain
        ctx.logger.debug("conversation history unavailable", exc_info=True)
        return None


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


def _project(
    decision: BrainDecision, state: dict[str, Any], usage: Any, ctx: AppContext | None = None
) -> dict[str, Any]:
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
        # The brain's own words when it gave them; otherwise the fixed refusal
        # from the route contract.  Never a tool step (see jarvis.agent.routing).
        reason = (decision.response_text or "").strip() or UNSUPPORTED_ANSWER
        return {
            **update,
            "request_kind": "unsupported",
            "halted_reason": reason,
        }

    if decision.request_type == "browse":
        # Fail closed: with no context or the flag off, no specialist loop can
        # run, so the browse request is an honest unsupported halt.
        agent_settings = getattr(getattr(ctx, "settings", None), "agent", None)
        if getattr(agent_settings, "use_specialists", False):
            return {
                **update,
                "plan": None,
                "request_kind": RequestKind.BROWSE.value,
                "specialist_goal": (
                    decision.goal or decision.response_text or state.get("user_input") or ""
                ),
                "specialist_phase": "decide",
                "specialist_calls": 0,
                "specialist_history": [],
                "error": None,
            }
        return {
            **update,
            "plan": None,
            "request_kind": "unsupported",
            "halted_reason": "Browser specialist is disabled.",
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
