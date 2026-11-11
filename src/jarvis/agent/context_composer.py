"""Task-specific, budgeted prompt composition (B9).

Every model-driven node used to assemble its own prompt from whatever happened
to be in state.  This module is the single composer for the two planning calls
(``brain`` and ``replan``): it chooses a compact capability directory, full
schemas for only the relevant candidates, bounded memory and bounded
conversation history, then enforces one explicit application-level budget.

Budget rules (docs/B9 report section 9):

* token counts from :func:`estimate_tokens` are **estimates** and always
  labelled that way; provider-reported counts stay in LLM telemetry;
* safety instructions, the current user request, typed output constraints and
  the selected tools' complete JSON schemas are never truncated;
* context is reduced in a fixed order: older conversation history, then
  low-relevance memory, then low-relevance candidate schemas, then execution
  observations;
* if the request still cannot fit, the caller fails closed with an honest
  message instead of sending a corrupt prompt.

The module makes no policy decision and never executes anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jarvis.agent.capability_index import (
    build_capability_index,
    catalogue_entries,
    discover_candidates,
    render_capability_directory,
)
from jarvis.agent.prompts import brain_system_prompt, brain_user_prompt
from jarvis.agent.tool_groups import trim_site_catalogue
from jarvis.browser.sites import render_site_catalogue, sites_from_settings
from jarvis.llm.prompts import replan_system_prompt, replan_user_prompt

__all__ = [
    "CONTEXT_BUDGET_EXCEEDED",
    "BrainContext",
    "ContextStats",
    "ReplanContext",
    "compose_brain_context",
    "compose_replan_context",
    "estimate_tokens",
]

#: Deterministic refusal when even the minimum sufficient prompt exceeds the
#: application budget.  Safety rules and JSON schemas are never truncated.
CONTEXT_BUDGET_EXCEEDED = (
    "That request is too large to process safely within the current context "
    "budget. Please split it into a smaller request."
)

#: A registry this small is cheaper to describe in full than to shortlist.
SMALL_REGISTRY_LIMIT = 8

#: Conservative character-per-token divisor.  Real tokenisers vary by model;
#: this deliberately over-estimates rather than under-estimates.
_CHARS_PER_TOKEN = 4

#: Conversation-history levels tried, largest first.
_HISTORY_LEVELS = (6, 4, 2, 0)

#: Replan observation levels tried, largest first.
_OBSERVATION_LEVELS = (8, 4, 2, 0)

#: Longest referent block accepted from state.
_REFERENT_CHARS = 1200


@dataclass(frozen=True, slots=True)
class ContextStats:
    """Safe, content-free measurements about one composed prompt."""

    estimated_input_tokens: int
    budget_tokens: int
    candidate_tools: int = 0
    schema_tools: int = 0
    memory_candidates: int = 0
    history_turns: int = 0
    trimmed: bool = False
    trim_actions: tuple[str, ...] = ()
    fits: bool = True

    def attempt_detail(self) -> dict[str, Any]:
        """Counts only: no prompt, memory, history or tool-argument content."""
        return {
            "estimated_input_tokens": max(int(self.estimated_input_tokens), 0),
            "candidate_tools": max(int(self.candidate_tools), 0),
            "schema_tools": max(int(self.schema_tools), 0),
            "memory_candidates": max(int(self.memory_candidates), 0),
            "history_turns": max(int(self.history_turns), 0),
            "context_trimmed": bool(self.trimmed),
        }


@dataclass(frozen=True, slots=True)
class BrainContext:
    """The exact strings and safe measurements handed to the brain call."""

    system: str
    user: str
    stats: ContextStats
    candidate_names: tuple[str, ...]
    schema_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReplanContext:
    """The exact strings and safe measurements handed to the replanner."""

    system: str
    user: str
    stats: ContextStats
    schema_names: tuple[str, ...]


def estimate_tokens(text: str) -> int:
    """Conservative **estimate** of input tokens (never provider-reported)."""
    return max(0, (len(text or "") + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN)


def _agent_settings(ctx: Any) -> Any:
    return getattr(getattr(ctx, "settings", None), "agent", None)


def _effective_budget(agent: Any) -> int:
    """Application-level input budget minus output reserve and safety margin.

    This is explicitly *not* a claim about any model's context window.  Unknown
    provider limits stay unknown; this is a conservative request cap JARVIS
    controls itself.
    """
    budget = int(getattr(agent, "context_budget_tokens", 24000) or 24000)
    margin = float(getattr(agent, "context_safety_margin", 0.15) or 0.0)
    reserved = int(getattr(agent, "context_reserved_output_tokens", 1024) or 0)
    return max(512, int(budget * max(0.0, 1.0 - margin)) - reserved)


def _index_for(ctx: Any) -> Any:
    index = getattr(ctx, "capability_index", None)
    if index is None:
        index = build_capability_index(ctx.registry)
    return index


def _history_block(state: dict[str, Any], ctx: Any, limit: int) -> str | None:
    """Bounded recent-turn block (already wrapped as untrusted data)."""
    if limit <= 0:
        return None
    manager = getattr(ctx, "conversations", None)
    render = getattr(manager, "render_history", None)
    if not callable(render):
        return None
    conversation_id = state.get("conversation_id")
    if not isinstance(conversation_id, str) or not conversation_id:
        return None
    try:
        return render(conversation_id, limit)
    except Exception:  # noqa: BLE001 - history must never break a task
        return None


def _memory(state: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    items = [x if isinstance(x, dict) else {} for x in (state.get("memory_context") or [])]
    return items[: max(0, limit)]


def _referent_summary(state: dict[str, Any]) -> str | None:
    """Resolve result-set/page/artifact referents into a bounded plain block."""
    try:
        refs = state.get("resolved_refs") or {}
        if not isinstance(refs, dict):
            return None
        parts: list[str] = []
        if refs.get("has_result_set") and refs.get("result_set_positions"):
            parts.append(
                "REFERENTS: result set positions: "
                + ",".join(str(x) for x in refs.get("result_set_positions", []))
            )
        if refs.get("has_page") and refs.get("page_url"):
            parts.append(
                f"REFERENTS: current page: <untrusted_data>{refs['page_url']}</untrusted_data>"
            )
        if refs.get("has_photo") and refs.get("photo_path"):
            parts.append(f"REFERENTS: last photo: {refs['photo_path']}")
        elif refs.get("has_file") and refs.get("file_path"):
            parts.append(f"REFERENTS: last file: {refs['file_path']}")
        if not parts:
            return None
        return "\n".join(parts)[:_REFERENT_CHARS]
    except Exception:  # noqa: BLE001 - referent context must never block a task
        return None


def _sites_block(ctx: Any, state: dict[str, Any], dynamic: bool) -> str:
    directory = sites_from_settings(ctx.settings)
    if dynamic:
        return trim_site_catalogue(directory, state)
    return render_site_catalogue(directory)


def _selection(
    state: dict[str, Any], ctx: Any, pinned: tuple[str, ...]
) -> tuple[bool, Any, list[str], list[str], str | None]:
    """Choose dynamic-vs-legacy mode and return names and the compact directory."""
    agent = _agent_settings(ctx)
    dynamic = bool(getattr(agent, "prompt_trim_enabled", True))
    index = _index_for(ctx)
    query = str(state.get("user_input") or "")
    if not dynamic:
        names = list(index.names())
        return False, index, names, names, None

    pinned_known = [name for name in pinned if index.get(name) is not None]
    if index.count() <= SMALL_REGISTRY_LIMIT:
        candidates = list(index.names())
    else:
        candidates = discover_candidates(
            index,
            query,
            limit=int(getattr(agent, "tool_candidate_limit", 5) or 5),
            pinned=pinned_known,
        )
    schema_limit = max(1, int(getattr(agent, "tool_schema_limit", 4) or 4))
    # Pinned tools are required: include every one, then fill with candidates.
    schema_names = list(pinned_known)
    for name in candidates:
        if len(schema_names) >= schema_limit:
            break
        if name not in schema_names:
            schema_names.append(name)
    if not pinned_known:
        schema_names = schema_names[:schema_limit]
    entries = catalogue_entries(index, ctx.registry, schema_names)
    # Only names that actually resolved to a canonical spec are advertised.
    resolved = [str(entry["name"]) for entry in entries]
    return True, index, candidates, resolved, render_capability_directory(index)


def _levels(maximum: int, preferred: tuple[int, ...]) -> list[int]:
    values = [value for value in preferred if value < maximum]
    values.extend(range(maximum, -1, -1))
    ordered: list[int] = []
    for value in [maximum, *values]:
        if value not in ordered:
            ordered.append(value)
    return ordered


def _trim_actions(
    history_max: int,
    history_used: int,
    memory_max: int,
    memory_used: int,
    schema_max: int,
    schema_used: int,
) -> tuple[str, ...]:
    actions: list[str] = []
    if history_used < history_max:
        actions.append(f"history:{history_max}->{history_used}")
    if memory_used < memory_max:
        actions.append(f"memory:{memory_max}->{memory_used}")
    if schema_used < schema_max:
        actions.append(f"schemas:{schema_max}->{schema_used}")
    return tuple(actions)


def compose_brain_context(
    state: dict[str, Any],
    ctx: Any,
    *,
    pinned_schema_tools: tuple[str, ...] = (),
) -> BrainContext:
    """Compose the smallest sufficient brain prompt within the request budget."""
    agent = _agent_settings(ctx)
    dynamic, _index, candidates, schema_names, directory = _selection(
        state, ctx, pinned_schema_tools
    )
    memory_all = _memory(state, len(state.get("memory_context") or []))
    conversation_id = state.get("conversation_id")
    history_max = _HISTORY_LEVELS[0] if isinstance(conversation_id, str) and conversation_id else 0
    memory_max = min(3, len(memory_all))
    schema_max = len(schema_names)
    min_schema = len([n for n in pinned_schema_tools if n in schema_names]) or (
        1 if schema_names else 0
    )
    effective = _effective_budget(agent)
    sites = _sites_block(ctx, state, dynamic)
    referents = _referent_summary(state)
    source = state.get("source") or "terminal"
    last_system = ""
    last_user = ""
    last_stats = ContextStats(0, effective, fits=False)
    for history_limit in _levels(history_max, _HISTORY_LEVELS):
        for memory_limit in range(memory_max, -1, -1):
            for schema_limit in range(schema_max, min_schema - 1, -1):
                system = brain_system_prompt(
                    catalogue_entries(_index, ctx.registry, schema_names[:schema_limit]),
                    capability_directory=directory if dynamic else None,
                )
                if getattr(agent, "use_specialists", False):
                    system += (
                        "\n12b. If the task needs several browser steps that depend on what "
                        "the page shows (searching, opening a result, reading a page, playing "
                        "a video), return request_type 'browse' with a goal and no actions. "
                        "Simple single actions stay 'action'."
                    )
                user = brain_user_prompt(
                    state.get("user_input") or "",
                    memory_context=memory_all[:memory_limit],
                    clarification_question=state.get("clarification_question"),
                    clarification_answer=state.get("clarification_answer"),
                    repair_error=state.get("error")
                    if int(state.get("validate_attempts") or 0) > 0
                    else None,
                    source=source,
                    clarification_count=int(state.get("clarification_count") or 0),
                    session_context=_session_block(state, ctx),
                    clarification_history=state.get("clarification_history"),
                    sites=sites,
                    referent_summary=referents,
                    conversation_history=_history_block(state, ctx, history_limit),
                )
                tokens = estimate_tokens(system) + estimate_tokens(user)
                actions = _trim_actions(
                    history_max, history_limit, memory_max, memory_limit, schema_max, schema_limit
                )
                stats = ContextStats(
                    estimated_input_tokens=tokens,
                    budget_tokens=effective,
                    candidate_tools=len(candidates),
                    schema_tools=schema_limit,
                    memory_candidates=memory_limit,
                    history_turns=history_limit,
                    trimmed=bool(actions),
                    trim_actions=actions,
                    fits=tokens <= effective,
                )
                last_system, last_user, last_stats = system, user, stats
                if stats.fits:
                    return BrainContext(
                        system=system,
                        user=user,
                        stats=stats,
                        candidate_names=tuple(candidates),
                        schema_names=tuple(schema_names[:schema_limit]),
                    )
    return BrainContext(
        system=last_system,
        user=last_user,
        stats=last_stats,
        candidate_names=tuple(candidates),
        schema_names=tuple(schema_names[: min_schema or 0]),
    )


def _session_block(state: dict[str, Any], ctx: Any) -> str | None:
    """Resolve a deictic reference to a prompt block (code decides, not the LLM)."""
    from jarvis.agent.session_context import SESSION_THREAD, render_resolution

    text = state.get("user_input") or ""
    answer = state.get("clarification_answer")
    if answer:
        text = f"{text}\n{answer}"
    try:
        return render_resolution(ctx.session_context.reference_resolution(SESSION_THREAD, text))
    except Exception:  # noqa: BLE001 - reference context must never block a task
        return None


def compose_replan_context(
    state: dict[str, Any],
    ctx: Any,
    *,
    toolcalls: list[dict[str, object]],
    goal: str,
) -> ReplanContext:
    """Compose a bounded replanner prompt with compact capabilities."""
    agent = _agent_settings(ctx)
    index = _index_for(ctx)
    dynamic = bool(getattr(agent, "prompt_trim_enabled", True))
    failing = [str(call.get("tool") or "") for call in toolcalls if call.get("tool")]
    query = " ".join([str(state.get("user_input") or ""), goal, *(failing[-2:])])
    if dynamic:
        names = (
            list(index.names())
            if index.count() <= SMALL_REGISTRY_LIMIT
            else discover_candidates(
                index,
                query,
                limit=int(getattr(agent, "tool_candidate_limit", 5) or 5),
            )
        )
        limit = max(1, int(getattr(agent, "tool_schema_limit", 4) or 4))
        schema_names = names[:limit]
        directory = render_capability_directory(index)
    else:
        schema_names = list(index.names())
        directory = None
    memory_all = _memory(state, len(state.get("memory_context") or []))
    memory_max = min(3, len(memory_all))
    schema_max = len(schema_names)
    observation_max = min(_OBSERVATION_LEVELS[0], len(toolcalls))
    effective = _effective_budget(agent)
    repair_error = state.get("error") if state.get("pending_replan") else None
    last_system = ""
    last_user = ""
    last_stats = ContextStats(0, effective, fits=False)
    # ``range(0, 0, -1)`` is empty; a replan with no relevant candidate schema
    # must still build one prompt (with the compact directory alone).
    schema_levels = list(range(schema_max, 0, -1)) or [0]
    for memory_limit in range(memory_max, -1, -1):
        for schema_limit in schema_levels:
            for observation_limit in _levels(observation_max, _OBSERVATION_LEVELS):
                system = replan_system_prompt(
                    catalogue_entries(index, ctx.registry, schema_names[:schema_limit]),
                    capability_directory=directory if dynamic else None,
                )
                user = replan_user_prompt(
                    user_input=state.get("user_input") or "",
                    goal=goal,
                    memory_context=memory_all[:memory_limit],
                    toolcalls=toolcalls[:observation_limit],
                    repair_error=repair_error,
                )
                tokens = estimate_tokens(system) + estimate_tokens(user)
                actions = _trim_actions(0, 0, memory_max, memory_limit, schema_max, schema_limit)
                if observation_limit < observation_max:
                    actions = (*actions, f"observations:{observation_max}->{observation_limit}")
                stats = ContextStats(
                    estimated_input_tokens=tokens,
                    budget_tokens=effective,
                    candidate_tools=len(schema_names),
                    schema_tools=schema_limit,
                    memory_candidates=memory_limit,
                    history_turns=0,
                    trimmed=bool(actions),
                    trim_actions=actions,
                    fits=tokens <= effective,
                )
                last_system, last_user, last_stats = system, user, stats
                if stats.fits:
                    return ReplanContext(
                        system=system,
                        user=user,
                        stats=stats,
                        schema_names=tuple(schema_names[:schema_limit]),
                    )
    return ReplanContext(
        system=last_system, user=last_user, stats=last_stats, schema_names=tuple(schema_names)
    )
