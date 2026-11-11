"""Capability-aware model selection (B8.1).

The pricing registry (:mod:`jarvis.llm.models`) says whether a model may be
used at all.  This module answers the second question: may *this model* serve
*this job*?  It never looks at a model's name to guess what it can do - only at
the :class:`~jarvis.llm.models.Capabilities` recorded for the registered
``(provider, model)`` pair.  A model whose capabilities are unknown (an
unregistered pair) is treated as **unsupported**, never as "probably fine".

Requirements are small, declarative, and closed:

* ``fast`` conversation needs structured output only (the brain always returns
  a typed :class:`~jarvis.agent.schemas.BrainDecision`);
* ``planner`` reasoning and tool-argument generation need structured output
  *and* tool planning;
* ``vision`` image understanding needs vision support.

There is deliberately no separate "summarisation" or "OCR" role: local OCR is
a deterministic local tool (no cloud call at all) and summarisation is served by
the planner requirement.  Fewer roles means fewer ways to mis-select a model.

Nothing here authorises anything: the PolicyEngine still decides whether an
action may run, and :func:`jarvis.llm.models.qualify` still decides whether a
model may be billed.  This module only refuses a model that provably cannot do
the work.
"""

from __future__ import annotations

from dataclasses import dataclass

from jarvis.llm.models import ModelSpec, model_spec
from jarvis.taskstore.vocabulary import TaskRequestClass

__all__ = [
    "CONVERSATION_REQUIREMENT",
    "PLANNER_REQUIREMENT",
    "TEXT_REQUIREMENT",
    "VISION_REQUIREMENT",
    "CapabilityRequirement",
    "eligible_candidates",
    "requirement_for_role",
    "requirement_for_task_class",
    "satisfies",
]

#: Historical model-role names (``settings.llm.models.<provider>.<role>``).
_ROLES = ("fast", "planner", "vision")


@dataclass(frozen=True, slots=True)
class CapabilityRequirement:
    """What one job demands of a model.

    ``min_context_tokens`` is ``None`` when no minimum is enforced (the
    default): an unknown context window is therefore only fatal when a caller
    explicitly demands one.
    """

    structured: bool = False
    tools: bool = False
    vision: bool = False
    text: bool = False
    min_context_tokens: int | None = None

    def label(self) -> str:
        """Short, secret-free description used in skip reasons and logs."""
        parts: list[str] = []
        if self.structured:
            parts.append("structured-output")
        if self.tools:
            parts.append("tool-planning")
        if self.vision:
            parts.append("vision")
        if self.text:
            parts.append("text-completion")
        if self.min_context_tokens:
            parts.append(f"context>={self.min_context_tokens}")
        return "+".join(parts) or "text"


#: A conversational reply: typed output, no tool catalogue needed.
CONVERSATION_REQUIREMENT = CapabilityRequirement(structured=True)

#: Planning, replanning and tool-argument generation.
PLANNER_REQUIREMENT = CapabilityRequirement(structured=True, tools=True)

#: Image understanding (screen/photo descriptions).
VISION_REQUIREMENT = CapabilityRequirement(vision=True)

#: An ordinary plain-text completion (``text()``).  Registration in the pricing
#: registry is *not* treated as implicit proof that a model can answer a
#: completion: the flag is declared on every registry entry and checked here.
TEXT_REQUIREMENT = CapabilityRequirement(text=True)

#: Unknown roles fail **closed** to the planner requirement: a typo must never
#: widen what a model is trusted to do.
_ROLE_REQUIREMENTS: dict[str, CapabilityRequirement] = {
    "fast": CONVERSATION_REQUIREMENT,
    "planner": PLANNER_REQUIREMENT,
    "vision": VISION_REQUIREMENT,
}

#: Task class -> the capability its model must have.  Local OCR never leaves the
#: machine, so it demands nothing from a cloud model.
_TASK_REQUIREMENTS: dict[TaskRequestClass, CapabilityRequirement] = {
    TaskRequestClass.CONVERSATION: CONVERSATION_REQUIREMENT,
    TaskRequestClass.ACTION: PLANNER_REQUIREMENT,
    TaskRequestClass.REASONING: PLANNER_REQUIREMENT,
    TaskRequestClass.CODING: PLANNER_REQUIREMENT,
    TaskRequestClass.SUMMARIZATION: PLANNER_REQUIREMENT,
    TaskRequestClass.TOOL_ARGS: PLANNER_REQUIREMENT,
    TaskRequestClass.LOCAL_OCR: CapabilityRequirement(),
}


def requirement_for_role(role: str) -> CapabilityRequirement:
    """Capability required by a model role (unknown roles use the planner set)."""
    return _ROLE_REQUIREMENTS.get(str(role or "").strip().lower(), PLANNER_REQUIREMENT)


def requirement_for_task_class(task_class: TaskRequestClass | str) -> CapabilityRequirement:
    """Capability required by a deterministic task class (unknown -> planner)."""
    try:
        known = TaskRequestClass(task_class)
    except (TypeError, ValueError):
        return PLANNER_REQUIREMENT
    return _TASK_REQUIREMENTS.get(known, PLANNER_REQUIREMENT)


def satisfies(spec: ModelSpec, requirement: CapabilityRequirement) -> tuple[bool, str]:
    """Whether ``spec`` can serve ``requirement``.

    Returns ``(ok, reason)``; ``reason`` is empty when satisfied and otherwise a
    short, secret-free explanation.  An unregistered model (``pricing_mode ==
    "unknown"``) has unknown capabilities and is refused.
    """
    if spec.pricing_mode == "unknown":
        return False, f"{spec.provider}/{spec.model}: capabilities are not registered"
    caps = spec.capabilities
    missing: list[str] = []
    if requirement.structured and not caps.supports_structured_output:
        missing.append("structured output")
    if requirement.tools and not caps.supports_tools:
        missing.append("tool planning")
    if requirement.vision and not caps.supports_vision:
        missing.append("vision")
    if requirement.text and not caps.supports_text:
        missing.append("plain text completion")
    if requirement.min_context_tokens is not None:
        window = spec.context_tokens
        if window is None:
            missing.append("a known context window")
        elif window < requirement.min_context_tokens:
            missing.append(f"a {requirement.min_context_tokens}-token context window")
    if missing:
        return False, f"{spec.provider}/{spec.model} lacks " + ", ".join(missing)
    return True, ""


def eligible_candidates(
    candidates: list[tuple[str, str]], role: str
) -> tuple[list[tuple[str, str]], list[str]]:
    """Split ``(provider, model)`` pairs into eligible ones and skip reasons.

    ``role`` decides the requirement.  Order is preserved so the caller keeps
    the configured provider order; this function never reorders providers and
    never sees a credential.
    """
    requirement = requirement_for_role(role)
    eligible: list[tuple[str, str]] = []
    reasons: list[str] = []
    for provider, model in candidates:
        ok, reason = satisfies(model_spec(provider, model), requirement)
        if ok:
            eligible.append((provider, model))
        else:
            reasons.append(f"skipped: {reason} (needs {requirement.label()})")
    return eligible, reasons
