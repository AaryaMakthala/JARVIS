"""B8.1 capability-aware model selection: requirements and fail-closed gates."""

from __future__ import annotations

import pytest

from jarvis.llm import capability
from jarvis.llm.capability import CapabilityRequirement
from jarvis.llm.models import Capabilities, ModelSpec, model_spec
from jarvis.taskstore.vocabulary import TaskRequestClass


def _spec(
    *,
    structured: bool = True,
    tools: bool = True,
    vision: bool = False,
    context_tokens: int | None = None,
) -> ModelSpec:
    return ModelSpec(
        "groq",
        "test-model",
        "free_tier",
        True,
        Capabilities(
            supports_tools=tools,
            supports_structured_output=structured,
            supports_vision=vision,
        ),
        context_tokens=context_tokens,
    )


def test_role_requirements_are_closed_and_fail_closed() -> None:
    assert capability.requirement_for_role("fast") == capability.CONVERSATION_REQUIREMENT
    assert capability.requirement_for_role("planner") == capability.PLANNER_REQUIREMENT
    assert capability.requirement_for_role("vision") == capability.VISION_REQUIREMENT
    # A typo must never widen what a model is trusted to do.
    assert capability.requirement_for_role("plannerr") == capability.PLANNER_REQUIREMENT
    assert capability.requirement_for_role("") == capability.PLANNER_REQUIREMENT


def test_conversation_needs_structured_output_but_not_tools() -> None:
    text_only = _spec(structured=True, tools=False)
    assert capability.satisfies(text_only, capability.CONVERSATION_REQUIREMENT) == (True, "")
    ok, reason = capability.satisfies(text_only, capability.PLANNER_REQUIREMENT)
    assert ok is False
    assert "tool planning" in reason


def test_model_without_structured_output_is_refused_for_every_role() -> None:
    plain = _spec(structured=False, tools=False)
    ok, reason = capability.satisfies(plain, capability.CONVERSATION_REQUIREMENT)
    assert ok is False
    assert "structured output" in reason
    assert capability.satisfies(plain, capability.VISION_REQUIREMENT)[0] is False


def test_vision_is_required_only_for_the_vision_role() -> None:
    no_vision = _spec(vision=False)
    assert capability.satisfies(no_vision, capability.PLANNER_REQUIREMENT)[0] is True
    ok, reason = capability.satisfies(no_vision, capability.VISION_REQUIREMENT)
    assert ok is False
    assert "vision" in reason
    assert capability.satisfies(_spec(vision=True), capability.VISION_REQUIREMENT)[0] is True


def test_unregistered_model_has_unknown_capabilities_and_is_refused() -> None:
    unknown = model_spec("groq", "made-up-model-name")
    assert unknown.pricing_mode == "unknown"
    for requirement in (
        capability.CONVERSATION_REQUIREMENT,
        capability.PLANNER_REQUIREMENT,
        capability.VISION_REQUIREMENT,
    ):
        ok, reason = capability.satisfies(unknown, requirement)
        assert ok is False
        assert "not registered" in reason


def test_unknown_context_window_is_unsupported_when_a_minimum_is_demanded() -> None:
    requirement = CapabilityRequirement(structured=True, min_context_tokens=4096)
    ok, reason = capability.satisfies(_spec(), requirement)
    assert ok is False
    assert "known context window" in reason
    assert capability.satisfies(_spec(context_tokens=8192), requirement)[0] is True
    ok, reason = capability.satisfies(_spec(context_tokens=1024), requirement)
    assert ok is False
    assert "4096-token" in reason


def test_task_class_requirements_keep_local_ocr_off_the_network() -> None:
    assert capability.requirement_for_task_class(TaskRequestClass.LOCAL_OCR) == (
        capability.CapabilityRequirement()
    )
    assert capability.requirement_for_task_class(TaskRequestClass.TOOL_ARGS) == (
        capability.PLANNER_REQUIREMENT
    )
    assert capability.requirement_for_task_class(TaskRequestClass.CONVERSATION) == (
        capability.CONVERSATION_REQUIREMENT
    )
    assert capability.requirement_for_task_class("no-such-class") == (
        capability.PLANNER_REQUIREMENT
    )


def test_eligible_candidates_preserve_order_and_explain_skips() -> None:
    candidates = [
        ("groq", "openai/gpt-oss-20b"),
        ("openrouter", "openrouter/free"),
        ("nvidia", "unregistered-model"),
    ]
    eligible, reasons = capability.eligible_candidates(candidates, "planner")
    assert eligible == [
        ("groq", "openai/gpt-oss-20b"),
        ("openrouter", "openrouter/free"),
    ]
    assert len(reasons) == 1
    assert "unregistered-model" in reasons[0]
    assert "needs structured-output+tool-planning" in reasons[0]


@pytest.mark.parametrize("role", ["fast", "planner", "vision"])
def test_every_registered_model_declares_what_its_role_needs(role: str) -> None:
    """The registry the factory trusts must be internally consistent.

    Selection never guesses capabilities from a model name, so a registered
    model that lacks structured output or tools is unusable for planning and a
    model without vision is unusable for image work.  Every registered entry is
    expected to declare the capabilities the default roles rely on.
    """
    from jarvis.llm import models as llm_models

    requirement = capability.requirement_for_role(role)
    specs = llm_models._FREE_ONLY_MODELS.values()
    assert specs
    for spec in specs:
        # Only the Gemini flash entries are vision models; the others are
        # legitimately skipped by the vision walk.
        if role == "vision" and not spec.capabilities.supports_vision:
            continue
        ok, reason = capability.satisfies(spec, requirement)
        assert ok is True, f"{spec.provider}/{spec.model} for {role}: {reason}"
