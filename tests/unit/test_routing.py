from __future__ import annotations

import pytest

from jarvis.agent.nodes.brain import brain
from jarvis.agent.schemas import BrainDecision
from jarvis.config import Settings
from jarvis.llm.client import FakeLLM
from jarvis.llm.routing import (
    TaskClass,
    classify_request,
    model_role_for,
    route_request,
)
from jarvis.taskstore import TaskRouteReason
from support import brain_action, make_spec, registry_with

TASK = "00000000-0000-4000-8000-0000000000bb"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", TaskClass.CONVERSATION),
        ("hello", TaskClass.CONVERSATION),
        ("thanks!", TaskClass.CONVERSATION),
        ("what is the capital of France?", TaskClass.CONVERSATION),
        ("open notepad", TaskClass.ACTION),
        ("search for cat videos", TaskClass.ACTION),
        ("send the report to Sam", TaskClass.ACTION),
        ("why does the sky look blue and explain the physics", TaskClass.REASONING),
        ("summarize this report", TaskClass.SUMMARIZATION),
        ("write a python function to sort a list", TaskClass.CODING),
        ("generate the tool arguments as JSON", TaskClass.TOOL_ARGS),
        ("read the text on my screen", TaskClass.LOCAL_OCR),
        ("what is on my screen", TaskClass.LOCAL_OCR),
        ("take a screenshot", TaskClass.ACTION),
        ("x" * 201, TaskClass.REASONING),
    ],
)
def test_classify_request(text: str, expected: TaskClass) -> None:
    assert classify_request(text) is expected


def test_has_image_routes_to_local_ocr() -> None:
    route = route_request("describe this", has_image=True)

    assert route.task_class is TaskClass.LOCAL_OCR
    assert route.model_role == "planner"
    assert route.reason_code is TaskRouteReason.HAS_IMAGE


def test_route_reason_is_deterministic() -> None:
    assert route_request("open notepad").reason_code is TaskRouteReason.ACTION
    assert route_request("hello").reason_code is TaskRouteReason.CONVERSATION
    assert route_request("summarize this").reason_code is TaskRouteReason.SUMMARIZATION
    assert route_request("write a python script").reason_code is TaskRouteReason.CODING
    assert route_request("generate tool arguments").reason_code is TaskRouteReason.TOOL_ARGUMENTS


def test_role_map_is_closed_and_conservative() -> None:
    assert model_role_for(TaskClass.CONVERSATION) == "fast"
    assert model_role_for(TaskClass.ACTION) == "planner"
    assert model_role_for(TaskClass.REASONING) == "planner"
    assert model_role_for(TaskClass.CODING) == "planner"
    assert model_role_for(TaskClass.SUMMARIZATION) == "planner"
    assert model_role_for(TaskClass.TOOL_ARGS) == "planner"
    assert model_role_for(TaskClass.LOCAL_OCR) == "planner"


def _ctx(llm: FakeLLM):
    registry = registry_with(make_spec("fake_echo", base_tier=0, record=[]))
    from jarvis.agent.context import make_app_context

    return make_app_context(Settings(), llm=llm, registry=registry)


def test_conversation_uses_the_fast_model() -> None:
    llm = FakeLLM(
        [BrainDecision(request_type="conversation", goal="greet", response_text="Hello there.")]
    )

    update = brain({"task_id": TASK, "source": "terminal", "user_input": "hello"}, _ctx(llm))

    assert update["plan"].kind == "conversation"
    assert llm.calls == [("structured", "fast", "BrainDecision")]


def test_action_uses_the_planner_model() -> None:
    llm = FakeLLM([brain_action("fake_echo", {"text": "cats"})])

    update = brain(
        {"task_id": TASK, "source": "terminal", "user_input": "search for cats"},
        _ctx(llm),
    )

    assert update["plan"].steps[0].tool == "fake_echo"
    assert llm.calls == [("structured", "planner", "BrainDecision")]


def test_reasoning_uses_the_planner_model() -> None:
    llm = FakeLLM(
        [BrainDecision(request_type="conversation", goal="explain", response_text="Because.")]
    )

    brain(
        {
            "task_id": TASK,
            "source": "terminal",
            "user_input": "why does ice float and explain the reason",
        },
        _ctx(llm),
    )

    assert llm.calls == [("structured", "planner", "BrainDecision")]
