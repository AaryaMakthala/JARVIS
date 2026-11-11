"""B8 acceptance: real graph paths, not isolated helpers.

Covers the observable B8 contract with a scripted provider, a temporary task
store and no network:

* Route A - time, date and arithmetic are answered deterministically with **no**
  model call;
* Route B - a conversational turn costs exactly one call on the ``fast`` role;
* Route C - a single-tool request is planned once, stays on the registered tool
  and never spawns an extra executor call;
* an unregistered tool proposed by the model is refused;
* every real provider attempt is persisted as an ``LLM_CALL`` event bound to
  its own task id (and two tasks keep their events apart).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from pydantic import BaseModel

from jarvis.agent.context import make_app_context
from jarvis.agent.graph import open_sqlite_checkpointer
from jarvis.agent.runner import run_task
from jarvis.agent.schemas import ActionIntent, BrainDecision
from jarvis.config import LLMSettings, ProviderModels, Settings
from jarvis.llm.client import CallMeta, Usage
from jarvis.llm.models import model_spec
from jarvis.llm.multi import MultiProviderClient
from jarvis.llm.provider import ProviderInfo
from jarvis.taskstore import TaskEventType
from jarvis.taskstore.runtime import open_runtime_task_store

PLANNER = "openai/gpt-oss-120b"
FAST = "openai/gpt-oss-20b"


class CountingClient:
    """Scripted provider client that records the role of every real call."""

    def __init__(self, decision: BaseModel) -> None:
        self.decision = decision
        self.roles: list[str] = []
        self.structured_calls = 0
        self.text_calls = 0
        self.last_call = CallMeta(provider="groq", model=PLANNER, role="")

    def resolve_model(self, role: str) -> str:
        return PLANNER

    def structured(self, **kwargs: Any) -> tuple[BaseModel, Usage]:
        self.structured_calls += 1
        self.roles.append(str(kwargs.get("model_role", "")))
        self.last_call = CallMeta(
            provider="groq",
            model=PLANNER,
            role=str(kwargs.get("model_role", "")),
            latency_ms=2,
            ok=True,
        )
        return self.decision, Usage(calls=1, prompt_tokens=5, completion_tokens=2)

    def text(self, **kwargs: Any) -> tuple[str, Usage]:
        self.text_calls += 1
        return "ok", Usage(calls=1)


def _settings() -> Settings:
    return Settings(
        llm=LLMSettings(
            provider_order=["groq"],
            strict_zero_cost=False,
            free_only=True,
            models={"groq": ProviderModels(planner=PLANNER, fast=FAST)},
        )
    )


def _multi(client: CountingClient) -> MultiProviderClient:
    spec = model_spec("groq", PLANNER)
    return MultiProviderClient([(ProviderInfo(name="groq", model=PLANNER), spec, client)])


def _conversation(text: str) -> BrainDecision:
    return BrainDecision(request_type="conversation", goal="reply", response_text=text)


def _action(tool: str, args: dict[str, Any]) -> BrainDecision:
    return BrainDecision(
        request_type="action",
        goal="do it",
        actions=[ActionIntent(tool=tool, args=args, rationale="test", expect="done")],
    )


def _run(tmp_path: Path, text: str, *, decision: BrainDecision, dry_run: bool = False):
    client = CountingClient(decision)
    ctx = make_app_context(_settings(), llm=_multi(client), dry_run=dry_run)
    saver = open_sqlite_checkpointer(str(tmp_path / "cp.db"))
    outcome = run_task(ctx, saver, text, thread_id=str(uuid4()))
    return outcome, client, ctx


def test_time_request_is_answered_locally_without_an_llm_call(tmp_path: Path) -> None:
    outcome, client, _ctx = _run(tmp_path, "what time is it", decision=_conversation("unused"))
    assert client.structured_calls == 0
    assert outcome.final_answer
    assert "unused" not in outcome.final_answer


def test_date_request_is_answered_locally_without_an_llm_call(tmp_path: Path) -> None:
    outcome, client, _ctx = _run(tmp_path, "what is today's date", decision=_conversation("unused"))
    assert client.structured_calls == 0
    assert outcome.final_answer
    assert outcome.results and outcome.results[0].ok


def test_arithmetic_uses_the_safe_local_route(tmp_path: Path) -> None:
    outcome, client, _ctx = _run(tmp_path, "what is 12 times 4", decision=_conversation("unused"))
    assert client.structured_calls == 0
    assert "48" in (outcome.final_answer or "")


def test_simple_conversation_uses_exactly_one_fast_call(tmp_path: Path) -> None:
    outcome, client, _ctx = _run(tmp_path, "hello there", decision=_conversation("Hello, sir."))
    assert client.structured_calls == 1
    assert client.roles == ["fast"]
    assert outcome.final_answer == "Hello, sir."


def test_allowlisted_app_request_uses_the_deterministic_route(tmp_path: Path) -> None:
    """Route A/C: a closed-vocabulary app open costs no model call at all."""
    outcome, client, _ctx = _run(
        tmp_path,
        "open notepad",
        decision=_conversation("unused"),
        dry_run=True,
    )
    assert client.structured_calls == 0
    plan = outcome.state.get("plan")
    assert plan is not None and plan.steps[0].tool == "open_app"
    assert outcome.results and outcome.results[0].ok is True


def test_single_tool_request_is_planned_once_and_uses_the_registry(
    tmp_path: Path,
) -> None:
    """Route C: one brain call, registered tool, validated args, no executor call."""
    outcome, client, _ctx = _run(
        tmp_path,
        "please open notepad",
        decision=_action("open_app", {"name": "notepad"}),
        dry_run=True,
    )
    assert client.structured_calls == 1
    plan = outcome.state.get("plan")
    assert plan is not None and plan.steps[0].tool == "open_app"
    assert outcome.results, "the dry run should report a planned tool result"
    assert outcome.results[0].ok is True
    assert "s1" in outcome.decisions


def test_unregistered_tool_is_refused_before_execution(tmp_path: Path) -> None:
    outcome, client, _ctx = _run(
        tmp_path,
        "do something impossible",
        decision=_action("made_up_tool", {}),
    )
    # The refusal itself may consume a repair/replan call; what matters is that
    # the unregistered tool never ran and no success is claimed.
    assert client.structured_calls >= 1
    assert outcome.results == []
    answer = (outcome.final_answer or "").lower()
    assert answer
    assert "completed" not in answer


def test_real_attempt_produces_an_llm_call_event(tmp_path: Path) -> None:
    client = CountingClient(_conversation("Hello, sir."))
    settings = _settings()
    ctx = make_app_context(settings, llm=_multi(client))
    store = open_runtime_task_store(settings, ctx.registry, path=tmp_path / "task_events.db")
    ctx.tasks = store
    saver = open_sqlite_checkpointer(str(tmp_path / "cp.db"))
    outcome = run_task(ctx, saver, "hello there", thread_id=str(uuid4()))

    events = [
        event
        for event in store.events(outcome.task_id)
        if event.event_type is TaskEventType.LLM_CALL
    ]
    assert len(events) == 1
    detail = events[0].detail
    assert detail["provider"] == "groq"
    assert detail["model"] == PLANNER
    assert detail["role"] == "fast"
    assert detail["outcome"] == "success"
    assert detail["task_class"] == "conversation"
    assert detail["prompt_tokens"] == 5
    assert detail["fallback_index"] == 0
    store.close()


def test_two_tasks_keep_their_attempts_apart(tmp_path: Path) -> None:
    settings = _settings()
    client = CountingClient(_conversation("Hello."))
    ctx = make_app_context(settings, llm=_multi(client))
    store = open_runtime_task_store(settings, ctx.registry, path=tmp_path / "task_events.db")
    ctx.tasks = store
    saver = open_sqlite_checkpointer(str(tmp_path / "cp.db"))
    first = run_task(ctx, saver, "hello there", thread_id=str(uuid4()))
    second = run_task(ctx, saver, "good morning", thread_id=str(uuid4()))

    assert first.task_id != second.task_id
    for task_id in (first.task_id, second.task_id):
        events = [
            event for event in store.events(task_id) if event.event_type is TaskEventType.LLM_CALL
        ]
        assert len(events) == 1, f"{task_id} must own exactly its own attempt"
    store.close()


@pytest.mark.parametrize("tool", ["delete_system_folder", "python_exec", "powershell"])
def test_dangerous_or_generic_tools_never_exist(tmp_path: Path, tool: str) -> None:
    outcome, _client, ctx = _run(tmp_path, "try something unsafe", decision=_action(tool, {}))
    assert tool not in ctx.registry.names()
    assert outcome.results == []
