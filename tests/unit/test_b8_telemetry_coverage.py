"""B8.5 coverage: every real LLM caller attributes its attempt to its task.

The first B8 pass wired attempt telemetry into the brain call only.  This file
covers the remaining real provider calls - the replanner, the browser
specialist, ``web_answer``, the WhatsApp summariser and WhatsApp contact
grounding - plus the negative cases (no task metadata, unreported tokens).

Everything here is mocked: a scripted client and a temporary SQLite task
store, with no network access.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from jarvis.agent.context import AppContext, make_app_context
from jarvis.agent.nodes.replan import replan as replan_step
from jarvis.agent.nodes.specialist import decide as specialist_decide
from jarvis.agent.schemas import ReplanDecision, SpecialistDecision
from jarvis.agent.state import Plan
from jarvis.config import LLMSettings, ProviderModels, Settings
from jarvis.llm.client import CallMeta, Usage
from jarvis.llm.models import model_spec
from jarvis.llm.multi import MultiProviderClient
from jarvis.llm.provider import ProviderInfo
from jarvis.llm.telemetry import LLMAttemptMeta, build_attempt_meta
from jarvis.messaging import grounding
from jarvis.taskstore import TaskEventType, TaskStore
from jarvis.taskstore.runtime import open_runtime_task_store
from jarvis.tools import web, whatsapp_web

PLANNER = "openai/gpt-oss-120b"
FAST = "openai/gpt-oss-20b"


class ScriptedClient:
    """Provider client with a scripted answer that records the role asked.

    ``supports_attempt_meta`` is deliberately absent: the composite client
    advertises it, and this inner client is exactly the "real provider" the
    attempt is attributed to.
    """

    def __init__(self, payload: BaseModel | None = None, *, report_tokens: bool = True) -> None:
        self.payload = payload
        self.report_tokens = report_tokens
        self.roles: list[str] = []
        self.calls: list[dict[str, Any]] = []
        self.last_call = CallMeta(provider="groq", model=PLANNER, role="")

    def resolve_model(self, role: str) -> str:
        return PLANNER

    def _usage(self) -> Usage:
        if not self.report_tokens:
            return Usage(calls=1)
        return Usage(calls=1, prompt_tokens=4, completion_tokens=2)

    def structured(self, **kwargs: Any) -> tuple[BaseModel, Usage]:
        self.roles.append(str(kwargs.get("model_role", "")))
        self.calls.append(kwargs)
        self.last_call = CallMeta(
            provider="groq", model=PLANNER, role=str(kwargs.get("model_role", "")), ok=True
        )
        assert self.payload is not None, "this test expected a structured payload"
        return self.payload, self._usage()

    def text(self, **kwargs: Any) -> tuple[str, Usage]:
        self.roles.append(str(kwargs.get("model_role", "")))
        self.calls.append(kwargs)
        return "Answer [1]", self._usage()


def _settings() -> Settings:
    return Settings(
        llm=LLMSettings(
            provider_order=["groq"],
            strict_zero_cost=False,
            free_only=True,
            models={"groq": ProviderModels(planner=PLANNER, fast=FAST)},
        )
    )


def _runtime(tmp_path: Path, client: ScriptedClient) -> tuple[TaskStore, AppContext]:
    settings = _settings()
    multi = MultiProviderClient(
        [(ProviderInfo(name="groq", model=PLANNER), model_spec("groq", PLANNER), client)]
    )
    ctx = make_app_context(settings, llm=multi)
    store = open_runtime_task_store(settings, ctx.registry, path=tmp_path / "task_events.db")
    ctx.tasks = store
    return store, ctx


def _llm_events(store: TaskStore, task_id: str) -> list[dict[str, Any]]:
    return [
        event.detail
        for event in store.events(task_id)
        if event.event_type is TaskEventType.LLM_CALL
    ]


def _new_task(store: TaskStore) -> str:
    task_id = str(uuid4())
    store.ensure_task(task_id, source="terminal")
    return task_id


def _search_stub(query: str) -> list[web.SearchResult]:
    del query
    return [web.SearchResult(title="T", url="https://example.com/a", snippet="S")]


def test_replan_node_records_its_provider_attempt(tmp_path: Path) -> None:
    store, ctx = _runtime(
        tmp_path, ScriptedClient(ReplanDecision(action="stop", message="cannot continue"))
    )
    task_id = _new_task(store)
    state: dict[str, Any] = {
        "task_id": task_id,
        "user_input": "open the file",
        "plan": Plan(kind="tool", goal="open the file"),
        "pending_replan": True,
        "replan_count": 0,
        "results": [],
        "source": "terminal",
    }

    update = replan_step(state, ctx)

    assert update  # the replanner produced a decision (or a visible halt)
    details = _llm_events(store, task_id)
    assert len(details) == 1
    assert details[0]["role"] == "planner"
    assert details[0]["model"] == PLANNER
    assert details[0]["outcome"] == "success"
    assert details[0]["task_class"] == "action"


def test_specialist_node_records_its_provider_attempt(tmp_path: Path) -> None:
    store, ctx = _runtime(
        tmp_path, ScriptedClient(SpecialistDecision(kind="done", summary="Done."))
    )
    task_id = _new_task(store)
    state: dict[str, Any] = {
        "task_id": task_id,
        "specialist_goal": "browse example.com",
        "specialist_calls": 0,
        "specialist_history": [],
    }

    update = specialist_decide(state, ctx)

    assert update.get("final_answer") == "Done."
    details = _llm_events(store, task_id)
    assert len(details) == 1
    assert details[0]["role"] == "planner"


def test_web_answer_records_its_provider_attempt(tmp_path: Path, monkeypatch: Any) -> None:
    store, ctx = _runtime(tmp_path, ScriptedClient())
    monkeypatch.setattr(web, "_search_ddgs", _search_stub)
    task_id = _new_task(store)
    tool_ctx = ctx.tool_context(attempt_meta=build_attempt_meta(ctx.llm, task_id, "action", store))

    result = web._run_web_answer(web.WebAnswerArgs(question="what is it?"), tool_ctx)

    assert result.ok is True
    details = _llm_events(store, task_id)
    assert len(details) == 1
    assert details[0]["role"] == "fast"
    assert details[0]["task_class"] == "action"


def test_whatsapp_summariser_records_its_provider_attempt(tmp_path: Path) -> None:
    store, ctx = _runtime(tmp_path, ScriptedClient())
    task_id = _new_task(store)
    tool_ctx = ctx.tool_context(attempt_meta=build_attempt_meta(ctx.llm, task_id, "action", store))
    messages = (
        SimpleNamespace(
            sequence=0, sender="Rahul", timestamp="10:00", is_outgoing=False, text="hello"
        ),
    )

    summary = whatsapp_web._summarize_messages("Rahul", messages, tool_ctx)

    assert summary == "Answer [1]"
    assert len(_llm_events(store, task_id)) == 1


def test_contact_grounding_forwards_attempt_metadata(monkeypatch: Any) -> None:
    """The grounding completion is a provider call on this task, not an orphan."""

    captured: dict[str, Any] = {}

    monkeypatch.setattr(grounding, "_deterministic", lambda store, query: (None, ()))

    def _fake_interpret(
        store: Any,
        query: str,
        llm: Any,
        *,
        clarification_answer: str = "",
        attempt_meta: Any = None,
    ) -> tuple[Any, None, tuple[str, ...], int, int]:
        del store, query, llm, clarification_answer
        captured["meta"] = attempt_meta
        return SimpleNamespace(name="Rahul"), None, (), 1, 0

    monkeypatch.setattr(grounding, "_interpret", _fake_interpret)
    step = SimpleNamespace(tool=min(grounding.GROUNDABLE_TOOLS), args={"contact": "my brother"})
    meta = LLMAttemptMeta(task_id=str(uuid4()))

    outcome = grounding.ground_contacts(
        [step], store=SimpleNamespace(), llm=object(), attempt_meta=meta
    )

    assert step.args["contact"] == "Rahul"
    assert captured["meta"] is meta
    assert outcome.clarification is None
    assert outcome.halted_reason is None


def test_tool_without_task_metadata_records_no_event(tmp_path: Path, monkeypatch: Any) -> None:
    """No attempt metadata (a bare tool context) means no invented task event."""

    store, ctx = _runtime(tmp_path, ScriptedClient())
    monkeypatch.setattr(web, "_search_ddgs", _search_stub)
    task_id = _new_task(store)

    result = web._run_web_answer(web.WebAnswerArgs(question="what is it?"), ctx.tool_context())

    assert result.ok is True
    assert _llm_events(store, task_id) == []


def test_two_concurrent_tasks_keep_their_attempts_apart(tmp_path: Path, monkeypatch: Any) -> None:
    store, ctx = _runtime(tmp_path, ScriptedClient())
    monkeypatch.setattr(web, "_search_ddgs", _search_stub)
    first, second = _new_task(store), _new_task(store)

    for task_id in (first, second):
        tool_ctx = ctx.tool_context(
            attempt_meta=build_attempt_meta(ctx.llm, task_id, "action", store)
        )
        assert web._run_web_answer(web.WebAnswerArgs(question="q"), tool_ctx).ok is True

    assert len(_llm_events(store, first)) == 1
    assert len(_llm_events(store, second)) == 1


def test_local_ocr_requires_no_cloud_vision_capability() -> None:
    """Image text extraction is local; the selector never demands a vision model.

    ``route_request(..., has_image=True)`` classifies the request as
    ``local_ocr`` and keeps the ordinary ``planner`` role, and neither that task
    class nor the planner requirement asks for vision - so local OCR can never
    be redirected to a cloud vision model by the selector.  The vision
    capability is only demanded by the explicit ``vision`` role, which the OCR
    path never requests.
    """
    from jarvis.llm.capability import (
        PLANNER_REQUIREMENT,
        requirement_for_role,
        requirement_for_task_class,
    )
    from jarvis.llm.routing import route_request

    route = route_request("read the text in this image", has_image=True)
    assert route.task_class.value == "local_ocr"
    assert route.model_role == "planner"
    assert requirement_for_task_class(route.task_class).vision is False
    assert requirement_for_task_class(route.task_class).structured is False
    assert PLANNER_REQUIREMENT.vision is False
    assert requirement_for_role("vision").vision is True


def test_unreported_tokens_are_omitted_not_invented(tmp_path: Path, monkeypatch: Any) -> None:
    store, ctx = _runtime(tmp_path, ScriptedClient(report_tokens=False))
    monkeypatch.setattr(web, "_search_ddgs", _search_stub)
    task_id = _new_task(store)
    tool_ctx = ctx.tool_context(attempt_meta=build_attempt_meta(ctx.llm, task_id, "action", store))

    assert web._run_web_answer(web.WebAnswerArgs(question="q"), tool_ctx).ok is True

    detail = _llm_events(store, task_id)[0]
    assert "prompt_tokens" not in detail
    assert "completion_tokens" not in detail
    assert "duration_ms" in detail
