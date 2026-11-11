"""B8.5: real provider attempts become LLM_CALL events (safe fields only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from jarvis.llm.client import CallMeta, LLMError, LLMTransientError, Usage
from jarvis.llm.models import Capabilities, ModelSpec
from jarvis.llm.multi import MultiProviderClient
from jarvis.llm.provider import ProviderInfo
from jarvis.llm.telemetry import (
    LLMAttemptMeta,
    LLMAttemptRecord,
    TaskStoreAttemptSink,
    new_attempt_id,
    task_failure_category,
)
from jarvis.taskstore import TaskEventType, TaskStoreVocabulary, open_task_store
from jarvis.taskstore.privacy import sanitize_detail
from jarvis.taskstore.vocabulary import TaskAttemptOutcome

TASK_ID = "00000000-0000-4000-8000-0000000000aa"


class _Answer(BaseModel):
    ok: bool = True


class _Sink:
    def __init__(self) -> None:
        self.records: list[LLMAttemptRecord] = []

    def record(self, attempt: LLMAttemptRecord) -> None:
        self.records.append(attempt)


class _FakeClient:
    """Minimal client with the composite's expected surface."""

    def __init__(self, model: str, *, error: Exception | None = None) -> None:
        self.model = model
        self.error = error
        self.last_call = CallMeta(provider="", model=model, role="")

    def resolve_model(self, role: str) -> str:
        return self.model

    def structured(self, **kwargs: Any) -> tuple[BaseModel, Usage]:
        if self.error is not None:
            raise self.error
        self.last_call = CallMeta(
            provider="fake", model=self.model, role=str(kwargs.get("model_role", "")), ok=True
        )
        return _Answer(), Usage(calls=1, prompt_tokens=11, completion_tokens=7)

    def text(self, **kwargs: Any) -> tuple[str, Usage]:
        if self.error is not None:
            raise self.error
        return "ok", Usage(calls=1, prompt_tokens=3, completion_tokens=1)


def _spec(*, structured: bool = True, tools: bool = True) -> ModelSpec:
    return ModelSpec(
        "groq",
        "fake-model",
        "free_tier",
        True,
        Capabilities(
            supports_tools=tools,
            supports_structured_output=structured,
            supports_vision=False,
        ),
    )


def _candidates(*clients: tuple[str, Any], spec: ModelSpec | None = None) -> list[Any]:
    return [
        (ProviderInfo(name=name, model=client.model, slot=1), spec or _spec(), client)
        for name, client in clients
    ]


def test_successful_attempt_records_actual_provider_model_and_tokens() -> None:
    sink = _Sink()
    client = MultiProviderClient(_candidates(("groq", _FakeClient("fake-model"))))
    meta = LLMAttemptMeta(task_id=TASK_ID, task_class="conversation", sink=sink)

    client.structured(system="s", user="u", schema=_Answer, attempt_meta=meta)

    assert len(sink.records) == 1
    record = sink.records[0]
    assert (record.provider, record.model, record.role) == ("groq", "fake-model", "planner")
    assert record.outcome is TaskAttemptOutcome.SUCCESS
    assert (record.prompt_tokens, record.completion_tokens) == (11, 7)
    assert record.fallback_index == 0
    assert record.task_class == "conversation"
    assert record.failure_category == ""


def test_fallback_records_the_actual_attempted_providers_in_order() -> None:
    sink = _Sink()
    failing = _FakeClient("fake-model", error=LLMTransientError("boom"))
    healthy = _FakeClient("fake-model")
    client = MultiProviderClient(
        _candidates(("groq", failing), ("nvidia", healthy)),
    )
    meta = LLMAttemptMeta(task_id=TASK_ID, task_class="action", sink=sink)

    client.structured(system="s", user="u", schema=_Answer, attempt_meta=meta)

    assert [(r.provider, r.fallback_index) for r in sink.records] == [
        ("groq", 0),
        ("nvidia", 1),
    ]
    assert sink.records[0].outcome is TaskAttemptOutcome.ERROR
    assert sink.records[0].failure_category == "llm"
    assert sink.records[1].outcome is TaskAttemptOutcome.SUCCESS


def test_capability_skip_is_not_recorded_as_an_attempt() -> None:
    sink = _Sink()
    incapable = _FakeClient("fake-model")
    client = MultiProviderClient(
        _candidates(("groq", incapable), spec=_spec(structured=False, tools=False)),
    )
    meta = LLMAttemptMeta(task_id=TASK_ID, task_class="action", sink=sink)

    with pytest.raises(LLMError):
        client.structured(system="s", user="u", schema=_Answer, attempt_meta=meta)
    # No request was sent, so there is nothing to record.
    assert sink.records == []


def test_telemetry_failure_never_breaks_the_call() -> None:
    class _BrokenSink:
        def record(self, attempt: LLMAttemptRecord) -> None:
            raise RuntimeError("store unavailable")

    client = MultiProviderClient(_candidates(("groq", _FakeClient("fake-model"))))
    meta = LLMAttemptMeta(task_id=TASK_ID, sink=_BrokenSink())
    answer, usage = client.structured(system="s", user="u", schema=_Answer, attempt_meta=meta)
    assert answer.ok is True
    assert usage.calls == 1


def test_failure_category_mapping_stays_inside_the_taskstore_vocabulary() -> None:
    assert task_failure_category("timeout", TaskAttemptOutcome.TIMEOUT) == "timeout"
    assert task_failure_category("cancelled", TaskAttemptOutcome.CANCELLED) == "cancelled"
    assert task_failure_category("rate_limit", TaskAttemptOutcome.ERROR) == "llm"
    assert task_failure_category("auth", TaskAttemptOutcome.ERROR) == "llm"
    assert task_failure_category("", TaskAttemptOutcome.SUCCESS) == ""


def _vocabulary() -> TaskStoreVocabulary:
    return TaskStoreVocabulary(
        providers=frozenset({"groq"}),
        models=frozenset({("groq", "fake-model")}),
        tools=frozenset({"get_time"}),
        nodes=frozenset({"brain"}),
        failure_codes=frozenset({"failed"}),
        failure_categories=frozenset(
            {
                "policy",
                "validation",
                "tool",
                "llm",
                "timeout",
                "cancelled",
                "verification",
                "internal",
                "unknown",
            }
        ),
    )


@pytest.fixture
def store(tmp_path: Path):
    opened = open_task_store(tmp_path / "task_events.db", create=True, vocabulary=_vocabulary())
    opened.ensure_task(TASK_ID, source="terminal")
    try:
        yield opened
    finally:
        opened.close()


def test_task_store_sink_persists_an_llm_call_event(store: Any) -> None:
    sink = TaskStoreAttemptSink(store)
    attempt_id = new_attempt_id()
    sink.record(
        LLMAttemptRecord(
            task_id=TASK_ID,
            provider="groq",
            model="fake-model",
            role="planner",
            outcome=TaskAttemptOutcome.SUCCESS,
            attempt_id=attempt_id,
            task_class="action",
            fallback_index=1,
            duration_ms=250,
            prompt_tokens=120,
            completion_tokens=30,
        )
    )

    events = [
        event for event in store.events(TASK_ID) if event.event_type is TaskEventType.LLM_CALL
    ]
    assert len(events) == 1
    detail = events[0].detail
    assert detail["provider"] == "groq"
    assert detail["model"] == "fake-model"
    assert detail["role"] == "planner"
    assert detail["outcome"] == "success"
    assert detail["fallback_index"] == 1
    assert detail["duration_ms"] == 250
    assert detail["attempt_id"] == attempt_id
    assert detail["task_class"] == "action"
    assert (detail["prompt_tokens"], detail["completion_tokens"]) == (120, 30)


def test_task_store_sink_records_timeout_and_failure_safely(store: Any) -> None:
    sink = TaskStoreAttemptSink(store)
    sink.record(
        LLMAttemptRecord(
            task_id=TASK_ID,
            provider="groq",
            model="fake-model",
            role="fast",
            outcome=TaskAttemptOutcome.TIMEOUT,
            duration_ms=30000,
            failure_category=task_failure_category("timeout", TaskAttemptOutcome.TIMEOUT),
            timed_out=True,
        )
    )
    detail = [e for e in store.events(TASK_ID) if e.event_type is TaskEventType.LLM_CALL][-1].detail
    assert detail["outcome"] == "timeout"
    assert detail["timed_out"] is True
    assert detail["failure_category"] == "timeout"


def test_privacy_policy_rejects_prompt_like_fields_on_llm_call() -> None:
    cleaned = sanitize_detail(
        "llm_call",
        {
            "provider": "groq",
            "model": "fake-model",
            "role": "planner",
            "prompt": "secret user text",
            "tool_args": {"path": "C:/Users/a/secret.txt"},
            "response": "model output",
        },
        _vocabulary(),
    )
    assert set(cleaned) == {"provider", "model", "role"}


def test_sink_swallows_a_closed_store() -> None:
    class _Closed:
        def append_event(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("closed")

    TaskStoreAttemptSink(_Closed()).record(
        LLMAttemptRecord(
            task_id=TASK_ID,
            provider="groq",
            model="fake-model",
            role="fast",
            outcome=TaskAttemptOutcome.SUCCESS,
        )
    )
