"""B8 contracts: plain-text capability and task-path telemetry attribution.

Two narrow contracts are closed here:

1. Ordinary text completion is a **declared, enforced** capability, not an
   implicit consequence of appearing in the pricing registry.  Every registered
   model declares ``supports_text`` and ``text()`` skips a candidate that does
   not.
2. Every production client a task actually uses is telemetry-aware.  The
   factory always returns the composite client (even for one provider), so an
   eligible task invocation emits exactly one ``LLM_CALL`` event per provider
   attempt, fallbacks stay individually recorded, and a missing task context
   attributes nothing.

No provider is contacted: the adapters are replaced in the factory for the
duration of these tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from pydantic import BaseModel

from jarvis.config import LLMSettings, ProviderModels, Settings
from jarvis.llm import models as llm_models
from jarvis.llm import provider as llm_provider
from jarvis.llm.capability import TEXT_REQUIREMENT, satisfies
from jarvis.llm.client import CallMeta, LLMError, LLMTransientError, Usage
from jarvis.llm.multi import MultiProviderClient
from jarvis.llm.provider import ProviderInfo, build_llm_client, build_provider_client
from jarvis.llm.telemetry import build_attempt_meta
from jarvis.taskstore import TaskEventType, TaskStore, TaskStoreVocabulary, open_task_store

PLANNER = "openai/gpt-oss-120b"
FAST = "openai/gpt-oss-20b"
OPENROUTER = "openrouter/free"


class _Answer(BaseModel):
    ok: bool = True


class _FakeProvider:
    """Scripted provider client with the surface the composite walk uses."""

    def __init__(self, model: str, *, error: Exception | None = None) -> None:
        self.model = model
        self.error = error
        self.last_call = CallMeta(provider="", model=model, role="")
        self.text_calls = 0
        self.structured_calls = 0

    def resolve_model(self, role: str) -> str:
        return self.model

    def structured(self, **kwargs: Any) -> tuple[BaseModel, Usage]:
        self.structured_calls += 1
        if self.error is not None:
            raise self.error
        self.last_call = CallMeta(
            provider="", model=self.model, role=str(kwargs.get("model_role", "")), ok=True
        )
        return _Answer(), Usage(calls=1, prompt_tokens=5, completion_tokens=2)

    def text(self, **kwargs: Any) -> tuple[str, Usage]:
        self.text_calls += 1
        if self.error is not None:
            raise self.error
        self.last_call = CallMeta(
            provider="", model=self.model, role=str(kwargs.get("model_role", "")), ok=True
        )
        return "ok", Usage(calls=1, prompt_tokens=3, completion_tokens=1)


class _MemoryKeyStore:
    """In-test ProviderKeyStore (no real keyring)."""

    def __init__(self, keys: dict[str, str]) -> None:
        self._keys = dict(keys)

    def get(self, name: str) -> str | None:
        return self._keys.get(name)

    def has(self, name: str) -> bool:
        return name in self._keys


def _settings(*provider_order: str, models: dict[str, ProviderModels] | None = None) -> Settings:
    return Settings(
        llm=LLMSettings(
            provider_order=list(provider_order),
            strict_zero_cost=False,
            free_only=True,
            models=models
            or {
                "groq": ProviderModels(planner=PLANNER, fast=FAST),
                "openrouter": ProviderModels(planner=OPENROUTER, fast=OPENROUTER),
            },
        )
    )


def _vocabulary() -> TaskStoreVocabulary:
    return TaskStoreVocabulary(
        providers=frozenset({"groq", "openrouter"}),
        models=frozenset({("groq", PLANNER), ("openrouter", OPENROUTER)}),
        tools=frozenset(),
        nodes=frozenset(),
        failure_codes=frozenset({"failed"}),
        failure_categories=frozenset({"llm", "timeout", "cancelled"}),
    )


@pytest.fixture
def store(tmp_path: Path):
    opened = open_task_store(tmp_path / "task_events.db", create=True, vocabulary=_vocabulary())
    task_id = str(uuid4())
    opened.ensure_task(task_id, source="terminal")
    try:
        yield opened, task_id
    finally:
        opened.close()


def _llm_events(store: TaskStore, task_id: str) -> list[dict[str, Any]]:
    return [
        event.detail
        for event in store.events(task_id)
        if event.event_type is TaskEventType.LLM_CALL
    ]


# ── contract 1: ordinary text completion ────────────────────────────────


def test_every_registered_model_declares_plain_text_support() -> None:
    registry = llm_models._FREE_ONLY_MODELS
    assert registry, "the pricing registry must not be empty"
    for (provider, model), spec in registry.items():
        ok, why = satisfies(spec, TEXT_REQUIREMENT)
        assert ok, f"{provider}/{model}: {why}"
        assert spec.capabilities.supports_text is True


def test_text_skips_a_model_without_the_text_capability() -> None:
    spec = llm_models.ModelSpec(
        "groq",
        "fake-model",
        "free_tier",
        True,
        llm_models.Capabilities(
            supports_tools=True,
            supports_structured_output=True,
            supports_vision=False,
            supports_text=False,
        ),
    )
    client = _FakeProvider("fake-model")
    multi = MultiProviderClient([(ProviderInfo(name="groq", model="fake-model"), spec, client)])

    with pytest.raises(LLMError):
        multi.text(system="s", user="u")

    assert client.text_calls == 0
    assert client.structured_calls == 0


def test_a_model_that_cannot_do_text_is_never_a_fallback_candidate(store: Any) -> None:
    opened, task_id = store
    capable = _FakeProvider(PLANNER)
    incapable_client = _FakeProvider(PLANNER)
    incapable_spec = llm_models.ModelSpec(
        "openrouter",
        PLANNER,
        "free_tier",
        True,
        llm_models.Capabilities(
            supports_tools=True,
            supports_structured_output=True,
            supports_vision=False,
            supports_text=False,
        ),
    )
    multi = MultiProviderClient(
        [
            (ProviderInfo(name="openrouter", model=PLANNER), incapable_spec, incapable_client),
            (
                ProviderInfo(name="groq", model=PLANNER),
                llm_models.model_spec("groq", PLANNER),
                capable,
            ),
        ]
    )
    meta = build_attempt_meta(multi, task_id, "action", opened)

    text, _usage = multi.text(system="s", user="u", attempt_meta=meta)

    assert text == "ok"
    assert incapable_client.text_calls == 0
    assert capable.text_calls == 1
    # Only the real attempt is recorded; the skip is not an attempt.
    events = _llm_events(opened, task_id)
    assert len(events) == 1
    assert events[0]["provider"] == "groq"


# ── contract 2: telemetry through the production client path ────────────


def test_production_factory_returns_a_telemetry_aware_composite(
    store: Any, monkeypatch: Any
) -> None:
    opened, task_id = store
    fake = _FakeProvider(PLANNER)
    monkeypatch.setitem(llm_provider._FACTORIES, "groq", lambda *args: fake)

    selection = build_llm_client(_settings("groq"), _MemoryKeyStore({"groq_api_key": "sk-test"}))

    assert selection.client is not None
    # One provider still goes through the composite: that is the single
    # telemetry/deadline boundary, so a one-provider install cannot lose events.
    assert isinstance(selection.client, MultiProviderClient)
    assert selection.client.supports_attempt_meta is True
    meta = build_attempt_meta(selection.client, task_id, "action", opened)
    assert meta is not None

    selection.client.text(system="s", user="u", attempt_meta=meta)

    events = _llm_events(opened, task_id)
    assert len(events) == 1
    assert events[0]["provider"] == "groq"
    assert events[0]["model"] == PLANNER
    assert events[0]["role"] == "fast"


def test_production_factory_records_each_fallback_attempt_exactly_once(
    store: Any, monkeypatch: Any
) -> None:
    opened, task_id = store
    limited = _FakeProvider(PLANNER, error=LLMTransientError("rate limited"))
    healthy = _FakeProvider(OPENROUTER)
    monkeypatch.setitem(llm_provider._FACTORIES, "groq", lambda *args: limited)
    monkeypatch.setitem(llm_provider._FACTORIES, "openrouter", lambda *args: healthy)

    selection = build_llm_client(
        _settings("groq", "openrouter"),
        _MemoryKeyStore({"groq_api_key": "sk-test", "openrouter_api_key": "sk-or-test"}),
    )
    assert selection.client is not None
    meta = build_attempt_meta(selection.client, task_id, "action", opened)

    answer, _usage = selection.client.structured(
        system="s", user="u", schema=_Answer, attempt_meta=meta
    )

    assert answer.ok is True
    events = _llm_events(opened, task_id)
    assert [(e["provider"], e["fallback_index"], e["outcome"]) for e in events] == [
        ("groq", 0, "error"),
        ("openrouter", 1, "success"),
    ]
    # Each attempt once - no wrapper + provider double-count.
    assert len(events) == 2
    assert limited.structured_calls == 1
    assert healthy.structured_calls == 1


def test_missing_task_context_fabricates_no_attribution(store: Any, monkeypatch: Any) -> None:
    opened, task_id = store
    fake = _FakeProvider(PLANNER)
    monkeypatch.setitem(llm_provider._FACTORIES, "groq", lambda *args: fake)
    selection = build_llm_client(_settings("groq"), _MemoryKeyStore({"groq_api_key": "sk-test"}))
    assert selection.client is not None

    # No task id, no store, or a client that does not advertise the feature.
    assert build_attempt_meta(selection.client, "", "action", opened) is None
    assert build_attempt_meta(selection.client, task_id, "action", None) is None
    assert build_attempt_meta(_FakeProvider(PLANNER), task_id, "action", opened) is None

    selection.client.text(system="s", user="u")

    assert _llm_events(opened, task_id) == []
    assert fake.text_calls == 1


def test_direct_provider_client_stays_outside_task_telemetry(monkeypatch: Any) -> None:
    """``jarvis doctor`` probes one provider directly; it has no task to attribute."""
    fake = _FakeProvider(PLANNER)
    monkeypatch.setitem(llm_provider._FACTORIES, "groq", lambda *args: fake)

    client = build_provider_client(
        _settings("groq"), _MemoryKeyStore({"groq_api_key": "sk-test"}), "groq"
    )

    assert not getattr(client, "supports_attempt_meta", False)
    assert build_attempt_meta(client, str(uuid4()), "action", object()) is None
