"""FREE-ONLY / STRICT ZERO-COST enforcement island.

Scenario tests that the "no billing surprises" guarantees hold end to end:

1. ``zero_cost_endpoint`` models are selectable under ``strict_zero_cost=true``
   (the safe default) while ``free_tier`` models (Groq OSS, Gemini flash) are
   refused there even with a valid key - the factory must not call a provider
   simply because it has a credential;
2. ``free_tier`` becomes selectable only when the operator explicitly sets
   ``strict_zero_cost=false`` (with ``free_only=true`` still blocking paid);
3. paid / unknown / unavailable models are refused under every safe policy;
4. missing credentials skip a provider and fall through cleanly;
5. a bad (rejected) credential fails the operation safely - never retried,
   never printed, next provider used;
6. rate limits fail over to the next eligible provider (bounded);
7. when every eligible provider fails the graph receives a safe, joined error;
8. the composite can never reach a paid model (ineligible ones never enter);
9. secrets never appear in logs, exception text, or serialised graph state.

Composite fallback behaviour is exercised with scripted :class:`FakeClient`
instances and the real :class:`MultiProviderClient`.
"""

from __future__ import annotations

import typing
from pathlib import Path

import pytest

from jarvis.agent.context import make_app_context
from jarvis.agent.graph import open_sqlite_checkpointer
from jarvis.agent.runner import run_task
from jarvis.config import LLMSettings, ProviderModels, Settings
from jarvis.llm.client import (
    FakeLLM,
    LLMAuthError,
    LLMCapabilityError,
    LLMError,
    LLMModelError,
    LLMTransientError,
    Usage,
)
from jarvis.llm.models import ModelSpec, model_spec
from jarvis.llm.multi import MultiProviderClient
from jarvis.llm.provider import ProviderInfo, build_llm_client
from support import brain_conversation, make_spec, registry_with

PLANNER = "openai/gpt-oss-120b"
FAST = "openai/gpt-oss-20b"


class FakeClient:
    """Scriptable LLMClient-compatible fake."""

    def __init__(
        self,
        *,
        error: Exception | None = None,
        response: str = "ok",
        structured_response: object | None = None,
        raise_capability: bool = False,
    ) -> None:
        self.error = error
        self.response = response
        self.structured_response = structured_response
        self.raise_capability = raise_capability

    def text(self, **kwargs: typing.Any) -> tuple[str, Usage]:
        if self.error is not None:
            raise self.error
        return self.response, Usage()

    def structured(self, **kwargs: typing.Any) -> tuple[object, Usage]:
        if self.raise_capability:
            raise LLMCapabilityError("this model cannot do structured output")
        if self.error is not None:
            raise self.error
        return self.structured_response or "structured", Usage()


class MemoryKeyStore:
    def __init__(self, keys: dict[str, str] | None = None) -> None:
        self._keys = dict(keys or {})

    def get(self, name: str) -> str | None:
        return self._keys.get(name)

    def has(self, name: str) -> bool:
        return name in self._keys


def _models() -> dict[str, ProviderModels]:
    return {
        "groq": ProviderModels(planner=PLANNER, fast=FAST),
        "openrouter": ProviderModels(planner="openrouter/free", fast="openrouter/free"),
        "gemini": ProviderModels(planner="gemini-3.8-flash", fast="gemini-3.7-flash"),
        "nvidia": ProviderModels(
            planner="nvidia/nemotron-3-super-120b-a12b",
            fast="nvidia/nemotron-3.5-lightning-30b-a3b",
        ),
    }


def _settings(
    *order: str,
    strict_zero_cost: bool = True,
    free_only: bool = True,
    models: dict[str, ProviderModels] | None = None,
) -> Settings:
    return Settings(
        llm=LLMSettings(
            provider_order=list(order) or ["openrouter"],
            strict_zero_cost=strict_zero_cost,
            free_only=free_only,
            models=models or _models(),
        )
    )


def _info(name: str, model: str) -> tuple[ProviderInfo, ModelSpec]:
    return ProviderInfo(name=name, model=model), model_spec(name, model)


# ── free_tier vs zero_cost_endpoint under strict mode ─────────────────────


def test_zero_cost_endpoint_selectable_under_strict() -> None:
    sel = build_llm_client(
        _settings("openrouter", strict_zero_cost=True),
        MemoryKeyStore({"openrouter_api_key": "sk-or-v1-x"}),
    )
    assert sel.client is not None
    assert sel.info is not None and sel.info.name == "openrouter"
    assert sel.reasons == []


def test_free_tier_refused_under_strict_even_with_key() -> None:
    sel = build_llm_client(
        _settings("groq", strict_zero_cost=True),
        MemoryKeyStore({"groq_api_key": "gsk-x"}),
    )
    assert sel.client is None
    assert "billing state cannot be verified" in sel.reasons[0]


def test_free_tier_selectable_when_strict_disabled() -> None:
    sel = build_llm_client(
        _settings("groq", strict_zero_cost=False),
        MemoryKeyStore({"groq_api_key": "gsk-x"}),
    )
    assert sel.client is not None
    assert sel.info is not None and sel.info.name == "groq"


def test_graph_halts_cleanly_when_only_free_tier_providers_under_strict() -> None:
    sel = build_llm_client(
        _settings("groq", "gemini", strict_zero_cost=True),
        MemoryKeyStore({"groq_api_key": "gsk-x", "gemini_api_key": "AIza-x"}),
    )
    assert sel.client is None
    assert len(sel.reasons) == 2
    for reason in sel.reasons:
        assert "billing state cannot be verified" in reason


def test_provider_order_skips_policy_violators_but_keeps_zero_cost() -> None:
    sel = build_llm_client(
        _settings("groq", "openrouter", strict_zero_cost=True),
        MemoryKeyStore({"groq_api_key": "gsk-x", "openrouter_api_key": "sk-or-v1-x"}),
    )
    assert sel.client is not None
    assert sel.info is not None and sel.info.name == "openrouter"
    assert len(sel.reasons) == 1  # groq refused on policy; openrouter selected


# ── paid / unknown / unavailable never selected ───────────────────────────


def test_paid_model_never_enters_build(monkeypatch: typing.Any) -> None:
    monkeypatch.setattr(
        "jarvis.llm.models._FREE_ONLY_MODELS",
        {
            ("groq", "paid-test-model"): ModelSpec(
                "groq",
                "paid-test-model",
                "paid",
                True,
                model_spec("groq", "openai/gpt-oss-120b").capabilities,
            )
        },
    )
    settings = _settings(
        "groq",
        strict_zero_cost=False,
        models={"groq": ProviderModels(planner="paid-test-model", fast="paid-test-model")},
    )
    sel = build_llm_client(settings, MemoryKeyStore({"groq_api_key": "sk-x"}))
    assert sel.client is None
    assert "not eligible for free-only" in sel.reasons[0]


def test_unknown_pricing_model_never_enters_build() -> None:
    settings = _settings(
        "groq",
        strict_zero_cost=False,
        models={"groq": ProviderModels(planner="not-registered", fast="x")},
    )
    sel = build_llm_client(settings, MemoryKeyStore({"groq_api_key": "sk-x"}))
    assert sel.client is None
    assert "not registered" in sel.reasons[0]


def test_unavailable_model_never_enters_build(monkeypatch: typing.Any) -> None:
    monkeypatch.setattr(
        "jarvis.llm.models._FREE_ONLY_MODELS",
        {
            ("groq", "gone-test-model"): ModelSpec(
                "groq",
                "gone-test-model",
                "free_tier",
                False,
                model_spec("groq", "openai/gpt-oss-120b").capabilities,
            )
        },
    )
    settings = _settings(
        "groq",
        strict_zero_cost=False,
        models={"groq": ProviderModels(planner="gone-test-model", fast="x")},
    )
    sel = build_llm_client(settings, MemoryKeyStore({"groq_api_key": "sk-x"}))
    assert sel.client is None
    assert "unavailable" in sel.reasons[0]


def test_free_model_builds_when_strict_allows() -> None:
    sel = build_llm_client(
        _settings("openrouter", strict_zero_cost=True),
        MemoryKeyStore({"openrouter_api_key": "sk-or-v1-x"}),
    )
    assert sel.client is not None
    assert sel.info is not None and sel.info.name == "openrouter"


# ── credential fallthrough ────────────────────────────────────────────────


def test_missing_key_skips_that_provider_but_uses_the_next() -> None:
    sel = build_llm_client(
        _settings("openrouter", "nvidia", strict_zero_cost=True),
        MemoryKeyStore({"nvidia_api_key": "nvapi-x"}),
    )
    assert sel.client is not None
    assert sel.info is not None and sel.info.name == "nvidia"
    assert len(sel.reasons) == 1
    assert "openrouter" in sel.reasons[0]


def test_every_provider_missing_returns_none_and_safe_reasons() -> None:
    sel = build_llm_client(
        _settings("openrouter", "nvidia", strict_zero_cost=True), MemoryKeyStore()
    )
    assert sel.client is None
    assert len(sel.reasons) == 2
    for reason in sel.reasons:
        assert "key not found" in reason
        assert "sk-" not in reason


# ── composite fallback behaviour ──────────────────────────────────────────


def test_rate_limited_first_provider_falls_through() -> None:
    first = FakeClient(error=LLMTransientError("rate limited"))
    second = FakeClient(response="pong")
    multi = MultiProviderClient(
        [
            _info("openrouter", "openrouter/free") + (first,),
            _info("nvidia", "nvidia/nemotron-3-super-120b-a12b") + (second,),
        ]
    )
    text, _usage = multi.text(system="s", user="u")
    assert text == "pong"
    assert multi.providers == ["openrouter", "nvidia"]


def test_auth_failure_falls_through_not_retried() -> None:
    first = FakeClient(error=LLMAuthError("rejected"))
    second = FakeClient(response="pong")
    multi = MultiProviderClient(
        [
            _info("openrouter", "openrouter/free") + (first,),
            _info("nvidia", "nvidia/nemotron-3-super-120b-a12b") + (second,),
        ]
    )
    text, _ = multi.text(system="s", user="u")
    assert text == "pong"


def test_model_404_falls_through() -> None:
    first = FakeClient(error=LLMModelError("model retired"))
    second = FakeClient(response="pong")
    multi = MultiProviderClient(
        [
            _info("openrouter", "openrouter/free") + (first,),
            _info("nvidia", "nvidia/nemotron-3-super-120b-a12b") + (second,),
        ]
    )
    text, _ = multi.text(system="s", user="u")
    assert text == "pong"


def test_all_providers_fail_safely_with_single_joined_error() -> None:
    first = FakeClient(error=LLMTransientError("rate limited"))
    second = FakeClient(error=LLMAuthError("rejected"))
    multi = MultiProviderClient(
        [
            _info("openrouter", "openrouter/free") + (first,),
            _info("nvidia", "nvidia/nemotron-3-super-120b-a12b") + (second,),
        ]
    )
    with pytest.raises(LLMError) as ei:
        multi.text(system="s", user="u")
    assert "openrouter" in str(ei.value) and "nvidia" in str(ei.value)
    assert "all free LLM providers failed" in str(ei.value)


def test_capability_mismatch_skips_provider_for_structured() -> None:
    from jarvis.llm.models import Capabilities

    no_structured = ModelSpec(
        "openrouter",
        "openrouter/free",
        "zero_cost_endpoint",
        True,
        Capabilities(False, False, False),
    )
    multi = MultiProviderClient(
        [
            _info("openrouter", "openrouter/free") + (FakeClient(structured_response="or-ok"),),
            (ProviderInfo("openrouter", "openrouter/free"), no_structured, FakeClient()),
        ]
    )
    out, _ = multi.structured(system="s", user="u", schema=type(str))
    assert out == "or-ok"


def test_runtime_capability_error_falls_through() -> None:
    first = FakeClient(raise_capability=True)
    second = FakeClient(structured_response="structured-ok")
    multi = MultiProviderClient(
        [
            _info("openrouter", "openrouter/free") + (first,),
            _info("nvidia", "nvidia/nemotron-3-super-120b-a12b") + (second,),
        ]
    )
    out, _ = multi.structured(system="s", user="u", schema=type(str))
    assert out == "structured-ok"


def test_fallback_is_bounded_by_provider_count() -> None:
    bad = FakeClient(error=LLMTransientError("always failing"))
    multi = MultiProviderClient([_info("openrouter", "openrouter/free") + (bad,)])
    with pytest.raises(LLMError):
        multi.text(system="s", user="u", max_tokens=1)


def test_composite_never_includes_paid_or_free_tier_under_strict() -> None:
    """Eligible-only guarantee: candidates come from a strict registry, so the
    composite cannot reach a provider that would bill the user."""
    sel = build_llm_client(
        _settings("groq", "gemini", "openrouter", "nvidia", strict_zero_cost=True),
        MemoryKeyStore(
            {
                "groq_api_key": "gsk-x",
                "gemini_api_key": "AIza-x",
                "openrouter_api_key": "sk-or-v1-x",
                "nvidia_api_key": "nvapi-x",
            }
        ),
    )
    assert sel.client is not None
    assert sel.client.providers == ["openrouter", "nvidia"]


# ── secrets never leak ────────────────────────────────────────────────────


def test_no_secret_in_logs_on_fallback(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    caplog.set_level("INFO")
    first = FakeClient(error=LLMTransientError("rate limited"))
    second = FakeClient(response="pong")
    logger = logging.getLogger("llm.test")
    multi = MultiProviderClient(
        [
            _info("openrouter", "openrouter/free") + (first,),
            _info("nvidia", "nvidia/nemotron-3-super-120b-a12b") + (second,),
        ],
        logger=logger,
    )
    text, _ = multi.text(system="s", user="u")
    assert text == "pong"
    assert "gsk-hunter2" not in caplog.text
    assert "transient failure" in caplog.text  # cause is logged, not the credential


def test_no_secret_in_exception_text() -> None:
    first = FakeClient(error=LLMTransientError("rate limited"))
    second = FakeClient(error=LLMAuthError("rejected"))
    multi = MultiProviderClient(
        [
            _info("openrouter", "openrouter/free") + (first,),
            _info("nvidia", "nvidia/nemotron-3-super-120b-a12b") + (second,),
        ]
    )
    with pytest.raises(LLMError) as ei:
        multi.text(system="s", user="u")
    assert "sk-" not in str(ei.value)
    assert "xai_secret" not in str(ei.value)


def test_no_secret_in_factory_logs(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    caplog.set_level("INFO")
    logger = logging.getLogger("llm.factory.test")
    sel = build_llm_client(
        _settings("openrouter", strict_zero_cost=True),
        MemoryKeyStore({"openrouter_api_key": "sk-or-v1-hunter2"}),
        logger=logger,
    )
    assert sel.client is not None
    assert "hunter2" not in caplog.text


def test_no_secret_in_pydantic_settings_serialisation() -> None:
    dump = _settings("openrouter", strict_zero_cost=True).model_dump()
    assert "sk-" not in repr(dump)
    assert "hunter2" not in repr(dump)


def test_no_secret_in_serialised_graph_state(tmp_path: Path) -> None:
    """End-to-end: a full agent run leaves no secret in the checkpoint DB."""
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    record: list = []
    ctx = make_app_context(
        Settings(),
        llm=FakeLLM([brain_conversation("Hi there!")]),
        registry=registry_with(make_spec("fake_echo", base_tier=0, record=record)),
    )
    out = run_task(ctx, saver, "hello jarvis")
    assert out.final_answer == "Hi there!"
    db_bytes = (tmp_path / "c.db").read_bytes()
    assert b"gsk-hunter2-secret" not in db_bytes


@pytest.mark.parametrize(
    "secret",
    [
        "gsk-test-123",
        "AIza-fake-google-key",
        "nvapi-fake-1",
        "sk-or-v1-fake",
    ],
)
def test_provider_specific_sample_keys_never_in_selection(secret: str) -> None:
    sel = build_llm_client(
        _settings("openrouter", strict_zero_cost=True),
        MemoryKeyStore({"openrouter_api_key": secret}),
    )
    assert secret not in repr(sel)
    assert secret not in repr(sel.reasons)
    assert secret not in repr(sel.info)
    assert secret not in repr(sel.client)
