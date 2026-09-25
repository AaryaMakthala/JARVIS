"""Provider factory: selection order, clean skip reasons, no key leaks.

``build_llm_client`` must never raise for the common bad-config cases - it
returns ``None`` client plus safe reasons so the graph halts cleanly.  The
key is read through the store protocol (keyring first, then env) and never
included in any return value.  Free-only enforcement is tested separately in
``test_freeonly.py`` / ``test_llm_models.py``.
"""

from __future__ import annotations

import typing

from jarvis.config import LLMSettings, ProviderModels, Settings
from jarvis.llm.provider import (
    LLMConfigError,
    build_llm_client,
    provider_has_credential,
    provider_status,
)
from jarvis.secrets import SecretStoreError

PLANNER = "openai/gpt-oss-120b"
FAST = "openai/gpt-oss-20b"


class MemoryKeyStore:
    """In-test ProviderKeyStore (no real keyring)."""

    def __init__(self, keys: dict[str, str] | None = None) -> None:
        self._keys = dict(keys or {})

    def get(self, name: str) -> str | None:
        return self._keys.get(name)

    def has(self, name: str) -> bool:
        return name in self._keys


class ThrowingStore(MemoryKeyStore):
    """A keyring that blows up (environmental failure is a skip reason)."""

    def get(self, name: str) -> str | None:
        raise SecretStoreError("keyring unavailable")

    def has(self, name: str) -> bool:
        raise SecretStoreError("keyring unavailable")


def _settings(
    provider_order: list[str],
    planner_model: str = PLANNER,
    fast_model: str = FAST,
    *,
    strict_zero_cost: bool = False,
) -> Settings:
    # These tests exercise *selection mechanics*; they opt out of strict
    # zero-cost mode so free-tier models (Groq) are usable.  The strict
    # policy itself is covered in test_freeonly.py / test_llm_models.py.
    return Settings(
        llm=LLMSettings(
            provider_order=provider_order,
            planner_model=planner_model,
            fast_model=fast_model,
            strict_zero_cost=strict_zero_cost,
        )
    )


def test_no_provider_configured_returns_none_cleanly() -> None:
    sel = build_llm_client(_settings([]), MemoryKeyStore())
    assert sel.client is None
    assert sel.info is None
    assert sel.reasons == []
    assert provider_status(_settings([]), MemoryKeyStore()) == []


def test_unknown_provider_is_skipped_with_reason() -> None:
    sel = build_llm_client(_settings(["mystery"]), MemoryKeyStore())
    assert sel.client is None
    assert "unknown provider" in sel.reasons[0]
    status = provider_status(_settings(["mystery"]), MemoryKeyStore())
    assert len(status) == 1
    assert status[0]["name"] == "mystery"
    assert status[0]["ok"] is False
    assert "unknown provider" in status[0]["reason"]


def test_missing_key_is_a_clean_skip_reason() -> None:
    sel = build_llm_client(_settings(["groq"]), MemoryKeyStore())
    assert sel.client is None
    assert "key not found" in sel.reasons[0]
    st = provider_status(_settings(["groq"]), MemoryKeyStore())[0]
    assert st["ok"] is False
    assert "groq_api_key" in st["reason"]


def test_gemini_skipped_when_no_key() -> None:
    sel = build_llm_client(_settings(["groq", "gemini"]), MemoryKeyStore())
    assert sel.client is None
    assert "groq" in sel.reasons[0] and "key" in sel.reasons[0]
    assert "gemini" in sel.reasons[1] and "key" in sel.reasons[1]


def test_throws_back_to_next_provider_then_none() -> None:
    sel = build_llm_client(_settings(["groq"]), ThrowingStore())
    assert sel.client is None
    assert "key" in sel.reasons[0].lower()


def test_credential_probe_returns_false_when_keyring_fails() -> None:
    assert provider_has_credential(ThrowingStore(), "groq") is False


def test_selection_with_key_and_model_builds_groq() -> None:
    store = MemoryKeyStore({"groq_api_key": "sk-test"})
    sel = build_llm_client(_settings(["groq"]), store)
    assert sel.client is not None
    assert sel.info is not None and sel.info.name == "groq"
    assert sel.reasons == []
    assert sel.info.model == PLANNER


def test_no_secret_ever_in_selection() -> None:
    store = MemoryKeyStore({"groq_api_key": "sk-hunter2"})
    sel = build_llm_client(_settings(["groq"]), store)
    blob = repr(sel)
    assert "hunter2" not in blob
    assert "sk-" not in blob
    blob_reasons = repr(sel.reasons)
    assert "sk-test" not in blob_reasons


def test_bad_first_provider_falls_back_to_good_second() -> None:
    store = MemoryKeyStore({"groq_api_key": "sk-test"})
    sel = build_llm_client(_settings(["gemini", "groq"]), store)
    assert sel.client is not None
    assert sel.info is not None and sel.info.name == "groq"
    assert len(sel.reasons) == 1
    assert "gemini" in sel.reasons[0].lower()
    assert "key not found" in sel.reasons[0]


def test_missing_models_are_reported() -> None:
    sel = build_llm_client(
        _settings(["groq"], planner_model="", fast_model=""),
        MemoryKeyStore({"groq_api_key": "k"}),
    )
    assert sel.client is None
    assert "no model configured" in sel.reasons[0]


def test_unregistered_model_is_blocked_by_free_only() -> None:
    sel = build_llm_client(
        _settings(["groq"], planner_model="some-model", fast_model="some-model"),
        MemoryKeyStore({"groq_api_key": "k"}),
    )
    assert sel.client is None
    assert "not registered" in sel.reasons[0]


def test_env_key_is_used_but_not_copied_to_keyring(
    monkeypatch: typing.Any,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk-from-env")
    store = MemoryKeyStore()  # keyring has nothing
    sel = build_llm_client(_settings(["groq"]), store)
    assert sel.client is not None
    assert sel.info is not None and sel.info.name == "groq"
    # env var read but never written into the credential store
    assert store.has("groq_api_key") is False


def test_env_key_is_used_when_keyring_read_fails(monkeypatch: typing.Any) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk-env-fallback")
    sel = build_llm_client(_settings(["groq"]), ThrowingStore())
    assert sel.client is not None
    assert sel.info is not None and sel.info.name == "groq"


def test_gemini_env_google_fallback(monkeypatch: typing.Any) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "AIza-env-key")
    settings = Settings(
        llm=LLMSettings(
            provider_order=["gemini"],
            strict_zero_cost=False,  # gemini is free-tier; not strict-eligible
            models={"gemini": ProviderModels(planner="gemini-3.8-flash", fast="gemini-3.7-flash")},
        )
    )
    sel = build_llm_client(settings, MemoryKeyStore())
    assert sel.client is not None
    assert sel.info is not None and sel.info.name == "gemini"


def test_multiple_eligible_providers_make_a_composite() -> None:
    store = MemoryKeyStore({"groq_api_key": "a", "openrouter_api_key": "b"})
    settings = Settings(
        llm=LLMSettings(
            provider_order=["groq", "openrouter"],
            strict_zero_cost=False,  # groq is free-tier; not strict-eligible
            models={
                "groq": ProviderModels(planner="openai/gpt-oss-120b", fast="openai/gpt-oss-20b"),
                "openrouter": ProviderModels(planner="openrouter/free", fast="openrouter/free"),
            },
        )
    )
    sel = build_llm_client(settings, store)
    assert sel.client is not None
    assert sel.info is not None and sel.info.name == "groq"
    # composite exposes the fallback order
    assert sel.client.providers == ["groq", "openrouter"]


def test_provider_status_report_shape() -> None:
    status = provider_status(_settings(["groq", "gemini"]), MemoryKeyStore({"groq_api_key": "k"}))
    assert status[0] == {
        "name": "groq",
        "label": "Groq",
        "model": PLANNER,
        "pricing_mode": "free_tier",
        "supports_tools": True,
        "supports_structured_output": True,
        "available": True,
        "ok": True,
        "reason": "configured",
    }
    assert status[1]["name"] == "gemini"
    assert status[1]["ok"] is False
    assert "api key" in status[1]["reason"].lower()


def test_provider_status_checks_fast_model() -> None:
    settings = Settings(
        llm=LLMSettings(
            provider_order=["groq"],
            strict_zero_cost=False,
            models={"groq": ProviderModels(planner=PLANNER, fast="unknown-fast-model")},
        )
    )
    status = provider_status(settings, MemoryKeyStore({"groq_api_key": "k"}))
    assert status[0]["ok"] is False
    assert "fast model" in status[0]["reason"]
    assert "not registered" in status[0]["reason"]


def test_provider_status_marks_free_tier_not_ok_under_strict() -> None:
    status = provider_status(
        _settings(["groq"], strict_zero_cost=True), MemoryKeyStore({"groq_api_key": "k"})
    )
    assert status[0]["ok"] is False
    assert "billing state cannot be verified" in status[0]["reason"]


def test_llm_config_error_is_raiseable_and_safe() -> None:
    err = LLMConfigError("some problem for the user")
    assert isinstance(err, RuntimeError)
    assert str(err) == "some problem for the user"


def test_legacy_per_provider_models_resolve() -> None:
    settings = Settings(
        llm=LLMSettings(
            provider_order=["groq"],
            models={
                "groq": ProviderModels(planner="openai/gpt-oss-120b", fast="openai/gpt-oss-20b")
            },
        )
    )
    assert settings.llm.model_for("groq", "planner") == "openai/gpt-oss-120b"
    assert settings.llm.model_for("groq", "fast") == "openai/gpt-oss-20b"
    # legacy fallback still works
    legacy = Settings(llm=LLMSettings(planner_model="openai/gpt-oss-120b"))
    assert legacy.llm.model_for("groq", "planner") == "openai/gpt-oss-120b"
