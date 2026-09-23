"""Provider factory: selection order, clean skip reasons, no key leaks.

``build_llm_client`` must never raise for the common bad-config cases - it
returns ``None`` client plus safe reasons so the graph halts cleanly.  The key
is read through the store protocol and never included in any return value.
"""

from __future__ import annotations

from typing import Any

from jarvis.config import LLMSettings, Settings
from jarvis.llm.provider import (
    LLMConfigError,
    build_llm_client,
    provider_status,
)


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

    def has(self, name: str) -> bool:
        raise RuntimeError("keyring unavailable")


def _settings(provider_order: list[str], planner_model: str = "some-model") -> Settings:
    return Settings(llm=LLMSettings(provider_order=provider_order, planner_model=planner_model))


def test_no_provider_configured_returns_none_cleanly() -> None:
    sel = build_llm_client(_settings([]), MemoryKeyStore())
    assert sel.client is None
    assert sel.info is None
    assert sel.reasons == ()
    assert provider_status(_settings([]), MemoryKeyStore()) == []


def test_unknown_provider_is_skipped_with_reason() -> None:
    sel = build_llm_client(_settings(["mystery"]), MemoryKeyStore())
    assert sel.client is None
    assert "unknown provider" in sel.reasons[0]
    status = provider_status(_settings(["mystery"]), MemoryKeyStore())
    assert status == [
        {
            "name": "mystery",
            "label": "mystery",
            "ok": False,
            "reason": "unknown provider in provider_order",
        }
    ]


def test_missing_key_is_a_clean_skip_reason() -> None:
    sel = build_llm_client(_settings(["groq"]), MemoryKeyStore())
    assert sel.client is None
    assert "key not found" in sel.reasons[0]
    st = provider_status(_settings(["groq"]), MemoryKeyStore())[0]
    assert st["ok"] is False
    assert "groq_api_key" in st["reason"]


def test_gemini_skipped_as_not_implemented() -> None:
    sel = build_llm_client(_settings(["groq", "gemini"]), MemoryKeyStore())
    assert sel.client is None
    assert "groq" in sel.reasons[0] and "key" in sel.reasons[0]
    assert "not implemented" in sel.reasons[1]


def test_throws_back_to_next_provider_then_none() -> None:
    sel = build_llm_client(_settings(["groq"]), ThrowingStore())
    assert sel.client is None
    assert "keyring" in sel.reasons[0].lower() or "key" in sel.reasons[0].lower()


def test_selection_with_key_and_model_builds_groq() -> None:
    store = MemoryKeyStore({"groq_api_key": "sk-test"})
    sel = build_llm_client(_settings(["groq"]), store)
    assert sel.client is not None
    assert sel.info is not None and sel.info.name == "groq"
    assert sel.reasons == ()
    assert sel.info.model == "some-model"


def test_no_secret_ever_in_selection() -> None:
    store = MemoryKeyStore({"groq_api_key": "sk-hunter2"})
    sel = build_llm_client(_settings(["groq"]), store)
    blob = repr(sel)
    assert "hunter2" not in blob
    assert "sk-" not in blob


def test_bad_first_provider_falls_back_to_good_second() -> None:
    store = MemoryKeyStore({"groq_api_key": "sk-test"})
    sel = build_llm_client(_settings(["gemini", "groq"]), store)
    assert sel.client is not None
    assert sel.info is not None and sel.info.name == "groq"
    assert len(sel.reasons) == 1
    assert "Gemini" in sel.reasons[0]


def test_missing_planner_model_is_reported() -> None:
    sel = build_llm_client(
        _settings(["groq"], planner_model=""), MemoryKeyStore({"groq_api_key": "k"})
    )
    assert sel.client is None
    assert "planner_model" in sel.reasons[0]


def test_provider_status_report_shape() -> None:
    status = provider_status(_settings(["groq", "gemini"]), MemoryKeyStore({"groq_api_key": "k"}))
    assert status[0] == {
        "name": "groq",
        "label": "Groq (primary)",
        "ok": True,
        "reason": "configured",
    }
    assert status[1]["ok"] is False
    assert "not implemented" in status[1]["reason"]


def test_llm_config_error_is_raiseable_and_safe() -> None:
    err = LLMConfigError("some problem for the user")
    assert isinstance(err, RuntimeError)
    assert str(err) == "some problem for the user"
    # Convenience: factories raise this; a ProviderFactory typing guard.
    _ = [Any for _ in ()]  # keep module-level import clean for stub typing tests
