"""SecretStore behaviour against a fake keyring backend."""

from __future__ import annotations

from typing import ClassVar

import keyring
import pytest
from keyring.backend import KeyringBackend

from jarvis.secrets import (
    MAX_KEY_SLOTS,
    PROVIDER_SECRETS,
    REDACTABLE_NAMES,
    SECRET_ENV_VARS,
    SECRET_NAMES,
    SECRET_SERVICE_NAME,
    SecretStore,
    SecretStoreError,
    check_store_access,
    slot_secret_name,
)


class FakeBackend(KeyringBackend):
    """In-memory keyring backend standing in for Windows Credential Manager."""

    priority = 90
    _store: ClassVar[dict[str, dict[str, str]]] = {}

    def set_password(self, service: str, username: str, password: str) -> None:
        self._store.setdefault(service, {})[username] = password

    def get_password(self, service: str, username: str) -> str | None:
        return self._store.get(service, {}).get(username)

    def delete_password(self, service: str, username: str) -> None:
        self._store.get(service, {}).pop(username, None)


@pytest.fixture
def fake_keyring(monkeypatch: pytest.MonkeyPatch) -> FakeBackend:
    FakeBackend._store.clear()
    backend = FakeBackend()
    monkeypatch.setattr("jarvis.secrets.keyring.get_password", backend.get_password)
    monkeypatch.setattr("jarvis.secrets.keyring.set_password", backend.set_password)
    monkeypatch.setattr("jarvis.secrets.keyring.delete_password", backend.delete_password)
    monkeypatch.setattr("jarvis.secrets.keyring.get_keyring", lambda: backend)
    return backend


def test_set_get_delete_roundtrip(fake_keyring: FakeBackend) -> None:
    store = SecretStore()
    assert store.has("groq_api_key") is False
    store.set("groq_api_key", "gsk-abc123")
    assert store.has("groq_api_key") is True
    assert store.get("groq_api_key") == "gsk-abc123"
    store.delete("groq_api_key")
    assert store.has("groq_api_key") is False


def test_delete_missing_is_noop(fake_keyring: FakeBackend) -> None:
    store = SecretStore()
    store.delete("ipc_token")
    assert store.has("ipc_token") is False


def test_get_missing_returns_none(fake_keyring: FakeBackend) -> None:
    assert SecretStore().get("gemini_api_key") is None


def test_backend_failure_raises_secret_store_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(service: str, username: str) -> str:
        raise keyring.errors.PasswordSetError("simulated failure")

    monkeypatch.setattr("jarvis.secrets.keyring.get_password", boom)
    store = SecretStore()
    with pytest.raises(SecretStoreError):
        store.get("groq_api_key")
    with pytest.raises(SecretStoreError):
        store.has("groq_api_key")


def test_backend_failure_on_write_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "jarvis.secrets.keyring.set_password",
        lambda service, user, pwd: (_ for _ in ()).throw(keyring.errors.PasswordSetError("x")),
    )
    with pytest.raises(SecretStoreError):
        SecretStore().set("groq_api_key", "gsk-abc123")


def test_check_store_access_reports_backend(fake_keyring: FakeBackend) -> None:
    assert check_store_access() == "FakeBackend"
    assert fake_keyring.get_password(SECRET_SERVICE_NAME, "__jarvis_probe__") is None


def test_check_store_access_none_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("jarvis.secrets.keyring.get_keyring", lambda: None)
    assert check_store_access() is None


def test_secret_names_and_redactables_align() -> None:
    assert set(SECRET_NAMES) == set(REDACTABLE_NAMES)
    assert SET_OF_NORMAL_NAMES == {
        "groq_api_key",
        "openrouter_api_key",
        "gemini_api_key",
        "nvidia_api_key",
        "tavily_api_key",
        "groq_2_api_key",
        "groq_3_api_key",
        "openrouter_2_api_key",
        "openrouter_3_api_key",
        "gemini_2_api_key",
        "gemini_3_api_key",
        "nvidia_2_api_key",
        "nvidia_3_api_key",
        "tavily_2_api_key",
        "tavily_3_api_key",
        "ipc_token",
        "password_hash",
        "virustotal_api_key",
    }


def test_slot_secret_name_naming() -> None:
    assert slot_secret_name("groq", 1) == "groq_api_key"
    assert slot_secret_name("groq", 2) == "groq_2_api_key"
    assert slot_secret_name("gemini", MAX_KEY_SLOTS) == "gemini_3_api_key"
    assert slot_secret_name("tavily", 1) == "tavily_api_key"


def test_slot_secret_name_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        slot_secret_name("groq", 0)
    with pytest.raises(ValueError):
        slot_secret_name("groq", MAX_KEY_SLOTS + 1)
    with pytest.raises(ValueError):
        slot_secret_name("unknown_provider", 2)


def test_every_slot_name_is_registered_keyring_only() -> None:
    """Every composable slot name is a registered secret with no env fallback."""
    for provider in PROVIDER_SECRETS:
        for slot in range(1, MAX_KEY_SLOTS + 1):
            name = slot_secret_name(provider, slot)
            assert name in SECRET_NAMES
            if slot > 1:
                assert name not in SECRET_ENV_VARS


def test_provider_secret_map_matches_secret_names() -> None:
    from jarvis.secrets import PROVIDER_SECRETS, SECRET_NAMES

    assert set(PROVIDER_SECRETS.values()) <= set(SECRET_NAMES)
    assert PROVIDER_SECRETS["groq"] == "groq_api_key"
    assert PROVIDER_SECRETS["openrouter"] == "openrouter_api_key"
    assert PROVIDER_SECRETS["gemini"] == "gemini_api_key"
    assert PROVIDER_SECRETS["nvidia"] == "nvidia_api_key"
    assert PROVIDER_SECRETS["tavily"] == "tavily_api_key"


def test_secret_env_vars_mapping() -> None:
    from jarvis.secrets import SECRET_ENV_VARS

    assert SECRET_ENV_VARS["groq_api_key"] == ("GROQ_API_KEY",)
    assert SECRET_ENV_VARS["openrouter_api_key"] == ("OPENROUTER_API_KEY",)
    assert SECRET_ENV_VARS["gemini_api_key"] == ("GEMINI_API_KEY", "GOOGLE_API_KEY")
    assert SECRET_ENV_VARS["nvidia_api_key"] == ("NVIDIA_API_KEY",)
    assert "ipc_token" not in SECRET_ENV_VARS  # never from env


SET_OF_NORMAL_NAMES = set(SECRET_NAMES)
