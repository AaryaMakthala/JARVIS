"""SecretStore behaviour against a fake keyring backend."""

from __future__ import annotations

from typing import ClassVar

import keyring
import pytest
from keyring.backend import KeyringBackend

from jarvis.secrets import (
    REDACTABLE_NAMES,
    SECRET_NAMES,
    SECRET_SERVICE_NAME,
    SecretStore,
    SecretStoreError,
    check_store_access,
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
        "gemini_api_key",
        "tavily_api_key",
        "ipc_token",
        "password_hash",
        "virustotal_api_key",
    }


SET_OF_NORMAL_NAMES = set(SECRET_NAMES)
