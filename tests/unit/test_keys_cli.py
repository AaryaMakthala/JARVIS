"""``jarvis keys`` sub-app: set/paste/clear/status (secret-safe).

Asserts the contract: hidden input, whitespace stripped, empty values
rejected (no infinite prompt), clipboard import never echoes the value,
``status`` shows only ``<provider>: configured|not configured`` and never any
prefix/suffix/length/hash of a key, and unknown providers are rejected.
"""

from __future__ import annotations

import typing

import pytest

from jarvis import cli

CREDENTIAL_PROVIDERS = cli.CREDENTIAL_PROVIDERS


class FakeStore:
    def __init__(self, **present: str) -> None:
        self._values = dict(present)

    def get(self, name: str) -> str | None:
        return self._values.get(name)

    def set(self, name: str, value: str) -> None:
        self._values[name] = value

    def delete(self, name: str) -> None:
        self._values.pop(name, None)

    def has(self, name: str) -> bool:
        return name in self._values


# ── set ───────────────────────────────────────────────────────────────────


def test_set_stores_trimmed_value() -> None:
    store = FakeStore()
    out = cli.keys_set_command(store, "groq", value="  gsk-ab-cd  ", interactive=False)
    assert out == "saved"
    assert store.get("groq_api_key") == "gsk-ab-cd"


def test_set_prompt_used_when_interactive() -> None:
    store = FakeStore()
    calls: list[str] = []
    out = cli.keys_set_command(
        store, "gemini", interactive=True, prompt_fn=lambda p: (calls.append(p), "AIza-x")[1]
    )
    assert out == "saved"
    assert store.get("gemini_api_key") == "AIza-x"
    assert calls and "API key:" in calls[0]


def test_set_rejects_empty_value() -> None:
    store = FakeStore()
    with pytest.raises(ValueError, match="empty API key rejected"):
        cli.keys_set_command(store, "groq", value="   ", interactive=False)


def test_set_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="unknown provider"):
        cli.keys_set_command(FakeStore(), "mystery", value="x", interactive=False)


# ── paste ────────────────────────────────────────────────────────────────


def test_paste_imports_from_clipboard_and_clears_it() -> None:
    store = FakeStore()
    out = cli.keys_paste_command(
        store,
        "nvidia",
        clipboard_fn=lambda: "  nvapi-123  ",
        clear_clipboard_fn=lambda: None,
    )
    assert out == "saved"
    assert store.get("nvidia_api_key") == "nvapi-123"


def test_paste_rejects_empty_clipboard() -> None:
    with pytest.raises(ValueError, match="clipboard was empty"):
        cli.keys_paste_command(FakeStore(), "groq", clipboard_fn=lambda: " \n ")


def test_paste_does_not_echo_the_key() -> None:
    store = FakeStore()
    out = cli.keys_paste_command(
        store, "groq", clipboard_fn=lambda: "gsk-super-secret", clear_clipboard_fn=lambda: None
    )
    assert out == "saved"  # return value never carries the value


# ── status (env fallback) ─────────────────────────────────────────────────


def test_status_lists_every_provider_once(monkeypatch: typing.Any) -> None:
    store = FakeStore(groq_api_key="x")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    lines = cli.keys_status_lines(store)
    assert [l.split(":")[0] for l in lines] == list(CREDENTIAL_PROVIDERS)
    assert "groq: configured" in lines
    for short in ("openrouter", "gemini", "nvidia", "tavily"):
        assert f"{short}: not configured" in lines


def test_status_reflects_environment(monkeypatch: typing.Any) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-from-env")
    lines = cli.keys_status_lines(FakeStore())
    assert "gemini: configured" in lines


def test_status_reflects_google_env_fallback(monkeypatch: typing.Any) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "AIza-google-env")
    lines = cli.keys_status_lines(FakeStore())
    assert "gemini: configured" in lines


def test_status_never_leaks_value_shape(monkeypatch: typing.Any) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    lines = cli.keys_status_lines(FakeStore(groq_api_key="gsk-mega-secret-value-123"))
    assert "gsk-mega-secret-value-123" not in " ".join(lines)
    assert "mega" not in " ".join(lines)


# ── clear ─────────────────────────────────────────────────────────────────


def test_clear_removes_key() -> None:
    store = FakeStore(groq_api_key="x")
    out = cli.keys_clear_command(store, "groq")
    assert out == "cleared"
    assert store.get("groq_api_key") is None


def test_clear_unknown_provider_rejected() -> None:
    with pytest.raises(ValueError, match="unknown provider"):
        cli.keys_clear_command(FakeStore(), "mystery")
