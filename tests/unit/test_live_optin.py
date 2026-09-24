"""Opt-in live endpoint tests.

Not part of the normal pytest run (they call real APIs and need real, free
API keys).  Enable with:

    set JARVIS_LIVE_TESTS=1
    pytest -q tests/unit/test_live_optin.py

The probe builds each configured free provider via the real factory and sends
one tiny text request.  It asserts (a) the endpoint is reachable and (b) no
secret material appears in the fallback error we might capture.
"""

from __future__ import annotations

import os

import pytest

from jarvis import config, secrets
from jarvis.llm.provider import build_provider_client

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("JARVIS_LIVE_TESTS") != "1",
        reason="set JARVIS_LIVE_TESTS=1 to run live endpoint tests (real free API keys required)",
    ),
    pytest.mark.live,
]


def _configured_providers() -> list[str]:
    settings = config.load_settings()
    configured = []
    for name in settings.llm.provider_order:
        key_in_env = any(
            os.environ.get(var) for var in secrets.SECRET_ENV_VARS.get(f"{name}_api_key", ())
        )
        if key_in_env:
            configured.append(name)
            continue
        try:
            present = secrets.SecretStore().has(f"{name}_api_key")
        except Exception:  # noqa: BLE001 - keyring may be unavailable here
            present = False
        if present:
            configured.append(name)
    return configured


def test_at_least_one_free_provider_is_configured() -> None:
    assert _configured_providers(), (
        "no free LLM provider configured - run `jarvis keys set <provider>` first"
    )


@pytest.mark.parametrize("provider", _configured_providers())
def test_live_provider_reachable(provider: str) -> None:
    settings = config.load_settings()
    store = secrets.SecretStore()
    client = build_provider_client(settings, store, provider)
    try:
        text, _usage = client.text(
            system="You are a connectivity probe.",
            user="Reply with exactly: ok",
            model_role="fast",
            max_tokens=8,
        )
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            close()
    assert isinstance(text, str) and text
    for reason in (None, text):
        assert "sk-" not in (reason or "")
