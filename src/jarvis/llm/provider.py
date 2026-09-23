"""Provider-neutral LLM client factory.

Selects the first configured client from ``settings.llm.provider_order`` and
builds it once.  Provider factories raise :class:`LLMConfigError` with a
user-facing reason when they cannot build; the selector records the reason,
tries the next provider, and returns ``(client, info, reasons)``.  If nothing
is configured every caller gets ``(None, None, reasons)`` so the graph halts
cleanly ("No LLM backend configured") instead of crashing.

Secrets are never logged or returned: a factory reads the API key from the
keyring and only ever says whether the key exists.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from jarvis.config import Settings

__all__ = [
    "LLMConfigError",
    "ProviderInfo",
    "ProviderKeyStore",
    "ProviderSelection",
    "build_llm_client",
    "provider_status",
]


class LLMConfigError(RuntimeError):
    """A provider could not be built (missing key/model, SDK absent, ...).

    The message is safe for the user; it never contains secret material.
    """


class ProviderKeyStore(Protocol):
    """The minimal keyring surface the factories need."""

    def get(self, name: str) -> str | None: ...
    def has(self, name: str) -> bool: ...


@dataclass(frozen=True)
class ProviderInfo:
    """Identity of the client that was built (for logs/doctor, no secrets)."""

    name: str
    model: str  # e.g. the planner model label as configured


@dataclass(frozen=True)
class ProviderSelection:
    """Result of a provider build attempt."""

    client: Any | None = None  # LLMClient | None
    info: ProviderInfo | None = None
    reasons: tuple[str, ...] = ()

    def __init__(
        self,
        client: Any | None = None,
        info: ProviderInfo | None = None,
        reasons: list[str] | None = None,
    ) -> None:
        object.__setattr__(self, "client", client)
        object.__setattr__(self, "info", info)
        object.__setattr__(self, "reasons", tuple(reasons or []))


ProviderFactory = Callable[[Settings, ProviderKeyStore], Any]


def _build_groq(settings: Settings, store: ProviderKeyStore) -> Any:
    """Build :class:`GroqClient` or raise :class:`LLMConfigError`."""
    try:
        key = store.get("groq_api_key")
    except Exception as exc:
        raise LLMConfigError(f"could not read the Groq API key: {exc}") from exc
    if not key:
        raise LLMConfigError("Groq API key not found - run `jarvis init` and set it")
    if not settings.llm.planner_model:
        raise LLMConfigError("llm.planner_model is unset - pick a model and set it in config.toml")
    try:
        from jarvis.llm.client import GroqClient

        return GroqClient(key, settings)
    except Exception as exc:
        raise LLMConfigError(f"Groq provider unavailable: {exc}") from exc


def _build_gemini(settings: Settings, store: ProviderKeyStore) -> Any:
    """Gemini is designed but not implemented yet; skip cleanly."""
    del settings, store
    raise LLMConfigError("Gemini provider is not implemented in this build yet")


_FACTORIES: dict[str, ProviderFactory] = {
    "groq": _build_groq,
    "gemini": _build_gemini,
}

_DESCRIPTIONS: dict[str, str] = {
    "groq": "Groq (primary)",
    "gemini": "Gemini (fallback, not implemented yet)",
}


def _store_from(obj: ProviderKeyStore) -> ProviderKeyStore:
    return obj


def build_llm_client(
    settings: Settings,
    store: ProviderKeyStore,
    *,
    logger: logging.Logger | None = None,
) -> ProviderSelection:
    """Build the first usable client by ``provider_order``.

    Returns ``ProviderSelection(client=None, reasons=[...])`` when no provider
    is available; ``reasons`` explain each skip and are safe to show.
    """
    reasons: list[str] = []
    for name in settings.llm.provider_order:
        factory = _FACTORIES.get(name)
        if factory is None:
            reasons.append(f"{name}: unknown provider in provider_order")
            continue
        try:
            key_store = _store_from(store)
            client = factory(settings, key_store)
        except LLMConfigError as exc:
            reasons.append(f"{name}: {exc}")
            continue
        info = ProviderInfo(name=name, model=settings.llm.planner_model or "")
        if logger is not None:
            logger.info("llm provider selected name=%s model=%s", name, info.model)
        return ProviderSelection(client=client, info=info, reasons=reasons)
    return ProviderSelection(client=None, info=None, reasons=reasons)


def provider_status(settings: Settings, store: ProviderKeyStore) -> list[dict[str, Any]]:
    """Diagnostic view of each provider without building a client (no SDK).

    Used by ``jarvis doctor`` to tell the user exactly what to fix.
    """
    rows: list[dict[str, Any]] = []
    for name in settings.llm.provider_order:
        description = _DESCRIPTIONS.get(name, name)
        if name not in _FACTORIES:
            rows.append(
                {
                    "name": name,
                    "label": description,
                    "ok": False,
                    "reason": "unknown provider in provider_order",
                }
            )
            continue
        if name == "gemini":
            rows.append(
                {
                    "name": name,
                    "label": description,
                    "ok": False,
                    "reason": "not implemented in this build yet",
                }
            )
            continue
        try:
            has_key = bool(store.has("groq_api_key"))
        except Exception:  # noqa: BLE001 - keyring failures are environmental
            has_key = False
        missing = []
        if not has_key:
            missing.append("groq_api_key")
        if not settings.llm.planner_model:
            missing.append("llm.planner_model")
        if missing:
            rows.append(
                {
                    "name": name,
                    "label": description,
                    "ok": False,
                    "reason": "missing: " + ", ".join(missing),
                }
            )
        else:
            rows.append({"name": name, "label": description, "ok": True, "reason": "configured"})
    return rows
