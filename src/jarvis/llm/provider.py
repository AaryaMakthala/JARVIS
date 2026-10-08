"""Provider-neutral LLM client factory (free-only + strict zero-cost).

Selects clients from ``settings.llm.provider_order``.  For each provider, in
order, the factory:

1. ignores providers with no credential (keyring, then environment variable) -
   one client is built per configured key slot (``llm.key_slots``) that has a
   credential, slot 1 first;
2. ignores models that the pricing policy refuses - under ``free_only=true``
   paid and unknown-priced models are blocked; under ``strict_zero_cost=true``
   only ``zero_cost_endpoint`` models pass and free-tier models are blocked
   (account billing state cannot be verified).  Enforced by the
   ``llm.models`` registry - never by prompt;
3. builds a client for the first compatible provider and returns it;
4. if every provider fails, returns ``ProviderSelection(client=None,
   reasons=[...])`` with safe, human-readable skip reasons so the graph halts
   cleanly ("No free LLM provider configured").

When more than one provider is eligible the selected client is a
:class:`MultiProviderClient` that falls through to the next eligible provider
per operation - and, crucially, NEVER to a paid provider: ineligible
(paid/free-tier-under-strict/unknown) providers are excluded before this
client exists.

Secrets are never logged, returned, or serialised: a factory reads the API
key from the store and only ever says whether the key exists.

FREE-ONLY / STRICT ZERO-COST MODE
---------------------------------
JARVIS never automatically falls back to a paid model.  A configured model
whose ``pricing_mode`` is not eligible is refused here before any client is
constructed (decision matrix in ``llm/models.qualify``).  If a provider later
prices a model or drops a free endpoint, update ``llm/models.py`` - provider
selection then fails safely instead of silently switching to something that
could bill the user.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from jarvis.config import Settings
from jarvis.llm import models as llm_models
from jarvis.secrets import SECRET_ENV_VARS, SecretStoreError, slot_secret_name

__all__ = [
    "LLMConfigError",
    "ProviderInfo",
    "ProviderKeyStore",
    "ProviderSelection",
    "build_llm_client",
    "build_provider_client",
    "provider_has_credential",
    "provider_status",
]

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


class LLMConfigError(RuntimeError):
    """A provider could not be built (missing key/model, not free, ...).

    The message is safe for the user; it never contains secret material.
    """


class ProviderKeyStore(Protocol):
    """The minimal credential surface the factories need."""

    def get(self, name: str) -> str | None: ...
    def has(self, name: str) -> bool: ...


@dataclass(frozen=True)
class ProviderInfo:
    """Identity of a selected client (for logs/doctor, no secrets)."""

    name: str
    model: str  # the planner model label as configured
    slot: int = 1  # credential key slot (1 = primary; Stage 5a failover)


@dataclass
class ProviderSelection:
    """Result of a provider build attempt."""

    client: Any | None = None  # LLMClient | None
    info: ProviderInfo | None = None
    reasons: list[str] = field(default_factory=list)


ProviderFactory = Callable[[Settings, str, str, str], Any]
# factory(settings, planner_model, fast_model, api_key) -> client


class _EnvBackedKeyStore:
    """Keyring-first, environment-second credential resolution.

    Lookup order per provider: (1) Windows keyring, (2) known environment
    variable(s), (3) missing.  Environment variables are read but never
    copied into the keyring.  A keyring failure still permits an environment
    fallback; the store error is re-raised only when no fallback is present.
    """

    def __init__(self, store: ProviderKeyStore) -> None:
        self._store = store

    def get(self, name: str) -> str | None:
        store_error: SecretStoreError | None = None
        try:
            stored = self._store.get(name)
        except SecretStoreError as exc:
            stored = None
            store_error = exc
        if stored:
            return stored
        for var in SECRET_ENV_VARS.get(name, ()):
            value = os.environ.get(var)
            if value:
                return value
        if store_error is not None:
            raise store_error
        return None

    def has(self, name: str) -> bool:
        return self.get(name) is not None


def _store_from(obj: ProviderKeyStore) -> _EnvBackedKeyStore:
    if isinstance(obj, _EnvBackedKeyStore):
        return obj
    return _EnvBackedKeyStore(obj)


def _resolve_models(settings: Settings, name: str) -> tuple[str, str]:
    """Return the free (planner, fast) model names for a provider.

    A missing role falls back to the other one (single-model providers).  An
    entirely unset provider is a configuration error.
    """
    planner = settings.llm.model_for(name, "planner")
    fast = settings.llm.model_for(name, "fast")
    if not planner and not fast:
        raise LLMConfigError(
            f"no model configured for {name} - run `jarvis init` or set [llm.models.{name}]"
        )
    return (planner or fast), (fast or planner)


def _guard_free(settings: Settings, name: str, planner: str, fast: str) -> None:
    """Refuse any configured model that is not eligible under pricing policy.

    ``free_only``/``strict_zero_cost`` from settings are passed straight to
    :func:`jarvis.llm.models.qualify`; a refused model raises before any
    client is constructed, so a paid or free-tier-under-strict model can
    never reach the graph.  When the provider has a vision model configured
    (per-provider ``vision`` or the legacy ``vision_model`` field) it is
    guarded the same way, so an ineligible vision model fails the build
    closed instead of waiting until image understanding is first used.
    """
    roles = [("planner", planner), ("fast", fast)]
    vision = settings.llm.model_for(name, "vision")
    if vision:
        roles.append(("vision", vision))
    for label, model in roles:
        allowed, _spec, reason = llm_models.qualify(
            name,
            model,
            free_only=settings.llm.free_only,
            strict_zero_cost=settings.llm.strict_zero_cost,
        )
        if not allowed:
            raise LLMConfigError(f"{name} {label} model: {reason}")


def _slot_secret_names(settings: Settings, name: str) -> list[str]:
    """Ordered keyring names to try for ``name``.

    Uses ``settings.llm.key_slots[name]`` when configured (Stage 5a key
    failover, owner-edited or hinted by ``jarvis keys set --slot``), else the
    single primary name for slot 1.
    """
    configured = settings.llm.key_slots.get(name)
    if configured:
        return configured
    return [slot_secret_name(name, 1)]


def _slot_number(name: str, secret: str, index: int) -> int:
    """Physical slot for a configured secret name.

    Parsed from the ``<provider>_<N>_api_key`` convention so health/log labels
    stay stable even if the owner reorders ``key_slots``; any other name falls
    back to its 1-based position in the list.
    """
    if secret == f"{name}_api_key":
        return 1
    tail = secret.removesuffix("_api_key").removeprefix(f"{name}_")
    return int(tail) if tail.isdigit() else index


def _credential_for(store: ProviderKeyStore, name: str, secret: str) -> str:
    """Read ``secret`` (keyring, then env) or raise a safe config error.

    The raised message never contains credential material - only the key's
    name and how to set it.
    """
    try:
        api_key = _store_from(store).get(secret)
    except Exception as exc:
        raise LLMConfigError(f"could not read the {name} API key: {exc}") from exc
    if not api_key:
        raise LLMConfigError(f"{name} API key not found - run `jarvis keys set {name}`")
    return api_key


def _build_groq(settings: Settings, planner: str, fast: str, api_key: str) -> Any:
    from jarvis.llm.client import GroqClient

    return GroqClient(
        api_key,
        settings,
        planner_model=planner,
        fast_model=fast,
    )


def _build_openrouter(settings: Settings, planner: str, fast: str, api_key: str) -> Any:
    from jarvis.llm.openai_compat import OpenAICompatClient

    return OpenAICompatClient(
        api_key,
        provider="openrouter",
        base_url=_OPENROUTER_BASE_URL,
        planner_model=planner,
        fast_model=fast,
        timeout=settings.llm.timeout_seconds,
        max_retries=settings.llm.max_retries,
    )


def _build_gemini(settings: Settings, planner: str, fast: str, api_key: str) -> Any:
    from jarvis.llm.gemini import GeminiClient

    return GeminiClient(
        api_key,
        planner_model=planner,
        fast_model=fast,
        timeout=settings.llm.timeout_seconds,
        max_retries=settings.llm.max_retries,
    )


def _build_nvidia(settings: Settings, planner: str, fast: str, api_key: str) -> Any:
    from jarvis.llm.openai_compat import OpenAICompatClient

    return OpenAICompatClient(
        api_key,
        provider="nvidia",
        base_url=_NVIDIA_BASE_URL,
        planner_model=planner,
        fast_model=fast,
        timeout=settings.llm.timeout_seconds,
        max_retries=settings.llm.max_retries,
    )


_FACTORIES: dict[str, ProviderFactory] = {
    "groq": _build_groq,
    "openrouter": _build_openrouter,
    "gemini": _build_gemini,
    "nvidia": _build_nvidia,
}


def _provider_label(name: str) -> str:
    return llm_models.PROVIDER_LABELS.get(name, name)


def provider_has_credential(store: ProviderKeyStore, name: str) -> bool:
    """Whether ``name`` has a credential reachable (keyring, then env).

    Safe for ``doctor``/``keys status``: returns a boolean, never a value or
    a credential-store exception.
    """
    try:
        return _store_from(store).has(f"{name}_api_key")
    except SecretStoreError:
        return False


def build_provider_client(settings: Settings, store: ProviderKeyStore, name: str) -> Any:
    """Build the client for one named provider, or raise :class:`LLMConfigError`.

    Enforces: provider known, credential present (keyring then env), models
    configured, and free-only eligibility.  Public so ``jarvis doctor`` can
    probe one provider at a time.
    """
    if name not in _FACTORIES:
        raise LLMConfigError(f"{name}: unknown provider in provider_order")
    api_key = _credential_for(store, name, slot_secret_name(name, 1))
    planner, fast = _resolve_models(settings, name)
    _guard_free(settings, name, planner, fast)
    try:
        return _FACTORIES[name](settings, planner, fast, api_key)
    except LLMConfigError:
        raise
    except Exception as exc:
        raise LLMConfigError(f"{name} provider unavailable: {exc}") from exc


def _build_provider_slots(
    settings: Settings, store: ProviderKeyStore, name: str
) -> tuple[list[tuple[int, Any]], str]:
    """Build one client per configured key slot that actually has a key.

    Returns ``(slot, client)`` pairs plus the planner model.  A missing
    backup key is skipped as long as at least one slot works; when no slot
    has a credential, or the provider fails model resolution/free-only
    guarding, a safe :class:`LLMConfigError` is raised (credentials are
    probed before models so a missing key stays the reported reason).
    """
    keys: list[tuple[int, str]] = []
    problem = ""
    for index, secret in enumerate(_slot_secret_names(settings, name), start=1):
        try:
            keys.append((_slot_number(name, secret, index), _credential_for(store, name, secret)))
        except LLMConfigError as exc:
            problem = str(exc)
    if not keys:
        raise LLMConfigError(problem)
    planner, fast = _resolve_models(settings, name)
    _guard_free(settings, name, planner, fast)
    built: list[tuple[int, Any]] = []
    for slot, api_key in keys:
        try:
            built.append((slot, _FACTORIES[name](settings, planner, fast, api_key)))
        except LLMConfigError as exc:
            problem = str(exc)
        except Exception as exc:  # noqa: BLE001 - a client constructor may fail any way
            problem = f"{name} provider unavailable: {exc}"
    if not built:
        raise LLMConfigError(problem)
    return built, planner


def build_llm_client(
    settings: Settings,
    store: ProviderKeyStore,
    *,
    logger: logging.Logger | None = None,
) -> ProviderSelection:
    """Build the first usable free client by ``provider_order``.

    Returns ``ProviderSelection(client=None, reasons=[...])`` when no free
    provider is available; each reason describes one skip and is safe to
    show.  When several providers qualify - or one provider has several key
    slots - the returned client is a :class:`MultiProviderClient` that falls
    through candidates per operation.
    """
    reasons: list[str] = []
    candidates: list[tuple[ProviderInfo, llm_models.ModelSpec, Any]] = []

    for name in settings.llm.provider_order:
        if name not in _FACTORIES:
            reasons.append(f"{name}: unknown provider in provider_order")
            continue
        try:
            built, planner = _build_provider_slots(settings, store, name)
        except LLMConfigError as exc:
            reasons.append(f"{name}: {exc}")
            continue
        spec = llm_models.model_spec(name, planner)
        for slot, client in built:
            candidates.append((ProviderInfo(name=name, model=planner, slot=slot), spec, client))
        if logger is not None:
            logger.info(
                "llm provider available name=%s slots=%s model=%s mode=%s",
                name,
                ",".join(str(slot) for slot, _client in built),
                planner,
                spec.pricing_mode,
            )

    if not candidates:
        return ProviderSelection(client=None, info=None, reasons=reasons)

    info, _spec, single = candidates[0]
    if len(candidates) == 1:
        client: Any = single
    else:
        from jarvis.llm.multi import MultiProviderClient

        client = MultiProviderClient(
            candidates,
            logger=logger,
            rotate_on_rate_limit=settings.llm.rotate_on_rate_limit,
        )
    return ProviderSelection(client=client, info=info, reasons=reasons)


def provider_status(settings: Settings, store: ProviderKeyStore) -> list[dict[str, Any]]:
    """Diagnostic view of each provider without building a client (no SDK).

    Used by ``jarvis doctor`` (never live) to tell the user exactly what to
    fix.  Reports presence only - credentials are never echoed.
    """
    rows: list[dict[str, Any]] = []
    key_store = _store_from(store)
    for name in settings.llm.provider_order:
        label = _provider_label(name)
        if name not in _FACTORIES:
            rows.append(
                {
                    "name": name,
                    "label": label,
                    "ok": False,
                    "reason": "unknown provider in provider_order",
                    "model": "",
                    "pricing_mode": "",
                    "supports_tools": False,
                    "supports_structured_output": False,
                    "available": False,
                }
            )
            continue

        try:
            planner, fast = _resolve_models(settings, name)
        except LLMConfigError as exc:
            rows.append(
                {
                    "name": name,
                    "label": label,
                    "ok": False,
                    "reason": str(exc),
                    "model": "",
                    "pricing_mode": "",
                    "supports_tools": False,
                    "supports_structured_output": False,
                    "available": False,
                }
            )
            continue

        spec = llm_models.model_spec(name, planner)
        try:
            _guard_free(settings, name, planner, fast)
        except LLMConfigError as exc:
            allowed = False
            reason = str(exc)
        else:
            allowed = True
            reason = ""
        base = {
            "name": name,
            "label": label,
            "model": planner,
            "pricing_mode": spec.pricing_mode,
            "supports_tools": spec.capabilities.supports_tools,
            "supports_structured_output": spec.capabilities.supports_structured_output,
            "available": spec.available,
        }

        try:
            has_key = key_store.has(f"{name}_api_key")
        except Exception as exc:  # noqa: BLE001 - keyring failures are environmental
            rows.append({**base, "ok": False, "reason": f"could not probe {name} API key: {exc}"})
            continue
        if not has_key:
            rows.append(
                {
                    **base,
                    "ok": False,
                    "reason": f"{name}_api_key: API key not found - run `jarvis keys set {name}`",
                }
            )
            continue
        if not allowed:
            rows.append({**base, "ok": False, "reason": reason})
            continue
        rows.append({**base, "ok": True, "reason": "configured"})
    return rows
