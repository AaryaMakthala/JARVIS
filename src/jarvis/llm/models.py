"""Provider/model metadata: pricing classification and eligibility.

FOUR pricing modes - JARVIS must never spend money it cannot verify:

* ``zero_cost_endpoint`` - the endpoint/route itself renders the model for
  free (e.g. OpenRouter's ``:free`` route, NVIDIA's free/trial NIM build
  endpoints).  These are the ONLY models allowed under ``strict_zero_cost``.
* ``free_tier`` - the model has a documented free tier BUT the same model
  also has paid token pricing (e.g. Groq OSS, Gemini flash).  Whether the
  account/plan avoids charges cannot be verified from the API, so under
  ``strict_zero_cost`` these are refused (fail-closed, never assumed).
* ``paid`` - priced models; refused unless the user explicitly sets
  ``free_only=false`` AND ``strict_zero_cost=false``.
* ``unknown`` - not in this registry (renamed/retired models).  Always
  refused, even with the opt-out flags: JARVIS never spends money it cannot
  price.

The default escapes entitlement checks to the registry:

        model available AND pricing known
        AND (mode is zero-cost or free-tier            <- free_only=true
             OR (paid AND free_only=false AND strict_zero_cost=false))
        AND NOT (mode is free-tier AND strict_zero_cost)

``strict_zero_cost=true`` (the safe default) admits ONLY ``zero_cost_endpoint``
models.  Free-tier models remain selectable when the operator explicitly sets
``strict_zero_cost=false`` and ``free_only=true``.  This is enforced in
:func:`qualify`, which the provider factory calls before constructing any
client; no prompt is ever consulted about pricing.

Classification truthfulness: a config points at a provider *feature* (e.g.
``openrouter/free``, NVIDIA's free endpoint) rather than at a specific paid
SKU, so a provider pricing/re-branding change must be reflected here before
``jarvis doctor`` and the factory will keep agreeing.  If a provider stops
serving a model for free, set ``available=False``; ``qualify`` then fails
safely instead of silently switching to something paid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PricingMode = Literal["zero_cost_endpoint", "free_tier", "paid", "unknown"]

#: Human-readable label per pricing mode (used by `jarvis doctor`).
PRICING_LABELS: dict[PricingMode, str] = {
    "zero_cost_endpoint": "zero-cost endpoint",
    "free_tier": "free tier",
    "paid": "paid",
    "unknown": "unknown",
}

#: Provider short name -> human-readable label (used by `jarvis init`, `keys`,
#: `doctor`).  The consultative order matches the conservative free-only
#: default (zero-cost providers first); the *effective* order is always
#: ``[llm] provider_order`` in config.toml.
PROVIDER_ORDER: tuple[str, ...] = ("openrouter", "nvidia", "gemini", "groq")

PROVIDER_LABELS: dict[str, str] = {
    "openrouter": "OpenRouter",
    "nvidia": "NVIDIA",
    "gemini": "Gemini",
    "groq": "Groq",
    "tavily": "Tavily",
}


@dataclass(frozen=True)
class Capabilities:
    """What an operation may demand of a model."""

    supports_tools: bool
    supports_structured_output: bool
    supports_vision: bool


@dataclass(frozen=True)
class ModelSpec:
    """Static metadata for one provider/model pair.  No secrets here."""

    provider: str
    model: str
    pricing_mode: PricingMode
    available: bool
    capabilities: Capabilities
    note: str = ""


def _cap(tools: bool, structured: bool, vision: bool = False) -> Capabilities:
    return Capabilities(
        supports_tools=tools,
        supports_structured_output=structured,
        supports_vision=vision,
    )


#: Curated registry of provider/model pairs.
#:   * ``zero_cost_endpoint`` = the endpoint renders the model free.
#:   * ``free_tier`` = documented free tier but paid pricing also exists; the
#:     account billing state cannot be verified -> refused under strict mode.
#:   * `available=False` means the provider no longer serves the model for
#:     free -> selected callers must fail instead of switching to paid.
#:   * `supports_tools`/`supports_structured_output` gate planning operations.
#: NVIDIA entries verified 2026-09 against build.nvidia.com ("Free Endpoint:
#: Available") - see the per-entry notes.  Source: docs/09 and live provider
#: pages; re-verify with `jarvis doctor --live`.
_FREE_ONLY_MODELS: dict[tuple[str, str], ModelSpec] = {
    # OpenRouter - the :free route dynamically picks currently-free models.
    ("openrouter", "openrouter/free"): ModelSpec(
        "openrouter",
        "openrouter/free",
        "zero_cost_endpoint",
        True,
        _cap(tools=True, structured=True),
        note="OpenRouter free route; dynamic (never a hard-coded free SKU). "
        "Free models change - capabilities are re-verified per operation and "
        "a throttled/exhausted free route falls through, never to a paid route.",
    ),
    # NVIDIA - free/trial NIM build endpoints (integrate.api.nvidia.com).
    ("nvidia", "nvidia/nemotron-3-super-120b-a12b"): ModelSpec(
        "nvidia",
        "nvidia/nemotron-3-super-120b-a12b",
        "zero_cost_endpoint",
        True,
        _cap(tools=True, structured=True),
        note="NVIDIA free/trial endpoint; NVIDIA flagged deprecation on the "
        "build page in 2026-09 - verify with doctor --live before relying on it",
    ),
    ("nvidia", "nvidia/nemotron-3.5-lightning-30b-a3b"): ModelSpec(
        "nvidia",
        "nvidia/nemotron-3.5-lightning-30b-a3b",
        "zero_cost_endpoint",
        True,
        _cap(tools=True, structured=True),
        note="NVIDIA free/trial endpoint (fast); 'Free Endpoint: Available' "
        "verified on build.nvidia.com 2026-09",
    ),
    # Gemini - documented Free Tier, same model family also paid on tiered
    # plans.  Classified free_tier, never zero_cost_endpoint.
    ("gemini", "gemini-3.8-flash"): ModelSpec(
        "gemini",
        "gemini-3.8-flash",
        "free_tier",
        True,
        _cap(tools=True, structured=True),
        note="Gemini Free Tier flash model; account billing state cannot be "
        "verified -> refused under strict_zero_cost",
    ),
    ("gemini", "gemini-3.7-flash"): ModelSpec(
        "gemini",
        "gemini-3.7-flash",
        "free_tier",
        True,
        _cap(tools=True, structured=True),
        note="Gemini Free Tier flash model; account billing state cannot be "
        "verified -> refused under strict_zero_cost",
    ),
    # Groq - free tier rate limits exist but Groq also publishes paid token
    # pricing for these models.  Classified free_tier, never zero-cost.
    ("groq", "openai/gpt-oss-120b"): ModelSpec(
        "groq",
        "openai/gpt-oss-120b",
        "free_tier",
        True,
        _cap(tools=True, structured=True),
        note="Groq Free tier planner model; account billing state cannot be "
        "verified -> refused under strict_zero_cost",
    ),
    ("groq", "openai/gpt-oss-20b"): ModelSpec(
        "groq",
        "openai/gpt-oss-20b",
        "free_tier",
        True,
        _cap(tools=True, structured=True),
        note="Groq Free tier fast model; account billing state cannot be "
        "verified -> refused under strict_zero_cost",
    ),
}

# A model that is NOT listed here is treated as ``pricing_mode="unknown"`` and
# is therefore never selectable.  Eligibility must be explicit and current.

_UNKNOWN_SPEC_CACHE: dict[str, ModelSpec] = {}


def model_spec(provider: str, model: str) -> ModelSpec:
    """Return the registry entry for (provider, model).

    Unknown pairs get a synthetic ``pricing_mode="unknown"`` spec so callers
    can reason uniformly.  Unknown specs are cached (frozen/immutable).
    """
    if not model:
        raise ValueError("model name must not be empty")
    known = _FREE_ONLY_MODELS.get((provider, model))
    if known is not None:
        return known
    key = f"{provider}/{model}"
    cached = _UNKNOWN_SPEC_CACHE.get(key)
    if cached is None:
        cached = ModelSpec(
            provider,
            model,
            "unknown",
            False,
            _cap(tools=False, structured=False),
            note="model is not registered in the pricing registry",
        )
        _UNKNOWN_SPEC_CACHE[key] = cached
    return cached


def pricing_message(spec: ModelSpec) -> str:
    """One-line human-readable pricing description for a spec.

    Used by ``jarvis doctor`` so a user understands *why* a provider was
    PASSed, WARNed or skipped (e.g. "openrouter/free = zero-cost endpoint").
    """
    mode = PRICING_LABELS[spec.pricing_mode]
    if not spec.available:
        return f"model is currently unavailable on {spec.provider} (pricing {mode})"
    return f"{mode}"


def qualify(
    provider: str,
    model: str,
    *,
    free_only: bool = True,
    strict_zero_cost: bool = True,
) -> tuple[bool, ModelSpec, str]:
    """Decide whether ``model`` may be used under the current pricing policy.

    Returns ``(allowed, spec, reason)``.  ``allowed`` is ``True`` only when
    the model is registered AND currently available AND its pricing mode is
    compatible with ``free_only`` / ``strict_zero_cost`` (see module doc for
    the exact matrix).  ``reason`` is a safe, user-facing explanation when
    refused (never contains secrets); ``""`` when allowed.
    """
    spec = model_spec(provider, model)

    if spec.pricing_mode == "unknown":
        return (
            False,
            spec,
            (
                f"model {model!r} is not registered in the pricing registry; "
                "pricing mode is unknown - refusing to use it"
            ),
        )
    if not spec.available:
        return (
            False,
            spec,
            (
                f"model {model!r} is currently marked unavailable on {provider}; "
                "it is not offered for free anymore - refusing to use it"
            ),
        )

    if spec.pricing_mode == "zero_cost_endpoint":
        return True, spec, ""

    if spec.pricing_mode == "free_tier":
        if strict_zero_cost:
            return (
                False,
                spec,
                (
                    f"model {model!r} is free-tier eligible but the account billing "
                    "state cannot be verified; strict_zero_cost allows only verified "
                    "zero-cost endpoints - refusing to use it"
                ),
            )
        # Without strict mode, free_tier is as cheap as it gets and is allowed
        # under free_only.  free_only=false still permits it (free_tier < paid).
        return True, spec, ""

    # paid
    if strict_zero_cost:
        return (
            False,
            spec,
            (
                f"model {model!r} is paid (pricing_mode=paid); strict_zero_cost "
                "allows only zero-cost endpoints - refusing to use it"
            ),
        )
    if free_only:
        return (
            False,
            spec,
            (f"model {model!r} is not eligible for free-only mode (pricing_mode=paid)."),
        )
    return True, spec, ""
