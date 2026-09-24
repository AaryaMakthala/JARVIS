"""Pricing registry: classification and the free-only / strict zero-cost policy.

``qualify`` is the enforcement point the provider factory and doctor both call.
Guarantees:

* only registered + currently-available models can pass;
* ``zero_cost_endpoint`` models pass under any policy (including strict);
* ``free_tier`` models pass under ``free_only=true`` but are REFUSED under
  ``strict_zero_cost=true`` (account billing state cannot be verified) - we
  never describe a free-tier provider as a guaranteed zero-cost provider;
* ``paid`` is refused under ``free_only``/``strict`` and only allowed with an
  explicit opt-out (``free_only=false`` AND ``strict_zero_cost=false``);
* ``unknown`` pricing is always refused;
* ``unavailable`` is always refused.
"""

from __future__ import annotations

import typing

import pytest

from jarvis.llm.models import (
    _FREE_ONLY_MODELS,
    PRICING_LABELS,
    PROVIDER_LABELS,
    PROVIDER_ORDER,
    Capabilities,
    ModelSpec,
    model_spec,
    pricing_message,
    qualify,
)

_GROQ_PLANNER = "openai/gpt-oss-120b"
_GROQ_FAST = "openai/gpt-oss-20b"


def _cap(tools: bool, structured: bool) -> Capabilities:
    return Capabilities(tools, structured, supports_vision=False)


def test_default_provider_order_and_labels() -> None:
    assert PROVIDER_ORDER == ("openrouter", "nvidia", "gemini", "groq")
    assert dict(PROVIDER_LABELS)["groq"] == "Groq"
    assert dict(PROVIDER_LABELS)["nvidia"] == "NVIDIA"


def test_pricing_labels_cover_all_modes() -> None:
    for mode in ("zero_cost_endpoint", "free_tier", "paid", "unknown"):
        assert PRICING_LABELS[mode]


def test_all_default_models_are_registered() -> None:
    pairs = list(_FREE_ONLY_MODELS)
    # 7 unique provider/model pairs: openrouter(1 - fast falls back to the
    # planner), nvidia(2), gemini(2), groq(2)
    assert len(pairs) == 7
    for (provider, model), spec in sorted(_FREE_ONLY_MODELS.items()):
        assert spec.pricing_mode in ("zero_cost_endpoint", "free_tier")
        assert spec.available is True, f"{provider}/{model} must be available"
        assert spec.capabilities.supports_tools is True
        assert spec.capabilities.supports_structured_output is True


def test_provider_classifications_are_honest() -> None:
    """Free-tier models must NOT be declared zero-cost; endpoints may be."""
    assert model_spec("openrouter", "openrouter/free").pricing_mode == "zero_cost_endpoint"
    assert (
        model_spec("nvidia", "nvidia/nemotron-3-super-120b-a12b").pricing_mode
        == "zero_cost_endpoint"
    )
    assert (
        model_spec("nvidia", "nvidia/nemotron-3.5-lightning-30b-a3b").pricing_mode
        == "zero_cost_endpoint"
    )
    assert model_spec("gemini", "gemini-3.8-flash").pricing_mode == "free_tier"
    assert model_spec("gemini", "gemini-3.7-flash").pricing_mode == "free_tier"
    assert model_spec("groq", _GROQ_PLANNER).pricing_mode == "free_tier"
    assert model_spec("groq", _GROQ_FAST).pricing_mode == "free_tier"


def test_known_spec_is_returned_verbatim() -> None:
    spec = model_spec("openrouter", "openrouter/free")
    assert spec.pricing_mode == "zero_cost_endpoint"
    assert spec.available is True
    assert spec.capabilities.supports_tools is True


def test_unknown_spec_is_synthetic_and_unavailable() -> None:
    spec = model_spec("groq", "some-groq-model")
    assert spec.pricing_mode == "unknown"
    assert spec.available is False
    assert spec.capabilities.supports_tools is False


def test_empty_model_name_is_rejected() -> None:
    with pytest.raises(ValueError):
        model_spec("groq", "")


def test_pricing_message_labels() -> None:
    assert "zero-cost endpoint" in pricing_message(model_spec("openrouter", "openrouter/free"))
    assert "free tier" in pricing_message(model_spec("gemini", "gemini-3.8-flash"))
    assert "unavailable" in pricing_message(model_spec("groq", "not-registered"))


# ── policy matrix ─────────────────────────────────────────────────────────


def test_zero_cost_endpoint_accepted_under_strict_zero_cost() -> None:
    allowed, spec, reason = qualify(
        "openrouter", "openrouter/free", free_only=True, strict_zero_cost=True
    )
    assert allowed is True
    assert spec.pricing_mode == "zero_cost_endpoint"
    assert reason == ""


def test_zero_cost_endpoint_accepted_without_free_only() -> None:
    allowed, _spec, _reason = qualify(
        "openrouter", "openrouter/free", free_only=False, strict_zero_cost=True
    )
    assert allowed is True


def test_free_tier_rejected_under_strict_zero_cost() -> None:
    for provider in ("groq", "gemini"):
        allowed, spec, reason = qualify(
            provider,
            _free_tier_model(provider),
            free_only=True,
            strict_zero_cost=True,
        )
        assert allowed is False
        assert spec.pricing_mode == "free_tier"
        assert "billing state cannot be verified" in reason


def test_free_tier_accepted_when_strict_zero_cost_is_false() -> None:
    for provider in ("groq", "gemini"):
        allowed, _spec, _reason = qualify(
            provider, _free_tier_model(provider), free_only=True, strict_zero_cost=False
        )
        assert allowed is True


def test_free_tier_allowed_even_without_free_only_but_never_paid() -> None:
    allowed, _spec, _reason = qualify(
        "groq", _GROQ_PLANNER, free_only=False, strict_zero_cost=False
    )
    assert allowed is True  # free_tier is never more expensive than paid


def test_unknown_pricing_is_never_allowed() -> None:
    for free_only in (True, False):
        for strict in (True, False):
            allowed, _spec, reason = qualify(
                "groq", "not-registered", free_only=free_only, strict_zero_cost=strict
            )
            assert allowed is False
            assert "not registered" in reason


def test_paid_model_refused_under_strict(monkeypatch: typing.Any) -> None:
    _inject(monkeypatch, "groq", "paid-test-model", "paid", True)
    allowed, _spec, reason = qualify(
        "groq", "paid-test-model", free_only=False, strict_zero_cost=True
    )
    assert allowed is False
    assert "strict_zero_cost" in reason


def test_paid_model_refused_in_free_only_mode(monkeypatch: typing.Any) -> None:
    _inject(monkeypatch, "groq", "paid-test-model", "paid", True)
    allowed, _spec, reason = qualify(
        "groq", "paid-test-model", free_only=True, strict_zero_cost=False
    )
    assert allowed is False
    assert "not eligible for free-only" in reason


def test_paid_model_allowed_only_with_full_opt_out(monkeypatch: typing.Any) -> None:
    _inject(monkeypatch, "groq", "paid-test-model", "paid", True)
    for free_only in (True, False):
        allowed, _spec, _reason = qualify(
            "groq", "paid-test-model", free_only=free_only, strict_zero_cost=True
        )
        assert allowed is False
    allowed, _spec, _reason = qualify(
        "groq", "paid-test-model", free_only=False, strict_zero_cost=False
    )
    assert allowed is True


def test_unavailable_model_refused_even_when_free(monkeypatch: typing.Any) -> None:
    _inject(monkeypatch, "groq", "gone-test-model", "free_tier", False)
    allowed, _spec, reason = qualify(
        "groq", "gone-test-model", free_only=True, strict_zero_cost=False
    )
    assert allowed is False
    assert "unavailable" in reason
    allowed, _spec, _reason = qualify(
        "groq", "gone-test-model", free_only=False, strict_zero_cost=False
    )
    assert allowed is False


def _free_tier_model(provider: str) -> str:
    if provider == "groq":
        return _GROQ_PLANNER
    if provider == "gemini":
        return "gemini-3.8-flash"
    raise AssertionError(f"no free-tier fixture for {provider}")


def _inject(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    model: str,
    pricing: str,
    available: bool,
) -> None:
    def patched() -> dict[tuple[str, str], ModelSpec]:
        base = dict(_FREE_ONLY_MODELS)
        base[(provider, model)] = ModelSpec(
            provider,
            model,
            pricing,  # type: ignore[arg-type]
            available,
            _cap(True, True),
            note="test-injected model",
        )
        return base

    monkeypatch.setattr("jarvis.llm.models._FREE_ONLY_MODELS", patched())


def test_capabilities_shape() -> None:
    spec = model_spec("openrouter", "openrouter/free")
    assert spec.capabilities.supports_vision is False
