"""B8 key rotation in the composite walk (Stage 5a).

Key rotation is for **availability only** (ROADMAP V2.1, B8): auth failures
and outages advance to the provider's next key slot, while a 429 cools only
the failing ``(provider, slot, role)`` and does *not* rotate keys unless
``llm.rotate_on_rate_limit`` is enabled (some provider terms forbid evading
rate limits by rotating accounts).  Model-scoped failures are recorded on
the provider-wide pseudo-slot 0, so one probe cools every slot of the
provider for that role.

These tests drive the real ``MultiProviderClient`` with scripted clients and
an injected clock.  The free-only gate is not involved here (see
``test_freeonly.py``).
"""

from __future__ import annotations

import logging
import typing

import pytest

from jarvis.llm.client import LLMAuthError, LLMCapabilityError, LLMError, Usage
from jarvis.llm.failures import FailureCategory, LLMCallFailure
from jarvis.llm.health import ProviderHealth
from jarvis.llm.models import ModelSpec, model_spec
from jarvis.llm.multi import MultiProviderClient
from jarvis.llm.provider import ProviderInfo

GROQ_MODEL = "openai/gpt-oss-120b"
NVIDIA_MODEL = "nvidia/nemotron-3-super-120b-a12b"

Candidate = tuple[ProviderInfo, ModelSpec, typing.Any]


class _Clock:
    """Manually advanced monotonic clock."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeClient:
    """Scriptable client that records how often each slot was probed."""

    def __init__(self, *, error: Exception | None = None, response: str = "ok") -> None:
        self.error = error
        self.response = response
        self.text_calls = 0
        self.structured_calls = 0

    def text(self, **kwargs: typing.Any) -> tuple[str, Usage]:
        self.text_calls += 1
        if self.error is not None:
            raise self.error
        return self.response, Usage()

    def structured(self, **kwargs: typing.Any) -> tuple[object, Usage]:
        self.structured_calls += 1
        if self.error is not None:
            raise self.error
        return self.response, Usage()


def _rate_limited() -> LLMError:
    return LLMError("rate limited", category=FailureCategory.RATE_LIMIT)


def _cand(name: str, slot: int, client: FakeClient) -> Candidate:
    model = GROQ_MODEL if name == "groq" else NVIDIA_MODEL
    return ProviderInfo(name=name, model=model, slot=slot), model_spec(name, model), client


def _groq_pair(s1: FakeClient, s2: FakeClient) -> list[Candidate]:
    return [_cand("groq", 1, s1), _cand("groq", 2, s2)]


def test_429_cools_the_slot_and_falls_through_without_rotating_keys() -> None:
    """Default (B8): a 429 cools that slot only; the backup key is not tried."""
    s1 = FakeClient(error=_rate_limited())
    s2 = FakeClient(response="backup")
    nvidia = FakeClient(response="nvidia-ok")
    multi = MultiProviderClient([*_groq_pair(s1, s2), _cand("nvidia", 1, nvidia)])
    text, _ = multi.text(system="s", user="u")
    assert text == "nvidia-ok"
    assert s1.text_calls == 1
    assert s2.text_calls == 0  # no key rotation on 429 by default
    assert nvidia.text_calls == 1
    assert multi.health.is_cooling("groq", "fast", slot=1) is True
    assert multi.health.is_cooling("groq", "fast", slot=2) is False


def test_429_rotates_to_the_next_slot_when_the_flag_is_enabled() -> None:
    s1 = FakeClient(error=_rate_limited())
    s2 = FakeClient(response="backup")
    nvidia = FakeClient(response="nvidia-ok")
    multi = MultiProviderClient(
        [*_groq_pair(s1, s2), _cand("nvidia", 1, nvidia)],
        rotate_on_rate_limit=True,
    )
    text, _ = multi.text(system="s", user="u")
    assert text == "backup"
    assert s1.text_calls == 1
    assert s2.text_calls == 1  # rotated to the next key slot
    assert nvidia.text_calls == 0  # the provider answered on slot 2


def test_auth_failure_advances_to_the_next_slot() -> None:
    """Availability failures rotate keys even with the flag off."""
    s1 = FakeClient(error=LLMAuthError("rejected"))
    s2 = FakeClient(response="backup")
    nvidia = FakeClient(response="nvidia-ok")
    multi = MultiProviderClient([*_groq_pair(s1, s2), _cand("nvidia", 1, nvidia)])
    text, _ = multi.text(system="s", user="u")
    assert text == "backup"
    assert s1.text_calls == 1
    assert s2.text_calls == 1
    assert nvidia.text_calls == 0
    assert multi.health.is_cooling("groq", "fast", slot=1) is True
    assert multi.health.is_cooling("groq", "fast", slot=2) is False


def test_capability_failure_cools_every_slot_with_one_probe() -> None:
    """Model-scoped failure: one probe records slot 0; every slot reports cooling."""
    clock = _Clock()
    health = ProviderHealth(clock)
    s1 = FakeClient(error=LLMCapabilityError("this model cannot do structured output"))
    s2 = FakeClient(response="backup")
    nvidia = FakeClient(response="nvidia-structured")
    multi = MultiProviderClient(
        [*_groq_pair(s1, s2), _cand("nvidia", 1, nvidia)],
        health=health,
        clock=clock,
    )
    out, _ = multi.structured(system="s", user="u", schema=type(str))
    assert out == "nvidia-structured"
    assert s1.structured_calls == 1
    assert s2.structured_calls == 0  # one probe, then on to the next provider
    # recorded once, provider-wide on slot 0 ...
    assert health.failure_count("groq", "planner", slot=0) == 1
    assert health.failure_count("groq", "planner", slot=2) == 0
    # ... so every real slot of that provider reports cooling for the role
    assert health.is_cooling("groq", "planner", slot=1) is True
    assert health.is_cooling("groq", "planner", slot=2) is True


def test_all_cooling_candidates_are_still_tried() -> None:
    """Nothing is ever dropped: every candidate is attempted, exactly once."""
    clock = _Clock()
    health = ProviderHealth(clock)
    for slot in (1, 2):
        health.record_failure(
            LLMCallFailure(
                provider="groq",
                model=GROQ_MODEL,
                role="fast",
                slot=slot,
                category=FailureCategory.SERVER_ERROR,
            )
        )
    health.record_failure(
        LLMCallFailure(
            provider="nvidia",
            model=NVIDIA_MODEL,
            role="fast",
            slot=1,
            category=FailureCategory.TIMEOUT,
        )
    )
    s1 = FakeClient(error=LLMError("s1 down"))
    s2 = FakeClient(error=LLMError("s2 down"))
    nvidia = FakeClient(error=LLMError("nvidia down"))
    multi = MultiProviderClient(
        [_cand("groq", 1, s1), _cand("groq", 2, s2), _cand("nvidia", 1, nvidia)],
        health=health,
        clock=clock,
    )
    with pytest.raises(LLMError):
        multi.text(system="s", user="u")
    assert s1.text_calls == 1
    assert s2.text_calls == 1
    assert nvidia.text_calls == 1


def test_rotation_logs_state_the_rule_but_never_key_material(
    caplog: pytest.LogCaptureFixture, monkeypatch: typing.Any
) -> None:
    monkeypatch.setenv("GROQ_2_API_KEY", "gsk-canary-999")
    caplog.set_level(logging.WARNING)
    logger = logging.getLogger("llm.rotation.test")
    s1 = FakeClient(error=_rate_limited())
    s2 = FakeClient(response="backup")
    nvidia = FakeClient(response="nvidia-ok")
    multi = MultiProviderClient(
        [*_groq_pair(s1, s2), _cand("nvidia", 1, nvidia)],
        logger=logger,
    )
    text, _ = multi.text(system="s", user="u")
    assert text == "nvidia-ok"
    assert "rate limited; not rotating key, trying next provider" in caplog.text
    assert "gsk-canary-999" not in caplog.text
    assert "_api_key" not in caplog.text
    assert "gsk-" not in caplog.text
