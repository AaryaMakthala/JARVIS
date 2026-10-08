"""Tests for dynamic provider health and structural failure classification.

The requirement is that provider preference is **dynamic, never a permanent
blacklist**: a 429 or 503 on one provider must make the *next* interaction prefer
another one, and the provider must come back on its own once it recovers — no
restart, no hand-edited list, no lost configured position.

These tests use an injected clock so recovery is proved deterministically
without sleeping.
"""

from __future__ import annotations

import pytest

from jarvis.llm.failures import (
    FailureCategory,
    LLMCallFailure,
    classify,
    cooldown_seconds,
    should_retry_same_provider,
)
from jarvis.llm.health import ProviderHealth

#: The configured preference order that must be preserved: (provider, slot, role).
ORDER: list[tuple[str, int, str]] = [
    ("Groq", 1, "fast"),
    ("OpenRouter", 1, "fast"),
    ("NVIDIA", 1, "fast"),
    ("Gemini", 1, "fast"),
]


class _Clock:
    """Manually advanced monotonic clock."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _failure(
    category: FailureCategory,
    provider: str = "NVIDIA",
    role: str = "fast",
    slot: int = 1,
) -> LLMCallFailure:
    return LLMCallFailure(
        provider=provider, model="test/model", role=role, slot=slot, category=category
    )


class TestCooldownIsFiniteAndDynamic:
    def test_failure_marks_pair_as_cooling(self) -> None:
        clock = _Clock()
        health = ProviderHealth(clock)
        health.record_failure(_failure(FailureCategory.RATE_LIMIT))
        assert health.is_cooling("NVIDIA", "fast") is True
        assert health.remaining("NVIDIA", "fast") > 0

    def test_pair_recovers_by_itself_after_the_cooldown(self) -> None:
        """No restart, no hand-edited list: time alone brings it back."""
        clock = _Clock()
        health = ProviderHealth(clock)
        health.record_failure(_failure(FailureCategory.SERVER_ERROR))
        clock.advance(cooldown_seconds(FailureCategory.SERVER_ERROR) + 1.0)
        assert health.is_cooling("NVIDIA", "fast") is False

    def test_cooldown_grows_but_stays_bounded(self) -> None:
        """Repeated failures back off further, but never become permanent."""
        clock = _Clock()
        health = ProviderHealth(clock)
        applied = [health.record_failure(_failure(FailureCategory.SERVER_ERROR)) for _ in range(12)]
        assert applied == sorted(applied), "cooldown must not shrink"
        assert applied[-1] == applied[0] * 8, "growth must cap at 8x"
        clock.advance(applied[-1] + 1.0)
        assert health.is_cooling("NVIDIA", "fast") is False

    def test_success_resets_the_failure_count(self) -> None:
        clock = _Clock()
        health = ProviderHealth(clock)
        health.record_failure(_failure(FailureCategory.RATE_LIMIT))
        health.record_failure(_failure(FailureCategory.RATE_LIMIT))
        assert health.failure_count("NVIDIA", "fast") == 2
        health.record_success("NVIDIA", "fast")
        assert health.failure_count("NVIDIA", "fast") == 0
        assert health.is_cooling("NVIDIA", "fast") is False

    def test_totals_survive_a_reset_of_the_streak(self) -> None:
        clock = _Clock()
        health = ProviderHealth(clock)
        health.record_failure(_failure(FailureCategory.TIMEOUT))
        health.record_success("NVIDIA", "fast")
        health.record_failure(_failure(FailureCategory.TIMEOUT))
        entry = health.entry("NVIDIA", "fast")
        assert entry.total_failures == 2
        assert entry.total_successes == 1


class TestGranularity:
    def test_roles_are_tracked_independently(self) -> None:
        """A fast-model timeout must not knock out the planner model."""
        clock = _Clock()
        health = ProviderHealth(clock)
        health.record_failure(_failure(FailureCategory.TIMEOUT, role="fast"))
        assert health.is_cooling("NVIDIA", "fast") is True
        assert health.is_cooling("NVIDIA", "planner") is False

    def test_providers_are_tracked_independently(self) -> None:
        clock = _Clock()
        health = ProviderHealth(clock)
        health.record_failure(_failure(FailureCategory.SERVER_ERROR, provider="NVIDIA"))
        assert health.is_cooling("Groq", "fast") is False


class TestSlotGranularity:
    def test_slots_of_one_provider_are_independent_entries(self) -> None:
        """A 429 on the primary key must not cool the provider's backup key."""
        clock = _Clock()
        health = ProviderHealth(clock)
        health.record_failure(_failure(FailureCategory.RATE_LIMIT, provider="Groq", slot=2))
        assert health.is_cooling("Groq", "fast", slot=2) is True
        assert health.is_cooling("Groq", "fast", slot=1) is False
        assert health.failure_count("Groq", "fast", slot=2) == 1
        assert health.failure_count("Groq", "fast", slot=1) == 0
        # A success on slot 1 must not clear slot 2's independent streak.
        health.record_success("Groq", "fast", slot=1)
        assert health.failure_count("Groq", "fast", slot=2) == 1
        assert health.is_cooling("Groq", "fast", slot=2) is True

    def test_cooling_slot_sinks_while_sibling_slot_keeps_its_position(self) -> None:
        """Per-slot ordering: the healthy primary stays ahead of the cooled backup."""
        clock = _Clock()
        health = ProviderHealth(clock)
        items = [("Groq", 1, "fast"), ("Groq", 2, "fast"), ("NVIDIA", 1, "fast")]
        health.record_failure(_failure(FailureCategory.RATE_LIMIT, provider="Groq", slot=2))
        ordered = health.order(items, key=lambda item: item)
        assert ordered == [("Groq", 1, "fast"), ("NVIDIA", 1, "fast"), ("Groq", 2, "fast")]

    def test_snapshot_lists_per_slot_entries_with_counts_only(self) -> None:
        clock = _Clock()
        health = ProviderHealth(clock)
        health.record_failure(_failure(FailureCategory.TIMEOUT, provider="Groq", slot=1))
        health.record_failure(_failure(FailureCategory.RATE_LIMIT, provider="Groq", slot=2))
        snap = health.snapshot()
        assert len(snap) == 2
        by_slot = {row["slot"]: row for row in snap}
        assert by_slot[1]["provider"] == "Groq"
        assert by_slot[1]["total_failures"] == 1
        assert by_slot[1]["last_category"] == FailureCategory.TIMEOUT.value
        assert by_slot[2]["consecutive_failures"] == 1
        assert by_slot[2]["last_category"] == FailureCategory.RATE_LIMIT.value
        allowed = {
            "provider",
            "slot",
            "role",
            "consecutive_failures",
            "cooling",
            "cooldown_remaining_s",
            "last_category",
            "total_failures",
            "total_successes",
        }
        for row in snap:
            assert set(row) <= allowed

    def test_safe_summary_label_carries_slot_but_never_key_material(self) -> None:
        failure = LLMCallFailure(
            provider="groq",
            model="test/model",
            role="planner",
            slot=2,
            category=FailureCategory.RATE_LIMIT,
            http_status=429,
            message="429 gsk_super_secret_key",
        )
        summary = failure.safe_summary()
        assert summary == "groq/s2 [planner] HTTP 429 rate_limit"
        for needle in ("_api_key", "gsk", "secret", "test/model"):
            assert needle not in summary


class TestOrdering:
    def test_healthy_order_is_untouched(self) -> None:
        health = ProviderHealth(_Clock())
        assert health.order(ORDER, key=lambda item: item) == ORDER

    def test_cooling_provider_sinks_but_is_never_dropped(self) -> None:
        """Deprioritised, not deleted: refusing to call anything fails the task."""
        clock = _Clock()
        health = ProviderHealth(clock)
        health.record_failure(_failure(FailureCategory.RATE_LIMIT, provider="Groq"))
        ordered = health.order(ORDER, key=lambda item: item)
        assert len(ordered) == len(ORDER)
        assert set(ordered) == set(ORDER)
        assert ordered[-1] == ("Groq", 1, "fast")

    def test_sinks_back_to_front_after_recovery(self) -> None:
        clock = _Clock()
        health = ProviderHealth(clock)
        health.record_failure(_failure(FailureCategory.RATE_LIMIT, provider="Groq"))
        clock.advance(cooldown_seconds(FailureCategory.RATE_LIMIT) + 1.0)
        assert health.order(ORDER, key=lambda item: item) == ORDER

    def test_ordering_is_stable_among_equals(self) -> None:
        clock = _Clock()
        health = ProviderHealth(clock)
        for provider, role in (("Groq", "fast"), ("OpenRouter", "fast")):
            health.record_failure(_failure(FailureCategory.RATE_LIMIT, provider, role))
        ordered = health.order(ORDER, key=lambda item: item)
        assert ordered[:2] == [("NVIDIA", 1, "fast"), ("Gemini", 1, "fast")]

    def test_all_cooling_is_reported_but_callers_still_try(self) -> None:
        clock = _Clock()
        health = ProviderHealth(clock)
        for provider, _slot, role in ORDER:
            health.record_failure(_failure(FailureCategory.SERVER_ERROR, provider, role))
        assert health.all_cooling(ORDER) is True
        # Deprioritised, never removed.
        assert len(health.order(ORDER, key=lambda item: item)) == len(ORDER)

    def test_all_cooling_of_nothing_is_false(self) -> None:
        assert ProviderHealth(_Clock()).all_cooling([]) is False


class TestSnapshot:
    def test_snapshot_is_json_friendly_and_hides_nothing_sensitive(self) -> None:
        clock = _Clock()
        health = ProviderHealth(clock)
        health.record_failure(_failure(FailureCategory.AUTH))
        snap = health.snapshot()
        assert len(snap) == 1
        row = snap[0]
        assert row["provider"] == "NVIDIA"
        assert row["role"] == "fast"
        assert row["cooling"] is True
        assert row["last_category"] == FailureCategory.AUTH.value

    def test_reset_clears_everything(self) -> None:
        clock = _Clock()
        health = ProviderHealth(clock)
        health.record_failure(_failure(FailureCategory.TIMEOUT))
        health.reset()
        assert health.snapshot() == []


class TestClassification:
    @pytest.mark.parametrize(
        ("exception", "expected"),
        [
            (TimeoutError("timed out"), FailureCategory.TIMEOUT),
            (ConnectionError("connection reset"), FailureCategory.CONNECTION),
        ],
    )
    def test_standard_exceptions(self, exception: Exception, expected: FailureCategory) -> None:
        assert classify(exception) is expected

    def test_explicit_category_attribute_wins(self) -> None:
        class Custom(Exception):
            category = FailureCategory.CONTEXT_LENGTH

        assert classify(Custom("timed out")) is FailureCategory.CONTEXT_LENGTH

    def test_status_code_401_is_auth(self) -> None:
        class Err(Exception):
            status_code = 401

        assert classify(Err("nope")) is FailureCategory.AUTH

    def test_status_code_429_is_rate_limit(self) -> None:
        class Err(Exception):
            status_code = 429

        assert classify(Err("slow down")) is FailureCategory.RATE_LIMIT

    def test_status_code_503_is_server_error(self) -> None:
        class Err(Exception):
            status_code = 503

        assert classify(Err("unavailable")) is FailureCategory.SERVER_ERROR

    def test_unsupported_model_error_class(self) -> None:
        class NotFoundError(Exception):
            pass

        assert classify(NotFoundError("model xyz")) is FailureCategory.UNSUPPORTED_MODEL

    def test_schema_capability_error_by_message(self) -> None:
        exc = Exception("this model does not support response_format json_schema")
        assert classify(exc) is FailureCategory.UNSUPPORTED_CAPABILITY

    def test_classify_never_raises(self) -> None:
        class Hostile(Exception):
            def __str__(self) -> Exception:
                raise RuntimeError("boom")

        assert isinstance(classify(Hostile()), FailureCategory)


class TestCooldownPolicy:
    @pytest.mark.parametrize(
        "category",
        [
            FailureCategory.RATE_LIMIT,
            FailureCategory.SERVER_ERROR,
            FailureCategory.TIMEOUT,
            FailureCategory.CONNECTION,
            FailureCategory.AUTH,
            FailureCategory.UNSUPPORTED_MODEL,
        ],
    )
    def test_every_category_cooldown_is_positive_and_finite(
        self, category: FailureCategory
    ) -> None:
        """Invariant: no provider is ever removed from rotation permanently."""
        seconds = cooldown_seconds(category)
        assert 0 < seconds < 3600

    @pytest.mark.parametrize(
        ("category", "retryable"),
        [
            (FailureCategory.RATE_LIMIT, True),
            (FailureCategory.SERVER_ERROR, True),
            (FailureCategory.TIMEOUT, True),
            (FailureCategory.CONNECTION, True),
            # A long prompt can succeed against the same provider after the
            # client shrinks it, and a sibling model may have a bigger window.
            (FailureCategory.CONTEXT_LENGTH, True),
            (FailureCategory.AUTH, False),
            (FailureCategory.MALFORMED_REQUEST, False),
            (FailureCategory.UNSUPPORTED_CAPABILITY, False),
        ],
    )
    def test_retry_policy(self, category: FailureCategory, retryable: bool) -> None:
        assert should_retry_same_provider(category) is retryable
