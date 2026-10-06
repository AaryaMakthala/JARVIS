"""get_time tool tests (Tier 0, read-only local clock).

The clock is injected by replacing :func:`jarvis.tools.clock.local_now`, so
every expected sentence is deterministic and independent of the machine's wall
clock (which is also why no test asserts the real time).
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from jarvis.config import Settings
from jarvis.policy import rules
from jarvis.tools.base import ToolContext
from jarvis.tools.clock import GetTimeArgs, make_get_time_spec, spoken_time
from jarvis.tools.registry import _FORBIDDEN_RE, build_default_registry


def _at(hour: int, minute: int = 0) -> datetime:
    """A deterministic local datetime (date is irrelevant to this tool).

    The tzinfo is fixed only to satisfy the lint rule against naive datetimes;
    ``spoken_time`` reads the wall-clock ``hour``/``minute`` fields alone, so
    the offset cannot change any expected sentence.
    """
    return datetime(2026, 10, 6, hour, minute, tzinfo=UTC)


#: Input time -> the exact sentence the tool must produce.
SENTENCES: list[tuple[int, int, str]] = [
    (15, 0, "It is 3 o'clock in the afternoon."),
    (15, 45, "It is 3:45 in the afternoon."),
    (12, 0, "It is 12 o'clock noon."),
    (0, 0, "It is 12 o'clock midnight."),
    (9, 5, "It is 9:05 in the morning."),
    (20, 30, "It is 8:30 in the evening."),
    # day-part boundaries
    (4, 59, "It is 4:59 at night."),
    (5, 0, "It is 5 o'clock in the morning."),
    (11, 59, "It is 11:59 in the morning."),
    (21, 0, "It is 9 o'clock at night."),
]


class TestSpokenTime:
    @pytest.mark.parametrize(("hour", "minute", "expected"), SENTENCES)
    def test_sentence(self, hour: int, minute: int, expected: str) -> None:
        assert spoken_time(_at(hour, minute)) == expected

    def test_minutes_are_zero_padded(self) -> None:
        """1-9 minutes read as two digits ("9:05"), never "9:5"."""
        assert spoken_time(_at(9, 5)) == "It is 9:05 in the morning."
        assert spoken_time(_at(8, 1)) == "It is 8:01 in the morning."

    def test_noon_and_midnight_are_exact_minute_zero(self) -> None:
        """Only exactly 12:00 is noon; 12:01 stays an ordinary minute."""
        assert spoken_time(_at(12, 0)) == "It is 12 o'clock noon."
        assert spoken_time(_at(12, 1)) == "It is 12:01 in the afternoon."
        assert spoken_time(_at(0, 0)) == "It is 12 o'clock midnight."
        assert spoken_time(_at(0, 1)) == "It is 12:01 at night."


class TestSpec:
    def test_tier_and_platform(self) -> None:
        spec = make_get_time_spec()
        assert spec.name == "get_time"
        assert spec.base_tier == 0
        assert spec.windows_only is False

    def test_name_is_not_policy_blocked(self) -> None:
        """The real rule must not trip on the name (imported, not re-implemented)."""
        spec = make_get_time_spec()
        assert rules.matches_blocked(spec, GetTimeArgs()) is False

    def test_description_has_no_forbidden_word(self) -> None:
        """`register()` rejects name+description on \\b(shell|powershell|cmd|exec|eval)\\b."""
        spec = make_get_time_spec()
        assert _FORBIDDEN_RE.search(f"{spec.name} {spec.description}") is None

    def test_description_states_the_limit(self) -> None:
        """The planner must be told it cannot answer other time zones / the date."""
        text = make_get_time_spec().description.lower()
        assert "other time zones" in text
        assert "date" in text

    def test_description_has_no_arithmetic_word_key(self) -> None:
        """Belt-and-braces against tests/unit/test_math_phrasings.py's guard."""
        spec = make_get_time_spec()
        pattern = r"\b(times|multiplied|divided|plus|minus|sum|product)\b\s*[:=]"
        assert not re.search(pattern, spec.description)

    def test_args_model_accepts_empty_and_rejects_extras(self) -> None:
        assert GetTimeArgs() == GetTimeArgs()
        with pytest.raises(ValidationError):
            GetTimeArgs(what="time")  # type: ignore[call-arg]

    def test_describe(self) -> None:
        spec = make_get_time_spec()
        assert spec.describe(GetTimeArgs()) == "tell you the time"


class TestRun:
    def _ctx(self) -> ToolContext:
        return ToolContext(settings=Settings())

    def test_returns_injected_time_and_verified_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("jarvis.tools.clock.local_now", lambda: _at(15, 45))
        result = make_get_time_spec().run_verified(GetTimeArgs(), self._ctx())
        assert result.ok is True
        assert result.output == "It is 3:45 in the afternoon."
        assert result.verified is None
        assert result.error is None

    def test_uses_the_injectable_clock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("jarvis.tools.clock.local_now", lambda: _at(9, 5))
        result = make_get_time_spec().run_verified(GetTimeArgs(), self._ctx())
        assert result.output == "It is 9:05 in the morning."

    def test_dry_run_reads_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom() -> datetime:
            raise AssertionError("dry run must not read the clock")

        monkeypatch.setattr("jarvis.tools.clock.local_now", _boom)
        ctx = ToolContext(settings=Settings(), dry_run=True)
        result = make_get_time_spec().run_verified(GetTimeArgs(), ctx)
        assert result.ok is True
        assert "[dry-run]" in result.output


class TestRegistration:
    def test_registered_in_default_registry(self) -> None:
        registry = build_default_registry(Settings())
        assert "get_time" in registry.names()
        assert registry.get("get_time").base_tier == 0
