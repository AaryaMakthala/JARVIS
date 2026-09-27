"""Dynamic per-provider, per-role health memory.

Requirement: provider preference is *dynamic*, not permanent.  A 429 or a 503
on NVIDIA must make the next interaction prefer another provider, and it must
come back on its own once the provider recovers — without a restart, without
losing its configured position permanently, and without anyone editing a list
by hand.

This module holds that state.  It is deliberately small and clock-injected so
the behaviour is fully deterministic in tests.

Granularity is ``(provider, role)``: NVIDIA's ``fast`` model timing out must
not knock out NVIDIA's ``planner`` model, because they are different endpoints
with different load.

**Nothing here is a blacklist.**  Every entry has a finite cooldown, a decaying
consecutive-failure count, and a reset on the next success.  A rotating an API
key or a provider re-publishing a model therefore recovers by itself: the next
call after the cooldown is a fresh probe.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

from jarvis.llm.failures import FailureCategory, LLMCallFailure, cooldown_seconds

T = TypeVar("T")


@dataclass
class ProviderHealthEntry:
    """Health of one ``(provider, role)`` pair.  Safe to log; no secrets."""

    provider: str
    role: str
    consecutive_failures: int = 0
    cooldown_until: float = 0.0
    last_category: FailureCategory = FailureCategory.NONE
    last_failure_at: float = 0.0
    last_success_at: float = 0.0
    total_failures: int = 0
    total_successes: int = 0
    history: list[FailureCategory] = field(default_factory=list)

    def is_cooling(self, now: float) -> bool:
        """Whether this pair must be skipped for the current call."""
        return now < self.cooldown_until

    def remaining(self, now: float) -> float:
        """Seconds of cooldown left (0 when usable now)."""
        return max(0.0, self.cooldown_until - now)

    def snapshot(self, now: float | None = None) -> dict[str, Any]:
        """Safe diagnostic view for ``jarvis doctor`` / logs.

        ``now`` is passed in by :class:`ProviderHealth` so the view reports
        against the *injected* clock.  Reading the real clock here would make
        ``cooling`` disagree with :meth:`is_cooling` in every test and in any
        caller that drives the health store from a fake timeline.
        """
        moment = time.monotonic() if now is None else now
        return {
            "provider": self.provider,
            "role": self.role,
            "consecutive_failures": self.consecutive_failures,
            "cooling": self.cooldown_until > moment,
            "cooldown_remaining_s": round(self.remaining(moment), 1),
            "last_category": self.last_category.value,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
        }


class ProviderHealth:
    """Cooldown/recovery memory for a set of providers.

    ``clock`` is injected so tests can advance time deterministically instead
    of sleeping.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._entries: dict[tuple[str, str], ProviderHealthEntry] = {}

    # ── recording ────────────────────────────────────────────────────────

    def record_success(self, provider: str, role: str) -> None:
        """Clear the failure state for ``(provider, role)`` (recovery)."""
        entry = self._entry(provider, role)
        entry.consecutive_failures = 0
        entry.cooldown_until = 0.0
        entry.last_category = FailureCategory.NONE
        entry.last_success_at = self._clock()
        entry.total_successes += 1

    def record_failure(self, failure: LLMCallFailure) -> float:
        """Record a failure and return the cooldown applied, in seconds.

        The cooldown grows with the consecutive-failure count (capped at four
        steps) so a provider that is briefly throttled is skipped briefly, while
        one that is genuinely down is skipped for longer — but it is always
        finite, and any success resets the count to zero.
        """
        now = self._clock()
        entry = self._entry(failure.provider, failure.role)
        entry.consecutive_failures += 1
        entry.total_failures += 1
        entry.last_category = failure.category
        entry.last_failure_at = now
        entry.history.append(failure.category)
        del entry.history[:-8]
        base = cooldown_seconds(failure.category)
        # 1x, 2x, 4x, 8x — bounded so a provider always returns within minutes.
        factor = min(2 ** (entry.consecutive_failures - 1), 8)
        applied = base * factor
        entry.cooldown_until = now + applied
        return applied

    # ── queries ──────────────────────────────────────────────────────────

    def entry(self, provider: str, role: str) -> ProviderHealthEntry:
        """The health entry for ``(provider, role)`` (created on demand)."""
        return self._entry(provider, role)

    def is_cooling(self, provider: str, role: str) -> bool:
        """Whether ``(provider, role)`` must be skipped for this call."""
        return self._entry(provider, role).is_cooling(self._clock())

    def remaining(self, provider: str, role: str) -> float:
        """Seconds until ``(provider, role)`` is retried."""
        return self._entry(provider, role).remaining(self._clock())

    def failure_count(self, provider: str, role: str) -> int:
        """Consecutive failures recorded for ``(provider, role)``."""
        return self._entry(provider, role).consecutive_failures

    def all_cooling(self, providers: Iterable[tuple[str, str]]) -> bool:
        """Whether every listed pair is cooling.

        The caller must still try them in that case — a cooling provider is
        deprioritised, never removed, because refusing to call anything would
        fail the interaction outright.
        """
        pairs = list(providers)
        return bool(pairs) and all(self.is_cooling(p, r) for p, r in pairs)

    def order(self, items: Sequence[T], key: Callable[[T], tuple[str, str]]) -> list[T]:
        """Return ``items`` reordered by current health, stably.

        Healthy pairs keep their configured order.  Cooling pairs sink, and
        among those the ones with more consecutive failures sink furthest — but
        **nothing is ever dropped**: a provider that is the only one left is
        still tried, and its success restores its position.

        A pair whose cooldown has *expired* goes back to its configured position
        as a fresh probe.  That is the difference between dynamic preference and
        a blacklist: without it, a single 429 would push a provider to the back
        of the rotation forever, because it would only ever be called after the
        others failed — which is exactly when there is nothing left to gain.
        Its failure streak is deliberately *not* cleared here, so a provider
        that fails again gets a longer cooldown on top of its configured slot.
        """
        now = self._clock()
        decorated: list[tuple[int, int, int, T]] = []
        for index, item in enumerate(items):
            provider, role = key(item)
            entry = self._entry(provider, role)
            cooling = entry.is_cooling(now)
            decorated.append(
                (1 if cooling else 0, entry.consecutive_failures if cooling else 0, index, item)
            )
        decorated.sort(key=lambda row: (row[0], row[1], row[2]))
        return [row[3] for row in decorated]

    def snapshot(self) -> list[dict[str, Any]]:
        """Safe diagnostic view of every tracked pair (on the injected clock)."""
        now = self._clock()
        return [entry.snapshot(now) for entry in self._entries.values()]

    def reset(self) -> None:
        """Forget all health state (test isolation)."""
        self._entries.clear()

    # ── internals ────────────────────────────────────────────────────────

    def _entry(self, provider: str, role: str) -> ProviderHealthEntry:
        pair = (provider, role)
        entry = self._entries.get(pair)
        if entry is None:
            entry = ProviderHealthEntry(provider=provider, role=role)
            self._entries[pair] = entry
        return entry
