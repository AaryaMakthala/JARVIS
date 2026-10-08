"""4.19: ``_verify_focus`` re-reads the foreground a bounded number of times.

``set_focus()`` can return before Windows has finished activating the window,
so the first read can transiently see ``foreground=''`` (the live Notepad
failure).  The poll must absorb that race while keeping the exact acceptance
condition and the exact existing error wording - 4.17c's non-retryable
classification depends on it - never sleeping for real in tests, and never
polling after a read exception.
"""

from __future__ import annotations

import pytest

from jarvis.tools import keyboard


def _install(
    monkeypatch: pytest.MonkeyPatch, titles: list[str | Exception]
) -> tuple[list[int], list[float]]:
    """Patch the foreground reader and the sleep hook; return their recorders.

    ``titles`` is read in order; the last entry repeats forever, so a
    one-element list models a foreground that never changes.
    """
    reads: list[int] = []
    sleeps: list[float] = []

    def _read() -> str:
        idx = len(reads)
        reads.append(idx)
        value = titles[idx] if idx < len(titles) else titles[-1]
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(keyboard, "_read_foreground_title", _read)
    monkeypatch.setattr(keyboard, "_pause", sleeps.append)
    return reads, sleeps


# ── the race is absorbed ────────────────────────────────────────────────


def test_wrong_twice_then_match_returns_success(monkeypatch: pytest.MonkeyPatch) -> None:
    reads, sleeps = _install(monkeypatch, ["Calculator", "", "Notepad - untitled"])
    assert keyboard._verify_focus("notepad") is None
    assert len(reads) == 3
    assert sleeps == [keyboard._VERIFY_POLL_S, keyboard._VERIFY_POLL_S]


def test_immediate_match_does_not_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    reads, sleeps = _install(monkeypatch, ["Untitled - Notepad"])
    assert keyboard._verify_focus("notepad") is None
    assert len(reads) == 1
    assert sleeps == []


# ── bounded, fail-closed, byte-identical errors ─────────────────────────


def test_wrong_forever_bounded_failure_keeps_the_existing_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads, sleeps = _install(monkeypatch, ["Calculator"])
    result = keyboard._verify_focus("notepad")
    assert result is not None and not result.ok
    assert result.error == (
        "focus verification failed: foreground is 'Calculator', expected 'notepad'"
    )
    assert len(reads) == keyboard._VERIFY_ATTEMPTS
    assert sleeps == [keyboard._VERIFY_POLL_S] * (keyboard._VERIFY_ATTEMPTS - 1)


def test_empty_forever_bounded_failure_keeps_the_existing_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live failure mode: foreground stays '' for the whole budget."""
    reads, sleeps = _install(monkeypatch, [""])
    result = keyboard._verify_focus("notepad")
    assert result is not None and not result.ok
    assert result.error == "focus verification failed: foreground is '', expected 'notepad'"
    assert len(reads) == keyboard._VERIFY_ATTEMPTS
    assert sleeps == [keyboard._VERIFY_POLL_S] * (keyboard._VERIFY_ATTEMPTS - 1)


def test_read_exception_fails_immediately_without_sleeping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads, sleeps = _install(monkeypatch, [RuntimeError("desktop session not available")])
    result = keyboard._verify_focus("notepad")
    assert result is not None and not result.ok
    assert result.error == (
        "focus verification failed: could not read foreground window: desktop session not available"
    )
    assert len(reads) == 1, "a read that raises must never be polled again"
    assert sleeps == []


# ── _pause is the only sleep path, and it stays injectable ──────────────


def test_pause_is_the_only_sleep_path_and_is_injectable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, ["Calculator"])

    def _trap(_seconds: float) -> None:
        raise AssertionError("real sleep attempted; _verify_focus bypassed _pause")

    sleeps: list[float] = []
    monkeypatch.setattr(keyboard, "_pause", sleeps.append)
    monkeypatch.setattr(keyboard.time, "sleep", _trap)

    result = keyboard._verify_focus("notepad")
    assert result is not None and not result.ok
    assert sleeps == [keyboard._VERIFY_POLL_S] * (keyboard._VERIFY_ATTEMPTS - 1)
