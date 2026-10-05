"""A plan-level ``plan_approval`` demands the strict approval words.

One interrupt can now cover several Tier 1 steps at once
(``jarvis.agent.batch_approval.TYPE_PLAN_APPROVAL``).  A colloquial "go" / "ok"
/ "sure" is acceptable for a *single* low-risk step, but not for a spoken
approval that releases a whole batch, so the word set is chosen by payload
``type`` in ``jarvis.voice.loop._approval_words``.

This test needs no audio, no graph and no Windows API: it pins the pure
word-selection function.
"""

from __future__ import annotations

from jarvis.agent.batch_approval import TYPE_PLAN_APPROVAL
from jarvis.daemon.confirmations import plan_hash
from jarvis.voice.loop import (
    _LOOSE_YES_WORDS,
    _STRICT_YES_WORDS,
    _approval_words,
)


def _plan_approval_payload() -> dict[str, object]:
    """The real payload shape built by ``batch_approval._plan_approval_payload``."""
    eligible = [("s1", "a" * 64), ("s2", "b" * 64)]
    return {
        "type": TYPE_PLAN_APPROVAL,
        "tier": 1,
        "summary": "2 steps",
        "action_hash": plan_hash([h for _, h in eligible]),
        "eligible": eligible,
        "plan_hash": plan_hash([h for _, h in eligible]),
        "resolved_paths": [],
    }


def test_plan_approval_rejects_go_ok_sure() -> None:
    words = _approval_words(_plan_approval_payload())
    for colloquial in ("go", "ok", "sure"):
        assert colloquial not in words, f"{colloquial!r} must never approve a plan"
    assert words == _STRICT_YES_WORDS
    # The single-step bucket is the only place these may still approve.
    assert {"go", "ok", "sure"} <= _LOOSE_YES_WORDS


def test_plan_approval_accepts_proceed() -> None:
    words = _approval_words(_plan_approval_payload())
    assert words == {"yes", "confirm", "proceed"}
    assert words == _STRICT_YES_WORDS
    for strict_word in _STRICT_YES_WORDS:
        assert strict_word in words


def test_single_tier1_loose_words_unchanged() -> None:
    # The per-step payload from batch_approval._single_step_payload, unchanged.
    single = {
        "type": "confirm",
        "step_id": "s1",
        "tier": 1,
        "summary": "create a file",
        "needs_unlock": False,
        "typed_confirmation": None,
        "resolved_paths": [],
        "action_hash": "c" * 64,
        "untrusted": False,
    }
    words = _approval_words(single)
    assert words == _LOOSE_YES_WORDS
    for colloquial in ("go", "ok", "sure", "yes", "y", "approve", "approved", "confirm"):
        assert colloquial in words
    # Empty / minimal payloads keep the loose set too (unchanged behaviour).
    assert _approval_words({"tier": 1}) == _LOOSE_YES_WORDS
    assert _approval_words({}) == _LOOSE_YES_WORDS
