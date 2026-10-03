"""Voice modes: NORMAL and AUTO.

Modes are a property of the voice loop (not persisted across restarts).
Phrase matching is in code (never the planner), and mode changes are logged
as events only (no transcripts).
"""

from __future__ import annotations

from enum import StrEnum


class VoiceMode(StrEnum):
    """Voice interaction mode."""

    NORMAL = "NORMAL"
    AUTO = "AUTO"


# Mode activation phrases (match in code, after wake word in NORMAL)
ACTIVATE_AUTO_PHRASES = frozenset(
    {
        "activate auto mode",
        "enter auto mode",
        "start auto mode",
        "auto mode on",
        "turn on auto mode",
    }
)

# Mode exit phrases (AUTO -> NORMAL)
SLEEP_PHRASES = frozenset(
    {
        "jarvis sleep",
        "go to sleep",
        "jarvis go to sleep",
        "sleep jarvis",
    }
)

# Sleep phrases when in NORMAL - should not change mode
SLEEP_PHRASES_NORMAL = frozenset(SLEEP_PHRASES)


def _normalise_mode_phrase(text: str) -> str:
    """Normalise text for mode phrase matching."""
    import re

    normalised = text.strip().lower()
    normalised = re.sub(r"auto\s*-?mode", "auto mode", normalised)
    normalised = re.sub(r"[^a-z0-9\s]", " ", normalised)
    normalised = re.sub(r"\s+", " ", normalised)
    return normalised.strip()


def _tokens_equal(a, b):
    return all(x == y for x, y in zip(a, b))


def classify_mode_phrase(text: str) -> str | None:
    """Classify a phrase as a mode command."""
    normalised = _normalise_mode_phrase(text)
    tokens = normalised.split() if normalised else []

    # Exact match
    if normalised in ACTIVATE_AUTO_PHRASES:
        return "activate_auto"
    if normalised in SLEEP_PHRASES:
        return "sleep"

    # Try with up to 4 prefix words (garbled), but tail must equal phrase exactly
    for max_prefix in range(1, 5):
        if max_prefix >= len(tokens):
            break
        tail = tokens[max_prefix:]
        tail_str = " ".join(tail)
        if tail_str in ACTIVATE_AUTO_PHRASES:
            return "activate_auto"
        if tail_str in SLEEP_PHRASES:
            return "sleep"

    return None
