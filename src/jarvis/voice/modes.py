"""Voice modes: NORMAL and AUTO.

Modes are a property of the voice loop (not persisted across restarts).
Phrase matching is in code (never the planner), and mode changes are logged
as events only (no transcripts).

Matching is mode-aware: :func:`classify_mode_phrase` takes the current mode, so
leaving AUTO can be said the short ways users actually say it ("sleep",
"jaro sleep", "auto mode off") without those words meaning anything in NORMAL.
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

# Phrases that leave AUTO (AUTO -> NORMAL).  Matched in BOTH modes on purpose:
# in AUTO they switch back to NORMAL, and in NORMAL the loop answers "Auto mode
# is off." - there is nothing to leave, and saying so is more use to the user
# than planning the words as a task.  The near-misses are listed verbatim rather
# than fuzzy-matched: whisper regularly renders "auto mode" as "auto mod" and
# "auto mood", and the live log had "Auto-mod off" reaching the planner.
#
# A bare "sleep" is deliberately NOT here - see :func:`_is_bare_sleep`.
SLEEP_PHRASES = frozenset(
    {
        "jarvis sleep",
        "go to sleep",
        "jarvis go to sleep",
        "sleep jarvis",
        "auto mode off",
        "auto mod off",
        "auto mood off",
        "turn off auto mode",
        "exit auto mode",
        "leave auto mode",
        "stop auto mode",
    }
)

# Sleep phrases when in NORMAL - should not change mode
SLEEP_PHRASES_NORMAL = frozenset(SLEEP_PHRASES)

#: The word that leaves AUTO on its own, bare or behind a garbled wake word.
_AUTO_SLEEP_WORD = "sleep"


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


def _is_bare_sleep(tokens: list[str]) -> bool:
    """Whether *tokens* is a bare "sleep", or a garbled wake word in front of it.

    Whole-utterance and at most two tokens long, which is what separates the
    mode command the user means ("sleep", "Jaro sleep", "Jarwae sleep") from an
    ordinary sentence that merely ends in the word ("i cant sleep", "how long
    should i sleep").  Those must reach the planner: guessing at intent is the
    LLM's job, and a mode switch is not.

    AUTO-only, because in NORMAL "sleep" is just a word.
    """
    if len(tokens) == 1:
        return tokens[0] == _AUTO_SLEEP_WORD
    if len(tokens) == 2:
        return tokens[1] == _AUTO_SLEEP_WORD
    return False


def classify_mode_phrase(text: str, mode: VoiceMode = VoiceMode.NORMAL) -> str | None:
    """Classify a phrase as a mode command for *mode*.

    Returns "activate_auto", "sleep", or ``None`` for everything else.  ``mode``
    decides the AUTO-only rules (see :func:`_is_bare_sleep`) and defaults to
    NORMAL, which is the mode-agnostic vocabulary the phrase sets below describe.
    """
    normalised = _normalise_mode_phrase(text)
    tokens = normalised.split() if normalised else []

    # AUTO-only, and ahead of the sets: neither contains a bare "sleep".
    if mode == VoiceMode.AUTO and _is_bare_sleep(tokens):
        return "sleep"

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
