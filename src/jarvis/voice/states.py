"""Explicit per-interaction voice lifecycle states.

The daemon must be able to *prove* what stage a voice interaction reached and
whether it always returned to :attr:`VoicePhase.READY`.  Previously the loop
logged free-form strings (``LISTENING`` / ``CAPTURING`` / ``PROCESSING`` …)
that were never validated, so an aborted interaction left no trace and a stuck
one looked identical to a healthy one.

This module owns the vocabulary and the legal transitions.  ``VoiceService``
keeps its own coarse service state (``off`` / ``starting`` / ``on`` /
``error``); this enum is the *interaction* state machine that runs inside one
running loop and is reset to ``READY`` after every wake.

Invariant: every interaction must terminate in ``READY`` (success or
recovered failure) or in ``ERROR``/``STOPPING``.  A transition that is not in
:data:`ALLOWED_TRANSITIONS` is rejected by :func:`check_transition`, which
makes an illegal jump a loud test failure rather than a silent log line.
"""

from __future__ import annotations

import itertools
from enum import StrEnum


class VoicePhase(StrEnum):
    """One stage of a single wake-to-answer interaction."""

    #: Service/loop booting; the microphone is opening.
    STARTING = "STARTING"
    #: Re-armed and listening for the wake word.  The only resting state.
    READY = "READY"
    #: Actively scoring audio for the wake word (a sub-phase of ``READY``).
    WAITING_FOR_WAKE = "WAITING_FOR_WAKE"
    #: The wake word scored above threshold this frame.
    WAKE_DETECTED = "WAKE_DETECTED"
    #: Reading the post-wake command utterance.
    LISTENING = "LISTENING"
    #: Utterance window closed; audio is complete and bounded.
    CAPTURE_COMPLETE = "CAPTURE_COMPLETE"
    #: Local speech-to-text is running on the captured audio.
    TRANSCRIBING = "TRANSCRIBING"
    #: The transcript is with the agent: classification, planning, tools.
    THINKING = "THINKING"
    #: The answer is being spoken.
    SPEAKING = "SPEAKING"
    #: Discarding stale audio and re-arming the detector for the next wake.
    RESETTING = "RESETTING"
    #: A stage failed; the loop is isolating the failure and will re-arm.
    RECOVERING = "RECOVERING"
    #: Unrecoverable for this loop (mic lost, thread dead).
    ERROR = "ERROR"
    #: Intentional shutdown (``stop()`` / daemon Ctrl+C).
    STOPPING = "STOPPING"


#: The states an interaction may legally move to from each state.  ``ERROR`` and
#: ``STOPPING`` are reachable from every state so a failure or a shutdown is
#: never blocked by where the loop happened to be; ``READY`` and ``RECOVERING``
#: are likewise always reachable, which is what guarantees a failed
#: interaction still returns to a listening state.
_TERMINAL: frozenset[VoicePhase] = frozenset({VoicePhase.ERROR, VoicePhase.STOPPING})
_ALWAYS: frozenset[VoicePhase] = _TERMINAL | {
    VoicePhase.READY,
    VoicePhase.RECOVERING,
    VoicePhase.WAITING_FOR_WAKE,
}

#: The happy path, in order.  Used by tests and by the status reporter's
#: ``interaction`` context manager to assert a complete interaction.
HAPPY_PATH: tuple[VoicePhase, ...] = (
    VoicePhase.READY,
    VoicePhase.WAKE_DETECTED,
    VoicePhase.LISTENING,
    VoicePhase.CAPTURE_COMPLETE,
    VoicePhase.TRANSCRIBING,
    VoicePhase.THINKING,
    VoicePhase.SPEAKING,
    VoicePhase.RESETTING,
    VoicePhase.READY,
)

ALLOWED_TRANSITIONS: dict[VoicePhase, frozenset[VoicePhase]] = {
    VoicePhase.STARTING: _ALWAYS | {VoicePhase.READY},
    VoicePhase.READY: _ALWAYS | {VoicePhase.WAKE_DETECTED},
    VoicePhase.WAITING_FOR_WAKE: _ALWAYS | {VoicePhase.WAKE_DETECTED},
    VoicePhase.WAKE_DETECTED: _ALWAYS
    | {
        VoicePhase.LISTENING,
        VoicePhase.SPEAKING,  # "Goodbye." for the stop-listening command
    },
    VoicePhase.LISTENING: _ALWAYS
    | {
        VoicePhase.CAPTURE_COMPLETE,
        VoicePhase.LISTENING,  # a chunked read stays in LISTENING
    },
    VoicePhase.CAPTURE_COMPLETE: _ALWAYS
    | {
        VoicePhase.TRANSCRIBING,
        VoicePhase.RESETTING,  # empty / too-short capture
    },
    VoicePhase.TRANSCRIBING: _ALWAYS
    | {
        VoicePhase.THINKING,
        VoicePhase.RESETTING,  # empty transcript
    },
    VoicePhase.THINKING: _ALWAYS
    | {
        VoicePhase.SPEAKING,
        VoicePhase.RESETTING,  # routed to nothing (dictation/paused)
    },
    VoicePhase.SPEAKING: _ALWAYS | {VoicePhase.RESETTING},
    VoicePhase.RESETTING: _ALWAYS | {VoicePhase.READY},
    VoicePhase.RECOVERING: _ALWAYS | {VoicePhase.RESETTING, VoicePhase.READY},
    VoicePhase.ERROR: frozenset({VoicePhase.STOPPING, VoicePhase.STARTING}),
    VoicePhase.STOPPING: frozenset(),
}

#: Monotonic interaction ids (``i1``, ``i2``, …) so every terminal line and
#: JSONL record for one wake can be correlated.
_interaction_counter = itertools.count(1)


def next_interaction_id() -> str:
    """Return the next monotonic interaction id (``i1``, ``i2``, ...)."""
    return f"i{next(_interaction_counter)}"


def reset_interaction_ids() -> None:
    """Restart the interaction counter (test isolation only)."""
    global _interaction_counter
    _interaction_counter = itertools.count(1)


#: Human-readable console labels.  The enum *value* is the machine vocabulary
#: (``WAKE_DETECTED``) used in JSONL records and in assertions, but the terminal
#: shows what a human reads: ``WAKE DETECTED``.
CONSOLE_LABELS: dict[VoicePhase, str] = {
    VoicePhase.WAKE_DETECTED: "WAKE DETECTED",
    VoicePhase.CAPTURE_COMPLETE: "CAPTURE COMPLETE",
}


def console_label(phase: VoicePhase) -> str:
    """Return the human-readable console label for ``phase``."""
    return CONSOLE_LABELS.get(phase, str(phase))


def check_transition(current: VoicePhase, target: VoicePhase) -> bool:
    """Return True when ``current -> target`` is a legal transition."""
    if current == target:
        return True
    return target in ALLOWED_TRANSITIONS.get(current, frozenset())
