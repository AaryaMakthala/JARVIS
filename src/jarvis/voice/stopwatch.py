"""Onset-triggered bare-cancel detection for the *speaking* state (D22).

Owner decision **D22** overrides ``ROADMAP_V2.1_AMENDMENTS.md.md`` section E: a
bare ``"stop"``, with **no wake word**, must cancel while JARVIS is THINKING and
while it is SPEAKING.  Section E had barge-in respond to the wake word only.

This module holds the one piece of genuinely new logic that D22 needs.  It is a
pure state machine over audio segments - it opens no stream, spawns no thread,
touches no device and imports nothing from :mod:`jarvis.voice.loop`, so the
import graph stays acyclic and it can be unit-tested without a microphone.

Why an *onset* trigger.  While JARVIS is speaking, the loop already reads the
microphone and scores every segment for the wake word; an utterance that is not
the wake word is currently discarded unread.  A bare ``"stop"`` has no wake-word
score to find, so the only way to notice it is ordinary speech energy - the same
:func:`~jarvis.voice.loop._chunk_is_speech` RMS gate the in-flight
``watch_for_stop`` already uses.  Requiring :data:`_MIN_ONSET_FRAMES`
*consecutive* speech frames keeps a click, a door or a chair creak from starting
a transcription of the room.

Self-echo.  The caller only ever feeds this trigger **after** the
``0.35 s`` onset guard (:data:`~jarvis.voice.loop._BARGE_IN_GUARD_S`), so the
answer's own onset cannot trip it.  On open speakers the guard can still be
exceeded; headphones remain the honest recommendation (docs/03 7.2).

Nothing here decides *whether* an utterance is a cancel request.  That stays in
the loop, matched by the fixed vocabulary in
:func:`~jarvis.voice.loop._is_cancel_request` - no LLM, no planner, so a stop
can never be planned as a new task.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable

from jarvis.voice.interfaces import AudioSegment

#: Consecutive speech frames that must be seen before an onset is reported.
#: The speaking loop polls 100 ms chunks (``loop._BARGE_IN_CHUNK_FRAMES``), so 2
#: frames is ~200 ms.  The in-flight watch needs 2 x 80 ms (~160 ms,
#: ``loop._MIN_CANCEL_SPEECH_FRAMES``): marginally stricter here, never looser.
_MIN_ONSET_FRAMES = 2

#: Longest utterance the *speaking* state will transcribe while looking for a
#: bare cancel.  Deliberately shorter than the in-flight watch's
#: ``_STOP_CAPTURE_MAX_S`` (4 s): "stop" is one or two words, and the microphone
#: has to come back to the loop promptly so wake-word barge-in is not delayed.
BARE_CANCEL_MAX_S = 2.0

#: Segments shorter than this carry no usable frame and never start an onset.
_MIN_SEGMENT_S = 0.05

SpeechPredicate = Callable[[AudioSegment], bool]


class OnsetTrigger:
    """Fires once, on the onset of an utterance, given a speech-energy gate.

    Feed one :class:`~jarvis.voice.interfaces.AudioSegment` at a time with
    :meth:`feed`.  It returns the onset samples (the last
    :data:`_MIN_ONSET_FRAMES` speech frames, flattened) the first time an
    utterance begins, and ``None`` on every other call until :meth:`reset`.

    The trigger is **latched**: after firing it stays quiet until reset, so one
    utterance can never be transcribed twice.  A non-speech frame clears the
    run, which is what makes the frames *consecutive* - a click followed by a
    word is two separate onsets, not one.

    Memory is bounded by the frame count, not by utterance length: the onset
    buffer is a :class:`~collections.deque` of at most ``min_frames`` segments
    regardless of how long the caller keeps feeding it.
    """

    def __init__(
        self,
        is_speech: SpeechPredicate,
        *,
        min_frames: int = _MIN_ONSET_FRAMES,
    ) -> None:
        if not callable(is_speech):
            raise TypeError("OnsetTrigger requires a callable is_speech")
        self._is_speech = is_speech
        self._min_frames = max(1, int(min_frames))
        self._frames: deque[list[float]] = deque(maxlen=self._min_frames)
        self._fired = False

    @property
    def fired(self) -> bool:
        """Whether an onset has been reported since the last :meth:`reset`."""
        return self._fired

    def feed(self, segment: AudioSegment) -> list[float] | None:
        """Consume one segment; return the onset samples when it fires.

        Returns ``None`` while the run is still too short, when the segment is
        not speech, and for every call after the first until :meth:`reset`.
        """
        if self._fired:
            return None
        if segment.duration_s < _MIN_SEGMENT_S or not self._is_speech(segment):
            self._frames.clear()
            return None
        self._frames.append(list(segment.samples))
        if len(self._frames) < self._min_frames:
            return None
        self._fired = True
        return [sample for frame in self._frames for sample in frame]

    def reset(self) -> None:
        """Re-arm the trigger after the caller has dealt with an onset.

        Called when an onset turned out not to be a cancel request, so the
        microphone stays watched for a later "stop" instead of going deaf for
        the rest of the answer.
        """
        self._fired = False
        self._frames.clear()
