"""Voice loop: wake → capture → STT → route → TTS.

Runs in a background thread.  The loop:

1. Waits for the wake word.
2. Captures a speech segment with the trailing-silence endpoint in
   :meth:`VoiceLoop._read_command` (there is no separate VAD stage; see
   :data:`_SILENCE_RMS`).
3. Transcribes via STT.
4. Routes the transcript against a fixed vocabulary *in code*, never the
   planner:
   - ``"stop listening"`` / ``"turn off voice"`` / ``"jarvis off"`` → voice off;
   - ``"stop"`` / ``"cancel"`` / ``"never mind"`` → cancel the current speech
     and task, then keep listening (never exits the loop);
   - ``"stop dictation"`` → stops the active dictation session;
   - anything else → submitted to the agent as a voice-sourced task.
5. Speaks the response via TTS, and a ``"Hey Jarvis"`` heard *while* it speaks
   purges the answer (:meth:`VoiceLoop._speak_with_barge_in`).

A Tier 1 confirmation is answered in a bounded **wake-free window**
(:meth:`VoiceLoop.confirm_by_voice`) that the *daemon's* worker thread opens
while this thread waits inside the task submission; the loop never runs a
confirmation dialogue itself.

Reliability contract (this is the part that matters most)
----------------------------------------------------------
**After every successful or recoverable interaction the loop returns to
``READY``, and the next ``"Hey Jarvis"`` works without restarting the
daemon.**  Concretely, that is enforced by:

* every interaction being wrapped in its own ``try``/``except`` (:meth:`_run`),
  so no single failure can end the loop thread;
* an idle timeout being **non-fatal** — it bounds one *wait*, not the loop: the
  loop reports ``RECOVERING``, returns to the armed ``READY`` state and waits
  again (it used to ``break`` out of the loop, which left the daemon alive with
  a permanently dead wake word).  An idle wait is deliberately **not** an
  interaction, so it does not spend the once-per-interaction re-arm (nothing
  was spoken, so there is no response echo to flush and no room to drain);
* :meth:`_rearm` running after *every* interaction, success or failure, in the
  order ``flush → quiet-start drain → wake_detector.reset() → counters``;
* an empty capture, an empty transcript, a dead route, a TTS error and an
  unexpected exception all reporting ``RECOVERING`` and then ``READY``;
* every answer passing through :func:`jarvis.agent.answer.spoken_answer`, so
  an internal routing document or a 900-character search dump is never read
  aloud.

The explicit lifecycle vocabulary lives in :mod:`jarvis.voice.states` and the
console/JSONL reporting in :mod:`jarvis.voice.status`.

No audio data ever leaves this module; only text is passed onward.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

from jarvis.agent.answer import spoken_answer
from jarvis.logging_setup import redact
from jarvis.voice.interfaces import (
    AudioInput,
    AudioSegment,
    SpeechToText,
    TextToSpeech,
    WakeWordDetector,
    WindowFocusChecker,
)
from jarvis.voice.states import VoicePhase, next_interaction_id
from jarvis.voice.status import (
    DEFAULT_WAKE_WORD,
    ProviderReport,
    VoiceStatusReporter,
)
from jarvis.voice.stopwatch import BARE_CANCEL_MAX_S, OnsetTrigger

logger = logging.getLogger(__name__)

# Voice commands (case-insensitive, matched against the *normalised* utterance:
# lower-cased, punctuation removed, whitespace collapsed - so "Stop." matches).
#
# Two vocabularies that must never be mixed:
#   * voice OFF (:data:`_STOP_LISTENING`): the user wants JARVIS to stop
#     listening altogether.  These are the explicit phrasings - the ones the CLI
#     advertises plus "turn off voice".  A bare "turn off" is deliberately NOT
#     here: "turn off the lights" is an ordinary command, not a voice command.
#   * cancel (:data:`_CANCEL_REQUESTS`): "stop"/"cancel"/"never mind" cancel the
#     current speech and the current task and leave the loop listening.
#
# "go to sleep" is absent from both on purpose: sleep is a *mode* change owned
# by Stage 2, so it must never fall through to the voice-off branch here.
_STOP_LISTENING = {
    "stop listening",
    "stop listening jarvis",
    "turn off voice",
    "turn off jarvis",
    "disable voice",
    "jarvis off",
}
_STOP_DICTATION = {"stop dictation", "stop dictating"}
_RESUME_DICTATION = {"resume", "resume dictation"}

#: Spoken phrases accepted as "cancel the task that is running right now".
#: Deliberately a fixed, closed vocabulary matched in code: deciding that
#: "stop" means stop must never be an LLM judgement, and none of these words
#: can be a command on their own.  A bare "stop" at the *command* position
#: (:meth:`VoiceLoop._route`) cancels the speech and the task and returns to
#: wake-listening - it never switches the voice off; while a task is in flight
#: the same vocabulary is matched by :meth:`VoiceLoop.watch_for_stop`.
_CANCEL_REQUESTS = frozenset(
    {
        "stop",
        "stop it",
        "stop that",
        "stop please",
        "cancel",
        "cancel it",
        "cancel that",
        "abort",
        "never mind",
        "nevermind",
    }
)

#: Whole words that mean "cancel the running task", matched anywhere in a
#: short utterance so "Hey Jarvis stop", "please stop" and "stop the task"
#: all work without exact-phrase matching.  Any of these appearing in a
#: *long* utterance is background speech, not an order, so the capture is
#: bounded by :data:`_MAX_CANCEL_TOKENS`.
_CANCEL_TOKENS = frozenset({"stop", "cancel", "abort"})

#: Longest utterance still considered a cancel request.  A short "stop" or
#: "Hey Jarvis, stop, I mean it" counts; a television playing a monologue
#: containing the word "stop" does not.
_MAX_CANCEL_TOKENS = 8

#: Short commands that must survive the unusable-utterance check below even
#: though they carry no intent verb ("yes" is not a verb, "lock" is).  The
#: stop/cancel/voice-off words are here for the same reason: they are matched in
#: code by :meth:`VoiceLoop._route` and must never be rejected as a fragment
#: before routing sees them, and a fragment gate must never be the reason a
#: user's "stop" is ignored.  Multi-word entries are matched token-wise, so
#: "stop" covers "stop listening" and "never" + "mind" cover "never mind".
_SHORT_COMMAND_WORDS = frozenset(
    {
        "yes",
        "no",
        "y",
        "n",
        "ok",
        "okay",
        "sure",
        "nope",
        "nah",
        "stop",
        "cancel",
        "abort",
        "never",
        "mind",
        "nevermind",
        "listening",
        "disable",
        "off",
    }
)

#: Deterministic intent signals: verbs, question words and the small set of
#: function words that turn a fragment into a command.  Used only to reject
#: *short* utterances that contain none of them ("shed on my system"), never as
#: a general-purpose parser or a minimum word count.
_INTENT_WORDS = frozenset(
    {
        # actions
        "open",
        "close",
        "type",
        "write",
        "create",
        "make",
        "delete",
        "remove",
        "rename",
        "copy",
        "move",
        "save",
        "send",
        "start",
        "stop",
        "cancel",
        "abort",
        "lock",
        "unlock",
        "restart",
        "reboot",
        "shut",
        "shutdown",
        "search",
        "find",
        "show",
        "list",
        "read",
        "play",
        "pause",
        "turn",
        "set",
        "get",
        "give",
        "take",
        "increase",
        "decrease",
        "volume",
        "brightness",
        "battery",
        "screenshot",
        "dictate",
        "whatsapp",
        "call",
        "message",
        "email",
        "browse",
        "google",
        "answer",
        "tell",
        # questions / polite framings
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "how",
        "is",
        "are",
        "was",
        "were",
        "do",
        "does",
        "did",
        "can",
        "could",
        "would",
        "should",
        "will",
        "please",
        "help",
        "explain",
    }
)

#: Longest utterance still treated as a *fragment* by the no-intent check.
#: Real commands of this length always contain at least one intent word; a
#: longer utterance is a sentence and is routed rather than guessed at.
_MAX_FRAGMENT_TOKENS = 5

#: "lock my system" / "lock this computer" / … → the canonical request the
#: whole system is calibrated around.  Rewrites only recognised natural
#: variants of the *existing* `lock_computer` action; it never invents a tool
#: and never touches the policy decision (the canonical text still goes through
#: brain → validate → policy_gate → confirmation → act exactly as before).
_LOCK_VARIANT_RE = re.compile(
    r"^(?:please\s+)?lock\s+(?:my|this|the|your|our)\s+"
    r"(?:system|computer|pc|machine|laptop|desktop|screen|workstation|it)$"
)

#: "shut down my system" / "shutdown my system" / …  There is **no** shutdown
#: tool in the registry (``shutdown`` is a hard-blocked name fragment, docs/03),
#: so recognising the phrasing deterministically means answering locally
#: instead of spending an LLM round trip on a plan that cannot exist.
_SHUTDOWN_VARIANT_RE = re.compile(
    r"^(?:please\s+)?(?:shut\s*down|shut|power\s*(?:down|off)|turn\s+off)\s+"
    r"(?:(?:my|this|the|your|our)\s+)?"
    r"(?:system|computer|pc|machine|laptop|desktop|workstation)$"
)

#: Fixed local reply for the shutdown phrasing: honest, no LLM, no tool.
SHUTDOWN_UNSUPPORTED_MSG = (
    "Shutting down the computer is not something I can do. "
    "You can shut down Windows from the Start menu."
)

#: Fixed, local acknowledgement spoken when an in-flight task is cancelled.
#: A constant by design: the cancellation is deterministic, so its wording is
#: owned by the loop rather than generated by the model it just interrupted.
STOP_ACKNOWLEDGEMENT = "Stopped."

#: Spoken prompt for a Tier 1 confirmation.  It names *only* what the user has
#: to say, never the configured wake token: the answer is given inside a
#: wake-free window (see :meth:`VoiceLoop._capture_spoken_answer`), so asking
#: for the wake word here would be asking for something the window does not
#: listen for.  A constant, like the cancellation acknowledgement.
CONFIRMATION_PROMPT = "Say yes to continue, or no to cancel."

#: Spoken when the wake-free window produced no usable answer.  Fixed wording,
#: and always a refusal (fail closed, docs/03 §7.8).
CONFIRMATION_REFUSED = "No response received. Action cancelled."

#: Bounded window for a spoken answer: a Tier 1 yes/no, or a clarification.
#: It opens *after* the prompt has been spoken, so the prompt is never
#: transcribed as the answer, and it needs no wake word.  Bounded twice - by
#: this value and by the trailing-silence end in ``_read_command`` - and an
#: empty/timeout/garbled answer is always a refusal.
DEFAULT_CONFIRM_WINDOW_S = 8.0

#: Spoken words accepted as an approval to a low-risk confirmation prompt
#: (Tier 1, no typed folder-name requirement).  Colloquial "go"/"ok"/"sure"
#: are allowed here because nothing destructive or sensitive can reach this
#: bucket (docs/03 §7.8 + D2): ``can_confirm_by_voice`` fails closed before
#: these ever match.
_LOOSE_YES_WORDS = {"yes", "y", "approve", "approved", "confirm", "go", "ok", "sure"}

#: Spoken words that may approve a destructive/high-risk confirmation.  D2:
#: such an action requires EXACTLY "yes" (plus the unambiguous "confirm" /
#: "proceed"); "go", "ok" and "sure" must never approve one.  Today the voice
#: path cannot reach these (see above), so the strict bucket is a defensive
#: boundary enforced in code and tested directly.
_STRICT_YES_WORDS = {"yes", "confirm", "proceed"}


def _approval_words(payload: dict[str, Any]) -> frozenset[str]:
    """The words that count as "yes" for ``payload``'s confirmation.

    A payload is treated as destructive/high-risk when it demands a typed
    folder-name confirmation or carries a Tier 2+ step.  Anything that reaches
    the voice matcher is normally Tier 1 + untagged, so the loose set applies;
    the strict set is the defensive fallback that can never loosen a refusal.
    """
    if payload.get("typed_confirmation") is not None:
        return _STRICT_YES_WORDS
    try:
        if int(payload.get("tier", 0)) >= 2:
            return _STRICT_YES_WORDS
    except (TypeError, ValueError):
        pass
    return _LOOSE_YES_WORDS


#: Session limits enforced by the loop (defaults, overridable per instance).
DEFAULT_MAX_SESSION_S = 1800.0
DEFAULT_MAX_DICTATION_CHARS = 20000

#: Longest single command window captured before STT (30 s, as in config.toml).
DEFAULT_MAX_SEGMENT_S = 30.0

#: Capture chunk length (100 ms at 16 kHz) read for silence detection.
_CAPTURE_CHUNK_FRAMES = 1600

#: Maximum pre-speech audio kept so whisper still sees the utterance onset
#: (200 ms at 16 kHz = 2 capture chunks).  Everything before that is dropped.
_CAPTURE_PREROLL_FRAMES = 2 * _CAPTURE_CHUNK_FRAMES

#: Heartbeat cadence for the "still alive" / audio-level INFO line (~3 s).
_WAKE_HEARTBEAT_S = 3.0

#: Bounded drain applied after every interaction so the re-armed wake detector
#: starts scoring from the same quiet baseline as the very first wake, instead
#: of over the tail of the response still ringing in the room/mic.  Event-
#: driven (the drain ends the moment a quiet chunk arrives) and capped so a
#: persistently noisy room can never wedge the loop.
DEFAULT_REARM_QUIET_GATE_S = 0.5
_REARM_QUIET_GATE_S = DEFAULT_REARM_QUIET_GATE_S

#: RMS (float samples in [-1, 1]) below which a chunk counts as silence.
_SILENCE_RMS = 0.01

#: Chunk read from the microphone while the answer is being spoken, so the wake
#: detector can score a barge-in ("Hey Jarvis") over the speakers.  Same 100 ms
#: granularity the wake wait uses, so barge-in latency matches wake latency.
_BARGE_IN_CHUNK_FRAMES = _CAPTURE_CHUNK_FRAMES

#: Grace period after speech starts during which a wake detection is ignored.
#: The first words of the answer are the most likely to be mistaken for a wake
#: (an answer that starts "Hey..." or simply the speaker's own onset), so
#: barge-in is armed only once the answer is actually under way.  This is a
#: short guard, not echo cancellation: with speakers on, the wake model can
#: still score the answer itself, so headphones are recommended (docs/09).
_BARGE_IN_GUARD_S = 0.35

#: Hard wall-clock ceiling on one spoken answer under barge-in.  The loop is
#: normally released by the engine reporting that speech finished, but a wedged
#: engine must never hold the voice thread (and the microphone) forever: at the
#: ceiling the speech is purged and the loop moves on.
_BARGE_IN_MAX_S = 300.0

#: Default end-of-command silence.  Was 0.7 s, which ended capture on a normal
#: mid-sentence hesitation ("what is the capacity of" + 0.7 s pause shipped
#: "Capacity of-" to the planner).  1.2 s clears an ordinary breath/pause while
#: still ending a finished command promptly; ``silence_timeout_s`` remains the
#: single knob, and ``max_segment_s`` still bounds the whole utterance.
DEFAULT_SILENCE_TIMEOUT_S = 1.2

#: Fixed, local "please repeat" prompt.  A constant by design: when the
#: transcript cannot be used the retry text must never be generated by the LLM.
REPEAT_PROMPT = "Could you repeat that?"

#: Trailing words that mean the transcription stopped mid-sentence (a dangling
#: article, conjunction, auxiliary or preposition).  Deliberately small and
#: unambiguous: it excludes "on"/"in"/"up"/"out"/"it" so natural commands
#: such as "turn it on" or "turn the volume up" are never rejected.
_DANGLING_TRAILERS = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "to",
        "for",
        "with",
        "from",
        "about",
        "and",
        "or",
        "but",
        "is",
        "are",
        "was",
        "were",
    }
)

#: Characters whisper uses when it had to cut a word or the audio ran out.
_TRUNCATION_MARKERS = ("-", "–", "—", "…", "...")

#: Edit distance within which a Latin transcript counts as the wake phrase
#: itself ("hey charvis", "hay jarvis") rather than a command.
_WAKE_MATCH_MAX_DISTANCE = 2

#: Longest non-Latin fragment considered "not a command".  The observed leak
#: was whisper rendering the wake phrase as the single Arabic word "هداريس"
#: (~"hadaris"); a longer non-Latin transcript is more likely a real (if
#: unsupported) utterance, so it is routed rather than refused.
_MAX_SCRIPTLESS_TOKENS = 3

#: Longest answer handed to TTS.  Reading a multi-source web-search dump aloud
#: produced 79 seconds of dead air; the full text is still reported on the
#: console (truncated) and kept in the task result.
DEFAULT_MAX_SPOKEN_CHARS = 300

#: Wake-model frame length (80 ms at 16 kHz) - the chunk openWakeWord scores.
_WAKE_CHUNK_FRAMES = 1280

#: Longest utterance captured after an in-flight wake trigger when looking for
#: a cancel request, and how long to wait for it to start.  Both are short on
#: purpose: this is a cancel path, not a command window, and the loop must
#: return to the task it is watching as soon as possible.
_STOP_CAPTURE_MAX_S = 4.0
_STOP_LISTEN_TIMEOUT_S = 5.0

#: Consecutive speech-energy frames (~160 ms) the in-flight watch requires before
#: it starts capturing.  One frame is a click, a door, a cough: two in a row is
#: a voice.  Kept local to the watch rather than reusing the wake wait's own
#: gate, which also has to *score* a model and so is much more expensive.
_MIN_CANCEL_SPEECH_FRAMES = 2

#: Pause before re-entering the wake wait after an idle timeout.  The idle
#: deadline is measured in *audio* seconds, so a stream that delivers frames
#: instantly (a test fake, or a device that replays a cached buffer) would
#: otherwise satisfy it thousands of times a second and busy-spin one core.
#: Yielding for a fixed period bounds that to a few waits per second.  On a real
#: microphone the pause is imperceptible because the deadline takes
#: ``idle_timeout_s`` of real audio to elapse in the first place.  Overridable
#: per instance (see ``VoiceLoop(idle_backoff_s=...)``) so a test can shrink it
#: without changing the production default.
_IDLE_BACKOFF_S = 0.5

#: Pause after a wake read that returned *no* samples at all.  A stalled or
#: half-closed device can return empty reads forever; without this the wake wait
#: would hot-spin a core doing no work, and with ``idle_timeout_s`` disabled
#: nothing would bound it at all.
_EMPTY_READ_BACKOFF_S = 0.02

#: Extra wall-clock allowance on top of the audio-second idle deadline, so a
#: completely stalled stream (no audio at all) still yields and re-arms instead
#: of waiting forever.
_IDLE_WALL_SLACK_S = 5.0


def _segment_rms(segment: AudioSegment) -> float:
    """Root-mean-square energy of ``segment.samples`` (0.0 for empty audio)."""
    samples = segment.samples
    if not samples:
        return 0.0
    acc = 0.0
    for sample in samples:
        acc += sample * sample
    return (acc / len(samples)) ** 0.5


def _chunk_is_speech(segment: AudioSegment, threshold: float = _SILENCE_RMS) -> bool:
    """Return True when ``segment`` carries voice-level energy.

    ``threshold`` is the RMS energy below which a chunk counts as silence;
    the loop passes its configured ``silence_threshold`` (config.toml
    ``[voice] silence_threshold``) so a noisy room can be tuned without
    code changes.
    """
    return _segment_rms(segment) >= threshold


def _normalise_speech(text: str) -> str:
    """Lower-case, strip punctuation and collapse whitespace.

    Keeps letters and digits from any script, so the wake-word comparison and
    the "no Latin letter" check both see the real words.
    """
    kept = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text.lower())
    return " ".join(kept.split())


def _edit_distance(left: str, right: str) -> int:
    """Levenshtein distance between two short strings (no dependency)."""
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, lch in enumerate(left, start=1):
        current = [i]
        for j, rch in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (0 if lch == rch else 1),
                )
            )
        previous = current
    return previous[-1]


def _has_latin_or_digit(text: str) -> bool:
    """Whether ``text`` contains an ASCII letter or any digit."""
    return any(("a" <= ch <= "z") or ch.isdigit() for ch in text)


def _transcript_problem(text: str, wake_word: str) -> str | None:
    """Return why ``text`` must not reach the agent, or ``None`` to route it.

    Deterministic and local: no LLM, no minimum word count, no retry loop.  It
    refuses only the transcripts where acting would be a guess, and the loop
    answers them with :data:`REPEAT_PROMPT` instead of a plan:

    * empty or punctuation-only audio;
    * the wake phrase itself — the wake word leaked into the command window, or
      the user repeated it because the "Yes?" ack was missed;
    * a short fragment with no Latin letter or digit, e.g. whisper's "هداريس"
      for the wake phrase on a quiet mic (an English-configured assistant
      cannot act on it, so it asks instead);
    * a transcript cut off mid-sentence ("what is the capacity of-"), detected
      from whisper's own trailing hyphen/ellipsis or a dangling trailer word;
    * a **short fragment with no intent word at all** ("shed on my system."):
      within a few tokens a real command always carries a verb, a question
      word or one of :data:`_SHORT_COMMAND_WORDS`, so a fragment without one
      would be a guess.  This is deliberately *not* a minimum word count:
      "stop", "yes", "lock" and "open notepad" all pass.

    Legitimate short commands ("lock my computer", "stop", "yes", "what time
    is it", "turn it on") match none of these and route unchanged.
    """
    cleaned = text.strip()
    if not cleaned:
        return "empty transcript"

    normalised = _normalise_speech(cleaned)
    if not normalised:
        return "no recognisable words"

    wake = _normalise_speech(wake_word)
    if wake and (
        normalised == wake
        or _edit_distance(normalised.replace(" ", ""), wake.replace(" ", ""))
        <= _WAKE_MATCH_MAX_DISTANCE
    ):
        return "wake phrase captured instead of a command"

    tokens = normalised.split()
    if len(tokens) <= _MAX_SCRIPTLESS_TOKENS and not _has_latin_or_digit(normalised):
        return "transcript is not recognisable as a command"

    if cleaned.endswith(_TRUNCATION_MARKERS) or tokens[-1] in _DANGLING_TRAILERS:
        return "transcript ended mid-sentence"

    # Fail fast on an obvious fragment: a short utterance with no intent word
    # and no whitelisted short command would only be guessed at by the planner
    # (a whole LLM round trip to reach "Could you repeat that?" anyway).
    if len(tokens) <= _MAX_FRAGMENT_TOKENS and not (
        _SHORT_COMMAND_WORDS & set(tokens) or _INTENT_WORDS & set(tokens)
    ):
        return "transcript has no recognisable command"
    return None


def _canonical_command(text: str) -> str:
    """Rewrite a recognised natural variant to the canonical phrasing.

    Deterministic and deliberately tiny: only the phrasings the live voice path
    kept mis-routing are matched, and each maps to wording the system already
    supports (``lock my system`` -> ``lock my computer`` -> the existing
    ``lock_computer`` action).  The rewritten text still travels the normal
    brain -> validate -> policy_gate -> confirmation -> act path, so nothing is
    bypassed.  Anything unrecognised is returned unchanged - this is not a
    general-purpose parser.
    """
    if _LOCK_VARIANT_RE.match(_normalise_speech(text)):
        return "lock my computer"
    return text


def is_shutdown_request(text: str) -> bool:
    """Whether ``text`` asks to power the machine off (no shutdown tool exists).

    Recognising the phrasing in code lets the loop answer honestly and locally
    instead of spending an LLM round trip on a plan that cannot exist:
    ``shutdown`` is a hard-blocked tool-name fragment (docs/03), so there is
    nothing to map it to.
    """
    return bool(_SHUTDOWN_VARIANT_RE.match(_normalise_speech(text)))


def _is_cancel_request(text: str) -> bool:
    """Whether a captured utterance is an order to cancel the running task.

    Pure, fixed-vocabulary matching in code: deciding that "stop" means *stop
    the task that is running right now* must never be an LLM judgement, and a
    cancel request must never be submitted to the planner (that would start a
    second task instead of stopping the first).

    Two layers, deliberately narrow:

    * the whole normalised utterance is in :data:`_CANCEL_REQUESTS` ("stop
      that", "never mind"), so a bare order is unambiguous;
    * otherwise a cancel *word* anywhere in a short utterance counts ("Hey
      Jarvis stop", "please stop the task").  The length bound
      (:data:`_MAX_CANCEL_TOKENS`) is what keeps background speech out: a
      television monologue that happens to contain "stop" is not an order, and
      a long utterance is not something this method gets to reinterpret.

    Anything else - including an empty transcript - is ``False``, and the
    caller leaves the running task alone.
    """
    normalised = _normalise_speech(text)
    if not normalised:
        return False
    if normalised in _CANCEL_REQUESTS:
        return True
    tokens = normalised.split()
    return len(tokens) <= _MAX_CANCEL_TOKENS and bool(_CANCEL_TOKENS & set(tokens))


class VoiceLoop:
    """Orchestrates the voice pipeline in a background thread.

    Use :meth:`start` / :meth:`stop` / :meth:`is_active` from any
    thread.  The loop itself runs in its own daemon thread.
    """

    def __init__(
        self,
        *,
        audio: AudioInput,
        wake_detector: WakeWordDetector | None = None,
        stt: SpeechToText,
        tts: TextToSpeech,
        focus_checker: WindowFocusChecker | None = None,
        submit_task: Callable[[str, str], Any] | None = None,
        on_dictation: Callable[[str], None] | None = None,
        on_exit: Callable[[str | None], None] | None = None,
        wake_word: str = "hey jarvis",
        listen_timeout_s: float = 30.0,
        idle_timeout_s: float = 120.0,
        max_session_s: float = DEFAULT_MAX_SESSION_S,
        max_dictation_chars: int = DEFAULT_MAX_DICTATION_CHARS,
        silence_timeout_s: float = DEFAULT_SILENCE_TIMEOUT_S,
        max_segment_s: float = DEFAULT_MAX_SEGMENT_S,
        silence_threshold: float = _SILENCE_RMS,
        rearm_quiet_gate_s: float = _REARM_QUIET_GATE_S,
        confirm_window_s: float = DEFAULT_CONFIRM_WINDOW_S,
        bare_stop_while_busy: bool = True,
        idle_backoff_s: float = _IDLE_BACKOFF_S,
        reporter: VoiceStatusReporter | None = None,
        report_status: bool = True,
        echo_transcript: bool = True,
        max_spoken_chars: int = DEFAULT_MAX_SPOKEN_CHARS,
        routing_report: Callable[[], ProviderReport | None] | None = None,
    ) -> None:
        self._audio = audio
        self._wake_detector = wake_detector
        self._stt = stt
        self._tts = tts
        self._focus_checker = focus_checker
        self._submit_task = submit_task
        self._on_dictation = on_dictation
        self._on_exit = on_exit
        self._wake_word = wake_word
        self._listen_timeout_s = listen_timeout_s
        self._idle_timeout_s = idle_timeout_s
        self._max_session_s = max_session_s
        self._max_dictation_chars = max_dictation_chars
        self._silence_timeout_s = max(silence_timeout_s, 0.0)
        self._max_segment_s = max_segment_s
        self._silence_threshold = max(silence_threshold, 0.0)
        #: Bounded quiet-start drain length, 0 disables it (see the constant).
        self._rearm_quiet_gate_s = max(rearm_quiet_gate_s, 0.0)
        #: Bounded wake-free window for a spoken answer (see
        #: :data:`DEFAULT_CONFIRM_WINDOW_S`).
        self._confirm_window_s = max(confirm_window_s, 0.0)
        #: D22: also honour a bare "stop" while speaking.  Gates only the
        #: speaking-state capability; the in-flight watch is unconditional.
        self._bare_stop_while_busy = bool(bare_stop_while_busy)
        #: Pause between a non-fatal idle timeout and the next wake wait.
        self._idle_backoff_s = max(idle_backoff_s, 0.0)
        #: Longest utterance handed to TTS (see :func:`spoken_answer`).
        self._max_spoken_chars = max(int(max_spoken_chars), 40)
        #: Optional provider/model probe, called after the agent answered, so
        #: the console can show which provider actually served the request.
        self._routing_report = routing_report
        self._reporter = reporter or VoiceStatusReporter(
            stream="auto",
            enabled=report_status,
            echo_transcript=echo_transcript,
            wake_word=wake_word or DEFAULT_WAKE_WORD,
        )

        self._running = False
        self._thread: threading.Thread | None = None
        self._dictating = False
        self._dictation_paused = False
        self._dictation_target_app: str = ""
        self._dictation_started: float = 0.0
        self._dictation_chars: int = 0
        self._idle_timed_out = False
        self._stop_event = threading.Event()
        self._stopped = threading.Event()
        self._audio_opened = False
        self._wake_frames_read = 0
        #: True while an in-flight stop watch session has armed the detector.
        #: See :meth:`watch_for_stop`: the detector must be re-armed once per
        #: session (its rolling window is what scores the wake word), never
        #: once per slice.
        self._stop_watch_armed = False
        self._phase_state = "off"
        self._interaction_id = ""

    # ── public API ──────────────────────────────────────────────────────

    @property
    def reporter(self) -> VoiceStatusReporter:
        """The lifecycle reporter (console + JSONL) used by this loop."""
        return self._reporter

    @property
    def interaction_id(self) -> str:
        """Id of the interaction currently running (empty between wakes)."""
        return self._interaction_id

    def _set_state(self, state: str) -> None:
        """Record and log a voice-phase transition.

        The state string is one of a fixed set (LISTENING / WAKE_DETECTED /
        CAPTURING / TRANSCRIBING / PROCESSING / SPEAKING / RESETTING) so a
        session's ``jarvis.jsonl`` shows exactly which phase each interaction
        is in and whether it ever returns to wake-listening.  No audio
        content or transcript wording is ever included.
        """
        self._phase_state = state
        logger.info("VOICE state=%s", state)

    def _recover(self, reason: str) -> None:
        """Report an isolated failure that the loop will recover from.

        Called from every per-phase guard so a failed interaction is visibly
        distinguishable from a successful one, and is always followed by the
        caller's re-arm back to ``READY``.
        """
        self._reporter.recovering(reason)

    def _ask_to_repeat(self, reason: str) -> None:
        """Speak the fixed local retry prompt; the transcript never reaches the agent.

        Used when the STT usability gate refuses a transcript (wake-only,
        garbled or clearly incomplete).  The prompt is :data:`REPEAT_PROMPT`, a
        module constant — no LLM, no plan and no tool call is involved.  The
        caller re-arms exactly once afterwards, so an unusable transcript costs
        one prompt and one ``RESETTING -> READY``, never a retry loop.
        """
        logger.info("VOICE STT_UNUSABLE")
        logger.info("voice boundary: stt unusable (%s); asking the user to repeat", reason)
        self._recover(f"could not use the command ({reason})")
        self._reporter.retry_prompt(REPEAT_PROMPT)
        try:
            self._tts.speak(REPEAT_PROMPT)
        except Exception:
            logger.exception("voice interaction failed at repeat-prompt tts; continuing to listen")

    def start(self) -> None:
        """Start the voice loop in a background thread (idempotent)."""
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._stopped.clear()
        self._idle_timed_out = False
        self._thread = threading.Thread(target=self._run, daemon=True, name="voice-loop")
        self._thread.start()
        logger.info("voice loop started (wake_word=%s)", self._wake_word)

    def stop(self) -> None:
        """Stop the voice loop and wait for the thread to finish.

        Waits on a ``_stopped`` event (set in the loop thread's ``finally``)
        instead of a fixed join so the audio device is never closed while the
        loop thread might still be reading from it.
        """
        if not self._running:
            return
        self._running = False
        self._stop_event.set()
        self._tts.stop()
        if not self._stopped.wait(timeout=10.0):
            logger.warning("voice loop thread did not stop within 10s; leaving audio open")
            self._reporter.error("loop thread did not stop within 10s")
            return
        self._audio.close()
        self._reporter.stopping()
        logger.info("voice loop stopped")

    def is_active(self) -> bool:
        """Return True while the loop thread is running."""
        return self._running

    @property
    def is_dictating(self) -> bool:
        """Return True when a dictation session is active."""
        return self._dictating

    def start_dictation(self, target_app: str = "notepad") -> None:
        """Begin a dictation session (called by the agent tool)."""
        self._dictating = True
        self._dictation_paused = False
        self._dictation_target_app = target_app
        self._dictation_started = time.monotonic()
        self._dictation_chars = 0
        logger.info("dictation started (target=%s)", target_app)

    def stop_dictation(self) -> None:
        """End the current dictation session."""
        self._dictating = False
        self._dictation_paused = False
        self._dictation_target_app = ""
        self._dictation_started = 0.0
        self._dictation_chars = 0
        logger.info("dictation stopped")

    def can_confirm_by_voice(self, payload: dict[str, Any]) -> bool:
        """Whether ``payload`` may be approved by voice (Tier 1 only).

        Fails closed: Tier 2+ and typed folder-name confirmations always need
        the terminal.  Unknown / missing fields are treated as a refusal.
        """
        tier = payload.get("tier", 0)
        try:
            tier = int(tier)
        except (TypeError, ValueError):
            return False
        if tier >= 2:
            return False
        return payload.get("typed_confirmation") is None

    def confirm_by_voice(
        self,
        payload: dict[str, Any],
        *,
        on_confirmation: Callable[[dict[str, Any]], Any],
        rearm_timeout_s: float | None = None,
    ) -> str:
        """Ask the user to approve/reject a confirmation by voice.

        The prompt is spoken, and only then does a **bounded wake-free window**
        open (see :meth:`_capture_spoken_answer`): no wake word is required to
        answer, which is what makes a spoken confirmation usable at all.  The
        window is opened *after* the prompt so the answer cannot be the prompt's
        own echo, and every failure mode is a refusal - a missing callback, an
        action that may not be confirmed by voice, silence, a timeout, or an
        answer that is not an approval word.

        ``on_confirmation`` is **required** and receives the resume answer
        ``{"approved": bool, "action_hash": str}`` exactly once.  There is no
        loop-level fallback callback: only the caller that published the
        pending confirmation can consume the answer, so a confirmation can never
        be approved by a listener that is not waiting for it.
        """
        payload = dict(payload or {})
        summary = payload.get("summary") or ""
        action_hash = payload.get("action_hash") or ""
        callback = on_confirmation

        if not self.can_confirm_by_voice(payload):
            logger.info("voice confirmation refused: action is not Tier 1")
            callback({"approved": False, "action_hash": action_hash})
            return f"Action requires terminal confirmation: {summary}"

        window_s = self._confirm_window_s if rearm_timeout_s is None else max(rearm_timeout_s, 0.0)
        try:
            self._tts.speak(f"{summary}. {CONFIRMATION_PROMPT}" if summary else CONFIRMATION_PROMPT)
        except Exception:
            logger.exception("voice confirmation prompt failed; refusing")
            callback({"approved": False, "action_hash": action_hash})
            return CONFIRMATION_REFUSED

        answer = self._capture_spoken_answer(window_s, kind="confirmation")
        if answer is None:
            # Distinct from a user refusal on purpose: nothing usable came back
            # from the window (silence, too short, or an empty transcript).
            # Both refuse, but only this one is the system's own doing.
            logger.info("voice confirmation refused: no usable answer in the window")
            callback({"approved": False, "action_hash": action_hash})
            return CONFIRMATION_REFUSED

        approved = answer in _approval_words(payload)
        callback({"approved": approved, "action_hash": action_hash})
        if approved:
            logger.info("voice confirmation approved")
            return "Approved."
        logger.info("voice confirmation refused by user: answer is not an approval word")
        return "Cancelled."

    def _capture_spoken_answer(self, window_s: float, *, kind: str) -> str | None:
        """Capture one spoken answer inside a bounded, wake-free window.

        ``kind`` is only for the log ("confirmation" / "clarification").  The
        window is bounded twice - by ``window_s`` of wall clock/samples and by
        the trailing-silence end in :meth:`_read_command` - and the microphone
        is flushed and drained back to a quiet baseline first, so neither the
        prompt's own echo nor stale ambient audio can be transcribed as the
        answer.

        Returns the normalised answer, or ``None`` for every failure (no
        speech, too short, transcription error, empty transcript).  ``None`` is
        always the *refusing* direction for both callers, so a window that
        mishears something fails closed.

        Every ``None`` is reported to the console with the reason it happened.
        A refusal is a safety decision the owner has to be able to audit, and
        "heard nothing" and "transcribed to nothing" are different failures
        that used to collapse into the same silent refusal.
        """
        logger.info("voice boundary: %s window open (%.1fs, no wake word)", kind, window_s)
        try:
            self._audio.flush()
        except Exception:
            logger.debug("%s window: audio flush failed", kind, exc_info=True)
        try:
            self._quiet_start_drain()
        except Exception:
            logger.debug("%s window: quiet drain failed", kind, exc_info=True)
        try:
            segment = self._read_command(
                int(window_s * 16_000),
                listen_timeout_s=window_s,
            )
        except Exception:
            logger.exception("%s window: capture failed", kind)
            self._report_window_answer(kind, "", outcome="capture failed")
            return None
        if segment.duration_s < 0.3:
            reason = "no speech detected" if segment.duration_s <= 0.0 else "too short to use"
            logger.info("voice boundary: %s window heard nothing (%s)", kind, reason)
            self._report_window_answer(kind, "", outcome=reason)
            return None
        try:
            result = self._stt.transcribe(segment)
        except Exception:
            logger.exception("%s window: transcription failed", kind)
            self._report_window_answer(kind, "", outcome="transcription failed")
            return None
        answer = _normalise_speech(result.text)
        logger.info("voice boundary: %s window closed (answer=%d chars)", kind, len(answer))
        if not answer:
            # Speech reached the recogniser but came back empty: a distinct
            # failure from silence, and the one that makes a short spoken
            # "no" indistinguishable from not having spoken at all.
            logger.info("voice boundary: %s window: speech captured but nothing recognised", kind)
            self._report_window_answer(kind, "", outcome="speech heard, nothing recognised")
            return None
        self._report_window_answer(kind, answer, outcome="captured")
        return answer

    def _report_window_answer(self, kind: str, answer: str, *, outcome: str) -> None:
        """Show one wake-free window's answer and outcome on the console.

        The words never reach the log (see
        :meth:`~jarvis.voice.status.VoiceStatusReporter.confirmation_answer`);
        reporting must never break the window, so any failure here is swallowed.
        """
        try:
            self._reporter.confirmation_answer(kind, answer, outcome=outcome)
        except Exception:
            logger.debug("%s window: answer reporting failed", kind, exc_info=True)

    def capture_free_text(
        self,
        prompt: str,
        *,
        on_answer: Callable[[str], Any] | None = None,
        rearm_timeout_s: float | None = None,
    ) -> str:
        """Ask a question and capture a spoken free-text answer.

        Used for clarification interrupts, and the same bounded wake-free
        window as a confirmation: the prompt is spoken, then the window opens.
        Fails closed: silence, a timeout, a transcription error or an empty
        transcript returns ``""`` - the caller resumes the graph with empty
        text and the clarify node reports "No clarification received.".

        ``on_answer`` (optional) receives the transcript once, including the
        empty string when the window failed closed, so callers can rely on
        exactly one callback per prompt.
        """
        window_s = self._confirm_window_s if rearm_timeout_s is None else max(rearm_timeout_s, 0.0)

        def report(answer: str) -> None:
            if on_answer is not None:
                on_answer(answer)

        try:
            self._tts.speak(prompt)
        except Exception:
            logger.exception("voice clarification prompt failed; answering empty (fail closed)")
            report("")
            return ""
        answer = self._capture_spoken_answer(window_s, kind="clarification")
        if answer is None:
            report("")
            return ""
        report(answer)
        return answer

    # ── main loop (runs in background thread) ───────────────────────────

    def _run(self) -> None:
        """Main voice loop body.

        Any unhandled exception is logged and reported through the
        ``on_exit`` callback with a fixed reason code so the owning
        service can move the state machine to ``error``.  The audio device
        is always closed in ``finally`` — even on a crash — so the mic is
        never left held by a dead thread.
        """
        reason: str | None = None
        try:
            self._reporter.state(VoicePhase.STARTING)
            self._audio.open()
            self._audio_opened = True
            self._set_state("LISTENING")
            if self._wake_detector is not None:
                self._wake_detector.reset()
            self._reporter.ready(self._wake_word)

            while self._running and not self._stop_event.is_set():
                # Phase 1: Wait for wake word (or voice activity)
                logger.info("VOICE WAKE_WAIT_RESTART")
                if not self._wait_for_wake():
                    if self._idle_timed_out:
                        # Non-fatal by design.  The idle timeout exists only so
                        # one silent wait does not hold the microphone open
                        # forever, so it releases the device and waits again.
                        # It used to ``break`` out of the loop, which left the
                        # daemon running with voice permanently dead: the wake
                        # word never worked again until a restart.
                        logger.info(
                            "voice loop idle timeout reached; releasing mic and "
                            "resuming the wake wait"
                        )
                        self._idle_timed_out = False
                        self._resume_after_idle()
                    continue

                # One interaction is isolated: an unexpected exception in any
                # phase is logged and the loop is re-armed to wake-listening,
                # then the worker keeps running.  No single failed or malformed
                # interaction may disable voice.  The wake detector is re-armed
                # EXACTLY once, here in ``_rearm``, after every interaction
                # (success or failure) — never per audio frame.
                try:
                    self._run_interaction()
                except Exception:
                    logger.exception("voice interaction failed; resetting and continuing to listen")
                    self._recover("unexpected failure inside the interaction")

                # Re-arm for the next wake: reset the wake-word model's rolling
                # buffer and discard audio captured while the response was
                # spoken, so a second "hey jarvis" is detected fresh and late
                # TTS output is never mistaken for a follow-up command.  A
                # failure here must not kill the loop.
                try:
                    self._rearm()
                except Exception:
                    logger.exception("voice re-arm failed after interaction; continuing to wait")
                    self._recover("re-arm failed; retrying")

        except Exception:
            logger.exception("voice loop crashed")
            reason = "mic-open-failed" if not self._audio_opened else "loop-crashed"
            self._reporter.error(reason)
        finally:
            self._running = False
            self._idle_timed_out = False
            try:
                self._audio.close()
            except Exception:
                logger.debug("error closing audio on exit", exc_info=True)
            if self._on_exit is not None:
                try:
                    self._on_exit(reason)
                except Exception:
                    logger.debug("voice loop on_exit callback failed", exc_info=True)
            self._stopped.set()

    def _wait_for_wake(self) -> bool:
        """Block until the wake word is detected.

        Returns True if the wake word was detected, False if the loop should
        stop (stop request) or the idle deadline elapsed.  An idle timeout is
        **not** fatal: the caller re-enters the wait, so the deadline bounds one
        wait, not the lifetime of the loop.

        The deadline is measured in **audio seconds actually consumed**, with a
        wall-clock slack on top.  Measuring it in audio keeps the re-arm rate
        bounded when a stream delivers frames faster than real time (a test
        fake, or a device replaying a buffer) — otherwise a wall-clock-only
        deadline is satisfied thousands of times a second and the loop
        busy-spins a core while doing nothing.  The slack still releases the
        microphone when the stream delivers no audio at all.  A stop request
        is honoured promptly either way, because both bounds are re-checked on
        the cancellable poll loop.

        A transient exception from the wake detector itself is logged and the
        detector re-armed, never allowed to kill the loop: model hiccups must
        not disable voice.  A failure of the underlying microphone read still
        surfaces as ``loop-crashed``.
        """
        logger.info("VOICE WAKE_WAIT_START")
        self._idle_timed_out = False
        self._reporter.waiting_for_wake()
        if self._wake_detector is None:
            # No wake-word model; use a simple voice-activity gate.
            segment = self._audio.read(4800)  # 300 ms
            if segment.duration_s > 0 and not self._stop_event.is_set():
                self._set_state("WAKE_DETECTED")
                self._reporter.wake_detected()
            return segment.duration_s > 0 and not self._stop_event.is_set()

        wait_start = time.monotonic()
        wall_deadline: float | None = None
        if self._idle_timeout_s > 0:
            wall_deadline = wait_start + self._idle_timeout_s + _IDLE_WALL_SLACK_S
        audio_s = 0.0

        # Read chunks and feed to the wake-word model.
        chunk_frames = 1280  # 80 ms at 16 kHz
        last_heartbeat = wait_start
        while self._running and not self._stop_event.is_set():
            if self._idle_timeout_s > 0 and (
                audio_s >= self._idle_timeout_s
                or (wall_deadline is not None and time.monotonic() >= wall_deadline)
            ):
                self._idle_timed_out = True
                self._idle_backoff()
                return False
            segment = self._audio.read(chunk_frames)
            if segment.duration_s < 0.05:
                # An *empty* read means the stream is stalled or half-closed: it
                # advances neither the audio clock nor the model, so yield
                # instead of spinning on it (see :meth:`_empty_read_backoff`).
                # A merely short read is real audio from a source that returns
                # less than was asked for, so it is discarded without a pause.
                if not segment.samples:
                    self._empty_read_backoff()
                logger.debug(
                    "voice boundary: wake read too short (duration_s=%.3f)",
                    segment.duration_s,
                )
                continue
            self._wake_frames_read += len(segment.samples)
            audio_s += segment.duration_s
            logger.debug(
                "voice boundary: wake frame (frames=%d duration_s=%.3f)",
                len(segment.samples),
                segment.duration_s,
            )
            if time.monotonic() - last_heartbeat >= _WAKE_HEARTBEAT_S:
                last_heartbeat = time.monotonic()
                logger.info(
                    "voice boundary: waiting for wake word "
                    "(frames_read=%d elapsed_s=%.1f audio_s=%.1f audio_rms=%.4f)",
                    self._wake_frames_read,
                    time.monotonic() - wait_start,
                    audio_s,
                    _segment_rms(segment),
                )
            try:
                result = self._wake_detector.detect(segment)
            except Exception:
                logger.exception("voice boundary: wake detector error; re-arming and continuing")
                self._recover("wake detector error; re-arming")
                self._rearm()
                continue
            if result.detected:
                logger.info("VOICE wake_detected")
                logger.info("VOICE WAKE_DETECTED")
                logger.info("voice boundary: wake trigger; entering interaction")
                self._set_state("WAKE_DETECTED")
                self._reporter.wake_detected()
                return True
        return False

    def _idle_backoff(self) -> None:
        """Yield briefly after an idle timeout so re-arming cannot busy-spin.

        A cancellable wait, so ``stop()`` is honoured immediately.
        """
        self._stop_event.wait(self._idle_backoff_s)

    def _empty_read_backoff(self) -> None:
        """Yield briefly when a wake read returned no audio at all.

        A stalled or half-closed microphone can return empty reads
        indefinitely.  Without this the wake wait would spin a core flat with
        no work done — and with ``idle_timeout_s`` disabled nothing would bound
        it.  Cancellable, so ``stop()`` is honoured immediately.
        """
        self._stop_event.wait(_EMPTY_READ_BACKOFF_S)

    def _resume_after_idle(self) -> None:
        """Return to the armed wake wait after a non-fatal idle timeout.

        Deliberately *not* :meth:`_rearm`.  An idle wait is not an interaction:
        nothing was spoken, so there is no response echo to flush and no room
        state to drain back to a quiet baseline, and the detector has been
        scoring silence the whole time.  Running the post-interaction reset here
        would spend the once-per-interaction re-arm on a wake that never
        happened and make "re-armed exactly once per interaction"
        unobservable.

        What the loop must guarantee is that it stays *armed*: this reports
        ``RECOVERING`` then ``READY``, so the next "hey jarvis" is still
        detected and a machine that stays silent for hours never needs the
        daemon restarted.
        """
        self._recover("idle timeout; still armed for the next wake word")
        self._reporter.ready(self._wake_word)

    def _run_interaction(self) -> None:
        """Run one wake-activated interaction (ack → capture → STT → route → TTS).

        Any ``return`` here falls back to :meth:`_rearm` in the caller, so
        the loop always returns to wake-listening after a wake, even when the
        utterance was noise, empty, or routed to a dead command.

        Each phase is individually guarded: a failure inside one interaction
        (TTS, capture, STT, routing, or response TTS) is logged and swallowed
        so the loop keeps listening for the next wake word.  A transient
        backend error must never permanently disable voice; only exceptions
        outside an interaction (device open, wake detector, stop) kill the
        loop.
        """
        logger.info("VOICE INTERACTION_START")
        logger.info("voice boundary: interaction start")
        self._interaction_id = next_interaction_id()
        self._reporter.bind(self._interaction_id, self._wake_word)

        # Phase 2: Acknowledge wake.  The state is still WAKE_DETECTED so the
        # JSONL shows the required sequence LISTENING → WAKE_DETECTED → "Yes?"
        # → CAPTURING; the transition to CAPTURING happens only when speech
        # capture (Phase 3) actually begins.
        logger.info("voice boundary: wake ack speech starting")
        try:
            self._tts.speak("Yes?")
        except Exception:
            logger.exception("voice interaction failed at wake-ack; continuing to listen")
            self._recover("wake acknowledgement failed")
            return
        logger.info("wake acknowledged — listening for command")

        # Phase 3: Capture speech.  Discard audio buffered before/during the
        # acknowledgement so the spoken command is read fresh and TTS output is
        # never mistaken for a follow-up command.
        self._set_state("CAPTURING")
        self._reporter.listening()
        self._reporter.capture_start(max_segment_s=self._max_segment_s)
        try:
            self._audio.flush()
        except Exception:
            logger.exception("voice interaction failed at capture-flush; continuing to listen")
            self._recover("microphone flush failed")
            return
        logger.info("VOICE capture_started")
        logger.info("VOICE COMMAND_CAPTURE_START")
        logger.info("voice boundary: capture start (max_segment_s=%.1f)", self._max_segment_s)
        try:
            segment = self._read_command(int(self._max_segment_s * 16_000))
        except Exception:
            logger.exception("voice interaction failed at capture; continuing to listen")
            self._recover("command capture failed")
            return
        logger.info("VOICE capture_finished")
        logger.info("VOICE COMMAND_CAPTURE_END")
        logger.info(
            "voice boundary: capture done (duration_s=%.3f samples=%d)",
            segment.duration_s,
            len(segment.samples),
        )
        self._reporter.capture_end(
            duration_s=segment.duration_s,
            samples=len(segment.samples),
        )
        if not segment.samples:
            logger.info("voice boundary: no command detected; returning to wake")
            self._reporter.stt_empty()
            return
        if segment.duration_s < 0.3:
            logger.info("voice boundary: capture too short — returning to wake")
            self._recover("captured audio too short")
            return  # too short, likely noise

        # Phase 4: Transcribe
        self._set_state("TRANSCRIBING")
        self._reporter.transcribing()
        logger.info("VOICE STT_START")
        logger.info("voice boundary: stt start (audio_s=%.3f)", segment.duration_s)
        try:
            result = self._stt.transcribe(segment)
        except Exception:
            logger.exception("voice interaction failed at stt; continuing to listen")
            self._recover("speech recognition failed")
            return

        # Command transcripts are logged only at DEBUG, through the
        # redaction filter.  INFO carries a character count only so
        # possible sensitive wording never reaches the log (docs/03 §10).
        # The console line is the *exact* transcript the agent will receive, so
        # a misrecognition is visible where it happens rather than surfacing as
        # a nonsensical task later.
        text = result.text.strip()
        logger.info("VOICE transcript=%d chars", len(text))
        logger.info("VOICE STT_RESULT")
        logger.info("voice boundary: stt done (chars=%d language=%s)", len(text), result.language)
        logger.info("voice transcript: %d chars", len(text))
        logger.debug("voice transcript: %r", redact(text))
        self._reporter.transcript(text, language=result.language or "")
        if not text:
            logger.info("voice boundary: stt empty — no command; returning to wake")
            self._reporter.stt_empty()
            return

        # STT usability gate: a wake-only, empty or mid-sentence transcript
        # must never reach the planner to be guessed at.  This runs *before*
        # THINKING so no task is ever submitted for it; the answer is the fixed
        # local REPEAT_PROMPT and one clean re-arm, not an LLM decision.
        problem = _transcript_problem(text, self._wake_word)
        if problem is not None:
            self._ask_to_repeat(problem)
            return

        # Phase 5: Route
        self._set_state("PROCESSING")
        self._reporter.thinking()
        logger.info("VOICE ROUTING_START")
        logger.info("voice boundary: routing command")
        try:
            response = self._route(text)
        except Exception:
            logger.exception("voice interaction failed at route; continuing to listen")
            self._recover("routing the request failed")
            return
        logger.info("VOICE ROUTING_COMPLETE")
        self._report_routing()
        if response is None:
            logger.info("voice boundary: route returned no response; returning to wake")
            return
        spoken = self._prepare_spoken(response)
        logger.info("VOICE RESPONSE_START")
        logger.info("voice boundary: response ready (chars=%d)", len(spoken))
        self._reporter.answer(spoken)

        self._set_state("SPEAKING")
        self._reporter.speaking(backend=self._tts_backend(), chars=len(spoken))
        try:
            logger.info("VOICE TTS_START")
            logger.info("voice boundary: tts response speech starting")
            completed = self._speak_with_barge_in(spoken)
            logger.info("VOICE TTS_COMPLETE")
            if not completed:
                # Cause-agnostic on purpose: ``_speak_with_barge_in`` already
                # logged *why* the answer stopped (wake word, bare "stop", voice
                # off, or the ceiling), so this line must not name one.
                logger.info("voice boundary: answer cut short; returning to wake")
                return
            logger.info("voice boundary: tts response speech done")
        except Exception:
            # A TTS failure is isolated: the answer was produced, the log says
            # so, and the loop still returns to READY.  Voice must never go
            # permanently deaf because a speech engine hiccuped.
            logger.exception("voice interaction failed at tts-response; continuing to listen")
            self._recover("speech synthesis failed")
            return
        logger.info("VOICE INTERACTION_COMPLETE")
        logger.info("voice boundary: interaction completed")

    def _report_routing(self) -> None:
        """Publish which provider/model served the request (best effort).

        The probe is injected (the daemon wires it to the live ``LLMClient``)
        so this module never imports the LLM layer.  A failure here is
        swallowed: missing routing metadata must never fail an interaction.
        """
        if self._routing_report is None:
            return
        try:
            report = self._routing_report()
        except Exception:
            logger.debug("routing report unavailable", exc_info=True)
            return
        if report is None:
            return
        try:
            self._reporter.routing(report)
        except Exception:
            logger.debug("routing report could not be printed", exc_info=True)

    def _tts_backend(self) -> str:
        """Best-effort backend label for the TTS engine (log-friendly)."""
        for attr in ("backend", "name", "engine_name"):
            value = getattr(self._tts, attr, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return type(self._tts).__name__

    def _speak_with_barge_in(self, text: str) -> bool:
        """Speak *text*, purging it if the wake word or a bare "stop" is heard.

        Returns ``True`` when the answer was spoken to the end and ``False``
        when the user barged in - with the wake word, or under **D22** with a
        bare ``"stop"`` - in which case the caller returns to wake-listening and
        says nothing further.  Neither submits anything to the agent.

        Two engine shapes are supported:

        * an engine that exposes the non-blocking ``start_speaking`` /
          ``is_speaking`` pair (``SapiSpVoiceEngine``).  The engine then speaks
          on the audio device while *this* thread keeps reading the
          microphone, so "Hey Jarvis" can purge the answer.  The microphone
          still has exactly one owner (this thread) and no extra Python thread
          is created;
        * any other engine keeps the blocking ``speak()`` the
          :class:`~jarvis.voice.interfaces.TextToSpeech` protocol guarantees.
          Its answers cannot be cut short; the fallback is logged, never hidden.

        Bounded three ways, so a wedged engine can never hold the voice thread
        (and the microphone) forever: the engine reporting completion,
        ``_stop_event``, and the :data:`_BARGE_IN_MAX_S` wall-clock ceiling.

        **D22.**  When ``bare_stop_while_busy`` is on, the same loop also feeds
        every post-guard segment to an :class:`~jarvis.voice.stopwatch.OnsetTrigger`
        so a bare ``"stop"`` can cut the answer off.  It is strictly additive
        and strictly *after* both the wake detection and the guard, so it can
        neither delay nor displace wake-word barge-in; a non-cancel utterance
        purges nothing and resets nothing.
        """
        start = getattr(self._tts, "start_speaking", None)
        speaking = getattr(self._tts, "is_speaking", None)
        if self._wake_detector is None or not callable(start) or not callable(speaking):
            if not callable(start) or not callable(speaking):
                logger.info("voice tts: engine cannot be interrupted; speaking without barge-in")
            self._tts.speak(text)
            return True
        try:
            self._wake_detector.reset()
        except Exception:
            logger.debug("barge-in: wake detector reset failed", exc_info=True)
        start(text)
        deadline = time.monotonic() + _BARGE_IN_MAX_S
        guard_until = time.monotonic() + _BARGE_IN_GUARD_S
        onset = (
            OnsetTrigger(lambda seg: _chunk_is_speech(seg, self._silence_threshold))
            if self._bare_stop_while_busy
            else None
        )
        while speaking():
            if self._stop_event.is_set():
                logger.info("barge-in: voice stopped; purging the answer")
                self._purge_speech()
                return False
            if time.monotonic() >= deadline:
                logger.warning("barge-in: speech ceiling reached; purging the answer")
                self._purge_speech()
                return False
            segment = self._audio.read(_BARGE_IN_CHUNK_FRAMES)
            if segment.duration_s < 0.05 or time.monotonic() < guard_until:
                continue
            try:
                detected = self._wake_detector.detect(segment).detected
            except Exception:
                logger.debug("barge-in: wake detection failed", exc_info=True)
                continue
            if detected:
                logger.info("VOICE TTS_INTERRUPTED")
                logger.info("voice boundary: wake word heard during speech; purging the answer")
                self._purge_speech()
                return False
            if onset is None:
                continue
            # Additive, strictly after the wake check and the onset guard: the
            # wake detector has already seen this segment, so a bare cancel can
            # never delay or displace wake-word barge-in.
            samples = onset.feed(segment)
            if samples is None:
                continue
            if self._capture_stop_request(
                preroll=samples,
                max_s=BARE_CANCEL_MAX_S,
                listen_timeout_s=BARE_CANCEL_MAX_S,
                where="speaking",
            ):
                # _capture_stop_request already logged VOICE STOP_REQUESTED,
                # the "bare cancel heard while speaking" line and the status
                # event.  All that is left here is the console-level fact that
                # speech was cut off, then the purge.
                logger.info("VOICE TTS_INTERRUPTED")
                logger.info("voice boundary: answer cut short by a bare stop; returning to wake")
                # Purge before returning: the caller's re-arm flushes the mic
                # and quiet-drains, which must happen on a silenced engine.
                self._purge_speech()
                return False
            # Not a cancel ("stop watch: heard N chars ... ignoring" was logged
            # by the capture).  Nothing is purged and the detector is not reset,
            # so the answer plays on and the wake word still works.  Re-arm the
            # onset trigger only: that utterance has been consumed.
            onset.reset()
        return True

    def _purge_speech(self) -> None:
        """Stop any in-progress speech, swallowing engine failures."""
        try:
            self._tts.stop()
        except Exception:
            logger.debug("voice tts: stop failed", exc_info=True)

    def _prepare_spoken(self, response: str) -> str:
        """Return the bounded, leak-free text that TTS will actually speak.

        :func:`~jarvis.agent.answer.spoken_answer` drops any internal routing
        payload (a JSON document echoed into the answer) and caps the length at
        a sentence boundary, so a document-length answer becomes a spoken
        sentence instead of 79 seconds of dead air.  The full text is still on
        the console and in the task result.
        """
        try:
            spoken = spoken_answer(response, budget=self._max_spoken_chars)
        except Exception:
            logger.debug("spoken answer preparation failed", exc_info=True)
            return response
        if spoken != response:
            logger.info(
                "voice boundary: spoken answer bounded (%d -> %d chars)",
                len(response),
                len(spoken),
            )
        return spoken

    def _quiet_start_drain(self) -> None:
        """Discard non-quiet audio until the mic returns to its baseline.

        After an interaction the speakers may still be ringing with the
        response, so a re-armed wake detector would score its first windows
        over that echo instead of the quiet a *first* wake always enjoys.
        Reading-and-discarding until the room drops back below the silence
        threshold restores identical input conditions for every re-arm,
        which keeps the second wake as detectable as the first.

        Bounded two ways so it can never wedge the loop: it stops once at most
        ``_rearm_quiet_gate_s`` of *audio* has been drained, and also at an
        absolute wall-clock deadline a little later, so a stalled microphone
        (repeated short reads that carry no audio) cannot hold the loop open.
        When a bound elapses with the room still non-quiet, detection simply
        proceeds (equivalent to the pre-gate behaviour).  The detector is
        not fed here — re-arm's ``wake_detector.reset()`` runs *after* this
        drain so the model's window starts clean at the first quiet frame.
        """
        if self._rearm_quiet_gate_s <= 0.0:
            return
        chunk_frames = 1280
        # On a live mic each read returns ~80 ms of audio, so ``rearm_quiet_gate_s``
        # of audio is only a handful of reads.  Bounding by *audio seconds*
        # (not wall time) keeps the drain event-driven AND bounded: the very
        # first quiet chunk ends it, a non-quiet room ends it after gate_s, and
        # the wall-clock deadline only guards against a wedged/stalled stream.
        # On test fakes, whose reads return whole segments at once, the audio
        # bound still caps how much scripted audio the drain can consume.
        deadline = time.monotonic() + self._rearm_quiet_gate_s + 1.0
        drained_s = 0.0
        while not self._stop_event.is_set() and drained_s < self._rearm_quiet_gate_s:
            if time.monotonic() >= deadline:
                return
            segment = self._audio.read(chunk_frames)
            if segment.duration_s < 0.05:
                continue
            drained_s += segment.duration_s
            if not _chunk_is_speech(segment, self._silence_threshold):
                logger.debug("voice reset: re-arm quiet gate satisfied")
                return

    def _reset_voice_state(self) -> None:
        """Return the loop to wake-listening after an interaction (idempotent).

        Discards any audio buffered while the response was spoken (so TTS
        output is never heard as a follow-up command), drains the mic back to
        a quiet baseline so the next wake is detected under identical input
        conditions to the first, re-arms the wake-word model's rolling buffer
        *after* that drain so its window starts clean, and clears transient
        per-interaction counters.  Safe to call after a success, an empty
        command, an STT/processing/TTS failure, a timeout, or an unexpected
        exception.  Exceptions inside the reset are swallowed per component so
        a partial reset can never kill the loop.
        """
        self._set_state("RESETTING")
        self._reporter.resetting()
        try:
            self._audio.flush()
            logger.debug("voice reset: audio flushed")
        except Exception:
            logger.debug("voice reset: audio flush failed", exc_info=True)
        try:
            self._quiet_start_drain()
        except Exception:
            logger.debug("voice reset: re-arm quiet gate failed", exc_info=True)
        if self._wake_detector is not None:
            try:
                self._wake_detector.reset()
                logger.debug("voice reset: wake detector re-armed")
            except Exception:
                logger.debug("voice reset: wake detector reset failed", exc_info=True)
        # Any in-flight stop watch session ends here too: the next task's watch
        # must arm itself (flush + reset) instead of inheriting this window.
        self._stop_watch_armed = False
        self._wake_frames_read = 0
        self._set_state("LISTENING")
        self._reporter.ready(self._wake_word)

    def _rearm(self) -> None:
        """Re-arm the wake word after an interaction has finished.

        Delegates to :meth:`_reset_voice_state` so success, empty command,
        timeout, and failure paths all converge on the same idempotent reset;
        the ``voice boundary: re-armed`` marker is the caller-visible proof
        that the next wake word will be picked up.
        """
        logger.info("VOICE WAKE_REARM")
        self._reset_voice_state()
        logger.info("voice boundary: re-armed for next wake")

    def _read_command(
        self,
        max_samples: int,
        listen_timeout_s: float | None = None,
        should_continue: Callable[[], bool] | None = None,
        spoken_preroll: Sequence[float] = (),
    ) -> AudioSegment:
        """Capture speech after the wake, ending on trailing silence.

        ``listen_timeout_s`` overrides :attr:`_listen_timeout_s` for this call
        (the in-flight cancel watch passes a much shorter window than a normal
        command captures).

        ``should_continue`` is an optional hand-over check, re-evaluated before
        every frame.  The in-flight stop watch uses it so a confirmation
        published by the worker thread can take the microphone back within one
        frame instead of being raced by a capture that is already under way;
        the ordinary command path passes nothing and behaves exactly as before.
        A capture that stops early returns whatever was collected so far, which
        is below every caller's speech threshold and therefore reads as "no
        utterance" - the safe direction, because the caller refuses rather than
        acting on a partial one.

        ``spoken_preroll`` is audio the caller has *already* established is
        speech - the in-flight watch's trigger frames, which would otherwise be
        thrown away.  It seeds the capture in phase B (already speaking, end on
        trailing silence) instead of phase A, so the utterance is not re-waited
        for and its onset survives into the transcript.

        Phase A — wait for speech: after the "Yes?" ack the loop reads short
        chunks and discards pre-speech silence (keeping a ~200 ms pre-roll so
        whisper still sees the utterance onset) until the first speech-energy
        chunk arrives or :attr:`_listen_timeout_s` elapses.  This is the core
        fix for "Yes?" then nothing: a pause after the ack no longer counts as
        "the utterance ended", and a quiet room never causes the loop to give
        up instantly.

        Phase B — capture: once speech starts, chunks accumulate until
        :attr:`_silence_timeout_s` of trailing silence has elapsed or the
        ``max_samples`` (``max_segment_s``) cap is hit, so the loop is never
        deaf for a fixed 30 s window and never captures an arbitrarily long
        run of room noise.

        Returns an :class:`AudioSegment` with no samples when no speech
        arrived within the command window — the caller falls back to
        wake-listening.
        """
        collected: list[float] = list(spoken_preroll)
        preroll: list[float] = []
        trailing = 0.0
        speech_started = bool(collected)
        sample_rate = 16_000
        wait_start = time.monotonic()
        timeout = self._listen_timeout_s if listen_timeout_s is None else listen_timeout_s
        timeout = max(timeout, 0.0)
        wait_deadline = wait_start + timeout
        last_heartbeat = wait_start

        while not self._stop_event.is_set() and len(collected) < max_samples:
            if should_continue is not None and not should_continue():
                # Another reader owns the microphone now: stop before consuming
                # another frame so the hand-over happens on this frame.
                logger.debug("voice boundary: command capture handed over")
                break
            segment = self._audio.read(_CAPTURE_CHUNK_FRAMES)
            samples = segment.samples
            if len(samples) <= 0:
                break

            if not speech_started:
                if timeout <= 0 or time.monotonic() >= wait_deadline:
                    logger.info("voice boundary: command window expired without speech")
                    break
                if _chunk_is_speech(segment, self._silence_threshold):
                    speech_started = True
                    sample_rate = segment.sample_rate or 16_000
                    collected.extend(preroll)
                    collected.extend(samples)
                    trailing = 0.0
                else:
                    preroll.extend(samples)
                    del preroll[:-_CAPTURE_PREROLL_FRAMES]
                    now = time.monotonic()
                    if now - last_heartbeat >= _WAKE_HEARTBEAT_S:
                        last_heartbeat = now
                        logger.info(
                            "voice boundary: waiting for command (elapsed_s=%.1f audio_rms=%.4f)",
                            now - wait_start,
                            _segment_rms(segment),
                        )
                    continue
            else:
                collected.extend(samples)
                if _chunk_is_speech(segment, self._silence_threshold):
                    trailing = 0.0
                else:
                    trailing += segment.duration_s
                    if self._silence_timeout_s > 0 and trailing >= self._silence_timeout_s:
                        break  # end of utterance

        return AudioSegment(
            samples=collected,
            sample_rate=sample_rate,
        )

    def watch_for_stop(
        self,
        budget_s: float,
        keep_watching: Callable[[], bool] | None = None,
    ) -> tuple[bool, bool]:
        """Listen for a spoken cancel request while a task is in flight.

        Called by the daemon's task wait (``DaemonServer._voice_submit``) on the
        **voice-loop thread** - the thread that already owns the microphone - so
        there is exactly one reader on the audio device.  It spends one bounded
        slice of audio looking for speech; any utterance short enough to be an
        order is captured once and matched against the fixed cancel vocabulary
        (data:`_CANCEL_REQUESTS` / :data:`_CANCEL_TOKENS`) - no LLM, no planner.

        The trigger is ordinary speech energy, **not** the wake word.  A stop
        request has to work while a task is running without the user first
        saying "Hey Jarvis", and the watch never *scores* the detector, so its
        rolling window is not fed frame by frame behind the loop's back.  The
        detector is touched exactly once, when the session arms (see below), and
        then again by the loop's ordinary :meth:`_rearm`, so the next
        :meth:`_wait_for_wake` starts from the same window it always did.

        Returns ``(watched, stop_requested)``:

        * ``watched`` is True only when the slice was actually spent listening,
          which lets the caller loop straight into the next slice instead of
          also sleeping; it is always bounded by ``budget_s`` of wall time, so
          a caller that trusts it can never spin the loop thread.
        * ``watched`` is False when there was nothing to watch - a stop request
          is already pending, no wake detector is available to restore
          afterwards, or ``keep_watching`` says another reader owns the
          microphone - and the caller must wait on its slot instead.

        The first slice of a session flushes the tail of the command that
        started the task and resets the wake detector's rolling window, so
        that utterance can neither be scored as a cancel request nor linger in
        the detector.  Arming is once per session, **not** per slice: the
        daemon calls this every ``_VOICE_POLL_INTERVAL_S`` while a task runs,
        and a per-slice reset would clear the window before it could ever fill.
        The session ends when the watch hands the microphone back, and
        :meth:`_reset_voice_state` ends it on the loop's own re-arm.

        ``keep_watching`` is re-checked before every frame (and around the
        capture) so the confirmation dialogue - which runs on the worker
        thread - always wins the microphone, and a task that has finished ends
        the watch immediately.  This is a *bounded* watch, not a listener: it
        exists only while one task is in flight and ends with it.
        """
        if budget_s <= 0:
            return False, False
        if self._wake_detector is None:
            # Nothing to restore afterwards: without a detector the loop cannot
            # go back to wake-listening, so standing down keeps the failure
            # honest instead of half-entering a watch.
            logger.warning("stop watch standing down: no wake-word detector to re-arm")
            return False, False
        if not self._stop_watch_armed:
            # First slice of this session: drop the tail of the command that
            # started the task so it cannot be misheard as "stop", and clear
            # the detector's rolling window of that same audio.
            try:
                self._audio.flush()
                self._wake_detector.reset()
            except Exception:
                logger.exception("stop watch: could not arm; standing down")
                self._stop_watch_armed = False
                return False, False
            self._stop_watch_armed = True

        deadline = time.monotonic() + budget_s
        speech_run = 0
        preroll: list[float] = []
        while time.monotonic() < deadline:
            if self._stop_event.is_set():
                # A stop request is already in progress (loop shutting down):
                # hand the wait back rather than watching against it.
                self._stop_watch_armed = False
                return False, False
            if keep_watching is not None and not keep_watching():
                # Someone else is about to read the microphone (confirmation or
                # a finished task): disarm so the next session starts fresh.
                self._stop_watch_armed = False
                return False, False
            segment = self._audio.read(_WAKE_CHUNK_FRAMES)
            if segment.duration_s < 0.05:
                if not segment.samples:
                    self._stop_event.wait(_EMPTY_READ_BACKOFF_S)
                speech_run = 0
                continue
            if not _chunk_is_speech(segment, self._silence_threshold):
                speech_run = 0
                continue
            # Two consecutive speech frames (~160 ms) so a click or a chair
            # creak does not start a transcription of the room.
            speech_run += 1
            # Keep the frames that made up the run: they are the *onset* of the
            # utterance, which is exactly what a transcript needs and what
            # _read_command cannot recover because they were consumed here.
            preroll.extend(segment.samples)
            del preroll[: -_MIN_CANCEL_SPEECH_FRAMES * _WAKE_CHUNK_FRAMES]
            if speech_run < _MIN_CANCEL_SPEECH_FRAMES:
                continue
            requested = self._capture_stop_request(keep_watching, preroll, where="thinking")
            # The session stays armed: the daemon immediately spends the next
            # slice on the same task, and re-arming per utterance would flush
            # and reset the detector every time the room made a noise.  It ends
            # when the microphone is handed back or the loop re-arms.
            return True, requested
        return True, False

    def _capture_stop_request(
        self,
        keep_watching: Callable[[], bool] | None = None,
        preroll: Sequence[float] = (),
        *,
        max_s: float = _STOP_CAPTURE_MAX_S,
        listen_timeout_s: float = _STOP_LISTEN_TIMEOUT_S,
        where: str = "thinking",
    ) -> bool:
        """Read one short utterance and decide whether it is a cancel request.

        Ownership first: ``keep_watching`` is re-checked before the capture,
        before every frame inside it (``_read_command``'s ``should_continue``)
        and once more before the transcription, so a confirmation published by
        the worker thread takes the microphone back within one frame instead of
        racing this read.

        No second flush here: arming already dropped the command that started
        the task, and the two trigger frames were consumed from the buffer to
        find the speech, so a flush now would only discard the onset of the very
        utterance that was just detected.  ``preroll`` is those trigger frames,
        handed to :meth:`_read_command` as speech already spoken, so the
        transcript starts where the voice did.

        Deterministic and local: the transcript is matched against the fixed
        cancel vocabulary by :func:`_is_cancel_request` and is never submitted
        to the agent, so a "stop" can never be planned as a new task.  Any
        failure (capture, STT, hand-over) is a ``False`` - the watch simply
        stands down and the running task is left alone.

        ``max_s``/``listen_timeout_s`` default to the in-flight watch's bounds.
        The speaking state passes :data:`~jarvis.voice.stopwatch.BARE_CANCEL_MAX_S`
        instead, because it shares the microphone with wake-word barge-in and
        has to hand it back promptly.  ``where`` only selects the wording of the
        log line so the two callers are distinguishable in the JSONL.
        """
        if keep_watching is not None and not keep_watching():
            return False
        try:
            segment = self._read_command(
                int(max_s * 16_000),
                listen_timeout_s=listen_timeout_s,
                should_continue=keep_watching,
                spoken_preroll=preroll,
            )
        except Exception:
            logger.exception("stop watch: capture failed")
            return False
        if segment.duration_s < 0.2:
            return False
        if keep_watching is not None and not keep_watching():
            # The confirmation owns the microphone now; do not spend whisper on
            # an utterance we can no longer act on ourselves.
            return False
        try:
            text = self._stt.transcribe(segment).text
        except Exception:
            logger.exception("stop watch: transcription failed")
            return False
        if not _is_cancel_request(text):
            logger.info(
                "stop watch: heard %d chars that are not a cancel request; ignoring",
                len(text),
            )
            return False
        logger.info("VOICE STOP_REQUESTED")
        if where == "speaking":
            # Wording only; the purge and the return-to-READY belong to
            # _speak_with_barge_in, which owns the TTS engine.
            logger.info("stop watch: bare cancel heard while speaking; purging")
        else:
            logger.info("stop watch: bare cancel heard while thinking; cancelling")
            logger.info("voice boundary: in-flight cancel request heard; cancelling the task")
        self._reporter.stop_request()
        return True

    def _route(self, text: str) -> str | None:
        """Route a voice transcript and return a spoken response (or None).

        The command vocabulary is matched against the *normalised* utterance, so
        punctuation from speech recognition ("Stop.", "cancel ?") lands on the
        same branch as the bare word.  The three stop-ish branches are
        deliberately distinct and are checked in this order:

        1. voice OFF (:data:`_STOP_LISTENING`) - stop listening for good;
        2. cancel (:data:`_CANCEL_REQUESTS`) - cancel the current speech and the
           current task, then keep listening (never exits the loop);
        3. dictation - see below.

        A cancel request matches the *whole* utterance only, unlike the in-flight
        watch (:meth:`watch_for_stop`) which also accepts a cancel word inside a
        short utterance.  At the command position there is no task to cancel
        yet, and a loose match would swallow a real request such as "stop the
        music".
        """
        normalised = _normalise_speech(text)

        # Stop listening entirely.
        if normalised in _STOP_LISTENING:
            self._running = False
            self._stop_event.set()
            return "Goodbye."

        # Power-off phrasings have no tool behind them: answer locally rather
        # than spend a planner round trip on a plan that cannot exist.
        if is_shutdown_request(text):
            logger.info("voice boundary: shutdown request answered locally (no tool exists)")
            return SHUTDOWN_UNSUPPORTED_MSG

        # Cancel: stop any speech and abandon the current task, but keep the
        # microphone open.  Reaching this branch means nothing was submitted yet
        # (a task already in flight is cancelled by ``watch_for_stop``), so the
        # acknowledgement is the whole answer and the loop returns to
        # wake-listening.  A bare "stop" here must NEVER switch voice off.
        if normalised in _CANCEL_REQUESTS:
            logger.info("VOICE STOP_REQUESTED")
            logger.info("voice boundary: cancel request at command position; loop stays active")
            try:
                self._tts.stop()
            except Exception:
                logger.debug("cancel: tts stop failed", exc_info=True)
            if self._dictating:
                self.stop_dictation()
                return "Dictation stopped."
            self._reporter.stop_request()
            return STOP_ACKNOWLEDGEMENT

        # Stop dictation
        if normalised in _STOP_DICTATION:
            if self._dictating:
                self.stop_dictation()
                return "Dictation stopped."
            return "No dictation is active."

        # Resume dictation
        if normalised in _RESUME_DICTATION:
            if not self._dictating:
                return None
            if self._focus_checker is not None and not self._focus_checker.is_foreground(
                self._dictation_target_app
            ):
                return f"Dictation paused: {self._dictation_target_app} lost focus."
            self._dictation_paused = False
            return None

        # Dictation mode: forward speech to the dictation callback
        if self._dictating:
            if self._on_dictation is None:
                return "Dictation is not connected - stop dictation first."

            # Pause (fail closed) when the target window lost focus; never
            # forward speech meant for another window.
            if self._focus_checker is not None and not self._focus_checker.is_foreground(
                self._dictation_target_app
            ):
                self._dictation_paused = True
                return f"Dictation paused: {self._dictation_target_app} lost focus."
            if self._dictation_paused:
                return f"Dictation paused: {self._dictation_target_app} lost focus."

            # Hard limits (docs/04 §2.5): session duration and total chars.
            if (
                self._max_session_s > 0
                and time.monotonic() - self._dictation_started >= self._max_session_s
            ):
                self.stop_dictation()
                return "Dictation stopped: session time limit reached."
            if (
                self._max_dictation_chars > 0
                and self._dictation_chars + len(text) > self._max_dictation_chars
            ):
                self.stop_dictation()
                return "Dictation stopped: character limit reached."

            self._dictation_chars += len(text)
            logger.info("voice dictation: %d chars (total %d)", len(text), self._dictation_chars)
            logger.debug("voice dictation: %r", redact(text))
            self._on_dictation(text)
            return None

        # Submit to the agent (recognised natural variants rewritten to the
        # canonical phrasing; everything else reaches the planner untouched).
        if self._submit_task is not None:
            try:
                outcome = self._submit_task(_canonical_command(text), "voice")
                return self._extract_response(outcome)
            except Exception:
                logger.exception("voice task submission failed")
                return "Sorry, I couldn't process that."

        return None

    def _extract_response(self, outcome: Any) -> str | None:
        """Extract a spoken response from an agent outcome."""
        if outcome is None:
            return None
        # A cancelled task is answered with the fixed local acknowledgement, not
        # with the error text: the cancellation was requested and already
        # reported when it was heard (see :meth:`watch_for_stop`).
        if getattr(outcome, "cancelled", False):
            return STOP_ACKNOWLEDGEMENT
        # A confirmation never reaches this thread: the daemon publishes the
        # pending confirmation and answers it on its own worker thread
        # (``DaemonServer._run_voice_confirmation`` -> ``confirm_by_voice``),
        # which is the only path that owns a resume callback.  If one ever
        # arrives here it is unhandled by design, so fail closed: say nothing
        # and let the graph's own policy gate deny the step.
        if getattr(outcome, "confirmation", None) is not None:
            logger.warning("voice outcome carried an unhandled confirmation; ignoring")
            return None
        if hasattr(outcome, "final_answer") and outcome.final_answer:
            return outcome.final_answer
        if hasattr(outcome, "error") and outcome.error:
            return f"Error: {outcome.error}"
        return None
