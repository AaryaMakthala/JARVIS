"""Voice loop: wake → capture → STT → route → TTS (Phase 5).

Runs in a background thread.  The loop:

1. Waits for the wake word (or falls back to voice-activity detection).
2. Captures a speech segment via the VAD.
3. Transcribes via STT.
4. Routes the transcript:
   - ``"stop listening"`` → stops the loop.
   - ``"stop dictation"`` → stops the active dictation session.
   - Tier 1 confirmation (yes/no) → answers directly.
   - Other → submits to the agent as a voice-sourced task.
5. Speaks the response via TTS.

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
import threading
import time
from collections.abc import Callable
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

logger = logging.getLogger(__name__)

# Voice commands (case-insensitive, exact match after stripping).
_STOP_LISTENING = {"stop listening", "stop", "turn off", "go to sleep"}
_STOP_DICTATION = {"stop dictation", "stop dictating"}
_RESUME_DICTATION = {"resume", "resume dictation"}

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
      from whisper's own trailing hyphen/ellipsis or a dangling trailer word.

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
    return None


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
        on_confirmation: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
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
        self._on_confirmation = on_confirmation
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
            logger.exception(
                "voice interaction failed at repeat-prompt tts; continuing to listen"
            )

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
        on_confirmation: Callable[[dict[str, Any]], Any] | None = None,
        rearm_timeout_s: float | None = None,
    ) -> str:
        """Ask the user to approve/reject a confirmation via voice.

        Re-arms the wake word before listening so an ambient "yes" cannot
        approve an action that was not addressed to JARVIS.  Fails closed:
        a missing detector, timeout, or unrecognised answer is a refusal.

        ``on_confirmation`` (defaults to the loop's callback) receives the
        resume answer ``{"approved": bool, "action_hash": str}``.
        """
        payload = dict(payload or {})
        summary = payload.get("summary") or ""
        action_hash = payload.get("action_hash") or ""
        callback = on_confirmation if on_confirmation is not None else self._on_confirmation

        if not self.can_confirm_by_voice(payload):
            if callback is not None:
                callback({"approved": False, "action_hash": action_hash})
            return f"Action requires terminal confirmation: {summary}"

        if not self._rearm_wake_for_confirmation(timeout_s=rearm_timeout_s):
            if callback is not None:
                callback({"approved": False, "action_hash": action_hash})
            return "No response received. Action cancelled."

        self._tts.speak(f"{summary}. Please say yes or no.")
        logger.info("voice boundary: confirmation listening for yes/no")
        # Discard audio captured before the prompt: stale ambient audio must
        # never be used to approve an action.
        self._audio.flush()
        seg = self._read_command(int(self._listen_timeout_s * 16_000))
        if seg.duration_s < 0.3:
            if callback is not None:
                callback({"approved": False, "action_hash": action_hash})
            return "No response received. Action cancelled."
        result = self._stt.transcribe(seg)
        answer = result.text.lower().strip()
        approved = answer in _approval_words(payload)
        if callback is not None:
            callback({"approved": approved, "action_hash": action_hash})
        if approved:
            logger.info("voice confirmation approved")
            return "Approved."
        logger.info("voice confirmation refused by user")
        return "Cancelled."

    def capture_free_text(
        self,
        prompt: str,
        *,
        on_answer: Callable[[str], Any] | None = None,
        rearm_timeout_s: float | None = None,
    ) -> str:
        """Ask a question and capture a spoken free-text answer.

        Used for clarification interrupts.  Re-arms the wake word first so an
        ambient utterance cannot be mistaken for an answer.  Fails closed:
        a missing detector, timeout, or an empty transcript returns ``""`` —
        the caller resumes the graph with empty text and the clarify node
        reports "No clarification received.".

        ``on_answer`` (defaults to nothing) receives the transcript once.
        """
        if not self._rearm_wake_for_confirmation(timeout_s=rearm_timeout_s):
            return ""

        self._tts.speak(prompt)
        logger.info("voice boundary: clarification listening for free text")
        self._audio.flush()
        seg = self._read_command(int(self._listen_timeout_s * 16_000))
        if seg.duration_s < 0.3:
            return ""
        result = self._stt.transcribe(seg)
        answer = result.text.strip()
        if on_answer is not None:
            on_answer(answer)
        if not answer:
            logger.info("voice clarification dropped: no speech captured")
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
            self._tts.speak(spoken)
            logger.info("VOICE TTS_COMPLETE")
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

    def _read_command(self, max_samples: int) -> AudioSegment:
        """Capture speech after the wake, ending on trailing silence.

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
        collected: list[float] = []
        preroll: list[float] = []
        trailing = 0.0
        speech_started = False
        sample_rate = 16_000
        wait_start = time.monotonic()
        wait_deadline = wait_start + max(self._listen_timeout_s, 0.0)
        last_heartbeat = wait_start

        while not self._stop_event.is_set() and len(collected) < max_samples:
            segment = self._audio.read(_CAPTURE_CHUNK_FRAMES)
            samples = segment.samples
            if len(samples) <= 0:
                break

            if not speech_started:
                if self._listen_timeout_s <= 0 or time.monotonic() >= wait_deadline:
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

    def _rearm_wake_for_confirmation(self, timeout_s: float | None = None) -> bool:
        """Require the wake word again before a voice approval.

        Prevents an un-addressed "yes" in the room from approving a pending
        action.  Fails closed: without a wake detector, or when the wake word
        is not re-detected within the timeout, the confirmation is refused.
        """
        if self._wake_detector is None:
            logger.warning("confirmation by voice refused: no wake-word detector to re-arm")
            return False
        timeout = self._listen_timeout_s if timeout_s is None else timeout_s
        self._tts.speak(f"Say {self._wake_word} to confirm.")
        # Discard the prompt's own acoustic echo from the mic, then reset the
        # detector so its rolling window is clean when listening begins.  This
        # must happen AFTER the prompt: resetting first would leave the prompt
        # echo to fill the very windows the confirmation wake is scored in.
        self._audio.flush()
        self._wake_detector.reset()
        deadline = time.monotonic() + timeout
        chunk_frames = 1280
        while not self._stop_event.is_set() and time.monotonic() < deadline:
            segment = self._audio.read(chunk_frames)
            if segment.duration_s < 0.05:
                continue
            if self._wake_detector.detect(segment).detected:
                return True
        logger.warning("confirmation by voice refused: wake word not re-detected")
        return False

    def _route(self, text: str) -> str | None:
        """Route a voice transcript and return a spoken response (or None)."""
        normalised = text.lower().strip()

        # Stop listening
        if normalised in _STOP_LISTENING:
            self._running = False
            self._stop_event.set()
            return "Goodbye."

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

        # Submit to the agent
        if self._submit_task is not None:
            try:
                outcome = self._submit_task(text, "voice")
                return self._extract_response(outcome)
            except Exception:
                logger.exception("voice task submission failed")
                return "Sorry, I couldn't process that."

        return None

    def _extract_response(self, outcome: Any) -> str | None:
        """Extract a spoken response from an agent outcome."""
        if outcome is None:
            return None
        # Handle the confirmation case
        if hasattr(outcome, "confirmation") and outcome.confirmation is not None:
            return self._handle_voice_confirmation(outcome)
        if hasattr(outcome, "final_answer") and outcome.final_answer:
            return outcome.final_answer
        if hasattr(outcome, "error") and outcome.error:
            return f"Error: {outcome.error}"
        return None

    def _handle_voice_confirmation(self, outcome: Any) -> str | None:
        """Handle a confirmation request via voice (Tier 1 only)."""
        conf = outcome.confirmation
        payload = conf if isinstance(conf, dict) else dict(conf or {})
        return self.confirm_by_voice(payload)
