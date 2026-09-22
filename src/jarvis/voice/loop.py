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

No audio data ever leaves this module; only text is passed onward.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from jarvis.logging_setup import redact
from jarvis.voice.interfaces import (
    AudioInput,
    AudioSegment,
    SpeechToText,
    TextToSpeech,
    WakeWordDetector,
    WindowFocusChecker,
)

logger = logging.getLogger(__name__)

# Voice commands (case-insensitive, exact match after stripping).
_STOP_LISTENING = {"stop listening", "stop", "turn off", "go to sleep"}
_STOP_DICTATION = {"stop dictation", "stop dictating"}
_RESUME_DICTATION = {"resume", "resume dictation"}

#: Spoken words accepted as an approval to a confirmation prompt.
_YES_WORDS = {"yes", "y", "approve", "approved", "confirm", "go", "ok", "sure"}

#: Session limits enforced by the loop (defaults, overridable per instance).
DEFAULT_MAX_SESSION_S = 1800.0
DEFAULT_MAX_DICTATION_CHARS = 20000

#: Longest single command window captured before STT (30 s, as in config.toml).
DEFAULT_MAX_SEGMENT_S = 30.0

#: Capture chunk length (100 ms at 16 kHz) read for silence detection.
_CAPTURE_CHUNK_FRAMES = 1600

#: RMS (float samples in [-1, 1]) below which a chunk counts as silence.
_SILENCE_RMS = 0.01


def _chunk_is_speech(segment: AudioSegment) -> bool:
    """Return True when ``segment`` carries voice-level energy."""
    samples = segment.samples
    if not samples:
        return False
    acc = 0.0
    for sample in samples:
        acc += sample * sample
    return (acc / len(samples)) ** 0.5 >= _SILENCE_RMS


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
        silence_timeout_s: float = 0.7,
        max_segment_s: float = DEFAULT_MAX_SEGMENT_S,
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

    # ── public API ──────────────────────────────────────────────────────

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
        logger.info("voice loop started")
        print("[DIAG] voice.loop.start(): thread 'voice-loop' launched", flush=True)

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
            return
        self._audio.close()
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
        approved = answer in _YES_WORDS
        if callback is not None:
            callback({"approved": approved, "action_hash": action_hash})
        if approved:
            logger.info("voice confirmation approved")
            return "Approved."
        logger.info("voice confirmation refused by user")
        return "Cancelled."

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
            self._audio.open()
            self._audio_opened = True
            if self._wake_detector is not None:
                self._wake_detector.reset()

            while self._running and not self._stop_event.is_set():
                # Phase 1: Wait for wake word (or voice activity)
                if not self._wait_for_wake():
                    if self._idle_timed_out:
                        logger.info("voice loop idle timeout reached; stopping")
                        break
                    continue

                self._run_interaction()

                # Re-arm for the next wake: reset the wake-word model's rolling
                # buffer and discard audio captured while the response was
                # spoken, so a second "hey jarvis" is detected fresh and late
                # TTS output is never mistaken for a follow-up command.  A
                # failure here must not kill the loop.
                try:
                    self._rearm()
                except Exception:
                    logger.exception("voice re-arm failed after interaction; continuing to wait")

        except Exception:
            logger.exception("voice loop crashed")
            reason = "mic-open-failed" if not self._audio_opened else "loop-crashed"
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
        stop (idle timeout or stop request).  The idle deadline is measured
        with ``time.monotonic()`` and re-checked on a cancellable poll loop so
        a stop request is never delayed by a long sleep.

        A transient exception from the wake detector itself is logged and the
        detector re-armed, never allowed to kill the loop: model hiccups must
        not disable voice.  A failure of the underlying microphone read still
        surfaces as ``loop-crashed``.
        """
        if self._wake_detector is None:
            # No wake-word model; use a simple voice-activity gate.
            segment = self._audio.read(4800)  # 300 ms
            return segment.duration_s > 0 and not self._stop_event.is_set()

        wait_start = time.monotonic()
        deadline: float | None = None
        if self._idle_timeout_s > 0:
            deadline = wait_start + self._idle_timeout_s

        # Read chunks and feed to the wake-word model.
        chunk_frames = 1280  # 80 ms at 16 kHz
        last_heartbeat = wait_start
        while self._running and not self._stop_event.is_set():
            if deadline is not None and time.monotonic() >= deadline:
                self._idle_timed_out = True
                return False
            segment = self._audio.read(chunk_frames)
            if segment.duration_s < 0.05:
                logger.debug(
                    "voice boundary: wake read too short (duration_s=%.3f)",
                    segment.duration_s,
                )
                continue
            self._wake_frames_read += len(segment.samples)
            logger.debug(
                "voice boundary: wake frame (frames=%d duration_s=%.3f)",
                len(segment.samples),
                segment.duration_s,
            )
            if time.monotonic() - last_heartbeat >= 3.0:
                last_heartbeat = time.monotonic()
                logger.info(
                    "voice boundary: waiting for wake word (frames_read=%d elapsed_s=%.1f)",
                    self._wake_frames_read,
                    time.monotonic() - wait_start,
                )
            try:
                result = self._wake_detector.detect(segment)
            except Exception:
                logger.exception("voice boundary: wake detector error; re-arming and continuing")
                self._rearm()
                continue
            if result.detected:
                logger.info("voice boundary: wake trigger; entering interaction")
                return True
        return False

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
        logger.info("voice boundary: interaction start")

        # Phase 2: Acknowledge wake
        logger.info("voice boundary: wake ack speech starting")
        try:
            self._tts.speak("Yes?")
        except Exception:
            logger.exception("voice interaction failed at wake-ack; continuing to listen")
            return
        logger.info("wake acknowledged — listening for command")

        # Phase 3: Capture speech.  Discard audio buffered before/during the
        # acknowledgement so the spoken command is read fresh and TTS output is
        # never mistaken for a follow-up command.
        try:
            self._audio.flush()
        except Exception:
            logger.exception("voice interaction failed at capture-flush; continuing to listen")
            return
        logger.info("voice boundary: capture start (max_segment_s=%.1f)", self._max_segment_s)
        try:
            segment = self._read_command(int(self._max_segment_s * 16_000))
        except Exception:
            logger.exception("voice interaction failed at capture; continuing to listen")
            return
        logger.info(
            "voice boundary: capture done (duration_s=%.3f samples=%d)",
            segment.duration_s,
            len(segment.samples),
        )
        if segment.duration_s < 0.3:
            logger.info("voice boundary: capture too short — returning to wake")
            return  # too short, likely noise

        # Phase 4: Transcribe
        logger.info("voice boundary: stt start (audio_s=%.3f)", segment.duration_s)
        try:
            result = self._stt.transcribe(segment)
        except Exception:
            logger.exception("voice interaction failed at stt; continuing to listen")
            return

        # Command transcripts are logged only at DEBUG, through the
        # redaction filter.  INFO carries a character count only so
        # possible sensitive wording never reaches the log (docs/03 §10).
        text = result.text.strip()
        logger.info("voice boundary: stt done (chars=%d language=%s)", len(text), result.language)
        logger.info("voice transcript: %d chars", len(text))
        logger.debug("voice transcript: %r", redact(text))
        if not text:
            logger.info("voice boundary: stt empty — no command; returning to wake")
            return

        # Phase 5: Route
        logger.info("voice boundary: routing command")
        try:
            response = self._route(text)
        except Exception:
            logger.exception("voice interaction failed at route; continuing to listen")
            return
        if response is None:
            logger.info("voice boundary: route returned no response; returning to wake")
            return
        logger.info("voice boundary: response ready (chars=%d)", len(response))

        try:
            logger.info("voice boundary: tts response speech starting")
            self._tts.speak(response)
            logger.info("voice boundary: tts response speech done")
        except Exception:
            logger.exception("voice interaction failed at tts-response; continuing to listen")
            return
        logger.info("voice boundary: interaction completed")

    def _rearm(self) -> None:
        """Re-arm the wake word after an interaction has finished.

        Resets the wake-word model's rolling buffer and discards any audio
        captured while the response was spoken, so a second wake word is
        detected fresh and late TTS audio is never heard as a command.
        """
        if self._wake_detector is not None:
            self._wake_detector.reset()
        self._audio.flush()
        logger.info("voice boundary: re-armed for next wake")

    def _read_command(self, max_samples: int) -> AudioSegment:
        """Capture speech until trailing silence, bounded by ``max_samples``.

        Reads short chunks so the loop is never deaf for a whole
        ``listen_timeout_s`` window: capture ends once :attr:`_silence_timeout_s`
        of trailing silence has accumulated after speech (or right away when
        nothing was spoken), so the loop returns to wake-listening promptly
        enough for a second wake word to be detected.
        """
        collected: list[float] = []
        total = 0
        trailing = 0.0
        sample_rate = 16_000
        while not self._stop_event.is_set() and total < max_samples:
            segment = self._audio.read(_CAPTURE_CHUNK_FRAMES)
            samples = segment.samples
            if len(samples) <= 0:
                break
            if total == 0:
                sample_rate = segment.sample_rate or 16_000
            collected.extend(samples)
            total += len(samples)
            if _chunk_is_speech(segment):
                trailing = 0.0
            else:
                trailing += segment.duration_s
                if self._silence_timeout_s > 0 and trailing >= self._silence_timeout_s:
                    break  # end of utterance (or no utterance at all)
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
