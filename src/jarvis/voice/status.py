"""Terminal + JSONL lifecycle reporting for the voice pipeline.

The daemon's console must show, for every interaction, exactly which stage was
reached and whether the loop returned to :attr:`~jarvis.voice.states.VoicePhase.READY`.
Before this module the only console output was a stray ``[DIAG]`` line, so a
user could not tell a healthy interaction from a stuck one.

The console vocabulary is fixed (one line per stage)::

    [VOICE] READY — waiting for "Hey Jarvis"
    [VOICE] WAKE DETECTED
    [VOICE] LISTENING
    [VOICE] CAPTURE COMPLETE
    [VOICE] TRANSCRIBING
    [VOICE] STT: "<exact transcript>"
    [VOICE] THINKING
    [LLM] PROVIDER: <provider>
    [LLM] MODEL: <model>
    [VOICE] ANSWER: "<clean answer>"
    [VOICE] SPEAKING
    [VOICE] RESETTING
    [VOICE] READY — waiting for "Hey Jarvis"

and, after an isolated failure, ``[VOICE] RECOVERING`` followed by ``READY``.
Everything else (capture start, per-chunk audio length, latency) is **JSONL
only** — useful for diagnosis without being console noise.

Design constraints (docs/02, AGENTS.md §6):

* **Library code never ``print``s.**  The reporter owns the single place a
  voice lifecycle line reaches the console, and the stream is injected — so
  tests assert on a ``StringIO`` and the daemon injects ``sys.stdout``.
* **Under ``pythonw`` there is no console.**  ``sys.stdout`` is ``None``, and
  the reporter degrades to JSONL-only instead of raising.
* **Credentials never reach either sink.**  Only a fixed vocabulary of event
  names, provider/model labels, durations, and failure categories is emitted.
  The transcript is written to the console verbatim (it is the user's own
  speech, on their own machine, and the requirement is an exact
  ``[VOICE] STT: "…"`` line) but only a *character count* and a *word count*
  reach the log file — no words at all, per docs/03 §10.
* **No internal state is ever printed.**  The reporter accepts only already
  sanitised strings; it never receives a graph dictionary, a plan, or a raw
  provider payload, so a leak is structurally impossible from this side.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, TextIO

from jarvis.logging_setup import redact
from jarvis.voice.states import VoicePhase, check_transition, console_label

logger = logging.getLogger(__name__)

#: Longest answer echoed to the console.  A longer final answer is a bug, not
#: something to read aloud, so it is truncated *for display only* — the value
#: the agent produced is unchanged and the full length is reported instead.
MAX_ECHO_CHARS = 400

#: Fallback wake word used only for the READY banner when none was configured.
DEFAULT_WAKE_WORD = "hey jarvis"


def wake_phrase(wake_word: str) -> str:
    """Render the configured wake word for display (``hey_jarvis`` -> ``Hey Jarvis``).

    Only the display form is touched; the detector's own model name is passed
    through untouched, so this can never change what is actually listened for.
    """
    cleaned = (wake_word or "").replace("_", " ").strip()
    if not cleaned:
        cleaned = DEFAULT_WAKE_WORD
    return " ".join(part.capitalize() for part in cleaned.split())


def resolve_stream() -> TextIO | None:
    """Return a writable console stream, or ``None`` under ``pythonw``."""
    stream = sys.stdout
    if stream is None or getattr(stream, "closed", False):
        return None
    return stream


@dataclass
class ProviderReport:
    """Safe routing metadata for one LLM operation (no secrets)."""

    provider: str = ""
    model: str = ""
    latency_ms: int = 0
    failure_category: str = ""
    fallback_from: str = ""
    fallback_to: str = ""
    degraded: bool = False
    attempts: int = 0

    def is_empty(self) -> bool:
        """Whether this report carries no routing information at all."""
        return not (
            self.provider
            or self.model
            or self.failure_category
            or self.fallback_from
            or self.fallback_to
        )


def report_from_call_meta(meta: Any, *, fallback_from: str = "") -> ProviderReport | None:
    """Build a :class:`ProviderReport` from a ``llm.client.CallMeta``.

    Duck-typed on purpose: ``jarvis.llm`` must not import ``jarvis.voice``, and
    a caller holding no ``CallMeta`` (a plain object, ``None``) simply gets no
    report rather than an exception.  Only provider/model labels, a latency,
    and a failure-category string are copied — never a prompt or a payload.
    """
    if meta is None:
        return None
    provider = str(getattr(meta, "provider", "") or "")
    model = str(getattr(meta, "model", "") or "")
    category = str(getattr(meta, "failure_category", "") or "")
    if not (provider or model or category):
        return None
    try:
        latency_ms = int(getattr(meta, "latency_ms", 0) or 0)
    except (TypeError, ValueError):
        latency_ms = 0
    return ProviderReport(
        provider=provider,
        model=model,
        latency_ms=latency_ms,
        failure_category=category,
        fallback_from=fallback_from,
        fallback_to=provider if fallback_from and fallback_from != provider else "",
        degraded=bool(getattr(meta, "degraded", False)),
    )


class VoiceStatusReporter:
    """Emits the voice lifecycle to the console and the JSONL log.

    Every method is safe to call from any thread and never raises: reporting
    must not be able to break an interaction.
    """

    def __init__(
        self,
        stream: TextIO | None | str = "auto",
        *,
        enabled: bool = True,
        echo_transcript: bool = True,
        wake_word: str = DEFAULT_WAKE_WORD,
    ) -> None:
        if stream == "auto":
            stream = resolve_stream()
        self._stream: TextIO | None = stream
        self._enabled = enabled
        self._echo_transcript = echo_transcript
        self._wake_word = wake_word or DEFAULT_WAKE_WORD
        self._lock = threading.Lock()
        self._interaction_id = ""
        self._phase: VoicePhase | None = None
        self._phase_started = 0.0
        self._illegal: list[str] = []

    # ── configuration ────────────────────────────────────────────────────

    @property
    def phase(self) -> VoicePhase | None:
        """The phase most recently reported, or ``None`` before the first one."""
        return self._phase

    @property
    def interaction_id(self) -> str:
        """The interaction id the reporter is currently tagging lines with."""
        return self._interaction_id

    @property
    def illegal_transitions(self) -> list[str]:
        """Every rejected ``current -> target`` jump, for tests and diagnosis.

        A non-empty list means the loop skipped a lifecycle stage, which is a
        real bug even though reporting must never be the thing that raises.
        """
        return list(self._illegal)

    def bind(self, interaction_id: str, wake_word: str | None = None) -> None:
        """Start a new interaction, tagged with ``interaction_id``."""
        with self._lock:
            self._interaction_id = interaction_id
            if wake_word:
                self._wake_word = wake_word
            self._illegal.clear()

    # ── core emitters ────────────────────────────────────────────────────

    def _write(self, line: str, event: str, *, console: bool = True, **fields: Any) -> None:
        """Write one console line and one JSONL record (both failure-proof)."""
        with self._lock:
            if console and self._enabled and self._stream is not None:
                try:
                    self._stream.write(line + "\n")
                    self._stream.flush()
                except Exception:  # noqa: BLE001 - reporting must never raise
                    self._stream = None
            try:
                detail = _fmt_fields(fields)
            except Exception:  # noqa: BLE001 - never let a field break the loop
                detail = ""
            logger.info("voice event=%s %s", event, detail)

    def state(
        self,
        phase: VoicePhase,
        *,
        detail: str = "",
        **fields: Any,
    ) -> None:
        """Report a lifecycle transition.

        ``detail`` is appended to the console line only; the JSONL record keeps
        the phase name, the elapsed time in the previous phase, and ``fields``.
        """
        with self._lock:
            previous = self._phase
            now = time.monotonic()
            spent = int((now - self._phase_started) * 1000) if previous is not None else 0
            self._phase = phase
            self._phase_started = now
            interaction_id = self._interaction_id
        suffix = f" — {detail}" if detail else ""
        self._write(
            f"[VOICE] {console_label(phase)}{suffix}",
            phase.name.lower(),
            interaction_id=interaction_id,
            voice_state=str(phase),
            previous_phase=str(previous) if previous is not None else "",
            phase_ms=spent,
            **fields,
        )

    def advance(self, phase: VoicePhase, *, detail: str = "", **fields: Any) -> bool:
        """Report ``phase`` only when the transition is legal.

        Returns True when the transition was accepted.  An illegal jump is
        recorded in :attr:`illegal_transitions` and logged at WARNING, but it
        is still reported: the console must show where the loop actually is
        even when that is not where the state machine says it should be.  A
        stuck or skipped stage therefore becomes a visible, testable defect
        rather than a silent divergence between the log and the machine.
        """
        current = self._phase
        if current is not None and not check_transition(current, phase):
            message = f"{current} -> {phase}"
            self._illegal.append(message)
            logger.warning("voice illegal state transition: %s", message)
            self.state(phase, detail=detail, illegal_from=str(current), **fields)
            return False
        self.state(phase, detail=detail, **fields)
        return True

    # ── interaction stages ───────────────────────────────────────────────

    def ready(self, wake_word: str | None = None) -> None:
        """Report that the loop is re-armed and waiting for the wake word.

        This is the resting state and the proof that a previous interaction
        finished: the exact same line is printed at start-up and after every
        wake, so a missing final ``READY`` is immediately visible.
        """
        with self._lock:
            if wake_word:
                self._wake_word = wake_word
            word = self._wake_word
        self.advance(
            VoicePhase.READY,
            detail=f'waiting for "{wake_phrase(word)}"',
            wake_word=word,
        )

    def waiting_for_wake(self) -> None:
        """Report that the detector is actively scoring audio (JSONL only)."""
        with self._lock:
            current = self._phase
        if current is VoicePhase.READY:
            return
        self.advance(VoicePhase.WAITING_FOR_WAKE)

    def wake_detected(self) -> None:
        """Report the wake word scoring above threshold."""
        self.state(VoicePhase.WAKE_DETECTED, detail="")

    def listening(self) -> None:
        """Report the start of command capture."""
        self.advance(VoicePhase.LISTENING)

    def capture_start(self, *, max_segment_s: float = 0.0) -> None:
        """Report the start of command capture (JSONL only; not console noise)."""
        self._write(
            "[VOICE] CAPTURE_START",
            "capture_start",
            console=False,
            interaction_id=self.interaction_id,
            voice_state=str(self.phase or ""),
            max_segment_s=round(max_segment_s, 2),
        )

    def capture_end(self, *, duration_s: float, samples: int = 0) -> None:
        """Report the end of capture and the exact audio duration.

        Console: one ``[VOICE] CAPTURE COMPLETE`` line (the required stage).
        JSONL: the two boundary records with the real numbers.
        """
        self.advance(VoicePhase.CAPTURE_COMPLETE, detail=f"audio={duration_s:.2f}s")
        self._write(
            "[VOICE] CAPTURE_END",
            "capture_end",
            console=False,
            interaction_id=self.interaction_id,
            voice_state=str(self.phase or ""),
            duration_s=round(duration_s, 3),
            samples=samples,
        )
        self._write(
            f"[VOICE] AUDIO DURATION: {duration_s:.2f}s",
            "audio_duration",
            console=False,
            interaction_id=self.interaction_id,
            voice_state=str(self.phase or ""),
            duration_s=round(duration_s, 3),
        )

    def transcribing(self) -> None:
        """Report that local speech-to-text is running on the captured audio."""
        self.advance(VoicePhase.TRANSCRIBING)

    def transcript(self, text: str, *, language: str = "") -> None:
        """Report the exact transcript.

        The console gets the verbatim words inside double quotes so it is
        obvious where speech recognition stopped and the agent began.  The
        **log record gets no words at all** — only the length and the word
        count.

        This is deliberate.  An earlier version logged a "redacted copy", but
        :func:`~jarvis.logging_setup.redact` only masks *recognised* secrets and
        ``+``-prefixed numbers, so for ordinary speech the "copy" was the whole
        utterance verbatim.  An arbitrary transcript is the user's own words,
        and AGENTS.md 6 requires message bodies to stay out of the log at INFO;
        the console is the channel the user opted into with ``echo_transcript``,
        the log file is not.
        """
        shown = (
            f'[VOICE] STT: "{text}"'
            if self._echo_transcript
            else f"[VOICE] STT: <{len(text)} chars>"
        )
        self._write(
            shown,
            "stt",
            interaction_id=self.interaction_id,
            voice_state=str(self.phase or ""),
            chars=len(text),
            words=len(text.split()) if text else 0,
            language=language,
        )

    def stt_empty(self) -> None:
        """Report that nothing usable was recognised."""
        self._write(
            "[VOICE] STT EMPTY — returning to READY",
            "stt_empty",
            interaction_id=self.interaction_id,
            voice_state=str(self.phase or ""),
        )

    def retry_prompt(self, prompt: str) -> None:
        """Report the local "please repeat" prompt (console + JSONL).

        Emitted instead of ``TRANSCRIBING -> THINKING`` when the transcript
        cannot be used, so the console shows why no answer followed.  The
        prompt is a constant owned by the loop: no LLM ever generates it, and
        nothing transcript-shaped reaches this sink.
        """
        self._write(
            f'[VOICE] RETRY: "{prompt}"',
            "stt_retry",
            interaction_id=self.interaction_id,
            voice_state=str(self.phase or ""),
            chars=len(prompt),
        )

    def thinking(self, report: ProviderReport | None = None) -> None:
        """Report the start of agent processing."""
        self.advance(VoicePhase.THINKING)
        if report is not None:
            self.routing(report)

    def routing(self, report: ProviderReport) -> None:
        """Report which provider/model actually served the call."""
        if report.is_empty():
            return
        if report.failure_category:
            self._write(
                f"[LLM] FAILURE: {report.failure_category}",
                "llm_failure",
                interaction_id=self.interaction_id,
                voice_state=str(self.phase or ""),
                provider=report.provider,
                failure_category=report.failure_category,
                fallback_to=report.fallback_to,
                attempts=report.attempts,
            )
        if report.fallback_from and report.fallback_to:
            self._write(
                f"[LLM] FALLBACK: {report.fallback_from} -> {report.fallback_to}",
                "llm_fallback",
                interaction_id=self.interaction_id,
                voice_state=str(self.phase or ""),
                provider=report.provider,
                model=report.model,
                fallback_from=report.fallback_from,
                fallback_to=report.fallback_to,
            )
        if report.provider:
            self._write(
                f"[LLM] PROVIDER: {report.provider}",
                "llm_provider",
                interaction_id=self.interaction_id,
                voice_state=str(self.phase or ""),
                provider=report.provider,
                model=report.model,
            )
        if report.model:
            self._write(
                f"[LLM] MODEL: {report.model}",
                "llm_model",
                interaction_id=self.interaction_id,
                voice_state=str(self.phase or ""),
                provider=report.provider,
                model=report.model,
            )
        if report.latency_ms:
            self._write(
                f"[LLM] LATENCY: {report.latency_ms}ms",
                "llm_latency",
                console=False,
                interaction_id=self.interaction_id,
                voice_state=str(self.phase or ""),
                provider=report.provider,
                model=report.model,
                latency_ms=report.latency_ms,
                attempts=report.attempts,
                degraded=report.degraded,
            )
        if report.degraded:
            self._write(
                "[LLM] DEGRADED: fast model (no tool actions)",
                "llm_degraded",
                interaction_id=self.interaction_id,
                voice_state=str(self.phase or ""),
                provider=report.provider,
                model=report.model,
            )

    def answer(self, text: str) -> None:
        """Report the final answer that will be spoken."""
        shown = f'[VOICE] ANSWER: "{text}"'
        if len(text) > MAX_ECHO_CHARS:
            shown = (
                f'[VOICE] ANSWER: "{text[:MAX_ECHO_CHARS]}…" (+{len(text) - MAX_ECHO_CHARS} chars)'
            )
            self._write(
                shown,
                "answer",
                interaction_id=self.interaction_id,
                voice_state=str(self.phase or ""),
                chars=len(text),
                truncated=True,
            )
            return
        self._write(
            shown,
            "answer",
            interaction_id=self.interaction_id,
            voice_state=str(self.phase or ""),
            chars=len(text),
        )

    def speaking(self, *, backend: str = "", chars: int = 0) -> None:
        """Report that the answer is being spoken, and by which backend."""
        self.advance(VoicePhase.SPEAKING)
        if backend:
            self._write(
                f"[VOICE] TTS: {backend}",
                "tts",
                console=False,
                interaction_id=self.interaction_id,
                voice_state=str(self.phase or ""),
                backend=backend,
                chars=chars,
            )

    def resetting(self) -> None:
        """Report the start of the post-interaction reset."""
        self.advance(VoicePhase.RESETTING)

    def recovering(self, reason: str) -> None:
        """Report an isolated failure that will be recovered from."""
        self.advance(VoicePhase.RECOVERING, detail=reason)

    def ready_again(self, wake_word: str | None = None) -> None:
        """Alias of :meth:`ready` that reads better at re-arm call sites."""
        self.ready(wake_word)

    def error(self, reason: str) -> None:
        """Report an unrecoverable voice failure."""
        self.state(VoicePhase.ERROR, detail=reason)

    def stopping(self) -> None:
        """Report an intentional shutdown."""
        self.state(VoicePhase.STOPPING)


def _fmt_fields(fields: dict[str, Any]) -> str:
    """Render safe scalar fields as ``k=v`` pairs for the JSONL message."""
    parts: list[str] = []
    for key, value in fields.items():
        if value is None or value == "" or value is False:
            continue
        if isinstance(value, bool):
            parts.append(f"{key}={str(value).lower()}")
        elif isinstance(value, (int, float)):
            parts.append(f"{key}={value}")
        else:
            parts.append(f"{key}={redact(str(value))}")
    return " ".join(parts)
