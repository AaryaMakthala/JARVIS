"""Tests for the voice status reporter.

The reporter is the daemon's proof of what a voice interaction did.  Two
properties matter and are both asserted here:

1. **The required terminal vocabulary is exact.**  A human reading the console,
   and a marker written for a specific log line, both depend on the literal text
   — ``[VOICE] WAKE DETECTED``, not ``[VOICE] WAKE_DETECTED``.
2. **The interaction always returns to READY.**  ``READY`` is printed at
   start-up *and* after every wake, so a missing final ``READY`` is an
   immediately visible, testable defect rather than a silently dead wake word.

Console output is captured in a :class:`io.StringIO` and diagnostics are disabled,
so no test writes to the real console or leaves a JSONL file behind.
"""

from __future__ import annotations

import io
from itertools import pairwise

import pytest

from jarvis.voice.states import (
    ALLOWED_TRANSITIONS,
    HAPPY_PATH,
    VoicePhase,
    check_transition,
    console_label,
    next_interaction_id,
    reset_interaction_ids,
)
from jarvis.voice.status import ProviderReport, VoiceStatusReporter


@pytest.fixture
def reporter() -> VoiceStatusReporter:
    """A reporter that writes to an in-memory buffer instead of the console."""
    return VoiceStatusReporter(stream=io.StringIO(), wake_word="Hey Jarvis")


@pytest.fixture
def out(reporter: VoiceStatusReporter) -> io.StringIO:
    """The buffer the reporter under test writes to."""
    stream = reporter._stream
    assert isinstance(stream, io.StringIO)
    return stream


def full_interaction(r: VoiceStatusReporter) -> None:
    """Drive one complete wake-to-answer interaction, in the happy-path order."""
    r.ready()
    r.waiting_for_wake()
    r.wake_detected()
    r.listening()
    r.capture_start(max_segment_s=30.0)
    r.capture_end(duration_s=1.24, samples=19840)
    r.transcribing()
    r.transcript("what is 2 times 2?")
    r.thinking()
    r.routing(ProviderReport(provider="Groq", model="openai/gpt-oss-20b", attempts=1))
    r.answer("4")
    r.speaking(backend="piper", chars=1)
    r.resetting()
    r.ready_again()


class TestRequiredTerminalVocabulary:
    """Every one of these lines is part of the contract; the text is exact."""

    @pytest.mark.parametrize(
        "expected",
        [
            '[VOICE] READY — waiting for "Hey Jarvis"',
            "[VOICE] WAKE DETECTED",
            "[VOICE] LISTENING",
            "[VOICE] CAPTURE COMPLETE",
            "[VOICE] TRANSCRIBING",
            '[VOICE] STT: "what is 2 times 2?"',
            "[VOICE] THINKING",
            "[LLM] PROVIDER: Groq",
            "[LLM] MODEL: openai/gpt-oss-20b",
            '[VOICE] ANSWER: "4"',
            "[VOICE] SPEAKING",
            "[VOICE] RESETTING",
        ],
    )
    def test_line_is_emitted(
        self, reporter: VoiceStatusReporter, out: io.StringIO, expected: str
    ) -> None:
        full_interaction(reporter)
        assert expected in out.getvalue()

    def test_one_interaction_has_no_illegal_transition(self, reporter: VoiceStatusReporter) -> None:
        full_interaction(reporter)
        assert reporter.illegal_transitions == []

    def test_underscored_labels_are_not_used(
        self, reporter: VoiceStatusReporter, out: io.StringIO
    ) -> None:
        """Regression: the enum value must not leak into human-facing text."""
        full_interaction(reporter)
        text = out.getvalue()
        assert "WAKE_DETECTED" not in text
        assert "CAPTURE_COMPLETE" not in text

    def test_console_labels_have_spaces_where_the_words_do(
        self, reporter: VoiceStatusReporter, out: io.StringIO
    ) -> None:
        full_interaction(reporter)
        assert "[VOICE] WAKE DETECTED" in out.getvalue()
        assert "[VOICE] CAPTURE COMPLETE" in out.getvalue()


class TestReadyIsAlwaysRestored:
    def test_ready_printed_at_startup_and_after_the_interaction(
        self, reporter: VoiceStatusReporter, out: io.StringIO
    ) -> None:
        full_interaction(reporter)
        assert out.getvalue().count("[VOICE] READY") == 2

    def test_consecutive_interactions_each_end_ready(
        self, reporter: VoiceStatusReporter, out: io.StringIO
    ) -> None:
        """The second wake must work exactly like the first."""
        for _ in range(3):
            full_interaction(reporter)
        assert out.getvalue().count("[VOICE] READY") == 6
        assert reporter.illegal_transitions == []


class TestTranscript:
    def test_exact_text_is_shown_inside_quotes(
        self, reporter: VoiceStatusReporter, out: io.StringIO
    ) -> None:
        reporter.transcript("what is 2 times 2?")
        assert '[VOICE] STT: "what is 2 times 2?"' in out.getvalue()

    def test_can_be_hidden_by_config(self) -> None:
        stream = io.StringIO()
        r = VoiceStatusReporter(stream=stream, echo_transcript=False)
        r.transcript("secret-ish utterance")
        text = stream.getvalue()
        assert "secret-ish utterance" not in text
        assert "[VOICE] STT: <" in text


class TestAnswerBoundaryAtTheReporter:
    def test_answer_is_shown_as_a_quoted_string(
        self, reporter: VoiceStatusReporter, out: io.StringIO
    ) -> None:
        reporter.answer("4")
        assert '[VOICE] ANSWER: "4"' in out.getvalue()

    def test_over_long_answer_is_truncated_for_display(
        self, reporter: VoiceStatusReporter, out: io.StringIO
    ) -> None:
        reporter.answer("x" * 5000)
        line = next(line for line in out.getvalue().splitlines() if "[VOICE] ANSWER" in line)
        assert len(line) < 5000
        assert "more" in line.lower() or "…" in line or "..." in line


class TestRecovery:
    def test_recovering_line_names_the_reason(
        self, reporter: VoiceStatusReporter, out: io.StringIO
    ) -> None:
        reporter.recovering("idle timeout; re-arming")
        assert "[VOICE] RECOVERING" in out.getvalue()
        assert "idle timeout" in out.getvalue()

    def test_an_interaction_recovering_from_stt_still_returns_to_ready(
        self, reporter: VoiceStatusReporter, out: io.StringIO
    ) -> None:
        """Invariant 10 (fail closed) and the READY invariant together."""
        reporter.ready()
        reporter.waiting_for_wake()
        reporter.wake_detected()
        reporter.listening()
        reporter.capture_end(duration_s=0.5)
        reporter.transcribing()
        reporter.stt_empty()
        reporter.resetting()
        reporter.ready_again()
        assert reporter.illegal_transitions == []
        assert out.getvalue().count("[VOICE] READY") == 2


class TestDiagnosticsLog:
    """Diagnostics go to the logging sink (JSONL handler), never to the console."""

    def test_console_stays_quiet_while_events_are_logged(
        self, reporter: VoiceStatusReporter, out: io.StringIO, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        caplog.set_level(logging.INFO, logger="jarvis.voice.status")
        reporter.capture_start(max_segment_s=30.0)
        assert out.getvalue() == ""
        assert "capture_start" in caplog.text

    def test_log_records_carry_the_phase_and_interaction(
        self, reporter: VoiceStatusReporter, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        caplog.set_level(logging.INFO, logger="jarvis.voice.status")
        reporter.ready()
        reporter.wake_detected()
        assert "event=ready" in caplog.text
        assert "event=wake_detected" in caplog.text
        assert "voice_state=" in caplog.text

    def test_logged_transcript_contains_no_words(
        self, reporter: VoiceStatusReporter, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The console shows the words; the log keeps only counts.

        A "redacted copy" is not enough for a transcript: redaction only masks
        *recognised* secrets, so ordinary speech would be stored verbatim.
        """
        import logging

        caplog.set_level(logging.INFO, logger="jarvis.voice.status")
        reporter.transcript("my number is 9876543210")
        assert "9876543210" not in caplog.text
        assert "my number" not in caplog.text
        assert "chars=" in caplog.text
        assert "words=" in caplog.text

    def test_no_raw_provider_payload_is_logged(
        self, reporter: VoiceStatusReporter, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        caplog.set_level(logging.INFO, logger="jarvis.voice.status")
        reporter.routing(ProviderReport(provider="Groq", model="openai/gpt-oss-20b", attempts=2))
        assert "Groq" in caplog.text
        assert "api_key" not in caplog.text

    def test_disabling_status_silences_the_console_only(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """``enabled=False`` hides the console; the log sink is independent.

        Turning off the on-screen status must not blind the diagnostics that
        ``jarvis doctor`` and bug reports depend on.
        """
        import logging

        caplog.set_level(logging.INFO, logger="jarvis.voice.status")
        stream = io.StringIO()
        quiet = VoiceStatusReporter(stream=stream, enabled=False)
        quiet.ready()
        quiet.wake_detected()
        assert stream.getvalue() == ""
        assert "event=wake_detected" in caplog.text

    def test_a_broken_stream_does_not_raise(self) -> None:
        """Reporting must never be able to break an interaction."""

        class Exploding(io.StringIO):
            def write(self, _text: str) -> int:
                raise OSError("console closed")

        r = VoiceStatusReporter(stream=Exploding())
        r.ready()
        r.wake_detected()
        r.answer("fine")


class TestStateMachineVocabulary:
    def test_happy_path_is_a_legal_chain(self) -> None:
        for current, target in pairwise(HAPPY_PATH):
            assert check_transition(current, target), f"{current} -> {target}"

    def test_console_labels_only_differ_where_words_are_split(self) -> None:
        assert console_label(VoicePhase.WAKE_DETECTED) == "WAKE DETECTED"
        assert console_label(VoicePhase.CAPTURE_COMPLETE) == "CAPTURE COMPLETE"
        assert console_label(VoicePhase.READY) == "READY"
        assert console_label(VoicePhase.THINKING) == "THINKING"

    def test_error_and_stopping_are_reachable_from_everywhere(self) -> None:
        """A stuck phase must never trap the loop."""
        for phase in ALLOWED_TRANSITIONS:
            if phase in (VoicePhase.ERROR, VoicePhase.STOPPING):
                continue
            assert check_transition(phase, VoicePhase.ERROR), phase
            assert check_transition(phase, VoicePhase.STOPPING), phase

    def test_ready_is_reachable_from_every_live_phase(self) -> None:
        for phase in ALLOWED_TRANSITIONS:
            if phase in (VoicePhase.ERROR, VoicePhase.STOPPING):
                continue
            assert check_transition(phase, VoicePhase.READY), phase

    def test_error_recovers_only_through_a_fresh_start(self) -> None:
        """An unrecoverable loop restarts the machine; it does not resume.

        ``ERROR -> READY`` is deliberately rejected: resuming a listening state
        from an error would hide the fact that the microphone or thread died.
        """
        assert check_transition(VoicePhase.ERROR, VoicePhase.READY) is False
        assert check_transition(VoicePhase.ERROR, VoicePhase.STARTING) is True
        assert check_transition(VoicePhase.ERROR, VoicePhase.STOPPING) is True

    def test_skipping_a_stage_is_rejected(self) -> None:
        assert check_transition(VoicePhase.READY, VoicePhase.SPEAKING) is False
        assert check_transition(VoicePhase.LISTENING, VoicePhase.THINKING) is False

    def test_interaction_ids_are_monotonic(self) -> None:
        reset_interaction_ids()
        first = next_interaction_id()
        second = next_interaction_id()
        assert first == "i1"
        assert second == "i2"
        reset_interaction_ids()
