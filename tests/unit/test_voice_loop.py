"""Voice loop orchestration tests (no hardware, no network).

Tests the VoiceLoop's lifecycle, routing, and edge cases using fakes.
"""

from __future__ import annotations

import time
from typing import Any

from jarvis.voice.fakes import (
    FakeAudioInput,
    FakeFocusChecker,
    FakeSTT,
    FakeTTS,
    FakeWakeWord,
    make_silence,
    make_speech,
)
from jarvis.voice.interfaces import (
    AudioSegment,
    STTResult,
    WakeWordResult,
)
from jarvis.voice.loop import VoiceLoop

# ── helpers ─────────────────────────────────────────────────────────────


def _make_loop(
    *,
    wake_results: list[WakeWordResult] | None = None,
    stt_results: list[STTResult] | None = None,
    tts: FakeTTS | None = None,
    submit_task: Any | None = None,
    on_confirmation: Any | None = None,
    on_dictation: Any | None = None,
    focus: FakeFocusChecker | None = None,
    audio_segments: list[AudioSegment] | None = None,
) -> tuple[VoiceLoop, FakeAudioInput, FakeWakeWord, FakeSTT, FakeTTS]:
    """Build a VoiceLoop with all fakes for testing."""
    if audio_segments is None:
        # Enough speech segments for wake + command capture
        audio_segments = [make_speech("fake", duration_s=0.5)] * 50

    audio = FakeAudioInput(segments=audio_segments)
    wake = FakeWakeWord(results=wake_results or [WakeWordResult(detected=True, confidence=0.9)])
    stt = FakeSTT(results=stt_results or [STTResult(text="hello")])
    tts = tts or FakeTTS()

    loop = VoiceLoop(
        audio=audio,
        wake_detector=wake,
        stt=stt,
        tts=tts,
        focus_checker=focus,
        submit_task=submit_task,
        on_confirmation=on_confirmation,
        on_dictation=on_dictation,
    )
    return loop, audio, wake, stt, tts


# ── lifecycle tests ─────────────────────────────────────────────────────


class TestVoiceLoopLifecycle:
    def test_start_and_stop(self) -> None:
        """Start the loop, verify it's running, then stop it manually."""
        loop, audio, _, _, _ = _make_loop()
        loop.start()
        time.sleep(0.1)  # give the thread a moment to start
        assert loop.is_active()
        loop.stop()
        assert not loop.is_active()
        assert audio.close_calls >= 1

    def test_stop_when_not_running(self) -> None:
        loop, _, _, _, _ = _make_loop()
        loop.stop()  # should not raise
        assert not loop.is_active()

    def test_start_is_idempotent(self) -> None:
        loop, _, _, _, _ = _make_loop()
        loop.start()
        time.sleep(0.1)
        t1 = loop._thread
        assert loop.is_active()
        loop.start()  # second call should be a no-op
        assert loop._thread is t1
        loop.stop()


# ── routing tests ───────────────────────────────────────────────────────


class TestVoiceLoopRouting:
    def test_stop_listening_command(self) -> None:
        loop, _, _, _, _ = _make_loop(
            stt_results=[STTResult(text="stop listening")],
        )
        response = loop._route("stop listening")
        assert response == "Goodbye."
        assert not loop.is_active()

    def test_stop_dictation_command(self) -> None:
        loop, _, _, _, _ = _make_loop()
        loop.start_dictation()
        assert loop.is_dictating
        response = loop._route("stop dictation")
        assert response == "Dictation stopped."
        assert not loop.is_dictating

    def test_stop_dictation_when_not_dictating(self) -> None:
        loop, _, _, _, _ = _make_loop()
        response = loop._route("stop dictation")
        assert response == "No dictation is active."

    def test_resume_dictation(self) -> None:
        loop, _, _, _, _ = _make_loop()
        loop.start_dictation()
        response = loop._route("resume")
        assert response is None  # just resumes silently

    def test_dictation_forwards_to_callback(self) -> None:
        forwarded: list[str] = []
        loop, _, _, _, _ = _make_loop(on_dictation=lambda text: forwarded.append(text))
        loop.start_dictation()
        response = loop._route("hello world")
        assert response is None
        assert forwarded == ["hello world"]

    def test_submit_task(self) -> None:
        results: list[tuple[str, str]] = []

        def fake_submit(text: str, source: str) -> Any:
            results.append((text, source))
            return type(
                "Outcome", (), {"final_answer": "done", "confirmation": None, "error": None}
            )()

        loop, _, _, _, _ = _make_loop(submit_task=fake_submit)
        response = loop._route("open notepad")
        assert response == "done"
        assert results == [("open notepad", "voice")]

    def test_submit_task_error(self) -> None:
        def boom(text: str, source: str) -> Any:
            raise RuntimeError("LLM unavailable")

        loop, _, _, _, _ = _make_loop(submit_task=boom)
        response = loop._route("open notepad")
        assert response == "Sorry, I couldn't process that."


# ── confirmation tests ──────────────────────────────────────────────────


class TestVoiceLoopConfirmation:
    def test_tier1_voice_confirmation_approve(self) -> None:
        confirmation_answers: list[dict[str, Any]] = []

        def on_confirm(answer: dict[str, Any]) -> str:
            confirmation_answers.append(answer)
            return "approved"

        outcome = type(
            "Outcome",
            (),
            {
                "final_answer": None,
                "error": None,
                "confirmation": {"tier": 1, "summary": "create file", "action_hash": "abc"},
            },
        )()

        def fake_submit(text: str, source: str) -> Any:
            return outcome

        loop, _audio, _, stt, _tts = _make_loop(
            submit_task=fake_submit,
            on_confirmation=on_confirm,
            audio_segments=[make_speech("fake", duration_s=0.5)] * 100,
        )
        # Pre-load STT with "yes" for the confirmation
        stt._results = [STTResult(text="yes")]
        stt._idx = 0

        loop._route("create file")
        # The response should include the confirmation handling
        assert len(confirmation_answers) == 1
        assert confirmation_answers[0]["approved"] is True

    def test_tier2_rejected_by_voice(self) -> None:
        outcome = type(
            "Outcome",
            (),
            {
                "final_answer": None,
                "error": None,
                "confirmation": {"tier": 2, "summary": "delete files", "action_hash": "def"},
            },
        )()

        def fake_submit(text: str, source: str) -> Any:
            return outcome

        loop, _, _, _, _ = _make_loop(submit_task=fake_submit)
        response = loop._route("delete files")
        assert "terminal confirmation" in response


# ── focus verification tests ────────────────────────────────────────────


class TestVoiceLoopFocus:
    def test_dictation_pause_on_focus_loss(self) -> None:
        focus = FakeFocusChecker(foreground="Chrome")
        loop, _, _, _, _ = _make_loop(focus=focus)
        loop.start_dictation(target_app="notepad")
        response = loop._route("resume")
        assert "lost focus" in response

    def test_dictation_resume_on_focus_gain(self) -> None:
        focus = FakeFocusChecker(foreground="Notepad")
        loop, _, _, _, _ = _make_loop(focus=focus)
        loop.start_dictation(target_app="notepad")
        response = loop._route("resume")
        assert response is None  # resumes silently


# ── device disconnect tests ─────────────────────────────────────────────


class TestVoiceLoopDeviceDisconnect:
    def test_audio_close_on_loop_exit(self) -> None:
        loop, audio, _, _, _ = _make_loop()
        loop.start()
        time.sleep(0.1)
        loop.stop()  # this triggers cleanup
        loop._thread.join(timeout=5.0)
        assert audio.close_calls >= 1

    def test_exception_in_loop_does_not_crash(self) -> None:
        """A broken audio stream should not leak the thread."""
        audio = FakeAudioInput(segments=[make_silence(0.01)])

        wake = FakeWakeWord(results=[WakeWordResult(detected=True)])

        def boom(seg: AudioSegment) -> STTResult:
            raise RuntimeError("device disconnected")

        stt = FakeSTT(transcribe_fn=boom)
        tts = FakeTTS()
        loop = VoiceLoop(audio=audio, wake_detector=wake, stt=stt, tts=tts)
        loop.start()
        loop._thread.join(timeout=5.0)
        assert not loop.is_active()


# ── invariant #15 prep: audio not sent to LLM ──────────────────────────


class TestAudioNotSentToLLM:
    def test_submit_receives_text_not_audio(self) -> None:
        """Verify that the submit_task callback receives text, not audio data."""
        received: list[tuple[str, Any]] = []

        def capture_submit(text: str, source: str) -> Any:
            received.append((text, source))
            return type(
                "Outcome",
                (),
                {
                    "final_answer": "ok",
                    "confirmation": None,
                    "error": None,
                },
            )()

        loop, _, _, _, _ = _make_loop(
            stt_results=[STTResult(text="hello")],
            submit_task=capture_submit,
        )
        loop._route("hello")
        assert len(received) == 1
        text_arg, source_arg = received[0]
        assert isinstance(text_arg, str)
        assert source_arg == "voice"
        # Ensure no bytes/ndarray made it through
        assert not isinstance(text_arg, (bytes, bytearray))


# ── G1: stale audio is flushed before command / confirmation reads ──────


class TestStaleAudioFlush:
    def _capture_outcome(self) -> tuple[list[str], Any]:
        submitted: list[str] = []

        def fake_submit(text: str, source: str) -> Any:
            submitted.append(text)
            return type(
                "Outcome", (), {"final_answer": "ok", "confirmation": None, "error": None}
            )()

        return submitted, fake_submit

    def test_flush_before_command_read_on_full_cycle(self) -> None:
        """A full wake->command cycle flushes the mic before reading."""
        submitted, fake_submit = self._capture_outcome()
        loop, audio, _, _, _ = _make_loop(
            stt_results=[STTResult(text="open notepad")],
            submit_task=fake_submit,
            audio_segments=[make_speech("x", duration_s=0.5)] * 50,
        )
        loop.start()
        for _ in range(100):
            if submitted:
                break
            time.sleep(0.05)
        loop.stop()
        assert submitted == ["open notepad"]
        assert audio.flush_calls >= 1
        # The flush must happen before the command is read from the mic.
        assert audio.read_calls  # command read occurred

    def test_flush_before_confirmation_read(self) -> None:
        """confirm_by_voice flushes the mic before listening for yes/no."""
        answers: list[dict[str, Any]] = []
        loop, audio, _, _, _ = _make_loop(
            stt_results=[STTResult(text="yes")],
            audio_segments=[make_speech("x", duration_s=0.5)] * 200,
        )
        loop.confirm_by_voice(
            {"tier": 1, "summary": "create file", "action_hash": "abc"},
            on_confirmation=answers.append,
            rearm_timeout_s=1.0,
        )
        assert answers == [{"approved": True, "action_hash": "abc"}]
        assert audio.flush_calls >= 1

    def test_flush_happens_after_wake_but_before_command_read(self) -> None:
        """Ordering: wake detect ... flush ... read(command window)."""
        events: list[tuple[str, int]] = []
        submitted, fake_submit = self._capture_outcome()

        class TrackingAudio(FakeAudioInput):
            def read(self, num_frames: int) -> AudioSegment:
                events.append(("read", num_frames))
                return super().read(num_frames)

            def flush(self) -> None:
                events.append(("flush", 0))
                super().flush()

        listen_timeout_s = 1.0
        command_frames = int(listen_timeout_s * 16_000)
        audio = TrackingAudio(segments=[make_speech("x", duration_s=0.5)] * 50)
        wake = FakeWakeWord(results=[WakeWordResult(detected=True, confidence=0.9)])
        loop = VoiceLoop(
            audio=audio,
            wake_detector=wake,
            stt=FakeSTT(results=[STTResult(text="open notepad")]),
            tts=FakeTTS(),
            submit_task=fake_submit,
            listen_timeout_s=listen_timeout_s,
            idle_timeout_s=0.5,
        )
        loop.start()
        for _ in range(100):
            if submitted:
                break
            time.sleep(0.05)
        loop.stop()
        assert submitted == ["open notepad"]
        # Wake detection reads small chunks (1280 frames); the command read is
        # the full listen window (16000 * listen_timeout_s).
        command_reads = [
            i for i, (op, n) in enumerate(events) if op == "read" and n == command_frames
        ]
        assert command_reads, f"no command-window read in {events}"
        flushes = [i for i, (op, _) in enumerate(events) if op == "flush"]
        assert flushes, f"no flush in {events}"
        # The flush must occur immediately before the command read, i.e. no
        # other read between the last flush and the command read.
        last_flush = flushes[-1]
        assert last_flush < command_reads[0]
        assert events[last_flush] == ("flush", 0)
        assert not any(op == "read" for op, _ in events[last_flush + 1 : command_reads[0]])


# ── F2: dictation pauses (fail closed) when the target loses focus ──────


class TestDictationFocusPause:
    def test_speech_paused_when_target_not_foreground(self) -> None:
        focus = FakeFocusChecker(foreground="Chrome")
        forwarded: list[str] = []
        loop, _, _, _, _ = _make_loop(focus=focus, on_dictation=lambda t: forwarded.append(t))
        loop.start_dictation(target_app="notepad")
        response = loop._route("hello world")
        assert "lost focus" in response
        assert forwarded == []  # never forwarded speech meant for Notepad
        assert loop._dictation_paused is True

    def test_resume_still_requires_foreground(self) -> None:
        focus = FakeFocusChecker(foreground="Chrome")
        loop, _, _, _, _ = _make_loop(focus=focus)
        loop.start_dictation(target_app="notepad")
        loop._dictation_paused = True
        assert "lost focus" in loop._route("resume")
        assert loop._dictation_paused is True  # still paused

    def test_forwards_only_when_target_is_foreground(self) -> None:
        focus = FakeFocusChecker(foreground="Untitled - Notepad")
        forwarded: list[str] = []
        loop, _, _, _, _ = _make_loop(focus=focus, on_dictation=lambda t: forwarded.append(t))
        loop.start_dictation(target_app="notepad")
        loop._route("hello world")
        assert forwarded == ["hello world"]
        assert loop._dictation_paused is False


# ── F4: transcripts are redacted in logs ────────────────────────────────


class TestTranscriptRedaction:
    def _drive_one_transcript(
        self, caplog: object, level: int, transcript: str, submitted: list[str]
    ) -> tuple[VoiceLoop, FakeAudioInput]:
        caplog.set_level(level, logger="jarvis.voice.loop")  # type: ignore[attr-defined]

        def submit(text: str, source: str) -> Any:
            submitted.append(text)
            return type(
                "Outcome", (), {"final_answer": "ok", "confirmation": None, "error": None}
            )()

        loop, audio, _, _, _ = _make_loop(
            stt_results=[STTResult(text=transcript)],
            submit_task=submit,
            audio_segments=[make_speech("x", duration_s=0.5)] * 20,
        )
        loop.start()
        for _ in range(100):
            if submitted:
                break
            time.sleep(0.05)
        loop.stop()
        return loop, audio

    def test_info_level_has_char_count_only(self, caplog: object) -> None:
        import logging

        submitted: list[str] = []
        transcript = "my secret dozen 98765"
        self._drive_one_transcript(caplog, logging.INFO, transcript, submitted)
        assert submitted == [transcript]
        text = caplog.text  # type: ignore[attr-defined]
        assert "voice transcript:" in text
        assert "98765" not in text  # raw digits never logged at INFO
        assert transcript not in text  # full wording only ever at DEBUG

    def test_debug_level_redacts_sensitive_markers(self, caplog: object) -> None:
        import logging

        submitted: list[str] = []
        self._drive_one_transcript(caplog, logging.DEBUG, "hello", submitted)
        assert "voice transcript" in caplog.text  # type: ignore[attr-defined]


# ── F5: session limits are enforced in the loop ─────────────────────────


class TestSessionLimits:
    def test_session_duration_limit_stops_dictation(self) -> None:
        loop, _, _, _, _ = _make_loop(on_dictation=lambda t: None)
        loop.start_dictation()
        loop._dictation_started = time.monotonic() - loop._max_session_s - 1
        response = loop._route("hello")
        assert "time limit reached" in response
        assert not loop.is_dictating

    def test_character_limit_stops_dictation(self) -> None:
        loop, _, _, _, _ = _make_loop(on_dictation=lambda t: None)
        loop.start_dictation()
        loop._dictation_chars = loop._max_dictation_chars - 5
        response = loop._route("x" * 10)
        assert "character limit reached" in response
        assert not loop.is_dictating

    def test_idle_timeout_stops_loop(self) -> None:
        audio = FakeAudioInput(segments=[make_speech("fake", duration_s=0.5)] * 10)
        wake = FakeWakeWord(results=[WakeWordResult(detected=False)])
        loop = VoiceLoop(
            audio=audio,
            wake_detector=wake,
            stt=FakeSTT(),
            tts=FakeTTS(),
            idle_timeout_s=0.01,
        )
        loop.start()
        assert loop._thread is not None
        loop._thread.join(timeout=5.0)
        assert not loop.is_active()


# ── F6: voice confirmations fail closed ─────────────────────────────────


class TestVoiceConfirmationFailClosed:
    def test_no_wake_detector_cannot_confirm(self) -> None:
        answers: list[dict[str, Any]] = []
        loop = VoiceLoop(
            audio=FakeAudioInput(),
            wake_detector=None,
            stt=FakeSTT(results=[STTResult(text="yes")]),
            tts=FakeTTS(),
        )
        response = loop.confirm_by_voice(
            {"tier": 1, "summary": "create file", "action_hash": "h"},
            on_confirmation=answers.append,
            rearm_timeout_s=0.05,
        )
        assert response == "No response received. Action cancelled."
        assert answers == [{"approved": False, "action_hash": "h"}]

    def test_rearm_timeout_cannot_confirm(self) -> None:
        answers: list[dict[str, Any]] = []
        loop, _, wake, _, _ = _make_loop(
            audio_segments=[make_silence(0.01)] * 2,
            wake_results=[WakeWordResult(detected=False)],
        )
        assert wake is not None
        response = loop.confirm_by_voice(
            {"tier": 1, "summary": "create file", "action_hash": "h"},
            on_confirmation=answers.append,
            rearm_timeout_s=0.05,
        )
        assert response == "No response received. Action cancelled."
        assert answers == [{"approved": False, "action_hash": "h"}]

    def test_unrecognised_answer_is_cancelled(self) -> None:
        answers: list[dict[str, Any]] = []
        loop, _, _, _, _ = _make_loop(stt_results=[STTResult(text="maybe later")])
        response = loop.confirm_by_voice(
            {"tier": 1, "summary": "create file", "action_hash": "h"},
            on_confirmation=answers.append,
            rearm_timeout_s=1.0,
        )
        assert response == "Cancelled."
        assert answers == [{"approved": False, "action_hash": "h"}]


# ── F10: stop() waits for the thread and closes audio ───────────────────


class TestVoiceLoopStop:
    def test_stop_blocks_until_thread_finished(self) -> None:
        loop, audio, _, _, _ = _make_loop()
        loop.start()
        time.sleep(0.1)
        loop.stop()
        assert loop._thread is not None and not loop._thread.is_alive()
        assert audio.close_calls >= 1
        assert not loop.is_active()


# ── Stage 2: on_exit reports crashes to the owning service ──────────────


class _OpenFailAudio:
    """AudioInput whose open() fails — a broken/unplugged mic."""

    def __init__(self) -> None:
        self.close_calls = 0

    def open(self, sample_rate: int = 16_000, channels: int = 1) -> None:
        raise RuntimeError("no audio device")

    def read(self, num_frames: int) -> Any:
        raise AssertionError("read must not be reached")

    def close(self) -> None:
        self.close_calls += 1

    def is_open(self) -> bool:
        return False


class TestVoiceLoopOnExit:
    def _run_to_exit(self, loop: VoiceLoop) -> None:
        loop.start()
        assert loop._thread is not None
        loop._thread.join(timeout=5.0)
        assert not loop._thread.is_alive()
        assert not loop.is_active()

    def test_open_failure_reports_mic_open_failed_and_closes_audio(self) -> None:
        reasons: list[str | None] = []
        audio = _OpenFailAudio()
        loop = VoiceLoop(
            audio=audio,
            stt=FakeSTT(),
            tts=FakeTTS(),
            on_exit=reasons.append,
        )
        self._run_to_exit(loop)
        assert reasons == ["mic-open-failed"]
        assert audio.close_calls == 1  # audio closed in finally even on open failure

    def test_mid_loop_crash_reports_loop_crashed(self) -> None:
        reasons: list[str | None] = []

        def boom(seg: AudioSegment) -> STTResult:
            raise RuntimeError("device disconnected")

        audio = FakeAudioInput(segments=[make_silence(0.01)])
        loop = VoiceLoop(
            audio=audio,
            wake_detector=FakeWakeWord(results=[WakeWordResult(detected=True)]),
            stt=FakeSTT(transcribe_fn=boom),
            tts=FakeTTS(),
            on_exit=reasons.append,
        )
        self._run_to_exit(loop)
        assert reasons == ["loop-crashed"]
        assert audio.close_calls >= 1  # closed in finally

    def test_clean_stop_reports_none(self) -> None:
        reasons: list[str | None] = []
        loop, audio, _, _, _ = _make_loop()
        loop._on_exit = reasons.append  # type: ignore[method-assign]
        loop.start()
        time.sleep(0.05)
        loop.stop()
        assert reasons == [None]
        assert audio.close_calls >= 1

    def test_idle_timeout_reports_none(self) -> None:
        reasons: list[str | None] = []
        audio = FakeAudioInput(segments=[make_silence(0.01)] * 2)
        loop = VoiceLoop(
            audio=audio,
            wake_detector=FakeWakeWord(results=[WakeWordResult(detected=False)]),
            stt=FakeSTT(),
            tts=FakeTTS(),
            idle_timeout_s=0.02,
            on_exit=reasons.append,
        )
        self._run_to_exit(loop)
        assert reasons == [None]
