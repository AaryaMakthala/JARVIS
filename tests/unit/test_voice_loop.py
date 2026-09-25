"""Voice loop orchestration tests (no hardware, no network).

Tests the VoiceLoop's lifecycle, routing, and edge cases using fakes.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from jarvis.agent.context import make_app_context
from jarvis.agent.graph import open_sqlite_checkpointer
from jarvis.agent.runner import run_task
from jarvis.config import LLMSettings, ProviderModels, Settings
from jarvis.llm.client import GroqClient
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
from jarvis.voice.loop import (
    _CAPTURE_CHUNK_FRAMES,
    _LOOSE_YES_WORDS,
    _STRICT_YES_WORDS,
    VoiceLoop,
    _approval_words,
)
from support import brain_conversation

# ── helpers ─────────────────────────────────────────────────────────────


def _scripted_wake(booleans: list[bool]) -> Callable[[AudioSegment], WakeWordResult]:
    """A ``detect_fn`` returning scripted results.

    Drive wake detection from a ``detect_fn`` instead of ``results=`` so the
    ``reset()`` re-arm call (which only rewinds the scripted ``results`` list)
    never replays ``True`` and turns a test into a busy loop: re-arming is the
    very behaviour under test.
    """

    def detect(_segment: AudioSegment) -> WakeWordResult:
        if booleans:
            return WakeWordResult(detected=booleans.pop(0))
        return WakeWordResult(detected=False)

    return detect


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

    def test_wake_stt_groq_graph_tts_pipeline(self, tmp_path: Path) -> None:
        calls: list[dict[str, Any]] = []
        response = brain_conversation("Four.").model_dump_json()

        def complete(**kwargs: Any) -> Any:
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=response))],
                usage=SimpleNamespace(prompt_tokens=12, completion_tokens=4),
            )

        settings = Settings(
            llm=LLMSettings(
                provider_order=["groq"],
                strict_zero_cost=False,
                models={
                    "groq": ProviderModels(
                        planner="openai/gpt-oss-120b",
                        fast="openai/gpt-oss-20b",
                    )
                },
            )
        )
        ctx = make_app_context(
            settings,
            llm=GroqClient("gsk-test", settings, completer=complete),
        )
        saver = open_sqlite_checkpointer(str(tmp_path / "checkpoints.db"))
        tts = FakeTTS()
        audio = FakeAudioInput(
            segments=[make_silence(1.0), *[make_speech("x", duration_s=0.5)] * 12]
        )
        wake = FakeWakeWord(detect_fn=_scripted_wake([True, False, False, False]))
        stt = FakeSTT(results=[STTResult(text="What is 2 plus 2?")])

        def submit(text: str, source: str) -> Any:
            return run_task(
                ctx,
                saver,
                text,
                source=source,
                thread_id="voice-groq-e2e",
            )

        loop = VoiceLoop(
            audio=audio,
            wake_detector=wake,
            stt=stt,
            tts=tts,
            submit_task=submit,
            listen_timeout_s=5.0,
            idle_timeout_s=0.3,
        )
        loop.start()
        assert loop._thread is not None
        loop._thread.join(timeout=10.0)
        assert tts.spoken == ["Yes?", "Four."]
        assert calls[0]["model"] == "openai/gpt-oss-120b"
        assert calls[0]["response_format"]["type"] == "json_schema"


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

    def test_transient_stt_error_keeps_loop_listening(self, caplog: object) -> None:
        """A transient STT failure must be logged, not kill the voice loop."""
        import logging

        caplog.set_level(logging.INFO, logger="jarvis.voice.loop")  # type: ignore[attr-defined]
        raised: list[str] = []

        def boom(seg: AudioSegment) -> STTResult:
            raised.append("boom")
            raise RuntimeError("whisper backend down")

        audio = FakeAudioInput(segments=[make_speech("x", duration_s=0.1)] * 20)
        wake = FakeWakeWord(detect_fn=_scripted_wake([True, False, False, False]))
        loop = VoiceLoop(
            audio=audio,
            wake_detector=wake,
            stt=FakeSTT(transcribe_fn=boom),
            tts=FakeTTS(),
            idle_timeout_s=5.0,
        )
        loop.start()
        for _ in range(100):
            if raised:
                break
            time.sleep(0.05)
        assert raised == ["boom"]
        # The failed interaction is swallowed: the loop is STILL listening.
        assert loop.is_active()
        assert loop._tts.spoken == ["Yes?"]  # the wake ack happened before the failure
        assert wake.reset_calls >= 1  # re-armed after the failed interaction
        loop.stop()
        assert not loop.is_active()
        text = caplog.text  # type: ignore[attr-defined]
        assert "voice interaction failed at stt" in text


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
        """Ordering: wake detect ... flush ... first capture chunk read."""
        events: list[tuple[str, int]] = []
        submitted, fake_submit = self._capture_outcome()

        class TrackingAudio(FakeAudioInput):
            def read(self, num_frames: int) -> AudioSegment:
                events.append(("read", num_frames))
                return super().read(num_frames)

            def flush(self) -> None:
                events.append(("flush", 0))
                super().flush()

        audio = TrackingAudio(segments=[make_speech("x", duration_s=0.5)] * 50)
        wake = FakeWakeWord(results=[WakeWordResult(detected=True, confidence=0.9)])
        loop = VoiceLoop(
            audio=audio,
            wake_detector=wake,
            stt=FakeSTT(results=[STTResult(text="open notepad")]),
            tts=FakeTTS(),
            submit_task=fake_submit,
            listen_timeout_s=1.0,
            idle_timeout_s=0.5,
        )
        loop.start()
        for _ in range(100):
            if submitted:
                break
            time.sleep(0.05)
        loop.stop()
        assert submitted == ["open notepad"]
        # Wake detection reads small 80 ms chunks; command capture starts with
        # a 100 ms chunk, so the first 1600-frame read marks capture begin.
        first_capture = -1
        for i, (op, n) in enumerate(events):
            if op == "read" and n == _CAPTURE_CHUNK_FRAMES:
                first_capture = i
                break
        assert first_capture > 0, f"no capture-chunk read in {events}"
        # The flush must be the audio op immediately before the first capture
        # chunk is read, so stale audio is never transcribed as the command.
        assert events[first_capture - 1] == ("flush", 0)


# ── JARVIS second-wake fix: re-arm + silence-segmented capture ───────────


class TestVoiceLoopRearmAndCapture:
    """The wake detector is re-armed and the mic flushed after every
    interaction, and command capture stops at trailing silence instead of a
    fixed 30 s window — so a second wake word is picked up, not swallowed."""

    def _submit(self, submitted: list[str]) -> Any:
        def submit(text: str, source: str) -> Any:
            submitted.append(text)
            return type(
                "Outcome", (), {"final_answer": "ok", "confirmation": None, "error": None}
            )()

        return submit

    def test_two_consecutive_wakes_both_process_commands(self) -> None:
        """One interaction must never swallow the next wake word."""
        submitted: list[str] = []
        audio = FakeAudioInput(segments=[make_speech("x", duration_s=0.5)] * 200)
        wake = FakeWakeWord(detect_fn=_scripted_wake([True, True, False, False]))
        loop = VoiceLoop(
            audio=audio,
            wake_detector=wake,
            stt=FakeSTT(transcribe_fn=lambda _seg: STTResult(text="open notepad")),
            tts=FakeTTS(),
            submit_task=self._submit(submitted),
            idle_timeout_s=0.3,
        )
        loop.start()
        assert loop._thread is not None
        loop._thread.join(timeout=10.0)
        assert not loop.is_active()
        assert submitted == ["open notepad", "open notepad"]
        # After each interaction the detector was re-armed and stale audio
        # (TTS echo etc.) flushed so it is never heard as the next command.
        assert wake.reset_calls >= 2
        assert audio.flush_calls >= 2

    def test_three_consecutive_wakes_process_all_commands(self) -> None:
        """Three wake→command cycles run end-to-end without any restart.

        Each interaction must speak ack + response and re-arm so the NEXT
        wake word is heard; no cycle may swallow or drop its successor."
        """
        submitted: list[str] = []
        audio = FakeAudioInput(segments=[make_speech("x", duration_s=0.5)] * 250)
        wake = FakeWakeWord(detect_fn=_scripted_wake([True, True, True, False, False]))
        loop = VoiceLoop(
            audio=audio,
            wake_detector=wake,
            stt=FakeSTT(transcribe_fn=lambda _seg: STTResult(text="open notepad")),
            tts=FakeTTS(),
            submit_task=self._submit(submitted),
            idle_timeout_s=0.3,
        )
        loop.start()
        assert loop._thread is not None
        loop._thread.join(timeout=10.0)
        assert not loop.is_active()
        assert submitted == ["open notepad", "open notepad", "open notepad"]
        assert loop._tts.spoken == ["Yes?", "ok", "Yes?", "ok", "Yes?", "ok"]
        assert audio.close_calls >= 1  # clean shutdown after all three cycles
        assert wake.reset_calls >= 3  # re-armed between every interaction

    def test_empty_utterance_returns_to_wake_listening(self) -> None:
        """A wake with no speech (or an STT miss) never wedges the loop."""
        submitted: list[str] = []
        audio = FakeAudioInput(segments=[make_silence(1.0)] * 40)
        wake = FakeWakeWord(detect_fn=_scripted_wake([True, False]))
        loop = VoiceLoop(
            audio=audio,
            wake_detector=wake,
            stt=FakeSTT(),  # transcribes "" (no text)
            tts=FakeTTS(),
            submit_task=self._submit(submitted),
            listen_timeout_s=0.2,  # short command window: nothing said → give up
            idle_timeout_s=0.3,
        )
        loop.start()
        assert loop._thread is not None
        loop._thread.join(timeout=10.0)
        assert not loop.is_active()
        assert submitted == []
        assert wake.reset_calls >= 1  # re-armed after the failed utterance

    def test_silence_terminates_capture_early(self) -> None:
        """Capture ends on trailing silence instead of filling the 30 s window."""
        audio = FakeAudioInput(segments=[make_speech("x", duration_s=0.5), make_silence(2.0)])
        loop = VoiceLoop(
            audio=audio,
            stt=FakeSTT(),
            tts=FakeTTS(),
        )
        seg = loop._read_command(int(loop._max_segment_s * 16_000))
        assert len(audio.read_calls) == 2  # speech chunk + one trailing-silence chunk
        assert 0.5 <= seg.duration_s < 5.0

    def test_capture_bounded_by_max_segment(self) -> None:
        """Continuous speech still caps at max_segment_s, not forever."""
        audio = FakeAudioInput(segments=[make_speech("x", duration_s=0.5)] * 10)
        loop = VoiceLoop(
            audio=audio,
            stt=FakeSTT(),
            tts=FakeTTS(),
            silence_timeout_s=999.0,  # only the window bound may stop capture
            max_segment_s=1.0,
        )
        seg = loop._read_command(int(loop._max_segment_s * 16_000))
        assert seg.duration_s == 1.0
        assert len(audio.read_calls) == 2

    def test_no_speech_capture_waits_for_speech_then_gives_up(self) -> None:
        """A command-less wake waits for speech, then returns empty — never deaf.

        Regression for the reported bug: the old capture treated the pause
        right after the "Yes?" ack as "end of utterance" and returned an empty
        segment instantly, so a user speaking a beat late was never heard
        ("Yes?" then nothing).  New capture waits up to ``listen_timeout_s``
        for the first speech chunk; if none arrives it returns an empty
        segment and the loop re-arms instead of wedging.
        """
        audio = FakeAudioInput(segments=[make_silence(3.0)])
        loop = VoiceLoop(
            audio=audio,
            stt=FakeSTT(),
            tts=FakeTTS(),
            silence_timeout_s=0.5,
            listen_timeout_s=0.2,
        )
        seg = loop._read_command(int(loop._max_segment_s * 16_000))
        assert len(seg.samples) == 0  # no speech within the command window
        assert seg.duration_s == 0.0
        assert audio.read_calls  # we did wait and probe for speech

    def test_late_command_after_ack_pause_is_captured(self) -> None:
        """Speech arriving AFTER a post-ack pause is still captured.

        The core fix: pre-speech silence (the user's normal pause after
        "Yes?") is discarded, not treated as "the command ended".  The
        discarded silence is bounded by ``_CAPTURE_PREROLL_FRAMES`` and the
        command window is bounded by ``listen_timeout_s``.
        """
        submitted: list[str] = []
        audio = FakeAudioInput(
            segments=[make_silence(1.0), *[make_speech("x", duration_s=0.5)] * 12]
        )
        wake = FakeWakeWord(detect_fn=_scripted_wake([True, False, False, False]))
        loop = VoiceLoop(
            audio=audio,
            wake_detector=wake,
            stt=FakeSTT(results=[STTResult(text="open notepad")]),
            tts=FakeTTS(),
            submit_task=self._submit(submitted),
            listen_timeout_s=5.0,
            idle_timeout_s=0.3,
        )
        loop.start()
        assert loop._thread is not None
        loop._thread.join(timeout=10.0)
        assert submitted == ["open notepad"]  # the pause did not eat the command
        assert loop._tts.spoken == ["Yes?", "ok"]
        assert wake.reset_calls >= 1


# ── Re-arm quiet-start: the response echo never reaches the detector ────


class TestRearmQuietStartGate:
    """After an interaction the response is still ringing in the room, so the
    re-armed wake detector must not score its first windows over that echo.

    The re-arm must drain the mic back to a quiet baseline BEFORE
    ``wake_detector.reset()`` so every re-arm starts from the same input
    conditions as the very first wake, and the drain must be bounded so a
    room that never goes quiet can never wedge the loop.
    """

    @staticmethod
    def _tone(fill: float, duration_s: float = 0.2) -> AudioSegment:
        """Constant-fill segment so audio content can be identified by value."""
        n = int(duration_s * 16_000)
        return AudioSegment(samples=[fill] * n, sample_rate=16_000)

    def _submit(self, submitted: list[str]) -> Any:
        def submit(text: str, source: str) -> Any:
            submitted.append(text)
            return type(
                "Outcome", (), {"final_answer": "ok", "confirmation": None, "error": None}
            )()

        return submit

    def test_response_echo_drained_before_detector_reopens(self) -> None:
        """The detector's post-reset window contains the quiet room and the
        NEXT wake, never the response echo.

        The audio stream models a real cycle: wake → command → the response
        echoing in the room → the room going quiet → the next wake.  The
        echo segments carry distinctive fills (0.3/0.15); if re-arm resets the
        detector before draining, those echoes are fed straight into the
        model's freshly reset window (the reported bug), which this test
        fails on.
        """
        submitted: list[str] = []
        wake_1 = self._tone(0.75)  # the first "hey jarvis"
        cmd_1 = [self._tone(0.5), self._tone(0.5)]  # the command
        end_1 = make_silence(1.0)  # trailing silence ends capture 1
        echo = [self._tone(0.3), self._tone(0.15)]  # response ringing out
        quiet = make_silence(0.4)  # the room returns to baseline
        wake_2 = self._tone(0.75)  # the second "hey jarvis"
        cmd_2 = [self._tone(0.5), self._tone(0.5)]
        end_2 = make_silence(1.0)
        tail = [make_silence(0.1)] * 10  # idle timeout stops the loop

        audio = FakeAudioInput(
            segments=[wake_1, *cmd_1, end_1, *echo, quiet, wake_2, *cmd_2, end_2, *tail]
        )
        wake = FakeWakeWord(detect_fn=_scripted_wake([True, True, False, False]))
        loop = VoiceLoop(
            audio=audio,
            wake_detector=wake,
            stt=FakeSTT(transcribe_fn=lambda _seg: STTResult(text="open notepad")),
            tts=FakeTTS(),
            submit_task=self._submit(submitted),
            idle_timeout_s=0.3,
        )
        loop.start()
        assert loop._thread is not None
        loop._thread.join(timeout=10.0)
        assert not loop.is_active()
        # Both cycles ran end-to-end.
        assert submitted == ["open notepad", "open notepad"]
        # The echo (fills 0.3 / 0.15) was drained during re-arm and never
        # fed to the detector; the real wake words (fill 0.75) were.
        assert all(seg.samples[0] not in (0.3, 0.15) for seg in wake.detect_calls)
        assert any(seg.samples[0] == 0.75 for seg in wake.detect_calls)
        # 1 startup reset + 1 re-arm per interaction — the quiet gate
        # itself must not add extra resets.
        assert wake.reset_calls == 3

    def test_reset_drains_echo_before_rearming_the_detector(self) -> None:
        """``_reset_voice_state`` order is flush → quiet drain → detector reset.

        Resetting before the drain would put the echo into the window the
        model starts scoring after the reset, so the reset must be the LAST
        step, after the mic reads show a quiet baseline.
        """
        events: list[str] = []

        class _EventAudio(FakeAudioInput):
            def flush(self) -> None:
                events.append("flush")
                super().flush()

            def read(self, num_frames: int) -> AudioSegment:
                events.append("read")
                return super().read(num_frames)

        class _EventWake(FakeWakeWord):
            def reset(self) -> None:
                events.append("reset")
                super().reset()

        audio = _EventAudio(segments=[make_speech("echo", duration_s=0.2), make_silence(0.4)])
        loop = VoiceLoop(
            audio=audio,
            wake_detector=_EventWake(),
            stt=FakeSTT(),
            tts=FakeTTS(),
        )
        loop._reset_voice_state()
        # flush, echo read (non-quiet → keep draining), quiet read (drain
        # done → return), and only THEN the detector reset.
        assert events == ["flush", "read", "read", "reset"]

    def test_quiet_gate_bounded_when_room_never_goes_quiet(self) -> None:
        """A room that stays loud must not wedge re-arm: the drain stops
        after ``rearm_quiet_gate_s`` of audio and detection proceeds, so
        repeated wakes keep being processed."""
        submitted: list[str] = []
        audio = FakeAudioInput(segments=[make_speech("x", duration_s=0.5)] * 250)
        wake = FakeWakeWord(detect_fn=_scripted_wake([True, True, False, False]))
        loop = VoiceLoop(
            audio=audio,
            wake_detector=wake,
            stt=FakeSTT(transcribe_fn=lambda _seg: STTResult(text="open notepad")),
            tts=FakeTTS(),
            submit_task=self._submit(submitted),
            idle_timeout_s=0.3,
            rearm_quiet_gate_s=0.1,  # bound the drain per re-arm
        )
        loop.start()
        assert loop._thread is not None
        loop._thread.join(timeout=10.0)
        assert not loop.is_active()
        assert submitted == ["open notepad", "open notepad"]
        assert wake.reset_calls == 3

    def test_confirmation_rearm_flushes_prompt_echo_before_listening(self) -> None:
        """Confirmation re-arm order is prompt → flush → reset → listen.

        Resetting before the prompt was spoken would leave the prompt's own
        "Say hey jarvis to confirm." echo in the very windows the user's
        confirming wake word is scored in, so the flush and reset must happen
        after the prompt, before any read.
        """
        events: list[str] = []

        class _EventAudio(FakeAudioInput):
            def flush(self) -> None:
                events.append("flush")
                super().flush()

        class _EventWake(FakeWakeWord):
            def reset(self) -> None:
                events.append("reset")
                super().reset()

            def detect(self, segment: AudioSegment) -> WakeWordResult:
                events.append("detect")
                return super().detect(segment)

        loop = VoiceLoop(
            audio=_EventAudio(segments=[make_speech("x", duration_s=0.5)] * 10),
            wake_detector=_EventWake(detect_fn=_scripted_wake([True])),
            stt=FakeSTT(),
            tts=FakeTTS(speak_fn=lambda _text: events.append("speak")),
        )
        assert loop._rearm_wake_for_confirmation(timeout_s=1.0)
        assert events == ["speak", "flush", "reset", "detect"]


# ── first wake after a quiet startup (service starts → detector armed → wake) ──


class TestFirstWakeAfterQuietStartup:
    """The very first wake word after the loop starts is detected and drives a
    full interaction even when the startup window was quiet.

    Contract pinned after the reported regression ("daemon starts, says
    nothing, first 'hey jarvis' gets no response"): the detector is armed
    exactly once at startup, quiet startup audio is fed without triggering,
    and the FIRST wake word that follows is never swallowed — neither by the
    re-arm quiet-start gate (which owns re-arm only, never startup) nor by any
    startup-only drain.  If that property is ever violated, this test fails.
    """

    @staticmethod
    def _tone(fill: float, duration_s: float = 0.2) -> AudioSegment:
        n = int(duration_s * 16_000)
        return AudioSegment(samples=[fill] * n, sample_rate=16_000)

    def _submit(self, submitted: list[str]) -> Any:
        def submit(text: str, source: str) -> Any:
            submitted.append(text)
            return type(
                "Outcome", (), {"final_answer": "ok", "confirmation": None, "error": None}
            )()

        return submit

    def test_quiet_startup_then_first_wake_drives_the_interaction(self) -> None:
        """Silence first, then the FIRST wake word: detected, acknowledged,
        processed, and the detector re-armed — exactly once after startup.

        Audio order models a realistic launch: the room is quiet while the
        loop starts listening, the user's first "hey jarvis" arrives, the
        command follows, capture ends on trailing silence, and the idle
        timeout stops the loop after the single interaction.
        """
        submitted: list[str] = []
        quiet = make_silence(0.4)  # startup window: microphone open, nothing said
        wake_1 = self._tone(0.75)  # the FIRST wake word after startup
        cmd_1 = [make_speech("x", duration_s=0.5), make_speech("x", duration_s=0.5)]
        end_1 = make_silence(1.0)  # trailing silence ends capture
        tail = [make_silence(0.1)] * 10  # idle timeout stops the loop

        audio = FakeAudioInput(segments=[quiet, wake_1, *cmd_1, end_1, *tail])
        # Silence is fed and scores False; the first wake word scores True.
        wake = FakeWakeWord(detect_fn=_scripted_wake([False, True, False, False]))
        loop = VoiceLoop(
            audio=audio,
            wake_detector=wake,
            stt=FakeSTT(results=[STTResult(text="open notepad")]),
            tts=FakeTTS(),
            submit_task=self._submit(submitted),
            idle_timeout_s=0.3,
        )
        loop.start()
        assert loop._thread is not None
        loop._thread.join(timeout=10.0)
        assert not loop.is_active()
        # The first wake was honoured end-to-end.
        assert submitted == ["open notepad"]
        assert loop._tts.spoken == ["Yes?", "ok"]
        # The wake word itself was fed to the detector (never skipped), and
        # the quiet window produced nothing.
        assert any(seg.samples[0] == 0.75 for seg in wake.detect_calls)
        assert any(seg.samples[0] == 0.0 for seg in wake.detect_calls)
        # Armed exactly once at startup, plus exactly one re-arm after the
        # interaction — a startup-only quiet gate / drain would add resets or
        # consume the wake chunks and fail the assertions above.
        assert wake.reset_calls == 2


# ── TASK 8 regressions: lifecycle, per-interaction failure isolation ────


class TestInteractionLifecycleRegression:
    """A successful interaction speaks ack + response and the loop keeps
    listening; a transient backend failure never disables voice."""

    def _submit_ok(self, submitted: list[str]) -> Any:
        def submit(text: str, source: str) -> Any:
            submitted.append(text)
            return type(
                "Outcome", (), {"final_answer": "ok", "confirmation": None, "error": None}
            )()

        return submit

    def test_successful_interaction_speaks_and_keeps_listening(self) -> None:
        submitted: list[str] = []
        audio = FakeAudioInput(segments=[make_speech("x", duration_s=0.2)] * 40)
        wake = FakeWakeWord(detect_fn=_scripted_wake([True, False, False, False]))
        loop = VoiceLoop(
            audio=audio,
            wake_detector=wake,
            stt=FakeSTT(results=[STTResult(text="open notepad")]),
            tts=FakeTTS(),
            submit_task=self._submit_ok(submitted),
            idle_timeout_s=5.0,
        )
        loop.start()
        for _ in range(200):
            if len(loop._tts.spoken) >= 2:
                break
            time.sleep(0.05)
        # The whole pipeline ran: wake ack, then the agent's spoken response.
        # (Wait for len == 2 before asserting: ``submit_task`` appends to
        # ``submitted`` just before the loop thread speaks the response, so
        # asserting on ``submitted`` alone races the "ok" TTS.)
        assert loop._tts.spoken == ["Yes?", "ok"]
        assert submitted == ["open notepad"]
        assert loop.is_active()  # still listening for the next wake word
        loop.stop()
        assert not loop.is_active()
        assert wake.reset_calls >= 1  # re-armed after the interaction


# ── Task: per-phase isolation — one failed interaction never wedges ─────


class TestInteractionLifecycleHardening:
    """The first wake must enter _run_interaction, the ack must complete
    before capture starts, every phase failure must only skip that phase
    (the loop re-arms and answers the NEXT wake), and the production INFO
    boundary diagnostics must actually be emitted through the loop's own
    logger."""

    def _submit(self, submitted: list[str], *, fail_first: bool = False) -> Any:
        state = {"calls": 0}

        def submit(text: str, source: str) -> Any:
            state["calls"] += 1
            if fail_first and state["calls"] == 1:
                raise RuntimeError("LLM down")
            submitted.append(text)
            return type(
                "Outcome", (), {"final_answer": "ok", "confirmation": None, "error": None}
            )()

        return submit

    class _FlakyCaptureAudio(FakeAudioInput):
        """Raises once on the first 1600-frame capture read, then recovers."""

        def __init__(self, segments: list[AudioSegment]) -> None:
            super().__init__(segments=segments)
            self._capture_boom = True

        def read(self, num_frames: int) -> AudioSegment:
            self.read_calls.append(num_frames)
            if num_frames == _CAPTURE_CHUNK_FRAMES and self._capture_boom:
                self._capture_boom = False
                raise RuntimeError("mic capture stall")
            if self._idx < len(self._segments):
                seg = self._segments[self._idx]
                self._idx += 1
                return seg
            return AudioSegment(samples=[0.0] * num_frames, sample_rate=self._sample_rate)

    def _run_to_idle(self, loop: VoiceLoop) -> None:
        assert loop._thread is not None
        loop._thread.join(timeout=10.0)
        assert not loop.is_active()

    def test_ack_speaks_before_command_capture_starts(self) -> None:
        """Ordering: wake → speak "Yes?" → flush → first capture chunk read."""
        events: list[tuple[str, Any]] = []

        class EventTTS(FakeTTS):
            def speak(self, text: str) -> None:
                events.append(("speak", text))
                super().speak(text)

        class EventAudio(FakeAudioInput):
            def flush(self) -> None:
                events.append(("flush", None))
                super().flush()

            def read(self, num_frames: int) -> AudioSegment:
                events.append(("read", num_frames))
                return super().read(num_frames)

        submitted: list[str] = []
        audio = EventAudio(segments=[make_speech("x", duration_s=0.5)] * 40)
        wake = FakeWakeWord(detect_fn=_scripted_wake([True, False, False]))
        loop = VoiceLoop(
            audio=audio,
            wake_detector=wake,
            stt=FakeSTT(results=[STTResult(text="open notepad")]),
            tts=EventTTS(),
            submit_task=self._submit(submitted),
            idle_timeout_s=0.3,
        )
        loop.start()
        self._run_to_idle(loop)
        assert submitted == ["open notepad"]
        assert loop._tts.spoken == ["Yes?", "ok"]
        assert ("speak", "Yes?") in events
        assert ("flush", None) in events
        first_capture = next(
            i for i, (op, n) in enumerate(events) if op == "read" and n == _CAPTURE_CHUNK_FRAMES
        )
        speak_idx = events.index(("speak", "Yes?"))
        flush_idx = events.index(("flush", None))
        assert speak_idx < flush_idx < first_capture

    def test_capture_failure_does_not_wedge_loop(self, caplog: object) -> None:
        import logging

        caplog.set_level(logging.INFO, logger="jarvis.voice.loop")  # type: ignore[attr-defined]
        submitted: list[str] = []
        audio = self._FlakyCaptureAudio(segments=[make_speech("x", duration_s=0.5)] * 100)
        wake = FakeWakeWord(detect_fn=_scripted_wake([True, True, False, False]))
        loop = VoiceLoop(
            audio=audio,
            wake_detector=wake,
            stt=FakeSTT(transcribe_fn=lambda _seg: STTResult(text="open notepad")),
            tts=FakeTTS(),
            submit_task=self._submit(submitted),
            idle_timeout_s=0.5,
        )
        loop.start()
        self._run_to_idle(loop)
        # First interaction died at capture; the SECOND wake still worked.
        assert loop._tts.spoken == ["Yes?", "Yes?", "ok"]
        assert submitted == ["open notepad"]
        assert wake.reset_calls >= 2  # re-armed even after the failed capture
        assert "voice interaction failed at capture" in caplog.text  # type: ignore[attr-defined]

    def test_stt_failure_does_not_wedge_loop(self, caplog: object) -> None:
        import logging

        caplog.set_level(logging.INFO, logger="jarvis.voice.loop")  # type: ignore[attr-defined]
        submitted: list[str] = []
        state = {"calls": 0}

        def flaky_stt(_seg: AudioSegment) -> STTResult:
            state["calls"] += 1
            if state["calls"] == 1:
                raise RuntimeError("whisper down")
            return STTResult(text="open notepad")

        wake = FakeWakeWord(detect_fn=_scripted_wake([True, True, False, False]))
        loop = VoiceLoop(
            audio=FakeAudioInput(segments=[make_speech("x", duration_s=0.5)] * 100),
            wake_detector=wake,
            stt=FakeSTT(transcribe_fn=flaky_stt),
            tts=FakeTTS(),
            submit_task=self._submit(submitted),
            idle_timeout_s=0.5,
        )
        loop.start()
        self._run_to_idle(loop)
        assert loop._tts.spoken == ["Yes?", "Yes?", "ok"]
        assert submitted == ["open notepad"]
        assert wake.reset_calls >= 2
        assert "voice interaction failed at stt" in caplog.text  # type: ignore[attr-defined]

    def test_routing_failure_does_not_wedge_loop(self) -> None:
        submitted: list[str] = []
        wake = FakeWakeWord(detect_fn=_scripted_wake([True, True, False, False]))
        loop = VoiceLoop(
            audio=FakeAudioInput(segments=[make_speech("x", duration_s=0.5)] * 100),
            wake_detector=wake,
            stt=FakeSTT(transcribe_fn=lambda _seg: STTResult(text="open notepad")),
            tts=FakeTTS(),
            submit_task=self._submit(submitted, fail_first=True),
            idle_timeout_s=0.5,
        )
        loop.start()
        self._run_to_idle(loop)
        # First route hit a dead submit_task; the loop still re-armed and the
        # SECOND wake completed normally.
        assert loop._tts.spoken == ["Yes?", "Sorry, I couldn't process that.", "Yes?", "ok"]
        assert submitted == ["open notepad"]
        assert wake.reset_calls >= 2

    def test_tts_response_failure_does_not_wedge_loop(self, caplog: object) -> None:
        """A failing response-speech (TTS down) skips only that phase."""
        import logging

        caplog.set_level(logging.INFO, logger="jarvis.voice.loop")  # type: ignore[attr-defined]
        submitted: list[str] = []
        state = {"calls": 0}

        def flaky_speak(text: str) -> None:
            if text != "Yes?":
                state["calls"] += 1
                if state["calls"] == 1:
                    raise RuntimeError("piper down")

        wake = FakeWakeWord(detect_fn=_scripted_wake([True, True, False, False]))
        loop = VoiceLoop(
            audio=FakeAudioInput(segments=[make_speech("x", duration_s=0.5)] * 100),
            wake_detector=wake,
            stt=FakeSTT(transcribe_fn=lambda _seg: STTResult(text="open notepad")),
            tts=FakeTTS(speak_fn=flaky_speak),
            submit_task=self._submit(submitted),
            idle_timeout_s=0.5,
        )
        loop.start()
        self._run_to_idle(loop)
        # First interaction: the command WAS processed and its response text
        # was enqueued, but the spoken response failed; the loop re-armed and
        # the SECOND wake completed fully.
        assert loop._tts.spoken == ["Yes?", "ok", "Yes?", "ok"]
        assert submitted == ["open notepad", "open notepad"]
        assert wake.reset_calls >= 2
        assert "voice interaction failed at tts-response" in caplog.text  # type: ignore[attr-defined]

    def test_unexpected_interaction_exception_is_contained(self, caplog: object) -> None:
        """An exception escaping every per-phase guard still keeps voice on.

        The boundary in ``_run`` catches it, resets to wake-listening, and the
        NEXT wake works — one malformed interaction can never disable voice.
        """
        import logging

        caplog.set_level(logging.INFO, logger="jarvis.voice.loop")  # type: ignore[attr-defined]
        submitted: list[str] = []
        state = {"calls": 0}

        def corrupt_stt(_seg: AudioSegment) -> STTResult:
            state["calls"] += 1
            # First call returns a malformed result (no .text) which blows up
            # in _run_interaction OUTSIDE the per-phase try/except guards.
            if state["calls"] == 1:
                return "not-a-result"  # type: ignore[return-value]
            return STTResult(text="open notepad")

        wake = FakeWakeWord(detect_fn=_scripted_wake([True, True, False, False]))
        loop = VoiceLoop(
            audio=FakeAudioInput(segments=[make_speech("x", duration_s=0.5)] * 100),
            wake_detector=wake,
            stt=FakeSTT(transcribe_fn=corrupt_stt),
            tts=FakeTTS(),
            submit_task=self._submit(submitted),
            idle_timeout_s=0.5,
        )
        loop.start()
        self._run_to_idle(loop)
        assert loop._tts.spoken == ["Yes?", "Yes?", "ok"]
        assert submitted == ["open notepad"]
        assert wake.reset_calls >= 2  # reset even though the interaction blew up
        assert "voice interaction failed; resetting and continuing to listen" in caplog.text  # type: ignore[attr-defined]

    def test_info_diagnostics_emitted_via_production_logger(self, caplog: object) -> None:
        """The per-phase INFO boundary markers the diagnostics rely on are
        emitted through jarvis.voice.loop at INFO, so they reach the root
        file handler in production."""
        import logging

        caplog.set_level(logging.INFO, logger="jarvis.voice.loop")  # type: ignore[attr-defined]
        submitted: list[str] = []
        wake = FakeWakeWord(detect_fn=_scripted_wake([True, False, False]))
        loop = VoiceLoop(
            audio=FakeAudioInput(segments=[make_speech("x", duration_s=0.5)] * 40),
            wake_detector=wake,
            stt=FakeSTT(results=[STTResult(text="open notepad")]),
            tts=FakeTTS(),
            submit_task=self._submit(submitted),
            idle_timeout_s=0.3,
        )
        loop.start()
        self._run_to_idle(loop)
        assert submitted == ["open notepad"]
        text = caplog.text  # type: ignore[attr-defined]
        for marker in (
            "VOICE state=LISTENING",
            "VOICE state=WAKE_DETECTED",
            "VOICE state=CAPTURING",
            "VOICE state=TRANSCRIBING",
            "VOICE state=PROCESSING",
            "VOICE state=SPEAKING",
            "VOICE state=RESETTING",
            "VOICE wake_detected",
            "VOICE capture_started",
            "VOICE capture_finished",
            "VOICE transcript=",
            "voice boundary: interaction start",
            "voice boundary: wake ack speech starting",
            "wake acknowledged — listening for command",
            "voice boundary: capture start",
            "voice boundary: capture done",
            "voice boundary: stt start",
            "voice boundary: stt done",
            "voice boundary: routing command",
            "voice boundary: response ready",
            "voice boundary: tts response speech starting",
            "voice boundary: tts response speech done",
            "voice boundary: interaction completed",
            "voice boundary: re-armed for next wake",
        ):
            assert marker in text, f"missing INFO marker: {marker}"
        # Transcript wording stays out of INFO; only its char count is logged.
        assert "open notepad" not in text


# ── wake-detector frame delivery + transient-error resilience ───────────


class TestWakeDetectorFrameDelivery:
    """Proof that audio frames reach the wake detector, that quiet audio
    never triggers it, and that a transient detector error is absorbed."""

    class _FixedAudio(FakeAudioInput):
        """AudioInput that returns exactly ``num_frames`` samples per read,
        mirroring the bounded-block behaviour of SoundDeviceAudioInput."""

        def read(self, num_frames: int) -> AudioSegment:
            self.read_calls.append(num_frames)
            return AudioSegment(samples=[0.01] * num_frames, sample_rate=16_000)

    def test_frames_reach_detector_but_quiet_audio_never_triggers(self) -> None:
        """Full 1280-sample chunks are fed to the detector; a sub-threshold
        detector stays silent and the loop keeps listening."""
        audio = self._FixedAudio()
        wake = FakeWakeWord(results=[WakeWordResult(detected=False)])
        loop = VoiceLoop(
            audio=audio,
            wake_detector=wake,
            stt=FakeSTT(),
            tts=FakeTTS(),
            idle_timeout_s=0.5,
        )
        loop.start()
        for _ in range(100):
            if len(wake.detect_calls) >= 3:
                break
            time.sleep(0.05)
        assert len(wake.detect_calls) >= 3, "no frames reached the wake detector"
        assert all(len(seg.samples) == 1280 for seg in wake.detect_calls)
        assert loop._tts.spoken == []  # quiet audio never produced a wake
        assert loop.is_active()
        loop.stop()

    def test_transient_detector_error_then_recovers(self, caplog: object) -> None:
        """A one-off wake-detector exception is logged, the detector re-armed,
        and a follow-up detection still drives a full interaction."""
        import logging

        caplog.set_level(logging.INFO, logger="jarvis.voice.loop")  # type: ignore[attr-defined]
        submitted: list[str] = []
        state = {"calls": 0}

        def flaky_detect(_seg: AudioSegment) -> WakeWordResult:
            state["calls"] += 1
            if state["calls"] == 1:
                raise RuntimeError("onnx hiccup")
            return WakeWordResult(detected=state["calls"] == 2)

        def submit(text: str, source: str) -> Any:
            submitted.append(text)
            return type(
                "Outcome", (), {"final_answer": "ok", "confirmation": None, "error": None}
            )()

        audio = FakeAudioInput(segments=[make_speech("x", duration_s=0.2)] * 60)
        wake = FakeWakeWord(detect_fn=flaky_detect)
        loop = VoiceLoop(
            audio=audio,
            wake_detector=wake,
            stt=FakeSTT(results=[STTResult(text="open notepad")]),
            tts=FakeTTS(),
            submit_task=submit,
            idle_timeout_s=5.0,
        )
        loop.start()
        for _ in range(200):
            if loop._tts.spoken == ["Yes?", "ok"]:
                break
            time.sleep(0.05)
        assert submitted == ["open notepad"]  # the SECOND detection drove the interaction
        assert loop._tts.spoken == ["Yes?", "ok"]
        assert loop.is_active()
        loop.stop()
        text = caplog.text  # type: ignore[attr-defined]
        assert "wake detector error" in text
        assert wake.reset_calls >= 2  # re-armed after the transient failure


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


# ── D2: the approval-word split is code, not prompt ─────────────────────


class TestApprovalWordsD2:
    """D2: a destructive/high-risk confirmation demands the strict word set.

    ``can_confirm_by_voice`` already fails closed before the words match, so
    the strict bucket is a defensive second boundary.  These tests pin the
    split so a future "loosen the words" edit breaks tests, not a demo.
    """

    def test_loose_set_serves_ordinary_tier1_payloads(self) -> None:
        assert _approval_words({"tier": 1}) == _LOOSE_YES_WORDS
        assert _approval_words({}) == _LOOSE_YES_WORDS

    def test_typed_folder_name_demands_the_strict_set(self) -> None:
        assert _approval_words({"tier": 1, "typed_confirmation": "Reports"}) == _STRICT_YES_WORDS

    def test_tier2_demands_the_strict_set(self) -> None:
        assert _approval_words({"tier": 2}) == _STRICT_YES_WORDS
        assert _approval_words({"tier": "3"}) == _STRICT_YES_WORDS

    def test_unparsable_tier_falls_back_to_loose_but_voice_is_still_blocked(self) -> None:
        # A junk tier can never reach the matcher: can_confirm_by_voice fails
        # first.  The word fallback only widens the loose set, never the strict.
        assert _approval_words({"tier": "nope"}) == _LOOSE_YES_WORDS
        loop, _, _, _, _ = _make_loop(stt_results=[STTResult(text="yes")])
        pushed: list[dict[str, Any]] = []
        response = loop.confirm_by_voice(
            {"tier": "nope", "summary": "x", "action_hash": "h"},
            on_confirmation=pushed.append,
            rearm_timeout_s=1.0,
        )
        assert response == "Action requires terminal confirmation: x"
        assert pushed == [{"approved": False, "action_hash": "h"}]

    def test_go_ok_sure_approve_an_ordinary_confirmation(self) -> None:
        for word in ("go", "ok", "sure"):
            pushed: list[dict[str, Any]] = []
            loop, _, _, _, _ = _make_loop(stt_results=[STTResult(text=word)])
            loop.confirm_by_voice(
                {"tier": 1, "summary": "open notepad", "action_hash": "h"},
                on_confirmation=pushed.append,
                rearm_timeout_s=1.0,
            )
            assert pushed == [{"approved": True, "action_hash": "h"}], word

    def test_go_ok_sure_cannot_approve_a_typed_payload(self) -> None:
        # Even if a strict payload slipped past can_confirm_by_voice, the word
        # matcher must refuse the colloquial approvals.  The match expression in
        # confirm_by_voice is exactly ``answer in _approval_words(payload)``,
        # so pin the membership directly.
        strict = _approval_words({"typed_confirmation": "Reports"})
        for word in ("go", "ok", "sure"):
            assert word not in strict, word
        assert "yes" in strict
        # And the payload still fails closed at the whole-function boundary.
        pushed: list[dict[str, Any]] = []
        loop, _, _, _, _ = _make_loop(stt_results=[STTResult(text="go")])
        loop.confirm_by_voice(
            {"tier": 1, "typed_confirmation": "Reports", "summary": "wipe", "action_hash": "h"},
            on_confirmation=pushed.append,
            rearm_timeout_s=1.0,
        )
        assert pushed == [{"approved": False, "action_hash": "h"}]

    def test_go_ok_sure_cannot_approve_a_tier2_payload(self) -> None:
        strict = _approval_words({"tier": 2})
        for word in ("go", "ok", "sure"):
            assert word not in strict, word
        pushed: list[dict[str, Any]] = []
        loop, _, _, _, _ = _make_loop(stt_results=[STTResult(text="ok")])
        loop.confirm_by_voice(
            {"tier": 2, "summary": "delete", "action_hash": "h"},
            on_confirmation=pushed.append,
            rearm_timeout_s=1.0,
        )
        assert pushed == [{"approved": False, "action_hash": "h"}]

    def test_yes_never_a_casualty_of_the_strict_split(self) -> None:
        # The unambiguous words must still match under the strict bucket.
        strict = _approval_words({"typed_confirmation": "Reports"})
        assert strict == _STRICT_YES_WORDS
        for word in _STRICT_YES_WORDS:
            assert word in strict, word


# ── voice clarification: capture_free_text ───────────────────────────────


class TestVoiceClarificationCapture:
    def test_returns_the_spoken_transcript(self) -> None:
        answers: list[str] = []
        loop, _, _, _, _ = _make_loop(
            stt_results=[STTResult(text="the report")],
            audio_segments=[make_speech("fake", duration_s=0.5)] * 50,
        )
        text = loop.capture_free_text("Which file?", on_answer=answers.append, rearm_timeout_s=1.0)
        assert text == "the report"
        assert answers == ["the report"]

    def test_empty_utterance_fails_closed(self) -> None:
        loop, _, _, _, _ = _make_loop(
            audio_segments=[make_silence(0.01)] * 2,
            wake_results=[WakeWordResult(detected=True)],
        )
        assert loop.capture_free_text("Which file?", rearm_timeout_s=1.0) == ""

    def test_empty_transcript_fails_closed(self) -> None:
        answers: list[str] = []
        loop, _, _, _, _ = _make_loop(
            stt_results=[STTResult(text="  ")],
            audio_segments=[make_speech("fake", duration_s=0.5)] * 50,
        )
        text = loop.capture_free_text("Which file?", on_answer=answers.append, rearm_timeout_s=1.0)
        assert text == ""
        assert answers == [""]  # stripped utterance → empty resume


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

        def boom_read(_n: int) -> AudioSegment:
            raise RuntimeError("device disconnected")

        audio = FakeAudioInput(segments=[make_silence(0.01)])
        audio.read = boom_read  # type: ignore[method-assign]
        loop = VoiceLoop(
            audio=audio,
            wake_detector=FakeWakeWord(results=[WakeWordResult(detected=False)]),
            stt=FakeSTT(),
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


# ── Exactly-once re-arm + full state machine (TASK: voice hardening) ─────


class TestStateMachineAndRearm:
    """The wake detector is re-armed exactly once after every interaction
    (success or failure), never on per-frame basis, and the INFO state
    sequence matches LISTENING → WAKE_DETECTED → CAPTURING → TRANSCRIBING →
    PROCESSING → SPEAKING → RESETTING → LISTENING."""

    def _submit(self, submitted: list[str]) -> Any:
        def submit(text: str, source: str) -> Any:
            submitted.append(text)
            return type(
                "Outcome", (), {"final_answer": "ok", "confirmation": None, "error": None}
            )()

        return submit

    def test_state_sequence_for_one_interaction(self, caplog: object) -> None:
        """Full cycle logs exactly the 8 required phase states, in order."""
        import logging
        import re

        caplog.set_level(logging.INFO, logger="jarvis.voice.loop")  # type: ignore[attr-defined]
        submitted: list[str] = []
        audio = FakeAudioInput(segments=[make_speech("x", duration_s=0.2)] * 40)
        wake = FakeWakeWord(detect_fn=_scripted_wake([True, False]))
        loop = VoiceLoop(
            audio=audio,
            wake_detector=wake,
            stt=FakeSTT(results=[STTResult(text="open notepad")]),
            tts=FakeTTS(),
            submit_task=self._submit(submitted),
            idle_timeout_s=0.3,
        )
        loop.start()
        assert loop._thread is not None
        loop._thread.join(timeout=10.0)
        assert submitted == ["open notepad"]
        states = re.findall(r"VOICE state=(\w+)", caplog.text)  # type: ignore[arg-type]
        assert states == [
            "LISTENING",
            "WAKE_DETECTED",
            "CAPTURING",
            "TRANSCRIBING",
            "PROCESSING",
            "SPEAKING",
            "RESETTING",
            "LISTENING",
        ]

    def test_wake_detector_rearmed_exactly_once_per_interaction(self) -> None:
        """Two interactions share the startup reset plus ONE re-arm each."""
        submitted: list[str] = []
        audio = FakeAudioInput(segments=[make_speech("x", duration_s=0.5)] * 250)
        wake = FakeWakeWord(detect_fn=_scripted_wake([True, True, False, False]))
        loop = VoiceLoop(
            audio=audio,
            wake_detector=wake,
            stt=FakeSTT(transcribe_fn=lambda _seg: STTResult(text="open notepad")),
            tts=FakeTTS(),
            submit_task=self._submit(submitted),
            idle_timeout_s=0.3,
        )
        loop.start()
        assert loop._thread is not None
        loop._thread.join(timeout=10.0)
        assert submitted == ["open notepad", "open notepad"]
        # 1 (startup) + 1 per interaction — no extra resets.
        assert wake.reset_calls == 3

    def test_wake_detector_rearmed_exactly_once_even_on_failure(self) -> None:
        """A failed interaction still re-arms exactly once (no double reset)."""
        submitted: list[str] = []
        state = {"calls": 0}

        def flaky_stt(_seg: AudioSegment) -> STTResult:
            state["calls"] += 1
            if state["calls"] == 1:
                raise RuntimeError("whisper down")
            return STTResult(text="open notepad")

        audio = FakeAudioInput(segments=[make_speech("x", duration_s=0.5)] * 250)
        wake = FakeWakeWord(detect_fn=_scripted_wake([True, True, False, False]))
        loop = VoiceLoop(
            audio=audio,
            wake_detector=wake,
            stt=FakeSTT(transcribe_fn=flaky_stt),
            tts=FakeTTS(),
            submit_task=self._submit(submitted),
            idle_timeout_s=0.3,
        )
        loop.start()
        assert loop._thread is not None
        loop._thread.join(timeout=10.0)
        assert submitted == ["open notepad"]  # only the second interaction routed
        # Startup reset + one re-arm per interaction (failed + success) = 3.
        assert wake.reset_calls == 3

    def test_no_per_frame_reset_while_waiting(self) -> None:
        """While listening for the wake word the detector is reset only at
        startup — never on every audio frame."""
        audio = FakeAudioInput(segments=[make_speech("x", duration_s=0.2)] * 60)
        wake = FakeWakeWord(results=[WakeWordResult(detected=False)])
        loop = VoiceLoop(
            audio=audio,
            wake_detector=wake,
            stt=FakeSTT(),
            tts=FakeTTS(),
            idle_timeout_s=0.2,
        )
        loop.start()
        assert loop._thread is not None
        loop._thread.join(timeout=10.0)
        assert len(wake.detect_calls) >= 5  # frames were actually fed
        assert wake.reset_calls == 1  # startup only, no per-frame resets

    def test_capture_waits_for_speech_then_ends_on_trailing_silence(self) -> None:
        """End of utterance = trailing silence, never a fixed 30 s window."""
        submitted: list[str] = []
        # Initial silence (the pause after "Yes?") then a short command then
        # trailing silence; the loop must capture all three correctly.
        audio = FakeAudioInput(
            segments=[make_silence(0.5), make_speech("x", duration_s=0.3), make_silence(1.5)]
        )
        wake = FakeWakeWord(detect_fn=_scripted_wake([True, False]))
        loop = VoiceLoop(
            audio=audio,
            wake_detector=wake,
            stt=FakeSTT(results=[STTResult(text="open notepad")]),
            tts=FakeTTS(),
            submit_task=self._submit(submitted),
            listen_timeout_s=5.0,
            silence_timeout_s=0.4,
            idle_timeout_s=0.3,
        )
        loop.start()
        assert loop._thread is not None
        loop._thread.join(timeout=10.0)
        assert submitted == ["open notepad"]
        assert loop._tts.spoken == ["Yes?", "ok"]


# ── Rate-limited RMS diagnostics (loop + wake detector) ──────────────────


class TestVoiceLevelDiagnostics:
    def test_wake_heartbeat_includes_audio_level(self, caplog: object, monkeypatch: object) -> None:
        """The wake-listening heartbeat carries an audio_rms field."""
        import logging

        import jarvis.voice.loop as loop_mod

        monkeypatch.setattr(loop_mod, "_WAKE_HEARTBEAT_S", 0.0)  # type: ignore[attr-defined]
        caplog.set_level(logging.INFO, logger="jarvis.voice.loop")  # type: ignore[attr-defined]
        audio = FakeAudioInput(segments=[make_speech("x", duration_s=0.2)] * 60)
        wake = FakeWakeWord(results=[WakeWordResult(detected=False)])
        loop = VoiceLoop(
            audio=audio,
            wake_detector=wake,
            stt=FakeSTT(),
            tts=FakeTTS(),
            idle_timeout_s=0.2,
        )
        loop.start()
        assert loop._thread is not None
        loop._thread.join(timeout=10.0)
        text = caplog.text  # type: ignore[attr-defined]
        assert "audio_rms=" in text
        assert "waiting for wake word" in text
        assert wake.reset_calls == 1

    def test_wake_heartbeat_is_rate_limited(self, caplog: object) -> None:
        """Heartbeat (and its audio_rms) is logged at ~3 s cadence, not per frame."""
        import logging

        caplog.set_level(logging.INFO, logger="jarvis.voice.loop")  # type: ignore[attr-defined]
        audio = FakeAudioInput(segments=[make_speech("x", duration_s=0.01)] * 300)
        wake = FakeWakeWord(results=[WakeWordResult(detected=False)])
        loop = VoiceLoop(
            audio=audio,
            wake_detector=wake,
            stt=FakeSTT(),
            tts=FakeTTS(),
            idle_timeout_s=0.2,
        )
        loop.start()
        assert loop._thread is not None
        loop._thread.join(timeout=10.0)
        heartbeat_lines = [
            line
            for line in caplog.text.splitlines()  # type: ignore[attr-defined]
            if "waiting for wake word" in line
        ]
        # Many frames processed but far fewer (at most a handful in 0.2 s of
        # listening) heartbeat lines: the cadence gate kept it rate-limited.
        assert len(wake.detect_calls) > len(heartbeat_lines) or not heartbeat_lines
        assert len(wake.detect_calls) >= 20
        assert all("audio_rms=" in line or "elapsed_s=" in line for line in heartbeat_lines)
