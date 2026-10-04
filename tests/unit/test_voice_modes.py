"""Tests for voice modes (AUTO/NORMAL) - Stage 2."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from tests.unit.test_daemon_voice import _FakeStore
from tests.unit.test_voice_loop import (
    FakeAsyncTTS,
    FakeAudioInput,
    FakeSTT,
    FakeTTS,
    FakeWakeWord,
    STTResult,
    _make_loop,
    _run_scripted,
    _scripted_wake,
    make_silence,
    make_speech,
)

import jarvis.voice.loop as loop_mod
from jarvis.config import Settings, VoiceSettings
from jarvis.daemon.server import DaemonServer
from jarvis.voice.loop import (
    CONFIRMATION_PROMPT,
    CONFIRMATION_REFUSED,
    VoiceLoop,
)
from jarvis.voice.modes import VoiceMode
from jarvis.voice.states import VoicePhase
from jarvis.voice.status import VoiceStatusReporter


class _PhaseRecorder(VoiceStatusReporter):
    """Records the reported phase names instead of printing them."""

    def __init__(self) -> None:
        super().__init__(stream=None, enabled=False)
        self.phases: list[str] = []

    def state(self, phase: VoicePhase, *, detail: str = "", **fields: Any) -> None:
        self.phases.append(str(phase))
        super().state(phase, detail=detail, **fields)


def _ok_answer(text: str) -> Any:
    return type("Outcome", (), {"final_answer": text, "confirmation": None, "error": None})()


class TestModeSwitching:
    """Mode switching by phrases."""

    def test_normal_to_auto_by_phrase(self) -> None:
        submitted: list[str] = []

        def fake_submit(text: str, source: str):
            submitted.append(text)
            return type("Outcome", (), {"final_answer": "ok", "confirmation": None})()

        loop, _, _, _, _ = _make_loop(submit_task=fake_submit)
        result = loop._route("activate auto mode")
        assert result == "Auto mode on."
        assert loop.mode == VoiceMode.AUTO
        assert submitted == []

    def test_normal_to_auto_variants(self) -> None:
        for phrase in ("enter auto mode", "start auto mode"):
            submitted: list[str] = []

            def fake_submit(text: str, source: str, _seen: list[str] = submitted):
                _seen.append(text)
                return type("Outcome", (), {"final_answer": "ok", "confirmation": None})()

            loop, _, _, _, _ = _make_loop(submit_task=fake_submit)
            result = loop._route(phrase)
            assert "auto mode" in result.lower() or "Auto mode" in result
            assert submitted == []

    def test_auto_to_normal_by_sleep(self) -> None:
        submitted: list[str] = []

        def fake_submit(text: str, source: str):
            submitted.append(text)
            return type("Outcome", (), {"final_answer": "ok", "confirmation": None})()

        loop, _, _, _, _ = _make_loop(submit_task=fake_submit)
        loop._route("activate auto mode")
        assert loop.mode == VoiceMode.AUTO
        result = loop._route("jarvis sleep")
        assert result == "Okay, sleeping."
        assert loop.mode == VoiceMode.NORMAL

    def test_sleep_in_normal_says_off_no_change(self) -> None:
        submitted: list[str] = []

        def fake_submit(text: str, source: str):
            submitted.append(text)
            return type("Outcome", (), {"final_answer": "ok", "confirmation": None})()

        loop, _, _, _, _ = _make_loop(submit_task=fake_submit)
        assert loop.mode == VoiceMode.NORMAL
        result = loop._route("go to sleep")
        assert result == "Auto mode is off."
        assert loop.mode == VoiceMode.NORMAL
        assert submitted == []

    def test_already_in_auto(self) -> None:
        submitted: list[str] = []

        def fake_submit(text: str, source: str):
            submitted.append(text)
            return type("Outcome", (), {"final_answer": "ok", "confirmation": None})()

        loop, _, _, _, _ = _make_loop(submit_task=fake_submit)
        loop._route("activate auto mode")
        result = loop._route("activate auto mode")
        assert result == "Already in auto mode."
        assert loop.mode == VoiceMode.AUTO


class TestAutoBehavior:
    """Additional AUTO behavior tests."""

    def test_auto_skips_wake_detector_three_commands(self) -> None:
        """AUTO reaches the planner with no wake word, three times over.

        The detector is scripted to *never* fire, so every command below can
        only have been captured because AUTO skipped the wake wait - and the
        loop is asserted not to have consulted the detector at all.
        """
        submitted: list[str] = []

        def fake_submit(text: str, source: str):
            submitted.append(text)
            return _ok_answer(f"ok:{text}")

        audio = FakeAudioInput(
            segments=[make_speech("cmd", duration_s=0.2), make_silence(1.3)] * 60
        )
        wake = FakeWakeWord(detect_fn=_scripted_wake([False] * 200))
        loop = VoiceLoop(
            audio=audio,
            wake_detector=wake,
            stt=FakeSTT(transcribe_fn=lambda _seg: STTResult(text="what time is it")),
            tts=FakeTTS(),
            submit_task=fake_submit,
            idle_timeout_s=10.0,
        )
        loop._set_mode(VoiceMode.AUTO, reason="test")
        loop.start()
        _run_scripted(loop, lambda: len(submitted) >= 3, what="three AUTO commands")

        assert len(submitted) >= 3
        assert submitted[:3] == ["what time is it"] * 3
        assert wake.detect_calls == [], "AUTO must not consult the wake detector"


class TestModeEdgeCases:
    """Edge cases for mode detection."""

    def test_activate_auto_variants_with_garbage_and_punct(self) -> None:
        from jarvis.voice import modes

        cases = [
            "Activate Auto Mode",
            "activate auto mode.",
            "Hedge R with Activate Auto Mode",
            "Here there is activate auto mode.",
        ]
        for c in cases:
            assert modes.classify_mode_phrase(c) == "activate_auto"

    def test_not_a_mode_phrase_in_middle(self) -> None:
        from jarvis.voice import modes

        assert modes.classify_mode_phrase("what is auto mode in cars") is None


class TestBareSleepLeaveAUTO:
    """The short ways a user says "leave AUTO", and the sentences that must not.

    Live evidence (Stage 2): in AUTO, "Jaro Sleep" and "Sleep." were rejected
    by the usability gate and "Auto-mod off" went to the planner, because only
    the fixed three-word phrases matched.
    """

    @pytest.mark.parametrize(
        "text",
        [
            "Sleep.",
            "Jaro Sleep",
            "Jarwae sleep",
            "Auto-mod off",
            "auto mode off",
            "auto mod off",
            "auto mood off",
            "turn off auto mode",
            "exit auto mode",
            "leave auto mode",
            "stop auto mode",
        ],
    )
    def test_leave_phrases_match_in_auto(self, text: str) -> None:
        from jarvis.voice import modes

        assert modes.classify_mode_phrase(text, VoiceMode.AUTO) == "sleep"

    @pytest.mark.parametrize(
        "text",
        [
            "i cant sleep",
            "what is sleep mode",
            "how long should i sleep",
        ],
    )
    def test_ordinary_sentences_never_match_in_auto(self, text: str) -> None:
        from jarvis.voice import modes

        assert modes.classify_mode_phrase(text, VoiceMode.AUTO) is None

    @pytest.mark.parametrize("text", ["Sleep.", "sleep", "Jaro Sleep"])
    def test_bare_sleep_is_unmatched_in_normal(self, text: str) -> None:
        from jarvis.voice import modes

        assert modes.classify_mode_phrase(text, VoiceMode.NORMAL) is None

    def test_auto_mode_off_still_answers_in_normal(self) -> None:
        """ "auto mode off" is unambiguous in NORMAL too: say it is already off."""
        from jarvis.voice import modes

        assert modes.classify_mode_phrase("auto mode off", VoiceMode.NORMAL) == "sleep"


class TestBareSleepRouting:
    """End-to-end through _route: the mode changes, and the gate is not reached."""

    def test_bare_sleep_leaves_auto(self) -> None:
        submitted: list[str] = []

        def fake_submit(text: str, source: str):
            submitted.append(text)
            return type("Outcome", (), {"final_answer": "ok", "confirmation": None})()

        loop, _, _, _, _ = _make_loop(submit_task=fake_submit)
        loop._route("activate auto mode")
        assert loop.mode == VoiceMode.AUTO
        assert loop._route("Sleep.") == "Okay, sleeping."
        assert loop.mode == VoiceMode.NORMAL
        assert submitted == []

    def test_garbled_wake_sleep_leaves_auto(self) -> None:
        submitted: list[str] = []

        def fake_submit(text: str, source: str):
            submitted.append(text)
            return type("Outcome", (), {"final_answer": "ok", "confirmation": None})()

        loop, _, _, _, _ = _make_loop(submit_task=fake_submit)
        loop._route("activate auto mode")
        assert loop._route("Jaro Sleep") == "Okay, sleeping."
        assert loop.mode == VoiceMode.NORMAL
        assert submitted == []

    def test_auto_mod_off_leaves_auto(self) -> None:
        submitted: list[str] = []

        def fake_submit(text: str, source: str):
            submitted.append(text)
            return type("Outcome", (), {"final_answer": "ok", "confirmation": None})()

        loop, _, _, _, _ = _make_loop(submit_task=fake_submit)
        loop._route("activate auto mode")
        assert loop._route("Auto-mod off") == "Okay, sleeping."
        assert loop.mode == VoiceMode.NORMAL
        assert submitted == []

    def test_auto_mode_off_in_normal_says_already_off(self) -> None:
        submitted: list[str] = []

        def fake_submit(text: str, source: str):
            submitted.append(text)
            return type("Outcome", (), {"final_answer": "ok", "confirmation": None})()

        loop, _, _, _, _ = _make_loop(submit_task=fake_submit)
        assert loop.mode == VoiceMode.NORMAL
        assert loop._route("auto mode off") == "Auto mode is off."
        assert loop.mode == VoiceMode.NORMAL
        assert submitted == []

    def test_bare_sleep_in_normal_is_not_a_mode_command(self) -> None:
        submitted: list[str] = []

        def fake_submit(text: str, source: str):
            submitted.append(text)
            return type("Outcome", (), {"final_answer": "ok", "confirmation": None})()

        loop, _, _, _, _ = _make_loop(submit_task=fake_submit)
        assert loop.mode == VoiceMode.NORMAL
        loop._route("Sleep.")
        assert loop.mode == VoiceMode.NORMAL
        assert submitted == ["Sleep."]

    def test_stop_auto_mode_wins_over_nothing_but_is_not_a_cancel(self) -> None:
        """Stop vocabulary is checked before mode phrases, so it still wins."""
        loop, _, _, _, _ = _make_loop()
        loop._route("activate auto mode")
        assert loop._route("stop") == "Stopped."
        assert loop.mode == VoiceMode.AUTO


class TestAutoIdleTimer:
    """The AUTO idle timer: it fires, and only genuine activity postpones it."""

    def test_auto_sleeps_after_the_idle_timeout(self) -> None:
        """Silence in AUTO puts the loop back to sleep, with an INFO line."""
        tts = FakeTTS()
        loop = VoiceLoop(
            audio=FakeAudioInput(segments=[make_silence(0.2)] * 400),
            wake_detector=FakeWakeWord(detect_fn=_scripted_wake([False] * 500)),
            stt=FakeSTT(transcribe_fn=lambda _seg: STTResult(text="hello")),
            tts=tts,
            auto_idle_timeout_s=0.2,
            idle_timeout_s=0.0,
        )
        loop._set_mode(VoiceMode.AUTO, reason="test")
        loop.start()
        _run_scripted(loop, lambda: loop.mode == VoiceMode.NORMAL, what="auto-sleep")

        assert loop.mode == VoiceMode.NORMAL
        assert "Going back to sleep." in tts.spoken
        assert loop._auto_idle_deadline is None

    @pytest.mark.parametrize("flag", ["_thinking", "_speaking_now", "_in_confirmation"])
    def test_an_expired_timer_is_held_off_while_busy(self, flag: str) -> None:
        """A deadline that has passed still does not fire mid-turn."""
        loop, _, _, _, _ = _make_loop()
        loop._set_mode(VoiceMode.AUTO, reason="test")
        loop._auto_idle_deadline = time.monotonic() - 1.0

        assert loop._check_auto_idle() is True
        setattr(loop, flag, True)
        assert loop._check_auto_idle() is False

    @pytest.mark.parametrize("transcript", ["", "shed on my system."])
    def test_a_non_interaction_does_not_postpone_auto_sleep(self, transcript: str) -> None:
        """An empty or gate-rejected capture is not activity.

        This is why the live log showed no auto-sleep: resetting the deadline on
        every unusable transcript would keep a silent room awake for ever.
        """
        loop, _, _, _, _ = _make_loop(stt_results=[STTResult(text=transcript)])
        loop._set_mode(VoiceMode.AUTO, reason="test")
        loop._auto_idle_timeout_s = 600.0
        deadline = loop._auto_idle_deadline
        assert deadline is not None

        loop._run_interaction()

        assert loop._auto_idle_deadline == deadline
        assert loop.mode == VoiceMode.AUTO


class TestMicrophoneOwnership:
    """Half-duplex: exactly one reader owns the microphone at a time."""

    def test_a_handover_happens_before_another_frame_is_read(self) -> None:
        loop, audio, _, _, _ = _make_loop()
        segment = loop._read_command(16_000, should_continue=lambda: False)
        assert segment.duration_s == 0.0
        assert audio.read_calls == [], "a frame was consumed after the hand-over"


class TestConfirmationWindowContract:
    """Ownership, the policy gate, and the phase vocabulary."""

    def test_confirmation_ownership_stays_with_the_publisher(self) -> None:
        """The loop thread can never approve, and the callback fires once."""
        loop, _, _, _, _ = _make_loop()
        carried = type(
            "Outcome",
            (),
            {"final_answer": "x", "confirmation": {"approved": True}, "error": None},
        )()
        assert loop._extract_response(carried) is None

        seen: list[dict] = []
        loop.confirm_by_voice(
            {"tier": 2, "summary": "s", "action_hash": "h"}, on_confirmation=seen.append
        )
        assert seen == [{"approved": False, "action_hash": "h"}]

    @pytest.mark.parametrize(
        "payload",
        [
            {"tier": 2},
            {"tier": 3},
            {"tier": 1, "typed_confirmation": "Notepad"},
            {"tier": "not-a-number"},
        ],
    )
    def test_policy_gate_refuses_everything_but_tier_1(self, payload: dict) -> None:
        """Tier 2+, a typed folder name, and an unreadable tier all refuse."""
        loop, _, _, _, tts = _make_loop()
        assert loop.can_confirm_by_voice(payload) is False

        seen: list[dict] = []
        spoken = loop.confirm_by_voice(
            {"summary": "delete a folder", "action_hash": "h", **payload},
            on_confirmation=seen.append,
        )
        assert seen == [{"approved": False, "action_hash": "h"}]
        assert spoken == "Action requires terminal confirmation: delete a folder"
        assert CONFIRMATION_PROMPT not in tts.spoken, "a window was opened for a refusal"

    def test_state_vocabulary_is_exactly_the_phase_enum(self) -> None:
        """Every reported phase is a VoicePhase name, and the turn is complete."""
        loop, _, _, _, _ = _make_loop(
            stt_results=[STTResult(text="what time is it")],
            submit_task=lambda _t, _s: _ok_answer("done"),
        )
        loop._reporter = _PhaseRecorder()
        loop._run_interaction()

        seen = loop._reporter.phases  # type: ignore[attr-defined]
        allowed = {phase.value for phase in VoicePhase}
        assert seen, "no phase was reported"
        assert [name for name in seen if name not in allowed] == []
        for name in ("LISTENING", "CAPTURE_COMPLETE", "TRANSCRIBING", "THINKING", "SPEAKING"):
            assert name in seen, f"{name} missing from {seen}"


class TestModeVocabularyCollisions:
    """Mode words colliding with dictation, cancel, and the wake prefix."""

    def test_mode_phrases_are_skipped_while_dictating(self) -> None:
        """Dictation wins: the words are typed, not read as a mode change."""
        typed: list[str] = []
        loop, _, _, _, _ = _make_loop(on_dictation=typed.append)
        loop._route("activate auto mode")
        assert loop.mode == VoiceMode.AUTO

        loop.start_dictation("notepad")
        assert loop._route("auto mode off") is None
        assert loop.mode == VoiceMode.AUTO
        assert typed == ["auto mode off"]

        assert loop._route("stop") == "Dictation stopped."
        assert loop.is_dictating is False
        assert not loop._stop_event.is_set(), "a cancel must never switch voice off"

    @pytest.mark.parametrize("text", ["jarvis stop listening", "hey jarvis stop listening"])
    def test_wake_prefix_is_stripped_before_the_vocabulary_in_auto(self, text: str) -> None:
        """AUTO users still say the wake word; it must not hide a voice-off."""
        loop, _, _, _, _ = _make_loop()
        loop._route("activate auto mode")
        assert loop._route(text) == "Goodbye."
        assert loop.mode == VoiceMode.NORMAL
        assert loop._running is False

    def test_barge_in_ends_the_answer_not_the_auto_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(loop_mod, "_BARGE_IN_GUARD_S", 0.0)
        tts = FakeAsyncTTS(slices=6)
        loop, _audio, wake, _stt, _ = _make_loop(tts=tts)
        loop._set_mode(VoiceMode.AUTO, reason="test")
        loop._auto_idle_timeout_s = 600.0
        wake._detect_fn = _scripted_wake([True] + [True] * 10)

        assert loop._speak_with_barge_in("a long answer") is False
        assert tts.stop_calls == 1
        assert loop.mode == VoiceMode.AUTO
        assert loop._auto_idle_deadline is not None


class TestVoiceOff:
    """Voice-off resets the mode, and voice-off inside a window is honoured."""

    def test_restart_after_voice_off_is_normal(self) -> None:
        """AUTO must not survive a voice-off: a restart needs the wake word."""
        loop, _, _, _, _ = _make_loop()
        loop._route("activate auto mode")
        assert loop.mode == VoiceMode.AUTO

        assert loop._route("stop listening") == "Goodbye."
        assert loop.mode == VoiceMode.NORMAL
        assert loop._running is False

        loop.start()
        try:
            assert loop.mode == VoiceMode.NORMAL
        finally:
            loop.stop()

    def test_stop_listening_in_a_confirmation_window_refuses_and_turns_voice_off(self) -> None:
        loop, _, _, stt, tts = _make_loop(
            audio_segments=[make_speech("stop listening", duration_s=0.5)] * 6
        )
        stt._transcribe_fn = lambda _seg: STTResult(text="stop listening")
        loop._set_mode(VoiceMode.AUTO, reason="test")

        seen: list[dict] = []
        spoken = loop.confirm_by_voice(
            {"tier": 1, "summary": "lock my computer", "action_hash": "h"},
            on_confirmation=seen.append,
            rearm_timeout_s=1.0,
        )

        # Refused, never approved, and the words never became the answer.
        assert seen == [{"approved": False, "action_hash": "h"}]
        assert spoken == CONFIRMATION_REFUSED
        assert loop._voice_off_requested is True
        # The worker thread only requests; the loop thread does the stopping.
        assert not loop._stop_event.is_set()
        assert loop._apply_pending_voice_off() is True
        assert loop._running is False
        assert loop.mode == VoiceMode.NORMAL
        assert "Goodbye." in tts.spoken
        assert loop._apply_pending_voice_off() is False

    def test_stop_listening_in_a_clarification_window_never_reaches_the_planner(self) -> None:
        loop, _, _, stt, _ = _make_loop(
            audio_segments=[make_speech("stop listening", duration_s=0.5)] * 6
        )
        stt._transcribe_fn = lambda _seg: STTResult(text="stop listening")

        reported: list[str] = []
        answer = loop.capture_free_text(
            "Which file?", on_answer=reported.append, rearm_timeout_s=1.0
        )

        assert answer == ""
        assert reported == [""], "the window must report one empty answer, nothing more"
        assert loop._voice_off_requested is True

    def test_a_cancel_in_a_clarification_window_never_reaches_the_planner(self) -> None:
        loop, _, _, stt, _ = _make_loop(audio_segments=[make_speech("stop", duration_s=0.5)] * 6)
        stt._transcribe_fn = lambda _seg: STTResult(text="stop")

        reported: list[str] = []
        answer = loop.capture_free_text(
            "Which file?", on_answer=reported.append, rearm_timeout_s=1.0
        )

        assert answer == ""
        assert reported == [""]
        assert loop._voice_off_requested is False, "a cancel is not a voice-off"

    def test_status_response_reports_the_loop_mode(self) -> None:
        class _Svc:
            state = "on"
            error_code = None

            def __init__(self, voice_loop: VoiceLoop) -> None:
                self._loop = voice_loop

        class _Conn:
            def __init__(self) -> None:
                self.sent: list[Any] = []

            async def send(self, msg: Any) -> None:
                self.sent.append(msg)

        loop, _, _, _, _ = _make_loop()
        server = DaemonServer(
            settings=Settings(voice=VoiceSettings(enabled=True)), store=_FakeStore()
        )
        server._voice_service = _Svc(loop)
        conn = _Conn()

        asyncio.run(server._handle_status(conn))
        assert conn.sent[-1].mode == "NORMAL"

        loop._set_mode(VoiceMode.AUTO, reason="test")
        asyncio.run(server._handle_status(conn))
        assert conn.sent[-1].mode == "AUTO"
