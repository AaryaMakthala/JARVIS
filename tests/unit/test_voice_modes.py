"""Tests for voice modes (AUTO/NORMAL) - Stage 2."""

from __future__ import annotations

import pytest
from tests.unit.test_voice_loop import (
    FakeAudioInput,
    FakeSTT,
    FakeTTS,
    FakeWakeWord,
    STTResult,
    _make_loop,
    _scripted_wake,
    make_speech,
)

from jarvis.voice.loop import VoiceLoop
from jarvis.voice.modes import VoiceMode


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

    def test_auto_skips_wake_detector_three_commands(self, caplog: object) -> None:
        """AUTO skips wake detector; three consecutive commands with no wake."""
        import logging

        caplog.set_level(logging.INFO, logger="jarvis.voice.loop")
        submitted: list[str] = []

        def fake_submit(text: str, source: str):
            submitted.append(text)
            return type("Outcome", (), {"final_answer": f"ok:{text}", "confirmation": None})()

        audio = FakeAudioInput(segments=[make_speech("test", duration_s=0.2)] * 40)
        wake = FakeWakeWord(detect_fn=_scripted_wake([False, False, False, False]))
        loop = VoiceLoop(
            audio=audio,
            wake_detector=wake,
            stt=FakeSTT(transcribe_fn=lambda _seg: STTResult(text="hello")),
            tts=FakeTTS(),
            submit_task=fake_submit,
            idle_timeout_s=10.0,
        )
        loop._set_mode(VoiceMode.AUTO, reason="test")
        loop.start()
        import time

        time.sleep(0.5)
        loop.stop()


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
