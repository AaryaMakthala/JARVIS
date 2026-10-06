"""Spoken lines are echoed on the console: once, redacted, display-only.

:func:`jarvis.voice.console_say.say_line` is the single writer of the
``[VOICE] SAY: "…"`` console line.  Three properties are pinned here:

* **visible** - a confirmation prompt that reaches the TTS engine also reaches
  the console (fake TTS proves the text was spoken, capsys proves it printed);
* **redacted and bounded** - secrets are masked and the printed line is cut to
  ``MAX_SAY_DISPLAY_CHARS`` for display only; the caller's text is untouched;
* **no transcripts at INFO** - the words never reach the JSONL log; at most a
  character count is logged, below INFO.

No audio, no network, no API key.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from jarvis.voice.console_say import MAX_SAY_DISPLAY_CHARS, say_line
from jarvis.voice.fakes import FakeAudioInput, FakeSTT, FakeTTS
from jarvis.voice.interfaces import STTResult
from jarvis.voice.loop import VoiceLoop


def test_say_line_prints_redacted(capsys: pytest.CaptureFixture[str]) -> None:
    secret = "sk-liveABCDEFGHIJKLMNOPQRSTUVWXYZ012345"  # matches redact's sk- pattern
    say_line(f"running with key {secret}")

    out = capsys.readouterr().out
    assert '[VOICE] SAY: "' in out
    assert secret not in out
    assert "<redacted>" in out


def test_say_line_truncates_display_only(capsys: pytest.CaptureFixture[str]) -> None:
    text = "word " * 300  # 1500 chars, far past the display bound
    say_line(text)

    out = capsys.readouterr().out
    say_lines = [line for line in out.splitlines() if line.startswith("[VOICE] SAY:")]
    assert len(say_lines) == 1
    line = say_lines[0]
    assert len(line) <= len('[VOICE] SAY: ""') + MAX_SAY_DISPLAY_CHARS + 1  # + ellipsis
    assert line.endswith('…"')
    # Display-only: the tail of the spoken text is not shown...
    assert out.count("word") <= MAX_SAY_DISPLAY_CHARS // 5 + 1
    # ...and the caller's copy is what it was.
    assert text == "word " * 300


def test_confirmation_prompt_is_printed(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The confirmation prompt spoken by ``confirm_by_voice`` hits the console."""
    tts = FakeTTS()
    loop = VoiceLoop(
        audio=FakeAudioInput(),
        stt=FakeSTT(results=[STTResult(text="yes")]),
        tts=tts,
    )
    # The audio window is irrelevant here: stub it so the test is instant and
    # deterministic, and keep the real prompt -> say_line -> TTS path.
    monkeypatch.setattr(loop, "_capture_spoken_answer", lambda window_s, kind="confirmation": "yes")

    answers: list[dict[str, Any]] = []
    loop.confirm_by_voice(
        {"tier": 1, "summary": "create file", "action_hash": "abc"},
        on_confirmation=answers.append,
    )

    out = capsys.readouterr().out
    assert out.count("[VOICE] SAY:") == 1  # exactly once, not duplicated
    assert "create file" in out
    assert "yes to continue" in out  # the prompt wording itself
    # The same text really went to the TTS engine (fake TTS recorded it).
    assert tts.spoken and "create file" in tts.spoken[0]
    assert answers == [{"approved": True, "action_hash": "abc"}]


def test_spoken_text_not_logged_at_info(caplog: pytest.LogCaptureFixture) -> None:
    phrase = "zeta-unique-spoken-phrase-do-not-log"
    with caplog.at_level(logging.DEBUG):
        say_line(phrase)

    # No record at INFO or above may carry the words (rule: no transcripts at INFO).
    offending = [
        record
        for record in caplog.records
        if record.levelno >= logging.INFO and phrase in record.getMessage()
    ]
    assert offending == []
    # Not even a lower-level record carries them: the only log of a say_line
    # is a character count.
    assert phrase not in caplog.text
    assert any("say_line" in record.getMessage() for record in caplog.records)
