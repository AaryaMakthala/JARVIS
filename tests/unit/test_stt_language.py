"""Language / initial_prompt plumbing into faster-whisper's transcribe.

Regression coverage for the mis-detection bug: short commands ("proceed",
"stop", ...) were auto-detected as Russian/Japanese nonsense because the model
ran in language-auto mode with no vocabulary conditioning and carried a
garbage previous transcript forward.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from jarvis.config import STT_INITIAL_PROMPT_HINT, VoiceSettings
from jarvis.voice.interfaces import AudioSegment
from jarvis.voice.stt import FasterWhisperSTT


class _FakeWhisperModel:
    """Stands in for faster_whisper.WhisperModel; records transcribe kwargs."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def transcribe(self, audio: Any, **kwargs: Any) -> tuple[list[Any], Any]:
        self.calls.append(kwargs)
        return [], SimpleNamespace(language="en")


def _make(
    language: str | None = None, initial_prompt: str | None = None
) -> tuple[FasterWhisperSTT, _FakeWhisperModel]:
    model = _FakeWhisperModel()
    kwargs: dict[str, Any] = {}
    if language is not None:
        kwargs["language"] = language
    if initial_prompt is not None:
        kwargs["initial_prompt"] = initial_prompt
    return FasterWhisperSTT(model, **kwargs), model


def _transcribe(stt: FasterWhisperSTT, model: _FakeWhisperModel) -> dict[str, Any]:
    stt.transcribe(AudioSegment(samples=[0.0] * 1600))
    assert len(model.calls) == 1
    return model.calls[0]


def test_language_en_passed_by_default() -> None:
    # The shipped config default is "en", and it reaches the whisper call.
    assert VoiceSettings().stt_language == "en"
    stt, model = _make(language=VoiceSettings().stt_language)
    kwargs = _transcribe(stt, model)
    assert kwargs["language"] == "en"
    # The fix for "one bad detection poisons the next": never condition on a
    # previous transcript.
    assert kwargs["condition_on_previous_text"] is False


def test_empty_language_means_auto() -> None:
    # voice.stt_language = "" must become faster-whisper auto-detect (None).
    stt, model = _make(language="")
    kwargs = _transcribe(stt, model)
    assert kwargs["language"] is None


def test_initial_prompt_passed() -> None:
    assert VoiceSettings().stt_initial_prompt == STT_INITIAL_PROMPT_HINT
    stt, model = _make(
        language="en",
        initial_prompt=VoiceSettings().stt_initial_prompt,
    )
    kwargs = _transcribe(stt, model)
    assert kwargs["initial_prompt"] == STT_INITIAL_PROMPT_HINT
    assert "proceed" in kwargs["initial_prompt"]


def test_initial_prompt_off_when_empty() -> None:
    stt, model = _make(language="en", initial_prompt="")
    kwargs = _transcribe(stt, model)
    assert kwargs["initial_prompt"] is None
