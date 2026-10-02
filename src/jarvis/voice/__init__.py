"""Voice pipeline: wake word, STT, TTS (optional extra).

Audio never leaves the machine; only text goes to LLM APIs.

All hardware/OS dependencies are behind protocols in
:mod:`jarvis.voice.interfaces` so tests use fakes from
:mod:`jarvis.voice.fakes`.
"""

from jarvis.voice.interfaces import (
    AudioInput,
    AudioSegment,
    SpeechToText,
    STTResult,
    TextToSpeech,
    WakeWordDetector,
    WakeWordResult,
    WindowFocusChecker,
)

__all__ = [
    "AudioInput",
    "AudioSegment",
    "STTResult",
    "SpeechToText",
    "TextToSpeech",
    "WakeWordDetector",
    "WakeWordResult",
    "WindowFocusChecker",
]
