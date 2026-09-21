"""Voice pipeline: wake word, STT, TTS, VAD (optional extra, Phase 5).

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
    VoiceCommand,
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
    "VoiceCommand",
    "WakeWordDetector",
    "WakeWordResult",
    "WindowFocusChecker",
]
