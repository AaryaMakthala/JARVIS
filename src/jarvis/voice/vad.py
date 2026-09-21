"""Voice-activity detection via sounddevice (Phase 5).

Provides a simple energy-based VAD that segments a continuous audio
stream into speech/silence regions.  The voice loop uses this to
determine when the user has finished speaking.

If ``sounddevice`` is not installed, :func:`create` returns ``None``.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from jarvis.voice.interfaces import AudioSegment

logger = logging.getLogger(__name__)

# Default parameters
_DEFAULT_SAMPLE_RATE = 16_000
_DEFAULT_CHANNELS = 1
_DEFAULT_BLOCK_SIZE = 480  # 30 ms at 16 kHz
_SILENCE_THRESHOLD = 0.01  # RMS energy below this is silence
_SILENCE_TIMEOUT_S = 0.7  # silence after speech to mark segment end
_MAX_SEGMENT_S = 30.0  # hard cap per segment


class SoundDeviceVAD:
    """Energy-based VAD using a sounddevice input stream.

    Constructed via :func:`create`.
    """

    def __init__(
        self,
        stream: Any,
        *,
        sample_rate: int = _DEFAULT_SAMPLE_RATE,
        silence_threshold: float = _SILENCE_THRESHOLD,
        silence_timeout_s: float = _SILENCE_TIMEOUT_S,
        max_segment_s: float = _MAX_SEGMENT_S,
    ) -> None:
        self._stream = stream
        self._sample_rate = sample_rate
        self._silence_threshold = silence_threshold
        self._silence_timeout_s = silence_timeout_s
        self._max_segment_s = max_segment_s

    def listen_for_speech(self) -> AudioSegment:
        """Block until a complete speech segment is captured.

        Returns an :class:`AudioSegment` containing the speech.
        """
        all_samples: list[float] = []
        speech_started = False
        last_speech_time = time.monotonic()

        while True:
            data, overflowed = self._stream.read(_DEFAULT_BLOCK_SIZE)
            if overflowed:
                logger.warning("audio buffer overflowed")
            samples = list(data[:, 0]) if data.ndim > 1 else list(data)
            samples = [float(s) for s in samples]
            rms = _rms(samples)

            now = time.monotonic()
            if rms >= self._silence_threshold:
                speech_started = True
                last_speech_time = now
                all_samples.extend(samples)
            elif speech_started:
                all_samples.extend(samples)
                if (now - last_speech_time) >= self._silence_timeout_s:
                    break
            # If not started, keep discarding silence.

            # Hard cap
            duration = len(all_samples) / self._sample_rate
            if duration >= self._max_segment_s:
                logger.warning("segment hit max duration %.1fs", self._max_segment_s)
                break

        return AudioSegment(samples=all_samples, sample_rate=self._sample_rate)

    def close(self) -> None:
        """Close the underlying audio stream."""
        try:
            self._stream.close()
        except Exception:
            logger.debug("error closing audio stream", exc_info=True)


def _rms(samples: list[float]) -> float:
    """Root-mean-square energy of a sample block."""
    if not samples:
        return 0.0
    n = len(samples)
    return (sum(s * s for s in samples) / n) ** 0.5


def create(
    sample_rate: int = _DEFAULT_SAMPLE_RATE,
    channels: int = _DEFAULT_CHANNELS,
    block_size: int = _DEFAULT_BLOCK_SIZE,
    silence_threshold: float = _SILENCE_THRESHOLD,
    silence_timeout_s: float = _SILENCE_TIMEOUT_S,
) -> SoundDeviceVAD | None:
    """Create a VAD; returns ``None`` if sounddevice is not installed."""
    try:
        import sounddevice as sd
    except ImportError:
        logger.warning(
            "sounddevice not installed — VAD unavailable (install with: pip install sounddevice)"
        )
        return None

    try:
        stream = sd.InputStream(
            samplerate=sample_rate,
            channels=channels,
            blocksize=block_size,
            dtype="float32",
        )
        stream.start()
    except Exception:
        logger.warning("failed to open audio input stream", exc_info=True)
        return None

    return SoundDeviceVAD(
        stream,
        sample_rate=sample_rate,
        silence_threshold=silence_threshold,
        silence_timeout_s=silence_timeout_s,
    )
