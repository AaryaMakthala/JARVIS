"""Wake-word detection via openWakeWord (Phase 5).

Uses the ONNX backend (the only option on Windows).  Speex noise
suppression is Linux-only and not used.

If ``openwakeword`` is not installed, :func:`create` returns ``None``
so callers can fall back to push-to-talk or a voice-activity-only loop.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from jarvis.voice.interfaces import AudioSegment, WakeWordResult

logger = logging.getLogger(__name__)

# Default wake-word model name shipped by openwakeword.
_DEFAULT_MODEL = "hey_jarvis"

# Minimum confidence threshold for a positive detection.
_DEFAULT_THRESHOLD = 0.5

#: How often the per-frame score is sampled into the INFO-level log (~2 s).
_INFO_SCORE_INTERVAL_S = 2.0


class OpenWakeWordDetector:
    """Real wake-word detector backed by openWakeWord + ONNX runtime.

    Constructed via :func:`create` which returns ``None`` when the
    dependency is missing so the rest of the voice pipeline can
    degrade gracefully.
    """

    def __init__(
        self,
        model: Any,
        *,
        model_name: str = _DEFAULT_MODEL,
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> None:
        self._model = model
        self._model_name = model_name
        self._threshold = threshold
        # Diagnostic counters (scores only, never audio content).
        self._frames_seen = 0
        self._last_score = 0.0
        self._max_score = 0.0
        self._last_info_ts = 0.0

    @property
    def frames_seen(self) -> int:
        """Number of audio samples fed to the detector since construction."""
        return self._frames_seen

    @property
    def last_score(self) -> float:
        """Most recent model score (0.0 when never invoked)."""
        return self._last_score

    @property
    def max_score(self) -> float:
        """Highest model score observed since construction."""
        return self._max_score

    def detect(self, segment: AudioSegment) -> WakeWordResult:
        """Run wake-word inference on the audio segment.

        The samples are scaled from the normalized float32 form that
        :class:`AudioSegment` carries into the 16-bit PCM format
        openWakeWord 0.6.0 requires: its mel-spectrogram preprocessor
        casts input to ``int16`` (``AudioFeatures._get_melspectrogram``),
        so feeding the [-1, 1] floats directly would truncate every sample
        to {0, ±1} and the detector would silently hear nothing.  Verified
        on real hardware: the same "hey jarvis" clip scores 0.966 as int16
        versus 0.000 as float32.

        The score is always recorded so the diagnostics can prove whether
        audio is reaching the model and *what* it scores; only detected
        frames are reported at INFO by the caller.
        """
        import numpy as np

        samples = np.asarray(segment.samples, dtype=np.float32)
        self._frames_seen += len(samples)
        audio_rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
        audio_int16 = np.round(np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
        prediction = self._model.predict(audio_int16)
        # openWakeWord returns a dict mapping model names to scores.
        score = 0.0
        if isinstance(prediction, dict) or hasattr(prediction, "get"):
            score = float(prediction.get(self._model_name, 0.0))
        self._last_score = score
        self._max_score = max(self._max_score, score)

        detected = score >= self._threshold
        logger.debug(
            "wake detector invoked (model=%s frames=%d score=%.4f threshold=%.2f)",
            self._model_name,
            len(samples),
            score,
            self._threshold,
        )
        now = time.monotonic()
        if now - self._last_info_ts >= _INFO_SCORE_INTERVAL_S:
            self._last_info_ts = now
            logger.info(
                "wake detector sample (model=%s frames_seen=%d last_score=%.4f "
                "max_score=%.4f threshold=%.2f audio_rms=%.4f)",
                self._model_name,
                self._frames_seen,
                score,
                self._max_score,
                self._threshold,
                audio_rms,
            )
        if detected:
            logger.info(
                "wake word detected: model=%s score=%.3f threshold=%.2f",
                self._model_name,
                score,
                self._threshold,
            )
        return WakeWordResult(
            detected=detected,
            confidence=score,
            model_name=self._model_name,
        )

    def reset(self) -> None:
        """Reset the model's internal rolling buffer."""
        if hasattr(self._model, "reset"):
            self._model.reset()
        self._last_score = 0.0
        logger.info(
            "wake detector reset (model=%s, frames_seen=%d)", self._model_name, self._frames_seen
        )


def create(
    model_name: str = _DEFAULT_MODEL,
    threshold: float = _DEFAULT_THRESHOLD,
) -> OpenWakeWordDetector | None:
    """Create a detector; returns ``None`` if openwakeword is not installed.

    The model file is downloaded on first use by openWakeWord.
    """
    try:
        from openwakeword.model import Model as OwwModel
    except ImportError:
        logger.warning(
            "openwakeword not installed — wake-word detection unavailable "
            "(install with: pip install openwakeword)"
        )
        return None

    try:
        model = OwwModel(
            wakeword_models=[model_name],
            inference_framework="onnx",
        )
    except Exception:
        logger.warning("failed to load wake-word model %r", model_name, exc_info=True)
        return None

    return OpenWakeWordDetector(model, model_name=model_name, threshold=threshold)
