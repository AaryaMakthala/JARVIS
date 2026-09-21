"""Wake-word detection via openWakeWord (Phase 5).

Uses the ONNX backend (the only option on Windows).  Speex noise
suppression is Linux-only and not used.

If ``openwakeword`` is not installed, :func:`create` returns ``None``
so callers can fall back to push-to-talk or a voice-activity-only loop.
"""

from __future__ import annotations

import logging
from typing import Any

from jarvis.voice.interfaces import AudioSegment, WakeWordResult

logger = logging.getLogger(__name__)

# Default wake-word model name shipped by openwakeword.
_DEFAULT_MODEL = "hey_jarvis"

# Minimum confidence threshold for a positive detection.
_DEFAULT_THRESHOLD = 0.5


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

    def detect(self, segment: AudioSegment) -> WakeWordResult:
        """Run wake-word inference on the audio segment."""
        import numpy as np

        audio_np = np.array(segment.samples, dtype=np.float32)
        prediction = self._model.predict(audio_np)
        # openWakeWord returns a dict mapping model names to scores.
        score = 0.0
        if isinstance(prediction, dict) or hasattr(prediction, "get"):
            score = float(prediction.get(self._model_name, 0.0))

        detected = score >= self._threshold
        if detected:
            logger.info("wake word detected: model=%s score=%.3f", self._model_name, score)
        return WakeWordResult(
            detected=detected,
            confidence=score,
            model_name=self._model_name,
        )

    def reset(self) -> None:
        """Reset the model's internal rolling buffer."""
        if hasattr(self._model, "reset"):
            self._model.reset()


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
