"""Speech-to-text via faster-whisper (Phase 5).

Uses the ``tiny`` or ``base`` model with ``int8`` quantisation for
CPU-only operation.  No GPU required.

If ``faster_whisper`` is not installed, :func:`create` returns ``None``.
"""

from __future__ import annotations

import logging
from typing import Any

from jarvis.voice.interfaces import AudioSegment, STTResult

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_SIZE = "base"
_DEFAULT_COMPUTE_TYPE = "int8"
_DEFAULT_LANGUAGE = None  # auto-detect


class FasterWhisperSTT:
    """Real STT backend backed by faster-whisper.

    Constructed via :func:`create`.
    """

    def __init__(
        self,
        model: Any,
        *,
        model_size: str = _DEFAULT_MODEL_SIZE,
        language: str | None = _DEFAULT_LANGUAGE,
        initial_prompt: str | None = None,
    ) -> None:
        self._model = model
        self._model_size = model_size
        # "" (config's "auto-detect" spelling) means None = faster-whisper
        # auto-detect; "" initial_prompt means "no vocabulary hint".
        self._language = language or None
        self._initial_prompt = initial_prompt or None

    def transcribe(self, segment: AudioSegment) -> STTResult:
        """Transcribe an audio segment to text."""
        import numpy as np

        audio_np = np.array(segment.samples, dtype=np.float32)
        # faster-whisper expects 1-D float32 numpy array at 16 kHz.
        segments, info = self._model.transcribe(
            audio_np,
            language=self._language,
            initial_prompt=self._initial_prompt,
            # Do not carry a previous (possibly garbage) transcript forward as
            # conditioning: one bad detection must not poison the next one.
            condition_on_previous_text=False,
            beam_size=1,
            vad_filter=False,  # we do our own VAD
        )
        texts = [seg.text for seg in segments]
        full_text = " ".join(texts).strip()
        language = getattr(info, "language", "") or ""
        return STTResult(
            text=full_text,
            language=language,
            duration_s=segment.duration_s,
        )


def create(
    model_size: str = _DEFAULT_MODEL_SIZE,
    compute_type: str = _DEFAULT_COMPUTE_TYPE,
    language: str | None = _DEFAULT_LANGUAGE,
    initial_prompt: str | None = None,
) -> FasterWhisperSTT | None:
    """Create an STT engine; returns ``None`` if faster-whisper is missing."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        logger.warning(
            "faster-whisper not installed — STT unavailable "
            "(install with: pip install faster-whisper)"
        )
        return None

    try:
        model = WhisperModel(
            model_size,
            device="cpu",
            compute_type=compute_type,
        )
    except Exception:
        logger.warning(
            "failed to load whisper model %r (compute_type=%s)",
            model_size,
            compute_type,
            exc_info=True,
        )
        return None

    return FasterWhisperSTT(
        model,
        model_size=model_size,
        language=language,
        initial_prompt=initial_prompt,
    )
