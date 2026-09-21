"""Text-to-speech via Piper, falling back to Windows SAPI (Phase 5).

Piper produces more natural speech but is GPL-3.0 and may not install
on every Windows/Python combination.  ``pyttsx3`` (wrapping Windows
SAPI5) is always available as a fallback.

:func:`create` tries Piper first; if it fails, falls back to SAPI.
If neither works, returns ``None``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class PiperTTSEngine:
    """TTS via piper-tts (GPL-3.0)."""

    def __init__(self, model_path: str, *, config_path: str | None = None) -> None:
        self._model_path = model_path
        self._config_path = config_path
        self._synth: Any = None

    def _ensure_loaded(self) -> None:
        if self._synth is not None:
            return
        from piper import PiperVoice

        self._synth = PiperVoice.load(self._model_path, config_path=self._config_path)

    def speak(self, text: str) -> None:
        """Speak *text* synchronously."""
        import numpy as np
        import sounddevice as sd

        self._ensure_loaded()
        # PiperVoice.synthesize_stream_raw returns raw PCM int16 chunks.
        audio_chunks: list[bytes] = list(self._synth.synthesize_stream_raw(text))
        raw = b"".join(audio_chunks)
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        sd.play(audio, self._synth.config.sample_rate)
        sd.wait()

    def stop(self) -> None:
        """Stop any in-progress speech."""
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:
            logger.debug("sounddevice stop failed", exc_info=True)


class SapiTTSEngine:
    """TTS via Windows SAPI (pyttsx3)."""

    def __init__(self) -> None:
        self._engine: Any = None

    def _ensure_loaded(self) -> None:
        if self._engine is not None:
            return
        import pyttsx3

        self._engine = pyttsx3.init()

    def speak(self, text: str) -> None:
        """Speak *text* synchronously."""
        self._ensure_loaded()
        self._engine.say(text)
        self._engine.runAndWait()

    def stop(self) -> None:
        """Stop any in-progress speech."""
        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception:
                logger.debug("pyttsx3 stop failed", exc_info=True)


def create(
    *,
    backend: str = "piper",
    model_path: str | None = None,
) -> PiperTTSEngine | SapiTTSEngine | None:
    """Create a TTS engine.

    Tries ``piper`` first if requested, then falls back to ``sapi``.
    Returns ``None`` if nothing works.
    """
    if backend == "piper":
        try:
            if model_path:
                return PiperTTSEngine(model_path)
            logger.warning(
                "piper backend requested but no model_path provided; falling back to SAPI"
            )
        except ImportError:
            logger.info("piper-tts not installed; falling back to SAPI")
        except Exception:
            logger.warning("piper-tts init failed", exc_info=True)

    # SAPI fallback
    try:
        return SapiTTSEngine()
    except ImportError:
        logger.warning(
            "pyttsx3 not installed — TTS unavailable (install with: pip install pyttsx3)"
        )
    except Exception:
        logger.warning("pyttsx3 init failed", exc_info=True)

    return None
