"""Voice pipeline diagnostics for ``jarvis voice doctor`` (Phase 5).

Probes the same factory functions the daemon's ``_build_voice_service`` uses,
in isolation, and returns a list of :class:`~jarvis.checks.Check` results so
the CLI can render them as a table.

Privacy: the microphone probe reads one short fixed-length segment and only
reports a sample count; STT is probed with silence; no speech is recorded to
disk and no transcript text is returned.
"""

from __future__ import annotations

import logging

from jarvis import config
from jarvis.checks import Check, _check
from jarvis.voice.interfaces import AudioSegment
from jarvis.voice.service import VOICE_HINTS

logger = logging.getLogger(__name__)

#: Maximum length of an exception string surfaced to the CLI.
_SANITIZE_LIMIT = 200


def run_voice_doctor(
    settings: config.Settings | None = None,
    *,
    mic: bool = True,
    stt: bool = True,
    tts: bool = True,
) -> list[Check]:
    """Diagnose the voice pipeline as the daemon wires it.

    ``mic`` / ``stt`` / ``tts`` flags skip the corresponding (potentially
    slow or downloading) probes.  The wake-word probe always runs: it is
    cheap and is the link that most often fails silently.
    """
    settings = settings or config.load_settings()
    checks: list[Check] = []
    vs = settings.voice

    if vs.enabled:
        checks.append(_check("voice_config", "PASS", "voice enabled in config.toml"))
    else:
        checks.append(
            _check(
                "voice_config",
                "WARN",
                "[voice] enabled = false so `jarvis on` is refused until it is set",
            )
        )

    if mic:
        checks.extend(_probe_audio(settings))
    checks.extend(_probe_wake(vs.wake_word))
    if stt:
        checks.append(_probe_stt(vs.stt_model))
    if tts:
        checks.append(_probe_tts(vs.tts_backend))
    return checks


def _probe_audio(settings: config.Settings) -> list[Check]:
    """Check the mic library, default input device, and a short read."""
    from jarvis.voice.audio_input import create as create_audio_input

    audio = create_audio_input(
        sample_rate=16_000,
        channels=1,
        input_device=settings.voice.input_device,
    )
    if audio is None:
        return [
            _check(
                "audio_library",
                "FAIL",
                "sounddevice not installed "
                "(install the voice extras: pip install jarvis-agent[voice])",
            )
        ]

    checks: list[Check] = [  # library import proves sounddevice is present
        _check("audio_library", "PASS", "sounddevice installed")
    ]
    device_name, err = _input_device_name(settings.voice.input_device)
    if err:
        checks.append(_check("mic_device", "FAIL", err))
    else:
        checks.append(_check("mic_device", "PASS", f"input device: {device_name}"))

    try:
        audio.open()
        try:
            segment = audio.read(16_000)
        finally:
            audio.close()
    except Exception as exc:
        logger.debug("voice doctor: mic probe failed", exc_info=exc)
        checks.append(_check("mic_open", "FAIL", f"could not read from the mic: {_sanitize(exc)}"))
    else:
        if segment.duration_s <= 0:
            checks.append(_check("mic_open", "FAIL", "mic opened but returned no audio"))
        else:
            checks.append(
                _check(
                    "mic_open",
                    "PASS",
                    f"read {len(segment.samples)} samples ({segment.duration_s:.2f}s)",
                )
            )
    return checks


def _input_device_name(input_device: int | str | None) -> tuple[str | None, str | None]:
    """Resolve the default (or configured) input device for display."""
    import sounddevice as sd

    try:
        if input_device is not None:
            info = sd.query_devices(input_device, "input")
        else:
            info = sd.query_devices(kind="input")
        name = str(info["name"])
        channels = int(info["max_input_channels"])
        return f"{name} ({channels} ch)", None
    except Exception as exc:
        logger.debug("voice doctor: device query failed", exc_info=exc)
        return None, f"no usable input device: {_sanitize(exc)}"


def _probe_wake(wake_word: str) -> list[Check]:
    """Load the wake model and run one inference on silence."""
    from jarvis.voice.wake import create as create_wake

    wake = create_wake(model_name=wake_word)
    if wake is None:
        return [
            _check(
                "wake_model",
                "FAIL",
                f"'{wake_word}': {VOICE_HINTS['wake-model-missing']}",
            )
        ]
    try:
        wake.reset()
        result = wake.detect(AudioSegment(samples=[0.0] * 1280))
    except Exception as exc:
        logger.debug("voice doctor: wake inference failed", exc_info=exc)
        return [_check("wake_detection", "FAIL", f"inference failed: {_sanitize(exc)}")]
    return [
        _check("wake_model", "PASS", f"model '{wake_word}' loaded"),
        _check(
            "wake_detection",
            "PASS",
            f"engine ready (silence probe confidence {result.confidence:.2f}); "
            "say 'hey jarvis' to verify live",
        ),
    ]


def _probe_stt(model_size: str) -> Check:
    """Load the STT model and transcribe one second of silence."""
    if not model_size:
        return _check("stt_model", "FAIL", "no stt_model set in config.toml")
    from jarvis.voice.stt import create as create_stt

    stt = create_stt(model_size=model_size)
    if stt is None:
        return _check("stt_model", "FAIL", VOICE_HINTS["stt-model-missing"])
    try:
        stt.transcribe(AudioSegment(samples=[0.0] * 16_000))
    except Exception as exc:
        logger.debug("voice doctor: STT probe failed", exc_info=exc)
        return _check("stt_model", "FAIL", f"inference failed: {_sanitize(exc)}")
    return _check(
        "stt_model",
        "PASS",
        f"'{model_size}' loaded and silent probe transcribed (first use downloads the model)",
    )


def _probe_tts(backend: str) -> Check:
    """Initialise the TTS engine without playing any audio."""
    if not backend:
        return _check("tts", "FAIL", "no tts_backend set in config.toml")
    from jarvis.voice.tts import create as create_tts

    tts = create_tts(backend=backend)
    if tts is None:
        return _check("tts", "FAIL", VOICE_HINTS["tts-unavailable"])
    return _check(
        "tts",
        "PASS",
        f"{type(tts).__name__} ready (doctor does not play audio)",
    )


def _sanitize(exc: BaseException) -> str:
    """Return a bounded, single-line error text; never the traceback."""
    text = (str(exc).strip() or type(exc).__name__).replace("\n", " ")
    if len(text) > _SANITIZE_LIMIT:
        text = text[:_SANITIZE_LIMIT] + "..."
    return text
