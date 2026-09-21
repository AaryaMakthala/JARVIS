"""sounddevice-backed :class:`AudioInput` — the single mic-stream owner.

Phase 5 architecture rule: *exactly one* component may own the
``sd.InputStream``.  That component is :class:`SoundDeviceAudioInput`.
It is the only place a microphone stream is created; wake-word detection,
VAD, and STT all consume the float frames this class hands them.

Capture runs on sounddevice's realtime callback thread, so the callback
must never block or allocate unbounded buffers.  Frames are pushed onto a
**bounded** queue; if the queue is full the *oldest* buffered frames are
dropped (counted) so the most recent audio wins and memory stays bounded.

Audio never leaves the machine and is never persisted or logged — only
float samples cross into this object (docs/03 §10, invariant).
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Callable

from jarvis.voice.interfaces import AudioSegment

logger = logging.getLogger(__name__)

_DEFAULT_SAMPLE_RATE = 16_000
_DEFAULT_CHANNELS = 1
_DEFAULT_BLOCK_SIZE = 1280  # 80 ms at 16 kHz
_DEFAULT_QUEUE_MAXSIZE = 8  # at most 8 blocks buffered (~640 ms)
_DEFAULT_READ_TIMEOUT_S = 0.5

_INT16_DIVISOR = 32_768.0

_CALLBACK_STATUS_LOG_INTERVAL_S = 5.0
_callback_last_status_log = 0.0


class VoiceInputError(Exception):
    """Raised when a mic stream cannot be opened or read.

    ``str(exc)`` is a **sanitized** message: it never contains the raw
    exception text, device-driver strings, or any data that could leak
    machine-identifying details (docs/03 §10).
    """


class SoundDeviceAudioInput:
    """Microphone input owning the single `sd.InputStream` (16 kHz mono).

    Constructed via :func:`create`.  Use as a context manager
    (``with input_device() as audio:``) so the stream is always released,
    even on error; a crash therefore cannot leave the mic open.
    """

    def __init__(
        self,
        *,
        sample_rate: int = _DEFAULT_SAMPLE_RATE,
        channels: int = _DEFAULT_CHANNELS,
        block_size: int = _DEFAULT_BLOCK_SIZE,
        queue_maxsize: int = _DEFAULT_QUEUE_MAXSIZE,
        read_timeout_s: float = _DEFAULT_READ_TIMEOUT_S,
        device: int | str | None = None,
        stream_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._block_size = block_size
        self._queue_maxsize = queue_maxsize
        self._read_timeout_s = read_timeout_s
        self._device = device
        self._stream_factory = stream_factory
        self._stream: Any = None
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=queue_maxsize)
        self._dropped_oldest = 0
        self._open = False

    # ── AudioInput protocol ──────────────────────────────────────────────

    def open(self) -> None:
        """Open the microphone stream (idempotent; clears stale audio)."""
        if self._open:
            return
        self._dropped_oldest = 0
        self._clear_queue()

        device: Any = None
        if self._device is not None:
            try:
                import sounddevice as sd

                device = self._resolve_device(sd, self._device)
            except ImportError:
                raise VoiceInputError(
                    "sounddevice is not installed — cannot open a microphone "
                    "(install with: pip install sounddevice)"
                ) from None

        try:
            if self._stream_factory is not None:
                self._stream = self._stream_factory(
                    sample_rate=self._sample_rate,
                    channels=self._channels,
                    block_size=self._block_size,
                    dtype="int16",
                    callback=self._callback,
                    device=device,
                )
            else:
                import sounddevice as sd

                self._stream = sd.InputStream(
                    sample_rate=self._sample_rate,
                    channels=self._channels,
                    block_size=self._block_size,
                    dtype="int16",
                    callback=self._callback,
                    device=device,
                )
            self._stream.start()
        except VoiceInputError:
            raise
        except Exception:
            logger.warning(
                "could not open the microphone — check that a microphone is "
                "connected and enabled on Windows"
            )
            raise VoiceInputError(
                "could not open the microphone (check that a microphone is "
                "connected and enabled on Windows)"
            ) from None
        self._open = True
        logger.info(
            "microphone open (device=%s)",
            self._device_label(device),
        )

    def read(self, num_frames: int) -> AudioSegment:
        """Read *num_frames* samples as float (blocking, bounded).

        Expected to be called in a tight loop (e.g. from the voice loop);
        returns whatever is available up to the queue timeout.  int16 input
        is converted to float by dividing by 32768.
        """
        frames: list[float] = []
        while len(frames) < num_frames:
            try:
                block = self._queue.get(timeout=self._read_timeout_s)
            except queue.Empty:
                break
            frames.extend(self._block_to_float(block))
        return AudioSegment(samples=frames, sample_rate=self._sample_rate)

    def close(self) -> None:
        """Close the stream and drain the queue (safe to call twice)."""
        if not self._open:
            return
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            logger.debug("error closing microphone stream", exc_info=True)
        self._stream = None
        self._open = False
        self._clear_queue()
        logger.info("microphone closed")

    def is_open(self) -> bool:
        """Return True while the stream is open."""
        return self._open

    # ── extra API used by the voice loop ─────────────────────────────────

    def flush(self) -> None:
        """Discard all buffered audio and reset the drop counter.

        Called before confirmation listening so stale audio captured before
        a prompt is never used to approve an action; also called after TTS
        so a voice response is never mistaken for a follow-up command.
        """
        self._clear_queue()
        self._dropped_oldest = 0
        logger.debug("audio input flushed")

    @property
    def dropped_oldest(self) -> int:
        """Number of oldest-blocks dropped since the last flush/open."""
        return self._dropped_oldest

    # ── internals ────────────────────────────────────────────────────────

    def _callback(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        """Realtime callback (never blocks, never grows the queue)."""
        global _callback_last_status_log
        if status:
            now = time.monotonic()
            if now - _callback_last_status_log >= _CALLBACK_STATUS_LOG_INTERVAL_S:
                _callback_last_status_log = now
                logger.warning("microphone stream status=%r", str(status))
        try:
            block = bytes(indata)  # int16 mono, C-contiguous
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                    self._dropped_oldest += 1
                except queue.Empty:
                    pass
            self._queue.put_nowait(block)
        except Exception:
            # Never let the callback thread crash; drop and count instead.
            logger.debug("audio callback dropped a block", exc_info=True)
            self._dropped_oldest += 1

    @staticmethod
    def _block_to_float(block: bytes) -> list[float]:
        """Convert an int16 PCM block to float samples (÷32768)."""
        from array import array

        raw = array("h")
        raw.frombytes(block)
        return [s / _INT16_DIVISOR for s in raw]

    @staticmethod
    def _resolve_device(sd: Any, device: int | str) -> Any:
        """Map *device* to a sounddevice device id (int or name substring).

        Raises :class:`VoiceInputError` with a sanitized message if the
        device cannot be found.  No raw exception text is included.
        """
        if isinstance(device, int):
            total = max(sd.query_devices().__len__(), 1)
            if device < 0 or device >= total:
                raise VoiceInputError(
                    f"voice.input_device {device} is out of range "
                    f"(0..{total - 1} on this machine)"
                )
            if int(sd.query_devices(device).get("max_input_channels", 0)) <= 0:
                raise VoiceInputError(
                    f"voice.input_device {device} is not a microphone "
                    "(it has no input channels)"
                )
            return device

        name = str(device)
        candidates: list[Any] = []
        try:
            count = len(sd.query_devices())
            for idx in range(count):
                dev = sd.query_devices(idx)
                if name.lower() in str(dev.get("name", "")).lower():
                    candidates.append(idx)
        except Exception:
            raise VoiceInputError("could not enumerate audio devices") from None
        if not candidates:
            raise VoiceInputError(
                f"voice.input_device {name!r} matched no microphone "
                "(use `jarvis doctor` to list devices)"
            )
        # Prefer an input-capable device from the *default* host API
        # (MME/DirectSound on Windows); otherwise take the first match.
        try:
            default_hostapi = int(sd.query_devices(kind="input").get("hostapi", -1))
        except Exception:
            default_hostapi = -1
        for idx in candidates:
            if int(sd.query_devices(idx).get("max_input_channels", 0)) <= 0:
                continue
            if int(sd.query_devices(idx).get("hostapi", -1)) == default_hostapi:
                return idx
        for idx in candidates:
            if int(sd.query_devices(idx).get("max_input_channels", 0)) > 0:
                return idx
        return candidates[0]

    @staticmethod
    def _device_label(device: Any) -> str:
        """Return a human-readable label for a resolved device id.

        Falls back to ``"default microphone"`` when sounddevice is not
        installed.  No raw device-driver text is included.
        """
        try:
            import sounddevice as sd

            dev = sd.query_devices(device if device is not None else "default")
            return str(dev.get("name", "default microphone"))
        except Exception:
            return "default microphone"

    def _clear_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break


def create(
    *,
    input_device: int | str | None = None,
    sample_rate: int = _DEFAULT_SAMPLE_RATE,
    channels: int = _DEFAULT_CHANNELS,
    block_size: int = _DEFAULT_BLOCK_SIZE,
    queue_maxsize: int = _DEFAULT_QUEUE_MAXSIZE,
    read_timeout_s: float = _DEFAULT_READ_TIMEOUT_S,
) -> "SoundDeviceAudioInput | None":
    """Create a microphone input backed by sounddevice.

    Returns ``None`` if sounddevice is not installed (the daemon then runs
    without live audio).  Device resolution happens in :meth:`open`, never
    at config-load time.
    """
    try:
        import sounddevice as sd  # noqa: F401
    except ImportError:
        logger.warning(
            "sounddevice not installed — voice audio unavailable "
            "(install with: pip install sounddevice)"
        )
        return None
    return SoundDeviceAudioInput(
        device=input_device,
        sample_rate=sample_rate,
        channels=channels,
        block_size=block_size,
        queue_maxsize=queue_maxsize,
        read_timeout_s=read_timeout_s,
    )
