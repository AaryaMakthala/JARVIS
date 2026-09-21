"""SoundDeviceAudioInput tests against a strict fake sounddevice (no hardware).

The fake ``InputStream`` accepts exactly the keyword parameters the real
``sounddevice.InputStream`` declares (verified from the library source),
so any wrong kwarg name — e.g. the previously-used ``sample_rate`` /
``block_size`` instead of ``samplerate``/``blocksize`` — raises
``TypeError`` exactly like the real library would.

Coverage (Stage 1 f):
* callback -> read round-trip and int16 -> float /32768 scaling
* exact-n reads with surplus carried over to the next read
* drop-oldest bounded queue with counter
* flush, idempotent close, close-unblocks-a-blocked-read
* device resolution errors (out of range, not a microphone, no name match,
  prefers the default host API) with sanitized VoiceInputError
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Any

import numpy as np
import pytest

from jarvis.voice.audio_input import (
    SoundDeviceAudioInput,
    VoiceInputError,
    _stream_kwargs,
)

_BLOCK = 1280
_READ_OK = 0.05  # keep stall-based reads fast in tests

# Real sd.InputStream.__init__ parameter names (from the library source).
_REAL_STREAM_PARAMS = frozenset(
    {
        "samplerate",
        "blocksize",
        "device",
        "channels",
        "dtype",
        "latency",
        "extra_settings",
        "callback",
        "finished_callback",
        "clip_off",
        "dither_off",
        "never_drop_input",
        "prime_output_buffers_using_stream_callback",
    }
)


def _int16_block(values: list[int]) -> np.ndarray:
    return np.array(values, dtype=np.int16)


class StrictInputStream:
    """Rejects any kwarg the real sd.InputStream does not take."""

    def __init__(self, **kwargs: Any) -> None:
        unknown = set(kwargs) - _REAL_STREAM_PARAMS
        if unknown:
            raise TypeError(
                f"InputStream got unexpected keyword arguments {sorted(unknown)}"
            )
        self.kwargs = dict(kwargs)
        self.started = False
        self.stopped = False
        self.closed = False

    @property
    def callback(self) -> Any:
        return self.kwargs["callback"]

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


def _device(index: int, name: str, *, hostapi: int = 0, ins: int = 2) -> dict[str, Any]:
    return {
        "index": index,
        "name": name,
        "hostapi": hostapi,
        "max_input_channels": ins,
        "max_output_channels": 0,
        "default_samplerate": 16000.0,
        "default_low_input_latency": 0.01,
        "default_high_input_latency": 0.1,
    }


class FakeSD:
    """Dict-shaped stand-in for `sounddevice.query_devices`."""

    def __init__(self, devices: list[dict[str, Any]], default_input: int | None = None) -> None:
        self._devices = list(devices)
        self._default_input = default_input

    def query_devices(self, device: Any = None, kind: str | None = None) -> Any:
        if kind is not None:
            if self._default_input is None:
                raise IndexError("no default input device")
            return dict(self._devices[self._default_input])
        if device is None:
            return tuple(dict(d) for d in self._devices)  # DeviceList-like
        if isinstance(device, int):
            if not 0 <= device < len(self._devices):
                raise IndexError(f"invalid device id {device}")
            return dict(self._devices[device])
        raw = str(device).lower()
        idx = next(
            (i for i, d in enumerate(self._devices) if raw in d["name"].lower()), None
        )
        if idx is None:
            raise ValueError(f"no device named {device!r}")
        return dict(self._devices[idx])


class TestOpen:
    def test_open_uses_real_sd_kwarg_names(self) -> None:
        audio = SoundDeviceAudioInput(stream_factory=StrictInputStream)
        audio.open()
        kw = audio._stream.kwargs
        assert kw["samplerate"] == 16000
        assert kw["blocksize"] == 1280
        assert kw["channels"] == 1
        assert kw["dtype"] == "int16"
        assert kw["device"] is None
        assert callable(kw["callback"])
        assert audio._stream.started
        assert audio.is_open()
        audio.close()

    def test_open_no_args_keeps_constructor_format(self) -> None:
        audio = SoundDeviceAudioInput(
            sample_rate=44100, channels=2, block_size=640, stream_factory=StrictInputStream
        )
        audio.open()
        kw = audio._stream.kwargs
        assert kw["samplerate"] == 44100
        assert kw["blocksize"] == 640
        assert kw["channels"] == 2
        audio.close()

    def test_open_accepts_protocol_params(self) -> None:
        audio = SoundDeviceAudioInput(stream_factory=StrictInputStream)
        audio.open(sample_rate=22050, channels=2)
        kw = audio._stream.kwargs
        assert kw["samplerate"] == 22050
        assert kw["channels"] == 2
        audio.close()

    def test_open_is_idempotent(self) -> None:
        audio = SoundDeviceAudioInput(stream_factory=StrictInputStream)
        audio.open()
        stream = audio._stream
        audio.open()  # no-op
        assert audio._stream is stream
        audio.close()

    def test_open_resolves_device_and_passes_int_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_sd = FakeSD([_device(0, "Realtek USB")], default_input=0)
        monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
        audio = SoundDeviceAudioInput(device="realtek", stream_factory=StrictInputStream)
        audio.open()
        assert audio._stream.kwargs["device"] == 0
        audio.close()

    def test_open_failure_raises_sanitized_error(self) -> None:
        def boom(**kwargs: Any) -> Any:
            raise RuntimeError("raw-driver secret detail")

        audio = SoundDeviceAudioInput(stream_factory=boom)
        with pytest.raises(VoiceInputError) as exc:
            audio.open()
        msg = str(exc.value)
        assert "could not open the microphone" in msg
        assert "secret" not in msg
        assert "driver" not in msg
        assert not audio.is_open()

    def test_open_clears_stale_audio(self) -> None:
        audio = SoundDeviceAudioInput(stream_factory=StrictInputStream, read_timeout_s=_READ_OK)
        audio.open()
        audio._stream.callback(_int16_block([1] * _BLOCK), _BLOCK, None, None)
        audio.close()
        audio.open()
        assert audio.read(1).samples == []  # stale block gone


class TestRead:
    def test_callback_to_read_scales_int16(self) -> None:
        audio = SoundDeviceAudioInput(stream_factory=StrictInputStream)
        audio.open()
        values = [0, 16384, 32767, -32768, -16384]
        audio._stream.callback(_int16_block(values), len(values), None, None)
        seg = audio.read(len(values))
        assert seg.sample_rate == 16000
        assert seg.samples == pytest.approx(
            [0.0, 0.5, 32767 / 32768.0, -1.0, -0.5]
        )
        audio.close()

    def test_read_returns_exact_n_and_keeps_surplus(self) -> None:
        audio = SoundDeviceAudioInput(
            stream_factory=StrictInputStream, queue_maxsize=10, read_timeout_s=5.0
        )
        audio.open()
        cb = audio._stream.callback
        for i in range(10):
            cb(_int16_block([i] * 40), 40, None, None)  # 400 samples total
        seg1 = audio.read(260)
        seg2 = audio.read(100)
        seg3 = audio.read(40)  # the surplus tail
        assert [len(s.samples) for s in (seg1, seg2, seg3)] == [260, 100, 40]
        assert seg1.samples[0] == 0.0
        assert seg3.samples[-1] == pytest.approx(9 / 32768.0)
        assert audio._read_buffer == []  # surplus fully drained
        audio.close()

    def test_read_outputs_different_sizes_from_same_block(self) -> None:
        audio = SoundDeviceAudioInput(stream_factory=StrictInputStream, read_timeout_s=5.0)
        audio.open()
        audio._stream.callback(_int16_block([7] * 10), 10, None, None)
        assert audio.read(3).samples == [7 / 32768.0] * 3
        assert audio.read(7).samples == [7 / 32768.0] * 7  # uses the surplus
        audio.close()

    def test_read_stalls_and_returns_partial(self) -> None:
        audio = SoundDeviceAudioInput(stream_factory=StrictInputStream, read_timeout_s=_READ_OK)
        audio.open()
        audio._stream.callback(_int16_block([1] * 5), 5, None, None)
        start = time.monotonic()
        seg = audio.read(100)  # only 5 available → stalls → partial
        elapsed = time.monotonic() - start
        assert seg.samples == [1 / 32768.0] * 5
        assert elapsed < 1.0
        audio.close()


class TestQueueDiscipline:
    def test_drop_oldest_when_queue_full(self) -> None:
        audio = SoundDeviceAudioInput(
            stream_factory=StrictInputStream, queue_maxsize=2, read_timeout_s=5.0
        )
        audio.open()
        cb = audio._stream.callback
        cb(_int16_block([1] * _BLOCK), _BLOCK, None, None)  # A
        cb(_int16_block([2] * _BLOCK), _BLOCK, None, None)  # B
        assert audio.dropped_oldest == 0
        cb(_int16_block([3] * _BLOCK), _BLOCK, None, None)  # C drops A
        assert audio.dropped_oldest == 1
        cb(_int16_block([4] * _BLOCK), _BLOCK, None, None)  # D drops B
        assert audio.dropped_oldest == 2
        # Only the newest two blocks survive.
        assert audio.read(_BLOCK).samples == pytest.approx([3 / 32768.0] * _BLOCK)
        assert audio.read(_BLOCK).samples == pytest.approx([4 / 32768.0] * _BLOCK)
        audio.close()

    def test_callback_cannot_crash_on_broken_indata(self) -> None:
        audio = SoundDeviceAudioInput(stream_factory=StrictInputStream)
        audio.open()

        class Broken:
            def __bytes__(self) -> bytes:
                raise RuntimeError("broken buffer")

        audio._callback(Broken(), 0, None, None)
        assert audio.dropped_oldest == 1  # dropped and counted, callback survived

    def test_status_warning_rate_limited(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        caplog.set_level(logging.WARNING, logger="jarvis.voice.audio_input")
        audio = SoundDeviceAudioInput(stream_factory=StrictInputStream)
        audio.open()
        status = object()
        audio._last_status_log = time.monotonic()  # warning was just emitted
        audio._callback(_int16_block([0] * 4), 4, None, status)
        audio._callback(_int16_block([0] * 4), 4, None, status)
        assert "microphone stream status" not in caplog.text
        caplog.clear()
        audio._last_status_log = 0.0  # long ago → allowed again
        audio._callback(_int16_block([0] * 4), 4, None, status)
        assert "microphone stream status" in caplog.text
        audio.close()


class TestFlushClose:
    def test_flush_clears_queue_surplus_and_drop_counter(self) -> None:
        audio = SoundDeviceAudioInput(
            stream_factory=StrictInputStream, queue_maxsize=2, read_timeout_s=_READ_OK
        )
        audio.open()
        cb = audio._stream.callback
        cb(_int16_block([1] * _BLOCK), _BLOCK, None, None)
        cb(_int16_block([2] * _BLOCK), _BLOCK, None, None)
        cb(_int16_block([3] * _BLOCK), _BLOCK, None, None)  # drops block 1
        audio.flush()
        assert audio.dropped_oldest == 0
        assert audio.read(1).samples == []
        cb(_int16_block([5] * _BLOCK), _BLOCK, None, None)
        assert audio.read(2).samples == pytest.approx([5 / 32768.0] * 2)
        audio.close()

    def test_close_is_idempotent(self) -> None:
        audio = SoundDeviceAudioInput(stream_factory=StrictInputStream)
        audio.close()  # never opened → no-op
        audio.open()
        stream = audio._stream
        audio.close()
        assert not audio.is_open()
        assert stream.closed
        assert audio._stream is None
        audio.close()  # already closed → no-op
        assert not audio.is_open()

    def test_close_unblocks_a_blocked_read_in_another_thread(self) -> None:
        audio = SoundDeviceAudioInput(stream_factory=StrictInputStream, read_timeout_s=10.0)
        audio.open()
        result: dict[str, Any] = {}

        def _reader() -> None:
            result["seg"] = audio.read(_BLOCK)

        thread = threading.Thread(target=_reader)
        start = time.monotonic()
        thread.start()
        time.sleep(0.1)
        audio.close()
        thread.join(timeout=1.0)
        elapsed = time.monotonic() - start
        assert not thread.is_alive()
        assert elapsed < 0.3  # spec: released within ~0.3 s of close()
        assert len(result["seg"].samples) < _BLOCK  # partial, not 10 s of waiting


class TestDeviceResolution:
    def test_int_out_of_range(self) -> None:
        sd = FakeSD([_device(0, "A"), _device(1, "B")])
        with pytest.raises(VoiceInputError, match="out of range"):
            SoundDeviceAudioInput._resolve_device(sd, 5)

    def test_int_not_a_microphone(self) -> None:
        sd = FakeSD([_device(0, "Speakers", ins=0)])
        with pytest.raises(VoiceInputError, match="not a microphone"):
            SoundDeviceAudioInput._resolve_device(sd, 0)

    def test_name_matches_nothing(self) -> None:
        sd = FakeSD([_device(0, "Realtek Audio")])
        with pytest.raises(VoiceInputError, match="matched no microphone"):
            SoundDeviceAudioInput._resolve_device(sd, "nonexistent")

    def test_name_prefers_input_capable_default_hostapi(self) -> None:
        devices = [
            _device(0, "Mike", hostapi=0, ins=2),
            _device(1, "Mike", hostapi=0, ins=0),  # not input-capable
            _device(2, "Mike", hostapi=1, ins=2),
        ]
        sd = FakeSD(devices, default_input=2)  # default input lives on host API 1
        assert SoundDeviceAudioInput._resolve_device(sd, "mike") == 2

    def test_name_falls_back_to_first_input_capable_match(self) -> None:
        devices = [_device(0, "Mike", hostapi=0, ins=0), _device(1, "Mike", hostapi=2, ins=2)]
        sd = FakeSD(devices, default_input=0)  # default host API 0 has no input match
        assert SoundDeviceAudioInput._resolve_device(sd, "mike") == 1


class TestStreamKwargsHelper:
    def test_uses_the_real_sd_parameter_names(self) -> None:
        kwargs = _stream_kwargs(
            sample_rate=16000, channels=1, block_size=1280, device=None, callback=lambda *a: None
        )
        assert set(kwargs) <= _REAL_STREAM_PARAMS
        assert "samplerate" in kwargs
        assert "blocksize" in kwargs

    @pytest.mark.voice
    def test_kwargs_exist_in_real_sounddevice_signature(self) -> None:
        import inspect

        sd = pytest.importorskip("sounddevice")
        params = set(inspect.signature(sd.InputStream.__init__).parameters)
        kwargs = _stream_kwargs(
            sample_rate=16000, channels=1, block_size=1280, device=None, callback=lambda *a: None
        )
        assert not (set(kwargs) - params), f"unknown kwargs for sd.InputStream: {set(kwargs) - params}"