"""OpenWakeWordDetector (real backend wrapper) unit tests.

The critical regression covered here is the **sample-format contract**:
openwakeword 0.6.0's mel-spectrogram preprocessor requires 16-bit PCM
input (``AudioFeatures._get_melspectrogram`` casts list input to
``np.int16``), so a detector fed the normalized float32 samples that
:class:`AudioSegment` carries would have every sample truncated to
{0, ±1} and would silently hear silence — the real-machine bug where
``jarvis on`` said "voice activated" but "hey jarvis" never woke the loop.

A strict fake model records exactly what the wrapper passes to
``model.predict`` and confirms it is int16, scaled from the segment's
floats, clipped to the valid range.  No real model or audio hardware is
involved.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pytest

from jarvis.voice.interfaces import AudioSegment
from jarvis.voice.wake import OpenWakeWordDetector, create

_LOUD_SEGMENT = [0.9, -0.9, 0.5, -0.5, 0.0, 1.0, -1.0, 0.25]


class _RecordingModel:
    """Fake Oww model that records every array handed to ``predict``."""

    def __init__(self, result: dict[str, Any], model_name: str = "hey_jarvis") -> None:
        self._result = {**result}
        self._model_name = model_name
        self.calls: list[np.ndarray] = []
        self.reset_calls = 0

    def predict(self, x: np.ndarray) -> dict[str, Any]:
        self.calls.append(np.array(x, copy=True))
        return dict(self._result)

    def reset(self) -> None:
        self.reset_calls += 1


class TestDetectSampleFormat:
    def test_predict_receives_int16_scaled_from_float_segment(self) -> None:
        model = _RecordingModel({"hey_jarvis": 0.9})
        detector = OpenWakeWordDetector(model, model_name="hey_jarvis", threshold=0.5)
        seg = AudioSegment(samples=_LOUD_SEGMENT)
        result = detector.detect(seg)

        assert result.detected
        assert result.confidence == pytest.approx(0.9)
        assert result.model_name == "hey_jarvis"

        sent = model.calls[0]
        # The contract: int16 PCM, never normalized float32 (which openwakeword
        # would truncate to ~0 and then never detect).
        assert sent.dtype == np.int16
        assert sent.shape == (len(_LOUD_SEGMENT),)
        assert sent.tolist() == pytest.approx(
            [29490, -29490, 16384, -16384, 0, 32767, -32767, 8192]
        )

    def test_float_samples_out_of_range_are_clipped(self) -> None:
        model = _RecordingModel({"hey_jarvis": 0.1})
        detector = OpenWakeWordDetector(model, model_name="hey_jarvis")
        detector.detect(AudioSegment(samples=[1.5, -2.0, 3.0]))
        sent = model.calls[0]
        assert sent.tolist() == [32767, -32767, 32767]

    def test_zero_silence_maps_to_zero_int16(self) -> None:
        model = _RecordingModel({"hey_jarvis": 0.0})
        detector = OpenWakeWordDetector(model, model_name="hey_jarvis")
        detector.detect(AudioSegment(samples=[0.0, 0.0, -0.0]))
        assert model.calls[0].tolist() == [0, 0, 0]


class TestThreshold:
    def test_score_equal_to_threshold_detects(self) -> None:
        detector = OpenWakeWordDetector(_RecordingModel({"hey_jarvis": 0.5}), threshold=0.5)
        assert detector.detect(AudioSegment(samples=[1.0])).detected

    def test_score_below_threshold_does_not_detect(self) -> None:
        detector = OpenWakeWordDetector(_RecordingModel({"hey_jarvis": 0.499}), threshold=0.5)
        assert not detector.detect(AudioSegment(samples=[1.0])).detected

    def test_unknown_model_key_scores_zero(self) -> None:
        detector = OpenWakeWordDetector(
            _RecordingModel({"some_other_model": 0.9}), model_name="hey_jarvis"
        )
        result = detector.detect(AudioSegment(samples=[1.0]))
        assert not result.detected
        assert result.confidence == 0.0

    def test_silence_never_triggers(self) -> None:
        detector = OpenWakeWordDetector(_RecordingModel({"hey_jarvis": 0.0}), threshold=0.5)
        result = detector.detect(AudioSegment(samples=[0.0] * 1280))
        assert not result.detected
        assert result.confidence == 0.0
        assert detector.max_score == 0.0


class TestScoreMetadata:
    def test_last_max_and_frames_seen_track_predictions(self) -> None:
        detector = OpenWakeWordDetector(
            _RecordingModel({"hey_jarvis": 0.9}), model_name="hey_jarvis"
        )
        for samples, score in (([1.0] * 1280, 0.9), ([0.5] * 1280, 0.4)):
            model = detector._model  # type: ignore[attr-defined]
            model._result = {"hey_jarvis": score}
            detector.detect(AudioSegment(samples=samples))
        assert detector.last_score == 0.4
        assert detector.max_score == 0.9
        assert detector.frames_seen == 2560

    def test_sample_info_line_carries_audio_rms(self, caplog: object) -> None:
        """The rate-limited INFO sample proves BOTH signal level (audio_rms)
        and model score reach the log — the diagnostic that distinguishes
        'mic content is silent/quiet' from 'model is not scoring'."""
        import logging

        caplog.set_level(logging.INFO, logger="jarvis.voice.wake")  # type: ignore[attr-defined]
        detector = OpenWakeWordDetector(
            _RecordingModel({"hey_jarvis": 0.9}), model_name="hey_jarvis"
        )
        detector.detect(AudioSegment(samples=[0.5] * 1280))
        text = caplog.text  # type: ignore[attr-defined]
        assert "audio_rms=0.5000" in text

    def test_reset_clears_last_score_and_forwards_to_model(self) -> None:
        model = _RecordingModel({"hey_jarvis": 0.9})
        detector = OpenWakeWordDetector(model, model_name="hey_jarvis")
        detector.detect(AudioSegment(samples=[1.0]))
        assert detector.last_score == 0.9
        detector.reset()
        assert detector.last_score == 0.0
        assert model.reset_calls == 1

    def test_detector_usable_after_reset(self) -> None:
        """A reset (re-arm) must not leave the detector in a dead state."""
        model = _RecordingModel({"hey_jarvis": 0.9})
        detector = OpenWakeWordDetector(model, model_name="hey_jarvis")
        assert detector.detect(AudioSegment(samples=[1.0])).detected
        detector.reset()
        assert detector.detect(AudioSegment(samples=[1.0])).detected


class TestReset:
    def test_reset_forwards_to_model(self) -> None:
        model = _RecordingModel({"hey_jarvis": 0.0})
        detector = OpenWakeWordDetector(model, model_name="hey_jarvis")
        detector.reset()
        assert model.reset_calls == 1


class TestCreate:
    def test_create_returns_none_when_openwakeword_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "openwakeword.model", None)
        assert create() is None

    def test_import_error_reports_a_broken_transitive_dependency(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``import_error`` surfaces the real exception; ``create`` returning
        ``None`` alone cannot tell "not installed" from "backend broken"."""
        from jarvis.voice import wake as wake_module

        monkeypatch.setitem(sys.modules, "openwakeword", None)
        reason = wake_module.import_error()
        assert reason is not None
        assert "openwakeword" in reason

    def test_create_returns_none_when_model_load_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _Boom:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("onnx load failed")

        fake_module = type("fake", (), {"Model": _Boom})()
        monkeypatch.setitem(sys.modules, "openwakeword.model", fake_module)
        assert create() is None

    def test_create_builds_a_detector_and_calls_really_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[dict[str, Any]] = []

        class _Ok:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.args = args
                self.kwargs = kwargs
                calls.append(kwargs)
                self.detect_result = {"hey_jarvis": 0.7}

            def predict(self, x: np.ndarray) -> dict[str, Any]:
                return {"hey_jarvis": 0.7}

            def reset(self) -> None:
                pass

        fake_module = type("fake", (), {"Model": _Ok})()
        monkeypatch.setitem(sys.modules, "openwakeword.model", fake_module)
        detector = create(model_name="hey_jarvis", threshold=0.5)
        assert isinstance(detector, OpenWakeWordDetector)
        assert calls[0]["wakeword_models"] == ["hey_jarvis"]
        result = detector.detect(AudioSegment(samples=[0.5]))
        assert result.detected
        assert result.confidence == pytest.approx(0.7)


class TestThresholdCalibration:
    def test_default_threshold_matches_the_measured_score_range(self) -> None:
        """Calibrated from real-model measurements, not a guess.

        The library's 0.5 never fired (a real human "hey jarvis" through the dev
        PC's mic peaked at 0.4363) and 0.25 still missed a degraded far-field
        utterance.  The default must sit inside the measured empty band: above
        the non-wake near-miss ceiling (~0.087 for "hey service") and below the
        genuine wake peak.  0.44 / 0.30 / 0.20 are the close, far and notably
        degraded cases; 0.09 is the measured non-wake ceiling.
        """
        from jarvis import config
        from jarvis.voice import wake as wake_module

        # The configured default and the code default must not drift apart:
        # the daemon and doctor both read the setting.
        assert config.Settings().voice.wake_threshold == wake_module._DEFAULT_THRESHOLD
        assert 0.087 < wake_module._DEFAULT_THRESHOLD < 0.4363

        def detected(score: float) -> bool:
            detector = OpenWakeWordDetector(
                _RecordingModel({"hey_jarvis": score}), model_name="hey_jarvis"
            )
            return detector.detect(AudioSegment(samples=[0.0] * 1280)).detected

        assert detected(0.44)  # normal close-range speech
        assert detected(0.30)  # degraded / far-field speech
        assert detected(0.20)  # notably degraded far-field speech
        assert not detected(0.09)  # measured non-wake near-miss ceiling (0.0866)
        assert not detected(0.05)  # plainly non-wake audio
