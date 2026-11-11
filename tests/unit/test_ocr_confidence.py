"""OCR confidence and PaddleOCR adapter tests with fake local engines."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from jarvis.config import PerceptionSettings, Settings
from jarvis.perception import ocr
from jarvis.perception.ocr import OcrBlock, OcrResult
from jarvis.perception.screen import ForegroundWindow
from jarvis.tools import screen as tools_screen
from jarvis.tools.base import ToolContext


class FakeEngine:
    """Callable fake OCR engine returning a canned result."""

    def __init__(self, result: OcrResult) -> None:
        self.result = result
        self.calls = 0

    def __call__(self, _image: bytes) -> OcrResult:
        self.calls += 1
        return self.result


def _use_engine(monkeypatch: pytest.MonkeyPatch, result: OcrResult) -> FakeEngine:
    engine = FakeEngine(result)
    monkeypatch.setattr(ocr, "_get_engine", lambda *_args, **_kwargs: engine)
    return engine


def _predictions() -> list[dict[str, Any]]:
    return [
        {
            "rec_texts": ["Hello", "World"],
            "rec_scores": [0.91, 0.49],
            "rec_boxes": [[[0, 0], [10, 10]], [[0, 20], [10, 30]]],
        }
    ]


def test_parse_paddle_result_keeps_text_boxes_and_confidence() -> None:
    result = ocr._parse_predictions(_predictions())
    assert result.text == "Hello\nWorld"
    assert [block.confidence for block in result.blocks] == [0.91, 0.49]
    assert result.blocks[0].box == (0.0, 0.0, 10.0, 10.0)
    assert result.mean_confidence == 0.70


def test_ocr_image_stays_backward_compatible(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_engine(
        monkeypatch,
        OcrResult(
            text="Hello\nWorld",
            blocks=(OcrBlock("Hello", 0.91), OcrBlock("World", 0.49)),
            mean_confidence=0.70,
        ),
    )
    assert ocr.ocr_image(b"png") == "Hello\nWorld"


def test_missing_score_is_zero_and_scores_are_clamped() -> None:
    predictions = [
        {
            "rec_texts": ["no-score", "too-high", "negative"],
            "rec_scores": [None, 1.4, -0.2],
            "rec_boxes": [],
        }
    ]
    result = ocr._parse_predictions(predictions)
    assert [block.confidence for block in result.blocks] == [0.0, 1.0, 0.0]
    assert result.mean_confidence == 0.33


def test_no_blocks_has_no_mean(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_engine(monkeypatch, OcrResult(text="", blocks=(), mean_confidence=None))
    result = ocr.ocr_image_with_confidence(b"png")
    assert result == OcrResult(text="", blocks=(), mean_confidence=None)


def test_unavailable_reason_names_missing_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    monkeypatch.setattr(ocr, "_module_available", lambda _name: True)
    settings = PerceptionSettings(ocr_model_dir=str(tmp_path))

    reason = ocr.unavailable_reason(settings)

    assert reason is not None
    assert ocr.DETECTION_MODEL in reason
    assert ocr.RECOGNITION_MODEL in reason

    (tmp_path / ocr.DETECTION_MODEL).mkdir()
    (tmp_path / ocr.RECOGNITION_MODEL).mkdir()

    assert ocr.unavailable_reason(settings) is None


def test_unavailable_reason_names_missing_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ocr, "_module_available", lambda name: name != "paddle")

    assert ocr.unavailable_reason(PerceptionSettings()) == "PaddlePaddle is not installed"


def test_engine_selects_small_models_and_explicit_local_dirs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    captured: dict[str, Any] = {}

    class FakePaddleOCR:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(ocr, "_load_paddleocr_class", lambda: FakePaddleOCR)
    (tmp_path / ocr.DETECTION_MODEL).mkdir()
    (tmp_path / ocr.RECOGNITION_MODEL).mkdir()
    settings = PerceptionSettings(ocr_model_dir=str(tmp_path))

    engine = ocr._PaddleEngine(settings)
    engine.close()

    assert captured["text_detection_model_name"] == "PP-OCRv6_small_det"
    assert captured["text_recognition_model_name"] == "PP-OCRv6_small_rec"
    assert captured["text_detection_model_dir"] == str(tmp_path / ocr.DETECTION_MODEL)
    assert captured["text_recognition_model_dir"] == str(tmp_path / ocr.RECOGNITION_MODEL)
    assert captured["device"] == "cpu"
    assert captured["engine"] == "paddle"
    assert captured["use_doc_orientation_classify"] is False
    assert captured["use_doc_unwarping"] is False
    assert captured["use_textline_orientation"] is False
    assert captured["closed"] is True


def _extract_with_confidence(
    monkeypatch: pytest.MonkeyPatch, *, mean: float, threshold: float
) -> tools_screen._Extracted:
    monkeypatch.setattr(tools_screen.ocr, "unavailable_reason", lambda _settings: None)
    monkeypatch.setattr(
        tools_screen.ocr,
        "ocr_image_with_confidence",
        lambda _image, _settings: OcrResult(
            text="hello", blocks=(OcrBlock("hello", mean),), mean_confidence=mean
        ),
    )
    monkeypatch.setattr(tools_screen.screen, "capture_foreground", lambda hwnd=0: b"png")
    monkeypatch.setattr(
        tools_screen.ui_text,
        "foreground_uia_text",
        lambda **_kwargs: SimpleNamespace(text="", controls=0, secure_controls=0),
    )
    settings = Settings(perception=PerceptionSettings(ocr_low_conf=threshold))
    return tools_screen._extract(ToolContext(settings=settings))


def test_screen_output_and_data_keep_low_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted = _extract_with_confidence(monkeypatch, mean=0.55, threshold=0.6)
    assert extracted.ocr_low_confidence is True
    assert tools_screen._confidence_note(extracted) == "\nOCR confidence: 0.55 (low)"
    data = tools_screen._data(ForegroundWindow("Title", "app.exe", 1), extracted)
    assert data["ocr_mean_confidence"] == 0.55
    assert data["ocr_block_confidences"] == [0.55]
    assert data["ocr_low_confidence"] is True


def test_confidence_is_not_marked_low_at_the_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted = _extract_with_confidence(monkeypatch, mean=0.6, threshold=0.6)
    assert extracted.ocr_low_confidence is False
    assert tools_screen._confidence_note(extracted) == "\nOCR confidence: 0.60"


def test_read_screen_puts_the_confidence_line_in_the_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted = _extract_with_confidence(monkeypatch, mean=0.42, threshold=0.6)
    monkeypatch.setattr(tools_screen, "is_windows", lambda: True)
    monkeypatch.setattr(
        tools_screen.screen,
        "foreground_window",
        lambda: ForegroundWindow("Title", "app.exe", 1),
    )
    monkeypatch.setattr(tools_screen, "_extract", lambda *_args, **_kwargs: extracted)
    settings = Settings()
    result = tools_screen.make_read_screen_spec().run_verified(
        tools_screen.ReadScreenArgs(), ToolContext(settings=settings)
    )
    assert result.ok and result.tainted is True
    assert "OCR confidence: 0.42 (low)" in result.output
    assert result.data["ocr_mean_confidence"] == 0.42
