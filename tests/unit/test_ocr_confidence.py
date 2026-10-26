"""OCR confidence passthrough tests with a fake local engine."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from jarvis.config import PerceptionSettings, Settings
from jarvis.perception import ocr
from jarvis.perception.ocr import OcrResult
from jarvis.perception.screen import ForegroundWindow
from jarvis.tools import screen as tools_screen
from jarvis.tools.base import ToolContext


class FakeEngine:
    """Callable fake RapidOCR engine returning a canned result."""

    def __init__(self, result: object) -> None:
        self.result = result
        self.calls = 0

    def __call__(self, _image: bytes) -> object:
        self.calls += 1
        return self.result


def _boxes() -> list[list[object]]:
    return [
        [[[0, 0], [10, 10]], "Hello", 0.91],
        [[[0, 20], [10, 30]], "World", 0.49],
    ]


def _use_engine(monkeypatch: pytest.MonkeyPatch, result: object) -> FakeEngine:
    engine = FakeEngine(result)
    monkeypatch.setattr(ocr, "_get_engine", lambda: engine)
    return engine


def test_block_and_mean_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_engine(monkeypatch, (_boxes(), 0.02))
    result = ocr.ocr_image_with_confidence(b"png")
    assert result.text == "Hello\nWorld"
    assert [block.confidence for block in result.blocks] == [0.91, 0.49]
    assert result.mean_confidence == 0.70


def test_ocr_image_stays_backward_compatible(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_engine(monkeypatch, _boxes())
    assert ocr.ocr_image(b"png") == "Hello\nWorld"


def test_missing_score_is_zero_and_scores_are_clamped(monkeypatch: pytest.MonkeyPatch) -> None:
    boxes = [
        [[[0, 0], [1, 1]], "no-score"],
        [[[0, 0], [1, 1]], "too-high", 1.4],
        [[[0, 0], [1, 1]], "negative", -0.2],
    ]
    _use_engine(monkeypatch, boxes)
    result = ocr.ocr_image_with_confidence(b"png")
    assert [block.confidence for block in result.blocks] == [0.0, 1.0, 0.0]
    assert result.mean_confidence == 0.33


def test_no_blocks_has_no_mean(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_engine(monkeypatch, ([], 0.01))
    result = ocr.ocr_image_with_confidence(b"png")
    assert result == OcrResult(text="", blocks=(), mean_confidence=None)


def _extract_with_confidence(
    monkeypatch: pytest.MonkeyPatch, *, mean: float, threshold: float
) -> tools_screen._Extracted:
    monkeypatch.setattr(tools_screen.ocr, "ocr_available", lambda: True)
    monkeypatch.setattr(
        tools_screen.ocr,
        "ocr_image_with_confidence",
        lambda _image: OcrResult(
            text="hello", blocks=(ocr.OcrBlock("hello", mean),), mean_confidence=mean
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
