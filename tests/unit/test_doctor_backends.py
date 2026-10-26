"""Unit tests for the optional-backend rows in ``jarvis doctor``."""

from __future__ import annotations

import pytest

from jarvis import doctor


def test_all_backends_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "_has", lambda module: True)
    monkeypatch.setattr(doctor, "backend_name", lambda: "rapidocr_onnxruntime")

    checks = {check.name: check for check in doctor.optional_backend_checks()}

    assert set(checks) == {"screenshot", "ocr", "volume", "playwright"}
    assert all(check.status == "PASS" for check in checks.values())
    assert checks["ocr"].detail == "ready (rapidocr_onnxruntime)"
    assert "chromium" in checks["playwright"].detail


def test_missing_backends_warn_but_never_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "_has", lambda module: False)
    monkeypatch.setattr(doctor, "backend_name", lambda: None)

    checks = doctor.optional_backend_checks()

    assert [check.name for check in checks] == ["screenshot", "ocr", "volume", "playwright"]
    assert all(check.status == "WARN" for check in checks)
    by_name = {check.name: check.detail for check in checks}
    assert "mss" in by_name["screenshot"]
    assert "PIL" in by_name["screenshot"]
    assert "read_screen" in by_name["ocr"]
    assert "pycaw" in by_name["volume"]
    assert ".[browser]" in by_name["playwright"]


def test_probe_failure_is_reported_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(module: str) -> bool:
        raise ValueError("bad module name")

    monkeypatch.setattr(doctor.importlib.util, "find_spec", broken)

    assert doctor._has("anything") is False
