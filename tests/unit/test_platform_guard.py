"""Platform guard helpers and lazy imports."""

from __future__ import annotations

import sys

import pytest

from jarvis import platform_guard


def test_is_windows_matches_runtime() -> None:
    assert platform_guard.is_windows() == (sys.platform == "win32")


@pytest.mark.parametrize(
    "version,expected",
    [((3, 11, 0), True), ((3, 12, 9), True), ((3, 10, 0), False), ((3, 13, 0), False)],
)
def test_python_supported(
    monkeypatch: pytest.MonkeyPatch, version: tuple[int, int, int], expected: bool
) -> None:
    monkeypatch.setattr(platform_guard.sys, "version_info", version, raising=False)
    assert platform_guard.is_python_supported() is expected


def test_is_64bit_on_current_interpreters() -> None:
    assert platform_guard.is_64bit() is (sys.maxsize > 2**32)


def test_require_windows_raises_off_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform_guard.sys, "platform", "darwin")
    with pytest.raises(platform_guard.PlatformError):
        platform_guard.require_windows("Dangerous feature")


def test_windows_only_decorator(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    @platform_guard.windows_only
    def touch() -> str:
        calls.append("ran")
        return "ok"

    monkeypatch.setattr(platform_guard.sys, "platform", "linux")
    with pytest.raises(platform_guard.PlatformError):
        touch()
    assert calls == []

    monkeypatch.setattr(platform_guard.sys, "platform", "win32")
    assert touch() == "ok"
    assert calls == ["ran"]


def test_lazy_import_loads_on_demand() -> None:
    lazy = platform_guard.LazyImport("json")
    assert lazy.dumps({"x": 1}) == '{"x": 1}'
    assert lazy._module is not None


def test_lazy_import_same_object_after_load() -> None:
    a = platform_guard.LazyImport("json")
    b = platform_guard.LazyImport("json")
    assert a._load() is b._load()
