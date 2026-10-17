"""open_path tool tests (Stage 8): Tier-0 viewer for files inside allowed roots.

No viewer is ever launched for real - the launcher is injected with a recorder.
The refusal behaviour that makes this safe-tier (denied executable/script suffix,
non-allowlisted suffix, relative/ADS input, outside roots, protected, missing)
is what these tests pin.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from jarvis.config import PolicySettings, Settings
from jarvis.policy import tiers
from jarvis.policy.rules import matches_blocked
from jarvis.tools.base import ToolContext, ToolResult
from jarvis.tools.open_in_app import (
    OPENABLE_SUFFIXES,
    OpenPathArgs,
    make_open_path_spec,
)
from jarvis.tools.registry import _FORBIDDEN_RE, build_default_registry

#: Every executable/script suffix the roadmap calls out (plus the rest of the
#: tested DENIED_FILE_SUFFIXES set the resolver refuses).
REFUSED_SUFFIXES = (
    ".exe",
    ".bat",
    ".cmd",
    ".ps1",
    ".lnk",
    ".msi",
    ".vbs",
    ".js",
    ".jar",
    ".scr",
    ".com",
    ".reg",
    ".dll",
    ".cpl",
)

#: Allowlisted suffixes exercised end-to-end.
ALLOWED_SUFFIXES = (".jpg", ".png", ".txt", ".md", ".pdf")


class RecordingLauncher:
    """Injectable ``ViewerFunc`` that records calls instead of launching."""

    def __init__(self, fail: bool = False) -> None:
        self.calls: list[str] = []
        self.fail = fail

    def __call__(self, path: str) -> None:
        self.calls.append(path)
        if self.fail:
            raise OSError("boom: no default viewer")


def _ctx(tmp_path: Path, *, dry_run: bool = False) -> ToolContext:
    settings = Settings(policy=PolicySettings(allowed_roots=[str(tmp_path)]))
    return ToolContext(settings=settings, dry_run=dry_run)


def _touch(root: Path, name: str, content: bytes = b"hi") -> Path:
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return target


def _run(path: str, ctx: ToolContext, launcher: RecordingLauncher | None = None) -> ToolResult:
    spec = make_open_path_spec(launcher=launcher or RecordingLauncher())
    return spec.run_verified(OpenPathArgs(path=path), ctx)


# ── registration / contract ─────────────────────────────────────────────


def test_registered_tier_zero_windows_only_and_name_safe() -> None:
    registry = build_default_registry()
    spec = registry.get("open_path")
    assert spec.base_tier == tiers.TIER_SAFE
    assert spec.windows_only is True
    assert spec.path_args == ("path",)
    assert not _FORBIDDEN_RE.search(spec.name)
    assert not _FORBIDDEN_RE.search(spec.description)
    assert not matches_blocked(spec, OpenPathArgs(path="x"))


# ── allowed extensions open ─────────────────────────────────────────────


@pytest.mark.parametrize("suffix", ALLOWED_SUFFIXES)
def test_allowed_extensions_open_in_default_viewer(tmp_path: Path, suffix: str) -> None:
    launcher = RecordingLauncher()
    target = _touch(tmp_path, f"photo{suffix}")
    result = _run(str(target), _ctx(tmp_path), launcher)
    assert result.ok
    assert launcher.calls == [str(target)]


def test_verified_is_none_with_a_note(tmp_path: Path) -> None:
    target = _touch(tmp_path, "notes.txt")
    result = _run(str(target), _ctx(tmp_path))
    assert result.ok
    assert result.verified is None
    assert "cannot be verified" in result.verify_note
    assert result.data["file"] == str(target)


def test_allowlist_is_document_and_image_only() -> None:
    assert OPENABLE_SUFFIXES >= {".jpg", ".png", ".txt", ".md", ".pdf"}
    assert not (OPENABLE_SUFFIXES & {".exe", ".bat", ".ps1", ".lnk", ".js"})


# ── refusals ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("suffix", REFUSED_SUFFIXES)
def test_executable_and_script_suffixes_refused(tmp_path: Path, suffix: str) -> None:
    launcher = RecordingLauncher()
    target = _touch(tmp_path, f"evil{suffix}", content=b"MZ")
    result = _run(str(target), _ctx(tmp_path), launcher)
    assert not result.ok
    assert launcher.calls == []


def test_non_allowlisted_suffix_refused(tmp_path: Path) -> None:
    launcher = RecordingLauncher()
    target = _touch(tmp_path, "payload.zip")
    result = _run(str(target), _ctx(tmp_path), launcher)
    assert not result.ok
    assert "default viewer only" in (result.error or "")
    assert launcher.calls == []


def test_relative_path_refused(tmp_path: Path) -> None:
    launcher = RecordingLauncher()
    result = _run("notes.txt", _ctx(tmp_path), launcher)
    assert not result.ok
    assert "absolute" in (result.error or "")
    assert launcher.calls == []


def test_alternate_data_stream_refused(tmp_path: Path) -> None:
    launcher = RecordingLauncher()
    target = _touch(tmp_path, "notes.txt")
    result = _run(f"{target}:evil.exe", _ctx(tmp_path), launcher)
    assert not result.ok
    assert launcher.calls == []


def test_missing_file_refused(tmp_path: Path) -> None:
    launcher = RecordingLauncher()
    missing = tmp_path / "nope.txt"
    result = _run(str(missing), _ctx(tmp_path), launcher)
    assert not result.ok
    assert "does not exist" in (result.error or "")
    assert launcher.calls == []


def test_outside_roots_refused(tmp_path: Path) -> None:
    launcher = RecordingLauncher()
    outside = tmp_path.parent / "elsewhere.jpg"
    outside.write_bytes(b"x")
    try:
        settings = Settings(policy=PolicySettings(allowed_roots=[str(tmp_path / "allowed")]))
        ctx = ToolContext(settings=settings)
        spec = make_open_path_spec(launcher=launcher)
        result = spec.run_verified(OpenPathArgs(path=str(outside)), ctx)
        assert not result.ok
        assert "outside the allowed folders" in (result.error or "")
        assert launcher.calls == []
    finally:
        if outside.exists():
            outside.unlink()


def test_protected_path_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import jarvis.tools.open_in_app as module

    monkeypatch.setattr(module.paths, "is_protected", lambda path: True)
    launcher = RecordingLauncher()
    target = _touch(tmp_path, "photo.jpg")
    result = _run(str(target), _ctx(tmp_path), launcher)
    assert not result.ok
    assert "protected" in (result.error or "")
    assert launcher.calls == []


# ── dry run / launcher failure ──────────────────────────────────────────


def test_dry_run_launches_nothing(tmp_path: Path) -> None:
    launcher = RecordingLauncher()
    target = _touch(tmp_path, "photo.jpg")
    result = _run(str(target), _ctx(tmp_path, dry_run=True), launcher)
    assert result.ok
    assert launcher.calls == []


def test_launcher_failure_returns_ok_false(tmp_path: Path) -> None:
    launcher = RecordingLauncher(fail=True)
    target = _touch(tmp_path, "photo.jpg")
    result = _run(str(target), _ctx(tmp_path), launcher)
    assert not result.ok
    assert "failed to open" in (result.error or "")
    assert "boom" in (result.error or "")


def test_describe_resolves_the_path(tmp_path: Path) -> None:
    target = _touch(tmp_path, "photo.jpg")
    spec = make_open_path_spec(launcher=RecordingLauncher())
    described: Any = spec.describe(OpenPathArgs(path=str(target)))
    assert "default viewer" in described
    assert "photo.jpg" in described
