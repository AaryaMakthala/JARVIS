"""Tests for :mod:`jarvis.tools.default_dirs` and ``PathSettings`` (Step D1a)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from jarvis import config
from jarvis.tools import default_dirs
from jarvis.tools.default_dirs import (
    default_save_dir,
    known_folder,
    location_label,
    resolve_location,
)


def _settings(
    default: Path | None = None, aliases: dict[str, str] | None = None
) -> config.Settings:
    return config.Settings(
        paths=config.PathSettings(
            default_save_dir=str(default) if default is not None else "",
            aliases=aliases or {},
        )
    )


def test_default_save_dir_uses_known_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    onedrive_desktop = tmp_path / "OneDrive" / "Desktop"
    monkeypatch.setattr(
        default_dirs, "known_folder", lambda name: onedrive_desktop if name == "desktop" else None
    )

    assert default_save_dir(_settings()) == onedrive_desktop


def test_default_save_dir_falls_back_to_userprofile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(default_dirs, "is_windows", lambda: True)
    monkeypatch.setattr(default_dirs, "_shget_known_folder", lambda folder_id: None)
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    assert known_folder("desktop") == tmp_path / "Desktop"


def test_bare_filename_goes_to_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(default_dirs, "known_folder", lambda name: None)

    result = resolve_location("notes.txt", _settings(default=tmp_path))

    assert result.kind == "resolved"
    assert result.path == tmp_path / "notes.txt"


def test_alias_resolves_to_alias_root(tmp_path: Path) -> None:
    root = tmp_path / "Jarvis"

    result = resolve_location(
        "jarvis/notes.txt", _settings(default=tmp_path, aliases={"jarvis": str(root)})
    )

    assert result.kind == "resolved"
    assert result.path == root / "notes.txt"


def test_alias_wins_over_real_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    alias_root = tmp_path / "KnownDesktop"
    (tmp_path / "desktop").mkdir()  # a real folder of the same name under the default
    monkeypatch.setattr(
        default_dirs, "known_folder", lambda name: alias_root if name == "desktop" else None
    )

    result = resolve_location("desktop/notes.txt", _settings(default=tmp_path))

    assert result.kind == "resolved"
    assert result.path == alias_root / "notes.txt"


def test_absolute_path_unchanged(tmp_path: Path) -> None:
    target = tmp_path / "x.txt"

    result = resolve_location(str(target), _settings(default=tmp_path))

    assert result.kind == "resolved"
    assert result.path == target


def test_existing_subfolder_under_default_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(default_dirs, "known_folder", lambda name: None)
    (tmp_path / "projects").mkdir()

    result = resolve_location("projects/a.txt", _settings(default=tmp_path))

    assert result.kind == "resolved"
    assert result.path == tmp_path / "projects" / "a.txt"


def test_unknown_folder_asks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(default_dirs, "known_folder", lambda name: None)

    result = resolve_location("Reports/notes.txt", _settings(default=tmp_path))

    assert result.kind == "clarify"
    assert result.path is None
    assert result.reason == "which folder?"


def test_non_windows_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(default_dirs, "is_windows", lambda: False)

    assert known_folder("documents") == Path.home() / "Documents"


def test_alias_must_be_absolute() -> None:
    with pytest.raises(ValidationError):
        config.PathSettings(aliases={"x": "relative/path"})


def test_location_label_desktop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    desktop = tmp_path / "OneDrive" / "Desktop"
    monkeypatch.setattr(
        default_dirs, "known_folder", lambda name: desktop if name == "desktop" else None
    )

    assert location_label(desktop / "notes.txt", _settings()) == "your Desktop"
