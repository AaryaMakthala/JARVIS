"""Whole-profile roots and protected folders (G3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.config import FilesSettings, PolicySettings, Settings
from jarvis.policy import paths


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``~`` at a temporary home so no real profile is touched."""
    root = tmp_path / "home"
    root.mkdir()
    monkeypatch.setenv("USERPROFILE", str(root))
    monkeypatch.setenv("HOME", str(root))
    monkeypatch.setenv("HOMEDRIVE", root.drive)
    monkeypatch.setenv("HOMEPATH", str(root)[len(root.drive) :])
    return root


def _settings(*, profile_wide: bool, roots: list[str] | None = None) -> Settings:
    return Settings(
        files=FilesSettings(profile_wide=profile_wide),
        policy=PolicySettings(allowed_roots=roots or []),
    )


def test_profile_wide_off_keeps_configured_roots(home: Path) -> None:
    workspace = home / "workspace"
    settings = _settings(profile_wide=False, roots=[str(workspace)])
    roots = paths.roots_from_settings(settings)
    assert paths.within_any_root(workspace, roots)
    assert not paths.within_any_root(home / "Documents", roots)


def test_profile_wide_replaces_roots_with_the_home_folder(home: Path) -> None:
    settings = _settings(profile_wide=True, roots=["C:/somewhere/else"])
    roots = paths.roots_from_settings(settings)
    assert roots == [paths.resolve_safe(str(home))]
    assert paths.within_any_root(home / "Documents" / "notes.txt", roots)


def test_parent_escape_is_outside_the_profile_root(home: Path) -> None:
    roots = paths.roots_from_settings(_settings(profile_wide=True))
    escaped = paths.resolve_safe(str(home / ".." / "outside.txt"))
    assert not paths.within_any_root(escaped, roots)


def test_symlink_escape_is_refused(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    link = home / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")
    roots = paths.roots_from_settings(_settings(profile_wide=True))
    resolved = paths.resolve_safe(str(link / "secret.txt"))
    assert not paths.within_any_root(resolved, roots)


DENIED = [
    "AppData/Local/x.txt",
    "Windows/x.txt",
    "Program Files/x.txt",
    "Program Files (x86)/x.txt",
    "ProgramData/x.txt",
    ".ssh/id_ed25519",
    ".gnupg/private-keys-v1.d/x",
    ".aws/credentials",
    ".azure/x",
    ".kube/config",
    ".config/gh/hosts.yml",
    "Documents/credentials/x",
    "Documents/secrets/x",
    "Documents/Wallets/x",
    "Documents/Keychain/x",
    "OneDrive/Personal Vault/x",
    "project/.venv/Scripts/python.exe",
    "project/.git/config",
]


@pytest.mark.parametrize("relative", DENIED)
def test_profile_wide_denies_protected_components(home: Path, relative: str) -> None:
    settings = _settings(profile_wide=True)
    target = paths.resolve_safe(str(home / relative))
    assert paths.is_protected(
        target, exact=[], containers=[], patterns=(), settings=settings
    )


def test_the_same_component_is_not_denied_when_profile_wide_is_off(home: Path) -> None:
    settings = _settings(profile_wide=False)
    target = paths.resolve_safe(str(home / ".aws" / "credentials"))
    assert not paths.is_protected(
        target, exact=[], containers=[], patterns=(), settings=settings
    )
