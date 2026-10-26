"""G4: file_search, process_list and volume_control with fakes and tmp_path."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from jarvis.config import FilesSettings, PolicySettings, Settings
from jarvis.tools import files, system
from jarvis.tools.base import ToolContext


def _settings(root: Path, *, profile_wide: bool = False) -> Settings:
    return Settings(
        policy=PolicySettings(allowed_roots=[str(root)]),
        files=FilesSettings(profile_wide=profile_wide),
    )


def test_file_search_is_limited_to_the_requested_number(tmp_path: Path) -> None:
    root = tmp_path / "files"
    root.mkdir()
    for index in range(5):
        (root / f"note-{index}.txt").write_text("x", encoding="utf-8")
    result = files._run_file_search(
        files.FileSearchArgs(query="note", max_results=3),
        ToolContext(settings=_settings(root)),
    )
    assert result.ok is True
    assert result.data["count"] == 3
    assert len(result.data["paths"]) == 3


def test_file_search_filters_by_extension_without_a_dot(tmp_path: Path) -> None:
    root = tmp_path / "files"
    root.mkdir()
    (root / "resume.txt").write_text("x", encoding="utf-8")
    (root / "resume.MD").write_text("x", encoding="utf-8")
    result = files._run_file_search(
        files.FileSearchArgs(query="resume", extension="MD"),
        ToolContext(settings=_settings(root)),
    )
    assert result.data["paths"] == [str((root / "resume.MD").resolve())]


def test_file_search_skips_profile_deny_listed_folders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    (home / "Documents").mkdir(parents=True)
    (home / ".git").mkdir()
    (home / ".aws").mkdir()
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("HOME", str(home))
    (home / "Documents" / "a.txt").write_text("x", encoding="utf-8")
    (home / ".git" / "a.txt").write_text("x", encoding="utf-8")
    (home / ".aws" / "a.txt").write_text("x", encoding="utf-8")
    result = files._run_file_search(
        files.FileSearchArgs(query="txt", extension="txt"),
        ToolContext(settings=_settings(home, profile_wide=True)),
    )
    assert result.data["paths"] == [str((home / "Documents" / "a.txt").resolve())]


def test_file_search_never_returns_files_outside_the_roots(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("x", encoding="utf-8")
    result = files._run_file_search(
        files.FileSearchArgs(query="secret"),
        ToolContext(settings=_settings(root)),
    )
    assert result.data["paths"] == []


class _Proc:
    """Minimal psutil process record."""

    def __init__(self, pid: int, name: str) -> None:
        self.info = {"pid": pid, "name": name}


def test_process_list_returns_names_and_pids_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import psutil

    monkeypatch.setattr(
        psutil,
        "process_iter",
        lambda attrs: [_Proc(10, "alpha.exe"), _Proc(20, "beta.exe")],
    )
    spec = system.make_process_list_spec()
    result = spec.run_verified(system.ProcessListArgs(), ToolContext(settings=Settings()))
    assert result.ok is True
    assert result.data["processes"] == [
        {"name": "alpha.exe", "pid": 10},
        {"name": "beta.exe", "pid": 20},
    ]
    assert "alpha.exe (10)" in result.output


def test_process_list_caps_at_forty(monkeypatch: pytest.MonkeyPatch) -> None:
    import psutil

    monkeypatch.setattr(
        psutil,
        "process_iter",
        lambda attrs: [_Proc(index, f"p{index}.exe") for index in range(60)],
    )
    result = system._run_process_list(
        system.ProcessListArgs(), ToolContext(settings=Settings())
    )
    assert result.ok is True
    assert len(result.data["processes"]) == 40


def test_process_list_reports_a_missing_psutil(monkeypatch: pytest.MonkeyPatch) -> None:
    def _missing() -> Any:
        raise system._BackendUnavailable("psutil is missing")

    monkeypatch.setattr(system, "_load_psutil", _missing)
    result = system._run_process_list(
        system.ProcessListArgs(), ToolContext(settings=Settings())
    )
    assert result.ok is False
    assert "psutil" in (result.error or "")


class _FakeVolume:
    """Stand-in for a pycaw IAudioEndpointVolume object."""

    def __init__(self, level: float = 0.42, muted: bool = False) -> None:
        self.level = level
        self.muted = muted

    def GetMasterVolumeLevelScalar(self) -> float:
        return self.level

    def SetMasterVolumeLevelScalar(self, value: float, _ctx: object) -> None:
        self.level = value

    def GetMute(self) -> int:
        return int(self.muted)

    def SetMute(self, value: int, _ctx: object) -> None:
        self.muted = bool(value)


def _volume_ctx() -> ToolContext:
    return ToolContext(settings=Settings())


def test_volume_get_reports_the_current_state(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeVolume()
    monkeypatch.setattr(system, "_load_volume_endpoint", lambda: fake)
    result = system._run_volume_control(
        system.VolumeControlArgs(action="get"), _volume_ctx()
    )
    assert result.ok is True
    assert result.data == {"level": 42, "muted": False}
    assert "42%" in result.output


def test_volume_set_applies_the_level(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeVolume()
    monkeypatch.setattr(system, "_load_volume_endpoint", lambda: fake)
    result = system._run_volume_control(
        system.VolumeControlArgs(action="set", level=30), _volume_ctx()
    )
    assert result.ok is True
    assert fake.level == 0.30
    assert result.data["level"] == 30


@pytest.mark.parametrize(("action", "muted"), [("mute", True), ("unmute", False)])
def test_volume_mute_actions(
    monkeypatch: pytest.MonkeyPatch, action: str, muted: bool
) -> None:
    fake = _FakeVolume(muted=not muted)
    monkeypatch.setattr(system, "_load_volume_endpoint", lambda: fake)
    result = system._run_volume_control(
        system.VolumeControlArgs(action=action), _volume_ctx()
    )
    assert result.ok is True
    assert fake.muted is muted
    assert result.data["muted"] is muted


def test_volume_set_requires_a_level() -> None:
    with pytest.raises(ValidationError):
        system.VolumeControlArgs(action="set")


def test_volume_reports_a_missing_pycaw(monkeypatch: pytest.MonkeyPatch) -> None:
    def _missing() -> Any:
        raise system._BackendUnavailable("pycaw is missing")

    monkeypatch.setattr(system, "_load_volume_endpoint", _missing)
    result = system._run_volume_control(
        system.VolumeControlArgs(action="get"), _volume_ctx()
    )
    assert result.ok is False
    assert "pycaw" in (result.error or "")


def test_new_tools_are_each_in_exactly_one_group() -> None:
    from jarvis.agent import tool_groups

    assignment = tool_groups.tool_group_map()
    assert assignment["file_search"] == "files"
    assert assignment["process_list"] == "system"
    assert assignment["volume_control"] == "system"
