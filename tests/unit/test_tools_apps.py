"""open_app tool tests (docs/04 §2.1: Tier 0, allowlisted launch, no shell).

The regression these cover: ``subprocess.Popen(["chrome.exe"])`` fails with
WinError 2 on a stock Windows install, because Chrome is not on ``PATH`` - it
registers itself in the ``App Paths`` registry key, which only ``ShellExecute``
consults.  The tool therefore resolves the configured command to a real
executable *before* launching.  Nothing here starts a process: ``Popen`` is
replaced by a recorder.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from jarvis.config import Settings
from jarvis.tools.apps import (
    OpenAppArgs,
    make_open_app_spec,
    resolve_command,
    split_command,
)
from jarvis.tools.base import ToolContext


def _ctx(command: str = "chrome.exe", *, dry_run: bool = False) -> ToolContext:
    settings = Settings()
    settings.apps.chrome = command
    return ToolContext(settings=settings, dry_run=dry_run)


class _Recorder:
    """Stand-in for ``subprocess.Popen`` that records the argv it was given."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str], **kwargs: object) -> object:
        self.calls.append(list(argv))
        return type("Proc", (), {"pid": 4242})()


# ── resolution ──────────────────────────────────────────────────────────


class TestResolveCommand:
    def test_absolute_path_is_used_as_given(self, tmp_path: Path) -> None:
        target = tmp_path / "thing.exe"
        target.write_text("", encoding="utf-8")
        assert resolve_command(str(target)) == str(target)

    def test_missing_absolute_path_does_not_fall_through(self) -> None:
        assert resolve_command(r"C:\nope\nothing.exe") is None

    def test_path_lookup_is_used_first(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        found = str(tmp_path / "onpath.exe")
        monkeypatch.setattr("jarvis.tools.apps.shutil.which", lambda name: found)
        assert resolve_command("onpath.exe") == found

    def test_app_paths_registry_is_the_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A bare name that is not on PATH still resolves (the Chrome case)."""
        chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
        monkeypatch.setattr("jarvis.tools.apps.shutil.which", lambda name: None)
        monkeypatch.setattr(
            "jarvis.tools.apps._app_paths_lookup",
            lambda name: chrome if name == "chrome.exe" else None,
        )
        assert resolve_command("chrome.exe") == chrome
        assert resolve_command("nothing-here.exe") is None

    def test_app_paths_lookup_is_read_only_and_safe(self) -> None:
        """The real registry lookup never raises, whatever the machine has."""
        from jarvis.tools.apps import _app_paths_lookup

        assert _app_paths_lookup("definitely-not-installed-xyz.exe") is None


# ── launch ──────────────────────────────────────────────────────────────


class TestOpenAppLaunch:
    def test_resolved_path_is_what_gets_launched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _Recorder()
        monkeypatch.setattr(subprocess, "Popen", recorder)
        monkeypatch.setattr(
            "jarvis.tools.apps.resolve_command", lambda name: r"C:\wherever\chrome.exe"
        )

        result = make_open_app_spec().run_verified(OpenAppArgs(name="chrome"), _ctx())

        assert recorder.calls == [[r"C:\wherever\chrome.exe"]]
        assert result.ok is True
        assert result.data["resolved"] == r"C:\wherever\chrome.exe"

    def test_configured_arguments_are_preserved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _Recorder()
        monkeypatch.setattr(subprocess, "Popen", recorder)
        monkeypatch.setattr(
            "jarvis.tools.apps.resolve_command", lambda name: r"C:\wherever\chrome.exe"
        )

        make_open_app_spec().run_verified(
            OpenAppArgs(name="chrome"), _ctx('"chrome.exe" --new-window')
        )

        assert recorder.calls == [[r"C:\wherever\chrome.exe", "--new-window"]]

    def test_unresolvable_command_reports_honestly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _Recorder()
        monkeypatch.setattr(subprocess, "Popen", recorder)
        monkeypatch.setattr("jarvis.tools.apps.resolve_command", lambda name: None)

        result = make_open_app_spec().run_verified(OpenAppArgs(name="chrome"), _ctx())

        assert result.ok is False
        assert recorder.calls == []
        assert "could not find an executable" in (result.error or "")
        # A failed run is not "verified": apply_verify only runs on ok results,
        # so the failure is reported by ok/error and never as a verified step.
        assert result.verified is None

    def test_unknown_app_lists_the_allowlist(self) -> None:
        result = make_open_app_spec().run_verified(OpenAppArgs(name="spotify"), _ctx())
        assert result.ok is False
        assert "unknown app" in (result.error or "")
        assert "notepad" in result.data["known"]

    def test_dry_run_launches_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _Recorder()
        monkeypatch.setattr(subprocess, "Popen", recorder)

        result = make_open_app_spec().run_verified(
            OpenAppArgs(name="chrome"), _ctx(dry_run=True)
        )

        assert recorder.calls == []
        assert result.ok is True
        assert "[dry-run]" in result.output


class TestSplitCommand:
    def test_quoted_program_with_arguments(self) -> None:
        assert split_command('"C:\\Program Files\\App\\app.exe" --flag x') == [
            r"C:\Program Files\App\app.exe",
            "--flag",
            "x",
        ]

    def test_bare_name_stays_a_single_token(self) -> None:
        assert split_command("chrome.exe") == ["chrome.exe"]
