"""apps_scan tests: deny-list, filter, discovery, writer, scan, validate_add.

Hermetic: fake .lnk readers, fake App Paths sources and tmp_path configs; the
real Start Menu, registry and owner's config.toml are never touched.
"""

from __future__ import annotations

import logging
import tomllib
from pathlib import Path
from typing import Any

import pytest

from jarvis.apps_scan import (
    INTERPRETER_STEMS,
    ShortcutReader,
    append_apps_entries,
    denied_reason,
    discover_start_menu,
    filter_candidates,
    list_entries,
    scan_command,
    scan_proposals,
    validate_add,
)
from jarvis.config import AppSettings, Settings

#: Owner-style config: [apps] sits between two sections, with an inline comment.
ORIGINAL = (
    '[llm]\nprovider = "groq"\n\n[apps]\n'
    'notepad = "C:\\\\Windows\\\\System32\\\\notepad.exe"  # keep\n'
    'chrome = "chrome.exe"\n\n[daemon]\nport = 0\n'
)


def _exe(directory: Path, name: str) -> Path:
    path = directory / name
    path.write_text("", encoding="utf-8")
    return path


def _settings(**apps: str) -> Settings:
    return Settings(apps=AppSettings(**apps))


def _write_config(tmp_path: Path, content: str = ORIGINAL) -> Path:
    config = tmp_path / "config.toml"
    config.write_text(content, encoding="utf-8")
    return config


def _lnk_reader(_lnk: Path) -> tuple[str, str]:
    raise AssertionError("no shortcut should be read in this test")


def _start_menu(tmp_path: Path, name: str) -> tuple[list[Path], ShortcutReader]:
    exe = _exe(tmp_path, f"{name}.exe")
    (tmp_path / f"{name}.lnk").write_text("", encoding="utf-8")

    def reader(_lnk: Path) -> tuple[str, str]:
        return str(exe), ""

    return [tmp_path], reader


def _scan(settings: Settings, config: Path, **options: Any) -> list[str]:
    """scan_command with hermetic defaults: no real Start Menu or registry."""
    options.setdefault("start_menu_roots", [])
    options.setdefault("reader", _lnk_reader)
    options.setdefault("app_paths_source", list)
    return scan_command(settings, config, **options)


class TestDenyList:
    @pytest.mark.parametrize(
        "stem",
        [
            "cmd",
            "powershell",
            "wscript",
            "python",
            "py",
            "unins000",
            "setup",
            # 4.13: terminals/shells, remote clients, package managers,
            # admin utilities - exact stems, never prefixes.
            "wsl",
            "wt",
            "windowsterminal",
            "bash",
            "sh",
            "zsh",
            "fish",
            "git-gui",
            "mintty",
            "ssh",
            "scp",
            "sftp",
            "ftp",
            "telnet",
            "curl",
            "wget",
            "winget",
            "windowspackagemanagerserver",
            "choco",
            "scoop",
            "pip",
            "reg",
            "schtasks",
            "sc",
            "net",
            "netsh",
            "taskkill",
            "wmic",
            "mmc",
            "mstsc",
            "diskpart",
            "bcdedit",
            "vssadmin",
            "takeown",
            "icacls",
            "cipher",
            "msconfig",
            "nvm",
            "dism",
            "sfc",
            "wevtutil",
        ],
    )
    def test_denylisted_stems_are_refused(self, stem: str) -> None:
        assert denied_reason(stem) is not None

    @pytest.mark.parametrize("stem", ["claude-ssh-0.9.3", "claude-ssh-latest"])
    def test_versioned_wrapper_fragments_are_refused(self, stem: str) -> None:
        # The stem changes every release, so an exact deny stem can never match.
        assert denied_reason(stem) is not None

    @pytest.mark.parametrize("stem", ["pycharm", "code", "spotify", "notepad", "netflix"])
    def test_similar_names_still_pass(self, stem: str) -> None:
        assert denied_reason(stem) is None

    def test_exec_shells_are_launcher_refusals_too(self) -> None:
        # open_in_app refuses these at run time; the scan denies them too.
        assert {"bash", "sh", "zsh", "fish", "git-bash", "wsl", "wt"} <= INTERPRETER_STEMS


class TestFilter:
    def test_filter_chain_rejects_every_bad_candidate(self, tmp_path: Path) -> None:
        exe = _exe(tmp_path, "code.exe")
        odd = _exe(tmp_path, "my app!.exe")
        notes = tmp_path / "notes.txt"
        notes.write_text("hi", encoding="utf-8")
        candidates = [
            ("code.exe", "", "start menu"),
            (str(tmp_path / "ghost.exe"), "", "start menu"),
            (str(notes), "", "start menu"),
            (str(exe), "--new-window", "start menu"),
            (str(odd), "", "start menu"),
            (str(exe), "", "start menu"),
        ]
        assert [p.name for p in filter_candidates(candidates, [])] == ["code"]
        assert filter_candidates(candidates, ["Code"]) == []

    @pytest.mark.parametrize(
        "name", ["cmd.exe", "powershell.exe", "python.exe", "unins000.exe", "setup.exe"]
    )
    def test_denylisted_executables_are_never_proposed(self, name: str, tmp_path: Path) -> None:
        assert filter_candidates([(str(_exe(tmp_path, name)), "", "start menu")], []) == []

    def test_duplicate_targets_collapse_and_order_is_stable(self, tmp_path: Path) -> None:
        code = _exe(tmp_path, "code.exe")
        other = _exe(tmp_path, "spotify.exe")
        candidates = [
            (str(code), "", "app paths"),
            (str(other), "", "start menu"),
            (f'"{code}"', "", "start menu"),
        ]
        found = filter_candidates(candidates, [])
        assert [p.name for p in found] == ["code", "spotify"]
        assert found[0].source == "app paths"

    def test_duplicate_names_collapse_to_the_first_sorted_path(self, tmp_path: Path) -> None:
        # The live odbcad32 bug: System32 and SysWOW64 both propose the same
        # name.  One entry per name, first sorted path wins (System32 sorts
        # before SysWOW64), regardless of candidate order.
        system32 = tmp_path / "System32"
        syswow64 = tmp_path / "SysWOW64"
        system32.mkdir()
        syswow64.mkdir()
        wanted = _exe(system32, "odbcad32.exe")
        other_arch = _exe(syswow64, "odbcad32.exe")

        found = filter_candidates(
            [(str(other_arch), "", "start menu"), (str(wanted), "", "start menu")], []
        )

        assert [p.name for p in found] == ["odbcad32"]
        assert found[0].target == str(wanted)

    def test_windowsapps_paths_are_skipped_as_version_pinned(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        store = tmp_path / "Program Files" / "WindowsApps"
        store.mkdir(parents=True)
        store_exe = _exe(store, "spotify_cli.exe")
        normal = _exe(tmp_path, "spotify.exe")

        with caplog.at_level(logging.DEBUG, logger="jarvis.apps_scan"):
            found = filter_candidates(
                [(str(store_exe), "", "start menu"), (str(normal), "", "start menu")], []
            )

        assert [p.name for p in found] == ["spotify"]
        assert "version-pinned path" in caplog.text


class TestDiscovery:
    def test_walk_finds_shortcuts_and_skips_unreadable_ones(self, tmp_path: Path) -> None:
        (tmp_path / "Vendor").mkdir()
        (tmp_path / "Vendor" / "Code.lnk").write_text("", encoding="utf-8")
        (tmp_path / "Broken.lnk").write_text("", encoding="utf-8")
        exe = _exe(tmp_path, "code.exe")

        def reader(lnk: Path) -> tuple[str, str]:
            if lnk.name == "Broken.lnk":
                raise OSError("corrupt shortcut")
            return str(exe), ""

        assert discover_start_menu([tmp_path], reader) == [(str(exe), "", "start menu")]

    def test_missing_root_discovers_nothing(self, tmp_path: Path) -> None:
        (tmp_path / "Code.lnk").write_text("", encoding="utf-8")
        assert discover_start_menu([tmp_path / "gone"], _lnk_reader) == []

    def test_without_a_reader(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "Code.lnk").write_text("", encoding="utf-8")
        monkeypatch.setattr("jarvis.apps_scan._com_reader", lambda: None)
        assert discover_start_menu([tmp_path], None) == []

    def test_unavailable_registry_source_is_fail_soft(self) -> None:
        def boom() -> list[tuple[str, str]]:
            raise RuntimeError("registry unavailable")

        assert scan_proposals(_settings(), [], _lnk_reader, boom) == []

    def test_scan_merges_both_sources_minus_configured(self, tmp_path: Path) -> None:
        code = _exe(tmp_path, "code.exe")
        roots, reader = _start_menu(tmp_path, "spotify")
        proposals = scan_proposals(
            _settings(code=str(code)), roots, reader, lambda: [("code.exe", f'"{code}"')]
        )
        assert [p.name for p in proposals] == ["spotify"]


class TestWriter:
    def test_inserts_under_apps_and_preserves_comments(self, tmp_path: Path) -> None:
        config = _write_config(tmp_path)
        exe = _exe(tmp_path, "code.exe")

        lines = append_apps_entries(config, {"code": str(exe)})

        raw = config.read_text(encoding="utf-8")
        parsed = tomllib.loads(raw)
        assert parsed["apps"]["code"] == str(exe)
        assert parsed["apps"]["notepad"] == "C:\\Windows\\System32\\notepad.exe"
        assert "# keep" in raw
        assert raw.index("code =") < raw.index("[daemon]")
        assert lines == [f"added code = {exe}", f"backup: {tmp_path / 'config.toml.bak'}"]
        assert (tmp_path / "config.toml.bak").read_text(encoding="utf-8") == ORIGINAL

    def test_creates_apps_section_when_missing(self, tmp_path: Path) -> None:
        original = '[llm]\nprovider = "groq"\n'
        config = _write_config(tmp_path, original)

        append_apps_entries(config, {"code": "C:/apps/code.exe"})

        parsed = tomllib.loads(config.read_text(encoding="utf-8"))
        assert parsed["apps"] == {"code": "C:/apps/code.exe"}
        assert (tmp_path / "config.toml.bak").read_text(encoding="utf-8") == original

    def test_dry_run_writes_nothing(self, tmp_path: Path) -> None:
        config = _write_config(tmp_path)
        lines = append_apps_entries(config, {"code": "C:/apps/code.exe"}, dry_run=True)

        assert config.read_text(encoding="utf-8") == ORIGINAL
        assert not (tmp_path / "config.toml.bak").exists()
        assert lines == ["[dry-run] would add code = C:/apps/code.exe"]

    def test_already_configured_entry_is_skipped(self, tmp_path: Path) -> None:
        config = _write_config(tmp_path)
        entries = {"notepad": "C:/other.exe", "code": "C:/apps/code.exe"}

        lines = append_apps_entries(config, entries)

        parsed = tomllib.loads(config.read_text(encoding="utf-8"))
        assert parsed["apps"]["notepad"] == "C:\\Windows\\System32\\notepad.exe"
        assert parsed["apps"]["code"] == "C:/apps/code.exe"
        assert lines[1] == "skipped 1 already-configured entr(ies)"

    @pytest.mark.parametrize(
        "content",
        ["[apps\nbroken", 'apps = { notepad = "notepad.exe" }\n', '[apps.editor]\ncommand = "x"\n'],
    )
    def test_bad_configs_are_refused_untouched(self, content: str, tmp_path: Path) -> None:
        config = _write_config(tmp_path, content)

        with pytest.raises(ValueError, match="edit config.toml manually"):
            append_apps_entries(config, {"code": "C:/apps/code.exe"})

        assert config.read_text(encoding="utf-8") == content
        assert not (tmp_path / "config.toml.bak").exists()

    def test_bad_name_and_missing_config_are_refused(self, tmp_path: Path) -> None:
        config = _write_config(tmp_path)
        with pytest.raises(ValueError, match="bad app name"):
            append_apps_entries(config, {"bad name": "C:/apps/code.exe"})
        with pytest.raises(ValueError, match="jarvis init"):
            append_apps_entries(tmp_path / "missing.toml", {"code": "C:/apps/code.exe"})
        assert config.read_text(encoding="utf-8") == ORIGINAL


class TestScanCommand:
    def test_lists_numbered_proposals_without_writing(self, tmp_path: Path) -> None:
        config = _write_config(tmp_path)
        roots, reader = _start_menu(tmp_path, "code")

        lines = _scan(_settings(), config, start_menu_roots=roots, reader=reader)

        assert lines[0].startswith("  1. code")
        assert "approve e.g. --approve 1" in lines[-1]
        assert config.read_text(encoding="utf-8") == ORIGINAL

    def test_duplicate_names_are_printed_once(self, tmp_path: Path) -> None:
        # The live symptom: the numbered list printed odbcad32 twice (System32
        # and SysWOW64) while scan_command's name-keyed dict silently kept one.
        config = _write_config(tmp_path)
        system32 = tmp_path / "System32"
        syswow64 = tmp_path / "SysWOW64"
        system32.mkdir()
        syswow64.mkdir()
        wanted = _exe(system32, "odbcad32.exe")
        other_arch = _exe(syswow64, "odbcad32.exe")

        lines = _scan(
            _settings(),
            config,
            app_paths_source=lambda: [
                ("syswow64.exe", str(other_arch)),
                ("system32.exe", str(wanted)),
            ],
        )

        proposal_lines = [line for line in lines if "odbcad32" in line]
        assert len(proposal_lines) == 1
        assert str(wanted) in proposal_lines[0]

    def test_approve_writes_only_the_chosen_entry(self, tmp_path: Path) -> None:
        config = _write_config(tmp_path)
        roots, reader = _start_menu(tmp_path, "code")

        lines = _scan(_settings(), config, approve="1", start_menu_roots=roots, reader=reader)

        parsed = tomllib.loads(config.read_text(encoding="utf-8"))
        assert parsed["apps"]["code"].endswith("code.exe")
        assert any(line.startswith("added code") for line in lines)

    @pytest.mark.parametrize(("spec", "message"), [("9", "out of range"), ("abc", "bad --approve")])
    def test_bad_approve(self, spec: str, message: str, tmp_path: Path) -> None:
        config = _write_config(tmp_path)
        roots, reader = _start_menu(tmp_path, "code")

        with pytest.raises(ValueError, match=message):
            _scan(_settings(), config, approve=spec, start_menu_roots=roots, reader=reader)

        assert config.read_text(encoding="utf-8") == ORIGINAL

    def test_scan_dry_run_reports_and_writes_nothing(self, tmp_path: Path) -> None:
        config = _write_config(tmp_path)
        roots, reader = _start_menu(tmp_path, "code")
        lines = _scan(
            _settings(), config, approve="1", dry_run=True, start_menu_roots=roots, reader=reader
        )

        assert any("[dry-run]" in line for line in lines)
        assert config.read_text(encoding="utf-8") == ORIGINAL

    def test_no_proposals_is_reported_not_errored(self, tmp_path: Path) -> None:
        assert _scan(_settings(), _write_config(tmp_path)) == [
            "no proposals found (nothing discovered or nothing unconfigured)"
        ]


class TestValidateAdd:
    def test_valid_entry_returns_the_absolute_path(self, tmp_path: Path) -> None:
        exe = _exe(tmp_path, "code.exe")
        assert validate_add(_settings(), "vscode", str(exe)) == str(exe)

    @pytest.mark.parametrize(
        ("name", "raw"),
        [
            ("", "C:/apps/code.exe"),
            ("bad name", "C:/apps/code.exe"),
            ("a" * 33, "C:/apps/code.exe"),
            ("with.dot", "C:/apps/code.exe"),
            ("code", "code.exe"),
            ("code", "{tmp}/ghost.exe"),
        ],
    )
    def test_bad_names_and_paths_are_refused(self, name: str, raw: str, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="bad app name|absolute|does not exist"):
            validate_add(_settings(), name, raw.format(tmp=tmp_path))

    @pytest.mark.parametrize("name", ["notes.txt", "setup.msi"])
    def test_only_exe_files_are_accepted(self, name: str, tmp_path: Path) -> None:
        path = tmp_path / name
        path.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="not an .exe"):
            validate_add(_settings(), "tool", str(path))

    def test_denylisted_executable_is_refused(self, tmp_path: Path) -> None:
        exe = _exe(tmp_path, "cmd.exe")
        with pytest.raises(ValueError, match="refusing"):
            validate_add(_settings(), "shell", str(exe))

    def test_duplicate_name_is_refused_case_insensitively(self, tmp_path: Path) -> None:
        exe = _exe(tmp_path, "code.exe")
        with pytest.raises(ValueError, match="already in"):
            validate_add(_settings(vscode=str(exe)), "VSCode", str(exe))


class TestListEntries:
    def test_entries_render_sorted(self) -> None:
        rows = list_entries(_settings(vscode="C:/apps/code.exe", spotify="C:/spotify.exe"))
        assert [row.split(" = ")[0] for row in rows] == [
            "calculator",
            "chrome",
            "notepad",
            "spotify",
            "vscode",
        ]
