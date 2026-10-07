"""Installed-app discovery and ``[apps]`` seeding for ``jarvis apps scan``.

Discovery is read-only (injectable ``.lnk`` reader, ``KEY_READ`` registry) and
fails soft per item.  The scan only PROPOSES ``[apps]`` entries:
:func:`append_apps_entries` writes with per-entry approval, preserves owner
comments, and fails closed on bad TOML / inline / ``[apps.sub]``-only files.
Never a registered tool.
"""

from __future__ import annotations

import logging
import os
import re
import tomllib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import tomli_w

from jarvis import config

logger = logging.getLogger(__name__)

#: Case-insensitive substrings that deny an executable stem (installers, and
#: versioned wrappers such as ``claude-ssh-<version>`` whose stem changes on
#: every release, so an exact stem can never match).
DENY_FRAGMENTS = frozenset({"unins", "setup", "instal", "update", "patch", "claude-ssh"})

#: Exact lower-cased executable stems that deny shells, script hosts, LOLBins,
#: interpreters, remote/pairing clients, package managers and admin utilities
#: (owner decision, 4.13).  Module constants, never config-driven (owner decision 5).
DENY_STEMS = frozenset(
    "cmd powershell powershell_ise pwsh wscript cscript mshta rundll32 regedit "  # noqa: SIM905
    "regedt32 regsvr32 msiexec certutil bitsadmin psexec conhost sysedit "
    "python pythonw py node npm npx deno ruby perl php java javaw dotnet "
    "wsl wt windowsterminal bash sh zsh fish git-bash mintty ssh scp sftp ftp "
    "telnet curl wget winget windowspackagemanagerserver choco chocolatey scoop "
    "pip pip3 reg schtasks sc net netsh taskkill wmic mmc mstsc diskpart bcdedit "
    "vssadmin takeown icacls cipher msconfig nvm git-gui dism sfc wevtutil".split()
)

#: Launcher stems ``open_in_app`` refuses: an interpreter EXECUTES the file
#: (4.5b); the shells too - ``bash notes.txt`` runs it as a script (4.13).
INTERPRETER_STEMS = frozenset(
    "python pythonw py node deno ruby perl php java javaw dotnet powershell "  # noqa: SIM905
    "pwsh cmd wscript cscript mshta bash sh zsh fish git-bash wsl wt".split()
)

#: Owner-approved key pattern for ``[apps]`` entries.
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")

#: ShortcutReader: .lnk -> (target, arguments).  Candidate: (target, arguments, source).
ShortcutReader = Callable[[Path], tuple[str, str]]
AppPathsSource = Callable[[], list[tuple[str, str]]]
Candidate = tuple[str, str, str]

_HEADER_RE = re.compile(r"(?m)^\[apps\][ \t]*(?:#.*)?$")


@dataclass(frozen=True)
class Proposal:
    """One owner-reviewable [apps] entry proposed by the scan."""

    name: str
    target: str
    source: str


def denied_reason(stem: str) -> str | None:
    """Return why ``stem`` is deny-listed, or None when it is proposable."""
    low = stem.lower()
    for fragment in DENY_FRAGMENTS:
        if fragment in low:
            return f"denied name fragment {fragment!r}"
    return "denied executable name" if low in DENY_STEMS else None


def default_start_menu_roots() -> list[Path]:
    """User + all-users Start Menu Programs folders (missing vars skipped)."""
    roots: list[Path] = []
    for var in ("APPDATA", "ProgramData"):
        if base := os.environ.get(var):
            roots.append(Path(base) / "Microsoft/Windows/Start Menu/Programs")
    return roots


def _com_reader() -> ShortcutReader | None:
    """Property-only .lnk reader; None when the COM object is unavailable."""
    try:
        import win32com.client

        shell = win32com.client.Dispatch("WScript.Shell")
    except Exception as exc:  # noqa: BLE001 - COM may fail any way; scan is optional
        logger.debug("WScript.Shell unavailable for scan: %s", exc)
        return None

    def read(lnk: Path) -> tuple[str, str]:
        link = shell.CreateShortCut(str(lnk))
        return str(link.TargetPath), str(link.Arguments)

    return read


def discover_start_menu(
    roots: Sequence[Path] | None = None,
    reader: ShortcutReader | None = None,
) -> list[Candidate]:
    """(target, arguments, source) per readable .lnk; failures are skipped."""
    resolved = list(default_start_menu_roots() if roots is None else roots)
    reader = reader or _com_reader()
    if reader is None:
        return []
    found: list[Candidate] = []
    for root in resolved:
        try:
            links = sorted(root.rglob("*.lnk"))
        except OSError as exc:
            logger.debug("cannot walk Start Menu root %s: %s", root, exc)
            continue
        for lnk in links:
            try:
                target, arguments = reader(lnk)
            except Exception as exc:  # noqa: BLE001 - one broken shortcut must not stop the walk
                logger.debug("unreadable shortcut %s: %s", lnk, exc)
                continue
            if target:
                found.append((target, arguments, "start menu"))
    return found


def _registry_app_paths() -> list[tuple[str, str]]:
    """Every App Paths default value; KEY_READ only, per-entry failures skip."""
    import winreg

    base = r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths"
    flags = (0, getattr(winreg, "KEY_WOW64_64KEY", 0), getattr(winreg, "KEY_WOW64_32KEY", 0))
    pairs: list[tuple[str, str]] = []
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        for flag in flags:
            try:
                with winreg.OpenKey(hive, base, 0, winreg.KEY_READ | flag) as key:
                    for position in range(winreg.QueryInfoKey(key)[0]):
                        try:
                            name = winreg.EnumKey(key, position)
                            with winreg.OpenKey(key, name, 0, winreg.KEY_READ) as sub:
                                pairs.append((name, str(winreg.QueryValueEx(sub, "")[0])))
                        except OSError:
                            continue
            except OSError:
                continue
    return pairs


def scan_proposals(
    settings: config.Settings,
    start_menu_roots: Sequence[Path] | None = None,
    reader: ShortcutReader | None = None,
    app_paths_source: AppPathsSource | None = None,
) -> list[Proposal]:
    """Propose [apps] entries from both sources, minus what is configured."""
    candidates = discover_start_menu(start_menu_roots, reader)
    try:
        pairs = (app_paths_source or _registry_app_paths)()
    except Exception as exc:  # noqa: BLE001 - unavailable registry must not abort the CLI
        logger.warning("App Paths source unavailable for scan: %s", exc)
        pairs = []
    for _name, value in pairs:
        if target := value.strip().strip('"'):
            candidates.append((target, "", "app paths"))
    return filter_candidates(candidates, [str(key) for key in settings.apps.model_dump()])


def filter_candidates(
    candidates: Iterable[Candidate],
    existing_names: Iterable[str],
) -> list[Proposal]:
    """Apply the owner's filter chain; one entry per name, stable order."""
    configured = {str(name).lower() for name in existing_names}
    out: list[Proposal] = []
    for target, arguments, source in candidates:
        path = Path(target.strip().strip('"'))
        if "\\windowsapps\\" in str(path).lower().replace("/", "\\"):
            # Store package paths are version-pinned (they change on every
            # app update, so the entry would go stale immediately), sit behind
            # package ACLs, and are often CLI helpers rather than the GUI app.
            logger.debug("skipping %s: version-pinned path", path)
            continue
        if not path.is_absolute() or not path.is_file() or path.suffix.lower() != ".exe":
            continue
        if arguments.strip() or denied_reason(path.stem) is not None:
            continue
        name = path.stem.lower()
        if name in configured or NAME_RE.match(name) is None:
            continue
        out.append(Proposal(name=name, target=str(path), source=source))
    out.sort(key=lambda proposal: (proposal.name, proposal.target.lower()))
    # One entry per name; the sort makes the first path win (System32 before
    # SysWOW64), so a duplicate name cannot be printed twice and then silently
    # collapsed by scan_command's name-keyed dict (4.13, the odbcad32 bug).
    by_name: dict[str, Proposal] = {}
    for proposal in out:
        by_name.setdefault(proposal.name, proposal)
    return list(by_name.values())


def _parse_indices(spec: str, total: int) -> list[int]:
    """Parse a comma-separated 1-based selection against `total` proposals."""
    picked: list[int] = []
    for chunk in spec.split(","):
        text = chunk.strip()
        if not text.isdigit():
            raise ValueError(f"bad --approve value {spec!r}; expected numbers like 1,3,5")
        index = int(text)
        if not 1 <= index <= total:
            raise ValueError(f"proposal {index} is out of range (1-{total})")
        picked.append(index)
    return picked


def _insert_block(raw: str, block: str) -> str:
    """Insert `block` under the [apps] header (before the next section)."""
    match = _HEADER_RE.search(raw)
    if match is None:
        raise ValueError("[apps] header missing; edit config.toml manually")
    rest = raw[match.end() :]
    next_header = re.search(r"(?m)^\[", rest)
    if next_header is None:
        return (raw if raw.endswith("\n") else raw + "\n") + block
    cut = match.end() + next_header.start()
    return raw[:cut] + block + raw[cut:]


def append_apps_entries(
    config_path: Path,
    entries: Mapping[str, str],
    *,
    dry_run: bool = False,
) -> list[str]:
    """Insert approved entries under [apps]; comments preserved; fail closed."""
    if not entries:
        return ["nothing to write"]
    if bad := [name for name in entries if NAME_RE.match(name) is None]:
        raise ValueError(f"bad app name {bad[0]!r}; edit config.toml manually")
    if not config_path.is_file():
        raise ValueError(f"no config.toml at {config_path}; run `jarvis init` first")
    try:
        raw = config_path.read_text(encoding="utf-8")
        current = tomllib.loads(raw)
    except OSError as exc:
        raise ValueError(f"cannot read {config_path}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"config.toml is invalid ({exc}); edit config.toml manually") from exc
    existing = current.get("apps")
    if existing is not None and (not isinstance(existing, dict) or _HEADER_RE.search(raw) is None):
        raise ValueError("[apps] is inline or not a table; edit config.toml manually")
    present = {str(key).lower() for key in (existing or {})}
    wanted = {name: value for name, value in entries.items() if name.lower() not in present}
    if not wanted:
        return ["nothing to write (entries already configured)"]
    skipped = len(entries) - len(wanted)
    verb = "[dry-run] would add" if dry_run else "added"
    lines = [f"{verb} {name} = {value}" for name, value in wanted.items()]
    if skipped:
        lines.append(f"skipped {skipped} already-configured entr(ies)")
    if dry_run:
        return lines
    block = "".join(tomli_w.dumps({name: value}) for name, value in wanted.items())
    base = raw if existing is not None else raw.rstrip("\n") + "\n\n[apps]\n"
    new_raw = _insert_block(base, block)
    backup = config_path.with_name(config_path.name + ".bak")
    try:
        backup.write_text(raw, encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot write backup {backup}: {exc}") from exc
    try:
        written = tomllib.loads(new_raw).get("apps") or {}
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"generated config invalid ({exc}); edit config.toml manually") from exc
    if any(str(written.get(name)) != value for name, value in wanted.items()):
        raise ValueError("write verification failed; edit config.toml manually")
    tmp = config_path.with_name(config_path.name + ".tmp")
    try:
        tmp.write_text(new_raw, encoding="utf-8")
        os.replace(tmp, config_path)
    except OSError as exc:
        raise ValueError(f"cannot write {config_path}: {exc}") from exc
    return [*lines, f"backup: {backup}"]


def validate_add(settings: config.Settings, name: str, raw_path: str) -> str:
    """Validate one owner-supplied entry (scan rules); returns the path."""
    if NAME_RE.match(name) is None:
        raise ValueError(f"bad app name {name!r}: letters, digits, _ or - (max 32)")
    path = Path(raw_path.strip().strip('"'))
    if not path.is_absolute():
        raise ValueError(f"path must be absolute, got {raw_path!r}")
    if not path.is_file():
        raise ValueError(f"file does not exist: {path}")
    if path.suffix.lower() != ".exe":
        raise ValueError(f"not an .exe: {path.name}")
    if (denied := denied_reason(path.stem)) is not None:
        raise ValueError(f"refusing {path.name}: {denied}")
    if name.lower() in {str(key).lower() for key in settings.apps.model_dump()}:
        raise ValueError(f"{name!r} is already in [apps]")
    return str(path)


def scan_command(
    settings: config.Settings,
    config_path: Path,
    approve: str = "",
    dry_run: bool = False,
    start_menu_roots: Sequence[Path] | None = None,
    reader: ShortcutReader | None = None,
    app_paths_source: AppPathsSource | None = None,
) -> list[str]:
    """Numbered proposal list; with --approve, write only the chosen entries."""
    proposals = scan_proposals(settings, start_menu_roots, reader, app_paths_source)
    if not proposals:
        return ["no proposals found (nothing discovered or nothing unconfigured)"]
    lines = [f"{i:>3}. {p.name:<16} {p.target}  [{p.source}]" for i, p in enumerate(proposals, 1)]
    if not approve:
        return [*lines, f"{len(proposals)} proposal(s); approve e.g. --approve 1"]
    chosen = [proposals[index - 1] for index in _parse_indices(approve, len(proposals))]
    entries = {proposal.name: proposal.target for proposal in chosen}
    return [*lines, "", *append_apps_entries(config_path, entries, dry_run=dry_run)]


def list_entries(settings: config.Settings) -> list[str]:
    """Render the configured [apps] entries, sorted, for `jarvis apps list`."""
    apps = settings.apps.model_dump()
    rows = [f"{key} = {value}" for key, value in apps.items() if isinstance(value, str) and value]
    return sorted(rows, key=str.lower) or ["no [apps] entries configured"]
