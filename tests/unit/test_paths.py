"""Phase 2 path-policy tests (docs/03_SECURITY_AND_POLICY.md section 5, T4).

Covers :func:`resolve_safe` rejection/sanitisation, protected-region rules
(exact, container enclosure, wildcard) and allowed-root containment.  Protected
regions are injected from ``tmp_path`` so tests are deterministic; real OS
dirs are only touched read-only when the environment actually has them.
"""

from __future__ import annotations

import os
import string
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from jarvis.policy.paths import (
    PathError,
    covers_any_root,
    is_protected,
    normalise_key,
    resolve_safe,
    within_any_root,
)


def _root_drive() -> str:
    return os.path.splitdrive(os.path.expanduser("~"))[0] or "C:"


# ---------------------------------------------------------------------------
# resolve_safe: rejection matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    ["", "   ", "a\x00b", r"\\server\share", r"\\.\device", r"\\?\C:\x"],
)
def test_resolve_safe_rejects_unsafe_forms(raw: str) -> None:
    with pytest.raises(PathError):
        resolve_safe(raw)


def test_resolve_safe_rejects_device_names() -> None:
    root = _root_drive() + os.sep
    for name in ("CON", "NUL", "PRN", "AUX", "COM1", "LPT9"):
        with pytest.raises(PathError):
            resolve_safe(f"{root}{name}")


def test_resolve_safe_rejects_alternate_data_streams_and_drive_relative() -> None:
    drive = _root_drive()
    for raw in (f"{drive}{os.sep}notes.txt:admin", f"{drive}relative"):
        with pytest.raises(PathError):
            resolve_safe(raw)
    if os.name != "nt":
        # ':' is an ordinary filename character on POSIX.
        assert resolve_safe("a:b").is_absolute()


def test_resolve_safe_rejects_none_and_non_string() -> None:
    with pytest.raises(PathError):
        resolve_safe(None)  # type: ignore[arg-type]
    with pytest.raises(PathError):
        resolve_safe(Path("x"))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# resolve_safe: positive resolution
# ---------------------------------------------------------------------------


def test_resolve_safe_expands_home_dir() -> None:
    expected = Path(os.path.expanduser("~")).resolve()
    assert resolve_safe(str(Path("~"))) == expected
    assert resolve_safe("~") == expected


def test_resolve_safe_expands_env_vars() -> None:
    if os.name == "nt":
        root = Path(os.environ["SystemRoot"]).resolve()
        assert resolve_safe("%SystemRoot%") == root
    else:
        home = Path(os.environ["HOME"]).resolve()
        assert resolve_safe("$HOME") == home


def test_resolve_safe_joins_cwd_and_collapses_dotdot() -> None:
    started = Path.cwd()
    resolved = resolve_safe("does-not-exist/../x")
    assert resolved.is_absolute()
    assert resolved == (started / "x").resolve()
    assert resolve_safe(f".{os.sep}a{os.sep}..{os.sep}b") == (started / "b").resolve()


def test_resolve_safe_normal_path_is_untouched(tmp_path: Path) -> None:
    asserted = tmp_path / "plain" / "file.txt"
    assert resolve_safe(str(asserted)).name == "file.txt"
    assert resolve_safe(str(asserted)).is_absolute()


# ---------------------------------------------------------------------------
# normalise_key / containment
# ---------------------------------------------------------------------------


def test_normalise_key_is_case_insensitive_on_windows() -> None:
    key = normalise_key("c:\\Users\\Bob\\File.txt")
    assert key == normalise_key("C:\\users\\bob\\file.txt")
    assert key.islower()


def test_within_any_root_respects_path_boundaries(tmp_path: Path) -> None:
    root = (tmp_path / "ws").resolve()
    inside = root / "a" / "b.txt"
    assert within_any_root(inside, [root]) is True
    assert within_any_root(root, [root]) is True
    assert within_any_root(root.parent, [root]) is False
    # 'wsX' shares a prefix with 'ws' but is a sibling folder, not a child.
    assert within_any_root(tmp_path / "wsX", [root]) is False


def test_covers_any_root_means_root_or_ancestor(tmp_path: Path) -> None:
    roots = [(tmp_path / "ws").resolve()]
    assert covers_any_root(tmp_path / "ws", roots) is True
    assert covers_any_root(tmp_path, roots) is True  # ancestor
    assert covers_any_root(tmp_path / "ws" / "sub", roots) is False  # descendant
    assert covers_any_root(tmp_path / "other", roots) is False
    if os.name == "nt":
        assert covers_any_root(tmp_path / "WS", roots) is True  # case-insensitive


# ---------------------------------------------------------------------------
# is_protected: injected exact / container / wildcard regions
# ---------------------------------------------------------------------------


def _regions(tmp_path: Path) -> tuple[list[Path], list[Path], tuple[str, ...]]:
    exact = [tmp_path / "sys", tmp_path / "home"]
    containers = [tmp_path / "localappdata", tmp_path / "roaming"]
    patterns = (
        f"{tmp_path}{os.sep}Users{os.sep}*{os.sep}AppData",
        f"{tmp_path}{os.sep}Users{os.sep}*{os.sep}AppData{os.sep}Roaming",
        f"{tmp_path}{os.sep}Users{os.sep}*{os.sep}.ssh",
    )
    for folder in (*exact, *containers, tmp_path / "Users"):
        folder.mkdir(parents=True, exist_ok=True)
    return exact, containers, patterns


def test_is_protected_exact_root_and_ancestors(tmp_path: Path) -> None:
    exact, _containers, _patterns = _regions(tmp_path)
    assert is_protected(tmp_path / "sys", exact=exact, containers=[], patterns=()) is True
    # an exact root protects equality/ancestors only, NOT descendants - the
    # SystemRoot-*file* case below is only protected once the container
    # enclosure rule (SystemRoot is also a container) applies.
    assert (
        is_protected(tmp_path / "sys" / "win.ini", exact=exact, containers=[], patterns=()) is False
    )
    # the parent of a protected dir covers it -> protected too
    assert is_protected(tmp_path, exact=exact, containers=[], patterns=()) is True
    # a sibling is not protected by an exact root
    assert is_protected(tmp_path / "elsewhere", exact=exact, containers=[], patterns=()) is False


def test_is_protected_container_enclosure(tmp_path: Path) -> None:
    _exact, containers, _patterns = _regions(tmp_path)
    deep = tmp_path / "localappdata" / "Google" / "Chrome" / "User Data" / "Default"
    assert is_protected(deep, exact=[], containers=containers, patterns=()) is True
    assert (
        is_protected(tmp_path / "localappdata", exact=[], containers=containers, patterns=())
        is True
    )
    assert (
        is_protected(tmp_path / "roaming" / "Mozilla", exact=[], containers=containers, patterns=())
        is True
    )
    assert (
        is_protected(tmp_path / "downloads", exact=[], containers=containers, patterns=()) is False
    )


def test_is_protected_wildcard_region(tmp_path: Path) -> None:
    _exact, _containers, patterns = _regions(tmp_path)
    users = tmp_path / "Users"
    (users / "bob" / "AppData" / "Roaming" / "cache").mkdir(parents=True)
    (users / "bob" / ".ssh").mkdir(parents=True)
    kwargs = {"exact": [], "containers": [], "patterns": patterns}

    assert is_protected(users / "bob" / "AppData", **kwargs) is True
    assert is_protected(users / "bob" / "AppData" / "Roaming" / "cache" / "f.db", **kwargs) is True
    assert is_protected(users / "bob" / "AppData" / "Local", **kwargs) is True
    assert is_protected(users / "bob" / ".ssh" / "id_rsa", **kwargs) is True
    # a single '*' never matches across two components
    assert is_protected(users / "bob" / "sub" / "AppData", **kwargs) is False
    assert is_protected(users / "AppData", **kwargs) is False
    # an ancestor that only *contains* a matching concrete dir is also protected
    assert is_protected(users / "bob", **kwargs) is True
    assert is_protected(users / "alice", **kwargs) is False
    # 'Users\*\.ssh' matches ANY single-component user's .ssh, not just bob's
    assert is_protected(users / "alice" / ".ssh" / "id_rsa", **kwargs) is True
    # two components under the '*' never match: 'Users\*\AppData' needs alice/sub
    assert is_protected(users / "alice" / "sub" / "AppData", **kwargs) is False


def test_is_protected_drive_root_is_always_protected(tmp_path: Path) -> None:
    drive = Path(os.path.splitdrive(str(tmp_path.resolve()))[0] or os.sep)
    assert is_protected(drive, exact=[], containers=[], patterns=()) is True


def test_real_os_protected_dirs_are_rejected(tmp_path: Path) -> None:
    if os.name != "nt":
        pytest.skip("SystemRoot/ProgramFiles only exist on Windows")
    assert is_protected(Path(os.environ["SystemRoot"])) is True
    assert is_protected(Path(os.environ["ProgramFiles"])) is True
    assert is_protected(Path(os.environ["LOCALAPPDATA"]) / "Google") is True
    assert is_protected(Path(os.path.expanduser("~"))) is True


def test_allowed_roots_that_live_under_home_are_not_protected(tmp_path: Path) -> None:
    """USERPROFILE is exact-only, so Desktop/Downloads can still be allowed roots."""
    if os.name != "nt":
        pytest.skip("USERPROFILE semantics are Windows-only")
    home = Path(os.path.expanduser("~")).resolve()
    for sub in ("Desktop", "Downloads"):
        assert is_protected(home / sub) is False


# ---------------------------------------------------------------------------
# Hypothesis properties
# ---------------------------------------------------------------------------


def _safe_components(parts: list[str]) -> list[str]:
    out = []
    for part in parts:
        upper = part.upper()
        if upper in {"CON", "NUL", "PRN", "AUX"} or upper.startswith(("COM", "LPT")):
            continue
        out.append(part)
    return out or ["default"]


@given(
    st.lists(
        st.text(min_size=1, max_size=12, alphabet=string.ascii_letters + string.digits + "_-"),
        min_size=1,
        max_size=4,
    )
)
def test_resolve_safe_hypothesis_never_raises_and_is_absolute(parts: list[str]) -> None:
    parts = _safe_components(parts)
    raw = os.sep.join(parts).lstrip(os.sep)
    if not raw:
        raw = "x"
    result = resolve_safe(raw)
    assert result.is_absolute()
    assert result.name == parts[-1]


@given(st.text(max_size=80))
def test_normalise_key_is_total_and_idempotent(raw: str) -> None:
    key = normalise_key(Path(raw))
    assert normalise_key(Path(key)) == key
    if os.name == "nt":
        assert key == normalise_key(Path(raw.upper()))


def test_dotted_device_name_variants_stay_ordinary_filenames() -> None:
    """Only the bare ambiguous forms are refused; ``NUL.txt`` is a real filename."""
    root = _root_drive() + os.sep
    for name in ("NUL.txt", "CON.log", "notes", "Com1.backup"):
        result = resolve_safe(f"{root}{name}")
        assert result.is_absolute()
        assert result.name == name
