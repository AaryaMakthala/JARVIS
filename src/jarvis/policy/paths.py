"""Windows-safe path policy (docs/03_SECURITY_AND_POLICY.md section 5).

Before any policy decision or file operation every path is resolved with
:func:`resolve_safe` to an absolute, real path.  We reject:

* NUL bytes, empty/whitespace-only paths;
* alternate data streams (``file.txt:stream``) and drive-relative forms;
* UNC (``\\\\server\\share``) and device paths (``\\\\.\\``, ``\\\\?\\``);
* bare device names at the root (``CON``, ``NUL``, ``COM1`` ...).

Phase 2 hardening (on top of Phase 1):

* :func:`resolve_safe` combines :meth:`~pathlib.Path.resolve` (which on
  Windows already expands junctions, 8.3 short names and symlinks of every
  existing component) with a Win32 ``GetLongPathName`` pass as backup, so a
  spelling like ``C:\\PROGRA~1`` cannot hide a protected directory;
* **enclosure**: a target that lies *inside* a protected region (AppData,
  ``.ssh``, System, Program Files, ProgramData, the JARVIS data dir, other
  users' AppData) is protected too, not just the root itself.  ``USERPROFILE``
  is intentionally *not* enclosure-protected - the allowed roots (Desktop,
  Downloads) live inside it - only equality/ancestor protected;
* Windows wildcard patterns (``%USERPROFILE drive%\\Users\\*\\AppData``) are
  projected against what actually exists on disk, bounded and side-effect free.

All comparisons run on the normalised (case-insensitive, ``normcase``) real
path, so ``..`` tricks, case and slash variations cannot slip past.
"""

from __future__ import annotations

import importlib
import os
import re
from functools import lru_cache
from pathlib import Path

from jarvis import config

PathStr = str  #: Marked in tool args models; the engine treats it as a path.

_DRIVE_RE = re.compile(r"^[A-Za-z]:")
_DEVICE_NAMES = {"CON", "NUL", "PRN", "AUX"}
_DEVICE_NAMES |= {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}
_ILLEGAL_CHARS = ":"  # drive letters handled separately; data streams use ':'

_SPECIAL_PREFIXES = ("\\\\", r"\\.\\", "\\\\?\\")

#: Windows 8.3 short-name components look like ``PROGRA~1``.  Used only to
#: decide when a Win32 long-name expansion pass is worth running.
_SHORT_NAME_RE = re.compile(r"~[0-9]+(?=$|[\\/])")

_WILD = "*"

#: Environment variables whose values form *container* protected roots: any
#: descendant is protected too.  These are all below ``USERPROFILE`` except the
#: OS roots, so none of them can accidentally contain an allowed root.
_CONTAINER_ENV_VARS = (
    "SystemRoot",
    "ProgramFiles",
    "ProgramFiles(x86)",
    "ProgramData",
    "LOCALAPPDATA",
    "APPDATA",
)

#: Browser profile folders (absolute, concrete).  Defence in depth: they are
#: already covered by ``LOCALAPPDATA``/``APPDATA`` being containers, but naming
#: them keeps the protected set explicit and readable.
_BROWSER_PROFILES = (
    ("LOCALAPPDATA", "Google", "Chrome", "User Data"),
    ("LOCALAPPDATA", "Chromium", "User Data"),
    ("LOCALAPPDATA", "Microsoft", "Edge", "User Data"),
    ("APPDATA", "Mozilla", "Firefox", "Profiles"),
)


class PathError(ValueError):
    """Raised when a path string is rejected before any operation happens."""


def resolve_safe(raw: str) -> Path:
    """Expand and absolutely resolve ``raw``, rejecting unsafe forms.

    Raises :class:`PathError` for anything that must not reach the filesystem.
    The returned :class:`~pathlib.Path` is absolute (``Path.resolve``) with
    ``~``/env vars expanded, ``..`` collapsed and - on Windows - junctions,
    symlinks, reparse points and 8.3 short names resolved to their long real
    path.  When the Win32 long-name pass cannot run, we keep the resolved path
    (existing prefixes were already long-formed and symlink-expanded).
    """
    if not isinstance(raw, str):
        raise PathError(f"path must be a string, got {type(raw).__name__}")
    value = raw.strip()
    if not value:
        raise PathError("path is empty")
    if "\x00" in value:
        raise PathError("path contains a NUL byte")
    if any(value.lower().startswith(prefix) for prefix in _SPECIAL_PREFIXES):
        raise PathError(f"UNC/device paths are not allowed: {value!r}")

    # Alternate data streams (foobar.txt:stream) and drive-relative forms
    # (C:foo) are Windows constructs; reject ':' anywhere it is not the drive
    # letter separator on Windows.  On POSIX ':' is a legal filename character.
    if os.name == "nt":
        colons = [i for i, ch in enumerate(value) if ch == ":"]
        if colons:
            if len(colons) != 1 or colons[0] != 1:
                raise PathError(f"path contains an illegal ':' character: {value!r}")
            if not re.match(r"^[A-Za-z]:[\\/]", value):
                raise PathError(f"drive-relative path is not allowed: {value!r}")

    expanded = os.path.expandvars(os.path.expanduser(value))
    if not expanded:
        raise PathError(f"path expands to empty: {value!r}")

    absolute = Path(expanded)
    if not absolute.is_absolute():
        absolute = Path.cwd() / absolute
    # resolve(strict=False) collapses '..' and, on Windows, resolves
    # symlinks/reparse points + 8.3 short names of existing components; it
    # never touches the filesystem when strict=False.
    resolved = absolute.resolve(strict=False)
    _reject_device_name(resolved, value)

    if os.name == "nt" and _SHORT_NAME_RE.search(os.path.normcase(str(resolved))):
        long_name = _expand_short_names(str(resolved))
        if long_name and long_name != str(resolved):
            long_path = Path(long_name)
            _reject_device_name(long_path, value)
            resolved = long_path
    return resolved


def _reject_device_name(path: Path, original: str) -> None:
    """Refuse a bare device name right at the root (CON/NUL/... abuse)."""
    stem = path.name.upper()
    if stem in _DEVICE_NAMES:
        raise PathError(f"device name is not allowed as a path: {original!r}")


@lru_cache(maxsize=4096)
def _expand_short_names(raw: str) -> str | None:
    """Best-effort Win32 8.3 -> long-name expansion; ``None`` when unavailable.

    ``GetLongPathNameW`` returns the input unchanged when nothing can be
    expanded, so callers compare the return value with the input.
    """
    if os.name != "nt":
        return None
    try:
        win32api = importlib.import_module("win32api")
        value: str = win32api.GetLongPathName(raw)
    except Exception:  # noqa: BLE001 - Win32 is best effort; resolve_safe kept the realpath
        return None
    return value or None


def normalise_key(path: Path | str) -> str:
    """Case-fold + normcase a resolved path for comparisons."""
    return os.path.normcase(str(path)).casefold()


# ---------------------------------------------------------------------------
# Allowed roots
# ---------------------------------------------------------------------------


def roots_from_settings(settings: config.Settings) -> list[Path]:
    """Expand ``settings.policy.allowed_roots`` into resolved absolute roots."""
    return [resolve_safe(value) for value in settings.policy.allowed_roots]


def within_any_root(path: Path, roots: list[Path]) -> bool:
    """Return ``True`` when ``path`` equals or lies inside one of ``roots``.

    Both the path and every root are normalised (case-insensitive on
    Windows).  The caller is responsible for resolving roots with
    :func:`resolve_safe`.
    """
    target = normalise_key(path)
    for root in roots:
        key = normalise_key(root)
        if target == key or target.startswith(key.rstrip(os.sep) + os.sep):
            return True
    return False


def covers_any_root(path: Path, roots: list[Path]) -> bool:
    """Return ``True`` when ``path`` is an allowed root or an ancestor of one.

    Used to refuse destructive operations (delete_path) that would remove an
    allowed folder itself instead of something inside it.
    """
    key = normalise_key(path)
    for root in roots:
        rk = normalise_key(root)
        if key == rk:
            return True
        if rk.startswith(key.rstrip(os.sep) + os.sep):
            return True
    return False


def ensure_allowed_root(path: Path, roots: list[Path]) -> None:
    """Raise :class:`PathError` unless ``path`` is inside one of ``roots``."""
    if not within_any_root(path, roots):
        sample = "\n  ".join(str(r) for r in roots)
        raise PathError(f"path is outside the allowed folders (allowed roots):\n  {sample}")


# ---------------------------------------------------------------------------
# Protected regions
# ---------------------------------------------------------------------------


def protected_roots(settings: config.Settings | None = None) -> list[Path]:
    """Exact/ancestor protected roots (absolute): OS roots, the home folder and
    the JARVIS data directory.

    A target that *equals* or is an *ancestor of* these is always protected.
    ``USERPROFILE`` is deliberately not enclosure-protected (allowed roots live
    under it) - see :func:`protected_containers`.  Never derived from config:
    these cannot be overridden.
    """
    _ = settings  # reserved: later phases may read classifier settings.
    raw = ["SystemRoot", "ProgramFiles", "ProgramFiles(x86)", "ProgramData", "USERPROFILE"]
    roots: list[Path] = []
    for var in raw:
        value = os.getenv(var)
        if value:
            roots.append(Path(value).resolve(strict=False))
    roots.append(config.user_data_dir().resolve(strict=False))
    return _dedupe(roots)


def protected_containers(settings: config.Settings | None = None) -> list[Path]:
    """Container protected roots: the target *itself and every descendant* of
    these are protected (AppData, ``.ssh``, OS dirs, browser profiles, the
    JARVIS data directory).

    None of these may contain an allowed root, so enclosure protection cannot
    shadow the user's own folders.
    """
    _ = settings
    roots: list[Path] = []
    for var in _CONTAINER_ENV_VARS:
        value = os.getenv(var)
        if value:
            roots.append(Path(value).resolve(strict=False))
    ssh = os.path.expanduser("~/.ssh")
    if ssh:
        roots.append(Path(ssh).resolve(strict=False))
    for env_var, *parts in _BROWSER_PROFILES:
        resolved = _browser_profile(env_var, tuple(parts))
        if resolved is not None:
            roots.append(resolved)
    return _dedupe(roots)


def _browser_profile(env_var: str, parts: tuple[str, ...]) -> Path | None:
    base = os.getenv(env_var)
    if not base:
        return None
    return Path(base).joinpath(*parts).resolve(strict=False)


def _dedupe(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for p in paths:
        key = normalise_key(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _wildcard_patterns() -> tuple[str, ...]:
    """Protected *any-user* regions abstracted to a single wildcard level."""
    home = os.path.expanduser("~")
    drive = os.path.splitdrive(home)[0] or os.path.splitdrive(os.getcwd())[0] or "C:"
    return (
        f"{drive}{os.sep}Users{os.sep}{_WILD}{os.sep}AppData",
        f"{drive}{os.sep}Users{os.sep}{_WILD}{os.sep}AppData{os.sep}Local",
        f"{drive}{os.sep}Users{os.sep}{_WILD}{os.sep}AppData{os.sep}Roaming",
        f"{drive}{os.sep}Users{os.sep}{_WILD}{os.sep}.ssh",
    )


def is_protected(
    path: Path,
    *,
    exact: list[Path] | None = None,
    containers: list[Path] | None = None,
    patterns: tuple[str, ...] | None = None,
) -> bool:
    """Return ``True`` when ``path`` may not be targeted by a file tool.

    Protected means one of:

    * a drive root;
    * equal to / an ancestor of an ``exact`` protected root (e.g. the home
      folder itself);
    * equal to / an ancestor of / **inside** a ``containers`` protected root
      (e.g. ``...\\AppData\\Roaming\\x``);
    * equal to / inside / an ancestor of a ``patterns`` wildcard region
      (any-user ``AppData`` / ``.ssh``).

    ``path`` must already be resolved via :func:`resolve_safe`.  ``exact``,
    ``containers`` and ``patterns`` are injectable so tests can pin them.
    """
    if exact is None:
        exact = protected_roots()
    if containers is None:
        containers = protected_containers()
    if patterns is None:
        patterns = _wildcard_patterns()

    key = normalise_key(path)
    if key == normalise_key(path.anchor):
        return True  # a drive root itself is always protected

    if _pattern_region_hit(key, path, patterns):
        return True

    for root in exact:
        root_key = normalise_key(root)
        if key == root_key:
            return True
        # path is an ancestor of an exact protected root -> reaches into it
        if root_key.startswith(key.rstrip(os.sep) + os.sep):
            return True

    for root in containers:
        root_key = normalise_key(root)
        if key == root_key:
            return True
        # ancestors of a container -> reaches into it
        if root_key.startswith(key.rstrip(os.sep) + os.sep):
            return True
        # descendants of a container -> enclosure protection
        if key.startswith(root_key.rstrip(os.sep) + os.sep):
            return True
    return False


def _pattern_region_hit(key: str, path: Path, patterns: tuple[str, ...]) -> bool:
    """``path`` equals/inside a wildcard region, or covers one's concrete dir."""
    components = key.split(os.sep)
    for pattern in patterns:
        pattern_comps = [part.casefold() for part in pattern.split(os.sep)]
        # (a) the path or any ancestor matches the pattern exactly.
        for i in range(len(components), 0, -1):
            if _components_match(components[:i], pattern_comps):
                return True
        # (b) the path is an ancestor of a concrete dir matching the pattern
        #     (e.g. target = C:\Users covers C:\Users\bob\AppData).
        for concrete in _expand_wildcard(pattern):
            concrete_key = normalise_key(concrete)
            if concrete_key.startswith(key.rstrip(os.sep) + os.sep):
                return True
    return False


def _components_match(target: list[str], pattern: list[str]) -> bool:
    """Exact-length, single-level wildcard match (``*`` matches one component)."""
    if len(target) != len(pattern):
        return False
    for target_part, pattern_part in zip(target, pattern):
        if pattern_part == _WILD.casefold():
            continue
        if target_part != pattern_part:
            return False
    return True


def _expand_wildcard(pattern: str) -> list[Path]:
    """Concrete on-disk paths matching a single-``*`` pattern (bounded).

    Only the directory right before the wildcard is listed, so the scan is
    individually bounded to the static prefix (e.g. ``C:\\Users``), never a
    recursive walk.
    """
    pattern_comps = pattern.split(os.sep)
    if _WILD not in pattern_comps:
        return []
    idx = pattern_comps.index(_WILD)
    prefix = os.sep.join(pattern_comps[:idx])
    suffix = pattern_comps[idx + 1 :]
    base = Path(prefix)
    if not base.is_dir():
        return []
    concretes: list[Path] = []
    try:
        for entry in base.iterdir():
            concretes.append(Path(os.sep.join([str(entry), *suffix])))
    except OSError:
        return concretes
    return concretes
