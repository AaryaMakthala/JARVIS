"""Path safety (first version, Phase 1).

Before any policy decision or file operation every path is resolved with
:func:`resolve_safe` to an absolute, real path.  We reject:

* NUL bytes, empty/whitespace-only paths;
* alternate data streams (``file.txt:stream``) and drive-relative forms;
* UNC (``\\\\server\\share``) and device paths (``\\\\.\\``, ``\\\\?\\``);
* bare device names at the root (``CON``, ``NUL``, ``COM1`` ...).

Containment checks (allowed roots, protected roots) are then performed on the
resolved path with case-insensitive comparison, so ``..`` tricks and
case/slash variations cannot slip past.  Full symlink/junction-aware logic is
completed in Phase 2 (see docs/03_SECURITY_AND_POLICY.md section 5).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from jarvis import config

PathStr = str  #: Marked in tool args models; the engine treats it as a path.

_DRIVE_RE = re.compile(r"^[A-Za-z]:")
_DEVICE_NAMES = {"CON", "NUL", "PRN", "AUX"}
_DEVICE_NAMES |= {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}
_ILLEGAL_CHARS = ":"  # drive letters handled separately; data streams use ':'

_SPECIAL_PREFIXES = ("\\\\", r"\\.\\", "\\\\?\\")


class PathError(ValueError):
    """Raised when a path string is rejected before any operation happens."""


def resolve_safe(raw: str) -> Path:
    """Expand and absolutely resolve ``raw``, rejecting unsafe forms.

    Raises :class:`PathError` for anything that must not reach the filesystem.
    The returned :class:`~pathlib.Path` is absolute (``Path.resolve``) with
    ``~``/env vars expanded and any ``..`` collapsed.
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
    # (C:foo) are Windows constructs; reject ':' anywhere it is not the CDrive
    # letter separator on Windows.  On POSIX ':' is a legal filename character.
    if os.name == "nt":
        colon = value.find(":")
        if colon != -1 and not (colon == 1 and re.match(r"^[A-Za-z]:", value)):
            raise PathError(f"path contains an illegal ':' character: {value!r}")

    expanded = os.path.expandvars(os.path.expanduser(value))
    if not expanded:
        raise PathError(f"path expands to empty: {value!r}")

    absolute = Path(expanded)
    if not absolute.is_absolute():
        absolute = Path.cwd() / absolute
    # resolve(strict=False) collapses '..' and resolves symlinks/junctions on
    # Windows; it never touches the filesystem when strict=False.
    resolved = absolute.resolve(strict=False)

    # A bare device name right at the root is rejected (CON/NUL/... abuse).
    stem = resolved.name.upper()
    if stem in _DEVICE_NAMES:
        raise PathError(f"device name is not allowed as a path: {value!r}")
    return resolved


def normalise_key(path: Path | str) -> str:
    """Case-fold + normcase a resolved path for comparisons."""
    return os.path.normcase(str(path)).casefold()


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


def protected_roots(settings: config.Settings | None = None) -> list[Path]:
    """Return the hard-coded protected root directories (absolute).

    These can never be a direct target of a file tool regardless of config -
    in this first version protection is *exact/ancestor* (you cannot touch the
    root itself or its parents), while containment inside AppData and friends
    is Phase 2 work.  The JARVIS data directory itself is included so assets
    there cannot be overwritten by a stray file tool.
    """
    _ = settings  # reserved: later phases may read classifier settings.
    raw = [
        os.getenv("SystemRoot"),
        os.getenv("ProgramFiles"),
        os.getenv("ProgramFiles(x86)"),
        os.getenv("ProgramData"),
        os.getenv("USERPROFILE"),
        os.path.expanduser("~/.ssh"),
    ]
    roots: list[Path] = []
    for value in raw:
        if value:
            roots.append(Path(value).resolve(strict=False))
    roots.append(config.user_data_dir().resolve(strict=False))
    return roots


def is_protected(path: Path, roots: list[Path] | None = None) -> bool:
    """Return ``True`` when ``path`` is a protected root, a drive root, or an
    ancestor of a protected root (i.e. the operation would reach into one).

    ``path`` must already be resolved via :func:`resolve_safe`.
    """
    if roots is None:
        roots = protected_roots()
    key = normalise_key(path)
    # A drive root itself (C:\) is always protected.
    if key == normalise_key(path.anchor):
        return True
    for root in roots:
        root_key = normalise_key(root)
        if key == root_key:
            return True
        # path is an ancestor of a protected root -> operation reaches into it
        if root_key.startswith(key.rstrip(os.sep) + os.sep):
            return True
    return False


def roots_from_settings(settings: config.Settings) -> list[Path]:
    """Expand ``settings.policy.allowed_roots`` into resolved absolute roots."""
    return [resolve_safe(value) for value in settings.policy.allowed_roots]


def ensure_allowed_root(path: Path, roots: list[Path]) -> None:
    """Raise :class:`PathError` unless ``path`` is inside one of ``roots``."""
    if not within_any_root(path, roots):
        sample = "\n  ".join(str(r) for r in roots)
        raise PathError(f"path is outside the allowed folders (allowed roots):\n  {sample}")
