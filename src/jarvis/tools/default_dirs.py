"""Default save folder and folder aliases (Step D1).

Turns whatever location the planner put in a path argument into a concrete
path, deterministically and without an LLM.  This module does **not** touch
``allowed_roots`` (the policy engine enforces that later) and never mutates
config.

Resolution order for :func:`resolve_location`:

a. an absolute path is used as given;
b. a bare name with no separator goes under the default save dir;
c. a path whose first segment (case-insensitive) is a known alias -- the
   built-ins ``desktop``/``documents``/``downloads``/``pictures`` (Windows
   Known Folders) or a ``[paths.aliases]`` entry -- resolves under that alias
   root; aliases win over a real folder of the same name;
d. any other relative path with a separator resolves under the default dir
   only if its first segment is an existing directory there; otherwise the
   caller is told to *ask which folder* (never guessed, never created).

The Windows Known Folder lookup is lazy and platform-guarded, so this module
imports cleanly on Linux/WSL.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jarvis.platform_guard import is_windows

__all__ = [
    "LocationResult",
    "default_save_dir",
    "known_folder",
    "location_label",
    "resolve_location",
]

#: Standard KNOWNFOLDERID GUIDs (RFC 4122 text form).
_FOLDER_IDS: dict[str, str] = {
    "desktop": "{B4BFCC3A-DB2C-424C-B029-7FE99A87C641}",
    "documents": "{FDD39AD0-238F-46AF-ADB4-6C85480369C7}",
    "downloads": "{374DE290-123F-4565-9164-39C4925E467B}",
    "pictures": "{33E28130-4E1E-4676-835A-98395C3BC3BB}",
}

#: Display fallback directory name per known folder (used off-Windows and when
#: the Windows API is unavailable).
_CANONICAL: dict[str, str] = {
    "desktop": "Desktop",
    "documents": "Documents",
    "downloads": "Downloads",
    "pictures": "Pictures",
}

_BUILTIN_ALIASES: tuple[str, ...] = ("desktop", "documents", "downloads", "pictures")

_DISPLAY: dict[str, str] = {
    "desktop": "your Desktop",
    "documents": "your Documents",
    "downloads": "your Downloads",
    "pictures": "your Pictures",
}


@dataclass(frozen=True)
class LocationResult:
    """Outcome of resolving a planner-supplied location.

    ``kind`` is ``"resolved"`` (use ``path``) or ``"clarify"`` (ask the user;
    ``path`` is ``None`` and ``reason`` holds the question hint).
    """

    kind: str
    path: Path | None
    label: str = ""
    reason: str = ""


def known_folder(name: str) -> Path | None:
    """Return a Windows Known Folder path, or ``None`` for an unknown name.

    On non-Windows returns ``Path.home() / <Canonical>``.  On Windows calls
    ``SHGetKnownFolderPath`` (through ``ctypes``, freed with
    ``CoTaskMemFree``); if that fails it falls back to
    ``%USERPROFILE%\\<Canonical>``.  The API is never required to import.
    """
    key = str(name or "").strip().lower()
    canonical = _CANONICAL.get(key)
    if canonical is None:
        return None

    if not is_windows():
        return Path.home() / canonical

    path = _shget_known_folder(_FOLDER_IDS[key])
    if path is not None:
        return path
    base = os.environ.get("USERPROFILE") or str(Path.home())
    return Path(base) / canonical


def _shget_known_folder(folder_id: str) -> Path | None:
    """Call ``SHGetKnownFolderPath`` via ctypes; ``None`` on any failure."""
    try:
        import ctypes

        class _GUID(ctypes.Structure):
            _fields_ = (
                ("Data1", ctypes.c_uint32),
                ("Data2", ctypes.c_uint16),
                ("Data3", ctypes.c_uint16),
                ("Data4", ctypes.c_ubyte * 8),
            )

        guid = _GUID()
        ole32 = ctypes.windll.ole32
        shell32 = ctypes.windll.shell32
        if ole32.CLSIDFromString(ctypes.c_wchar_p(folder_id), ctypes.byref(guid)) != 0:
            return None
        ptr = ctypes.c_wchar_p()
        if shell32.SHGetKnownFolderPath(ctypes.byref(guid), 0, None, ctypes.byref(ptr)) != 0:
            return None
        try:
            value = ptr.value
        finally:
            ole32.CoTaskMemFree(ctypes.cast(ptr, ctypes.c_void_p))
        return Path(value) if value else None
    except Exception:  # noqa: BLE001 - the API is best effort; caller falls back
        return None


def default_save_dir(settings: Any) -> Path:
    """Return the folder a bare filename goes to.

    ``settings.paths.default_save_dir`` when set, else the user's real Desktop
    (Known Folder).  Falls back to ``Path.home()/Desktop`` if even that fails.
    """
    raw = getattr(getattr(settings, "paths", None), "default_save_dir", "") or ""
    if str(raw).strip():
        return Path(os.path.expandvars(os.path.expanduser(str(raw))))
    return known_folder("desktop") or (Path.home() / "Desktop")


def resolve_location(value: str, settings: Any) -> LocationResult:
    """Resolve a planner-supplied location string (rules a-d in the docstring)."""
    raw = str(value or "").strip()
    if not raw:
        return LocationResult("clarify", None, reason="which folder?")

    expanded = os.path.expandvars(os.path.expanduser(raw))
    candidate = Path(expanded)
    if candidate.is_absolute():
        return LocationResult("resolved", candidate, location_label(candidate, settings))

    parts = _split_parts(raw)
    first = parts[0].lower() if parts else ""
    aliases = _alias_roots(settings)
    if first and first in aliases:
        root = aliases[first]
        path = root.joinpath(*parts[1:]) if len(parts) > 1 else root
        return LocationResult("resolved", path, location_label(path, settings))

    default = default_save_dir(settings)
    if not _has_separator(raw):
        path = default / raw
        return LocationResult("resolved", path, location_label(path, settings))

    if parts and (default / parts[0]).is_dir():
        path = default.joinpath(*parts)
        return LocationResult("resolved", path, location_label(path, settings))
    return LocationResult("clarify", None, reason="which folder?")


def location_label(path: Path, settings: Any) -> str:
    """Human label for a resolved path: ``"your Desktop"``, alias name, parent."""
    target = _key(path)
    for name in _BUILTIN_ALIASES:
        root = known_folder(name)
        if root is not None and _under(target, _key(root)):
            return _DISPLAY[name]

    custom = getattr(getattr(settings, "paths", None), "aliases", {}) or {}
    for name, value in custom.items():
        root = _key(Path(os.path.expandvars(os.path.expanduser(str(value)))))
        if _under(target, root):
            return str(name)

    parent_name = Path(path).parent.name
    return parent_name or str(Path(path).parent)


def _alias_roots(settings: Any) -> dict[str, Path]:
    """Lower-cased alias name -> root.  Built-ins first, config aliases add/override."""
    roots: dict[str, Path] = {}
    for name in _BUILTIN_ALIASES:
        folder = known_folder(name)
        if folder is not None:
            roots[name] = folder
    custom = getattr(getattr(settings, "paths", None), "aliases", {}) or {}
    for name, value in custom.items():
        roots[str(name).lower()] = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return roots


def _split_parts(raw: str) -> list[str]:
    """Split a relative path into non-empty segments (both separators)."""
    normalised = raw.replace("\\", "/")
    return [part for part in normalised.split("/") if part not in ("", ".")]


def _has_separator(raw: str) -> bool:
    return "/" in raw or "\\" in raw


def _key(path: Path | str) -> str:
    return os.path.normcase(str(path))


def _under(target: str, root: str) -> bool:
    return target == root or target.startswith(root.rstrip(os.sep) + os.sep)
