"""Platform guard helpers.

Windows-only imports (``pywinauto``, ``win32*``, ``winreg``, ``ctypes.windll``)
must go behind this module so unit tests can import on any OS. Use
:func:`lazy_import` for heavy/optional modules instead of module-level imports.
"""

from __future__ import annotations

import functools
import importlib
import sys
import threading
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def is_windows() -> bool:
    """Return ``True`` when running on Microsoft Windows."""
    return sys.platform == "win32"


def is_python_supported() -> bool:
    """Return ``True`` for Python 3.11 or 3.12 (JARVIS' supported range)."""
    return (3, 11) <= sys.version_info[:2] < (3, 13)


def is_64bit() -> bool:
    """Return ``True`` when running a 64-bit interpreter (ML wheels require it)."""
    return sys.maxsize > 2**32


class PlatformError(RuntimeError):
    """Raised when a Windows-only operation is attempted elsewhere."""


def require_windows(context: str = "This feature") -> None:
    """Raise :class:`PlatformError` when not on Windows."""
    if not is_windows():
        raise PlatformError(f"{context} is only supported on Windows.")


def windows_only(func: F) -> F:
    """Decorate a callable so it refuses to run on non-Windows platforms."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        require_windows(func.__name__)
        return func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


class LazyImport:
    """Import a module on first attribute access (thread-safe).

    Example::

        win32 = LazyImport("win32api")
        win32.MessageBox(...)   # import happens here
    """

    def __init__(self, module_name: str) -> None:
        self._module_name = module_name
        self._module: Any = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        if self._module is None:
            with self._lock:
                if self._module is None:
                    self._module = importlib.import_module(self._module_name)
        return self._module

    def __getattr__(self, item: str) -> Any:
        return getattr(self._load(), item)
