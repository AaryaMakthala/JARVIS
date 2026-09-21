"""Shared doctor ``Check`` result type (CLI ``doctor`` and ``voice doctor``).

Kept in its own module so ``jarvis.cli`` and ``jarvis.voice.doctor`` can both
build check tables without importing the other (the voice module must stay
importable without dragging the CLI's typer/langgraph imports into the
daemon process).
"""

from __future__ import annotations

from dataclasses import dataclass

_CHECK_STATUS = ("PASS", "WARN", "FAIL")


@dataclass(frozen=True)
class Check:
    """Single doctor check result."""

    name: str
    status: str  # one of _CHECK_STATUS
    detail: str


def _check(name: str, status: str, detail: str) -> Check:
    if status not in _CHECK_STATUS:
        raise ValueError(f"unknown check status {status!r}")
    return Check(name, status, detail)
