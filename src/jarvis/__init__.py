"""JARVIS - a safe Windows desktop AI agent.

The LLM proposes actions; a deterministic policy engine decides; the user
confirms anything sensitive. This package implements the JARVIS agent.
"""

from __future__ import annotations

from importlib import metadata

__all__ = ["__version__"]


def _version() -> str:
    try:
        return metadata.version("jarvis-agent")
    except metadata.PackageNotFoundError:
        return "0.1.0.dev0"


__version__ = _version()
