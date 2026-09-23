"""Local skill/failure/preference memory.

Phase 6+ implements the real backends; this package exposes the shared
:class:`MemoryBackend` protocol, :class:`MemoryRecord` and the deterministic
:class:`NullMemory` used until then.
"""

from __future__ import annotations

from jarvis.memory.base import MemoryBackend, MemoryKind, MemoryRecord, NullMemory

__all__ = ["MemoryBackend", "MemoryKind", "MemoryRecord", "NullMemory"]
