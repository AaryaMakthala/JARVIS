"""Memory backend interfaces (light integration phase).

JARVIS keeps three kinds of local, non-secret memory: ``session`` (within one
conversation), ``episodic`` (what was done earlier and whether it worked) and
``preference`` (the user's stable choices).  Phase 8 implements the embedding
backends; this module defines the boundary every backend must satisfy plus a
deterministic :class:`NullMemory` so the graph runs without a store.

Invariant: **no secret material is ever stored in memory** (docs/03 §5).  The
interfaces take/return :class:`MemoryRecord` only.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

MemoryKind = Literal["session", "episodic", "preference"]


class MemoryRecord(BaseModel):
    """One retrievable memory fragment (never secrets)."""

    kind: MemoryKind = "session"
    text: str = Field(min_length=1)
    meta: dict = Field(default_factory=dict)


@runtime_checkable
class MemoryBackend(Protocol):
    """What a memory store must provide to the agent."""

    def retrieve(self, query: str, limit: int = 3) -> list[MemoryRecord]:
        """Return the ``limit`` most relevant records for ``query`` (bounded)."""

    def remember(self, record: MemoryRecord) -> None:
        """Store a record for later retrieval (no-op for read-only backends)."""


class NullMemory:
    """Deterministic no-op backend: never returns anything, never stores."""

    def retrieve(self, query: str, limit: int = 3) -> list[MemoryRecord]:
        del query, limit
        return []

    def remember(self, record: MemoryRecord) -> None:
        del record  # discarded: without a backend there is no memory to write
