"""Memory backend interfaces (light integration phase).

JARVIS keeps three kinds of local, non-secret memory: ``session`` (within one
conversation), ``episodic`` (what was done earlier and whether it worked) and
``preference`` (the user's stable choices).  Phase 8 implements the embedding
backends; this module defines the boundary every backend must satisfy plus a
deterministic :class:`NullMemory` so the graph runs without a store.

Invariant: **no secret material is ever stored in memory** (docs/03 §5).  The
retrieval interfaces take/return :class:`MemoryRecord` only; telemetry is a
separate :meth:`MemoryBackend.record_task` sink that carries counters (and a
redacted copy of the command) and is never handed to the model.
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

    def record_task(
        self,
        task_id: str,
        *,
        status: str = "running",
        source: str = "",
        user_input: str = "",
        api_calls: int = 0,
        tokens: int = 0,
        steps: int = 0,
        replans: int = 0,
    ) -> None:
        """Store per-task telemetry counters (no-op for read-only backends).

        Separate from :meth:`remember` on purpose.  A ``MemoryRecord`` is
        retrieved by the planner, so it must stay a small, verified,
        non-tainted *example*; a task row is a counter (calls, tokens, steps,
        replans) that is never fed back to the model and is only read by
        humans, the audit self-review and the benchmark.

        ``status="running"`` opens the row; any other status finishes it, so
        one task is one row even across a confirmation resume.
        """


class NullMemory:
    """Deterministic no-op backend: never returns anything, never stores."""

    def retrieve(self, query: str, limit: int = 3) -> list[MemoryRecord]:
        del query, limit
        return []

    def remember(self, record: MemoryRecord) -> None:
        del record  # discarded: without a backend there is no memory to write

    def record_task(
        self,
        task_id: str,
        *,
        status: str = "running",
        source: str = "",
        user_input: str = "",
        api_calls: int = 0,
        tokens: int = 0,
        steps: int = 0,
        replans: int = 0,
    ) -> None:
        """No-op: with no store there is nowhere to record telemetry."""
        del task_id, status, source, user_input, api_calls, tokens, steps, replans
