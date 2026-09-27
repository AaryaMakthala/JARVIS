"""Failure memory: what went wrong, so retries and replans stop repeating it.

Phase 8 of ``docs/05_BUILD_PLAN.md`` asks for a failure log that informs retry
decisions.  Records are **examples of past failures**, surfaced to the planner
alongside skills but always labelled as failures, so the model can avoid the
approach that did not work.  They are never instructions and never executed.

The same anti-secret guard as :mod:`jarvis.memory.skills` applies: a failure
message that looks like it contains a credential is stored with the offending
text replaced rather than persisted verbatim.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from jarvis.memory.db import looks_secret, now_iso, prune_failures, redact_secrets, safe_plan

__all__ = ["MAX_ERROR_CHARS", "Failure", "FailureStore"]

#: Failures come from tool error strings, which can be arbitrarily long.  A
#: remembered failure is a hint, not an archive.
MAX_ERROR_CHARS = 300


@dataclass(frozen=True)
class Failure:
    """One recorded failed step."""

    id: int
    goal_text: str
    step: dict[str, Any]
    error: str
    created_at: str = ""

    def as_memory_text(self) -> str:
        """Bounded, honest sentence for the planner prompt."""
        tool = str(self.step.get("tool") or "a tool")
        goal = self.goal_text or "a similar request"
        error = (self.error or "it failed").strip()[:MAX_ERROR_CHARS]
        return (
            f'Earlier attempt to "{goal}" failed at {tool}: {error}. Do not repeat that approach.'
        )


class FailureStore:
    """Append-only log of failed steps, bounded by an LRU-ish cap."""

    def __init__(self, conn: sqlite3.Connection, *, max_rows: int = 200) -> None:
        self._conn = conn
        self._max_rows = int(max_rows)

    def record(
        self,
        goal_text: str,
        step: dict[str, Any] | None = None,
        error: str = "",
    ) -> Failure | None:
        """Append one failure; ``None`` when the entry is empty and skipped."""
        # Truncate first (the text is unbounded), then redact.  A tool error is
        # untrusted third-party text: "login failed for hunter2, password
        # hunter2" must not become a durable row on disk.
        message = redact_secrets((error or "").strip())[:MAX_ERROR_CHARS]
        if not message:
            return None
        safe_step = safe_plan(step or {})
        goal = (goal_text or "").strip()[:200]
        if looks_secret(goal):
            goal = "[redacted goal]"
        cursor = self._conn.execute(
            "INSERT INTO failures (goal_text, step_json, error, created_at) VALUES (?, ?, ?, ?)",
            (goal, json.dumps(safe_step, ensure_ascii=False), message, now_iso()),
        )
        prune_failures(self._conn, self._max_rows)
        return Failure(
            id=int(cursor.lastrowid or 0),
            goal_text=goal,
            step=safe_step,
            error=message,
            created_at=now_iso(),
        )

    def count(self) -> int:
        """Number of stored failures."""
        return int(self._conn.execute("SELECT COUNT(*) FROM failures").fetchone()[0])

    def recent(self, limit: int = 3, goal_text: str | None = None) -> list[Failure]:
        """Most recent failures, optionally restricted to one goal substring."""
        if limit <= 0:
            return []
        if goal_text:
            rows = self._conn.execute(
                "SELECT * FROM failures WHERE goal_text LIKE ? ORDER BY id DESC LIMIT ?",
                (f"%{goal_text}%", limit),
            )
        else:
            rows = self._conn.execute("SELECT * FROM failures ORDER BY id DESC LIMIT ?", (limit,))
        return [self._to_failure(row) for row in rows]

    def repeated_tools(self, goal_text: str, limit: int = 5) -> list[str]:
        """Tool names that already failed for ``goal_text`` (most recent first).

        This is the signal a replan wants: "you already tried ``web_answer`` on
        this goal and it did not verify".
        """
        seen: list[str] = []
        for failure in self.recent(limit=limit * 4, goal_text=goal_text):
            tool = str(failure.step.get("tool") or "").strip()
            if tool and tool not in seen:
                seen.append(tool)
            if len(seen) >= limit:
                break
        return seen

    def clear(self) -> int:
        """Delete every failure (used by ``jarvis reset --memory``)."""
        cur = self._conn.execute("DELETE FROM failures")
        return int(cur.rowcount or 0)

    @staticmethod
    def _to_failure(row: sqlite3.Row) -> Failure:
        try:
            step = json.loads(row["step_json"] or "{}")
        except (TypeError, ValueError):
            step = {}
        return Failure(
            id=int(row["id"]),
            goal_text=str(row["goal_text"] or ""),
            step=step if isinstance(step, dict) else {},
            error=str(row["error"] or ""),
            created_at=str(row["created_at"] or ""),
        )
