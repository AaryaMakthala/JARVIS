"""Task telemetry: one durable, per-task metrics row (``task_log``).

``docs/01_PROJECT_SPEC.md`` (Observability) requires "JSON-lines log, task log
table, per-task API-call and token counts", and both the Phase 8 acceptance
criterion ("API calls per task drop or stay equal (measured)") and the Phase 9
benchmark metrics depend on it.  The ``brain`` and ``replan`` nodes already
accumulate ``api_calls``/``tokens`` in ``AgentState``; this module is where that
number stops being in-memory-only and becomes measurable after the process
exits.

What is stored is deliberately narrow - **counters, not content**:

======================  =====================================================
Column                  Meaning
======================  =====================================================
``task_id``             primary key; ``intake`` and ``respond`` upsert the
                        same id, so a task is one row even across a
                        confirmation resume
``status``              one of :data:`STATUSES` (see
                        :func:`jarvis.agent.nodes.telemetry.task_status`)
``api_calls``/``tokens`` LLM calls and prompt+completion tokens spent
``steps``/``replans``   planned steps and replans used (the verify/replan loop)
``user_input``          the user's command text, truncated and redacted
``duration_ms``         start -> finish, computed here from the stored
                        ``started_at`` so no extra state key is needed
======================  =====================================================

Safety: ``user_input`` is the only free text, and it is third-party text (the
user can paste a credential).  It goes through the same
:func:`~jarvis.memory.db.redact_secrets` guard as the failure log, and a row
whose text still looks like a live credential is replaced wholesale rather
than stored mangled.  Nothing here is ever fed back to the planner: these rows
are for humans and for the benchmark, not for retrieval.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from typing import Any

from jarvis.memory.db import insert_task_log, looks_secret, now_iso, prune_task_log, redact_secrets

__all__ = [
    "MAX_INPUT_CHARS",
    "STATUSES",
    "TaskLogStore",
    "TaskRecord",
    "TaskSummary",
]

_LOG = logging.getLogger("memory")

#: The user's command text is a hint for a human reading the log, not an
#: archive.  Benchmark tasks are one line; a pasted paragraph is truncated.
MAX_INPUT_CHARS = 300

#: Closed status vocabulary.  Deliberately *not* free text: a status is
#: queried and grouped (benchmark rollups, ``jarvis status``), so a typo
#: cannot invent a new bucket.  ``expired`` is reserved for the daemon's
#: stale-confirmation sweep (docs/11 section on restart) and is accepted here
#: so that sweep can write it even though the graph never produces it.
STATUSES = (
    "running",
    "completed",
    "answered",
    "partial",
    "failed",
    "halted",
    "cancelled",
    "expired",
)

_REDACTED_INPUT = "[redacted]"

#: Cap on how many rows any single read returns, so a report built from this
#: store cannot grow with the store.
_LIST_CAP = 5


@dataclass(frozen=True)
class TaskRecord:
    """One task's metrics row."""

    task_id: str
    source: str = ""
    user_input: str = ""
    status: str = "running"
    api_calls: int = 0
    tokens: int = 0
    steps: int = 0
    replans: int = 0
    started_at: str = ""
    finished_at: str = ""
    duration_ms: int | None = None
    peak_rss_mb: float | None = None

    @property
    def finished(self) -> bool:
        """True once the row carries a terminal status and a finish time."""
        return bool(self.finished_at) and self.status != "running"


@dataclass(frozen=True)
class TaskSummary:
    """Aggregate counts over the retained rows (the benchmark's raw material)."""

    tasks: int = 0
    completed: int = 0
    failed: int = 0
    unfinished: int = 0
    api_calls: int = 0
    tokens: int = 0
    replans: int = 0

    @property
    def api_calls_per_task(self) -> float:
        """Mean LLM calls per task - the metric claim C3 compares across runs."""
        return (self.api_calls / self.tasks) if self.tasks else 0.0


class TaskLogStore:
    """Bounded, thread-serialised access to the ``task_log`` table.

    The connection is shared with the rest of
    :class:`~jarvis.memory.store.SqliteMemory`; the caller's lock covers it, so
    this class does not add a second one.
    """

    def __init__(self, conn: sqlite3.Connection, *, max_rows: int = 500) -> None:
        self._conn = conn
        self._max_rows = int(max_rows)

    def start(self, task_id: str, *, source: str = "", user_input: str = "") -> None:
        """Open (or re-open) the ``running`` row for ``task_id``.

        Idempotent per id: calling it twice for one task keeps a single row
        with the earliest ``started_at``, so a resume cannot reset the clock
        and inflate the measured duration.
        """
        self._write(
            task_id=task_id,
            source=source,
            user_input=user_input,
            status="running",
            started_at=now_iso(),
        )

    def finish(
        self,
        task_id: str,
        *,
        status: str,
        api_calls: int = 0,
        tokens: int = 0,
        steps: int = 0,
        replans: int = 0,
        source: str = "",
        user_input: str = "",
    ) -> None:
        """Upsert the terminal row, keeping the ``started_at`` already stored.

        ``status`` is validated against :data:`STATUSES` and an unknown value
        becomes ``"failed"``: an unrecognised status must never be reported as
        a success.
        """
        if status not in STATUSES or status == "running":
            _LOG.warning("task_log: unusable status %r for %s; recording failure", status, task_id)
            status = "failed"
        finished_at = now_iso()
        self._write(
            task_id=task_id,
            source=source,
            user_input=user_input,
            status=status,
            api_calls=api_calls,
            tokens=tokens,
            steps=steps,
            replans=replans,
            started_at=None,  # keep whatever intake recorded
            finished_at=finished_at,
        )

    # ------------------------------------------------------------- reading

    def get(self, task_id: str) -> TaskRecord | None:
        """One row by id, or ``None``."""
        row = self._conn.execute("SELECT * FROM task_log WHERE task_id = ?", (task_id,)).fetchone()
        return self._to_record(row) if row is not None else None

    def recent(self, limit: int = 20, status: str | None = None) -> list[TaskRecord]:
        """Most recently started rows, newest first."""
        if limit <= 0:
            return []
        if status:
            rows = self._conn.execute(
                "SELECT * FROM task_log WHERE status = ? ORDER BY started_at DESC, task_id DESC"
                " LIMIT ?",
                (status, limit),
            )
        else:
            rows = self._conn.execute(
                "SELECT * FROM task_log ORDER BY started_at DESC, task_id DESC LIMIT ?",
                (limit,),
            )
        return [self._to_record(row) for row in rows]

    def summary(self) -> TaskSummary:
        """Aggregate the retained rows into benchmark-ready totals.

        Only *finished* rows contribute to the success split, so a task that
        crashed (still ``running``) is counted as unfinished rather than
        silently inflating the completion rate.
        """
        row = self._conn.execute(
            """
            SELECT COUNT(*) AS tasks,
                   COALESCE(SUM(api_calls), 0) AS api_calls,
                   COALESCE(SUM(tokens), 0) AS tokens,
                   COALESCE(SUM(replans), 0) AS replans
            FROM task_log
            """
        ).fetchone()
        tasks = int(row["tasks"]) if row is not None else 0
        completed = self._count(("completed", "answered", "partial"))
        failed = self._count(("failed", "halted", "cancelled", "expired"))
        return TaskSummary(
            tasks=tasks,
            completed=completed,
            failed=failed,
            unfinished=max(tasks - completed - failed, 0),
            api_calls=int(row["api_calls"]) if row is not None else 0,
            tokens=int(row["tokens"]) if row is not None else 0,
            replans=int(row["replans"]) if row is not None else 0,
        )

    def count(self) -> int:
        """Number of retained rows."""
        return int(self._conn.execute("SELECT COUNT(*) FROM task_log").fetchone()[0])

    def failure_repeats(self, limit: int = _LIST_CAP) -> list[tuple[str, int]]:
        """Commands that failed more than once, worst first.

        The self-review's "repeated failures" signal (docs/04_TOOLS_SPEC.md
        section 8): a command that has failed three times is a real problem
        the user should see, while three different commands that each failed
        once is noise.  The command text is returned as stored - already
        truncated and redacted by :func:`safe_input`.
        """
        if limit <= 0:
            return []
        rows = self._conn.execute(
            """
            SELECT user_input, COUNT(*) AS hits
            FROM task_log
            WHERE status IN ('failed', 'halted') AND user_input <> ''
            GROUP BY user_input
            HAVING hits > 1
            ORDER BY hits DESC, user_input ASC
            LIMIT ?
            """,
            (limit,),
        )
        return [(str(row["user_input"]), int(row["hits"])) for row in rows]

    def clear(self) -> int:
        """Delete every row (used by ``jarvis reset --memory``)."""
        cur = self._conn.execute("DELETE FROM task_log")
        return int(cur.rowcount or 0)

    # ------------------------------------------------------------- writing

    def _count(self, statuses: tuple[str, ...]) -> int:
        placeholders = ", ".join("?" for _ in statuses)
        row = self._conn.execute(
            f"SELECT COUNT(*) FROM task_log WHERE status IN ({placeholders})",
            statuses,
        ).fetchone()
        return int(row[0]) if row else 0

    def _write(
        self,
        *,
        task_id: str,
        source: str = "",
        user_input: str = "",
        status: str,
        api_calls: int = 0,
        tokens: int = 0,
        steps: int = 0,
        replans: int = 0,
        started_at: str | None = None,
        finished_at: str | None = None,
    ) -> None:
        """Merge one write into the existing row for ``task_id``.

        Read-modify-write on purpose: ``intake`` records the start and
        ``respond`` the outcome minutes later, and ``INSERT OR REPLACE`` would
        have the second call drop the first call's ``started_at``.
        """
        if not task_id:
            return
        previous = self.get(task_id)
        insert_task_log(
            self._conn,
            {
                "task_id": task_id,
                "source": source or (previous.source if previous else ""),
                "user_input": safe_input(user_input) or (previous.user_input if previous else ""),
                "status": status,
                "api_calls": api_calls,
                "tokens": tokens,
                "steps": steps,
                "replans": replans,
                "started_at": started_at or (previous.started_at if previous else "") or now_iso(),
                "finished_at": finished_at or (previous.finished_at if previous else ""),
                "duration_ms": _duration_ms(
                    started_at or (previous.started_at if previous else ""),
                    finished_at or (previous.finished_at if previous else ""),
                ),
                "peak_rss_mb": _current_rss_mb(),
            },
        )
        prune_task_log(self._conn, self._max_rows)

    @staticmethod
    def _to_record(row: Any) -> TaskRecord:
        return TaskRecord(
            task_id=str(row["task_id"]),
            source=str(row["source"] or ""),
            user_input=str(row["user_input"] or ""),
            status=str(row["status"] or "running"),
            api_calls=int(row["api_calls"] or 0),
            tokens=int(row["tokens"] or 0),
            steps=int(row["steps"] or 0),
            replans=int(row["replans"] or 0),
            started_at=str(row["started_at"] or ""),
            finished_at=str(row["finished_at"] or ""),
            duration_ms=None if row["duration_ms"] is None else int(row["duration_ms"]),
            peak_rss_mb=None if row["peak_rss_mb"] is None else float(row["peak_rss_mb"]),
        )


# --------------------------------------------------------------- helpers


def safe_input(text: str) -> str:
    """Bounded, credential-free form of the user's command text.

    ``docs/03_SECURITY_AND_POLICY.md`` section 5 sanctions keeping the command
    text (the benchmark and the audit self-review both need it) but the user
    can paste a key into the prompt, so the value is redacted before it is
    stored and dropped entirely if anything still looks like a credential.
    """
    cleaned = redact_secrets((text or "").strip())[:MAX_INPUT_CHARS]
    if looks_secret(cleaned):
        return _REDACTED_INPUT
    return cleaned


def _duration_ms(started_at: str, finished_at: str) -> int | None:
    """Elapsed milliseconds between two stored ISO stamps, or ``None``.

    ``None`` when either end is missing or unparseable - an unknown duration
    is honest, a zero would read as "instant".
    """
    if not started_at or not finished_at:
        return None
    try:
        from datetime import datetime

        start = datetime.fromisoformat(started_at)
        end = datetime.fromisoformat(finished_at)
    except ValueError:
        return None
    return max(int((end - start).total_seconds() * 1000), 0)


def _current_rss_mb() -> float | None:
    """Process RSS in MB, or ``None`` when ``psutil`` is unavailable.

    Best effort by design: the row is telemetry, so a missing number must not
    stop the write.
    """
    try:
        import psutil
    except ImportError:  # pragma: no cover - psutil is a hard dependency
        return None
    try:
        rss = int(psutil.Process().memory_info().rss)
    except Exception:  # noqa: BLE001 - telemetry must never break a task
        return None
    return round(rss / (1024 * 1024), 1)
