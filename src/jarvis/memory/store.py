"""The concrete :class:`~jarvis.memory.base.MemoryBackend` (Phase 8).

:class:`SqliteMemory` is what ``AppContext.memory`` finally receives, so the
``memory_retrieve`` node stops short-circuiting and ``memory_save`` has
somewhere to write.  It composes the three stores (:mod:`jarvis.memory.skills`,
:mod:`jarvis.memory.failures`, :mod:`jarvis.memory.prefs`) behind the small
protocol from :mod:`jarvis.memory.base` and adds the operational wrapper the
agent needs:

* **thread-safe.**  The daemon's worker thread and the IPC threads share one
  connection, so every statement is serialised by a re-entrant lock.
* **fail soft.**  A memory problem must never fail a user task: every public
  method logs and degrades (retrieve returns ``[]``, remember does nothing)
  instead of raising into the graph.
* **read-only in ``dry_run``.**  ``--dry-run`` must not create a database, so
  the factory refuses to open the file and the graph gets
  :class:`~jarvis.memory.base.NullMemory`.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Self

from jarvis.memory import embeddings
from jarvis.memory.base import MemoryRecord, NullMemory
from jarvis.memory.db import open_db
from jarvis.memory.failures import FAILURE_SIMILARITY, FailureStore
from jarvis.memory.prefs import PreferenceStore
from jarvis.memory.skills import SkillStore
from jarvis.memory.tasklog import TaskLogStore

__all__ = ["SqliteMemory", "open_memory"]


class SqliteMemory:
    """SQLite-backed memory: skills, failures and preferences in one file."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        settings: Any | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._conn = conn
        self._settings = settings
        self._log = logger or logging.getLogger("memory")
        self._lock = threading.RLock()
        memory = getattr(settings, "memory", None)
        threshold = float(getattr(memory, "similarity_threshold", embeddings.DEFAULT_SIMILARITY))
        self.skills = SkillStore(
            conn,
            similarity_threshold=threshold,
            max_rows=int(getattr(memory, "max_skills", 500)),
        )
        self.failures = FailureStore(
            conn,
            max_rows=int(getattr(memory, "max_failures", 200)),
            similarity_threshold=float(
                getattr(memory, "failure_similarity_threshold", FAILURE_SIMILARITY)
            ),
        )
        self.preferences = PreferenceStore(
            conn, max_rows=int(getattr(memory, "max_preferences", 50))
        )
        self.task_log = TaskLogStore(conn, max_rows=int(getattr(memory, "max_task_log", 500)))

    # ------------------------------------------------------- MemoryBackend

    def retrieve(self, query: str, limit: int = 3) -> list[MemoryRecord]:
        """Relevant skills, then preferences, then relevant recent failures.

        Ordering is deliberate: a verified earlier *approach* is the most
        useful thing for the planner, a preference is stable context, and a
        failure is a "prefer another approach" hint worth the least prompt
        space.  The list is capped at ``limit`` here as well as in the node, so
        no caller can exceed it.

        Every source is *scoped to ``query``**.  Failures used to be the most
        recent N regardless of the request, which meant one remembered failure
        was injected into every prompt and could refuse an unrelated command;
        they are now filtered by goal similarity like the skills.
        """
        if limit <= 0 or not (query or "").strip():
            return []
        records: list[MemoryRecord] = []
        with self._lock:
            records.extend(self._skill_records(query, limit))
            records.extend(self._preference_records(limit - len(records)))
            records.extend(self._failure_records(query, limit - len(records)))
        return records[:limit]

    def remember(self, record: MemoryRecord) -> None:
        """Store one record according to its ``kind`` (used by the graph/tests).

        ``session`` records are intentionally *not* persisted: they belong to
        one conversation and the checkpoint already holds them.  Writing them
        would grow the database with text the user never asked us to keep.
        """
        if record.kind == "session":
            return
        with self._lock:
            if record.kind == "preference":
                for key, value in _pairs(record):
                    try:
                        self.preferences.set(key, value)
                    except ValueError as exc:
                        self._log.info("preference not stored: %s", exc)
                return
            self._remember_episodic(record)

    # ---------------------------------------------------------------- metrics

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
        """Upsert this task's telemetry row; a failure never fails a task.

        ``status="running"`` opens the row (``intake``), anything else closes
        it (``respond``).  Both calls merge into the same ``task_id`` row, so a
        task that paused on a confirmation keeps the start time it really had
        instead of restarting the clock on resume.
        """
        if not task_id:
            return
        with self._lock:
            try:
                if status == "running":
                    self.task_log.start(task_id, source=source, user_input=user_input)
                else:
                    self.task_log.finish(
                        task_id,
                        status=status,
                        source=source,
                        user_input=user_input,
                        api_calls=api_calls,
                        tokens=tokens,
                        steps=steps,
                        replans=replans,
                    )
            except sqlite3.Error as exc:
                self._log.warning("task telemetry not recorded: %s", exc)

    # ------------------------------------------------------------- helpers

    def _remember_episodic(self, record: MemoryRecord) -> None:
        meta = record.meta or {}
        if meta.get(_FAILURE_META_KEY) is not None:
            step = meta.get("step") if isinstance(meta.get("step"), dict) else {}
            state = meta.get("step_state") if isinstance(meta.get("step_state"), dict) else None
            self.failures.record(str(meta.get("goal_text") or ""), step, record.text, state=state)
            return
        tools = meta.get("tools")
        self.skills.save(
            str(meta.get("goal_text") or record.text),
            plan=meta.get("plan") or {},
            tools_used=[str(t) for t in tools] if isinstance(tools, list) else None,
        )

    def _skill_records(self, query: str, limit: int) -> list[MemoryRecord]:
        if limit <= 0:
            return []
        try:
            matches = self.skills.find(query, limit=limit)
        except sqlite3.Error as exc:
            self._log.warning("skill retrieval failed: %s", exc)
            return []
        return [
            MemoryRecord(
                kind="episodic",
                text=skill.as_memory_text(),
                meta={
                    "skill_id": skill.id,
                    "goal_text": skill.goal_text,
                    "tools": skill.tools_used,
                    "success_count": skill.success_count,
                },
            )
            for skill in matches
        ]

    def _preference_records(self, limit: int) -> list[MemoryRecord]:
        if limit <= 0:
            return []
        try:
            prefs = self.preferences.all(limit=limit)
        except sqlite3.Error as exc:
            self._log.warning("preference retrieval failed: %s", exc)
            return []
        return [
            MemoryRecord(kind="preference", text=pref.as_memory_text(), meta={"key": pref.key})
            for pref in prefs
        ]

    def _failure_records(self, query: str, limit: int) -> list[MemoryRecord]:
        if limit <= 0:
            return []
        try:
            related = self.failures.relevant(query, limit=limit)
        except sqlite3.Error as exc:
            self._log.warning("failure retrieval failed: %s", exc)
            return []
        return [
            MemoryRecord(
                kind="episodic",
                text=failure.as_memory_text(),
                meta={"failure_id": failure.id, "tool": failure.step.get("tool", "")},
            )
            for failure in related
        ]

    def close(self) -> None:
        """Close the connection (idempotent)."""
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:  # pragma: no cover - close rarely fails
                pass

    def counts(self) -> dict[str, int]:
        """Row counts per table, for ``jarvis doctor`` / ``jarvis skills``.

        Learned memory only, and only what the planner can still see: the
        ``task_log`` is telemetry rather than something the user "learned", so
        it is reported through :meth:`task_metrics` instead, and quarantined
        failures are reported separately by :meth:`quarantined_failures`.
        """
        with self._lock:
            return {
                "skills": self.skills.count(),
                "failures": self.failures.count(),
                "preferences": self.preferences.count(),
            }

    def quarantined_failures(self) -> int:
        """Failure rows kept on disk but withheld from the planner."""
        with self._lock:
            try:
                return self.failures.quarantined_count()
            except sqlite3.Error as exc:
                self._log.warning("quarantined failure count unreadable: %s", exc)
                return 0

    def task_metrics(self) -> Any:
        """Aggregate task telemetry, or ``None`` when it cannot be read."""
        with self._lock:
            try:
                return self.task_log.summary()
            except sqlite3.Error as exc:
                self._log.warning("task telemetry not readable: %s", exc)
                return None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


#: Key that marks a ``MemoryRecord`` as belonging in the failure log rather
#: than the skill table when it is passed back to :meth:`SqliteMemory.remember`
#: (round-tripping a retrieved record through ``remember`` must not turn a
#: "this failed before" hint into a reusable skill).
_FAILURE_META_KEY = "failure_id"


def _pairs(record: MemoryRecord) -> list[tuple[str, str]]:
    """Read preference key/values out of a record's ``meta``.

    Accepts either ``meta={"key": ..., "value": ...}`` or a
    ``meta={"preferences": {...}}`` mapping, so a caller (or a future tool) does
    not have to know which shape the node happened to produce.
    """
    meta = record.meta or {}
    if isinstance(meta.get("preferences"), dict):
        return [(str(k), str(v)) for k, v in meta["preferences"].items()]
    key, value = meta.get("key"), meta.get("value")
    if key and value is not None:
        return [(str(key), str(value))]
    return []


def open_memory(
    path: str | Path | None = None,
    *,
    settings: Any | None = None,
    logger: logging.Logger | None = None,
    dry_run: bool = False,
) -> Any:
    """Build the memory backend for this run, or :class:`NullMemory`.

    Returns ``NullMemory`` (never raises) when memory is disabled, when the
    caller is in ``dry_run`` (writing a database for a rehearsal would be a
    side effect), or when the store cannot be opened - a corrupt or
    read-only ``memory.db`` must degrade to "no memory", not break the daemon.
    """
    log = logger or logging.getLogger("memory")
    if settings is not None and not bool(
        getattr(getattr(settings, "memory", None), "enabled", True)
    ):
        log.info("memory disabled by settings; running without a memory store")
        return NullMemory()
    if dry_run:
        log.info("dry-run: not opening the memory store")
        return NullMemory()

    if path is None:
        from jarvis import config as jarvis_config

        path = jarvis_config.memory_db()
    try:
        conn = open_db(path)
    except (OSError, sqlite3.Error) as exc:
        log.warning("could not open memory store at %s: %s", path, exc)
        return NullMemory()
    return SqliteMemory(conn, settings=settings, logger=log)
