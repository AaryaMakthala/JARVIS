"""SQLite connection and schema for the local memory store (Phase 8).

One database file (``config.memory_db()``, normally
``%LOCALAPPDATA%\\jarvis\\memory.db``) holds every kind of non-secret memory:
learned *skills*, *failures*, user *preferences* and the *task_log* used by
``jarvis audit`` for its self-review section.  The schema is the one documented
in ``docs/02_ARCHITECTURE.md`` section 9.

Design rules that matter for safety:

* **Local only.** No network, no model download, no code execution.  Opening
  this file can never phone home.
* **Secrets never land here.** Callers pass already-redacted text (see
  :mod:`jarvis.memory.skills`); the store additionally refuses rows whose
  ``goal_text``/``plan_json`` contain an obvious credential, so a mistake in a
  caller fails closed instead of persisting a password.
* **Bounded.** :func:`init_schema` creates the tables; the pruning helpers
  (``prune_skills``/``prune_failures``) keep rows under the configured caps so
  the file cannot grow without limit.
* **Boring.** Plain ``sqlite3`` from the standard library, no ORM, no global
  state: every function takes the connection or opens one it owns.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "SCHEMA_VERSION",
    "connect",
    "init_schema",
    "looks_secret",
    "prune_failures",
    "prune_skills",
    "redact_secrets",
    "safe_plan",
    "safe_value",
    "secret_free_rows",
    "secret_key",
    "user_version",
]

#: Bumped when the schema changes.  ``init_schema`` applies the statements in
#: :data:`SCHEMA` (all ``IF NOT EXISTS``) and records the version in
#: ``PRAGMA user_version`` so a future migration can branch on it.  Existing
#: databases are never dropped: skills and preferences outlive upgrades.
SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS skills (
  id INTEGER PRIMARY KEY,
  goal_text TEXT NOT NULL,
  plan_json TEXT NOT NULL,
  embedding BLOB NOT NULL,
  tools_used TEXT NOT NULL,
  success_count INTEGER DEFAULT 1,
  fail_count INTEGER DEFAULT 0,
  created_at TEXT NOT NULL,
  last_used_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS failures (
  id INTEGER PRIMARY KEY,
  goal_text TEXT,
  step_json TEXT,
  error TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS preferences (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS task_log (
  task_id TEXT PRIMARY KEY,
  source TEXT,
  user_input TEXT,
  status TEXT,
  api_calls INTEGER,
  tokens INTEGER,
  steps INTEGER,
  replans INTEGER,
  started_at TEXT,
  finished_at TEXT,
  duration_ms INTEGER,
  peak_rss_mb REAL
);

CREATE INDEX IF NOT EXISTS idx_skills_last_used ON skills (last_used_at DESC);
CREATE INDEX IF NOT EXISTS idx_failures_created ON failures (created_at DESC);
"""

#: Argument/field *names* that usually carry credential material.  Matched
#: case-insensitively as a substring, so ``new_password``/``apiKey``/
#: ``user_pin`` are all caught.  Deliberately broad: a false positive only
#: costs us a remembered example or a stored preference, while a false negative
#: would persist a credential.
SECRET_KEY_HINTS = (
    "password",
    "passwd",
    "passcode",
    "secret",
    "api_key",
    "apikey",
    "token",
    "credential",
    "pin",
    "otp",
    "authorization",
    "cookie",
)

_REDACTED = "[redacted]"

#: ``<secret-ish key><separator><value>`` in JSON, TOML, query-string or CLI
#: form.  The **value** is what decides: a key that merely *looks* sensitive
#: (``{"new_password": "[redacted]"}``) is already sanitised and must not be
#: rejected, while ``{"new_password": "hunter2"}`` must be.
_SECRET_ASSIGNMENT = re.compile(
    r"[a-z0-9_.-]*(?:" + "|".join(SECRET_KEY_HINTS) + r")[a-z0-9_.-]*"
    r"[\"']?\s*(?P<sep>[:=])\s*[\"']?(?P<value>[^\"',;}\s]*)",
    re.IGNORECASE,
)


def redact_secrets(text: str) -> str:
    """Replace every credential-ish ``key=value`` pair in free text.

    Applied to any string that comes from outside our own code - tool error
    messages, goals, remembered prose.  A tool that echoes a password back in
    its error string must not be able to persist it.  The key is kept (it is
    useful context: "the *password* was wrong") and only the value is dropped.
    """
    if not text:
        return ""

    def _sub(match: re.Match[str]) -> str:
        value = match.group("value")
        if not value or value == _REDACTED:
            return match.group(0)
        return f"{match.group(0)[: match.start('value') - match.start(0)]}{_REDACTED}"

    return _SECRET_ASSIGNMENT.sub(_sub, text)


def secret_key(name: str) -> bool:
    """True when a field *name* suggests it holds a credential."""
    low = (name or "").lower()
    return any(hint in low for hint in SECRET_KEY_HINTS)


def looks_secret(text: str) -> bool:
    """True when ``text`` pairs a credential-ish name with a real value.

    Used to fail closed on writes (:func:`secret_free_rows`) and to decide
    whether a remembered example is safe to show the planner.  A redacted
    placeholder does **not** count, so redaction and rejection compose: the
    caller can redact first and still pass the guard afterwards.
    """
    if not text:
        return False
    for match in _SECRET_ASSIGNMENT.finditer(text):
        value = match.group("value")
        if value and value != _REDACTED:
            return True
    return False


def secret_free_rows(text: str, plan_json: str) -> bool:
    """True when both fields are safe to persist.

    The fail-closed guard for :func:`jarvis.memory.skills.SkillStore.save`: a
    row that looks like it carries a secret is *rejected*, not redacted, so the
    caller can report honestly instead of storing a mangled example.
    """
    return not (looks_secret(text) or looks_secret(plan_json))


def redact(value: str) -> str:
    """Replace ``value`` with a fixed marker (never logs the original)."""
    del value
    return _REDACTED


def safe_value(key: str, value: Any) -> Any:
    """Recursively strip credential material out of one plan argument.

    Handles the three shapes a plan carries: a scalar, a nested mapping, and a
    list/tuple of mappings (a planner's ``steps`` is a list, so a naive
    ``isinstance(value, dict)`` check would let a secret inside a step through -
    a bug this helper exists to prevent).
    """
    if secret_key(key):
        return _REDACTED
    if isinstance(value, dict):
        return safe_plan(value)
    if isinstance(value, (list, tuple)):
        return [safe_value("", item) for item in value]
    return value


def safe_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Copy ``plan`` with credential-looking argument values replaced.

    Redaction happens *before* :func:`secret_free_rows` and the two compose: a
    value that looks like a real credential is rejected outright, while a field
    that merely *smells* sensitive is stored as ``[redacted]``.
    """
    return {key: safe_value(key, value) for key, value in plan.items()}


def user_version(conn: sqlite3.Connection) -> int:
    """Return the schema version recorded in the database."""
    row = conn.execute("PRAGMA user_version").fetchone()
    return int(row[0]) if row else 0


def connect(path: str | Path, *, read_only: bool = False) -> sqlite3.Connection:
    """Open ``path`` with the pragmas the memory store relies on.

    * ``row_factory=sqlite3.Row`` so callers can read columns by name.
    * ``foreign_keys=ON`` (defensive; the schema has no FKs yet).
    * ``journal_mode=WAL`` for a real file so a crashed daemon cannot corrupt
      the store and so the audit reader is not blocked by a writer.
    * ``busy_timeout`` so a concurrent daemon/CLI access waits instead of
      raising ``database is locked`` (the daemon holds the graph, the CLI
      sometimes reads skills at the same time).
    * ``check_same_thread=False``: the daemon's worker thread and its IPC
      threads share the connection, and we serialise access with a lock
      (:class:`jarvis.memory.store.SqliteMemory`) rather than per-thread
      connections.
    """
    target = str(path)
    if target != ":memory:":
        parent = Path(target).expanduser().parent
        if str(parent):
            parent.mkdir(parents=True, exist_ok=True)
        target = str(Path(target).expanduser())
    conn = sqlite3.connect(
        target,
        check_same_thread=False,
        timeout=5.0,
        isolation_level=None,  # explicit transactions only; see store.py
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    if read_only:
        return conn
    if target != ":memory:":
        conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_schema(conn: sqlite3.Connection) -> sqlite3.Connection:
    """Create every table/index if missing and stamp the schema version.

    Idempotent: safe to call on every open, and safe on a database written by
    an older build (all statements are ``IF NOT EXISTS``, so no data is lost).
    """
    conn.executescript(SCHEMA)
    if user_version(conn) < SCHEMA_VERSION:
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    return conn


def open_db(path: str | Path) -> sqlite3.Connection:
    """Convenience: :func:`connect` + :func:`init_schema`."""
    return init_schema(connect(path))


def prune_skills(conn: sqlite3.Connection, max_rows: int) -> int:
    """Drop the least recently used skills above ``max_rows``.

    Returns the number of rows deleted.  Never runs when ``max_rows <= 0`` (a
    caller that wants "no pruning" passes 0).
    """
    if max_rows <= 0:
        return 0
    total = int(conn.execute("SELECT COUNT(*) FROM skills").fetchone()[0])
    if total <= max_rows:
        return 0
    cur = conn.execute(
        """
        DELETE FROM skills WHERE id IN (
          SELECT id FROM skills ORDER BY last_used_at ASC, id ASC LIMIT ?
        )
        """,
        (total - max_rows,),
    )
    return int(cur.rowcount or 0)


def prune_failures(conn: sqlite3.Connection, max_rows: int) -> int:
    """Drop the oldest failures above ``max_rows``; returns rows deleted."""
    if max_rows <= 0:
        return 0
    total = int(conn.execute("SELECT COUNT(*) FROM failures").fetchone()[0])
    if total <= max_rows:
        return 0
    cur = conn.execute(
        """
        DELETE FROM failures WHERE id IN (
          SELECT id FROM failures ORDER BY created_at ASC, id ASC LIMIT ?
        )
        """,
        (total - max_rows,),
    )
    return int(cur.rowcount or 0)


def prune_preferences(conn: sqlite3.Connection, max_rows: int) -> int:
    """Drop least-recently-updated preferences above ``max_rows``."""
    if max_rows <= 0:
        return 0
    total = int(conn.execute("SELECT COUNT(*) FROM preferences").fetchone()[0])
    if total <= max_rows:
        return 0
    cur = conn.execute(
        """
        DELETE FROM preferences WHERE key IN (
          SELECT key FROM preferences ORDER BY updated_at ASC, key ASC LIMIT ?
        )
        """,
        (total - max_rows,),
    )
    return int(cur.rowcount or 0)


def now_iso() -> str:
    """UTC timestamp in the format stored in every ``*_at`` column."""
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat(timespec="seconds")


def insert_task_log(conn: sqlite3.Connection, row: dict) -> None:
    """Insert/replace one ``task_log`` row (used by the runner and benchmarks)."""
    columns: Sequence[str] = (
        "task_id",
        "source",
        "user_input",
        "status",
        "api_calls",
        "tokens",
        "steps",
        "replans",
        "started_at",
        "finished_at",
        "duration_ms",
        "peak_rss_mb",
    )
    values = [row.get(name) for name in columns]
    placeholders = ", ".join("?" for _ in columns)
    conn.execute(
        f"INSERT OR REPLACE INTO task_log ({', '.join(columns)}) VALUES ({placeholders})",
        values,
    )
