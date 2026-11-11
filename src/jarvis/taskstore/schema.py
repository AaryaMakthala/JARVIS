"""SQLite connection, schema and version handling for the task event store."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 3

REQUIRED_TABLES = frozenset({"tasks", "task_events"})
REQUIRED_INDEXES = frozenset(
    {
        "idx_task_events_one_created",
        "idx_task_events_task_seq",
        "idx_tasks_owner_status",
        "idx_tasks_status_updated",
        "idx_tasks_updated",
    }
)

REQUIRED_COLUMNS: dict[str, frozenset[str]] = {
    "tasks": frozenset(
        {
            "task_id",
            "source",
            "stage",
            "status",
            "created_at",
            "started_at",
            "finished_at",
            "updated_at",
            "version",
            "retry_count",
            "active_tool",
            "active_step_id",
            "approval_state",
            "failure_category",
            "failure_code",
            "owner_instance",
            "owner_pid",
            "owner_process_create_time",
        }
    ),
    "task_events": frozenset(
        {
            "event_id",
            "task_id",
            "seq",
            "event_type",
            "stage",
            "status",
            "detail_json",
            "created_at",
        }
    ),
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
  task_id TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  stage TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  started_at TEXT,
  finished_at TEXT,
  updated_at TEXT NOT NULL,
  version INTEGER NOT NULL DEFAULT 0,
  retry_count INTEGER NOT NULL DEFAULT 0,
  active_tool TEXT,
  active_step_id TEXT,
  approval_state TEXT,
  failure_category TEXT,
  failure_code TEXT,
  owner_instance TEXT,
  owner_pid INTEGER,
  owner_process_create_time REAL
);

CREATE TABLE IF NOT EXISTS task_events (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
  seq INTEGER NOT NULL,
  event_type TEXT NOT NULL,
  stage TEXT,
  status TEXT,
  detail_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  UNIQUE (task_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_task_events_task_seq
  ON task_events (task_id, seq);
CREATE UNIQUE INDEX IF NOT EXISTS idx_task_events_one_created
  ON task_events (task_id) WHERE event_type = 'task_created';
CREATE INDEX IF NOT EXISTS idx_tasks_updated
  ON tasks (updated_at DESC, task_id DESC);
CREATE INDEX IF NOT EXISTS idx_tasks_status_updated
  ON tasks (status, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_tasks_owner_status
  ON tasks (owner_instance, status);
"""


class SchemaError(RuntimeError):
    """The database is missing a compatible task-store schema."""


def user_version(conn: sqlite3.Connection) -> int:
    """Return the schema version recorded in the database."""
    row = conn.execute("PRAGMA user_version").fetchone()
    return int(row[0]) if row else 0


def table_names(conn: sqlite3.Connection) -> set[str]:
    """Return user table names in the database."""
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return {str(row[0]) for row in rows}


def index_names(conn: sqlite3.Connection) -> set[str]:
    """Return user index names in the database."""
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'index' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return {str(row[0]) for row in rows}


def columns(conn: sqlite3.Connection, table: str) -> set[str]:
    """Return column names for ``table``."""
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {str(row[1]) for row in rows}


def connect(path: str | Path, *, create: bool, read_only: bool) -> sqlite3.Connection:
    """Open the task database with the pragmas this store relies on."""
    target = str(path)
    if target == ":memory:":
        if read_only:
            raise SchemaError("in-memory task store cannot be read-only")
        conn = sqlite3.connect(
            target,
            check_same_thread=False,
            timeout=5.0,
            isolation_level=None,
        )
    elif read_only:
        resolved = Path(target).expanduser().resolve(strict=False)
        if not resolved.is_file():
            raise FileNotFoundError("task store does not exist")
        uri = f"{resolved.as_uri()}?mode=ro"
        conn = sqlite3.connect(
            uri,
            uri=True,
            check_same_thread=False,
            timeout=5.0,
            isolation_level=None,
        )
    else:
        resolved = Path(target).expanduser().resolve(strict=False)
        if create:
            resolved.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(resolved),
            check_same_thread=False,
            timeout=5.0,
            isolation_level=None,
        )

    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    if read_only:
        conn.execute("PRAGMA query_only = ON")
    elif target != ":memory:":
        conn.execute("PRAGMA journal_mode = WAL")
    return conn


def verify_connection(
    conn: sqlite3.Connection,
    *,
    read_only: bool,
    memory: bool,
) -> None:
    """Validate the connection pragmas the task store depends on."""
    foreign_keys = conn.execute("PRAGMA foreign_keys").fetchone()
    if not foreign_keys or int(foreign_keys[0]) != 1:
        raise SchemaError("task store connection must enable foreign keys")

    busy_timeout = conn.execute("PRAGMA busy_timeout").fetchone()
    if not busy_timeout or int(busy_timeout[0]) <= 0:
        raise SchemaError("task store connection must set busy_timeout")

    if read_only:
        query_only = conn.execute("PRAGMA query_only").fetchone()
        if not query_only or int(query_only[0]) != 1:
            raise SchemaError("read-only task store connection must be query_only")
    elif not memory:
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()
        if not journal_mode or str(journal_mode[0]).lower() != "wal":
            raise SchemaError("task store connection must use WAL")


def initialize(conn: sqlite3.Connection) -> None:
    """Create the tables/indexes if missing and record the current version."""
    conn.executescript(SCHEMA)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


def migrate_v2_to_v3(conn: sqlite3.Connection) -> None:
    """Add task ownership metadata transactionally to a v2 database."""
    existing = columns(conn, "tasks")
    additions = (
        ("owner_instance", "TEXT"),
        ("owner_pid", "INTEGER"),
        ("owner_process_create_time", "REAL"),
    )
    conn.execute("BEGIN IMMEDIATE")
    try:
        for name, kind in additions:
            if name not in existing:
                conn.execute(f"ALTER TABLE tasks ADD COLUMN {name} {kind}")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tasks_owner_status "
            "ON tasks (owner_instance, status)"
        )
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    except BaseException:
        conn.rollback()
        raise
    else:
        conn.commit()


def validate(conn: sqlite3.Connection, *, read_only: bool) -> None:
    """Verify the schema and apply the version marker when writable."""
    version = user_version(conn)
    if version > SCHEMA_VERSION:
        raise SchemaError(
            f"task store schema version {version} is newer than supported {SCHEMA_VERSION}"
        )

    missing_tables = REQUIRED_TABLES - table_names(conn)
    if missing_tables:
        names = ", ".join(sorted(missing_tables))
        raise SchemaError(f"task store schema is not initialized (missing {names})")

    if version < 2:
        raise SchemaError(
            f"task store schema version {version} is older than supported {SCHEMA_VERSION}"
        )
    if version == 2:
        if read_only:
            raise SchemaError(
                f"task store schema version {version} is older than supported {SCHEMA_VERSION}"
            )
        migrate_v2_to_v3(conn)

    for table, required in REQUIRED_COLUMNS.items():
        missing_columns = required - columns(conn, table)
        if missing_columns:
            names = ", ".join(sorted(missing_columns))
            raise SchemaError(f"task store table {table} is missing columns: {names}")

    missing_indexes = REQUIRED_INDEXES - index_names(conn)
    if missing_indexes:
        names = ", ".join(sorted(missing_indexes))
        raise SchemaError(f"task store is missing indexes: {names}")
