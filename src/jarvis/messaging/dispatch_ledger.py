"""Narrow local record of dispatched WhatsApp text operations (W5).

W5 can only verify a message it can *identify*.  The transient W4 dispatch
context (``task_id:action_hash``) lives in the tool context and dies with the
task, so this module keeps the smallest record that a later task needs:

* the opaque operation hash (``sha256(operation_id)``, the same identity the
  duplicate-send guard uses);
* the saved contact display name (the identity the user asked for);
* a sha256 digest of the sent text (never the text itself);
* the dispatch time;
* a non-sensitive outcome string (``pending``/``dispatched``/``unknown``).

No message body, no phone number, no authentication material.  Verification
matches the digest against the conversation's visible messages, so the body
never has to be stored to be compared.

The ledger is a *verification convenience*, not a safety control: writing it is
best-effort and a failure is logged rather than turning a completed send into
an error.  A missing record only ever makes verification say "I have no record
of that message", never "it was delivered".  Reading, however, fails closed:
an unreadable or corrupt ledger raises :class:`DispatchLedgerError` so the
caller refuses instead of guessing.

Storage is SQLite, matching the W4.4 duplicate-send guard (atomic single-statement
writes, ``BEGIN IMMEDIATE`` for the prune, a bounded busy timeout).  Records are
pruned to :data:`RETENTION_S`.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "RETENTION_S",
    "DispatchLedger",
    "DispatchLedgerError",
    "DispatchRecord",
]

logger = logging.getLogger(__name__)

#: How long a dispatch stays verifiable.  Long enough to answer "did my message
#: reach Mom?" the next morning; short enough not to become a chat archive.
RETENTION_S = 7 * 24 * 3600.0

#: Bounded wait for another process's write lock before refusing (seconds).
LEDGER_BUSY_TIMEOUT_S = 2.0

#: Outcomes are a closed, non-sensitive vocabulary.
OUTCOME_PENDING = "pending"
OUTCOME_DISPATCHED = "dispatched"
OUTCOME_UNKNOWN = "unknown"

_TABLE = """
CREATE TABLE IF NOT EXISTS dispatches (
    operation_hash TEXT PRIMARY KEY,
    contact TEXT NOT NULL,
    message_sha256 TEXT NOT NULL,
    dispatched_at REAL NOT NULL,
    outcome TEXT NOT NULL
)
"""

_INDEX = """
CREATE INDEX IF NOT EXISTS idx_dispatches_contact
    ON dispatches (contact, dispatched_at DESC)
"""


class DispatchLedgerError(OSError):
    """The dispatch ledger could not be used; the caller must not guess."""


@dataclass(frozen=True)
class DispatchRecord:
    """One previously dispatched WhatsApp text message (no body, no number)."""

    operation_hash: str
    contact: str
    message_sha256: str
    dispatched_at: float
    outcome: str


def _operation_hash(operation_id: str) -> str:
    """Opaque identity of one approval (same shape as the send guard key)."""
    return hashlib.sha256(str(operation_id or "").encode("utf-8")).hexdigest()


class DispatchLedger:
    """Bounded, privacy-preserving SQLite record of dispatched messages."""

    def __init__(
        self,
        path: Path | str,
        *,
        retention_s: float = RETENTION_S,
        clock: Any = None,
        busy_timeout_s: float = LEDGER_BUSY_TIMEOUT_S,
    ) -> None:
        self.path = Path(path)
        self.retention_s = float(retention_s)
        self._clock = clock if clock is not None else time.time
        self._busy_timeout_s = float(busy_timeout_s)

    # ── storage ──────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        """Open the ledger, creating the schema; raise on any storage problem."""
        conn: sqlite3.Connection | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.path), timeout=self._busy_timeout_s)
            conn.isolation_level = None
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute(_TABLE)
            conn.execute(_INDEX)
        except (sqlite3.Error, OSError) as exc:
            if conn is not None:
                try:
                    conn.close()
                except sqlite3.Error:
                    logger.debug("closing the unusable dispatch ledger failed", exc_info=True)
            raise DispatchLedgerError(
                f"the WhatsApp dispatch ledger is unusable ({exc})"
            ) from exc
        return conn

    # ── operations ───────────────────────────────────────────────────────

    def record(
        self,
        operation_id: str,
        contact: str,
        text: str,
        *,
        outcome: str = OUTCOME_PENDING,
    ) -> str | None:
        """Record one dispatch attempt; returns its opaque hash.

        ``None`` means the operation had no identity (nothing can be verified),
        which is a programming error rather than a storage failure.
        """
        identifier = str(operation_id or "").strip()
        name = " ".join(str(contact or "").split())
        if not identifier or not name:
            return None
        from jarvis.messaging.verification import digest_text

        digest = digest_text(text)
        now = float(self._clock())
        operation_hash = _operation_hash(identifier)
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "DELETE FROM dispatches WHERE dispatched_at <= ?",
                (now - self.retention_s,),
            )
            conn.execute(
                "INSERT OR REPLACE INTO dispatches "
                "(operation_hash, contact, message_sha256, dispatched_at, outcome) "
                "VALUES (?, ?, ?, ?, ?)",
                (operation_hash, name, digest, now, str(outcome)),
            )
            conn.execute("COMMIT")
        except sqlite3.Error as exc:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                logger.debug("the dispatch ledger rollback failed", exc_info=True)
            raise DispatchLedgerError(
                f"the WhatsApp dispatch ledger could not be written ({exc})"
            ) from exc
        finally:
            conn.close()
        return operation_hash

    def mark(self, operation_id: str, outcome: str) -> bool:
        """Update the outcome of a recorded attempt; ``False`` when unknown."""
        identifier = str(operation_id or "").strip()
        if not identifier:
            return False
        conn = self._connect()
        try:
            cursor = conn.execute(
                "UPDATE dispatches SET outcome = ? WHERE operation_hash = ?",
                (str(outcome), _operation_hash(identifier)),
            )
            return cursor.rowcount == 1
        except sqlite3.Error as exc:
            raise DispatchLedgerError(
                f"the WhatsApp dispatch ledger could not be updated ({exc})"
            ) from exc
        finally:
            conn.close()

    def latest(self, contact: str) -> DispatchRecord | None:
        """The newest verifiable dispatch for ``contact`` inside retention.

        Raises :class:`DispatchLedgerError` when the ledger exists but cannot be
        read, so a corrupt store can never be mistaken for "nothing sent".
        """
        name = " ".join(str(contact or "").split())
        if not name:
            return None
        if not self.path.exists():
            return None
        if not self.path.is_file():
            raise DispatchLedgerError("the WhatsApp dispatch ledger path is not a file")
        now = float(self._clock())
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT operation_hash, contact, message_sha256, dispatched_at, outcome "
                "FROM dispatches WHERE contact = ? AND dispatched_at > ? "
                "ORDER BY dispatched_at DESC LIMIT 1",
                (name, now - self.retention_s),
            ).fetchone()
        except sqlite3.Error as exc:
            raise DispatchLedgerError(
                f"the WhatsApp dispatch ledger could not be read ({exc})"
            ) from exc
        finally:
            conn.close()
        if row is None:
            return None
        return DispatchRecord(
            operation_hash=str(row[0]),
            contact=str(row[1]),
            message_sha256=str(row[2]),
            dispatched_at=float(row[3]),
            outcome=str(row[4]),
        )
