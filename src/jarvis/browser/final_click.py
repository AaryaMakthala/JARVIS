"""Guarded final click (browser_final_click) - implementation."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from jarvis.config import Settings
from jarvis.memory.db import connect, init_schema


ALLOWED_ACTIONS = {"send", "submit", "post", "publish", "subscribe", "book"}
PAYMENT_WORDS = {
    "buy",
    "purchase",
    "pay",
    "payment",
    "payments",
    "checkout",
    "order",
    "place order",
    "donate",
}


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", str(s or "")).lower().split())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_hex(s: str) -> str:
    return hashlib.sha256(str(s or "").encode("utf-8")).hexdigest()


def init_final_click_schema(conn: Any) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS final_click_journal (
            id TEXT PRIMARY KEY,
            task_id TEXT,
            host TEXT,
            action TEXT,
            target_fingerprint TEXT,
            confirmation_sha256 TEXT,
            state TEXT NOT NULL,
            outcome_code TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_final_click_task ON final_click_journal(task_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_final_click_state ON final_click_journal(state)")
    conn.commit()


@dataclass(frozen=True)
class FinalClickJournalRow:
    id: str
    task_id: str
    host: str
    action: str
    target_fingerprint: str
    confirmation_sha256: str
    state: str
    outcome_code: str
    created_at: str
    updated_at: str


class FinalClickJournal:
    def __init__(self, path: str | None = None):
        self.path = path
        self._conn = None

    def _get(self):
        if self._conn is not None:
            return self._conn
        conn = connect(self.path)
        init_schema(conn)
        init_final_click_schema(conn)
        self._conn = conn
        return conn

    def prune_old(self, max_rows: int = 50) -> None:
        conn = self._get()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM final_click_journal")
        row = cur.fetchone()
        count = int(row[0]) if row else 0
        if count <= max_rows:
            return
        cur.execute(
            "SELECT id FROM final_click_journal ORDER BY created_at ASC LIMIT ?",
            (count - max_rows,),
        )
        ids = [r[0] for r in cur.fetchall()]
        if not ids:
            return
        cur.execute(
            f"DELETE FROM final_click_journal WHERE id IN ({','.join(['?'] * len(ids))})",
            ids,
        )
        conn.commit()

    def reserve(self, row: FinalClickJournalRow) -> bool:
        conn = self._get()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO final_click_journal (id, task_id, host, action, target_fingerprint, confirmation_sha256, state, outcome_code, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    row.id,
                    row.task_id,
                    row.host,
                    row.action,
                    row.target_fingerprint,
                    row.confirmation_sha256,
                    row.state,
                    row.outcome_code,
                    row.created_at,
                    row.updated_at,
                ),
            )
            conn.commit()
            self.prune_old(50)
            return True
        except Exception:
            return False

    def get_by_id(self, row_id: str) -> dict[str, Any] | None:
        conn = self._get()
        cur = conn.cursor()
        cur.execute("SELECT * FROM final_click_journal WHERE id=?", (row_id,))
        row = cur.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))

    def has_indeterminate_or_dispatched(self, task_id: str) -> bool:
        conn = self._get()
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM final_click_journal WHERE task_id=? AND state IN ('dispatched','indeterminate')",
            (task_id,),
        )
        row = cur.fetchone()
        return int(row[0]) if row else 0 > 0

    def update_state(self, row_id: str, state: str, outcome_code: str = "") -> bool:
        conn = self._get()
        cur = conn.cursor()
        cur.execute(
            "UPDATE final_click_journal SET state=?, outcome_code=?, updated_at=? WHERE id=?",
            (state, outcome_code, _now_iso(), row_id),
        )
        conn.commit()
        return cur.rowcount > 0

    def delete(self, row_id: str) -> None:
        conn = self._get()
        cur = conn.cursor()
        cur.execute("DELETE FROM final_click_journal WHERE id=?", (row_id,))
        conn.commit()

    def total_count_all(self) -> int:
        conn = self._get()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM final_click_journal")
        row = cur.fetchone()
        return int(row[0]) if row else 0


_journal = FinalClickJournal()


def get_journal() -> FinalClickJournal:
    return _journal


def classify_label_for_payment(name: str) -> str | None:
    n = _norm(name)
    if not n:
        return None
    for w in PAYMENT_WORDS:
        wnorm = _norm(w)
        if wnorm and wnorm in n:
            return "payment"
    return None


def label_matches_allowed_action(name: str, action: str) -> bool:
    n = _norm(name)
    a = _norm(action)
    if not n or action not in ALLOWED_ACTIONS:
        return False
    synonyms = {
        "send": ["send"],
        "submit": ["submit"],
        "post": ["post"],
        "publish": ["publish"],
        "subscribe": ["subscribe", "follow"],
        "book": ["book", "reserve", "rsvp"],
    }
    for s in synonyms.get(action, [action]):
        if _norm(s) in n:
            return True
    return False
