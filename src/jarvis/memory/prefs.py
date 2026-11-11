"""User preferences: small, explicit, inspectable facts the human stated.

A preference is a short key/value the user (or ``jarvis`` on their behalf) chose
to remember, e.g. ``search_engine=duckduckgo`` or ``units=metric``.  Unlike
skills these are read back *verbatim* and offered to the planner as stable
context, so the rules are strict:

* **No inference.**  Only what was explicitly set is stored.  Nothing is
  guessed from behaviour.
* **No secrets.**  A key that looks credential-ish is rejected outright, and a
  value that looks credential-ish is rejected too.
* **Inspectable and deletable.**  ``jarvis skills`` (and ``jarvis doctor``)
  list them; the user can delete any single key.
* Values are bounded in length, and the number of keys is bounded, so the
  preferences block of a prompt can never grow without limit.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

from jarvis.memory.db import (
    looks_like_token,
    looks_secret,
    now_iso,
    prune_preferences,
    secret_key,
)

__all__ = [
    "FACT_KEY_PREFIX",
    "MAX_KEY_CHARS",
    "MAX_VALUE_CHARS",
    "Preference",
    "PreferenceStore",
    "fact_details",
    "normalise_key",
]

MAX_KEY_CHARS = 64
MAX_VALUE_CHARS = 400

#: Prefix for explicit facts remembered in the preferences table.  A fact key
#: looks like ``fact.<category>.<slug>_<hash>``: the category is for
#: inspection, and the short hash keeps two subjects with the same slug
#: distinct.  No schema change is needed; the existing key/value table already
#: gives atomic overwrite, pruning and deletion.
FACT_KEY_PREFIX = "fact."

_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")


def fact_details(key: str) -> tuple[str, str] | None:
    """Return ``(category, slug)`` for a fact key, or ``None`` for a preference."""
    if not key.startswith(FACT_KEY_PREFIX):
        return None
    rest = key[len(FACT_KEY_PREFIX) :]
    category, separator, slug = rest.partition(".")
    if not separator or not category or not slug:
        return None
    return category, slug


def normalise_key(key: str) -> str:
    """Lower-case and validate a preference key; raises ``ValueError`` if unusable.

    Lower-casing makes ``Units`` and ``units`` the same preference.  The
    character set excludes whitespace and separators that would make the key
    ambiguous in a prompt.
    """
    value = (key or "").strip().lower()
    if not value:
        raise ValueError("preference key must not be blank")
    if len(value) > MAX_KEY_CHARS:
        raise ValueError(f"preference key longer than {MAX_KEY_CHARS} characters")
    if not _KEY_RE.match(value):
        raise ValueError(
            "preference key must be lower-case letters, digits, dot, dash or underscore"
        )
    if secret_key(value) or looks_like_token(value):
        raise ValueError("preference key looks like a credential; refusing to store it")
    return value


@dataclass(frozen=True)
class Preference:
    """One stored preference."""

    key: str
    value: str
    updated_at: str = ""

    @property
    def category(self) -> str:
        """``preference`` for ordinary settings, else the fact category."""
        details = fact_details(self.key)
        return details[0] if details is not None else "preference"

    @property
    def provenance(self) -> str:
        """Where the value came from: always the human's explicit words."""
        return "user_explicit"

    def as_memory_text(self) -> str:
        """The form handed to the planner as stable user context."""
        details = fact_details(self.key)
        if details is not None:
            return f"Remembered fact ({details[0]}): {self.value}"
        return f"User preference: {self.key} = {self.value}"


class PreferenceStore:
    """Read/write access to the ``preferences`` table."""

    def __init__(self, conn: sqlite3.Connection, *, max_rows: int = 50) -> None:
        self._conn = conn
        self._max_rows = int(max_rows)

    @property
    def max_rows(self) -> int:
        """Maximum number of rows this store retains."""
        return self._max_rows

    def get(self, key: str, default: str | None = None) -> str | None:
        """One value by key, or ``default``."""
        try:
            normalised = normalise_key(key)
        except ValueError:
            return default
        row = self._conn.execute(
            "SELECT value FROM preferences WHERE key = ?", (normalised,)
        ).fetchone()
        return str(row["value"]) if row else default

    def set(self, key: str, value: str) -> Preference:
        """Store one preference; raises ``ValueError`` on an unusable key/value."""
        normalised = normalise_key(key)
        text = (value or "").strip()
        if not text:
            raise ValueError("preference value must not be blank")
        if len(text) > MAX_VALUE_CHARS:
            raise ValueError(f"preference value longer than {MAX_VALUE_CHARS} characters")
        if looks_secret(text):
            raise ValueError("preference value looks like a credential; refusing to store it")
        stamp = now_iso()
        self._conn.execute(
            "INSERT INTO preferences (key, value, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
            (normalised, text, stamp),
        )
        prune_preferences(self._conn, self._max_rows)
        return Preference(key=normalised, value=text, updated_at=stamp)

    def all(self, limit: int = 50) -> list[Preference]:
        """Every stored preference, most recently updated first."""
        if limit <= 0:
            return []
        rows = self._conn.execute(
            "SELECT key, value, updated_at FROM preferences ORDER BY updated_at DESC, key ASC LIMIT ?",
            (limit,),
        )
        return [
            Preference(
                key=str(row["key"]), value=str(row["value"]), updated_at=str(row["updated_at"])
            )
            for row in rows
        ]

    def delete(self, key: str) -> bool:
        """Forget one preference; ``True`` when a row was removed."""
        try:
            normalised = normalise_key(key)
        except ValueError:
            return False
        cur = self._conn.execute("DELETE FROM preferences WHERE key = ?", (normalised,))
        return bool(cur.rowcount)

    def count(self) -> int:
        """Number of stored preferences."""
        return int(self._conn.execute("SELECT COUNT(*) FROM preferences").fetchone()[0])

    def clear(self) -> int:
        """Delete every preference (used by ``jarvis reset --memory``)."""
        cur = self._conn.execute("DELETE FROM preferences")
        return int(cur.rowcount or 0)
