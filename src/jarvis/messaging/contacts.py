"""Contacts and the send rate limit (Stage 8, slice 8.6.4).

These primitives used to live in :mod:`jarvis.tools.whatsapp`.  They were
moved here because the *messaging* domain owns them, not the WhatsApp tool:
every channel adapter needs the same contact list and the same send rate
limit, and the reusable flow in :mod:`jarvis.messaging.flow` resolves the
recipient and checks the rate limit once, for all of them.

``jarvis.tools.whatsapp`` re-exports every name, so the old import paths keep
working and there is exactly one implementation of each rule:

* a phone number is stored as plain digits and never leaves the machine in
  full (only the last two digits are ever shown);
* a send is refused unless **exactly one** contact matches - zero or several
  matches are a clarification, never a guess;
* the rate limit is checked *before* any side effect and is persisted across
  restarts.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import BaseModel

    from jarvis.tools.base import ToolContext

logger = logging.getLogger(__name__)

__all__ = [
    "Contact",
    "ContactStore",
    "RateLimitError",
    "RateLimiter",
    "make_recipient_precheck",
    "masked_number",
    "normalize_number",
    "recipient_error",
    "resolve_single",
]

#: Contacts file schema version (bump to force a migration in a future phase).
_CONTACTS_VERSION = 1
_MAX_CONTACTS = 500  # hard cap so a corrupted file cannot grow unbounded


# Privacy (docs/03): phone numbers never surface in full - keep last 2 digits.
def masked_number(number: str) -> str:
    """Render a phone number with everything but the last 2 digits hidden."""
    digits = normalize_number(number)
    if len(digits) <= 2:
        return "+" + "•" * len(digits)
    return "+" + "•" * (len(digits) - 2) + digits[-2:]


def normalize_number(raw: str) -> str:
    """Return E.164-ish digits (no leading ``+``); raises ValueError otherwise.

    Accepts an optional leading ``+`` and common separators (space, -, parens).
    The result is 8-15 digits with no separators, ready for the
    ``whatsapp://send?phone=`` URI which expects plain digits.
    """
    text = raw.strip()
    body = text.removeprefix("+")
    digits = re.sub(r"[^0-9]", "", body)
    if not text or not (8 <= len(digits) <= 15):
        raise ValueError(
            f"phone number must be 8-15 digits (E.164-ish), got {len(digits) or 0} digit(s)"
        )
    return digits


@dataclass(frozen=True)
class Contact:
    """A single WhatsApp contact (name only; number stored as plain digits)."""

    name: str
    number: str
    created: str = ""


def _created() -> str:
    import datetime

    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


class ContactStore:
    """File-backed contact list, edited only through ``jarvis contacts …``."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    # ── persistence ──────────────────────────────────────────────────────

    def load(self) -> list[Contact]:
        """Read and validate the contacts file; empty on any problem."""
        if not self.path.is_file():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            records = data["contacts"]
        except Exception:  # noqa: BLE001 - a bad file must never block sending
            logger.warning("contacts file unreadable, treating as empty: %s", self.path)
            return []
        contacts: list[Contact] = []
        for record in records[:_MAX_CONTACTS]:
            try:
                contacts.append(
                    Contact(
                        name=str(record["name"]).strip(),
                        number=normalize_number(str(record["number"])),
                        created=str(record.get("created", "")),
                    )
                )
            except Exception:  # skip malformed records
                logger.debug("skipping malformed contacts record", exc_info=True)
                continue
        return contacts

    def save(self, contacts: list[Contact]) -> None:
        """Atomically write the contact list."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _CONTACTS_VERSION,
            "contacts": [
                {"name": c.name, "number": c.number, "created": c.created} for c in contacts
            ],
        }
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    # ── operations ───────────────────────────────────────────────────────

    def add(self, name: str, number: str) -> str:
        """Add a contact after validation; returns a status line.

        Raises ValueError on invalid input or duplicates so the CLI can render
        the reason and exit 1.
        """
        name = name.strip()
        if not name:
            raise ValueError("contact name is empty")
        digits = normalize_number(number)
        contacts = self.load()
        if any(c.name.lower() == name.lower() for c in contacts):
            raise ValueError(f"a contact named {name!r} already exists")
        if any(c.number == digits for c in contacts):
            raise ValueError(f"the number +{digits} is already assigned to another contact")
        contacts.append(Contact(name=name, number=digits, created=_created()))
        self.save(contacts)
        return f"added contact {name!r} (+{digits})"

    def remove(self, name: str) -> str:
        """Remove a contact by name (case-insensitive); raises ValueError."""
        contacts = self.load()
        remaining = [c for c in contacts if c.name.lower() != name.strip().lower()]
        if len(remaining) == len(contacts):
            raise ValueError(f"no contact named {name!r}")
        self.save(remaining)
        return f"removed contact {name!r}"

    def resolve(self, query: str) -> list[Contact]:
        """Case-insensitive lookup: exact name match first, then substring."""
        q = query.strip().lower()
        contacts = self.load()
        exact = [c for c in contacts if c.name.lower() == q]
        if exact:
            return exact
        return [c for c in contacts if q in c.name.lower()]


class RateLimitError(Exception):
    """Raised when the messaging send rate limit is exceeded."""


class RateLimiter:
    """Persisted send-frequency gate (max/hour + minimum spacing, docs §2.6)."""

    def __init__(
        self,
        path: Path | str,
        *,
        max_per_hour: int = 10,
        min_interval_s: float = 5.0,
        clock: object | None = None,
    ) -> None:
        if max_per_hour <= 0:
            raise ValueError("max_per_hour must be >= 1")
        self.path = Path(path)
        self.max_per_hour = max_per_hour
        self.min_interval_s = min_interval_s
        self._clock = clock if clock is not None else time.time
        self._lock = threading.Lock()
        self._events = self._load()

    def _load(self) -> list[float]:
        if not self.path.is_file():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return [float(t) for t in data.get("events", [])]
        except Exception:  # noqa: BLE001 - treat a corrupt log as empty
            logger.warning("rate-limit log unreadable, treating as empty: %s", self.path)
            return []

    def _prune(self, now: float) -> list[float]:
        cutoff = now - 3600.0
        return [t for t in self._events if t > cutoff]

    def check(self) -> None:
        """Raise :class:`RateLimitError` when this send must be refused."""
        with self._lock:
            now = self._clock()
            self._events = self._prune(now)
            if self._events and (now - max(self._events)) < self.min_interval_s:
                raise RateLimitError(
                    f"messaging rate limit: wait at least {self.min_interval_s:.0f}s between sends"
                )
            if len(self._events) >= self.max_per_hour:
                raise RateLimitError(
                    f"messaging rate limit: at most {self.max_per_hour} sends per hour"
                )

    def record(self) -> None:
        """Record a successfully sent message (persisted across restarts)."""
        with self._lock:
            now = self._clock()
            self._events = self._prune(now)
            self._events.append(now)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"events": self._events}), encoding="utf-8")

    def traffic_count(self) -> int:
        """Number of sends recorded in the current window (read-only)."""
        with self._lock:
            return len(self._prune(self._clock()))


def resolve_single(store: ContactStore, query: str) -> tuple[Contact | None, str | None]:
    """Return exactly one contact, or the error text listing the options.

    Shared by every channel adapter and by the WhatsApp tools so that contact
    resolution has exactly one implementation.
    """
    matches = store.resolve(query)
    if len(matches) == 1:
        return matches[0], None
    if not matches:
        known = ", ".join(c.name for c in store.load()) or "(no contacts yet)"
        return None, (
            f"no contact matches {query!r}. Known contacts: {known}. "
            "Add one with `jarvis contacts add`."
        )
    names = ", ".join(c.name for c in matches)
    return None, f"multiple contacts match {query!r}: {names} — be more specific"


def recipient_error(store: ContactStore, query: str) -> str | None:
    """The recipient failure for ``query``, or ``None`` when exactly one matches.

    This is the read-only form of :func:`resolve_single`, used by the messaging
    tools' ``precheck`` hook.  It has no side effects and never guesses.
    """
    _contact, error = resolve_single(store, query)
    return error


def make_recipient_precheck(
    store: ContactStore | None = None,
) -> Callable[[BaseModel, ToolContext], str | None]:
    """Build a read-only ``ToolSpec.precheck`` that validates the recipient.

    ``store`` is the test injection point; the default reads the real contacts
    file at check time.  The check runs before any confirmation and again just
    before dispatch, so an unknown or ambiguous recipient is refused without
    opening a chat, staging a file, prompting for approval or unlocking.
    """

    def precheck(args: BaseModel, ctx: ToolContext) -> str | None:
        del ctx
        chosen = store
        if chosen is None:
            from jarvis import config

            chosen = ContactStore(config.contacts_file())
        return recipient_error(chosen, str(getattr(args, "contact", "") or ""))

    return precheck
