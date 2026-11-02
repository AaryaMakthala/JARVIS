"""The WhatsApp Web channel (Stage 8, slice 8.6.4).

File messages through the isolated WhatsApp Web profile.  The channel owns the
*mechanics* of that channel - acquiring the narrow page, driving the chat,
staging the attachment and pressing Send - and nothing else: the reusable
flow (:mod:`jarvis.messaging.flow`) has already resolved the recipient and
checked the rate limit, and the tool has already bound the approval to the
file's content.

Every step fails closed (docs/03 invariant 9, docs/04 §2.6):

1. the file must still be the bytes the approval was bound to;
2. the same (contact, content) pair must not have been attempted recently;
3. the chat identity must be proven by the header;
4. the attachment preview must be visible inside that chat's composer;
5. the file is re-hashed immediately before the click, so a file that changed
   while it was being staged is never sent;
6. exactly one Send control may look like Send.

If any of these cannot be proven, **nothing is sent and the draft is left in
WhatsApp** for the owner.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jarvis.browser.whatsapp_web import (
    ConversationSummary,
    WhatsAppDriver,
    WhatsAppError,
    WhatsAppMessage,
    WhatsAppSendUncertain,
)
from jarvis.messaging.base import (
    CAPABILITY_FILE,
    CAPABILITY_TEXT,
    CHANNEL_WHATSAPP_WEB,
    MessageError,
    MessageOutcome,
    MessageRequest,
)
from jarvis.messaging.contacts import Contact
from jarvis.messaging.dispatch_ledger import DispatchLedger
from jarvis.messaging.resolver import normalize_name
from jarvis.policy import paths

logger = logging.getLogger(__name__)

__all__ = [
    "RESEND_WINDOW_S",
    "ConversationRead",
    "FileSendGuard",
    "GuardStorageError",
    "WhatsAppWebAdapter",
    "WhatsAppWebTextAdapter",
    "content_digest",
    "operation_guard_key",
]

#: How long a send attempt blocks an identical retry (same contact + content).
#: Long enough to absorb retries/replans/restarts, short enough that a genuine
#: "send it again" later in the day is not blocked forever.
RESEND_WINDOW_S = 600.0

#: Fallback cap when the policy module does not expose its content-hash cap.
_DEFAULT_HASH_CAP = 16 * 1024 * 1024

@dataclass(frozen=True)
class ConversationRead:
    """The result of a read-only WhatsApp conversation read.

    ``messages`` are already bounded and chronological.  The message text is
    untrusted external content: it is returned to the tool, never to the
    planner as instructions.
    """

    header: str
    messages: tuple[WhatsAppMessage, ...]


class GuardStorageError(OSError):
    """The duplicate-send guard storage could not be used, so sending is refused.

    It subclasses :class:`OSError` deliberately: both channel adapters already
    wrap ``is_recent``/``claim`` in ``except OSError`` and refuse to send, so a
    broken store can never be read as "nothing was sent yet".
    """


#: How long a claim waits for another *process* to release SQLite's write lock
#: before refusing.  Bounded on purpose: a long wait only delays a refusal.
GUARD_BUSY_TIMEOUT_S = 2.0

_CLAIM_TABLE = (
    "CREATE TABLE IF NOT EXISTS claims (key TEXT PRIMARY KEY, claimed_at REAL NOT NULL)"
)


class FileSendGuard:
    """Persisted, **cross-process** "this operation was already attempted" guard.

    Storage is a small SQLite database with one row per claimed key.  ``claim``
    runs ``BEGIN IMMEDIATE`` and then ``INSERT OR IGNORE`` on a PRIMARY KEY, so
    two *processes* racing for the same key cannot both win: SQLite's write lock
    serialises them (bounded by ``busy_timeout_s``) and the loser's insert
    reports 0 affected rows.  No Python lock is used as the security mechanism;
    the database is the authority, which is what makes it work between processes.

    The claim is committed **before** the irreversible click.  A process that
    crashes after claiming leaves the row behind, so the same authorization
    cannot be replayed, and an unknown click outcome is equally non-retryable
    while the row is inside :data:`RESEND_WINDOW_S`.

    The guard stores only opaque sha256 keys and a timestamp - never message
    text, file paths or phone numbers.

    Every storage problem (unwritable directory, corrupt file, busy timeout,
    failed write) raises :class:`GuardStorageError`; a broken guard fails closed
    instead of authorizing a send.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        window_s: float = RESEND_WINDOW_S,
        clock: Any = None,
        busy_timeout_s: float = GUARD_BUSY_TIMEOUT_S,
    ) -> None:
        self.path = Path(path)
        self.window_s = float(window_s)
        self._clock = clock if clock is not None else time.time
        self._busy_timeout_s = float(busy_timeout_s)

    # ── storage ──────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        """Open the guard database, creating the table when needed.

        Raises :class:`GuardStorageError` for every SQLite failure, so a locked,
        corrupt or unwritable store can never look like an empty one.
        """
        conn: sqlite3.Connection | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.path), timeout=self._busy_timeout_s)
            conn.isolation_level = None  # explicit BEGIN/COMMIT below
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute(_CLAIM_TABLE)
        except (sqlite3.Error, OSError) as exc:
            # Never leave a handle on a corrupt/locked store: on Windows that
            # keeps the file locked and hides the original problem.
            if conn is not None:
                try:
                    conn.close()
                except sqlite3.Error:
                    logger.debug("closing the unusable guard failed", exc_info=True)
            raise GuardStorageError(
                f"the WhatsApp duplicate-send guard is unusable ({exc})"
            ) from exc
        return conn

    # ── the guard contract ───────────────────────────────────────────────

    def is_recent(self, key: str) -> bool:
        """Whether ``key`` was claimed inside the window (read-only)."""
        if not self.path.exists():
            return False
        if not self.path.is_file():
            raise GuardStorageError(
                "the WhatsApp duplicate-send guard path is not a file"
            )
        now = float(self._clock())
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT 1 FROM claims WHERE key = ? AND claimed_at > ?",
                (str(key), now - self.window_s),
            ).fetchone()
        except sqlite3.Error as exc:
            raise GuardStorageError(
                f"the WhatsApp duplicate-send guard could not be read ({exc})"
            ) from exc
        finally:
            conn.close()
        return row is not None

    def claim(self, key: str) -> bool:
        """Atomically claim ``key``; ``False`` when an attempt is already recent.

        The prune, duplicate check and insert share one ``BEGIN IMMEDIATE``
        transaction, so the decision is serialised across processes.  ``False``
        is a safe refusal, never "cannot tell".
        """
        now = float(self._clock())
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "DELETE FROM claims WHERE claimed_at <= ?", (now - self.window_s,)
            )
            cursor = conn.execute(
                "INSERT OR IGNORE INTO claims (key, claimed_at) VALUES (?, ?)",
                (str(key), now),
            )
            claimed = cursor.rowcount == 1
            conn.execute("COMMIT")
        except sqlite3.Error as exc:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                logger.debug("the guard rollback also failed", exc_info=True)
            raise GuardStorageError(
                f"the WhatsApp duplicate-send guard could not be recorded ({exc})"
            ) from exc
        finally:
            conn.close()
        return claimed


def content_digest(target: Path, max_bytes: int) -> str | None:
    """sha256 of ``target``, or ``None`` when hashing is unavailable/refused."""
    content_hash = getattr(paths, "content_hash", None)
    if content_hash is None:
        return None
    try:
        return str(content_hash(target, max_bytes=max_bytes))
    except Exception:  # noqa: BLE001 - any refusal means "cannot bind this file"
        logger.warning("the attachment could not be hashed for content binding")
        return None


def guard_key(contact: Contact, digest: str) -> str:
    """Opaque key for one (contact number, content) send attempt."""
    payload = f"{contact.number}\n{digest}".encode()
    return hashlib.sha256(payload).hexdigest()


def operation_guard_key(operation_id: str) -> str:
    """Opaque key for one approved operation (task id + action hash).

    The key is operation-bound, not content-bound: a retry/replan of the *same*
    approval cannot send twice, while a genuinely new task with the same text
    is a new approval and is allowed.
    """
    payload = str(operation_id or "").encode()
    return hashlib.sha256(payload).hexdigest()


def resend_refusal(contact: Contact, guard: FileSendGuard) -> str:
    """The refusal text for a duplicate (contact, content) send attempt."""
    minutes = max(1, int(guard.window_s // 60))
    return (
        f"this file was already sent to {contact.name} in the last {minutes} minutes; "
        "refusing to send it again"
    )


class WhatsAppWebAdapter:
    """File messages through the isolated WhatsApp Web profile."""

    name = CHANNEL_WHATSAPP_WEB

    def __init__(
        self,
        *,
        holder: Any = None,
        guard: FileSendGuard,
        store: Any = None,
        max_bytes: int = _DEFAULT_HASH_CAP,
    ) -> None:
        self._holder = holder
        self._guard = guard
        self._store = store
        self._max_bytes = int(max_bytes)

    def _name_key_is_unique(self, contact: Contact) -> bool:
        """Whether no other stored contact shares this name's matching key.

        The adapter is given the same ContactStore the flow resolved through.
        If the store is missing or unreadable this returns ``False`` (fail
        closed): a name-only header must then equal the whole exact saved name.
        """
        store = self._store
        if store is None:
            return False
        try:
            contacts = store.load()
        except Exception:  # noqa: BLE001 - an unreadable store is not a proof
            logger.warning("the contact store could not be read for header identity")
            return False
        key = normalize_name(contact.name)
        return sum(1 for item in contacts if normalize_name(item.name) == key) == 1

    # ── the channel contract ─────────────────────────────────────────────

    def capabilities(self) -> frozenset[str]:
        """This channel carries files."""
        return frozenset({CAPABILITY_FILE})

    def send(self, request: MessageRequest, contact: Contact) -> MessageOutcome:
        """Attach one file to the verified chat and press Send."""
        target = Path(request.path) if request.path else None
        if target is None:
            raise MessageError("there is no file to send")
        if not request.digest:
            raise MessageError("the file's content is not bound to this approval; refusing to send")
        key = guard_key(contact, request.digest)
        try:
            recent = self._guard.is_recent(key)
        except OSError as exc:
            raise MessageError(
                f"the duplicate-send guard could not be read ({exc}); refusing to send"
            ) from exc
        if recent:
            raise MessageError(resend_refusal(contact, self._guard))

        page = self._open_page()
        if page is None:
            raise MessageError("the WhatsApp browser session is not available; refusing to send")
        driver = WhatsAppDriver(page)
        try:
            driver.open_chat_by_phone(contact.number, contact.name)
        except WhatsAppError as exc:
            raise MessageError(f"could not open the WhatsApp chat: {exc}") from exc
        try:
            driver.attach(target, expect_name=target.name)
        except WhatsAppError as exc:
            raise MessageError(f"could not stage the attachment: {exc}") from exc

        # The gate hashed at approval time and act re-hashed immediately before
        # dispatch; this last read proves the bytes are still the approved ones
        # at the moment of the click (the file could have changed while
        # attaching).
        if content_digest(target, self._max_bytes) != request.digest:
            raise MessageError(
                "the file changed while the attachment was being prepared; "
                "refusing to send (the draft is left in WhatsApp)"
            )

        # Claim immediately before the click: a retry, replan or restarted
        # daemon cannot send the same bytes to the same contact twice inside
        # the window.
        try:
            claimed = self._guard.claim(key)
        except OSError as exc:
            raise MessageError(
                f"the duplicate-send guard could not be recorded ({exc}); refusing to send"
            ) from exc
        if not claimed:
            raise MessageError(resend_refusal(contact, self._guard))

        try:
            outcome = driver.send(expect_name=target.name)
        except WhatsAppError as exc:
            raise MessageError(
                f"the WhatsApp file send could not be confirmed: {exc} "
                "(the duplicate-send guard stays in place)"
            ) from exc

        logger.info(
            "whatsapp file sent contact=%s phone_digits=%d name=%s bytes=%d",
            contact.name,
            len(contact.number),
            target.name,
            target.stat().st_size,
        )
        return MessageOutcome(
            channel=self.name,
            contact=contact.name,
            sent=True,
            detail=f"sent WhatsApp file {target.name!r} to {contact.name}",
            data={
                "recipient": contact.name,
                "phone_digits": contact.number,
                "file": target.name,
                "content_sha256": request.digest,
                "control": outcome.control,
                "staged_count": outcome.staged_count,
                "chat_confirmed": True,
            },
        )

    # ── channel extras ───────────────────────────────────────────────────

    def open_chat(self, contact: Contact) -> str:
        """Open the verified chat for ``contact`` and return its header."""
        page = self._open_page()
        if page is None:
            raise MessageError("the WhatsApp browser session is not available; refusing to open")
        driver = WhatsAppDriver(page)
        try:
            return driver.open_chat_by_phone(contact.number, contact.name)
        except WhatsAppError as exc:
            raise MessageError(f"could not open the WhatsApp chat: {exc}") from exc

    def open_chat_by_search(self, contact: Contact) -> str:
        """Open ``contact``'s chat through WhatsApp's own number search.

        Same guards as :meth:`open_chat`: the page comes from the isolated
        WhatsApp profile, the session must be authenticated, and the driver
        refuses an ambiguous hit or an unproven header.  The query is the
        contact's trusted number, so a display name alone can never select a
        conversation.
        """
        page = self._open_page()
        if page is None:
            raise MessageError("the WhatsApp browser session is not available; refusing to open")
        driver = WhatsAppDriver(page)
        try:
            return driver.open_chat_by_search(
                contact.number,
                contact.name,
                name_is_unique=self._name_key_is_unique(contact),
            )
        except WhatsAppError as exc:
            raise MessageError(f"could not open the WhatsApp chat: {exc}") from exc

    def list_unread(self, limit: int) -> tuple[ConversationSummary, ...]:
        """List bounded unread conversations without opening one (W6).

        The page comes from the isolated WhatsApp profile and the driver proves
        the exact origin and an authenticated session first.  Only the visible
        conversation list is read: no chat is opened, so this operation does
        not change read receipts.  Nothing is sent, staged or clicked.
        """
        page = self._open_page()
        if page is None:
            raise MessageError("the WhatsApp browser session is not available; refusing to read")
        driver = WhatsAppDriver(page)
        try:
            conversations = driver.list_unread(limit)
        except WhatsAppError as exc:
            raise MessageError(f"could not read the WhatsApp conversation list: {exc}") from exc
        logger.info("whatsapp unread conversations listed count=%d", len(conversations))
        return conversations

    def read_messages(self, contact: Contact, limit: int) -> ConversationRead:
        """Read recent messages from ``contact``'s verified chat (read-only).

        The chat is opened through the W2 number-search route, so the
        conversation is bound to the trusted contact number rather than to a
        display name.  The driver proves the session and the chat identity
        before and after the read; no message is ever sent, staged or clicked.
        """
        page = self._open_page()
        if page is None:
            raise MessageError("the WhatsApp browser session is not available; refusing to read")
        driver = WhatsAppDriver(page)
        try:
            header = driver.open_chat_by_search(
                contact.number,
                contact.name,
                name_is_unique=self._name_key_is_unique(contact),
            )
            messages = driver.recent_messages(limit)
        except WhatsAppError as exc:
            raise MessageError(f"could not read the WhatsApp conversation: {exc}") from exc
        logger.info(
            "whatsapp messages read contact=%s phone_digits=%d count=%d",
            contact.name,
            len(contact.number),
            len(messages),
        )
        return ConversationRead(header=header, messages=messages)

    # ── internals ────────────────────────────────────────────────────────

    def _open_page(self) -> Any | None:
        """Start/return the WhatsApp profile's narrow page, or ``None``."""
        holder = self._holder
        if holder is None:
            return None
        try:
            session = holder.acquire()
        except Exception:
            logger.warning("the WhatsApp browser holder failed", exc_info=True)
            return None
        if session is None:
            return None
        try:
            return session.whatsapp_page()
        except Exception:
            logger.warning("the WhatsApp page is not available", exc_info=True)
            return None


class WhatsAppWebTextAdapter(WhatsAppWebAdapter):
    """Text messages through the isolated WhatsApp Web profile (W4).

    The reusable flow has already resolved the recipient and checked the rate
    limit.  This adapter opens the chat through the W2 number-search route,
    composes the exact approved text, claims an operation-bound duplicate-send
    guard immediately before the click, and presses Send only when the chat,
    the composer contents and the single Send control all agree.

    An unobservable click outcome is **not** an exception to the tool: it is a
    successful ``MessageOutcome`` with ``send_outcome="unknown"`` and
    ``delivery_verified=False``.  That keeps the graph from retrying a message
    that may already have gone out.  The adapter never claims delivery.
    """

    def __init__(
        self,
        *,
        holder: Any = None,
        guard: FileSendGuard,
        operation_id: str,
        store: Any = None,
        ledger: DispatchLedger | None = None,
    ) -> None:
        super().__init__(holder=holder, guard=guard, store=store)
        self._operation_id = str(operation_id or "")
        self._ledger = ledger

    # ── dispatch ledger (W5) ─────────────────────────────────────────

    def _record_dispatch(self, contact: Contact, text: str) -> None:
        """Best-effort: remember this attempt so a later task can verify it.

        Failure is logged, never raised: the send already passed its own
        guard, and a missing record only means a later verification says
        "I have no record", never that the message was delivered.
        """
        ledger = self._ledger
        if ledger is None:
            return
        try:
            ledger.record(self._operation_id, contact.name, text, outcome="pending")
        except OSError:
            logger.warning("the WhatsApp dispatch ledger could not record this attempt")

    def _mark_dispatch(self, outcome: str) -> None:
        """Best-effort: record the observed send outcome (no message body)."""
        ledger = self._ledger
        if ledger is None:
            return
        try:
            ledger.mark(self._operation_id, outcome)
        except OSError:
            logger.warning("the WhatsApp dispatch ledger could not be updated")

    def capabilities(self) -> frozenset[str]:
        """This channel carries text."""
        return frozenset({CAPABILITY_TEXT})

    def send(self, request: MessageRequest, contact: Contact) -> MessageOutcome:
        """Send exactly ``request.text`` to the verified chat, once per approval."""
        text = str(request.text or "")
        if not text.strip():
            raise MessageError("there is no message to send")
        operation_id = self._operation_id.strip()
        if not operation_id:
            raise MessageError(
                "the message is not bound to an approved operation; refusing to send"
            )

        key = operation_guard_key(operation_id)
        try:
            recent = self._guard.is_recent(key)
        except OSError as exc:
            raise MessageError(
                f"the duplicate-send guard could not be read ({exc}); refusing to send"
            ) from exc
        if recent:
            raise MessageError(
                "this approved message was already attempted; refusing to send it again"
            )

        page = self._open_page()
        if page is None:
            raise MessageError("the WhatsApp browser session is not available; refusing to send")
        driver = WhatsAppDriver(page)
        try:
            driver.open_chat_by_search(
                contact.number,
                contact.name,
                name_is_unique=self._name_key_is_unique(contact),
            )
        except WhatsAppError as exc:
            raise MessageError(f"could not open the WhatsApp chat: {exc}") from exc
        try:
            driver.compose_text(text)
        except WhatsAppError as exc:
            raise MessageError(f"could not compose the WhatsApp message: {exc}") from exc

        # Claim immediately before the click: a retry, replan or restarted
        # daemon cannot send the same approved operation twice inside the window.
        try:
            claimed = self._guard.claim(key)
        except OSError as exc:
            raise MessageError(
                f"the duplicate-send guard could not be recorded ({exc}); refusing to send"
            ) from exc
        if not claimed:
            raise MessageError(
                "this approved message was already attempted; refusing to send it again"
            )

        # W5: remember the attempt before the irreversible click, so a crash
        # after the click still leaves the message verifiable.
        self._record_dispatch(contact, text)

        try:
            outcome = driver.send_text(text)
        except WhatsAppSendUncertain:
            logger.warning(
                "whatsapp text send outcome is unknown contact=%s phone_digits=%d chars=%d",
                contact.name,
                len(contact.number),
                len(text),
            )
            self._mark_dispatch("unknown")
            return MessageOutcome(
                channel=self.name,
                contact=contact.name,
                sent=True,
                detail=(
                    f"the WhatsApp Send action for {contact.name} was attempted but "
                    "its outcome is unknown; not retrying automatically"
                ),
                data={
                    "recipient": contact.name,
                    "phone_digits": contact.number,
                    "send_outcome": "unknown",
                    "delivery_verified": False,
                    "chat_confirmed": True,
                    "control": "",
                },
            )
        except WhatsAppError as exc:
            raise MessageError(
                f"the WhatsApp message send could not be confirmed: {exc} "
                "(the duplicate-send guard stays in place)"
            ) from exc

        self._mark_dispatch("dispatched")
        logger.info(
            "whatsapp text send dispatched contact=%s phone_digits=%d chars=%d",
            contact.name,
            len(contact.number),
            len(text),
        )
        return MessageOutcome(
            channel=self.name,
            contact=contact.name,
            sent=True,
            detail=f"dispatched the WhatsApp message to {contact.name}; delivery is not verified",
            data={
                "recipient": contact.name,
                "phone_digits": contact.number,
                "send_outcome": "dispatched",
                "delivery_verified": False,
                "chat_confirmed": True,
                "control": outcome.control,
            },
        )
