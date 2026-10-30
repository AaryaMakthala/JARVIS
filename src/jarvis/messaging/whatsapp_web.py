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
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jarvis.browser.whatsapp_web import (
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
from jarvis.policy import paths

logger = logging.getLogger(__name__)

__all__ = [
    "RESEND_WINDOW_S",
    "ConversationRead",
    "FileSendGuard",
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

#: One lock per guard file, shared by every instance in this process.  The tool
#: builds a fresh :class:`FileSendGuard` for each dispatch, so an instance lock
#: alone would let two concurrent dispatches of the *same* approval both read
#: the file before either writes it.  The lock makes ``claim`` atomic for every
#: guard pointed at the same file (the persisted file still covers restarts).
_GUARD_LOCKS: dict[str, threading.Lock] = {}
_GUARD_LOCKS_GUARD = threading.Lock()


def _guard_lock(path: Path) -> threading.Lock:
    """Return the process-wide lock for one guard file (created on demand)."""
    key = os.path.normcase(str(path.resolve(strict=False)))
    with _GUARD_LOCKS_GUARD:
        lock = _GUARD_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _GUARD_LOCKS[key] = lock
        return lock


@dataclass(frozen=True)
class ConversationRead:
    """The result of a read-only WhatsApp conversation read.

    ``messages`` are already bounded and chronological.  The message text is
    untrusted external content: it is returned to the tool, never to the
    planner as instructions.
    """

    header: str
    messages: tuple[WhatsAppMessage, ...]


class FileSendGuard:
    """Persisted, short-lived "this operation was already attempted" guard.

    ``claim`` atomically records an attempt and returns ``False`` when the same
    opaque key was attempted inside :data:`RESEND_WINDOW_S`.  The file send uses
    a sha256 of (contact number, content digest); the text send uses a sha256 of
    the approved operation id.  Either way the file holds no readable phone
    numbers, paths or message text.  This is a safety guard, not an
    authorization control: a corrupt/missing file is treated as empty, while the
    approval binding remains the security boundary.

    Every guard for the same file shares one process-wide lock, so concurrent
    dispatches of the same approval cannot both claim it (the tool constructs a
    new guard per invocation).  The persisted file covers restarts; a second
    *process* is outside this lock and relies on the recorded window.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        window_s: float = RESEND_WINDOW_S,
        clock: Any = None,
    ) -> None:
        self.path = Path(path)
        self.window_s = float(window_s)
        self._clock = clock if clock is not None else time.time
        self._lock = _guard_lock(self.path)

    def is_recent(self, key: str) -> bool:
        """Whether ``key`` was attempted inside the window (read-only)."""
        with self._lock:
            entries = self._prune(self._load(), float(self._clock()))
            return key in entries

    def claim(self, key: str) -> bool:
        """Atomically record ``key``; ``False`` when an attempt is recent."""
        with self._lock:
            now = float(self._clock())
            entries = self._prune(self._load(), now)
            if key in entries:
                return False
            entries[key] = now
            self._write(entries)
            return True

    def _load(self) -> dict[str, float]:
        if not self.path.is_file():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            records = data.get("entries", {})
        except Exception:  # noqa: BLE001 - a bad guard file must not authorize a send
            logger.warning("the WhatsApp resend guard is unreadable; treating it as empty")
            return {}
        if not isinstance(records, dict):
            return {}
        entries: dict[str, float] = {}
        for key, timestamp in records.items():
            try:
                entries[str(key)] = float(timestamp)
            except (TypeError, ValueError):
                continue
        return entries

    def _prune(self, entries: dict[str, float], now: float) -> dict[str, float]:
        cutoff = now - self.window_s
        return {key: ts for key, ts in entries.items() if ts > cutoff}

    def _write(self, entries: dict[str, float]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"entries": entries}), encoding="utf-8")
        tmp.replace(self.path)


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
        max_bytes: int = _DEFAULT_HASH_CAP,
    ) -> None:
        self._holder = holder
        self._guard = guard
        self._max_bytes = int(max_bytes)

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
            return driver.open_chat_by_search(contact.number, contact.name)
        except WhatsAppError as exc:
            raise MessageError(f"could not open the WhatsApp chat: {exc}") from exc

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
            header = driver.open_chat_by_search(contact.number, contact.name)
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
    ) -> None:
        super().__init__(holder=holder, guard=guard)
        self._operation_id = str(operation_id or "")

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
            driver.open_chat_by_search(contact.number, contact.name)
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

        try:
            outcome = driver.send_text(text)
        except WhatsAppSendUncertain:
            logger.warning(
                "whatsapp text send outcome is unknown contact=%s phone_digits=%d chars=%d",
                contact.name,
                len(contact.number),
                len(text),
            )
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
