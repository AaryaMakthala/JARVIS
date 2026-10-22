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
import threading
import time
from pathlib import Path
from typing import Any

from jarvis.browser.whatsapp_web import WhatsAppDriver, WhatsAppError
from jarvis.messaging.base import (
    CAPABILITY_FILE,
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
    "FileSendGuard",
    "WhatsAppWebAdapter",
    "content_digest",
]

#: How long a send attempt blocks an identical retry (same contact + content).
#: Long enough to absorb retries/replans/restarts, short enough that a genuine
#: "send it again" later in the day is not blocked forever.
RESEND_WINDOW_S = 600.0

#: Fallback cap when the policy module does not expose its content-hash cap.
_DEFAULT_HASH_CAP = 16 * 1024 * 1024


class FileSendGuard:
    """Persisted, short-lived "this file was already attempted" guard.

    ``claim`` atomically records an attempt and returns ``False`` when the same
    (contact, content) pair was attempted inside :data:`RESEND_WINDOW_S`.  The
    key is a sha256, so the file holds no readable phone numbers or paths.
    This is a safety guard, not an authorization control: a corrupt/missing
    file is treated as empty, while the content-hash approval binding remains
    the security boundary.
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
        self._lock = threading.Lock()

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
