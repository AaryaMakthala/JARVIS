"""Guarded WhatsApp Web file send and chat open (Stage 8, Slice 8.4c).

Two tools live here, and they are deliberately separate:

* ``whatsapp_send_file`` -- attach exactly one local file to one verified chat
  and press Send.  Tier 2; requires the same content-hash approval binding as
  the rest of Slice 8.4 (the policy gate hashes the resolved path, act
  recomputes it immediately before dispatch and hands the approved digests to
  the tool, and the tool re-hashes again immediately before it clicks Send).
* ``whatsapp_open_chat`` -- open one verified chat for a contact and do nothing
  else.  Tier 1.  It never stages or sends a file, so "open Rahul's chat" and
  "send this to Rahul" cannot be confused.

The page is reached only through ``ToolContext.whatsapp_browser`` and the
narrow :class:`~jarvis.browser.whatsapp_web.WhatsAppPage` surface; no generic
DOM handle, URL or selector ever comes from the planner.  Every refusal is a
``ToolResult(ok=False, ...)`` with a message that is safe to show the owner:
no full phone numbers and no file contents.

Fail-closed rules honoured here:

* an unknown or ambiguous contact is refused with the known options (the brain
  is what asks the clarifying question -- the tool never guesses);
* the rate limit is checked before a browser is started or a chat is opened;
* the chat header must prove the contact identity before anything is staged;
* the attachment preview must be visible in the composer of that chat;
* the file must be bound to the approval by content, and the content must be
  byte-identical at the moment of the click;
* a send attempt is persisted in a short-lived guard so a retry, a replan or a
  daemon restart can never send the same file to the same contact twice.

Sending stays disabled unless the approval is content-bound: ``act`` places the
gate-approved sha256 digests on the tool context only after its own re-read of
the same resolved paths matches, and this tool refuses when they are absent.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from jarvis import config
from jarvis.browser.whatsapp_web import WhatsAppDriver, WhatsAppError
from jarvis.policy import paths
from jarvis.tools.base import ToolContext, ToolResult, ToolSpec
from jarvis.tools.whatsapp import (
    Contact,
    ContactStore,
    RateLimiter,
    RateLimitError,
    masked_number,
    resolve_single,
)

logger = logging.getLogger(__name__)

__all__ = [
    "APPROVED_HASHES_ATTR",
    "RESEND_WINDOW_S",
    "FileSendGuard",
    "WhatsAppOpenChatArgs",
    "WhatsAppSendFileArgs",
    "make_whatsapp_open_chat_spec",
    "make_whatsapp_send_file_spec",
]

#: ``act`` sets this attribute on the ``ToolContext`` with the sha256 digests the
#: policy gate bound the approval to, after act's own re-read matched them
#: (Slice 8.4d).  Reading it defensively keeps this tool fail-closed on an old
#: checkpoint or any dispatch that did not pass through the binding.
APPROVED_HASHES_ATTR = "approved_content_hashes"

#: How long a send attempt blocks an identical retry (same contact + content).
#: Long enough to absorb retries/replans/restarts, short enough that a genuine
#: "send it again" later in the day is not blocked forever.
RESEND_WINDOW_S = 600.0

#: Fallback cap when the policy module does not expose its content-hash cap.
_DEFAULT_HASH_CAP = 16 * 1024 * 1024

_MAX_DESCRIBE_CHARS = 100


class WhatsAppSendFileArgs(BaseModel):
    """Arguments for ``whatsapp_send_file`` -- a contact and one local file."""

    contact: str = Field(
        min_length=1, max_length=100, description="Contact name from `jarvis contacts`"
    )
    path: str = Field(
        min_length=1,
        max_length=4096,
        description=(
            "Absolute local file path to attach. The approval is bound to the "
            "file's sha256 content, not only to the path."
        ),
    )


class WhatsAppOpenChatArgs(BaseModel):
    """Arguments for ``whatsapp_open_chat`` -- a contact name only."""

    contact: str = Field(
        min_length=1, max_length=100, description="Contact name from `jarvis contacts`"
    )


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


def _guard_key(contact: Contact, digest: str) -> str:
    """Opaque key for one (contact number, content) send attempt."""
    payload = f"{contact.number}\n{digest}".encode()
    return hashlib.sha256(payload).hexdigest()


def _hash_cap(ctx: ToolContext) -> int:
    """The effective cap: the setting, never above the content-binding cap."""
    configured = max(1, int(ctx.settings.whatsapp.max_file_bytes))
    hard = int(getattr(paths, "CONTENT_HASH_MAX_BYTES", _DEFAULT_HASH_CAP))
    return min(configured, hard)


def _content_digest(target: Path, max_bytes: int) -> str | None:
    """sha256 of ``target``, or ``None`` when hashing is unavailable/refused."""
    content_hash = getattr(paths, "content_hash", None)
    if content_hash is None:
        return None
    try:
        return str(content_hash(target, max_bytes=max_bytes))
    except Exception:  # noqa: BLE001 - any refusal means "cannot bind this file"
        logger.warning("the attachment could not be hashed for content binding")
        return None


def _approved_digests(ctx: ToolContext) -> tuple[str, ...]:
    """The digests act/gate bound this dispatch to (empty when unwired)."""
    value = getattr(ctx, APPROVED_HASHES_ATTR, None)
    if not isinstance(value, (tuple, list)):
        return ()
    return tuple(str(item) for item in value if isinstance(item, str) and item)


def _describe_contact(store: ContactStore | None, query: str) -> str:
    """The owner-facing contact label with a masked number."""
    store = store or ContactStore(config.contacts_file())
    matches = store.resolve(query)
    if len(matches) == 1:
        return f"{query!r} ({masked_number(matches[0].number)})"
    return f"{query!r} (unknown contact)"


def _describe_send_file(store: ContactStore | None, args: WhatsAppSendFileArgs) -> str:
    """Confirmation summary: exactly which file and which (masked) contact."""
    who = _describe_contact(store, args.contact)
    name = Path(args.path).name or args.path
    if len(name) > _MAX_DESCRIBE_CHARS:
        name = name[: _MAX_DESCRIBE_CHARS - 1] + "\u2026"
    return f"send WhatsApp file {name!r} to {who} (full path: {args.path!r})"


def _describe_open_chat(store: ContactStore | None, args: WhatsAppOpenChatArgs) -> str:
    """Confirmation summary for opening a chat (no file is involved)."""
    return f"open the WhatsApp chat with {_describe_contact(store, args.contact)}"


def _require_windows(tool_name: str) -> ToolResult | None:
    """Refuse on a non-Windows host (the policy engine also blocks this)."""
    from jarvis.platform_guard import is_windows

    if not is_windows():
        return ToolResult(ok=False, error=f"{tool_name} is only supported on Windows")
    return None


def _resolve_contact(
    store: ContactStore | None, query: str
) -> tuple[Contact | None, ToolResult | None]:
    """Resolve exactly one contact, or the refusal listing the options."""
    store = store or ContactStore(config.contacts_file())
    contact, error = resolve_single(store, query)
    if contact is None:
        return None, ToolResult(
            ok=False, error=error or f"no contact matches {query!r}; refusing to continue"
        )
    return contact, None


def _resend_refusal(contact: Contact, guard: FileSendGuard) -> str:
    """The refusal text for a duplicate (contact, content) send attempt."""
    minutes = max(1, int(guard.window_s // 60))
    return (
        f"this file was already sent to {contact.name} in the last {minutes} minutes; "
        "refusing to send it again"
    )


def _limiter_for(limiter: RateLimiter | None, ctx: ToolContext) -> RateLimiter:
    """Use the injected limiter (tests) or the shared persisted one."""
    if limiter is not None:
        return limiter
    return RateLimiter(
        config.whatsapp_ratelimit_file(),
        max_per_hour=ctx.settings.whatsapp.max_per_hour,
        min_interval_s=ctx.settings.whatsapp.min_interval_s,
        clock=time.time,
    )


def _open_page(ctx: ToolContext) -> Any | None:
    """Start/return the WhatsApp profile's narrow page, or ``None`` (fail closed)."""
    holder = ctx.whatsapp_browser
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


def _resolve_file(raw: str, ctx: ToolContext) -> tuple[Path | None, int, ToolResult | None]:
    """Resolve the path and enforce the configured size cap."""
    try:
        target = paths.resolve_safe(raw)
    except paths.PathError as exc:
        return None, 0, ToolResult(ok=False, error=f"the file path is not allowed: {exc}")
    if not target.is_file():
        return (
            None,
            0,
            ToolResult(ok=False, error="the attachment is not a regular file; refusing to send"),
        )
    try:
        size = target.stat().st_size
    except OSError as exc:
        return None, 0, ToolResult(ok=False, error=f"the attachment could not be read: {exc}")
    cap = _hash_cap(ctx)
    if size > cap:
        return (
            None,
            0,
            ToolResult(
                ok=False,
                error=f"the attachment is larger than the {cap}-byte WhatsApp file cap; "
                "nothing was staged",
            ),
        )
    return target, size, None


def _send_file_run(
    store: ContactStore | None,
    limiter: RateLimiter | None,
    guard: FileSendGuard | None,
    args: WhatsAppSendFileArgs,
    ctx: ToolContext,
) -> ToolResult:
    """The full guarded flow; nothing is sent unless every check passes."""
    refusal = _require_windows("whatsapp_send_file")
    if refusal is not None:
        return refusal

    contact, refusal = _resolve_contact(store, args.contact)
    if refusal is not None:
        return refusal
    if contact is None:  # unreachable; keeps the type check honest
        return ToolResult(ok=False, error="the contact could not be resolved")

    limiter = _limiter_for(limiter, ctx)
    try:
        limiter.check()
    except RateLimitError as exc:
        return ToolResult(ok=False, error=str(exc))

    target, size, refusal = _resolve_file(args.path, ctx)
    if refusal is not None:
        return refusal
    if target is None:  # unreachable; keeps the type check honest
        return ToolResult(ok=False, error="the attachment could not be resolved")

    approved = _approved_digests(ctx)
    if not approved:
        return ToolResult(
            ok=False,
            error="the file's content is not bound to this approval; refusing to send",
        )
    digest = _content_digest(target, _hash_cap(ctx))
    if digest is None:
        return ToolResult(
            ok=False,
            error="the file's content could not be bound to this approval "
            "(missing, unreadable, or over the hash cap); refusing to send",
        )
    if digest not in approved:
        return ToolResult(
            ok=False,
            error="the file's content does not match the approved file; refusing to send",
        )

    guard = guard or FileSendGuard(config.whatsapp_sent_files_file())
    key = _guard_key(contact, digest)
    try:
        recent = guard.is_recent(key)
    except OSError as exc:
        return ToolResult(
            ok=False,
            error=f"the duplicate-send guard could not be read ({exc}); refusing to send",
        )
    if recent:
        return ToolResult(ok=False, error=_resend_refusal(contact, guard))

    page = _open_page(ctx)
    if page is None:
        return ToolResult(
            ok=False,
            error="the WhatsApp browser session is not available; refusing to send",
        )
    driver = WhatsAppDriver(page)

    try:
        driver.open_chat_by_phone(contact.number, contact.name)
    except WhatsAppError as exc:
        return ToolResult(ok=False, error=f"could not open the WhatsApp chat: {exc}")

    try:
        driver.attach(target, expect_name=target.name)
    except WhatsAppError as exc:
        return ToolResult(ok=False, error=f"could not stage the attachment: {exc}")

    # The gate hashed at approval time and act re-hashed immediately before
    # dispatch; this last read proves the bytes are still the approved ones at
    # the moment of the click (the file could have changed while attaching).
    if _content_digest(target, _hash_cap(ctx)) != digest:
        return ToolResult(
            ok=False,
            error="the file changed while the attachment was being prepared; "
            "refusing to send (the draft is left in WhatsApp)",
        )

    # Claim immediately before the click: a retry, replan or restarted daemon
    # cannot send the same bytes to the same contact twice inside the window.
    try:
        claimed = guard.claim(key)
    except OSError as exc:
        return ToolResult(
            ok=False,
            error=f"the duplicate-send guard could not be recorded ({exc}); refusing to send",
        )
    if not claimed:
        return ToolResult(ok=False, error=_resend_refusal(contact, guard))

    try:
        outcome = driver.send(expect_name=target.name)
    except WhatsAppError as exc:
        return ToolResult(
            ok=False,
            error=f"the WhatsApp file send could not be confirmed: {exc} "
            "(the duplicate-send guard stays in place)",
        )

    try:
        limiter.record()
    except OSError:
        # The send already happened; a bookkeeping failure must not turn this
        # into a retryable failure (that could send the file twice).
        logger.warning("the WhatsApp rate-limit record could not be written", exc_info=True)

    logger.info(
        "whatsapp file sent contact=%s phone_digits=%d name=%s bytes=%d",
        contact.name,
        len(contact.number),
        target.name,
        size,
    )
    return ToolResult(
        ok=True,
        output=f"sent WhatsApp file {target.name!r} to {contact.name}",
        verified=True,
        data={
            "recipient": contact.name,
            "phone_digits": contact.number,
            "file": target.name,
            "content_sha256": digest,
            "control": outcome.control,
            "staged_count": outcome.staged_count,
            "chat_confirmed": True,
        },
    )


def _open_chat_run(
    store: ContactStore | None,
    args: WhatsAppOpenChatArgs,
    ctx: ToolContext,
) -> ToolResult:
    """Open one verified chat and do nothing else (never stages a file)."""
    refusal = _require_windows("whatsapp_open_chat")
    if refusal is not None:
        return refusal

    contact, refusal = _resolve_contact(store, args.contact)
    if refusal is not None:
        return refusal
    if contact is None:  # unreachable; keeps the type check honest
        return ToolResult(ok=False, error="the contact could not be resolved")

    page = _open_page(ctx)
    if page is None:
        return ToolResult(
            ok=False, error="the WhatsApp browser session is not available; refusing to open"
        )
    driver = WhatsAppDriver(page)
    try:
        header = driver.open_chat_by_phone(contact.number, contact.name)
    except WhatsAppError as exc:
        return ToolResult(ok=False, error=f"could not open the WhatsApp chat: {exc}")
    return ToolResult(
        ok=True,
        output=f"opened the WhatsApp chat with {contact.name}",
        verified=True,
        data={"recipient": contact.name, "phone_digits": contact.number, "header": header},
    )


def make_whatsapp_send_file_spec(
    store: ContactStore | None = None,
    limiter: RateLimiter | None = None,
    guard: FileSendGuard | None = None,
) -> ToolSpec:
    """Build the ``whatsapp_send_file`` spec (Tier 2 - Confirm + unlocked).

    ``store`` / ``limiter`` / ``guard`` are injection points for tests; the
    defaults read the real contacts file, the shared send log and the persisted
    resend guard.  ``bind_content_hash`` opts this one tool into the 8.4d
    approval-to-content binding; no other tool is affected.
    """
    options: dict[str, Any] = {
        "name": "whatsapp_send_file",
        "description": (
            "Send exactly one local file to exactly one known WhatsApp contact "
            "through the isolated WhatsApp Web profile. Requires an unlocked "
            "JARVIS session. The approval is bound to the file's sha256 content. "
            "If the chat, the attachment preview or the content cannot be "
            "verified, nothing is sent and the draft is left for the user."
        ),
        "args_model": WhatsAppSendFileArgs,
        "base_tier": 2,
        "windows_only": True,
        "timeout_s": 120,
        "path_args": ("path",),
        "budget_cost": 1,
        "bind_content_hash": True,
        "run": lambda args, ctx: _send_file_run(store, limiter, guard, args, ctx),
        "describe": lambda args: _describe_send_file(store, args),
    }
    return ToolSpec(**options)


def make_whatsapp_open_chat_spec(store: ContactStore | None = None) -> ToolSpec:
    """Build the ``whatsapp_open_chat`` spec (Tier 1 - Confirm).

    Opening a chat is navigation only: it never stages or sends anything.  The
    URL is the driver's hard-coded ``web.whatsapp.com`` deep link, so the
    planner cannot supply one.
    """
    return ToolSpec(
        name="whatsapp_open_chat",
        description=(
            "Open the existing WhatsApp Web chat for exactly one known contact "
            "and stop. This only navigates to the conversation; it never sends "
            "a message or attaches a file."
        ),
        args_model=WhatsAppOpenChatArgs,
        base_tier=1,
        windows_only=True,
        timeout_s=60,
        budget_cost=1,
        run=lambda args, ctx: _open_chat_run(store, args, ctx),
        describe=lambda args: _describe_open_chat(store, args),
    )
