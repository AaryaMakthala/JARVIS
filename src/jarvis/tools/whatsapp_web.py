"""Guarded WhatsApp Web file send and chat open (Stage 8, Slices 8.4c/8.6.4).

Both tools are thin: the reusable messaging flow
(:mod:`jarvis.messaging.flow`) resolves the recipient and checks the send
rate limit, and the WhatsApp Web channel adapter
(:mod:`jarvis.messaging.whatsapp_web`) owns the browser mechanics and fails
closed.  What is left here is the *policy* work the tools must do before a
message can leave the machine: resolving the file inside the policy sandbox,
enforcing the size cap and binding the approval to the file's content.

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

import logging
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from jarvis import config
from jarvis.messaging.base import CHANNEL_WHATSAPP_WEB, MessageError, MessageRequest
from jarvis.messaging.contacts import (
    ContactStore,
    RateLimiter,
    make_recipient_precheck,
    masked_number,
)
from jarvis.messaging.flow import MessageFlow
from jarvis.messaging.whatsapp_web import (
    RESEND_WINDOW_S,
    FileSendGuard,
    WhatsAppWebAdapter,
    content_digest,
)
from jarvis.policy import paths
from jarvis.tools.base import ToolContext, ToolResult, ToolSpec

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


def _hash_cap(ctx: ToolContext) -> int:
    """The effective cap: the setting, never above the content-binding cap."""
    configured = max(1, int(ctx.settings.whatsapp.max_file_bytes))
    hard = int(getattr(paths, "CONTENT_HASH_MAX_BYTES", _DEFAULT_HASH_CAP))
    return min(configured, hard)


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


def _adapter(ctx: ToolContext, guard: FileSendGuard | None) -> WhatsAppWebAdapter:
    """The WhatsApp Web channel adapter, wired to this context."""
    return WhatsAppWebAdapter(
        holder=ctx.whatsapp_browser,
        guard=guard or FileSendGuard(config.whatsapp_sent_files_file()),
        max_bytes=_hash_cap(ctx),
    )


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

    store = store or ContactStore(config.contacts_file())
    limiter = _limiter_for(limiter, ctx)
    flow = MessageFlow([_adapter(ctx, guard)], store=store, limiter=limiter)
    request = MessageRequest(channel=CHANNEL_WHATSAPP_WEB, contact=args.contact)

    # 1. the recipient and the rate limit, before anything is opened
    try:
        prepared = flow.prepare(request)
    except MessageError as exc:
        return ToolResult(ok=False, error=str(exc))

    # 2. the file, its size cap and the approval's content binding
    target, _size, refusal = _resolve_file(args.path, ctx)
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
    digest = content_digest(target, _hash_cap(ctx))
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

    # 3. the delivery itself: the adapter re-hashes the file immediately
    #    before the click and claims the duplicate-send guard
    request = MessageRequest(
        channel=CHANNEL_WHATSAPP_WEB,
        contact=args.contact,
        path=str(target),
        digest=digest,
    )
    try:
        outcome = flow.dispatch(prepared, request)
    except MessageError as exc:
        return ToolResult(ok=False, error=str(exc))

    return ToolResult(
        ok=True,
        output=outcome.detail,
        verified=True,
        data=dict(outcome.data),
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

    store = store or ContactStore(config.contacts_file())
    adapter = _adapter(ctx, None)
    flow = MessageFlow([adapter], store=store)
    try:
        contact = flow.resolve_contact(args.contact)
        header = adapter.open_chat(contact)
    except MessageError as exc:
        return ToolResult(ok=False, error=str(exc))
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
        "precheck": make_recipient_precheck(store),
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
        precheck=make_recipient_precheck(store),
        describe=lambda args: _describe_open_chat(store, args),
    )
