"""The ``whatsapp_send`` tool (Phase 6; Stage 8, slice 8.6.4).

Single-recipient ``whatsapp_send`` per docs/04_TOOLS_SPEC.md §2.6:

1. Load ``contacts.json`` (edited only via ``jarvis contacts …``) and resolve
   the recipient; zero or multiple matches → error listing options.
2. Enforce rate limits *before* opening any window.
3. Open ``whatsapp://send?phone=<digits>&text=<urlencoded>``.
4. Wait for the WhatsApp window; **verify the chat identity from the UI**.
   If the chat header cannot be read or does not match the contact → never
   press Enter (fail closed, docs/03 §1); the draft is left for the user.
5. Verify the message text is present in the draft box (when readable),
   press Enter, then best-effort confirm delivery.

The WhatsApp window is reached only through :class:`DesktopWindowProvider`, so
tests inject a fake provider and never touch the desktop (docs/06 Phase 6).

Slice 8.6.4 moved the messaging rules into :mod:`jarvis.messaging`: the
contact list, the rate limit and the *flow* (resolve the recipient → check the
rate limit → dispatch to the channel) are shared with every other channel,
and this tool now only adapts the shared flow to the WhatsApp desktop
channel.  The names below are re-exported so the old import paths keep
working.
"""

from __future__ import annotations

import logging
import time

from pydantic import BaseModel, Field

from jarvis import config
from jarvis.messaging.base import CHANNEL_WHATSAPP, MessageError, MessageRequest
from jarvis.messaging.contacts import (
    Contact,
    ContactStore,
    RateLimiter,
    RateLimitError,
    make_recipient_precheck,
    masked_number,
    normalize_number,
    resolve_single,
)
from jarvis.messaging.flow import MessageFlow
from jarvis.messaging.whatsapp_desktop import (
    DesktopWindowProvider,
    WhatsAppDesktopAdapter,
    build_desktop_provider,
)
from jarvis.tools.base import ToolContext, ToolResult, ToolSpec

logger = logging.getLogger(__name__)

__all__ = [
    "Contact",
    "ContactStore",
    "DesktopWindowProvider",
    "RateLimitError",
    "RateLimiter",
    "WhatsAppSendArgs",
    "make_whatsapp_send_spec",
    "masked_number",
    "normalize_number",
    "resolve_single",
]


# ── the tool ─────────────────────────────────────────────────────────────


class WhatsAppSendArgs(BaseModel):
    """Arguments for the ``whatsapp_send`` tool."""

    contact: str = Field(
        min_length=1, max_length=100, description="Contact name from `jarvis contacts`"
    )
    message: str = Field(min_length=1, max_length=1000, description="Message text (max 1000 chars)")


def _build_provider(settings: config.Settings) -> DesktopWindowProvider:
    """Build the real desktop provider (monkeypatched in tests)."""
    return build_desktop_provider(settings.whatsapp.app_name)


def _whatsapp_send_run(
    store: ContactStore | None,
    limiter: RateLimiter | None,
    args: WhatsAppSendArgs,
    ctx: ToolContext,
) -> ToolResult:
    """Send through the shared flow; never sends unless the chat is verified."""
    from jarvis.platform_guard import is_windows

    if not is_windows():
        return ToolResult(ok=False, error="whatsapp_send is only supported on Windows")

    store = store or ContactStore(config.contacts_file())
    limiter = limiter or RateLimiter(
        config.whatsapp_ratelimit_file(),
        max_per_hour=ctx.settings.whatsapp.max_per_hour,
        min_interval_s=ctx.settings.whatsapp.min_interval_s,
        clock=time.time,
    )
    adapter = WhatsAppDesktopAdapter(
        app_name=ctx.settings.whatsapp.app_name,
        window_timeout_s=ctx.settings.whatsapp.window_timeout_s,
        provider=_build_provider(ctx.settings),
    )
    flow = MessageFlow([adapter], store=store, limiter=limiter)
    request = MessageRequest(channel=CHANNEL_WHATSAPP, contact=args.contact, text=args.message)
    try:
        outcome = flow.run(request)
    except MessageError as exc:
        return ToolResult(ok=False, error=str(exc))

    return ToolResult(
        ok=True,
        output=outcome.detail,
        data=dict(outcome.data),
    )


def _describe_whatsapp(store: ContactStore | None, args: WhatsAppSendArgs) -> str:
    """Confirmation summary: masked number + exact message (docs §2.6 step 2)."""
    store = store or ContactStore(config.contacts_file())
    masked = "unknown contact"
    matches = store.resolve(args.contact)
    if len(matches) == 1:
        masked = masked_number(matches[0].number)
    return f"send WhatsApp message to {args.contact!r} ({masked}): {args.message!r}"


def _whatsapp_send_verify(
    _store: ContactStore | None,
    _limiter: RateLimiter | None,
    args: WhatsAppSendArgs,
    result: ToolResult,
    ctx: ToolContext,
) -> ToolResult:
    """Attach the best-effort read-back confirmation to a successful send."""
    if not result.ok:
        return result
    confirmed = result.data.get("chat_confirmed")
    return result.model_copy(update={"verified": confirmed})


def make_whatsapp_send_spec(
    store: ContactStore | None = None,
    limiter: RateLimiter | None = None,
) -> ToolSpec:
    """Build the ``whatsapp_send`` spec (Tier 2 - Confirm + unlocked).

    ``store`` / ``limiter`` are optional injection points for tests; the
    default spec reads the real contacts file and rate-limit log.
    """
    return ToolSpec(
        name="whatsapp_send",
        description=(
            "Send a WhatsApp message to exactly one known contact. Requires an "
            "unlocked JARVIS session. If the open WhatsApp chat cannot be "
            "verified against the contact, nothing is sent and the draft is "
            "left for the user."
        ),
        args_model=WhatsAppSendArgs,
        base_tier=2,
        windows_only=True,
        timeout_s=30,
        run=lambda args, ctx: _whatsapp_send_run(store, limiter, args, ctx),
        precheck=make_recipient_precheck(store),
        verify=lambda args, result, ctx: _whatsapp_send_verify(store, limiter, args, result, ctx),
        describe=lambda args: _describe_whatsapp(store, args),
    )
