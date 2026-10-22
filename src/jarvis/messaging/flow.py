"""The reusable messaging flow (Stage 8, slice 8.6.4).

One pipeline, many channels.  Whatever the channel, the flow does the same
things in the same order:

1. **Channel.**  An unknown channel is refused (a typo must never become a
   silent no-op).
2. **Recipient.**  Exactly one contact must match - zero or several matches
   are a clarification, never a guess (the same rule the WhatsApp tools
   always used).
3. **Rate limit.**  The send rate limit is checked *before* any side effect,
   so a refused send never opens a browser or a window.
4. **Payload.**  Exactly one of ``text`` / ``path``, and the channel must be
   able to carry it.
5. **Delivery.**  The channel adapter does the work and fails closed.
6. **Read-back.**  The rate limit is recorded only after a successful
   delivery, and the outcome carries what the owner can be told.

A tool that needs to do policy work *between* the recipient check and the
delivery (resolving a file, binding the approval to its content) calls
:meth:`MessageFlow.prepare` and :meth:`MessageFlow.dispatch` separately.
That work has no side effects, so the order of the safety checks is
unchanged: recipient, rate limit, content binding, then delivery.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

from jarvis import config
from jarvis.messaging.base import (
    CAPABILITY_FILE,
    CAPABILITY_TEXT,
    MessageChannel,
    MessageError,
    MessageOutcome,
    MessageRequest,
)
from jarvis.messaging.contacts import (
    Contact,
    ContactStore,
    RateLimiter,
    RateLimitError,
    resolve_single,
)

logger = logging.getLogger(__name__)

__all__ = ["MessageFlow", "PreparedMessage"]


@dataclass(frozen=True)
class PreparedMessage:
    """A request whose recipient is resolved and whose rate limit was checked."""

    channel: MessageChannel
    contact: Contact


class MessageFlow:
    """The shared messaging pipeline every channel adapter plugs into."""

    def __init__(
        self,
        channels: Iterable[MessageChannel],
        *,
        store: ContactStore | None = None,
        limiter: RateLimiter | None = None,
    ) -> None:
        self._channels: dict[str, MessageChannel] = {}
        for channel in channels:
            self._channels[str(channel.name)] = channel
        self._contacts = store
        self._limiter = limiter

    # ── channels ──────────────────────────────────────────────────────────

    def channel_names(self) -> tuple[str, ...]:
        """The names of the registered channels."""
        return tuple(sorted(self._channels))

    def channel(self, name: str) -> MessageChannel:
        """Return one registered channel, or raise :class:`MessageError`."""
        channel = self._channels.get(str(name))
        if channel is None:
            known = ", ".join(self.channel_names()) or "(none)"
            raise MessageError(f"unknown messaging channel {name!r}; known channels: {known}")
        return channel

    # ── the flow ──────────────────────────────────────────────────────────

    def prepare(self, request: MessageRequest) -> PreparedMessage:
        """Resolve the recipient and check the rate limit.

        No side effect happens here: nothing is opened, staged or sent.
        """
        channel = self.channel(request.channel)
        contact = self.resolve_contact(request.contact)
        self._check_rate_limit()
        return PreparedMessage(channel=channel, contact=contact)

    def resolve_contact(self, query: str) -> Contact:
        """Resolve exactly one contact, or raise :class:`MessageError`."""
        contact, error = resolve_single(self._contact_store(), query)
        if contact is None:
            raise MessageError(error or f"no contact matches {query!r}")
        return contact

    def dispatch(self, prepared: PreparedMessage, request: MessageRequest) -> MessageOutcome:
        """Deliver a prepared request through its channel adapter."""
        self._check_payload(prepared.channel, request)
        try:
            outcome = prepared.channel.send(request, prepared.contact)
        except MessageError:
            raise
        except Exception as exc:
            raise MessageError(f"the message could not be sent: {exc}") from exc
        self._record_rate_limit()
        return outcome

    def run(self, request: MessageRequest) -> MessageOutcome:
        """Prepare and dispatch in one call."""
        return self.dispatch(self.prepare(request), request)

    # ── internals ────────────────────────────────────────────────────────

    def _contact_store(self) -> ContactStore:
        return self._contacts or ContactStore(config.contacts_file())

    def _check_rate_limit(self) -> None:
        limiter = self._limiter
        if limiter is None:
            return
        try:
            limiter.check()
        except RateLimitError as exc:
            raise MessageError(str(exc)) from exc

    def _record_rate_limit(self) -> None:
        limiter = self._limiter
        if limiter is None:
            return
        try:
            limiter.record()
        except OSError:
            # The message already went out; a bookkeeping failure must not
            # turn a delivered message into a retryable failure (that could
            # send it twice).
            logger.warning("the messaging rate-limit record could not be written", exc_info=True)

    def _check_payload(self, channel: MessageChannel, request: MessageRequest) -> None:
        """Exactly one of text / file, and the channel must be able to carry it."""
        has_text = bool(request.text.strip())
        has_file = bool(request.path.strip())
        if has_text and has_file:
            raise MessageError("a message carries either text or a file, never both")
        if not has_text and not has_file:
            raise MessageError("a message needs either text or a file")
        capabilities = frozenset(channel.capabilities())
        wanted = CAPABILITY_TEXT if has_text else CAPABILITY_FILE
        if wanted not in capabilities:
            raise MessageError(
                f"the {channel.name} channel cannot carry a {wanted} message "
                f"(it can carry: {', '.join(sorted(capabilities)) or 'nothing'})"
            )
