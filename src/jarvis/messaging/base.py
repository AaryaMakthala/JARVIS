"""The messaging contract every channel adapter implements (Stage 8, 8.6.4).

A *channel* is one way of delivering a message to one contact: the WhatsApp
desktop app, the WhatsApp Web profile, and - later - mail or SMS.  The
contract is deliberately tiny:

* a channel names itself and says what it can carry (``text`` / ``file``);
* it sends one already-resolved request to one already-resolved contact.

Everything that must be identical for every channel - resolving the recipient,
respecting the send rate limit, and turning the result into a read-back -
lives in the flow (:mod:`jarvis.messaging.flow`), not here.

The types are plain dataclasses so a channel adapter can be tested without a
browser, a desktop or a network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from jarvis.messaging.contacts import Contact

__all__ = [
    "CAPABILITY_FILE",
    "CAPABILITY_TEXT",
    "CHANNEL_WHATSAPP",
    "CHANNEL_WHATSAPP_WEB",
    "MessageChannel",
    "MessageError",
    "MessageOutcome",
    "MessageRequest",
]

#: The desktop WhatsApp application (text messages).
CHANNEL_WHATSAPP = "whatsapp"

#: The isolated WhatsApp Web profile in the managed browser (file messages).
CHANNEL_WHATSAPP_WEB = "whatsapp_web"

#: What a channel can carry.
CAPABILITY_TEXT = "text"
CAPABILITY_FILE = "file"


class MessageError(RuntimeError):
    """A message could not be delivered safely.

    The message is safe to show to the owner: it never contains a full phone
    number, a file's contents or the message body.
    """


@dataclass(frozen=True)
class MessageRequest:
    """One message for one contact on one channel.

    ``path`` and ``digest`` are only set for a file message.  ``digest`` is the
    sha256 the *approval* was bound to; the channel re-reads the file and
    compares it right before the irreversible step, so a file that changed
    since the approval is never sent.
    """

    channel: str
    contact: str
    text: str = ""
    path: str = ""
    digest: str = ""


@dataclass(frozen=True)
class MessageOutcome:
    """What a channel did with a request."""

    channel: str
    contact: str
    sent: bool
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class MessageChannel(Protocol):
    """One delivery channel (WhatsApp desktop, WhatsApp Web, ...)."""

    name: str

    def capabilities(self) -> frozenset[str]:
        """The message kinds this channel can carry."""

    def send(self, request: MessageRequest, contact: Contact) -> MessageOutcome:
        """Deliver ``request`` to ``contact`` or raise :class:`MessageError`."""
