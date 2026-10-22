"""Reusable messaging flow and channel adapters (Stage 8, slice 8.6.4).

The messaging domain lives here, not in a tool:

* :mod:`jarvis.messaging.contacts` - the contact list, the phone-number
  masking and the persisted send rate limit;
* :mod:`jarvis.messaging.base` - the channel contract (a request, an outcome,
  an error);
* :mod:`jarvis.messaging.flow` - the one pipeline every channel uses
  (recipient -> rate limit -> payload -> delivery -> read-back);
* :mod:`jarvis.messaging.whatsapp_desktop` - text messages through the
  WhatsApp desktop application;
* :mod:`jarvis.messaging.whatsapp_web` - file messages through the isolated
  WhatsApp Web profile.

A new channel (mail, SMS, a second messaging app) is a new adapter plus a
thin tool; it never touches the safety rules, because those live in the flow
and in the policy engine.
"""

from __future__ import annotations

from jarvis.messaging.base import (
    CAPABILITY_FILE,
    CAPABILITY_TEXT,
    CHANNEL_WHATSAPP,
    CHANNEL_WHATSAPP_WEB,
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
    masked_number,
    normalize_number,
    resolve_single,
)
from jarvis.messaging.flow import MessageFlow, PreparedMessage

__all__ = [
    "CAPABILITY_FILE",
    "CAPABILITY_TEXT",
    "CHANNEL_WHATSAPP",
    "CHANNEL_WHATSAPP_WEB",
    "Contact",
    "ContactStore",
    "MessageChannel",
    "MessageError",
    "MessageFlow",
    "MessageOutcome",
    "MessageRequest",
    "PreparedMessage",
    "RateLimitError",
    "RateLimiter",
    "masked_number",
    "normalize_number",
    "resolve_single",
]
