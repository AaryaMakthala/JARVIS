"""JARVIS-owned managed browser package (Stage 7, Slice 6).

A single headed, JARVIS-owned Chrome context driven through Playwright for
``browser_goto`` / ``browser_read`` / ``browser_click`` / ``browser_type``.

Stage 8 adds a second, *narrow* profile: the WhatsApp driver
(:class:`~jarvis.browser.whatsapp_web.WhatsAppDriver`) talks to WhatsApp Web
through :class:`~jarvis.browser.whatsapp_page.PlaywrightWhatsAppPage`.  That
surface is deliberately not reachable from the ``browser_*`` tools.

Importing this package never starts Chrome and never requires Playwright to be
installed (lifecycle only happens on :meth:`PlaywrightBrowser.start`).
"""

from __future__ import annotations

from jarvis.browser.context import (
    BrowserError,
    BrowserSession,
    ElementRef,
    Observation,
    PlaywrightBrowser,
    profile_lock,
)
from jarvis.browser.lazy import LazyProfileBrowser
from jarvis.browser.tools import (
    BrowserClickArgs,
    BrowserGotoArgs,
    BrowserReadArgs,
    BrowserTypeArgs,
    make_browser_click_spec,
    make_browser_goto_spec,
    make_browser_read_spec,
    make_browser_type_spec,
)
from jarvis.browser.whatsapp_page import PlaywrightWhatsAppPage
from jarvis.browser.whatsapp_web import (
    WHATSAPP_BASE_URL,
    WHATSAPP_ORIGIN,
    WHATSAPP_SEND_URL,
    AttachmentCandidate,
    AttachmentScan,
    SearchHit,
    SendControl,
    SendOutcome,
    WhatsAppDriver,
    WhatsAppError,
    WhatsAppPage,
)

__all__ = [
    "WHATSAPP_BASE_URL",
    "WHATSAPP_ORIGIN",
    "WHATSAPP_SEND_URL",
    "AttachmentCandidate",
    "AttachmentScan",
    "BrowserClickArgs",
    "BrowserError",
    "BrowserGotoArgs",
    "BrowserReadArgs",
    "BrowserSession",
    "BrowserTypeArgs",
    "ElementRef",
    "LazyProfileBrowser",
    "Observation",
    "PlaywrightBrowser",
    "PlaywrightWhatsAppPage",
    "SearchHit",
    "SendControl",
    "SendOutcome",
    "WhatsAppDriver",
    "WhatsAppError",
    "WhatsAppPage",
    "make_browser_click_spec",
    "make_browser_goto_spec",
    "make_browser_read_spec",
    "make_browser_type_spec",
    "profile_lock",
]
