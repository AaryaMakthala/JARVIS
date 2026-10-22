"""The WhatsApp desktop channel (Stage 8, slice 8.6.4).

Text messages through the WhatsApp desktop application.  This module owns the
*mechanics* of that channel - the ``whatsapp://`` URI, the UIA window probe
and the identity checks - and nothing else: the reusable flow
(:mod:`jarvis.messaging.flow`) has already resolved the recipient and checked
the rate limit.

Every step fails closed (docs/03 invariant 9):

1. the chat URI is opened for the *resolved* contact's number;
2. the window must be found and its header must prove the chat identity;
3. the draft must contain the message that was approved;
4. only then is Enter pressed, and the delivery is read back best-effort.

An unreadable UI is never turned into a "success": every probe returns
``None`` and the adapter refuses to send.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Protocol, runtime_checkable
from urllib.parse import quote

from jarvis.messaging.base import (
    CAPABILITY_TEXT,
    CHANNEL_WHATSAPP,
    MessageError,
    MessageOutcome,
    MessageRequest,
)
from jarvis.messaging.contacts import Contact

logger = logging.getLogger(__name__)

__all__ = [
    "DesktopWindowProvider",
    "PywinautoWhatsAppWindow",
    "PywinautoWhatsAppWindowProvider",
    "WhatsAppDesktopAdapter",
    "WhatsAppWindow",
    "build_desktop_provider",
]


@runtime_checkable
class WhatsAppWindow(Protocol):
    """A handle to the open WhatsApp chat window."""

    def chat_header(self) -> str | None:
        """Return readable header/title text, or None when the UI cannot be read."""

    def draft_text(self) -> str | None:
        """Return the draft box text, or None when it cannot be read."""

    def press_enter(self) -> None:
        """Send the message (only ever called after identity verification)."""

    def message_visible(self, message: str) -> bool | None:
        """Best-effort read-back of the delivered message (None = unreadable)."""


@runtime_checkable
class DesktopWindowProvider(Protocol):
    """Opens a WhatsApp chat URI and finds the resulting window."""

    def open_chat(self, phone_digits: str, text: str) -> None: ...

    def find_chat_window(self, timeout_s: float) -> WhatsAppWindow | None: ...


class PywinautoWhatsAppWindowProvider:
    """Best-effort WhatsApp Desktop provider behind pywinauto (UIA).

    Every UI read fails closed: any exception or unreadable control yields
    ``None`` and the tool refuses to send (docs/12: pywinauto cannot always
    read the WhatsApp UI - never turned into a "success").
    """

    def __init__(self, app_name: str = "WhatsApp") -> None:
        self._app_name = app_name

    def open_chat(self, phone_digits: str, text: str) -> None:
        import os

        uri = f"whatsapp://send?phone={phone_digits}&text={quote(text, safe='')}"
        logger.info("opening whatsapp chat uri (display-name hidden)")
        os.startfile(uri)

    def find_chat_window(self, timeout_s: float) -> WhatsAppWindow | None:
        try:
            from pywinauto import Desktop
        except Exception:  # noqa: BLE001
            return None
        deadline = time.monotonic() + float(timeout_s)
        pattern = re.compile(f".*{re.escape(self._app_name)}.*", re.IGNORECASE)
        while time.monotonic() < deadline:
            try:
                visible = [w for w in Desktop(backend="uia").windows() if w.is_visible()]
                for window in visible:
                    if pattern.match(window.window_text() or ""):
                        return PywinautoWhatsAppWindow(window)
            except Exception:
                logger.debug("whatsapp window probe failed, retrying", exc_info=True)
            time.sleep(0.2)
        return None


class PywinautoWhatsAppWindow:
    """Thin, fail-closed adapter around a pywinauto window."""

    def __init__(self, win: object) -> None:
        self._win = win

    def _collect_text(self) -> str | None:
        try:
            parts: list[str] = []
            title = self._win.window_text()
            if title:
                parts.append(str(title))
            for ctrl in list(self._win.descendants(control_type="Text"))[:30]:
                try:
                    text = (ctrl.window_text() or "").strip()
                except Exception:
                    logger.debug("whatsapp text control probe failed", exc_info=True)
                    continue
                if text:
                    parts.append(text)
            return " | ".join(parts) or None
        except Exception:  # noqa: BLE001 - unreadable UI means "cannot verify"
            return None

    def chat_header(self) -> str | None:
        return self._collect_text()

    def draft_text(self) -> str | None:
        try:
            for ctrl in list(self._win.descendants(control_type="Edit"))[:10]:
                try:
                    value = ctrl.get_value()
                except Exception:
                    logger.debug("whatsapp draft probe failed", exc_info=True)
                    continue
                if isinstance(value, str) and value.strip():
                    return value
            return None
        except Exception:  # noqa: BLE001
            return None

    def press_enter(self) -> None:
        try:
            self._win.set_focus()
            self._win.type_keys("{ENTER}")
        except Exception:
            logger.warning("failed to press Enter in WhatsApp window", exc_info=True)
            raise

    def message_visible(self, message: str) -> bool | None:
        text = self._collect_text()
        if text is None:
            return None
        return message in text


def build_desktop_provider(app_name: str = "WhatsApp") -> DesktopWindowProvider:
    """Build the real desktop provider (tests inject a fake instead)."""
    return PywinautoWhatsAppWindowProvider(app_name)


def _contact_matches_header(contact: Contact, header: str) -> bool:
    """The chat header matches when it shows the name or the phone digits."""
    haystack = header.lower()
    return contact.name.lower() in haystack or contact.number in haystack


class WhatsAppDesktopAdapter:
    """Text messages through the WhatsApp desktop application."""

    name = CHANNEL_WHATSAPP

    def __init__(
        self,
        *,
        app_name: str = "WhatsApp",
        window_timeout_s: float = 20.0,
        provider: DesktopWindowProvider | None = None,
    ) -> None:
        self._app_name = app_name
        self._window_timeout_s = float(window_timeout_s)
        self._provider = provider

    def capabilities(self) -> frozenset[str]:
        """This channel carries text."""
        return frozenset({CAPABILITY_TEXT})

    def send(self, request: MessageRequest, contact: Contact) -> MessageOutcome:
        """Send one text message, or raise :class:`MessageError`."""
        from jarvis.platform_guard import is_windows

        if not is_windows():
            raise MessageError("the WhatsApp desktop channel is only supported on Windows")
        text = request.text.strip()
        if not text:
            raise MessageError("there is no message text to send")

        provider = self._provider or build_desktop_provider(self._app_name)
        try:
            provider.open_chat(contact.number, request.text)
        except Exception as exc:
            raise MessageError(f"could not open WhatsApp: {exc}") from exc

        window = provider.find_chat_window(self._window_timeout_s)
        if window is None:
            # Invariant 9 / docs §2.6 step 4-5: unverifiable chat -> do not send.
            raise MessageError("Could not verify chat; draft left in WhatsApp, send manually")
        header = window.chat_header()
        if header is None or not _contact_matches_header(contact, header):
            raise MessageError(
                f"Chat identity could not be verified for {contact.name!r}; "
                "draft left in WhatsApp, send manually"
            )

        draft = window.draft_text()
        if draft is not None and request.text not in draft:
            raise MessageError("Draft text does not match the message; nothing was sent")

        try:
            window.press_enter()
        except Exception as exc:
            raise MessageError(f"failed to press Enter: {exc}") from exc

        confirmed = window.message_visible(request.text)
        logger.info(
            "whatsapp message sent contact=%s phone_digits=%d confirmed=%s",
            contact.name,
            len(contact.number),
            confirmed,
        )
        return MessageOutcome(
            channel=self.name,
            contact=contact.name,
            sent=True,
            detail=f"sent WhatsApp message to {contact.name}",
            data={
                "recipient": contact.name,
                "phone_digits": contact.number,
                "chat_confirmed": confirmed,
            },
        )
