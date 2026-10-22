"""Fail-closed WhatsApp Web driver (Stage 8, Slice 8.4b).

This module owns the *policy* of talking to WhatsApp Web; the DOM mechanics
live in :mod:`jarvis.browser.whatsapp_page`.  The two are split on purpose:
the policy is unit-testable with a fake page, and the adapter is a thin
Playwright wrapper.

Safety model (docs/03 invariant 9, ROADMAP_V2 Stage 8)
------------------------------------------------------
Every step fails closed.  If the chat identity cannot be proven, the attachment
cannot be proven, or the Send control is not unambiguous, **nothing is sent and
the draft is left in place** for the owner.

The gates, in order:

1. **Origin.** The page must be exactly ``https://web.whatsapp.com``.  A
   redirect anywhere else stops the task.  ``web.whatsapp.com`` is deliberately
   *not* in ``browser.allowed_domains`` - the general browser session can never
   reach it.
2. **Chat identity.** The conversation header must contain the contact's saved
   name or the phone digits.  An unreadable header is not a pass.
3. **Attachment preview.** The staged file's name must be visible *inside the
   chat composer* of the active chat panel, and the DOM node that names it
   must sit in a small container strictly below the panel.  A bare match
   elsewhere on the page (an old message, another pane) is not proof of what
   is staged and is refused.  This is the hard fail-closed gate the owner
   asked for.
4. **Send control.** The controls of the active chat are *enumerated* and
   matched by their accessible name - no selector is guessed.  Exactly one may
   look like Send, and it must state how many items are staged (``Send 1
   selected``).  A count other than one is refused; a bare ``Send`` is only
   accepted when the file input itself lists exactly the expected file.

The surface below is deliberately narrow.  There is no generic
``evaluate``/``locator``/``click`` escape hatch, so the planner (and the
``browser_*`` tools) cannot reach WhatsApp through it.  The only way in is
:class:`WhatsAppDriver`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from jarvis.policy.rules import url_origin

__all__ = [
    "WHATSAPP_BASE_URL",
    "WHATSAPP_ORIGIN",
    "WHATSAPP_SEND_URL",
    "AttachmentCandidate",
    "AttachmentScan",
    "SearchHit",
    "SendControl",
    "SendOutcome",
    "WhatsAppDriver",
    "WhatsAppError",
    "WhatsAppPage",
]

logger = logging.getLogger("jarvis.browser.whatsapp")

#: The only origin the driver will ever talk to.
WHATSAPP_ORIGIN = "https://web.whatsapp.com"
WHATSAPP_BASE_URL = f"{WHATSAPP_ORIGIN}/"
#: Deep link used to open a chat for a known phone number.
WHATSAPP_SEND_URL = f"{WHATSAPP_ORIGIN}/send?phone={{digits}}"

#: A Send control must say *what* it sends: "Send", "Send 1 selected", ...
_SEND_NAME_RE = re.compile(r"^send(?:\s+(?P<count>\d+))?(?:\s+selected)?$", re.IGNORECASE)

#: How much text the container holding the attachment evidence may hold.  An
#: attachment chip is a few characters; a chat transcript is thousands.  The
#: bound keeps "the file name appears somewhere in the chat" from passing as
#: "the file name is the staged preview".
_MAX_PREVIEW_CONTAINER_CHARS = 200


class WhatsAppError(RuntimeError):
    """A WhatsApp step could not be completed safely.

    Raised for every refusal (unverifiable chat, unscoped attachment preview,
    ambiguous Send control, ...).  The message is safe to show to the owner:
    it never contains message bodies or full phone numbers.
    """


@dataclass(frozen=True)
class SearchHit:
    """One WhatsApp search result (the left-hand pane)."""

    name: str
    detail: str = ""


@dataclass(frozen=True)
class SendControl:
    """One control that may be WhatsApp's Send button."""

    name: str
    index: int = 0


@dataclass(frozen=True)
class AttachmentCandidate:
    """One DOM node whose label or text mentions the staged file's name."""

    scope: str
    source: str
    text: str = ""


@dataclass(frozen=True)
class AttachmentScan:
    """The result of scanning the chat panel for the staged file's name.

    ``composer`` holds the candidates inside the composer (the accepted scope);
    ``panel`` holds the ones elsewhere in the chat panel (diagnostic only - the
    driver refuses to send on those).

    ``composer_container_below_panel`` records the *ancestry* check: the
    container that holds the candidates must sit strictly below the chat panel,
    so a match on the whole page (or on the panel itself) is never proof.
    ``composer_container_text_len`` bounds how much text that container holds -
    an attachment chip is small, a chat transcript is not.
    """

    panel_found: bool
    composer: tuple[AttachmentCandidate, ...] = ()
    panel: tuple[AttachmentCandidate, ...] = ()
    composer_container_below_panel: bool = False
    panel_container_below_panel: bool = False
    composer_container_text_len: int = 0
    panel_container_text_len: int = 0

    @property
    def composer_scoped(self) -> bool:
        """Whether the file name is visible in the composer of this panel."""
        return bool(self.composer) and self.composer_container_below_panel


@dataclass(frozen=True)
class SendOutcome:
    """What the driver did when it pressed Send."""

    header: str
    control: str
    staged_count: int | None = None


@runtime_checkable
class WhatsAppPage(Protocol):
    """The narrow page surface the WhatsApp driver is allowed to use.

    Implemented by :class:`jarvis.browser.whatsapp_page.PlaywrightWhatsAppPage`
    (live) and by test fakes.  There is deliberately no generic DOM handle:
    no ``evaluate``, no ``locator``, no free-form ``click``.
    """

    def url(self) -> str:
        """The current page URL."""

    def goto(self, url: str) -> None:
        """Navigate to an absolute URL."""

    def chat_header(self) -> str:
        """The conversation header text ("" when it cannot be read)."""

    def fill_search(self, query: str) -> None:
        """Type a query into the WhatsApp search box."""

    def search_results(self) -> tuple[SearchHit, ...]:
        """The visible search results (bounded)."""

    def open_search_result(self, hit: SearchHit) -> None:
        """Open exactly one search result; refuse when it is not unique."""

    def set_input_files(self, path: str) -> None:
        """Stage a file through the hidden ``input[type=file]`` (approach B)."""

    def staged_file_names(self) -> tuple[str, ...]:
        """The file names the file input currently holds (may be empty)."""

    def attachment_scan(self, filename: str) -> AttachmentScan:
        """Scan the chat panel for nodes naming ``filename``."""

    def send_controls(self) -> tuple[SendControl, ...]:
        """The named controls of the active chat (bounded)."""

    def click_send(self, name: str) -> None:
        """Click the single control whose accessible name is exactly ``name``."""


def normalize_text(value: str) -> str:
    """Collapse whitespace and strip a page-derived string."""
    return " ".join(str(value or "").split())


def digits_only(value: str) -> str:
    """Return the digits of ``value`` (no separators, no ``+``)."""
    return re.sub(r"\D", "", str(value or ""))


def header_matches(header: str, name: str = "", digits: str = "") -> bool:
    """Whether a chat header proves the identity of the open chat.

    The header passes when it shows the contact's saved name or the phone
    digits (separators and spacing are ignored).  An empty header never
    passes - "cannot read" is not "verified".
    """
    text = normalize_text(header)
    if not text:
        return False
    folded = text.casefold()
    if name and normalize_text(name).casefold() in folded:
        return True
    wanted = digits_only(digits)
    return bool(wanted and wanted in digits_only(text))


def is_send_name(name: str) -> bool:
    """Whether ``name`` looks like WhatsApp's Send control."""
    return bool(_SEND_NAME_RE.match(normalize_text(name)))


def send_count(name: str) -> int | None:
    """The number of items a Send control says it will send, if it says."""
    match = _SEND_NAME_RE.match(normalize_text(name))
    if match is None:
        return None
    raw = match.group("count")
    return int(raw) if raw is not None else None


class WhatsAppDriver:
    """Drives one WhatsApp Web chat, refusing anything it cannot prove.

    The driver holds no global state: every call takes its inputs as
    arguments and returns a small dataclass.  All refusals raise
    :class:`WhatsAppError`; nothing is ever sent implicitly.
    """

    def __init__(self, page: WhatsAppPage, *, logger: logging.Logger | None = None) -> None:
        """Wrap a narrow page surface (a live one or a fake)."""
        self._page = page
        self._logger = logger or logging.getLogger("jarvis.browser.whatsapp")
        self._expected_name = ""
        self._expected_digits = ""

    # ── opening a chat ─────────────────────────────────────────────────────

    def open_chat_by_phone(self, digits: str, name: str = "") -> str:
        """Open the chat for ``digits`` through the WhatsApp deep link.

        Returns the verified chat header.  Raises :class:`WhatsAppError` when
        the origin or the chat identity cannot be proven.
        """
        number = digits_only(digits)
        if not 8 <= len(number) <= 15:
            raise WhatsAppError("refusing to open a chat for a number that is not 8-15 digits")
        self._expected_name = normalize_text(name)
        self._expected_digits = number
        self._page.goto(WHATSAPP_SEND_URL.format(digits=number))
        self._require_origin()
        header = self._read_header()
        self._verify_header(header)
        self._logger.info("whatsapp chat opened for a verified contact (digits=%d)", len(number))
        return header

    def open_chat_by_name(self, name: str, digits: str = "") -> str:
        """Open the chat for a contact found through the WhatsApp search.

        The search result must be unique; an ambiguous result is refused
        instead of picking the first one.  The header check still applies.
        """
        query = normalize_text(name)
        if not query:
            raise WhatsAppError("a contact name is required to open a WhatsApp chat")
        self._expected_name = query
        self._expected_digits = digits_only(digits)
        self._page.goto(WHATSAPP_BASE_URL)
        self._require_origin()
        self._page.fill_search(query)
        hits = self._page.search_results()
        exact = [hit for hit in hits if normalize_text(hit.name).casefold() == query.casefold()]
        if len(exact) == 1:
            chosen = exact[0]
        elif len(exact) > 1:
            raise WhatsAppError("the WhatsApp search returned more than one exact match; refusing")
        elif len(hits) == 1:
            chosen = hits[0]
        elif not hits:
            raise WhatsAppError("the WhatsApp search returned no contact; refusing")
        else:
            raise WhatsAppError("the WhatsApp search result was not unique; refusing")
        self._page.open_search_result(chosen)
        header = self._read_header()
        self._verify_header(header)
        return header

    # ── attaching ──────────────────────────────────────────────────────────

    def attach(self, path: Path | str, expect_name: str = "") -> AttachmentScan:
        """Stage ``path`` in the composer and prove it is the expected file.

        Approach B (the hidden ``input[type=file]``) is used because it needs
        no clicks.  The staged file is only accepted when the file name is
        visible inside the composer of the verified chat.
        """
        target = Path(path)
        name = normalize_text(expect_name) or target.name
        if not name:
            raise WhatsAppError("the attachment has no file name to verify")
        if not target.is_file():
            raise WhatsAppError("the attachment file does not exist; nothing was staged")
        self._verify_header(self._read_header())
        self._page.set_input_files(str(target))
        scan = self._page.attachment_scan(name)
        if not scan.panel_found:
            raise WhatsAppError(
                "the WhatsApp chat panel could not be located; refusing to continue"
            )
        if not scan.composer:
            raise WhatsAppError(
                "the staged file's name is not visible in the chat composer; "
                "refusing to continue (the draft is left in WhatsApp)"
            )
        if not scan.composer_container_below_panel:
            raise WhatsAppError(
                "the attachment preview could not be scoped to the chat panel; refusing to continue"
            )
        if scan.composer_container_text_len > _MAX_PREVIEW_CONTAINER_CHARS:
            raise WhatsAppError(
                "the attachment preview could not be isolated from the chat; refusing to continue"
            )
        staged = self._page.staged_file_names()
        if staged:
            if name not in staged:
                raise WhatsAppError(
                    "the staged file does not match the expected attachment; refusing"
                )
            if len(staged) != 1:
                raise WhatsAppError("more than one file is staged; refusing to continue")
        self._logger.info("whatsapp attachment staged and verified (name=%s)", name)
        return scan

    # ── sending ────────────────────────────────────────────────────────────

    def send(self, expect_name: str = "") -> SendOutcome:
        """Press Send, but only when the chat, the file and the control agree.

        The header and the preview are re-verified immediately before the
        click, so a chat switch or a cleared preview stops the send.
        """
        name = normalize_text(expect_name)
        header = self._read_header()
        self._verify_header(header)
        if name:
            scan = self._page.attachment_scan(name)
            if not scan.composer_scoped:
                raise WhatsAppError(
                    "the attachment preview is no longer visible in the composer; refusing to send"
                )
        controls = self._page.send_controls()
        matches = [control for control in controls if is_send_name(control.name)]
        if not matches:
            raise WhatsAppError("no WhatsApp Send control was found; refusing to send")
        if len(matches) > 1:
            raise WhatsAppError("more than one WhatsApp Send control was found; refusing to send")
        control = matches[0]
        count = send_count(control.name)
        if count is None:
            staged = self._page.staged_file_names()
            if not staged or (name and name not in staged):
                raise WhatsAppError(
                    "the Send control does not say what it sends and the staged file is not "
                    "listed; refusing to send"
                )
        elif count != 1:
            raise WhatsAppError(f"{count} items are staged for sending; expected exactly one")
        self._page.click_send(normalize_text(control.name))
        self._logger.info("whatsapp send performed after verification (control=%r)", control.name)
        return SendOutcome(header=header, control=control.name, staged_count=count)

    # ── internals ──────────────────────────────────────────────────────────

    def _require_origin(self) -> None:
        """Refuse anything that is not exactly the WhatsApp origin."""
        origin = url_origin(self._page.url())
        if origin != WHATSAPP_ORIGIN:
            raise WhatsAppError(
                f"the WhatsApp page is not on {WHATSAPP_ORIGIN}; refusing to continue"
            )

    def _read_header(self) -> str:
        """Read the conversation header ("" when unreadable)."""
        try:
            return normalize_text(self._page.chat_header())
        except Exception as exc:  # noqa: BLE001 - an unreadable header is not a pass
            self._logger.warning("the WhatsApp chat header could not be read: %s", exc)
            return ""

    def _verify_header(self, header: str) -> None:
        """Fail closed unless the header proves the chat identity."""
        if not header:
            raise WhatsAppError("the WhatsApp chat header could not be read; refusing to continue")
        if not header_matches(header, self._expected_name, self._expected_digits):
            raise WhatsAppError(
                "the WhatsApp chat header does not match the requested contact; refusing to continue"
            )
