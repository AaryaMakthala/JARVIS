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
import unicodedata
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol, runtime_checkable

from jarvis.messaging.resolver import normalize_name
from jarvis.policy.rules import url_origin

__all__ = [
    "MAX_CONVERSATION_NAME_CHARS",
    "MAX_CONVERSATION_PREVIEW_CHARS",
    "MAX_CONVERSATION_TIMESTAMP_CHARS",
    "MAX_RECENT_MESSAGES",
    "MAX_STATUS_HINTS",
    "MAX_UNREAD_CONVERSATIONS",
    "WHATSAPP_BASE_URL",
    "WHATSAPP_ORIGIN",
    "WHATSAPP_SEND_URL",
    "AttachmentCandidate",
    "AttachmentScan",
    "ConversationSummary",
    "SearchHit",
    "SendControl",
    "SendOutcome",
    "WhatsAppDriver",
    "WhatsAppError",
    "WhatsAppLoggedOut",
    "WhatsAppMessage",
    "WhatsAppNotReady",
    "WhatsAppPage",
    "WhatsAppQRRequired",
    "WhatsAppSendUncertain",
    "WhatsAppSessionError",
    "WhatsAppSessionState",
    "WhatsAppSessionUnknown",
    "composer_matches",
    "header_confirms_recipient",
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

#: Readiness budget used after every WhatsApp navigation.  The page adapter
#: clamps it to its own hard maximum, so a caller cannot raise it arbitrarily.
_READY_TIMEOUT_S = 20.0

#: Hard cap on how many recent messages one read may return.  A read is a
#: bounded look at the open conversation, never a conversation dump.
MAX_RECENT_MESSAGES = 20

#: W6: hard cap on how many unread conversations one discovery may return.
#: Discovery is a bounded look at the visible conversation list, never a dump.
MAX_UNREAD_CONVERSATIONS = 20

#: W6: per-field bounds for one conversation-list row.  These are display
#: bound only; the fields are still untrusted external content.
MAX_CONVERSATION_NAME_CHARS = 120
MAX_CONVERSATION_PREVIEW_CHARS = 160
MAX_CONVERSATION_TIMESTAMP_CHARS = 40

#: W5: bounds on the status evidence one message may report.
MAX_STATUS_HINTS = 8
MAX_STATUS_HINT_CHARS = 80


class WhatsAppError(RuntimeError):
    """A WhatsApp step could not be completed safely.

    Raised for every refusal (unverifiable chat, unscoped attachment preview,
    ambiguous Send control, ...).  The message is safe to show to the owner:
    it never contains message bodies or full phone numbers.
    """


class WhatsAppSessionState(str, Enum):
    """The authentication state of the WhatsApp Web page.

    Produced by one read-only probe (``WhatsAppPage.session_state``) and never
    inferred from the chat header.  ``UNKNOWN`` is the fail-closed default: a
    page that cannot be classified is never treated as authenticated.
    """

    AUTHENTICATED = "authenticated"
    QR_REQUIRED = "qr_required"
    LOGGED_OUT = "logged_out"
    UNKNOWN = "unknown"


#: Owner-facing refusal per state.  Safe to show: no QR payload, cookie, token,
#: message body or phone number is ever included.
_SESSION_MESSAGES: dict[WhatsAppSessionState, str] = {
    WhatsAppSessionState.AUTHENTICATED: "the WhatsApp session is authenticated",
    WhatsAppSessionState.QR_REQUIRED: (
        "WhatsApp Web needs a QR-code sign-in; open the WhatsApp browser profile "
        "and link this device before trying again"
    ),
    WhatsAppSessionState.LOGGED_OUT: (
        "the WhatsApp Web session is logged out; link this device again before trying again"
    ),
    WhatsAppSessionState.UNKNOWN: (
        "the WhatsApp Web session could not be authenticated; refusing to continue"
    ),
}


class WhatsAppSessionError(WhatsAppError):
    """The WhatsApp session is not authenticated; nothing was staged or sent.

    Carries the detected :attr:`state`, so a caller can report the real cause
    (QR sign-in needed, logged out, unrecognizable page) instead of a generic
    chat-header failure.  Still a :class:`WhatsAppError`, so every existing
    ``except WhatsAppError`` site keeps working.
    """

    def __init__(self, state: WhatsAppSessionState, message: str = "") -> None:
        """Record the detected state and the owner-facing refusal text."""
        self.state = state
        super().__init__(message or _SESSION_MESSAGES[state])


class WhatsAppQRRequired(WhatsAppSessionError):
    """WhatsApp Web is showing the QR-code sign-in surface."""


class WhatsAppLoggedOut(WhatsAppSessionError):
    """WhatsApp Web reports that this device/session was logged out."""


class WhatsAppSessionUnknown(WhatsAppSessionError):
    """The page could not be classified, so authentication cannot be assumed."""


class WhatsAppNotReady(WhatsAppSessionUnknown):
    """The page stayed unrecognizable for the whole readiness budget."""


class WhatsAppSendUncertain(WhatsAppError):
    """The Send click was attempted, but its result could not be observed.

    A subclass of :class:`WhatsAppError` so every existing refusal path keeps
    working.  It is distinct because the caller must **not** treat it as "not
    sent": the click may have reached WhatsApp.  The caller reports the outcome
    as unknown and never retries automatically.
    """


@dataclass(frozen=True)
class SearchHit:
    """One WhatsApp search result (the left-hand pane)."""

    name: str
    detail: str = ""


@dataclass(frozen=True)
class ConversationSummary:
    """One visible row of the WhatsApp conversation list (W6 discovery).

    The display name, preview and timestamp are **untrusted external content**:
    they are data for the owner, never instructions.  Only fields the page
    actually exposed are set; ``unread_count`` stays ``None`` when no count is
    visible and the other text fields stay empty when unavailable.  Nothing in
    this dataclass proves contact identity - that is decided against
    ``ContactStore`` by the tool layer.
    """

    name: str
    unread: bool = False
    unread_count: int | None = None
    preview: str = ""
    timestamp: str = ""


@dataclass(frozen=True)
class WhatsAppMessage:
    """One visible text message from the verified conversation.

    The message text is **untrusted external content**.  It is data for the
    owner and the summarizer, never an instruction to JARVIS.  ``sender`` and
    ``timestamp`` are empty when the page did not expose them; ``is_outgoing``
    is ``None`` when the page did not say.  ``sequence`` is the chronological
    position in this bounded read (0 is the oldest returned message).
    """

    text: str
    sender: str = ""
    timestamp: str = ""
    is_outgoing: bool | None = None
    sequence: int = 0
    #: Status evidence the page exposed for this bubble (W5): the
    #: ``data-icon`` names and accessible labels of its status node.  Empty
    #: when the page exposed none.  Classification lives in
    #: :mod:`jarvis.messaging.verification` and only understands WhatsApp's
    #: own status vocabulary, so unknown markup stays ``unknown``.
    status_hints: tuple[str, ...] = ()


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

    def session_state(self) -> WhatsAppSessionState:
        """The page's authentication state (read-only probe)."""

    def wait_for_ready(self, timeout_s: float = _READY_TIMEOUT_S) -> WhatsAppSessionState:
        """Poll the session state, bounded, until it is recognizable."""

    def chat_header(self) -> str:
        """The conversation header text ("" when it cannot be read)."""

    def conversation_summaries(self, limit: int) -> tuple[ConversationSummary, ...]:
        """The visible unread conversation rows (read-only; never clicks)."""

    def recent_messages(self, limit: int) -> tuple[WhatsAppMessage, ...]:
        """The most recent visible text messages of the open conversation."""

    def fill_search(self, query: str) -> None:
        """Type a query into the WhatsApp search box."""

    def search_results(self) -> tuple[SearchHit, ...]:
        """The visible search results (bounded)."""

    def open_search_result(self, hit: SearchHit) -> None:
        """Open exactly one search result; refuse when it is not unique."""

    def fill_composer(self, text: str) -> None:
        """Replace the active chat composer's contents with ``text``."""

    def composer_text(self) -> str:
        """The active chat composer's visible text ("" when it cannot be read)."""

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
    digits (separators and spacing are ignored).  The name comparison uses the
    shared emoji-insensitive key, so a header showing "Mom" proves a saved
    "Mom \u2764\ufe0f".  An empty header never passes - "cannot read" is not "verified".
    """
    text = normalize_text(header)
    if not text:
        return False
    wanted = digits_only(digits)
    if wanted and wanted in digits_only(text):
        return True
    expected = normalize_name(name)
    return bool(expected) and normalize_name(text) == expected


def _exact_name_key(value: str) -> str:
    """Exact saved-name key: NFC + casefold + whitespace collapse, emoji kept.

    Only used to decide whether an ambiguous name-only header contains the
    *whole* saved display name (including its decoration), which is the one
    name-only form that can distinguish two contacts sharing an emoji-ignored
    key.
    """
    return unicodedata.normalize("NFC", normalize_text(value)).casefold()


def header_confirms_recipient(
    header: str,
    name: str = "",
    digits: str = "",
    *,
    unique_name: bool = True,
) -> bool:
    """Whether a chat header proves this recipient (search routes).

    When the header shows a number it must contain the expected number, so a
    header naming a different number is refused even when the name matches.

    A header with no number is compared with the saved name using the same
    emoji-insensitive key as contact resolution.  ``unique_name`` records
    whether the trusted contact is the only ContactStore entry with that key:
    when it is not, an emoji-only difference cannot distinguish the contacts,
    so only the whole exact saved name is accepted (a bare "Mom" or a different
    decoration is refused).  An unmatched ordinary word, a substring, or an
    unrelated name never passes.
    """
    text = normalize_text(header)
    if not text:
        return False
    seen = digits_only(text)
    wanted = digits_only(digits)
    if seen:
        return bool(wanted) and wanted in seen
    expected = normalize_name(name)
    if not expected:
        # A name made only of emoji has no letters/digits to compare: it is
        # inherently ambiguous, so only the whole exact saved name is proof.
        return _exact_name_key(text) == _exact_name_key(name)
    if normalize_name(text) != expected:
        return False
    return unique_name or _exact_name_key(text) == _exact_name_key(name)


def _composer_visible(value: str) -> str:
    """Normalise a composer value for exact comparison (never for sending)."""
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u200b", "").replace("\ufeff", "").replace("\u00a0", " ")
    # WhatsApp's contenteditable often exposes a trailing newline; remove only
    # that artifact, never internal newlines or spaces the owner typed.
    return text.rstrip("\n")


def composer_matches(expected: str, actual: str) -> bool:
    """Whether the composer holds exactly the approved message text.

    Newlines are preserved; only invisible formatting differences that WhatsApp
    itself introduces (CRLF, zero-width characters, non-breaking spaces and a
    trailing newline) are ignored.  Any other difference is a refusal.
    """
    return _composer_visible(expected) == _composer_visible(actual)


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


def _session_error(state: WhatsAppSessionState) -> WhatsAppSessionError:
    """Build the specific refusal for an unauthenticated state."""
    if state is WhatsAppSessionState.QR_REQUIRED:
        return WhatsAppQRRequired(state)
    if state is WhatsAppSessionState.LOGGED_OUT:
        return WhatsAppLoggedOut(state)
    return WhatsAppSessionUnknown(state)


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
        self._expected_name_is_unique = True

    # ── session ────────────────────────────────────────────────────────────

    def require_authenticated(self) -> WhatsAppSessionState:
        """Refuse unless the page proves an authenticated WhatsApp session.

        The exact-origin check runs first, so the session probe never touches a
        page that is not exactly ``https://web.whatsapp.com``.  A probe failure,
        an unrecognized value or an unreadable page is ``unknown`` and refuses:
        authentication is never assumed.
        """
        self._require_origin()
        state = self._probe_session()
        if state is WhatsAppSessionState.AUTHENTICATED:
            return state
        raise _session_error(state)

    # ── W6: unread discovery (no chat is opened) ───────────────────────────

    def list_unread(self, limit: int) -> tuple[ConversationSummary, ...]:
        """List bounded unread conversations from the visible conversation list.

        This is deliberately not a chat-opening operation.  The exact origin
        and an authenticated session are required, then the page reads only
        the left-hand conversation list; nothing is clicked, no conversation is
        opened and no composer/Send control is touched, so read receipts are
        not changed by this call.  ``limit`` is validated (1..20) instead of
        being clamped.  Only rows the page marked unread are returned, and
        every text field is bounded and normalised.
        """
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise WhatsAppError("the unread limit must be an integer; refusing to read")
        if not 1 <= limit <= MAX_UNREAD_CONVERSATIONS:
            raise WhatsAppError(
                f"the unread limit must be between 1 and {MAX_UNREAD_CONVERSATIONS}; "
                "refusing to read"
            )
        self.require_authenticated()
        try:
            raw = self._page.conversation_summaries(limit)
        except WhatsAppError:
            raise
        except Exception as exc:
            raise WhatsAppError(
                "the WhatsApp conversation list could not be read; refusing to continue"
            ) from exc

        summaries: list[ConversationSummary] = []
        for item in tuple(raw or ()):
            if not isinstance(item, ConversationSummary) or not item.unread:
                continue
            name = normalize_text(item.name)[:MAX_CONVERSATION_NAME_CHARS]
            if not name:
                continue
            count = item.unread_count
            summaries.append(
                ConversationSummary(
                    name=name,
                    unread=True,
                    unread_count=(
                        count
                        if isinstance(count, int) and not isinstance(count, bool) and count > 0
                        else None
                    ),
                    preview=normalize_text(item.preview)[:MAX_CONVERSATION_PREVIEW_CHARS],
                    timestamp=normalize_text(item.timestamp)[:MAX_CONVERSATION_TIMESTAMP_CHARS],
                )
            )
            if len(summaries) >= limit:
                break
        self._logger.info("whatsapp unread conversations listed count=%d", len(summaries))
        return tuple(summaries)

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
        # The page is rendered client-side, so wait (bounded) for a recognizable
        # state before reading identity, then refuse unless it is authenticated.
        self._wait_until_ready()
        self.require_authenticated()
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
        self._wait_until_ready()
        self.require_authenticated()
        self._page.fill_search(query)
        hits = self._page.search_results()
        if not hits:
            raise WhatsAppError("the WhatsApp search returned no contact; refusing")
        expected = normalize_name(query)
        exact = [hit for hit in hits if normalize_name(hit.name) == expected]
        if len(exact) > 1:
            raise WhatsAppError("the WhatsApp search returned more than one exact match; refusing")
        if not exact:
            # A partial hit is *not* an identity: opening it would be a guess.
            raise WhatsAppError(
                "the WhatsApp search returned no exact match for that contact; refusing"
            )
        self._page.open_search_result(exact[0])
        header = self._read_header()
        self._verify_header_for_search(header)
        return header

    def open_chat_by_search(
        self,
        digits: str,
        name: str = "",
        *,
        name_is_unique: bool = True,
    ) -> str:
        """Open ``digits``' chat through WhatsApp's own search.

        The query is the *number*, never a display name, so the selected
        conversation is bound to the trusted number rather than to whatever
        name WhatsApp happens to show.  Exactly one hit is required, and the
        opened header must either show that number or match the expected saved
        name under the shared emoji-insensitive key.

        ``name_is_unique`` must be false when another ContactStore entry shares
        the saved name's emoji-insensitive key; an emoji-only header then cannot
        distinguish them, so only the whole exact saved name is accepted.
        """
        number = digits_only(digits)
        if not 8 <= len(number) <= 15:
            raise WhatsAppError("refusing to search for a number that is not 8-15 digits")
        self._expected_name = normalize_text(name)
        self._expected_digits = number
        self._expected_name_is_unique = bool(name_is_unique)
        self._page.goto(WHATSAPP_BASE_URL)
        self._require_origin()
        self._wait_until_ready()
        self.require_authenticated()
        self._page.fill_search(number)
        hits = self._page.search_results()
        if not hits:
            raise WhatsAppError("the WhatsApp search returned no chat for that number; refusing")
        if len(hits) > 1:
            raise WhatsAppError(
                "the WhatsApp search returned more than one chat for that number; refusing"
            )
        self._page.open_search_result(hits[0])
        header = self._read_header()
        self._verify_header_for_search(header)
        self._logger.info(
            "whatsapp chat opened through a number search (digits=%d)", len(number)
        )
        return header

    # ── reading ────────────────────────────────────────────────────────────

    def recent_messages(self, limit: int) -> tuple[WhatsAppMessage, ...]:
        """Read the most recent visible messages of the *verified* chat.

        The session and the chat identity are proven before and after the read,
        so a conversation switch during the read is refused.  The limit is
        validated here (1..:data:`MAX_RECENT_MESSAGES`): an out-of-range request
        is refused rather than silently clamped.  Only visible text is returned;
        attachments, empty containers and malformed entries are skipped.
        """
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise WhatsAppError("the message limit must be an integer; refusing to read")
        if not 1 <= limit <= MAX_RECENT_MESSAGES:
            raise WhatsAppError(
                f"the message limit must be between 1 and {MAX_RECENT_MESSAGES}; "
                "refusing to read"
            )
        self.require_authenticated()
        before = self._read_header()
        self._verify_header_for_search(before)
        try:
            raw = self._page.recent_messages(limit)
        except WhatsAppError:
            raise
        except Exception as exc:
            raise WhatsAppError(
                "the WhatsApp conversation could not be read; refusing to continue"
            ) from exc
        after = self._read_header()
        self._verify_header_for_search(after)
        if normalize_text(before).casefold() != normalize_text(after).casefold():
            raise WhatsAppError(
                "the WhatsApp conversation changed while it was being read; refusing"
            )
        messages: list[WhatsAppMessage] = []
        # A correct page already returns at most ``limit`` chronological items;
        # taking the tail too means even a misbehaving page cannot make this read
        # return more than the caller asked for, and keeps the *newest* window.
        for item in tuple(raw or ())[-limit:]:
            if not isinstance(item, WhatsAppMessage):
                continue
            text = normalize_text(item.text)
            if not text:
                continue
            messages.append(
                WhatsAppMessage(
                    text=text,
                    sender=normalize_text(item.sender),
                    timestamp=normalize_text(item.timestamp),
                    is_outgoing=item.is_outgoing,
                    sequence=len(messages),
                    status_hints=tuple(
                        str(hint)[:MAX_STATUS_HINT_CHARS]
                        for hint in (item.status_hints or ())[:MAX_STATUS_HINTS]
                    ),
                )
            )
            if len(messages) >= limit:
                break
        self._logger.info("whatsapp messages read count=%d", len(messages))
        return tuple(messages)

    # ── text messages ──────────────────────────────────────────────────────

    def compose_text(self, text: str) -> None:
        """Type the approved message into the verified chat's composer.

        The session and the chat identity are proven *before* anything is
        typed, and the composer is read back immediately afterwards.  A
        composer that does not hold exactly ``text`` is a refusal; nothing is
        sent here.
        """
        message = str(text or "")
        if not message.strip():
            raise WhatsAppError("refusing to compose an empty WhatsApp message")
        self.require_authenticated()
        self._verify_header_for_search(self._read_header())
        try:
            self._page.fill_composer(message)
        except WhatsAppError:
            raise
        except Exception as exc:
            raise WhatsAppError(
                "the WhatsApp message composer could not be filled; refusing to send"
            ) from exc
        actual = self._read_composer()
        if not composer_matches(message, actual):
            raise WhatsAppError(
                "the WhatsApp composer does not contain the approved message; refusing to send"
            )
        self._logger.info("whatsapp message composed for a verified contact (chars=%d)", len(message))

    def send_text(self, expected_text: str) -> SendOutcome:
        """Press Send for exactly the composed message, refusing anything else.

        Immediately before the click the session, the chat identity and the
        composer contents are re-proven.  Exactly one Send control may match.
        A click whose result cannot be observed raises
        :class:`WhatsAppSendUncertain` (the click may have reached WhatsApp),
        never a plain "nothing was sent" refusal.
        """
        message = str(expected_text or "")
        if not message.strip():
            raise WhatsAppError("refusing to send an empty WhatsApp message")
        self.require_authenticated()
        header = self._read_header()
        self._verify_header_for_search(header)
        if not composer_matches(message, self._read_composer()):
            raise WhatsAppError(
                "the WhatsApp composer no longer contains the approved message; refusing to send"
            )
        controls = self._page.send_controls()
        matches = [control for control in controls if is_send_name(control.name)]
        if not matches:
            raise WhatsAppError("no WhatsApp Send control was found; refusing to send")
        if len(matches) > 1:
            raise WhatsAppError("more than one WhatsApp Send control was found; refusing to send")
        control = matches[0]
        count = send_count(control.name)
        if count is not None and count != 1:
            raise WhatsAppError(f"{count} items are staged for sending; expected exactly one")
        try:
            self._page.click_send(normalize_text(control.name))
        except WhatsAppSendUncertain:
            raise
        except WhatsAppError:
            # A definite refusal from the adapter (e.g. the control stopped
            # being unique) means nothing was clicked; keep it a plain refusal.
            raise
        except Exception as exc:
            raise WhatsAppSendUncertain(
                "the WhatsApp Send action was attempted but its outcome is unknown"
            ) from exc
        self._logger.info("whatsapp text send performed after verification (control=%r)", control.name)
        return SendOutcome(header=header, control=control.name)

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
        # Authentication is proven on the live page *before* anything is staged,
        # so a logged-out or QR-only profile can never receive an attachment.
        self.require_authenticated()
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
        self.require_authenticated()
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

    def _probe_session(self) -> WhatsAppSessionState:
        """The page's session state; any failure is ``unknown``, never a pass."""
        try:
            raw = self._page.session_state()
        except Exception as exc:  # noqa: BLE001 - a failed probe is not a pass
            self._logger.warning("the WhatsApp session probe failed: %s", exc)
            return WhatsAppSessionState.UNKNOWN
        if isinstance(raw, WhatsAppSessionState):
            return raw
        try:
            return WhatsAppSessionState(str(raw))
        except ValueError:
            self._logger.warning("the WhatsApp session probe returned an unrecognized state")
            return WhatsAppSessionState.UNKNOWN

    def _wait_until_ready(self) -> WhatsAppSessionState:
        """Run the page's bounded readiness wait, wrapping a failure.

        The wait itself lives in the narrow page adapter (no sleep, no polling
        and no DOM access here), so the driver can never spin or hold a generic
        page handle.
        """
        try:
            return self._page.wait_for_ready(_READY_TIMEOUT_S)
        except WhatsAppError:
            raise
        except Exception as exc:
            raise WhatsAppNotReady(
                WhatsAppSessionState.UNKNOWN,
                "the WhatsApp page did not become ready; refusing to continue",
            ) from exc

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

    def _read_composer(self) -> str:
        """Read the composer's text ("" when unreadable)."""
        try:
            return str(self._page.composer_text() or "")
        except Exception as exc:  # noqa: BLE001 - an unreadable composer is not a pass
            self._logger.warning("the WhatsApp message composer could not be read: %s", exc)
            return ""

    def _verify_header_for_search(self, header: str) -> None:
        """Fail closed unless the opened chat proves exactly this recipient."""
        if not header:
            raise WhatsAppError("the WhatsApp chat header could not be read; refusing to continue")
        if not header_confirms_recipient(
            header,
            self._expected_name,
            self._expected_digits,
            unique_name=self._expected_name_is_unique,
        ):
            raise WhatsAppError(
                "the opened WhatsApp chat does not prove the requested contact; refusing to continue"
            )

    def _verify_header(self, header: str) -> None:
        """Fail closed unless the header proves the chat identity."""
        if not header:
            raise WhatsAppError("the WhatsApp chat header could not be read; refusing to continue")
        if not header_matches(header, self._expected_name, self._expected_digits):
            raise WhatsAppError(
                "the WhatsApp chat header does not match the requested contact; refusing to continue"
            )
