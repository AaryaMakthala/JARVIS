"""WhatsApp tools and contact store (Phase 6).

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
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable
from urllib.parse import quote

from pydantic import BaseModel, Field

from jarvis import config
from jarvis.tools.base import ToolContext, ToolResult, ToolSpec

logger = logging.getLogger(__name__)

#: Contacts file schema version (bump to force a migration in a future phase).
_CONTACTS_VERSION = 1
_MAX_CONTACTS = 500  # hard cap so a corrupted file cannot grow unbounded


# Privacy (docs/03): phone numbers never surface in full - keep last 2 digits.
def masked_number(number: str) -> str:
    """Render a phone number with everything but the last 2 digits hidden."""
    digits = normalize_number(number)
    if len(digits) <= 2:
        return "+" + "•" * len(digits)
    return "+" + "•" * (len(digits) - 2) + digits[-2:]


def normalize_number(raw: str) -> str:
    """Return E.164-ish digits (no leading ``+``); raises ValueError otherwise.

    Accepts an optional leading ``+`` and common separators (space, -, parens).
    The result is 8-15 digits with no separators, ready for the
    ``whatsapp://send?phone=`` URI which expects plain digits.
    """
    text = raw.strip()
    body = text.removeprefix("+")
    digits = re.sub(r"[^0-9]", "", body)
    if not text or not (8 <= len(digits) <= 15):
        raise ValueError(
            f"phone number must be 8-15 digits (E.164-ish), got {len(digits) or 0} digit(s)"
        )
    return digits


@dataclass(frozen=True)
class Contact:
    """A single WhatsApp contact (name only; number stored as plain digits)."""

    name: str
    number: str
    created: str = ""


def _created() -> str:
    import datetime

    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


class ContactStore:
    """File-backed contact list, edited only through ``jarvis contacts …``."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    # ── persistence ──────────────────────────────────────────────────────

    def load(self) -> list[Contact]:
        """Read and validate the contacts file; empty on any problem."""
        if not self.path.is_file():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            records = data["contacts"]
        except Exception:  # noqa: BLE001 - a bad file must never block sending
            logger.warning("contacts file unreadable, treating as empty: %s", self.path)
            return []
        contacts: list[Contact] = []
        for record in records[:_MAX_CONTACTS]:
            try:
                contacts.append(
                    Contact(
                        name=str(record["name"]).strip(),
                        number=normalize_number(str(record["number"])),
                        created=str(record.get("created", "")),
                    )
                )
            except Exception:  # skip malformed records
                logger.debug("skipping malformed contacts record", exc_info=True)
                continue
        return contacts

    def save(self, contacts: list[Contact]) -> None:
        """Atomically write the contact list."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _CONTACTS_VERSION,
            "contacts": [
                {"name": c.name, "number": c.number, "created": c.created} for c in contacts
            ],
        }
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    # ── operations ───────────────────────────────────────────────────────

    def add(self, name: str, number: str) -> str:
        """Add a contact after validation; returns a status line.

        Raises ValueError on invalid input or duplicates so the CLI can render
        the reason and exit 1.
        """
        name = name.strip()
        if not name:
            raise ValueError("contact name is empty")
        digits = normalize_number(number)
        contacts = self.load()
        if any(c.name.lower() == name.lower() for c in contacts):
            raise ValueError(f"a contact named {name!r} already exists")
        if any(c.number == digits for c in contacts):
            raise ValueError(f"the number +{digits} is already assigned to another contact")
        contacts.append(Contact(name=name, number=digits, created=_created()))
        self.save(contacts)
        return f"added contact {name!r} (+{digits})"

    def remove(self, name: str) -> str:
        """Remove a contact by name (case-insensitive); raises ValueError."""
        contacts = self.load()
        remaining = [c for c in contacts if c.name.lower() != name.strip().lower()]
        if len(remaining) == len(contacts):
            raise ValueError(f"no contact named {name!r}")
        self.save(remaining)
        return f"removed contact {name!r}"

    def resolve(self, query: str) -> list[Contact]:
        """Case-insensitive lookup: exact name match first, then substring."""
        q = query.strip().lower()
        contacts = self.load()
        exact = [c for c in contacts if c.name.lower() == q]
        if exact:
            return exact
        return [c for c in contacts if q in c.name.lower()]


class RateLimitError(Exception):
    """Raised when the WhatsApp send rate limit is exceeded."""


class RateLimiter:
    """Persisted send-frequency gate (max/hour + minimum spacing, docs §2.6)."""

    def __init__(
        self,
        path: Path | str,
        *,
        max_per_hour: int = 10,
        min_interval_s: float = 5.0,
        clock: object | None = None,
    ) -> None:
        if max_per_hour <= 0:
            raise ValueError("max_per_hour must be >= 1")
        self.path = Path(path)
        self.max_per_hour = max_per_hour
        self.min_interval_s = min_interval_s
        self._clock = clock if clock is not None else time.time
        self._lock = threading.Lock()
        self._events = self._load()

    def _load(self) -> list[float]:
        if not self.path.is_file():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return [float(t) for t in data.get("events", [])]
        except Exception:  # noqa: BLE001 - treat a corrupt log as empty
            logger.warning("rate-limit log unreadable, treating as empty: %s", self.path)
            return []

    def _prune(self, now: float) -> list[float]:
        cutoff = now - 3600.0
        return [t for t in self._events if t > cutoff]

    def check(self) -> None:
        """Raise :class:`RateLimitError` when this send must be refused."""
        with self._lock:
            now = self._clock()
            self._events = self._prune(now)
            if self._events and (now - max(self._events)) < self.min_interval_s:
                raise RateLimitError(
                    f"WhatsApp rate limit: wait at least {self.min_interval_s:.0f}s between sends"
                )
            if len(self._events) >= self.max_per_hour:
                raise RateLimitError(
                    f"WhatsApp rate limit: at most {self.max_per_hour} sends per hour"
                )

    def record(self) -> None:
        """Record a successfully sent message (persisted across restarts)."""
        with self._lock:
            now = self._clock()
            self._events = self._prune(now)
            self._events.append(now)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"events": self._events}), encoding="utf-8")

    def traffic_count(self) -> int:
        """Number of sends recorded in the current window (read-only)."""
        with self._lock:
            return len(self._prune(self._clock()))


# ── window provider (tests use a fake; the real one is Windows-only) ─────


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


def _build_provider(settings: config.Settings) -> DesktopWindowProvider:
    """Build the real desktop provider (monkeypatched in tests)."""
    return PywinautoWhatsAppWindowProvider(settings.whatsapp.app_name)


class PywinautoWhatsAppWindowProvider:
    """Best-effort WhatsApp Desktop provider behind pywinauto (UIA).

    Every UI read fails closed: any exception or unreadable control yields
    ``None`` and the tool refuses to send (docs/12: pywinauto cannot always
    read the WhatsApp UI — never turned into a "success").
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


# ── the tool ─────────────────────────────────────────────────────────────


class WhatsAppSendArgs(BaseModel):
    """Arguments for the ``whatsapp_send`` tool."""

    contact: str = Field(
        min_length=1, max_length=100, description="Contact name from `jarvis contacts`"
    )
    message: str = Field(min_length=1, max_length=1000, description="Message text (max 1000 chars)")


def _resolve_single(store: ContactStore, query: str) -> tuple[Contact | None, str | None]:
    """Return exactly one contact, or the error text listing the options."""
    matches = store.resolve(query)
    if len(matches) == 1:
        return matches[0], None
    if not matches:
        known = ", ".join(c.name for c in store.load()) or "(no contacts yet)"
        return None, (
            f"no contact matches {query!r}. Known contacts: {known}. "
            "Add one with `jarvis contacts add`."
        )
    names = ", ".join(c.name for c in matches)
    return None, f"multiple contacts match {query!r}: {names} — be more specific"


def _contact_matches_header(contact: Contact, header: str) -> bool:
    """The chat header matches when it shows the name or the phone digits."""
    haystack = header.lower()
    return contact.name.lower() in haystack or contact.number in haystack


def _whatsapp_send_run(
    store: ContactStore | None,
    limiter: RateLimiter | None,
    args: WhatsAppSendArgs,
    ctx: ToolContext,
) -> ToolResult:
    """Execute the §2.6 flow; never sends unless the chat is verified."""
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

    contact, error = _resolve_single(store, args.contact)
    if error is not None:
        return ToolResult(ok=False, error=error)

    try:
        limiter.check()
    except RateLimitError as exc:
        return ToolResult(ok=False, error=str(exc))

    provider = _build_provider(ctx.settings)
    try:
        provider.open_chat(contact.number, args.message)
    except Exception as exc:  # noqa: BLE001
        return ToolResult(ok=False, error=f"could not open WhatsApp: {exc}")

    window = provider.find_chat_window(ctx.settings.whatsapp.window_timeout_s)
    if window is None:
        # Invariant 9 / docs §2.6 step 4-5: unverifiable chat → do not send.
        return ToolResult(
            ok=False,
            error="Could not verify chat; draft left in WhatsApp, send manually",
        )
    header = window.chat_header()
    if header is None or not _contact_matches_header(contact, header):
        return ToolResult(
            ok=False,
            error="Chat identity could not be verified for "
            f"{contact.name!r}; draft left in WhatsApp, send manually",
        )

    draft = window.draft_text()
    if draft is not None and args.message not in draft:
        return ToolResult(
            ok=False,
            error="Draft text does not match the message; nothing was sent",
        )

    try:
        window.press_enter()
    except Exception as exc:  # noqa: BLE001
        return ToolResult(ok=False, error=f"failed to press Enter: {exc}")

    confirmed = window.message_visible(args.message)
    limiter.record()
    logger.info(
        "whatsapp sent contact=%s phone_digits=%d confirmed=%s",
        contact.name,
        len(contact.number),
        confirmed,
    )
    return ToolResult(
        ok=True,
        output=f"sent WhatsApp message to {contact.name}",
        data={
            "recipient": contact.name,
            "phone_digits": contact.number,
            "chat_confirmed": confirmed,
        },
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
        verify=lambda args, result, ctx: _whatsapp_send_verify(store, limiter, args, result, ctx),
        describe=lambda args: _describe_whatsapp(store, args),
    )
