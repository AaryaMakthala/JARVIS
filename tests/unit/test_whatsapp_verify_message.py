"""W5: read-only verification of a dispatched WhatsApp message's status.

No browser, no network, no live WhatsApp, no sending.  The real driver, adapter
and tool run against a fake narrow page and a real SQLite dispatch ledger in a
temporary directory.

What these tests pin down:

* status is derived only from WhatsApp's visible status vocabulary, and a
  missing/contradictory indicator is ``unknown`` - never guessed upward;
* the target message is bound to the recorded dispatch (contact + text digest),
  so the planner cannot point verification at another conversation or message;
* identical bodies are ambiguous and yield ``unknown`` instead of a pick;
* verification is read-only: no Send, no composer, no reply/forward/react,
  no attachment, no arbitrary navigation, no JavaScript;
* an unknown result never retries a send.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from jarvis.browser.whatsapp_page import PlaywrightWhatsAppPage
from jarvis.browser.whatsapp_web import (
    WHATSAPP_BASE_URL,
    SearchHit,
    SendControl,
    WhatsAppMessage,
    WhatsAppSessionState,
)
from jarvis.config import Settings
from jarvis.messaging.contacts import ContactStore
from jarvis.messaging.dispatch_ledger import (
    RETENTION_S,
    DispatchLedger,
    DispatchLedgerError,
)
from jarvis.messaging.verification import (
    MessageStatus,
    digest_text,
    match_outgoing,
    status_from_hints,
)
from jarvis.tools.base import ToolContext, ToolResult
from jarvis.tools.registry import build_default_registry
from jarvis.tools.whatsapp_web import (
    MAX_VERIFY_WAIT_S,
    WhatsAppSendMessageArgs,
    WhatsAppVerifyMessageArgs,
    make_whatsapp_send_message_spec,
    make_whatsapp_verify_message_spec,
)

_PHONE = "919876543210"
_OTHER_PHONE = "919876543211"
_MOM_PHONE = "919876543212"
_BODY = "I'll be late tonight"
_NOW = 1_700_000_000.0


@pytest.fixture(autouse=True)
def _windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test runs as if on Windows unless it overrides this."""
    monkeypatch.setattr("jarvis.platform_guard.is_windows", lambda: True)


class _Clock:
    """A controllable clock for the ledger and the guard."""

    def __init__(self, now: float = _NOW) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _store(tmp_path: Path, *contacts: tuple[str, str]) -> ContactStore:
    """A store that tolerates being built twice for the same test directory."""
    store = ContactStore(tmp_path / "contacts.json")
    known = {(contact.name, contact.number) for contact in store.load()}
    for name, number in contacts or (("Rahul", _PHONE),):
        if (name, number) not in known:
            store.add(name, number)
    return store


def _ledger(tmp_path: Path, clock: Any = None) -> DispatchLedger:
    return DispatchLedger(tmp_path / "dispatches.sqlite3", clock=clock or _Clock())


def _record(
    ledger: DispatchLedger,
    *,
    contact: str = "Rahul",
    text: str = _BODY,
    operation_id: str = "t1:abc",
    outcome: str = "dispatched",
) -> None:
    ledger.record(operation_id, contact, text, outcome=outcome)


def _msg(
    text: str = _BODY,
    *,
    outgoing: bool | None = True,
    timestamp: str = "10:41",
    hints: tuple[str, ...] = ("msg-dblcheck",),
    sender: str = "",
) -> WhatsAppMessage:
    return WhatsAppMessage(
        text=text,
        sender=sender,
        timestamp=timestamp,
        is_outgoing=outgoing,
        status_hints=hints,
    )


class _Page:
    """A narrow WhatsApp page stand-in that records every interaction."""

    def __init__(
        self,
        *,
        windows: list[tuple[WhatsAppMessage, ...]] | None = None,
        hits: tuple[SearchHit, ...] = (SearchHit(name="Rahul"),),
        headers: tuple[str, ...] = ("Rahul",),
        state: WhatsAppSessionState = WhatsAppSessionState.AUTHENTICATED,
        url: str = WHATSAPP_BASE_URL,
        raise_on_read: Exception | None = None,
        forbid_send: bool = True,
    ) -> None:
        #: Read-only tests keep this true so any Send-control enumeration during
        #: verification fails loudly; the W4 send test opts out explicitly.
        self.forbid_send = forbid_send
        self.windows = list(windows or [(_msg(),)])
        self.hits = hits
        self.headers = headers
        self.state = state
        self.forced_url = url
        self.raise_on_read = raise_on_read
        self.current_url = url
        self.visited: list[str] = []
        self.searched: list[str] = []
        self.opened: list[SearchHit] = []
        self.read_limits: list[int] = []
        self.read_calls = 0
        self.fills: list[str] = []
        self.composer_reads = 0
        self.staged_paths: list[str] = []
        self.clicked: list[str] = []
        self.scans: list[str] = []
        self._header_calls = 0

    def url(self) -> str:
        return self.current_url

    def goto(self, url: str) -> None:
        self.visited.append(url)
        self.current_url = self.forced_url

    def session_state(self) -> WhatsAppSessionState:
        return self.state

    def wait_for_ready(self, timeout_s: float = 20.0) -> WhatsAppSessionState:
        del timeout_s
        return self.state

    def chat_header(self) -> str:
        index = min(self._header_calls, len(self.headers) - 1)
        self._header_calls += 1
        return self.headers[index]

    def recent_messages(self, limit: int) -> tuple[Any, ...]:
        self.read_limits.append(limit)
        self.read_calls += 1
        if self.raise_on_read is not None:
            raise self.raise_on_read
        index = min(self.read_calls - 1, len(self.windows) - 1)
        return self.windows[index]

    def fill_search(self, query: str) -> None:
        self.searched.append(query)

    def search_results(self) -> tuple[SearchHit, ...]:
        return self.hits

    def open_search_result(self, hit: SearchHit) -> None:
        self.opened.append(hit)

    def fill_composer(self, text: str) -> None:
        self.fills.append(text)

    def composer_text(self) -> str:
        self.composer_reads += 1
        return self.fills[-1] if self.fills else ""

    def set_input_files(self, path: str) -> None:
        self.staged_paths.append(path)

    def staged_file_names(self) -> tuple[str, ...]:
        return ()

    def attachment_scan(self, filename: str) -> Any:
        self.scans.append(filename)
        raise AssertionError("verification must never scan attachments")

    def send_controls(self) -> tuple[Any, ...]:
        if self.forbid_send:
            raise AssertionError("verification must never enumerate Send controls")
        return (SendControl(name="Send", index=0),)

    def click_send(self, name: str) -> None:
        self.clicked.append(name)


class _Session:
    def __init__(self, page: _Page) -> None:
        self._page = page

    def whatsapp_page(self) -> _Page:
        return self._page


class _Holder:
    def __init__(self, page: _Page | None) -> None:
        self.page = page
        self.acquires = 0

    def acquire(self) -> _Session | None:
        self.acquires += 1
        return _Session(self.page) if self.page is not None else None


def _ctx(holder: _Holder, *, operation_id: str = "") -> ToolContext:
    return ToolContext(settings=Settings(), whatsapp_browser=holder, operation_id=operation_id)


def _spec(
    tmp_path: Path,
    *,
    store: ContactStore | None = None,
    ledger: DispatchLedger | None = None,
):
    return make_whatsapp_verify_message_spec(
        store=store or _store(tmp_path),
        ledger=ledger or _ledger(tmp_path),
    )


def _verify(
    tmp_path: Path,
    page: _Page,
    *,
    store: ContactStore | None = None,
    ledger: DispatchLedger | None = None,
    contact: str = "Rahul",
    wait_seconds: int = 0,
    expect: str | None = None,
) -> ToolResult:
    return _spec(tmp_path, store=store, ledger=ledger).execute(
        WhatsAppVerifyMessageArgs(
            contact=contact, wait_seconds=wait_seconds, expect=expect  # type: ignore[arg-type]
        ),
        _ctx(_Holder(page)),
    )


# ── status model: only visible evidence is mapped ─────────────────────────


@pytest.mark.parametrize(
    "hints,expected",
    [
        (("msg-check",), MessageStatus.SENT),
        (("msg-dblcheck",), MessageStatus.DELIVERED),
        (("msg-dblcheck-ack",), MessageStatus.READ),
        (("msg-time",), MessageStatus.DISPATCHED),
        (("Sent",), MessageStatus.SENT),
        (("Delivered",), MessageStatus.DELIVERED),
        (("Read",), MessageStatus.READ),
        (("Pending",), MessageStatus.DISPATCHED),
        ((), MessageStatus.UNKNOWN),
        (("something-else",), MessageStatus.UNKNOWN),
        (("unread",), MessageStatus.UNKNOWN),
    ],
)
def test_status_from_hints_maps_only_known_vocabulary(
    hints: tuple[str, ...], expected: MessageStatus
) -> None:
    assert status_from_hints(hints, is_outgoing=True).status is expected


def test_contradictory_indicators_are_unknown() -> None:
    evidence = status_from_hints(("msg-dblcheck", "msg-dblcheck-ack"), is_outgoing=True)
    assert evidence.status is MessageStatus.UNKNOWN
    assert evidence.reason == "contradictory_evidence"


def test_an_incoming_message_never_carries_a_status() -> None:
    assert status_from_hints(("msg-dblcheck-ack",), is_outgoing=False).status is MessageStatus.UNKNOWN
    assert status_from_hints(("msg-dblcheck-ack",), is_outgoing=None).status is MessageStatus.UNKNOWN


def test_status_ranks_only_order_evidence() -> None:
    assert (
        MessageStatus.UNKNOWN.rank
        < MessageStatus.DISPATCHED.rank
        < MessageStatus.SENT.rank
        < MessageStatus.DELIVERED.rank
        < MessageStatus.READ.rank
    )


# ── message identification ────────────────────────────────────────────────


def test_the_exact_outgoing_message_is_identified() -> None:
    match = match_outgoing([_msg()], digest=digest_text(_BODY))
    assert match.status is MessageStatus.DELIVERED
    assert match.index == 0
    assert match.evidence == ("msg-dblcheck",)


def test_an_incoming_message_with_the_same_text_is_not_the_target() -> None:
    match = match_outgoing([_msg(outgoing=False)], digest=digest_text(_BODY))
    assert match.status is MessageStatus.UNKNOWN
    assert match.reason == "not_outgoing"


def test_missing_direction_metadata_is_unknown() -> None:
    match = match_outgoing([_msg(outgoing=None)], digest=digest_text(_BODY))
    assert match.status is MessageStatus.UNKNOWN
    assert match.reason == "not_outgoing"


def test_a_content_mismatch_is_unknown() -> None:
    match = match_outgoing([_msg("a different message")], digest=digest_text(_BODY))
    assert match.status is MessageStatus.UNKNOWN
    assert match.reason == "no_matching_message"


def test_a_missing_timestamp_is_unknown() -> None:
    match = match_outgoing([_msg(timestamp="")], digest=digest_text(_BODY))
    assert match.status is MessageStatus.UNKNOWN
    assert match.reason == "missing_timestamp"


def test_duplicate_identical_bodies_are_ambiguous() -> None:
    match = match_outgoing([_msg(), _msg()], digest=digest_text(_BODY))
    assert match.status is MessageStatus.UNKNOWN
    assert match.reason == "duplicate_matching_messages"


def test_whitespace_normalisation_does_not_change_identity() -> None:
    assert digest_text("I'll  be late   tonight") == digest_text(_BODY)


# ── the dispatch ledger: privacy, retention, fail-closed ──────────────────


def test_the_ledger_records_and_finds_the_newest_dispatch(tmp_path: Path) -> None:
    clock = _Clock()
    ledger = _ledger(tmp_path, clock)
    _record(ledger, text="first", operation_id="t1:one")
    clock.now += 5.0  # distinct dispatch times, so "newest" is unambiguous
    _record(ledger, text="second", operation_id="t1:two")

    record = ledger.latest("Rahul")

    assert record is not None
    assert record.message_sha256 == digest_text("second")
    assert record.outcome == "dispatched"


def test_dispatches_are_isolated_per_contact(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    _record(ledger, contact="Rahul")
    assert ledger.latest("Ali") is None


def test_the_ledger_stores_no_body_and_no_number(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    _record(ledger)
    raw = (tmp_path / "dispatches.sqlite3").read_bytes()
    assert _BODY.encode() not in raw
    assert _PHONE.encode() not in raw


def test_an_expired_dispatch_is_no_longer_verifiable(tmp_path: Path) -> None:
    clock = _Clock()
    ledger = _ledger(tmp_path, clock)
    _record(ledger)

    later = _ledger(tmp_path, _Clock(_NOW + RETENTION_S + 1))
    assert later.latest("Rahul") is None


def test_a_corrupt_ledger_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "dispatches.sqlite3"
    path.write_bytes(b"not a sqlite database")
    with pytest.raises(DispatchLedgerError):
        DispatchLedger(path).latest("Rahul")


def test_a_missing_operation_id_is_not_recorded(tmp_path: Path) -> None:
    assert _ledger(tmp_path).record("", "Rahul", _BODY) is None


# ── tool: status reporting ────────────────────────────────────────────────


def test_a_delivered_message_is_reported_as_delivered(tmp_path: Path) -> None:
    page = _Page(windows=[(_msg(hints=("msg-dblcheck",)),)])
    ledger = _ledger(tmp_path)
    _record(ledger)

    result = _verify(tmp_path, page, ledger=ledger)

    assert result.ok, result.error
    assert result.data["status"] == "delivered"
    assert result.verified is True
    assert "delivered to Rahul" in result.output
    assert _PHONE not in result.output
    assert _PHONE not in str(result.data)


def test_a_read_message_is_reported_as_read(tmp_path: Path) -> None:
    page = _Page(windows=[(_msg(hints=("msg-dblcheck-ack",)),)])
    ledger = _ledger(tmp_path)
    _record(ledger)

    result = _verify(tmp_path, page, ledger=ledger)

    assert result.data["status"] == "read"
    assert result.verified is True
    assert "has read your message" in result.output


def test_a_sent_message_does_not_claim_delivery(tmp_path: Path) -> None:
    page = _Page(windows=[(_msg(hints=("msg-check",)),)])
    ledger = _ledger(tmp_path)
    _record(ledger)

    result = _verify(tmp_path, page, ledger=ledger)

    assert result.data["status"] == "sent"
    assert result.verified is False
    assert "delivery is not confirmed" in result.output


def test_a_dispatched_message_without_evidence_is_not_delivered(tmp_path: Path) -> None:
    page = _Page(windows=[(_msg(hints=()),)])
    ledger = _ledger(tmp_path)
    _record(ledger)

    result = _verify(tmp_path, page, ledger=ledger)

    assert result.data["status"] == "unknown"
    assert result.verified is False
    assert "couldn't reliably determine" in result.output
    assert result.data["reason"] == "no_status_evidence"


def test_an_unknown_outcome_record_is_still_verifiable(tmp_path: Path) -> None:
    page = _Page(windows=[(_msg(hints=("msg-dblcheck-ack",)),)])
    ledger = _ledger(tmp_path)
    _record(ledger, outcome="unknown")

    result = _verify(tmp_path, page, ledger=ledger)

    assert result.data["send_outcome"] == "unknown"
    assert result.data["status"] == "read"


def test_duplicate_identical_bodies_produce_an_honest_unknown(tmp_path: Path) -> None:
    page = _Page(windows=[(_msg(), _msg())])
    ledger = _ledger(tmp_path)
    _record(ledger)

    result = _verify(tmp_path, page, ledger=ledger)

    assert result.data["status"] == "unknown"
    assert result.data["reason"] == "duplicate_matching_messages"
    assert "won't resend it" in result.output


def test_a_changed_conversation_refuses(tmp_path: Path) -> None:
    # open -> before -> after: the after header still proves the contact by its
    # number, yet it is a different header, which is exactly the switch case.
    page = _Page(windows=[(_msg(),)], headers=("Rahul", "Rahul", _PHONE))
    ledger = _ledger(tmp_path)
    _record(ledger)

    result = _verify(tmp_path, page, ledger=ledger)

    assert result.ok is False
    assert "changed while it was being read" in result.error


def test_a_wrong_conversation_header_refuses(tmp_path: Path) -> None:
    page = _Page(windows=[(_msg(),)], headers=("Someone Else",))
    ledger = _ledger(tmp_path)
    _record(ledger)

    result = _verify(tmp_path, page, ledger=ledger)

    assert result.ok is False
    assert "could not read the WhatsApp conversation" in result.error
    assert page.clicked == []


# ── tool: operation binding ───────────────────────────────────────────────


def test_no_dispatch_record_is_an_explicit_unknown(tmp_path: Path) -> None:
    page = _Page()
    result = _verify(tmp_path, page)

    assert result.ok is True
    assert result.data["status"] == "unknown"
    assert result.data["reason"] == "no_dispatch_record"
    assert "no record" in result.output
    assert page.visited == [] and page.read_limits == []
    assert page.clicked == []


def test_a_record_for_another_contact_is_not_used(tmp_path: Path) -> None:
    page = _Page()
    ledger = _ledger(tmp_path)
    _record(ledger, contact="Ali")

    result = _verify(tmp_path, page, ledger=ledger)

    assert result.data["reason"] == "no_dispatch_record"


def test_an_unknown_contact_is_refused_without_a_browser(tmp_path: Path) -> None:
    page = _Page()
    holder = _Holder(page)
    result = _spec(tmp_path).execute(
        WhatsAppVerifyMessageArgs(contact="Nobody"), _ctx(holder)
    )

    assert result.ok is False
    assert result.error
    assert holder.acquires == 0


def test_an_ambiguous_contact_is_refused(tmp_path: Path) -> None:
    store = _store(tmp_path, ("Ali Khan", "9112345678"), ("Ali Ahmed", _PHONE))
    ledger = _ledger(tmp_path)
    _record(ledger, contact="Ali Khan")

    result = _verify(tmp_path, _Page(), store=store, ledger=ledger, contact="Ali")

    assert result.ok is False
    assert "multiple contacts match" in result.error


def test_a_corrupt_ledger_refuses_instead_of_guessing(tmp_path: Path) -> None:
    path = tmp_path / "dispatches.sqlite3"
    path.write_bytes(b"not a sqlite database")
    ledger = DispatchLedger(path, clock=_Clock())

    result = _verify(tmp_path, _Page(), ledger=ledger)

    assert result.ok is False
    assert "dispatch record could not be read" in result.error


def test_the_args_cannot_name_a_message_or_an_operation() -> None:
    fields = set(WhatsAppVerifyMessageArgs.model_fields)
    assert fields == {"contact", "wait_seconds", "expect"}
    assert not {"message", "text", "operation", "task_id", "url", "selector"} & fields


def test_an_over_long_wait_is_rejected_by_the_schema() -> None:
    with pytest.raises(ValidationError):
        WhatsAppVerifyMessageArgs(contact="Rahul", wait_seconds=MAX_VERIFY_WAIT_S + 1)


def test_an_arbitrary_expectation_is_rejected_by_the_schema() -> None:
    with pytest.raises(ValidationError):
        WhatsAppVerifyMessageArgs(contact="Rahul", expect="delivered-ish")  # type: ignore[arg-type]


# ── authentication, origin and identity ──────────────────────────────────


@pytest.mark.parametrize(
    "state,fragment",
    [
        (WhatsAppSessionState.QR_REQUIRED, "QR"),
        (WhatsAppSessionState.LOGGED_OUT, "logged out"),
        (WhatsAppSessionState.UNKNOWN, "authenticated"),
    ],
)
def test_an_unauthenticated_session_refuses(
    tmp_path: Path, state: WhatsAppSessionState, fragment: str
) -> None:
    ledger = _ledger(tmp_path)
    _record(ledger)
    result = _verify(tmp_path, _Page(state=state), ledger=ledger)

    assert result.ok is False
    assert fragment.lower() in result.error.lower()


def test_a_foreign_origin_is_refused(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    _record(ledger)
    page = _Page(url="https://evil.example/")

    result = _verify(tmp_path, page, ledger=ledger)

    assert result.ok is False
    assert "whatsapp" in result.error.lower()


def test_an_unavailable_page_refuses(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    _record(ledger)
    result = _spec(tmp_path, ledger=ledger).execute(
        WhatsAppVerifyMessageArgs(contact="Rahul"), _ctx(_Holder(None))
    )

    assert result.ok is False
    assert "not available" in result.error


# ── bounded polling ──────────────────────────────────────────────────────


def test_a_status_can_advance_from_sent_to_delivered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("jarvis.tools.whatsapp_web.VERIFY_POLL_INTERVAL_S", 0.0)
    monkeypatch.setattr("jarvis.tools.whatsapp_web.time.sleep", lambda _s: None)
    page = _Page(
        windows=[
            (_msg(hints=("msg-check",)),),
            (_msg(hints=("msg-dblcheck",)),),
        ]
    )
    ledger = _ledger(tmp_path)
    _record(ledger)

    result = _verify(tmp_path, page, ledger=ledger, wait_seconds=10, expect="delivered")

    assert result.data["status"] == "delivered"
    assert page.read_calls == 2


def test_a_status_can_advance_from_delivered_to_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("jarvis.tools.whatsapp_web.VERIFY_POLL_INTERVAL_S", 0.0)
    monkeypatch.setattr("jarvis.tools.whatsapp_web.time.sleep", lambda _s: None)
    page = _Page(
        windows=[
            (_msg(hints=("msg-dblcheck",)),),
            (_msg(hints=("msg-dblcheck-ack",)),),
        ]
    )
    ledger = _ledger(tmp_path)
    _record(ledger)

    result = _verify(tmp_path, page, ledger=ledger, wait_seconds=10, expect="read")

    assert result.data["status"] == "read"
    assert result.verified is True


def test_an_already_reached_status_is_not_polled_again(tmp_path: Path) -> None:
    page = _Page(windows=[(_msg(hints=("msg-dblcheck-ack",)),)])
    ledger = _ledger(tmp_path)
    _record(ledger)

    result = _verify(tmp_path, page, ledger=ledger, wait_seconds=10, expect="read")

    assert result.data["status"] == "read"
    assert page.read_calls == 1


def test_an_unknown_status_is_not_polled(tmp_path: Path) -> None:
    page = _Page(windows=[(_msg(hints=()),)])
    ledger = _ledger(tmp_path)
    _record(ledger)

    result = _verify(tmp_path, page, ledger=ledger, wait_seconds=10, expect="read")

    assert result.data["status"] == "unknown"
    assert page.read_calls == 1


# ── read-only guarantees ─────────────────────────────────────────────────


def test_verification_never_sends_composes_or_attaches(tmp_path: Path) -> None:
    page = _Page(windows=[(_msg(hints=("msg-dblcheck",)),)])
    ledger = _ledger(tmp_path)
    _record(ledger)

    result = _verify(tmp_path, page, ledger=ledger)

    assert result.ok
    assert page.clicked == []
    assert page.fills == []
    assert page.composer_reads == 0
    assert page.staged_paths == []
    assert page.scans == []


def test_verification_only_navigates_to_the_whatsapp_base_for_the_w2_route(
    tmp_path: Path,
) -> None:
    page = _Page(windows=[(_msg(),)])
    ledger = _ledger(tmp_path)
    _record(ledger)

    _verify(tmp_path, page, ledger=ledger)

    assert page.visited == [WHATSAPP_BASE_URL]
    assert page.searched == [_PHONE]


def test_verification_reads_only_a_bounded_window(tmp_path: Path) -> None:
    page = _Page(windows=[(_msg(),)])
    ledger = _ledger(tmp_path)
    _record(ledger)

    _verify(tmp_path, page, ledger=ledger)

    assert page.read_limits == [20]


def test_an_unknown_result_never_retries_a_send(tmp_path: Path) -> None:
    page = _Page(windows=[(_msg(hints=()),)])
    ledger = _ledger(tmp_path)
    _record(ledger)

    first = _verify(tmp_path, page, ledger=ledger)
    second = _verify(tmp_path, page, ledger=ledger)

    assert first.data["status"] == "unknown"
    assert second.data["status"] == "unknown"
    assert page.clicked == []
    assert page.fills == []


def test_the_message_body_is_never_logged_or_returned(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    page = _Page(windows=[(_msg(),)])
    ledger = _ledger(tmp_path)
    _record(ledger)

    result = _verify(tmp_path, page, ledger=ledger)

    assert _BODY not in caplog.text
    assert _BODY not in str(result.data)
    assert _BODY not in result.output


def test_every_expected_result_has_a_visible_message(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    _record(ledger)
    for hints in (("msg-check",), ("msg-dblcheck",), ("msg-dblcheck-ack",), ()):
        result = _verify(tmp_path, _Page(windows=[(_msg(hints=hints),)]), ledger=ledger)
        assert result.output.strip()
    assert _verify(tmp_path, _Page(), ledger=_ledger(tmp_path)).output.strip()


# ── existing W4 protection is untouched ──────────────────────────────────


def test_the_w4_duplicate_send_guard_still_refuses_a_second_attempt(
    tmp_path: Path,
) -> None:
    from jarvis.messaging.contacts import RateLimiter
    from jarvis.messaging.whatsapp_web import FileSendGuard

    page = _Page(forbid_send=False)
    holder = _Holder(page)
    spec = make_whatsapp_send_message_spec(
        store=_store(tmp_path),
        limiter=RateLimiter(tmp_path / "ratelimit.json", max_per_hour=10, min_interval_s=0.0),
        guard=FileSendGuard(tmp_path / "text_guard.sqlite3", clock=_Clock()),
        ledger=_ledger(tmp_path),
    )
    args = WhatsAppSendMessageArgs(contact="Rahul", message=_BODY)

    first = spec.execute(args, _ctx(holder, operation_id="t1:abc"))
    second = spec.execute(args, _ctx(holder, operation_id="t1:abc"))

    assert first.ok, first.error
    assert second.ok is False
    assert "already attempted" in second.error


# ── the live page parser exposes status hints (W5) ───────────────────────


class _EvalPage:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.scripts: list[str] = []

    def evaluate(self, script: str, arg: Any) -> Any:
        self.scripts.append(script)
        return self.result


def test_the_page_adapter_carries_status_hints_into_the_message() -> None:
    raw = {
        "panelFound": True,
        "messages": [
            {
                "text": _BODY,
                "pre": "[10:41, 02/11/2026] Rahul: ",
                "meta": "10:41",
                "outgoing": True,
                "hints": ["msg-dblcheck", "Delivered", 7],
            }
        ],
    }
    page = PlaywrightWhatsAppPage(_EvalPage(raw))

    messages = page.recent_messages(5)

    assert len(messages) == 1
    assert messages[0].status_hints == ("msg-dblcheck", "Delivered")
    assert "data-icon" in page._page.scripts[0]


def test_the_page_adapter_reports_no_hints_when_the_page_exposes_none() -> None:
    raw = {
        "panelFound": True,
        "messages": [{"text": _BODY, "pre": "", "meta": "10:41", "outgoing": True}],
    }

    messages = PlaywrightWhatsAppPage(_EvalPage(raw)).recent_messages(5)

    assert messages[0].status_hints == ()


# ── registration ─────────────────────────────────────────────────────────


def test_the_verify_tool_is_registered_as_a_read_only_tier_1_tool() -> None:
    registry = build_default_registry()
    spec = registry.get("whatsapp_verify_message")

    assert spec.base_tier == 1
    assert spec.args_model is WhatsAppVerifyMessageArgs
    assert spec.precheck is not None
    assert spec.path_args == ()
    assert spec.bind_content_hash is False
    assert "whatsapp_verify_message" in registry.names()


def test_the_verify_tool_is_advertised_without_send_privileges() -> None:
    registry = build_default_registry()
    spec = registry.get("whatsapp_verify_message")

    assert "never sends" in spec.description
    assert "retries" in spec.description


def test_the_verification_tool_is_contact_groundable() -> None:
    from jarvis.messaging.grounding import GROUNDABLE_TOOLS

    assert "whatsapp_verify_message" in GROUNDABLE_TOOLS


def test_grounding_rewrites_the_verification_contact(tmp_path: Path) -> None:
    from jarvis.agent.state import Step
    from jarvis.messaging.grounding import ground_contacts

    store = _store(tmp_path, ("Mom ❤️", _MOM_PHONE))
    step = Step(
        id="s1",
        tool="whatsapp_verify_message",
        args={"contact": "Mom"},
        rationale="check it",
    )

    outcome = ground_contacts([step], store=store, llm=None)

    assert outcome.halted_reason is None
    assert step.args["contact"] == "Mom ❤️"


def test_the_verify_spec_does_not_expose_a_send_path() -> None:
    spec = make_whatsapp_verify_message_spec()
    assert spec.name == "whatsapp_verify_message"
    assert spec.base_tier == 1
