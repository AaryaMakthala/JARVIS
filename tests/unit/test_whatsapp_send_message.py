"""W4: confirmed WhatsApp text messaging through the verified web route.

No browser, no Playwright, no network.  The real driver, adapter and tool run
against a fake narrow page, a fake contact store, a fake rate limiter and a
persisted duplicate-send guard.

What this pins down:

* the tool is Tier 2, costs one budget unit and has no path/content-hash
  binding: the approval is bound to the canonical args (contact + exact text);
* the exact text is shown in the confirmation summary and is never logged,
  returned in ``ToolResult.data`` or allowed into a validation error;
* the chat is opened through the W2 number search and the composer is read back
  immediately before the click; any mismatch refuses and sends nothing;
* an unobservable click is reported as ``send_outcome="unknown"`` with
  ``verified=None`` (not retried), and the operation-bound guard still prevents
  a second attempt of the same approval;
* ``act`` hands the tool a stable ``task_id:action_hash`` operation id.
"""

from __future__ import annotations

import json
import logging
import threading
import unicodedata
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError
from w41_helpers import page_is_verified

from jarvis.agent.context import make_app_context
from jarvis.agent.nodes.act import act
from jarvis.agent.state import Plan, Step
from jarvis.agent.tool_groups import GROUPS
from jarvis.browser.whatsapp_page import PlaywrightWhatsAppPage
from jarvis.browser.whatsapp_web import (
    WHATSAPP_BASE_URL,
    SearchHit,
    SendControl,
    WhatsAppError,
    WhatsAppSendUncertain,
    WhatsAppSessionState,
    composer_matches,
)
from jarvis.config import Settings
from jarvis.messaging.contacts import ContactStore, RateLimiter
from jarvis.messaging.whatsapp_web import FileSendGuard
from jarvis.policy import rules
from jarvis.tools.base import ToolContext, ToolResult, ToolSpec
from jarvis.tools.registry import build_default_registry
from jarvis.tools.whatsapp_web import (
    MAX_MESSAGE_CHARS,
    WhatsAppSendMessageArgs,
    make_whatsapp_send_message_spec,
)
from support import registry_with

_PHONE = "919876543210"
_MOM_PHONE = "919876543211"
_MOM_BLUE_PHONE = "919876543212"
_MOM_LOGO_PHONE = "919876543213"
_SECRET = "SECRET-TEXT-do-not-log"


@pytest.fixture(autouse=True)
def _windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test runs as if on Windows unless it overrides this."""
    monkeypatch.setattr("jarvis.platform_guard.is_windows", lambda: True)


class _Clock:
    """A deterministic clock so the duplicate-send window is stable."""

    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class _TextPage:
    """Fake narrow WhatsApp page that records every compose and click."""

    def __init__(
        self,
        *,
        header: str = "Rahul",
        hits: tuple[SearchHit, ...] = (SearchHit(name="Rahul"),),
        controls: tuple[SendControl, ...] = (SendControl(name="Send", index=0),),
        state: WhatsAppSessionState = WhatsAppSessionState.AUTHENTICATED,
        click_error: Exception | None = None,
        on_fill: Any = None,
        on_composer_read: Any = None,
        redirect_to: str = "",
        session_error: Exception | None = None,
    ) -> None:
        self.current_url = WHATSAPP_BASE_URL
        self.header = header
        self.hits = hits
        self.controls = controls
        self.state = state
        self.click_error = click_error
        self.on_fill = on_fill
        self.on_composer_read = on_composer_read
        self.redirect_to = redirect_to
        self.session_error = session_error
        self.searched: list[str] = []
        self.opened: list[SearchHit] = []
        self.fills: list[str] = []
        self.composer = ""
        self.composer_reads = 0
        self.clicked: list[str] = []

    def url(self) -> str:
        return self.current_url

    def goto(self, url: str) -> None:
        self.current_url = self.redirect_to or url

    def session_state(self) -> WhatsAppSessionState:
        if self.session_error is not None:
            raise self.session_error
        return self.state

    def wait_for_ready(self, timeout_s: float = 20.0) -> WhatsAppSessionState:
        del timeout_s
        return self.state

    def chat_header(self) -> str:
        return self.header

    def fill_search(self, query: str) -> None:
        self.searched.append(query)

    def search_results(self) -> tuple[SearchHit, ...]:
        return self.hits

    def open_search_result(self, hit: SearchHit) -> None:
        self.opened.append(hit)

    def fill_composer(self, text: str) -> None:
        self.fills.append(text)
        self.composer = text
        if self.on_fill is not None:
            self.on_fill(self)

    def composer_text(self) -> str:
        self.composer_reads += 1
        if self.on_composer_read is not None:
            self.on_composer_read(self, self.composer_reads)
        return self.composer

    def send_controls(self) -> tuple[SendControl, ...]:
        return self.controls

    def click_send(self, name: str) -> None:
        # A plain WhatsAppError is a definite pre-click refusal (the real page
        # raises it when the control is not unique); an uncertain/generic error
        # means the click was attempted.
        definite = isinstance(self.click_error, WhatsAppError) and not isinstance(
            self.click_error, WhatsAppSendUncertain
        )
        if not definite:
            self.clicked.append(name)
        if self.click_error is not None:
            raise self.click_error


class _Session:
    def __init__(self, page: _TextPage) -> None:
        self._page = page

    def whatsapp_page(self) -> _TextPage:
        return self._page


class _Holder:
    """Stands in for ``ToolContext.whatsapp_browser`` (a LazyProfileBrowser)."""

    def __init__(self, page: _TextPage | None) -> None:
        self.page = page
        self.acquires = 0

    def acquire(self) -> _Session | None:
        self.acquires += 1
        return _Session(self.page) if self.page is not None else None


def _store(tmp_path: Path, *contacts: tuple[str, str]) -> ContactStore:
    """Build a contact store; Rahul is the default when no contacts are named.

    A fixture that needs several contacts passes them all with distinct numbers so
    ContactStore's duplicate-name/number guards are never triggered by test data.
    """
    store = ContactStore(tmp_path / "contacts.json")
    known = {unicodedata.normalize("NFC", c.name).casefold() for c in store.load()}
    for name, number in contacts or (("Rahul", _PHONE),):
        key = unicodedata.normalize("NFC", name).casefold()
        if key in known:
            continue
        store.add(name, number)
        known.add(key)
    return store


def _emoji_store(tmp_path: Path) -> ContactStore:
    """A store that contains a decorated contact with a flag."""
    return _store(tmp_path, ("🇮🇳 Mom ❤️", _MOM_LOGO_PHONE))


def _limiter(tmp_path: Path) -> RateLimiter:
    return RateLimiter(tmp_path / "ratelimit.json", max_per_hour=10, min_interval_s=0.0)


def _guard(tmp_path: Path, clock: _Clock | None = None) -> FileSendGuard:
    return FileSendGuard(tmp_path / "text_guard.json", clock=clock or _Clock())


def _ctx(
    holder: _Holder | None,
    *,
    operation_id: str = "t1:abc",
    dry_run: bool = False,
) -> ToolContext:
    return ToolContext(
        settings=Settings(),
        whatsapp_browser=holder,
        operation_id=operation_id,
        dry_run=dry_run,
    )


def _spec(
    tmp_path: Path,
    *,
    store: ContactStore | None = None,
    limiter: RateLimiter | None = None,
    guard: FileSendGuard | None = None,
) -> ToolSpec:
    return make_whatsapp_send_message_spec(
        store=store or _store(tmp_path),
        limiter=limiter or _limiter(tmp_path),
        guard=guard or _guard(tmp_path),
    )


# ── args / registration ───────────────────────────────────────────────────


def test_args_are_contact_and_exact_message() -> None:
    assert set(WhatsAppSendMessageArgs.model_fields) == {"contact", "message"}


def test_whitespace_only_and_nul_messages_are_refused() -> None:
    with pytest.raises(ValidationError, match="visible text"):
        WhatsAppSendMessageArgs(contact="Rahul", message=" \n\t ")
    with pytest.raises(ValidationError, match="NUL"):
        WhatsAppSendMessageArgs(contact="Rahul", message="hello\x00world")


def test_over_long_message_does_not_leak_into_the_validation_error() -> None:
    secret = "s" * (MAX_MESSAGE_CHARS + 1)
    with pytest.raises(ValidationError) as caught:
        WhatsAppSendMessageArgs(contact="Rahul", message=secret)
    assert secret[:32] not in str(caught.value)


def test_extra_arguments_are_forbidden() -> None:
    with pytest.raises(ValidationError):
        WhatsAppSendMessageArgs(contact="Rahul", message="hi", path="x")


def test_registered_as_tier2_without_path_or_content_binding() -> None:
    registry = build_default_registry()
    spec = registry.get_optional("whatsapp_send_message")
    assert spec is not None
    assert spec.base_tier == 2
    assert spec.windows_only is True
    assert spec.budget_cost == 1
    assert spec.path_args == ()
    assert spec.bind_content_hash is False
    assert set(spec.args_model.model_fields) == {"contact", "message"}
    assert "whatsapp_send_message" in GROUPS["messaging"]


# ── composer comparison / page adapter mechanics ──────────────────────────


def test_composer_matches_only_ignores_whatsapp_formatting_artifacts() -> None:
    expected = "first line\nsecond line"
    assert composer_matches(expected, "first line\r\nsecond line\u200b\n")
    assert composer_matches(expected, "first line\nsecond line")
    assert not composer_matches(expected, "first line\nsecond line ")
    assert not composer_matches(expected, "first line\nsecond line!")


class _Locator:
    def __init__(self, count: int) -> None:
        self._count = count
        self.filled: list[str] = []

    def count(self) -> int:
        return self._count

    @property
    def first(self) -> _Locator:
        return self

    def fill(self, text: str) -> None:
        self.filled.append(text)


class _RawPage:
    def __init__(self, *, locators: dict[str, _Locator] | None = None, result: Any = "") -> None:
        self.locators = locators or {}
        self.result = result
        self.evaluated: list[Any] = []

    def locator(self, selector: str) -> _Locator:
        return self.locators.get(selector, _Locator(0))

    def evaluate(self, script: str, arg: Any) -> Any:
        self.evaluated.append((script, arg))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_fill_composer_uses_the_scoped_footer_selector() -> None:
    scoped = _Locator(1)
    raw = _RawPage(locators={"#main footer [contenteditable=\"true\"][role=\"textbox\"]": scoped})
    page = PlaywrightWhatsAppPage(raw)

    page.fill_composer("hello\nworld")

    assert scoped.filled == ["hello\nworld"]


def test_composer_text_reads_the_composer_and_fails_closed() -> None:
    assert PlaywrightWhatsAppPage(_RawPage(result="hello")).composer_text() == "hello"
    assert PlaywrightWhatsAppPage(_RawPage(result=RuntimeError("boom"))).composer_text() == ""


def test_click_send_distinguishes_uncertain_from_ambiguous() -> None:
    uncertain = PlaywrightWhatsAppPage(_RawPage(result=RuntimeError("lost")))
    with pytest.raises(WhatsAppSendUncertain):
        uncertain.click_send("Send")

    ambiguous = PlaywrightWhatsAppPage(_RawPage(result=2))
    with pytest.raises(WhatsAppError) as caught:
        ambiguous.click_send("Send")
    assert not isinstance(caught.value, WhatsAppSendUncertain)


# ── send_message ──────────────────────────────────────────────────────────


def test_send_message_happy_path(tmp_path: Path) -> None:
    message = "Hello Rahul\nsecond line"
    page = _TextPage()
    holder = _Holder(page)
    ctx = _ctx(holder)

    result = _spec(tmp_path).execute(
        WhatsAppSendMessageArgs(contact="Rahul", message=message), ctx
    )

    assert result.ok, result.error
    assert result.verified is None
    assert "dispatched" in result.verify_note
    assert page.searched == [_PHONE]
    assert page.fills == [message]
    assert page.clicked == ["Send"]
    assert page.composer == message
    assert result.data["send_outcome"] == "dispatched"
    assert result.data["delivery_verified"] is False
    assert result.data["recipient"] == "Rahul"
    assert message not in result.output
    assert message not in json.dumps(result.data)


def test_send_message_requires_an_approved_operation_id(tmp_path: Path) -> None:
    page = _TextPage()
    holder = _Holder(page)

    result = _spec(tmp_path).execute(
        WhatsAppSendMessageArgs(contact="Rahul", message="hi"), _ctx(holder, operation_id="")
    )

    assert result.ok is False
    assert "not bound to an approved operation" in result.error
    assert holder.acquires == 0
    assert page.fills == [] and page.clicked == []


def test_a_retry_of_the_same_operation_cannot_send_twice(tmp_path: Path) -> None:
    page = _TextPage()
    holder = _Holder(page)
    ctx = _ctx(holder)
    spec = _spec(tmp_path)
    args = WhatsAppSendMessageArgs(contact="Rahul", message="hi")

    first = spec.execute(args, ctx)
    second = spec.execute(args, ctx)

    assert first.ok, first.error
    assert second.ok is False
    assert "already attempted" in second.error
    assert holder.acquires == 1  # the duplicate is refused before the browser
    assert page.clicked == ["Send"]


def test_a_different_operation_may_send_the_same_text(tmp_path: Path) -> None:
    page = _TextPage()
    holder = _Holder(page)
    spec = _spec(tmp_path)
    args = WhatsAppSendMessageArgs(contact="Rahul", message="hi")

    first = spec.execute(args, _ctx(holder, operation_id="t1:a"))
    second = spec.execute(args, _ctx(holder, operation_id="t1:b"))

    assert first.ok, first.error
    assert second.ok, second.error
    assert holder.acquires == 2
    assert page.clicked == ["Send", "Send"]


def test_a_composer_mismatch_refuses_and_clicks_nothing(tmp_path: Path) -> None:
    page = _TextPage(on_fill=lambda p: setattr(p, "composer", p.composer + "!"))
    holder = _Holder(page)

    result = _spec(tmp_path).execute(
        WhatsAppSendMessageArgs(contact="Rahul", message="hi"), _ctx(holder)
    )

    assert result.ok is False
    assert "does not contain the approved message" in result.error
    assert page.clicked == []


def test_an_ambiguous_send_control_refuses_and_clicks_nothing(tmp_path: Path) -> None:
    page = _TextPage(
        controls=(SendControl(name="Send", index=0), SendControl(name="Send", index=1))
    )
    holder = _Holder(page)

    result = _spec(tmp_path).execute(
        WhatsAppSendMessageArgs(contact="Rahul", message="hi"), _ctx(holder)
    )

    assert result.ok is False
    assert "more than one WhatsApp Send control" in result.error
    assert page.clicked == []


def test_an_uncertain_click_is_reported_and_not_retried(tmp_path: Path) -> None:
    page = _TextPage(click_error=WhatsAppSendUncertain("lost"))
    holder = _Holder(page)
    ctx = _ctx(holder)
    spec = _spec(tmp_path)
    args = WhatsAppSendMessageArgs(contact="Rahul", message="hi")

    first = spec.execute(args, ctx)
    second = spec.execute(args, ctx)

    assert first.ok, first.error
    assert first.verified is None
    assert first.data["send_outcome"] == "unknown"
    assert first.data["delivery_verified"] is False
    assert "unknown" in first.verify_note
    assert page.clicked == ["Send"]
    assert second.ok is False
    assert "already attempted" in second.error
    assert holder.acquires == 1


def test_a_new_guard_instance_still_refuses_the_same_approval(tmp_path: Path) -> None:
    """A restarted daemon (fresh guard instance, same file) cannot resend."""
    page = _TextPage()
    holder = _Holder(page)
    args = WhatsAppSendMessageArgs(contact="Rahul", message="hi")

    first = _spec(tmp_path, guard=_guard(tmp_path)).execute(args, _ctx(holder))
    second = _spec(tmp_path, guard=_guard(tmp_path)).execute(args, _ctx(holder))

    assert first.ok, first.error
    assert second.ok is False
    assert "already attempted" in second.error
    assert page.clicked == ["Send"]


def test_concurrent_dispatches_of_one_approval_click_send_once(tmp_path: Path) -> None:
    """Two invocations of the same approval cannot both claim the guard."""
    barrier = threading.Barrier(2)
    pages = [
        _TextPage(on_fill=lambda page: barrier.wait(timeout=5)),
        _TextPage(on_fill=lambda page: barrier.wait(timeout=5)),
    ]
    specs = [_spec(tmp_path, guard=_guard(tmp_path)) for _ in range(2)]
    args = WhatsAppSendMessageArgs(contact="Rahul", message="hi")
    results: list[ToolResult] = []
    results_lock = threading.Lock()

    def run(spec: ToolSpec, page: _TextPage) -> None:
        result = spec.execute(args, _ctx(_Holder(page)))
        with results_lock:
            results.append(result)

    threads = [
        threading.Thread(target=run, args=(specs[index], pages[index])) for index in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not any(thread.is_alive() for thread in threads)
    assert sum(1 for result in results if result.ok) == 1, [result.error for result in results]
    assert sum(len(page.clicked) for page in pages) == 1
    assert any("already attempted" in (result.error or "") for result in results)


@pytest.mark.parametrize(
    "state,fragment",
    [
        (WhatsAppSessionState.QR_REQUIRED, "QR"),
        (WhatsAppSessionState.LOGGED_OUT, "logged out"),
        (WhatsAppSessionState.UNKNOWN, "could not be authenticated"),
    ],
)
def test_unauthenticated_sessions_refuse_before_composing(
    tmp_path: Path, state: WhatsAppSessionState, fragment: str
) -> None:
    page = _TextPage(state=state)

    result = _spec(tmp_path).execute(
        WhatsAppSendMessageArgs(contact="Rahul", message="hi"), _ctx(_Holder(page))
    )

    assert result.ok is False
    assert fragment in (result.error or "")
    assert page.fills == [] and page.clicked == []


def test_an_unavailable_session_probe_refuses(tmp_path: Path) -> None:
    page = _TextPage(session_error=RuntimeError("probe failed"))

    result = _spec(tmp_path).execute(
        WhatsAppSendMessageArgs(contact="Rahul", message="hi"), _ctx(_Holder(page))
    )

    assert result.ok is False
    assert "could not be authenticated" in (result.error or "")
    assert page.fills == [] and page.clicked == []


def test_a_foreign_origin_refuses_before_composing(tmp_path: Path) -> None:
    page = _TextPage(redirect_to="https://evil.example/send")

    result = _spec(tmp_path).execute(
        WhatsAppSendMessageArgs(contact="Rahul", message="hi"), _ctx(_Holder(page))
    )

    assert result.ok is False
    assert "not on https://web.whatsapp.com" in (result.error or "")
    assert page.searched == [] and page.fills == [] and page.clicked == []


def test_a_chat_identity_change_after_compose_refuses_before_click(tmp_path: Path) -> None:
    page = _TextPage(on_fill=lambda current: setattr(current, "header", "Someone Else"))

    result = _spec(tmp_path).execute(
        WhatsAppSendMessageArgs(contact="Rahul", message="hi"), _ctx(_Holder(page))
    )

    assert result.ok is False
    assert "could not be confirmed" in (result.error or "")
    assert page.clicked == []


def test_text_changed_after_compose_refuses_before_click(tmp_path: Path) -> None:
    def tamper(page: _TextPage, read_number: int) -> None:
        if read_number == 2:
            page.composer = page.composer + "!"

    page = _TextPage(on_composer_read=tamper)

    result = _spec(tmp_path).execute(
        WhatsAppSendMessageArgs(contact="Rahul", message="hi"), _ctx(_Holder(page))
    )

    assert result.ok is False
    assert "no longer contains the approved message" in (result.error or "")
    assert page.clicked == []


def test_a_definite_click_refusal_is_not_reported_as_uncertain(tmp_path: Path) -> None:
    page = _TextPage(click_error=WhatsAppError("Send control is no longer unique"))

    result = _spec(tmp_path).execute(
        WhatsAppSendMessageArgs(contact="Rahul", message="hi"), _ctx(_Holder(page))
    )

    assert result.ok is False
    assert "could not be confirmed" in (result.error or "")
    assert result.data.get("send_outcome") is None
    assert page.clicked == []


def test_a_generic_click_failure_after_dispatch_is_uncertain(tmp_path: Path) -> None:
    page = _TextPage(click_error=RuntimeError("browser lost the page"))

    result = _spec(tmp_path).execute(
        WhatsAppSendMessageArgs(contact="Rahul", message="hi"), _ctx(_Holder(page))
    )

    assert result.ok is True
    assert result.verified is None
    assert result.data["send_outcome"] == "unknown"
    assert result.data["delivery_verified"] is False
    assert page.clicked == ["Send"]


def test_a_composer_refusal_does_not_include_the_message(tmp_path: Path) -> None:
    page = _TextPage(on_fill=lambda current: setattr(current, "composer", current.composer + "!"))

    result = _spec(tmp_path).execute(
        WhatsAppSendMessageArgs(contact="Rahul", message=_SECRET), _ctx(_Holder(page))
    )

    assert result.ok is False
    assert _SECRET not in (result.error or "")
    assert _SECRET not in result.output


def test_whatsapp_send_message_is_not_voice_tier2_eligible() -> None:
    from types import SimpleNamespace

    from jarvis.voice.tier2_voice import VOICE_TIER2_TOOLS, allowed

    settings = SimpleNamespace(voice=SimpleNamespace(allow_tier2_by_voice=True))
    ok, reason = allowed("whatsapp_send_message", {"tier": 2, "allowed": True}, settings)

    assert "whatsapp_send_message" not in VOICE_TIER2_TOOLS
    assert ok is False
    assert "cannot be approved by voice" in reason


def test_rate_limit_refuses_before_the_browser(tmp_path: Path) -> None:
    holder = _Holder(_TextPage())
    limiter = RateLimiter(tmp_path / "rl.json", max_per_hour=1, min_interval_s=0.0)
    limiter.record()
    spec = _spec(tmp_path, limiter=limiter)

    result = spec.execute(
        WhatsAppSendMessageArgs(contact="Rahul", message="hi"), _ctx(holder)
    )

    assert result.ok is False
    assert "rate limit" in result.error
    assert holder.acquires == 0


def test_unknown_and_ambiguous_contacts_are_refused_with_options(tmp_path: Path) -> None:
    store = _store(tmp_path, ("Ali Khan", "9112345678"), ("Ali Ahmed", _PHONE))
    holder = _Holder(_TextPage())
    spec = _spec(tmp_path, store=store)
    ctx = _ctx(holder)

    unknown = spec.execute(WhatsAppSendMessageArgs(contact="Nobody", message="hi"), ctx)
    assert unknown.ok is False
    assert "no contact matches" in unknown.error
    assert "Ali Khan" in unknown.error

    ambiguous = spec.execute(WhatsAppSendMessageArgs(contact="Ali", message="hi"), ctx)
    assert ambiguous.ok is False
    assert "multiple contacts match" in ambiguous.error
    assert "Ali Khan" in ambiguous.error and "Ali Ahmed" in ambiguous.error
    assert holder.acquires == 0


def test_contact_store_still_rejects_duplicate_contacts(tmp_path: Path) -> None:
    """Test fixes must not weaken the store's duplicate protections."""
    store = _store(tmp_path, ("Mom ❤️", _MOM_PHONE))

    with pytest.raises(ValueError, match="already exists"):
        store.add("Mom ❤️", _MOM_BLUE_PHONE)
    with pytest.raises(ValueError, match="already assigned"):
        store.add("Mom 💙", _MOM_PHONE)


def test_an_emoji_saved_contact_is_found_by_the_bare_word_only_when_unique(
    tmp_path: Path,
) -> None:
    """"Mom" resolves to "Mom ❤️" only because exactly one saved contact normalises to "mom"."""
    page = _TextPage(header="Mom ❤️")
    store = _store(tmp_path, ("Mom ❤️", _MOM_PHONE))
    holder = _Holder(page)
    spec = _spec(tmp_path, store=store)
    ctx = _ctx(holder)

    result = spec.execute(
        WhatsAppSendMessageArgs(contact="Mom", message="hi"), ctx
    )

    assert result.ok, result.error
    assert result.data["recipient"] == "Mom ❤️"
    assert holder.acquires == 1
    assert page_is_verified(page, _MOM_PHONE)


def test_two_emoji_decorated_mom_contacts_are_ambiguous(tmp_path: Path) -> None:
    """"Mom" must not silently pick between "Mom ❤️" and "Mom 💙"."""
    page = _TextPage(header="Mom ❤️")
    store = _store(tmp_path, ("Mom ❤️", _MOM_PHONE), ("Mom 💙", _MOM_BLUE_PHONE))
    holder = _Holder(page)
    spec = _spec(tmp_path, store=store)
    ctx = _ctx(holder)

    result = spec.execute(
        WhatsAppSendMessageArgs(contact="Mom", message="hi"), ctx
    )

    assert result.ok is False
    assert "multiple contacts match" in result.error
    assert "Mom ❤️" in result.error and "Mom 💙" in result.error
    assert holder.acquires == 0


def test_the_exact_decorated_name_resolves_its_contact(
    tmp_path: Path,
) -> None:
    page = _TextPage(header="Mom ❤️")
    store = _store(tmp_path, ("Mom ❤️", _MOM_PHONE))
    holder = _Holder(page)
    spec = _spec(tmp_path, store=store)
    ctx = _ctx(holder)

    result = spec.execute(
        WhatsAppSendMessageArgs(contact="Mom ❤️", message="hi"), ctx
    )

    assert result.ok, result.error
    assert result.data["recipient"] == "Mom ❤️"
    assert holder.acquires == 1
    assert page_is_verified(page, _MOM_PHONE)


def test_mom_flag_love_is_found_by_the_bare_word_only_when_unique(
    tmp_path: Path,
) -> None:
    """"🇮🇳 Mom ❤️" is found by "Mom" because only one saved contact normalises to "mom"."""
    page = _TextPage(header="🇮🇳 Mom ❤️")
    store = _emoji_store(tmp_path)
    holder = _Holder(page)
    spec = _spec(tmp_path, store=store)
    ctx = _ctx(holder)

    result = spec.execute(
        WhatsAppSendMessageArgs(contact="Mom", message="hi"), ctx
    )

    assert result.ok, result.error
    assert result.data["recipient"] == "🇮🇳 Mom ❤️"
    assert holder.acquires == 1
    assert page_is_verified(page, _MOM_LOGO_PHONE)


def test_mamá_heart_is_found_by_the_bare_word_only_when_unique(
    tmp_path: Path,
) -> None:
    """"Mamá ❤️" can be found by "Mamá" because the canonical NFC forms still match."""
    page = _TextPage(header="Mamá ❤️")
    store = _store(tmp_path, ("Mamá ❤️", _MOM_PHONE))
    holder = _Holder(page)
    spec = _spec(tmp_path, store=store)
    ctx = _ctx(holder)

    result = spec.execute(
        WhatsAppSendMessageArgs(contact="Mamá", message="hi"), ctx
    )

    assert result.ok, result.error
    assert result.data["recipient"] == "Mamá ❤️"
    assert holder.acquires == 1
    assert page_is_verified(page, _MOM_PHONE)


def test_two_mom_contacts_with_different_emoji_remain_ambiguous(
    tmp_path: Path,
) -> None:
    """Two differently decorated Mom contacts with different numbers are still ambiguous."""
    page = _TextPage(header="Mom ❤️")
    store = _store(
        tmp_path,
        ("Mom ❤️", _MOM_PHONE),
        ("Mom 💙", _MOM_BLUE_PHONE),
    )
    holder = _Holder(page)
    spec = _spec(tmp_path, store=store)
    ctx = _ctx(holder)

    result = spec.execute(
        WhatsAppSendMessageArgs(contact="Mom", message="hi"), ctx
    )

    assert result.ok is False
    assert "multiple contacts match" in result.error
    assert "Mom ❤️" in result.error and "Mom 💙" in result.error
    assert holder.acquires == 0


def test_an_unknown_decorated_contact_is_refused_with_options(
    tmp_path: Path,
) -> None:
    page = _TextPage(header="Mom ❤️")
    store = _store(tmp_path, ("Mom ❤️", _MOM_PHONE))
    holder = _Holder(page)
    spec = _spec(tmp_path, store=store)
    ctx = _ctx(holder)

    result = spec.execute(
        WhatsAppSendMessageArgs(contact="Dad 💙", message="hi"), ctx
    )

    assert result.ok is False
    assert "no contact matches" in result.error
    assert "Mom ❤️" in result.error
    assert holder.acquires == 0


def test_empty_contact_store_produces_a_clear_response(tmp_path: Path) -> None:
    page = _TextPage()
    store = ContactStore(tmp_path / "empty.json")
    holder = _Holder(page)
    spec = _spec(tmp_path, store=store)
    ctx = _ctx(holder)

    result = spec.execute(
        WhatsAppSendMessageArgs(contact="Mom", message="hi"), ctx
    )

    assert result.ok is False
    assert "no contact matches" in result.error
    assert "(no contacts yet)" in result.error
    assert holder.acquires == 0


def test_a_chat_that_does_not_prove_the_contact_is_refused(tmp_path: Path) -> None:
    page = _TextPage(header="Someone Else")
    holder = _Holder(page)

    result = _spec(tmp_path).execute(
        WhatsAppSendMessageArgs(contact="Rahul", message="hi"), _ctx(holder)
    )

    assert result.ok is False
    assert "could not open the WhatsApp chat" in result.error
    assert page.fills == [] and page.clicked == []


def test_non_windows_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("jarvis.platform_guard.is_windows", lambda: False)
    page = _TextPage()
    holder = _Holder(page)

    result = _spec(tmp_path).execute(
        WhatsAppSendMessageArgs(contact="Rahul", message="hi"), _ctx(holder)
    )

    assert result.ok is False
    assert "only supported on Windows" in result.error
    assert holder.acquires == 0
    assert page.fills == [] and page.clicked == []


def test_describe_shows_the_exact_message_and_masks_the_number(tmp_path: Path) -> None:
    summary = _spec(tmp_path).describe(
        WhatsAppSendMessageArgs(contact="Rahul", message="hello\nworld")
    )

    assert "hello\nworld" in summary
    assert "Rahul" in summary
    assert "\u2022" * 8 + "10" in summary
    assert _PHONE not in summary


def test_the_message_is_never_logged(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    page = _TextPage()
    holder = _Holder(page)
    spec = _spec(tmp_path)

    result = spec.execute(
        WhatsAppSendMessageArgs(contact="Rahul", message=_SECRET), _ctx(holder)
    )
    assert result.ok, result.error
    assert _SECRET not in caplog.text


def test_an_unknown_outcome_does_not_log_the_message(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    page = _TextPage(click_error=WhatsAppSendUncertain("lost"))
    holder = _Holder(page)
    spec = _spec(tmp_path)

    result = spec.execute(
        WhatsAppSendMessageArgs(contact="Rahul", message=_SECRET), _ctx(holder)
    )
    assert result.ok, result.error
    assert _SECRET not in caplog.text


# ── act wiring ────────────────────────────────────────────────────────────


class _OpArgs(BaseModel):
    contact: str
    message: str


def _op_spec(record: list[str]) -> ToolSpec:
    def run(args: _OpArgs, ctx: ToolContext) -> ToolResult:
        del args
        record.append(ctx.operation_id)
        return ToolResult(ok=True, output="sent")

    return ToolSpec(
        name="fake_send_message",
        description="fake confirmed text send",
        args_model=_OpArgs,
        base_tier=2,
        run=run,
        describe=lambda args: "send a message",
    )


def test_act_sets_the_operation_id_to_task_and_action_hash(tmp_path: Path) -> None:
    record: list[str] = []
    spec = _op_spec(record)
    ctx = make_app_context(Settings(), registry=registry_with(spec))
    step = Step(
        id="s1",
        tool="fake_send_message",
        args={"contact": "Rahul", "message": "hi"},
        rationale="send it",
    )
    action = str(rules.action_hash(spec, spec.args_model.model_validate(step.args)))
    state: dict[str, Any] = {
        "plan": Plan(goal="send the message", steps=[step]),
        "step_index": 0,
        "task_id": "t1",
        "approved_hashes": [action],
        "results": [],
        "user_input": "send a message to Rahul",
    }

    update = act(state, ctx)

    results = list(update.get("results") or [])
    assert results, "act must record a result"
    assert results[-1].ok, results[-1].error
    assert record == [f"t1:{action}"]
