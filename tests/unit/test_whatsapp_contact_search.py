"""W2: verified WhatsApp contact search.

No browser, no Playwright, no network: the real driver, adapter and tool run
against a fake narrow page with a fake contact store.

Two layers are pinned:

* the *safe alternative* actually wired into ``whatsapp_open_chat`` - a search
  keyed to the trusted contact's **number**, requiring exactly one hit and a
  header that proves that number (or is exactly the saved name);
* the name-search driver primitive, which is tightened as well: a partial hit is
  never selected and a substring header never proves identity.

Name-only search is deliberately **not** reachable from any tool: a display
name cannot distinguish two people who share it, so it is not identity evidence
on its own.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from jarvis.browser.whatsapp_web import (
    WHATSAPP_BASE_URL,
    WHATSAPP_SEND_URL,
    AttachmentScan,
    SearchHit,
    SendControl,
    WhatsAppDriver,
    WhatsAppError,
    WhatsAppSessionState,
    header_confirms_recipient,
)
from jarvis.config import Settings
from jarvis.messaging.contacts import ContactStore
from jarvis.tools.base import ToolContext, ToolResult
from jarvis.tools.whatsapp_web import WhatsAppOpenChatArgs, make_whatsapp_open_chat_spec

_PHONE = "919876543210"
_MOM_PHONE = "919876543211"
_MOM_BLUE_PHONE = "919876543212"


@pytest.fixture(autouse=True)
def _windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test runs as if on Windows unless it overrides this."""
    monkeypatch.setattr("jarvis.platform_guard.is_windows", lambda: True)


class _Page:
    """A fake narrow WhatsApp page (records every search and every action)."""

    def __init__(
        self,
        *,
        hits: tuple[SearchHit, ...] = (),
        header: str = "Rahul",
        state: WhatsAppSessionState = WhatsAppSessionState.AUTHENTICATED,
        redirect_to: str = "",
    ) -> None:
        self.hits = hits
        self.header = header
        self.state = state
        self.redirect_to = redirect_to
        self.current_url = WHATSAPP_BASE_URL
        self.visited: list[str] = []
        self.searched: list[str] = []
        self.opened: list[SearchHit] = []
        self.staged_paths: list[str] = []
        self.clicked: list[str] = []
        self.controls: tuple[SendControl, ...] = ()

    # ── WhatsAppPage protocol ──────────────────────────────────────────────

    def url(self) -> str:
        return self.current_url

    def goto(self, url: str) -> None:
        self.visited.append(url)
        self.current_url = self.redirect_to or url

    def session_state(self) -> WhatsAppSessionState:
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

    def set_input_files(self, path: str) -> None:
        self.staged_paths.append(path)

    def staged_file_names(self) -> tuple[str, ...]:
        return ()

    def attachment_scan(self, filename: str) -> AttachmentScan:
        del filename
        return AttachmentScan(panel_found=False)

    def send_controls(self) -> tuple[SendControl, ...]:
        return self.controls

    def click_send(self, name: str) -> None:
        self.clicked.append(name)


class _Session:
    def __init__(self, page: _Page) -> None:
        self._page = page

    def whatsapp_page(self) -> _Page:
        return self._page


class _Holder:
    """Stands in for ``ToolContext.whatsapp_browser`` (a LazyProfileBrowser)."""

    def __init__(self, page: _Page | None) -> None:
        self.page = page
        self.acquires = 0

    def acquire(self) -> _Session | None:
        self.acquires += 1
        return _Session(self.page) if self.page is not None else None


def _store(tmp_path: Path, *contacts: tuple[str, str]) -> ContactStore:
    """Build a contact store; Rahul is the default when no contacts are named.

    Every intended contact gets a distinct number, so a test never collides with
    ContactStore's (correct) duplicate-number guard.
    """
    store = ContactStore(tmp_path / "contacts.json")
    for name, number in contacts or (("Rahul", _PHONE),):
        store.add(name, number)
    return store


def _ctx(holder: _Holder) -> ToolContext:
    return ToolContext(settings=Settings(), whatsapp_browser=holder)


def _open(tmp_path: Path, page: _Page, **args: Any) -> tuple[ToolResult, _Page, _Holder]:
    return _open_contact(tmp_path, page, "Rahul", **args)


def _open_contact(
    tmp_path: Path,
    page: _Page,
    contact: str,
    *contacts: tuple[str, str],
    **args: Any,
) -> tuple[ToolResult, _Page, _Holder]:
    holder = _Holder(page)
    spec = make_whatsapp_open_chat_spec(store=_store(tmp_path, *contacts))
    result = spec.execute(WhatsAppOpenChatArgs(contact=contact, **args), _ctx(holder))
    return result, page, holder


def _search(tmp_path: Path, page: _Page) -> ToolResult:
    result, _page, _holder = _open(tmp_path, page, via="search")
    return result


# ── the wired route: number search ─────────────────────────────────────────


def test_number_search_opens_the_verified_chat(tmp_path: Path) -> None:
    page = _Page(hits=(SearchHit(name="Rahul"),))

    result, page, _holder = _open(tmp_path, page, via="search")

    assert result.ok, result.error
    assert result.verified is True
    assert result.data["via"] == "search"
    assert page.visited == [WHATSAPP_BASE_URL]
    # The query is the trusted number, never the display name.
    assert page.searched == [_PHONE]
    assert page.opened == [SearchHit(name="Rahul")]
    assert page.staged_paths == [] and page.clicked == []


def test_number_search_refuses_when_nothing_matches(tmp_path: Path) -> None:
    page = _Page(hits=())

    result = _search(tmp_path, page)

    assert result.ok is False
    assert "no chat for that number" in (result.error or "")
    assert page.opened == []


def test_number_search_refuses_multiple_hits(tmp_path: Path) -> None:
    page = _Page(hits=(SearchHit(name="Rahul"), SearchHit(name="Rahul Office")))

    result = _search(tmp_path, page)

    assert result.ok is False
    assert "more than one chat" in (result.error or "")
    assert page.opened == []


def test_number_search_refuses_a_header_naming_another_number(tmp_path: Path) -> None:
    """The opened chat shows a different number: it is not the contact."""
    page = _Page(hits=(SearchHit(name="Rahul"),), header="Rahul +91 99999 11111")

    result = _search(tmp_path, page)

    assert result.ok is False
    assert "does not prove" in (result.error or "")


def test_number_search_refuses_a_header_that_is_not_exactly_the_contact(
    tmp_path: Path,
) -> None:
    page = _Page(hits=(SearchHit(name="Rahul"),), header="Rahul Sharma")

    result = _search(tmp_path, page)

    assert result.ok is False
    assert "does not prove" in (result.error or "")


def test_number_search_refuses_an_unauthenticated_session(tmp_path: Path) -> None:
    page = _Page(hits=(SearchHit(name="Rahul"),), state=WhatsAppSessionState.QR_REQUIRED)

    result = _search(tmp_path, page)

    assert result.ok is False
    assert "QR-code sign-in" in (result.error or "")
    assert page.searched == [] and page.opened == []


def test_number_search_refuses_a_foreign_origin(tmp_path: Path) -> None:
    page = _Page(hits=(SearchHit(name="Rahul"),), redirect_to="https://evil.example/send")

    result = _search(tmp_path, page)

    assert result.ok is False
    assert "not on https://web.whatsapp.com" in (result.error or "")
    assert page.searched == [] and page.opened == []


def test_an_unknown_contact_never_starts_the_browser(tmp_path: Path) -> None:
    page = _Page(hits=(SearchHit(name="Rahul"),))
    holder = _Holder(page)
    spec = make_whatsapp_open_chat_spec(store=_store(tmp_path))

    result = spec.execute(WhatsAppOpenChatArgs(contact="Nobody", via="search"), _ctx(holder))

    assert result.ok is False
    assert holder.acquires == 0
    assert page.searched == [] and page.opened == []


def test_the_phone_route_is_still_the_default(tmp_path: Path) -> None:
    page = _Page()

    result, page, _holder = _open(tmp_path, page)

    assert result.ok, result.error
    assert result.data["via"] == "phone"
    assert page.visited == [WHATSAPP_SEND_URL.format(digits=_PHONE)]
    assert page.searched == []


def test_search_is_read_only_apart_from_opening_the_chat(tmp_path: Path) -> None:
    page = _Page(hits=(SearchHit(name="Rahul"),))

    _search(tmp_path, page)

    assert page.clicked == []
    assert page.staged_paths == []


# ── the name-search primitive is tightened ─────────────────────────────────


def test_a_partial_hit_is_never_selected() -> None:
    page = _Page(hits=(SearchHit(name="Rahul Sharma"),))

    with pytest.raises(WhatsAppError, match="no exact match"):
        WhatsAppDriver(page).open_chat_by_name("Rahul")

    assert page.opened == []


def test_a_substring_header_does_not_prove_identity() -> None:
    page = _Page(hits=(SearchHit(name="Rahul"),), header="Rahul Sharma")

    with pytest.raises(WhatsAppError, match="does not prove"):
        WhatsAppDriver(page).open_chat_by_name("Rahul")


def test_name_search_still_opens_a_unique_exact_hit() -> None:
    page = _Page(hits=(SearchHit(name="Rahul"),), header="Rahul")

    assert WhatsAppDriver(page).open_chat_by_name("Rahul") == "Rahul"
    assert page.opened == [SearchHit(name="Rahul")]


def test_number_search_opens_a_decorated_contact_when_the_header_matches(
    tmp_path: Path,
) -> None:
    """An exact decorated saved name opens when the header proves that identity."""
    page = _Page(hits=(SearchHit(name="Mom ❤️"),), header="Mom ❤️")

    # The trusted number is the identity; the decorated header only confirms the
    # opened chat matches the resolved saved contact.
    result, page, _holder = _open_contact(
        tmp_path, page, "Mom ❤️", ("Mom ❤️", _MOM_PHONE), via="search"
    )

    assert result.ok, result.error
    assert result.data["recipient"] == "Mom ❤️"
    assert page.searched == [_MOM_PHONE]
    assert page.opened == [SearchHit(name="Mom ❤️")]


def test_a_bare_word_opens_a_unique_decorated_contact(tmp_path: Path) -> None:
    """"Mom" resolves the only saved contact whose name normalises to "mom"."""
    page = _Page(hits=(SearchHit(name="Mom ❤️"),), header="Mom ❤️")

    result, page, holder = _open_contact(
        tmp_path, page, "Mom", ("Mom ❤️", _MOM_PHONE), via="search"
    )

    assert result.ok, result.error
    assert result.data["recipient"] == "Mom ❤️"
    assert holder.acquires == 1
    assert page.searched == [_MOM_PHONE]


def test_number_search_accepts_a_unique_emoji_difference(tmp_path: Path) -> None:
    """"Mom ❤️" is saved; a unique header "Mom 💙" proves the same identity."""
    page = _Page(hits=(SearchHit(name="Mom ❤️"),), header="Mom 💙")

    result, page, _holder = _open_contact(
        tmp_path, page, "Mom ❤️", ("Mom ❤️", _MOM_PHONE), via="search"
    )

    assert result.ok, result.error
    assert result.data["recipient"] == "Mom ❤️"
    assert page.searched == [_MOM_PHONE]


def test_number_search_accepts_a_bare_name_header_for_a_unique_contact(
    tmp_path: Path,
) -> None:
    """A header without the saved emoji proves the unique saved contact."""
    page = _Page(hits=(SearchHit(name="Mom ❤️"),), header="Mom")

    result, page, _holder = _open_contact(
        tmp_path, page, "Mom ❤️", ("Mom ❤️", _MOM_PHONE), via="search"
    )

    assert result.ok, result.error
    assert result.data["recipient"] == "Mom ❤️"


def test_a_name_only_header_cannot_resolve_an_ambiguous_emoji_key(
    tmp_path: Path,
) -> None:
    """Two contacts share the key "mom"; a bare header cannot distinguish them."""
    page = _Page(hits=(SearchHit(name="Mom ❤️"),), header="Mom")

    result, _page, _holder = _open_contact(
        tmp_path,
        page,
        "Mom ❤️",
        ("Mom ❤️", _MOM_PHONE),
        ("Mom 💙", _MOM_BLUE_PHONE),
        via="search",
    )

    assert result.ok is False
    assert "does not prove" in (result.error or "")


def test_a_different_decoration_cannot_resolve_an_ambiguous_emoji_key(
    tmp_path: Path,
) -> None:
    page = _Page(hits=(SearchHit(name="Mom ❤️"),), header="Mom 🥰")

    result, _page, _holder = _open_contact(
        tmp_path,
        page,
        "Mom ❤️",
        ("Mom ❤️", _MOM_PHONE),
        ("Mom 💙", _MOM_BLUE_PHONE),
        via="search",
    )

    assert result.ok is False
    assert "does not prove" in (result.error or "")


def test_an_exact_decorated_header_resolves_an_ambiguous_emoji_key(
    tmp_path: Path,
) -> None:
    """The whole exact saved name is distinguishing evidence."""
    page = _Page(hits=(SearchHit(name="Mom ❤️"),), header="Mom ❤️")

    result, _page, _holder = _open_contact(
        tmp_path,
        page,
        "Mom ❤️",
        ("Mom ❤️", _MOM_PHONE),
        ("Mom 💙", _MOM_BLUE_PHONE),
        via="search",
    )

    assert result.ok, result.error
    assert result.data["recipient"] == "Mom ❤️"


def test_a_matching_header_number_resolves_an_ambiguous_emoji_key(
    tmp_path: Path,
) -> None:
    """The trusted number remains the stronger identity evidence."""
    page = _Page(
        hits=(SearchHit(name="Mom ❤️"),),
        header=f"Mom ❤️ +91 {_MOM_PHONE[-10:-5]} {_MOM_PHONE[-5:]}",
    )

    result, _page, _holder = _open_contact(
        tmp_path,
        page,
        "Mom ❤️",
        ("Mom ❤️", _MOM_PHONE),
        ("Mom 💙", _MOM_BLUE_PHONE),
        via="search",
    )

    assert result.ok, result.error


def test_number_search_refuses_a_header_naming_a_different_phone(tmp_path: Path) -> None:
    page = _Page(
        hits=(SearchHit(name="Mom ❤️"),), header="Mom ❤️ +91 99999 11111"
    )

    result, page, _holder = _open_contact(
        tmp_path, page, "Mom ❤️", ("Mom ❤️", _MOM_PHONE), via="search"
    )

    assert result.ok is False
    assert "does not prove" in (result.error or "")
    assert page.staged_paths == [] and page.clicked == []


def test_ambiguous_decorated_names_are_refused_before_the_browser(
    tmp_path: Path,
) -> None:
    """A bare "Mom" never picks between "Mom ❤️" and "Mom 💙"."""
    page = _Page(hits=(SearchHit(name="Mom ❤️"),), header="Mom ❤️")

    result, page, holder = _open_contact(
        tmp_path,
        page,
        "Mom",
        ("Mom ❤️", _MOM_PHONE),
        ("Mom 💙", _MOM_BLUE_PHONE),
        via="search",
    )

    assert result.ok is False
    assert "multiple contacts match" in (result.error or "")
    assert "Mom ❤️" in (result.error or "") and "Mom 💙" in (result.error or "")
    assert holder.acquires == 0
    assert page.visited == [] and page.searched == []


# ── the header rule itself ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("header", "name", "digits", "expected"),
    [
        ("Rahul", "Rahul", _PHONE, True),
        ("Rahul +91 98765 43210", "Rahul", _PHONE, True),
        ("+91 98765 43210", "", _PHONE, True),
        ("Rahul Sharma", "Rahul", _PHONE, False),
        ("Rahul +91 99999 11111", "Rahul", _PHONE, False),
        ("", "Rahul", _PHONE, False),
        ("Rahul", "", "", False),
        ("Mom ❤️", "Mom ❤️", _PHONE, True),
        ("Mom", "Mom ❤️", _PHONE, True),
        ("Mom 💙", "Mom ❤️", _PHONE, True),
        ("Mom 🥰", "Mom ❤️", _PHONE, True),
        ("❤️ Mom", "Mom ❤️", _PHONE, True),
        ("Dad", "Mom ❤️", _PHONE, False),
        ("Mamá ❤️", "Mamá ❤️", _PHONE, True),
        ("Mamá", "Mamá ❤️", _PHONE, True),
        ("Mama", "Mamá ❤️", _PHONE, False),
    ],
)
def test_header_confirms_recipient(header: str, name: str, digits: str, expected: bool) -> None:
    assert header_confirms_recipient(header, name, digits) is expected


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("Mom", False),
        ("Mom 🥰", False),
        ("Mom ❤️", True),
        ("Mom ❤️ +91 98765 43211", True),
    ],
)
def test_header_confirms_recipient_when_the_name_key_is_ambiguous(
    header: str, expected: bool
) -> None:
    """A name-only header must distinguish two contacts sharing the key "mom"."""
    assert (
        header_confirms_recipient(
            header, "Mom ❤️", _MOM_PHONE, unique_name=False
        )
        is expected
    )
