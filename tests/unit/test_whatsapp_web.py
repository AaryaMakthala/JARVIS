"""WhatsApp Web driver tests (Stage 8, Slice 8.4b).

No browser, no Playwright, no network: the driver is exercised against a fake
narrow page, which lets every refusal path be tested deterministically.

The tests assert the driver's own policy:

* the deep link and the search path both verify the chat identity;
* a redirect away from ``https://web.whatsapp.com`` stops the task;
* the staged attachment must be visible **in the composer** - a match
  elsewhere in the chat panel is not proof and is refused;
* exactly one Send control may match and it must say that one item is staged;
* nothing is ever clicked on a refusal path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.browser.whatsapp_web import (
    WHATSAPP_BASE_URL,
    WHATSAPP_ORIGIN,
    WHATSAPP_SEND_URL,
    AttachmentCandidate,
    AttachmentScan,
    SearchHit,
    SendControl,
    WhatsAppDriver,
    WhatsAppError,
    header_matches,
    is_send_name,
    send_count,
)

_PHONE = "919876543210"


class _FakePage:
    """A fake narrow WhatsApp page (records what the driver did)."""

    def __init__(
        self,
        *,
        url: str = WHATSAPP_BASE_URL,
        header: str = "Mom",
        hits: tuple[SearchHit, ...] = (),
        staged: tuple[str, ...] = (),
        scan: AttachmentScan | None = None,
        controls: tuple[SendControl, ...] = (),
        redirect_to: str = "",
    ) -> None:
        self.current_url = url
        self.header = header
        self.hits = hits
        self.staged = staged
        self.scan = scan if scan is not None else _composer_scan()
        self.controls = controls
        self.redirect_to = redirect_to
        self.visited: list[str] = []
        self.searched: list[str] = []
        self.opened: list[SearchHit] = []
        self.staged_paths: list[str] = []
        self.clicked: list[str] = []

    # ── WhatsAppPage protocol ───────────────────────────────────────────

    def url(self) -> str:
        return self.current_url

    def goto(self, url: str) -> None:
        self.visited.append(url)
        self.current_url = self.redirect_to or url

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
        return self.staged

    def attachment_scan(self, filename: str) -> AttachmentScan:
        del filename
        return self.scan

    def send_controls(self) -> tuple[SendControl, ...]:
        return self.controls

    def click_send(self, name: str) -> None:
        self.clicked.append(name)


def _composer_scan(name: str = "photo.jpg") -> AttachmentScan:
    """A scan whose file-name evidence sits inside the composer."""
    return AttachmentScan(
        panel_found=True,
        composer=(AttachmentCandidate(scope="composer", source="aria-label", text=name),),
        panel=(),
        composer_container_below_panel=True,
    )


def _open(page: _FakePage) -> WhatsAppDriver:
    """A driver that has already opened (and verified) the test chat."""
    driver = WhatsAppDriver(page)
    driver.open_chat_by_phone(_PHONE, "Mom")
    return driver


# ── name helpers ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Send", True),
        ("send", True),
        ("Send 1 selected", True),
        ("Send 2 selected", True),
        ("Send message", False),
        ("Send photo", False),
        ("Cancel", False),
        ("", False),
    ],
)
def test_is_send_name(name: str, expected: bool) -> None:
    assert is_send_name(name) is expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Send", None),
        ("Send 1 selected", 1),
        ("Send 3 selected", 3),
        ("Send message", None),
    ],
)
def test_send_count(name: str, expected: int | None) -> None:
    assert send_count(name) == expected


@pytest.mark.parametrize(
    ("header", "name", "digits", "expected"),
    [
        ("Mom", "Mom", "", True),
        ("mom", "Mom", "", True),
        ("Mom +91 98765 43210", "Mom", _PHONE, True),
        ("+91 98765 43210", "", _PHONE, True),
        ("Mum", "Mom", "", False),
        ("", "Mom", _PHONE, False),
        ("Mom", "", "", False),
    ],
)
def test_header_matches(header: str, name: str, digits: str, expected: bool) -> None:
    assert header_matches(header, name, digits) is expected


# ── opening a chat ────────────────────────────────────────────────────────


def test_open_chat_by_phone_uses_the_deep_link_and_verifies_the_header() -> None:
    page = _FakePage(header="Mom")
    driver = WhatsAppDriver(page)

    header = driver.open_chat_by_phone(_PHONE, "Mom")

    assert header == "Mom"
    assert page.visited == [WHATSAPP_SEND_URL.format(digits=_PHONE)]


def test_open_chat_by_phone_refuses_a_foreign_origin() -> None:
    page = _FakePage(redirect_to="https://example.invalid/send?phone=1")
    driver = WhatsAppDriver(page)

    with pytest.raises(WhatsAppError, match="not on https://web.whatsapp.com"):
        driver.open_chat_by_phone(_PHONE, "Mom")

    assert WHATSAPP_ORIGIN == "https://web.whatsapp.com"


def test_open_chat_by_phone_refuses_a_mismatched_header() -> None:
    page = _FakePage(header="Someone else")
    driver = WhatsAppDriver(page)

    with pytest.raises(WhatsAppError, match="does not match"):
        driver.open_chat_by_phone(_PHONE, "Mom")


def test_open_chat_by_phone_refuses_an_unreadable_header() -> None:
    page = _FakePage(header="")
    driver = WhatsAppDriver(page)

    with pytest.raises(WhatsAppError, match="could not be read"):
        driver.open_chat_by_phone(_PHONE, "Mom")


@pytest.mark.parametrize("digits", ["", "123", "9" * 16])
def test_open_chat_by_phone_refuses_a_bad_number(digits: str) -> None:
    page = _FakePage()
    driver = WhatsAppDriver(page)

    with pytest.raises(WhatsAppError, match="8-15 digits"):
        driver.open_chat_by_phone(digits, "Mom")

    assert page.visited == []


def test_open_chat_by_name_opens_the_unique_exact_hit() -> None:
    page = _FakePage(hits=(SearchHit(name="Mom", detail="98765"), SearchHit(name="Dad")))
    driver = WhatsAppDriver(page)

    driver.open_chat_by_name("Mom")

    assert page.visited == [WHATSAPP_BASE_URL]
    assert page.searched == ["Mom"]
    assert [hit.name for hit in page.opened] == ["Mom"]


def test_open_chat_by_name_refuses_two_exact_matches() -> None:
    page = _FakePage(hits=(SearchHit(name="Mom"), SearchHit(name="Mom")))
    driver = WhatsAppDriver(page)

    with pytest.raises(WhatsAppError, match="more than one exact match"):
        driver.open_chat_by_name("Mom")

    assert page.opened == []


def test_open_chat_by_name_refuses_an_ambiguous_search() -> None:
    page = _FakePage(hits=(SearchHit(name="Mom's sister"), SearchHit(name="Mom's brother")))
    driver = WhatsAppDriver(page)

    with pytest.raises(WhatsAppError, match="not unique"):
        driver.open_chat_by_name("Mom")

    assert page.opened == []


def test_open_chat_by_name_refuses_an_empty_search() -> None:
    page = _FakePage(hits=())
    driver = WhatsAppDriver(page)

    with pytest.raises(WhatsAppError, match="no contact"):
        driver.open_chat_by_name("Mom")


# ── attaching ─────────────────────────────────────────────────────────────


def test_attach_stages_the_file_and_returns_the_scan(tmp_path: Path) -> None:
    target = tmp_path / "photo.jpg"
    target.write_bytes(b"jpeg")
    page = _FakePage(staged=("photo.jpg",))
    driver = _open(page)

    scan = driver.attach(target)

    assert page.staged_paths == [str(target)]
    assert scan.composer_scoped is True


def test_attach_refuses_a_missing_file(tmp_path: Path) -> None:
    page = _FakePage()
    driver = _open(page)

    with pytest.raises(WhatsAppError, match="does not exist"):
        driver.attach(tmp_path / "nope.jpg")

    assert page.staged_paths == []


def test_attach_refuses_when_the_name_is_only_elsewhere_in_the_panel(tmp_path: Path) -> None:
    """A match outside the composer is not proof of what is staged."""
    target = tmp_path / "photo.jpg"
    target.write_bytes(b"jpeg")
    page = _FakePage(
        scan=AttachmentScan(
            panel_found=True,
            composer=(),
            panel=(AttachmentCandidate(scope="panel", source="aria-label", text="photo.jpg"),),
            panel_container_below_panel=True,
        ),
        staged=("photo.jpg",),
    )
    driver = _open(page)

    with pytest.raises(WhatsAppError, match="not visible in the chat composer"):
        driver.attach(target)


def test_attach_refuses_when_the_composer_evidence_is_not_scoped(tmp_path: Path) -> None:
    target = tmp_path / "photo.jpg"
    target.write_bytes(b"jpeg")
    page = _FakePage(
        scan=AttachmentScan(
            panel_found=True,
            composer=(AttachmentCandidate(scope="composer", source="text", text="photo.jpg"),),
            composer_container_below_panel=False,
        ),
        staged=("photo.jpg",),
    )
    driver = _open(page)

    with pytest.raises(WhatsAppError, match="could not be scoped"):
        driver.attach(target)


def test_attach_refuses_when_the_panel_is_missing(tmp_path: Path) -> None:
    target = tmp_path / "photo.jpg"
    target.write_bytes(b"jpeg")
    page = _FakePage(scan=AttachmentScan(panel_found=False))
    driver = _open(page)

    with pytest.raises(WhatsAppError, match="chat panel could not be located"):
        driver.attach(target)


def test_attach_refuses_a_contradicting_staged_file(tmp_path: Path) -> None:
    target = tmp_path / "photo.jpg"
    target.write_bytes(b"jpeg")
    page = _FakePage(staged=("other.jpg",))
    driver = _open(page)

    with pytest.raises(WhatsAppError, match="does not match the expected attachment"):
        driver.attach(target)


def test_attach_refuses_when_more_than_one_file_is_staged(tmp_path: Path) -> None:
    target = tmp_path / "photo.jpg"
    target.write_bytes(b"jpeg")
    page = _FakePage(staged=("photo.jpg", "leftover.jpg"))
    driver = _open(page)

    with pytest.raises(WhatsAppError, match="more than one file is staged"):
        driver.attach(target)


def test_attach_refuses_a_preview_container_that_is_too_large(tmp_path: Path) -> None:
    """A name inside a huge region is not an isolated attachment preview."""
    target = tmp_path / "photo.jpg"
    target.write_bytes(b"jpeg")
    page = _FakePage(
        scan=AttachmentScan(
            panel_found=True,
            composer=(
                AttachmentCandidate(scope="composer", source="aria-label", text="photo.jpg"),
            ),
            composer_container_below_panel=True,
            composer_container_text_len=5000,
        ),
        staged=("photo.jpg",),
    )
    driver = _open(page)

    with pytest.raises(WhatsAppError, match="could not be isolated"):
        driver.attach(target)


def test_attach_refuses_when_the_chat_identity_is_wrong(tmp_path: Path) -> None:
    target = tmp_path / "photo.jpg"
    target.write_bytes(b"jpeg")
    page = _FakePage(header="Mom", staged=("photo.jpg",))
    driver = WhatsAppDriver(page)
    driver.open_chat_by_phone(_PHONE, "Mom")
    page.header = "Someone else"

    with pytest.raises(WhatsAppError, match="does not match"):
        driver.attach(target)


# ── sending ───────────────────────────────────────────────────────────────


def _ready_driver(tmp_path: Path, **kwargs: object) -> tuple[_FakePage, WhatsAppDriver, Path]:
    target = tmp_path / "photo.jpg"
    target.write_bytes(b"jpeg")
    page = _FakePage(staged=("photo.jpg",), **kwargs)  # type: ignore[arg-type]
    driver = WhatsAppDriver(page)
    driver.open_chat_by_phone(_PHONE, "Mom")
    return page, driver, target


def test_send_clicks_the_verified_control(tmp_path: Path) -> None:
    page, driver, target = _ready_driver(tmp_path, controls=(SendControl(name="Send 1 selected"),))

    outcome = driver.send(target.name)

    assert page.clicked == ["Send 1 selected"]
    assert outcome.staged_count == 1
    assert outcome.header == "Mom"


def test_send_refuses_when_no_control_matches(tmp_path: Path) -> None:
    page, driver, target = _ready_driver(tmp_path, controls=(SendControl(name="Cancel"),))

    with pytest.raises(WhatsAppError, match="no WhatsApp Send control"):
        driver.send(target.name)

    assert page.clicked == []


def test_send_refuses_two_matching_controls(tmp_path: Path) -> None:
    page, driver, target = _ready_driver(
        tmp_path,
        controls=(SendControl(name="Send 1 selected"), SendControl(name="Send")),
    )

    with pytest.raises(WhatsAppError, match="more than one WhatsApp Send control"):
        driver.send(target.name)

    assert page.clicked == []


def test_send_refuses_when_more_than_one_item_is_staged(tmp_path: Path) -> None:
    page, driver, target = _ready_driver(tmp_path, controls=(SendControl(name="Send 2 selected"),))

    with pytest.raises(WhatsAppError, match="2 items are staged"):
        driver.send(target.name)

    assert page.clicked == []


def test_send_with_a_bare_send_needs_staged_file_evidence(tmp_path: Path) -> None:
    page = _FakePage(staged=(), controls=(SendControl(name="Send"),))
    driver = WhatsAppDriver(page)
    driver.open_chat_by_phone(_PHONE, "Mom")

    with pytest.raises(WhatsAppError, match="does not say what it sends"):
        driver.send("photo.jpg")

    assert page.clicked == []


def test_send_with_a_bare_send_uses_the_staged_file_evidence(tmp_path: Path) -> None:
    page = _FakePage(staged=("photo.jpg",), controls=(SendControl(name="Send"),))
    driver = WhatsAppDriver(page)
    driver.open_chat_by_phone(_PHONE, "Mom")

    outcome = driver.send("photo.jpg")

    assert page.clicked == ["Send"]
    assert outcome.staged_count is None


def test_send_refuses_when_the_preview_is_gone(tmp_path: Path) -> None:
    page = _FakePage(
        staged=("photo.jpg",),
        controls=(SendControl(name="Send 1 selected"),),
        scan=AttachmentScan(panel_found=True),
    )
    driver = WhatsAppDriver(page)
    driver.open_chat_by_phone(_PHONE, "Mom")

    with pytest.raises(WhatsAppError, match="no longer visible"):
        driver.send("photo.jpg")

    assert page.clicked == []


def test_send_refuses_when_the_chat_changed(tmp_path: Path) -> None:
    page, driver, target = _ready_driver(tmp_path, controls=(SendControl(name="Send 1 selected"),))
    page.header = "Someone else"

    with pytest.raises(WhatsAppError, match="does not match"):
        driver.send(target.name)

    assert page.clicked == []


def test_send_requires_an_open_chat(tmp_path: Path) -> None:
    page = _FakePage(header="", controls=(SendControl(name="Send 1 selected"),))
    driver = WhatsAppDriver(page)

    with pytest.raises(WhatsAppError, match="could not be read"):
        driver.send("photo.jpg")

    assert page.clicked == []
