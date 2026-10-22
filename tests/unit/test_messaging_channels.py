"""Stage 8.6.4: the WhatsApp channel adapters.

The adapters are the channel-specific half of the messaging package: the
desktop app (text) and the WhatsApp Web profile (files).  The flow is tested
in ``test_messaging_flow.py``; here every adapter is driven directly against a
fake desktop window or a fake narrow page - no WhatsApp, no browser, no
network - and every refusal is pinned down.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from jarvis.browser.whatsapp_web import (
    WHATSAPP_BASE_URL,
    AttachmentCandidate,
    AttachmentScan,
    SendControl,
)
from jarvis.messaging.base import MessageError, MessageRequest
from jarvis.messaging.contacts import Contact
from jarvis.messaging.whatsapp_desktop import WhatsAppDesktopAdapter
from jarvis.messaging.whatsapp_web import (
    FileSendGuard,
    WhatsAppWebAdapter,
    content_digest,
)

_PHONE = "919876543210"
_CONTACT = Contact(name="Rahul", number=_PHONE)


@pytest.fixture(autouse=True)
def _windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test runs as if on Windows unless it overrides this."""
    monkeypatch.setattr("jarvis.platform_guard.is_windows", lambda: True)


# ── the desktop channel ───────────────────────────────────────────────────


class _Window:
    def __init__(
        self,
        *,
        header: str | None = "Rahul",
        draft: str | None = "hello",
        confirmed: bool | None = True,
    ) -> None:
        self.header = header
        self.draft = draft
        self.confirmed = confirmed
        self.entered = False

    def chat_header(self) -> str | None:
        return self.header

    def draft_text(self) -> str | None:
        return self.draft

    def press_enter(self) -> None:
        self.entered = True

    def message_visible(self, message: str) -> bool | None:
        del message
        return self.confirmed


class _Provider:
    def __init__(self, window: _Window | None) -> None:
        self.window = window
        self.opened: list[tuple[str, str]] = []

    def open_chat(self, phone_digits: str, text: str) -> None:
        self.opened.append((phone_digits, text))

    def find_chat_window(self, timeout_s: float) -> _Window | None:
        del timeout_s
        return self.window


def _desktop(provider: _Provider) -> WhatsAppDesktopAdapter:
    return WhatsAppDesktopAdapter(provider=provider)


def _text_request(text: str = "hello") -> MessageRequest:
    return MessageRequest(channel="whatsapp", contact="Rahul", text=text)


def test_the_desktop_channel_sends_and_reads_back() -> None:
    window = _Window()
    provider = _Provider(window)

    outcome = _desktop(provider).send(_text_request(), _CONTACT)

    assert outcome.sent is True
    assert outcome.channel == "whatsapp"
    assert outcome.data["chat_confirmed"] is True
    assert provider.opened == [(_PHONE, "hello")]
    assert window.entered is True


def test_the_desktop_channel_refuses_off_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("jarvis.platform_guard.is_windows", lambda: False)
    with pytest.raises(MessageError, match="only supported on Windows"):
        _desktop(_Provider(_Window())).send(_text_request(), _CONTACT)


def test_the_desktop_channel_refuses_an_unverifiable_window() -> None:
    with pytest.raises(MessageError, match="Could not verify chat"):
        _desktop(_Provider(None)).send(_text_request(), _CONTACT)


def test_the_desktop_channel_refuses_an_unreadable_header() -> None:
    window = _Window(header=None)
    with pytest.raises(MessageError, match="Chat identity could not be verified"):
        _desktop(_Provider(window)).send(_text_request(), _CONTACT)
    assert window.entered is False


def test_the_desktop_channel_refuses_a_foreign_header() -> None:
    window = _Window(header="Someone Else")
    with pytest.raises(MessageError, match="Chat identity could not be verified"):
        _desktop(_Provider(window)).send(_text_request(), _CONTACT)
    assert window.entered is False


def test_the_desktop_channel_refuses_a_draft_that_is_not_the_message() -> None:
    window = _Window(draft="something different")
    with pytest.raises(MessageError, match="Draft text does not match"):
        _desktop(_Provider(window)).send(_text_request(), _CONTACT)
    assert window.entered is False


def test_the_desktop_channel_refuses_an_empty_message() -> None:
    with pytest.raises(MessageError, match="no message text"):
        _desktop(_Provider(_Window())).send(_text_request("   "), _CONTACT)


def test_the_desktop_channel_only_carries_text() -> None:
    assert _desktop(_Provider(_Window())).capabilities() == frozenset({"text"})


# ── the WhatsApp Web channel ──────────────────────────────────────────────


def _scan(name: str) -> AttachmentScan:
    return AttachmentScan(
        panel_found=True,
        composer=(AttachmentCandidate(scope="composer", source="aria-label", text=name),),
        panel=(),
        composer_container_below_panel=True,
    )


class _Page:
    """Fake narrow WhatsApp page (records every action the driver took)."""

    def __init__(
        self,
        *,
        header: str = "Rahul",
        controls: tuple[SendControl, ...] = (SendControl(name="Send", index=0),),
        on_stage: Any = None,
    ) -> None:
        self.current_url = WHATSAPP_BASE_URL
        self.header = header
        self.controls = controls
        self.on_stage = on_stage
        self.visited: list[str] = []
        self.staged_paths: list[str] = []
        self.staged: tuple[str, ...] = ()
        self.clicked: list[str] = []

    def url(self) -> str:
        return self.current_url

    def goto(self, url: str) -> None:
        self.visited.append(url)
        self.current_url = url

    def chat_header(self) -> str:
        return self.header

    def fill_search(self, query: str) -> None:
        del query

    def search_results(self) -> tuple[Any, ...]:
        return ()

    def open_search_result(self, hit: Any) -> None:
        del hit

    def set_input_files(self, path: str) -> None:
        self.staged_paths.append(path)
        self.staged = (Path(path).name,)
        if self.on_stage is not None:
            self.on_stage()

    def staged_file_names(self) -> tuple[str, ...]:
        return self.staged

    def attachment_scan(self, filename: str) -> AttachmentScan:
        return _scan(filename)

    def send_controls(self) -> tuple[SendControl, ...]:
        return self.controls

    def click_send(self, name: str) -> None:
        self.clicked.append(name)


class _Session:
    def __init__(self, page: _Page | None) -> None:
        self._page = page

    def whatsapp_page(self) -> _Page | None:
        return self._page


class _Holder:
    """Stands in for ``ToolContext.whatsapp_browser`` (a LazyProfileBrowser)."""

    def __init__(self, page: _Page | None) -> None:
        self.page = page
        self.acquires = 0

    def acquire(self) -> _Session | None:
        self.acquires += 1
        return _Session(self.page) if self.page is not None else None


def _web(
    tmp_path: Path, holder: _Holder, *, guard: FileSendGuard | None = None
) -> WhatsAppWebAdapter:
    return WhatsAppWebAdapter(
        holder=holder,
        guard=guard or FileSendGuard(tmp_path / "guard.json"),
        max_bytes=1024 * 1024,
    )


def _file(tmp_path: Path, data: bytes = b"approved bytes") -> Path:
    path = tmp_path / "photo.jpg"
    path.write_bytes(data)
    return path


def _file_request(path: Path, digest: str) -> MessageRequest:
    return MessageRequest(channel="whatsapp_web", contact="Rahul", path=str(path), digest=digest)


def test_the_web_channel_sends_a_verified_file(tmp_path: Path) -> None:
    target = _file(tmp_path)
    digest = content_digest(target, 1024 * 1024)
    assert digest is not None
    page = _Page()
    holder = _Holder(page)

    outcome = _web(tmp_path, holder).send(_file_request(target, digest), _CONTACT)

    assert outcome.sent is True
    assert outcome.data["file"] == "photo.jpg"
    assert outcome.data["content_sha256"] == digest
    assert page.clicked == ["Send"]


def test_the_web_channel_refuses_an_unbound_file(tmp_path: Path) -> None:
    target = _file(tmp_path)
    holder = _Holder(_Page())

    with pytest.raises(MessageError, match="not bound to this approval"):
        _web(tmp_path, holder).send(_file_request(target, ""), _CONTACT)
    assert holder.acquires == 0


def test_the_web_channel_refuses_a_file_without_a_path(tmp_path: Path) -> None:
    holder = _Holder(_Page())
    request = MessageRequest(channel="whatsapp_web", contact="Rahul", digest="abc")
    with pytest.raises(MessageError, match="no file to send"):
        _web(tmp_path, holder).send(request, _CONTACT)


def test_the_web_channel_refuses_a_missing_browser(tmp_path: Path) -> None:
    target = _file(tmp_path)
    digest = content_digest(target, 1024 * 1024)
    assert digest is not None
    holder = _Holder(None)

    with pytest.raises(MessageError, match="browser session is not available"):
        _web(tmp_path, holder).send(_file_request(target, digest), _CONTACT)


def test_the_web_channel_refuses_a_foreign_chat(tmp_path: Path) -> None:
    target = _file(tmp_path)
    digest = content_digest(target, 1024 * 1024)
    assert digest is not None
    page = _Page(header="Someone Else")

    with pytest.raises(MessageError, match="could not open the WhatsApp chat"):
        _web(tmp_path, _Holder(page)).send(_file_request(target, digest), _CONTACT)
    assert page.clicked == []


def test_the_web_channel_refuses_a_file_that_changed_while_staging(tmp_path: Path) -> None:
    target = _file(tmp_path, data=b"original")
    digest = content_digest(target, 1024 * 1024)
    assert digest is not None
    page = _Page(on_stage=lambda: target.write_bytes(b"tampered"))

    with pytest.raises(MessageError, match="changed while the attachment was being prepared"):
        _web(tmp_path, _Holder(page)).send(_file_request(target, digest), _CONTACT)
    assert page.clicked == []


def test_the_web_channel_refuses_a_recent_duplicate(tmp_path: Path) -> None:
    target = _file(tmp_path)
    digest = content_digest(target, 1024 * 1024)
    assert digest is not None
    guard = FileSendGuard(tmp_path / "guard.json")
    adapter = _web(tmp_path, _Holder(_Page()), guard=guard)

    adapter.send(_file_request(target, digest), _CONTACT)
    with pytest.raises(MessageError, match="already sent"):
        adapter.send(_file_request(target, digest), _CONTACT)


def test_the_web_channel_only_carries_files(tmp_path: Path) -> None:
    assert _web(tmp_path, _Holder(_Page())).capabilities() == frozenset({"file"})


def test_the_web_channel_opens_a_chat_without_sending(tmp_path: Path) -> None:
    page = _Page()
    header = _web(tmp_path, _Holder(page)).open_chat(_CONTACT)

    assert header == "Rahul"
    assert page.staged_paths == [] and page.clicked == []


def test_the_content_digest_is_the_approved_sha256(tmp_path: Path) -> None:
    target = _file(tmp_path, data=b"approved bytes")
    assert content_digest(target, 1024 * 1024) == hashlib.sha256(b"approved bytes").hexdigest()
