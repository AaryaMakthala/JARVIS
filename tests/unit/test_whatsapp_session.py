"""W0: explicit WhatsApp Web session state and bounded readiness.

No browser, no Playwright, no network: the real driver runs against a fake
narrow page, so every state and every refusal is deterministic.

What these tests pin down:

* the four documented states (``authenticated`` / ``qr_required`` /
  ``logged_out`` / ``unknown``), and that a probe failure is ``unknown``;
* every unauthenticated state refuses ``open_chat_by_phone``,
  ``open_chat_by_name``, ``attach`` and ``send`` *before* anything is searched,
  staged or clicked;
* the session probe never runs on a foreign origin;
* a readiness timeout refuses without reading chat identity;
* refusals never log a phone number or a message body.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from jarvis.browser.whatsapp_web import (
    WHATSAPP_BASE_URL,
    WHATSAPP_SEND_URL,
    AttachmentCandidate,
    AttachmentScan,
    SearchHit,
    SendControl,
    WhatsAppDriver,
    WhatsAppError,
    WhatsAppLoggedOut,
    WhatsAppNotReady,
    WhatsAppQRRequired,
    WhatsAppSessionError,
    WhatsAppSessionState,
    WhatsAppSessionUnknown,
)

_PHONE = "919876543210"

_UNAUTHENTICATED = (
    WhatsAppSessionState.QR_REQUIRED,
    WhatsAppSessionState.LOGGED_OUT,
    WhatsAppSessionState.UNKNOWN,
)


def _scan(name: str = "photo.jpg") -> AttachmentScan:
    """A scan whose file-name evidence sits inside the composer."""
    return AttachmentScan(
        panel_found=True,
        composer=(AttachmentCandidate(scope="composer", source="aria-label", text=name),),
        panel=(),
        composer_container_below_panel=True,
    )


class _FakePage:
    """A fake narrow WhatsApp page with an explicit, settable session state."""

    def __init__(
        self,
        *,
        state: WhatsAppSessionState | str = WhatsAppSessionState.AUTHENTICATED,
        url: str = WHATSAPP_BASE_URL,
        header: str = "Mom",
        hits: tuple[SearchHit, ...] = (SearchHit(name="Mom"),),
        staged: tuple[str, ...] = ("photo.jpg",),
        scan: AttachmentScan | None = None,
        controls: tuple[SendControl, ...] = (SendControl(name="Send 1 selected"),),
        probe_error: Exception | None = None,
        ready_error: Exception | None = None,
        redirect_to: str = "",
    ) -> None:
        self.state = state
        self.current_url = url
        self.header = header
        self.hits = hits
        self.staged = staged
        self.scan = scan if scan is not None else _scan()
        self.controls = controls
        self.probe_error = probe_error
        self.ready_error = ready_error
        self.redirect_to = redirect_to
        self.header_reads = 0
        self.probes = 0
        self.waits = 0
        self.visited: list[str] = []
        self.searched: list[str] = []
        self.opened: list[SearchHit] = []
        self.staged_paths: list[str] = []
        self.clicked: list[str] = []

    # ── WhatsAppPage protocol ──────────────────────────────────────────────

    def url(self) -> str:
        return self.current_url

    def goto(self, url: str) -> None:
        self.visited.append(url)
        self.current_url = self.redirect_to or url

    def session_state(self) -> WhatsAppSessionState:
        self.probes += 1
        if self.probe_error is not None:
            raise self.probe_error
        return self.state  # type: ignore[return-value]

    def wait_for_ready(self, timeout_s: float = 20.0) -> WhatsAppSessionState:
        del timeout_s
        self.waits += 1
        if self.ready_error is not None:
            raise self.ready_error
        return self.session_state()

    def chat_header(self) -> str:
        self.header_reads += 1
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


def _opened(page: _FakePage) -> WhatsAppDriver:
    """A driver whose authenticated chat is already open."""
    driver = WhatsAppDriver(page)
    driver.open_chat_by_phone(_PHONE, "Mom")
    return driver


def _file(tmp_path: Path) -> Path:
    path = tmp_path / "photo.jpg"
    path.write_bytes(b"jpeg")
    return path


# ── the state contract ─────────────────────────────────────────────────────


def test_the_four_states_are_the_documented_values() -> None:
    assert {state.value for state in WhatsAppSessionState} == {
        "authenticated",
        "qr_required",
        "logged_out",
        "unknown",
    }


def test_session_errors_remain_whatsapp_errors() -> None:
    assert issubclass(WhatsAppSessionError, WhatsAppError)
    assert issubclass(WhatsAppQRRequired, WhatsAppSessionError)
    assert issubclass(WhatsAppLoggedOut, WhatsAppSessionError)
    assert issubclass(WhatsAppSessionUnknown, WhatsAppSessionError)
    assert issubclass(WhatsAppNotReady, WhatsAppSessionError)


def test_authenticated_require_returns_the_state() -> None:
    page = _FakePage()
    assert WhatsAppDriver(page).require_authenticated() is WhatsAppSessionState.AUTHENTICATED
    assert page.probes == 1


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (WhatsAppSessionState.QR_REQUIRED, WhatsAppQRRequired),
        (WhatsAppSessionState.LOGGED_OUT, WhatsAppLoggedOut),
        (WhatsAppSessionState.UNKNOWN, WhatsAppSessionUnknown),
    ],
)
def test_require_authenticated_raises_the_specific_state(
    state: WhatsAppSessionState, expected: type[WhatsAppSessionError]
) -> None:
    page = _FakePage(state=state)
    with pytest.raises(expected) as caught:
        WhatsAppDriver(page).require_authenticated()
    assert caught.value.state is state


def test_a_probe_exception_is_unknown_never_a_pass() -> None:
    page = _FakePage(probe_error=RuntimeError("probe blew up"))
    with pytest.raises(WhatsAppSessionUnknown) as caught:
        WhatsAppDriver(page).require_authenticated()
    assert caught.value.state is WhatsAppSessionState.UNKNOWN


def test_an_unrecognized_probe_value_is_unknown() -> None:
    page = _FakePage(state="something-new")
    with pytest.raises(WhatsAppSessionUnknown) as caught:
        WhatsAppDriver(page).require_authenticated()
    assert caught.value.state is WhatsAppSessionState.UNKNOWN


def test_the_probe_never_runs_on_a_foreign_origin() -> None:
    page = _FakePage(url="https://evil.example/send")
    with pytest.raises(WhatsAppError, match="not on"):
        WhatsAppDriver(page).require_authenticated()
    assert page.probes == 0


# ── refusals before any side effect ────────────────────────────────────────


@pytest.mark.parametrize("state", _UNAUTHENTICATED)
def test_open_chat_by_phone_refuses_and_reads_no_identity(state: WhatsAppSessionState) -> None:
    page = _FakePage(state=state)
    with pytest.raises(WhatsAppSessionError) as caught:
        WhatsAppDriver(page).open_chat_by_phone(_PHONE, "Mom")
    assert caught.value.state is state
    assert page.header_reads == 0
    assert page.staged_paths == [] and page.clicked == []


@pytest.mark.parametrize("state", _UNAUTHENTICATED)
def test_open_chat_by_name_refuses_before_searching(state: WhatsAppSessionState) -> None:
    page = _FakePage(state=state)
    with pytest.raises(WhatsAppSessionError) as caught:
        WhatsAppDriver(page).open_chat_by_name("Mom")
    assert caught.value.state is state
    assert page.searched == [] and page.opened == []
    assert page.header_reads == 0


@pytest.mark.parametrize("state", _UNAUTHENTICATED)
def test_attach_refuses_before_staging(
    state: WhatsAppSessionState, tmp_path: Path
) -> None:
    page = _FakePage()
    driver = _opened(page)
    page.state = state
    with pytest.raises(WhatsAppSessionError) as caught:
        driver.attach(_file(tmp_path))
    assert caught.value.state is state
    assert page.staged_paths == []
    assert page.clicked == []


@pytest.mark.parametrize("state", _UNAUTHENTICATED)
def test_send_refuses_before_clicking(state: WhatsAppSessionState) -> None:
    page = _FakePage()
    driver = _opened(page)
    page.state = state
    with pytest.raises(WhatsAppSessionError) as caught:
        driver.send("photo.jpg")
    assert caught.value.state is state
    assert page.clicked == []


# ── readiness ──────────────────────────────────────────────────────────────


def test_a_readiness_timeout_refuses_before_reading_identity() -> None:
    page = _FakePage(
        ready_error=WhatsAppNotReady(WhatsAppSessionState.UNKNOWN, "readiness timed out")
    )
    with pytest.raises(WhatsAppNotReady, match="readiness timed out"):
        WhatsAppDriver(page).open_chat_by_phone(_PHONE, "Mom")
    assert page.waits == 1
    assert page.header_reads == 0
    assert page.staged_paths == [] and page.clicked == []


def test_a_page_error_during_readiness_is_a_readiness_refusal() -> None:
    page = _FakePage(ready_error=RuntimeError("the page crashed"))
    with pytest.raises(WhatsAppNotReady):
        WhatsAppDriver(page).open_chat_by_phone(_PHONE, "Mom")
    assert page.header_reads == 0


def test_readiness_is_waited_for_before_identity_on_both_navigation_paths() -> None:
    phone_page = _FakePage()
    WhatsAppDriver(phone_page).open_chat_by_phone(_PHONE, "Mom")
    assert phone_page.waits == 1

    name_page = _FakePage()
    WhatsAppDriver(name_page).open_chat_by_name("Mom")
    assert name_page.waits == 1


# ── authenticated behaviour is unchanged ───────────────────────────────────


def test_authenticated_phone_open_still_uses_the_deep_link_and_header() -> None:
    page = _FakePage(header="Mom")
    header = WhatsAppDriver(page).open_chat_by_phone(_PHONE, "Mom")
    assert header == "Mom"
    assert page.visited == [WHATSAPP_SEND_URL.format(digits=_PHONE)]
    assert page.header_reads == 1


def test_authenticated_name_open_still_uses_the_unique_search_hit() -> None:
    page = _FakePage(hits=(SearchHit(name="Mom"),))
    assert WhatsAppDriver(page).open_chat_by_name("Mom") == "Mom"
    assert page.searched == ["Mom"]
    assert len(page.opened) == 1


def test_authenticated_attach_and_send_still_complete(tmp_path: Path) -> None:
    page = _FakePage()
    driver = _opened(page)
    target = _file(tmp_path)

    driver.attach(target)
    outcome = driver.send("photo.jpg")

    assert page.staged_paths == [str(target)]
    assert page.clicked == ["Send 1 selected"]
    assert outcome.control == "Send 1 selected"
    assert outcome.staged_count == 1


# ── secrets ────────────────────────────────────────────────────────────────


def test_refusals_never_log_a_phone_number_or_message_body(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="jarvis.browser.whatsapp")
    page = _FakePage(state=WhatsAppSessionState.QR_REQUIRED)

    with pytest.raises(WhatsAppQRRequired):
        WhatsAppDriver(page).open_chat_by_phone(_PHONE, "Mom")

    assert _PHONE not in caplog.text
    assert "+91" not in caplog.text
