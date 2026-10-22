"""Playwright adapter tests for the WhatsApp page surface (Stage 8, Slice 8.4b).

These tests drive the *installed* Chrome against a local HTML fixture that
imitates the WhatsApp chat panel.  There is no network, no WhatsApp account
and no real message: the fixture only records the click in ``window.__sent``.

They are marked ``windows_only`` because they need Chrome, and they skip
cleanly when Playwright or Chrome is unavailable, so ``pytest -q`` stays green
on a machine without a browser.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from jarvis.browser.whatsapp_page import PlaywrightWhatsAppPage
from jarvis.browser.whatsapp_web import WHATSAPP_BASE_URL, WhatsAppDriver, WhatsAppError

pytestmark = pytest.mark.windows_only

_PHONE = "919876543210"

#: A minimal imitation of the WhatsApp chat panel.
#:
#: ``preview_in_composer`` decides whether the staged file's name is visible in
#: the composer (the accepted scope) or only in the message list (not proof).
_PAGE = """
<!doctype html>
<html>
  <body>
    <div id="main">
      <header><span title="Mom">Mom</span></header>
      <div role="main" id="messages">
        <div class="msg">an older message</div>
        {extra}
      </div>
      <footer>
        <input type="file" id="attach" style="display: none;">
        <button id="attach-button" aria-label="Attach">+</button>
        {preview}
        <button id="send" aria-label="Send 1 selected">Send</button>
      </footer>
    </div>
    <script>
      window.__sent = false;
      document.getElementById('send')
        .addEventListener('click', function () {{ window.__sent = true; }});
    </script>
  </body>
</html>
"""

_PREVIEW_IN_COMPOSER = '<div class="preview" aria-label="photo.jpg">photo.jpg</div>'
_PREVIEW_IN_MESSAGES = '<div class="msg" aria-label="photo.jpg">sent earlier</div>'


def _write_fixture(tmp_path: Path, preview: str, extra: str = "") -> Path:
    """Write the local HTML fixture and return its path."""
    path = tmp_path / "whatsapp.html"
    path.write_text(_PAGE.format(preview=preview, extra=extra), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def playwright() -> Any:
    """A started Playwright driver (skipped when Playwright is missing)."""
    module = pytest.importorskip("playwright.sync_api")
    try:
        driver = module.sync_playwright().start()
    except Exception as exc:  # noqa: BLE001 - a missing browser is a skip
        pytest.skip(f"Playwright could not start: {exc}")
    try:
        yield driver
    finally:
        driver.stop()


@pytest.fixture(scope="module")
def chrome_page(playwright: Any) -> Any:
    """A page in the installed Chrome (skipped when Chrome is missing)."""
    try:
        browser = playwright.chromium.launch(channel="chrome", headless=True)
    except Exception as exc:  # noqa: BLE001 - a missing browser is a skip
        pytest.skip(f"Chrome could not be launched: {exc}")
    page = browser.new_page()
    try:
        yield page
    finally:
        browser.close()


class _LocalWhatsAppPage(PlaywrightWhatsAppPage):
    """The adapter pointed at a local fixture instead of the real WhatsApp."""

    def __init__(self, page: Any, fixture: Path) -> None:
        """Remember the fixture the adapter must load."""
        super().__init__(page)
        self._fixture = fixture

    def goto(self, url: str) -> None:
        """Load the local fixture (never the network)."""
        del url
        self._page.goto(self._fixture.as_uri(), wait_until="domcontentloaded")

    def url(self) -> str:
        """Pretend the page is on WhatsApp (the driver checks the origin)."""
        return WHATSAPP_BASE_URL


def _open_fixture(
    chrome_page: Any, tmp_path: Path, preview: str, extra: str = ""
) -> _LocalWhatsAppPage:
    """Load the fixture and wrap it in the adapter."""
    fixture = _write_fixture(tmp_path, preview, extra)
    chrome_page.goto(fixture.as_uri(), wait_until="domcontentloaded")
    return _LocalWhatsAppPage(chrome_page, fixture)


# ── reading ───────────────────────────────────────────────────────────────


def test_chat_header_reads_the_contact_name(chrome_page: Any, tmp_path: Path) -> None:
    adapter = _open_fixture(chrome_page, tmp_path, _PREVIEW_IN_COMPOSER)

    assert adapter.chat_header() == "Mom"


def test_attachment_scan_finds_the_name_in_the_composer(chrome_page: Any, tmp_path: Path) -> None:
    adapter = _open_fixture(chrome_page, tmp_path, _PREVIEW_IN_COMPOSER)

    scan = adapter.attachment_scan("photo.jpg")

    assert scan.panel_found is True
    assert scan.composer_scoped is True
    assert [candidate.source for candidate in scan.composer] == ["aria-label"]
    assert scan.panel == ()


def test_attachment_scan_reports_a_name_outside_the_composer(
    chrome_page: Any, tmp_path: Path
) -> None:
    adapter = _open_fixture(chrome_page, tmp_path, "", _PREVIEW_IN_MESSAGES)

    scan = adapter.attachment_scan("photo.jpg")

    assert scan.panel_found is True
    assert scan.composer == ()
    assert scan.composer_scoped is False
    assert len(scan.panel) == 1


def test_attachment_scan_reports_no_panel(chrome_page: Any, tmp_path: Path) -> None:
    fixture = tmp_path / "empty.html"
    fixture.write_text("<html><body><p>nothing</p></body></html>", encoding="utf-8")
    chrome_page.goto(fixture.as_uri(), wait_until="domcontentloaded")
    adapter = _LocalWhatsAppPage(chrome_page, fixture)

    scan = adapter.attachment_scan("photo.jpg")

    assert scan.panel_found is False


def test_send_controls_are_enumerated_and_named(chrome_page: Any, tmp_path: Path) -> None:
    """Controls are found by their accessible name, not by a guessed selector."""
    adapter = _open_fixture(chrome_page, tmp_path, _PREVIEW_IN_COMPOSER)

    controls = adapter.send_controls()

    assert [control.name for control in controls] == ["Attach", "Send 1 selected"]


def test_set_input_files_stages_the_file(chrome_page: Any, tmp_path: Path) -> None:
    adapter = _open_fixture(chrome_page, tmp_path, _PREVIEW_IN_COMPOSER)
    target = tmp_path / "photo.jpg"
    target.write_bytes(b"jpeg")

    adapter.set_input_files(str(target))

    assert adapter.staged_file_names() == ("photo.jpg",)


# ── the whole flow, offline ───────────────────────────────────────────────


def test_driver_sends_only_after_every_gate_passes(chrome_page: Any, tmp_path: Path) -> None:
    adapter = _open_fixture(chrome_page, tmp_path, _PREVIEW_IN_COMPOSER)
    driver = WhatsAppDriver(adapter)
    target = tmp_path / "photo.jpg"
    target.write_bytes(b"jpeg")

    driver.open_chat_by_phone(_PHONE, "Mom")
    driver.attach(target)
    outcome = driver.send(target.name)

    assert outcome.staged_count == 1
    assert chrome_page.evaluate("() => window.__sent") is True


def test_driver_refuses_when_the_preview_is_outside_the_composer(
    chrome_page: Any, tmp_path: Path
) -> None:
    adapter = _open_fixture(chrome_page, tmp_path, "", _PREVIEW_IN_MESSAGES)
    driver = WhatsAppDriver(adapter)
    target = tmp_path / "photo.jpg"
    target.write_bytes(b"jpeg")

    driver.open_chat_by_phone(_PHONE, "Mom")
    with pytest.raises(WhatsAppError, match="not visible in the chat composer"):
        driver.attach(target)

    assert chrome_page.evaluate("() => window.__sent") is False
