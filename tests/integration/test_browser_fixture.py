"""Opt-in live browser tests against the offline fixture site.

The real managed wrapper (:class:`jarvis.browser.context.PlaywrightBrowser`)
launches a visible Chrome window and is Windows-only.  ``fixture_server``
serves ``tests/fixtures/site`` from 127.0.0.1 on a free port; nothing here
reaches the internet.

    set JARVIS_BROWSER_TESTS=1
    .venv\\Scripts\\python.exe -m pytest -q tests/integration/test_browser_fixture.py -m browser
"""

from __future__ import annotations

import os
import time
from urllib.parse import urlsplit

import pytest

pytestmark = [
    pytest.mark.browser,
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("JARVIS_BROWSER_TESTS") != "1",
        reason=(
            "set JARVIS_BROWSER_TESTS=1 to run the offline browser fixture "
            "(launches a visible Chrome window)"
        ),
    ),
    pytest.mark.skipif(os.name != "nt", reason="the managed Chrome browser is Windows-only"),
]


def _element(elements, *, role: str, name: str):
    """Return the observed element with this role/name or fail with context."""
    for element in elements:
        if element.role == role and element.name == name:
            return element
    seen = [(element.role, element.name) for element in elements]
    raise AssertionError(f"no {role} named {name!r}; observed {seen}")


def _wait_for_path(browser, path: str, *, timeout: float = 10.0) -> None:
    """Wait until the browser is on ``path`` and its DOM is ready."""
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        last = browser.current_url()
        if urlsplit(last).path == path:
            browser.wait(seconds=5, load_state="domcontentloaded")
            return
        time.sleep(0.1)
    raise AssertionError(f"page did not navigate to {path!r}; current URL was {last!r}")


@pytest.fixture(scope="module")
def browser():
    """One managed browser for the module; missing deps are a skip."""
    try:
        from jarvis.browser.context import BrowserError, PlaywrightBrowser
    except Exception as exc:  # noqa: BLE001 - a missing optional dep is a skip
        pytest.skip(f"browser deps missing ({type(exc).__name__}: {exc})")

    from jarvis.config import BrowserSettings, Settings

    settings = Settings(browser=BrowserSettings(allowed_domains=["127.0.0.1"]))
    session = PlaywrightBrowser(settings, profile="general")
    try:
        session.start()
    except BrowserError as exc:
        pytest.skip(f"browser deps missing or browser unavailable: {exc}")
    try:
        yield session
    finally:
        session.close()


def test_browser_goto_fixture(browser, fixture_server):
    result = browser.goto(f"{fixture_server}/")
    assert result["title"] == "Fixture Home"
    assert urlsplit(result["url"]).hostname == "127.0.0.1"

    observation = browser.read()
    assert observation.title == "Fixture Home"
    assert urlsplit(observation.url).hostname == "127.0.0.1"
    assert _element(observation.elements, role="link", name="Shop")
    assert _element(observation.elements, role="link", name="Search")


def test_browser_read_fixture(browser, fixture_server):
    browser.goto(f"{fixture_server}/search")
    observation = browser.read()
    search = _element(observation.elements, role="textbox", name="Search")

    browser.type_text(search.ref, "jarvis")
    browser.press_key("Enter")
    _wait_for_path(browser, "/results")

    observation = browser.read()
    assert observation.title == "Results"
    assert urlsplit(observation.url).hostname == "127.0.0.1"
    assert _element(observation.elements, role="link", name="Result One")


def test_browser_find_open_result_fixture(browser, fixture_server):
    browser.goto(f"{fixture_server}/results")
    observation = browser.read()
    result = _element(observation.elements, role="link", name="Result One")
    external = _element(observation.elements, role="link", name="External target")
    assert urlsplit(result.url).hostname == "127.0.0.1"
    assert urlsplit(external.url).hostname == "example.invalid"

    browser.click(result.ref)
    _wait_for_path(browser, "/result1")

    final = browser.read()
    assert final.title == "Result One"
    assert urlsplit(final.url).hostname == "127.0.0.1"
    assert final.url != external.url
