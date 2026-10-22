"""Managed-browser driver tests (Stage 7, Slice 6).

Playwright is never imported and Chrome is never launched: every test injects a
fake Playwright/``sync_api`` object, a fake persistent context and fake pages/
locators.  The tests assert the driver's own logic (lazy start, launch
arguments, per-profile serialization, redirect fail-closed, bounded reads,
transient references, identity re-validation and credential refusal).
"""

from __future__ import annotations

import threading

import pytest

from jarvis.browser import context as context_mod
from jarvis.browser.context import BrowserError, PlaywrightBrowser, profile_lock
from jarvis.config import Settings, browser_profile_dir

_TAG_ROLES = {
    "button": "button",
    "a": "link",
    "input": "textbox",
    "textarea": "textbox",
    "select": "combobox",
}


class _FakeElement:
    def __init__(
        self,
        tag: str,
        name: str = "",
        *,
        role: str | None = None,
        input_type: str = "",
        visible: bool = True,
        aria_label: str | None = None,
    ) -> None:
        self.tag = tag
        self.name = name
        self.role = role
        self.input_type = input_type
        self.visible = visible
        self.aria_label = aria_label
        self.clicks = 0
        self.filled: str | None = None

    def live_role(self) -> str:
        if self.role:
            return self.role.strip().lower()
        return _TAG_ROLES.get(self.tag, "unknown")

    def live_name(self) -> str:
        return self.aria_label if self.aria_label else self.name


class _FakeHandle:
    def __init__(self, element: _FakeElement) -> None:
        self._element = element

    def evaluate(self, _fn: str) -> str:
        return self._element.tag


class _FakeLocator:
    def __init__(self, element: _FakeElement) -> None:
        self._element = element

    def is_visible(self) -> bool:
        return self._element.visible

    def get_attribute(self, name: str) -> str | None:
        if name == "role":
            return self._element.role
        if name == "type":
            return self._element.input_type or None
        if name == "aria-label":
            return self._element.aria_label
        return None

    def element_handle(self) -> _FakeHandle:
        return _FakeHandle(self._element)

    def inner_text(self) -> str:
        return self._element.name

    def click(self, timeout: int | None = None) -> None:
        del timeout
        self._element.clicks += 1

    def fill(self, text: str, timeout: int | None = None) -> None:
        del timeout
        self._element.filled = text


class _FakeListLocator:
    def __init__(self, locators: list[_FakeLocator]) -> None:
        self._locators = locators

    def all(self) -> list[_FakeLocator]:
        return list(self._locators)


class _FakeGroupLocator:
    def __init__(self, locators: list[_FakeLocator]) -> None:
        self._locators = locators

    def count(self) -> int:
        return len(self._locators)

    def nth(self, index: int) -> _FakeLocator:
        if index < 0 or index >= len(self._locators):
            raise AssertionError("nth index out of range")
        return self._locators[index]


class _FakePage:
    def __init__(
        self,
        elements: list[_FakeElement] | None = None,
        *,
        url: str = "https://example.com/page",
        title: str = "Example",
    ) -> None:
        self._elements = elements if elements is not None else []
        self.url = url
        self._title = title
        self.redirect_to: str | None = None
        self.goto_calls: list[str] = []

    def goto(self, url: str, wait_until: str | None = None, timeout: int | None = None) -> None:
        del wait_until, timeout
        self.goto_calls.append(url)
        self.url = self.redirect_to or url

    def title(self) -> str:
        return self._title

    def locator(self, _selector: str) -> _FakeListLocator:
        return _FakeListLocator([_FakeLocator(e) for e in self._elements])

    def get_by_role(
        self, role: str, name: str | None = None, exact: bool = False
    ) -> _FakeGroupLocator:
        del exact
        matches = []
        for e in self._elements:
            if e.live_role() != role:
                continue
            if name is not None and e.live_name() != name:
                continue
            matches.append(_FakeLocator(e))
        return _FakeGroupLocator(matches)


class _FakeChromium:
    def __init__(self, driver: _FakeDriver) -> None:
        self._driver = driver

    def launch_persistent_context(self, **kwargs: object) -> _FakeContext:
        self._driver.starts += 1
        self._driver.launch_kwargs = kwargs
        ctx = _FakeContext(self._driver)
        self._driver.context = ctx
        return ctx


class _FakeContext:
    def __init__(self, driver: _FakeDriver) -> None:
        self._driver = driver
        self.pages = [driver.page]
        self.closed = False

    def new_page(self) -> _FakePage:
        page = _FakePage()
        self._driver.page = page
        self.pages.append(page)
        return page

    def close(self) -> None:
        self.closed = True
        self._driver.stops += 1


class _FakePlaywright:
    def __init__(self, driver: _FakeDriver) -> None:
        self._driver = driver
        self.chromium = _FakeChromium(driver)

    def stop(self) -> None:
        self._driver.stops += 1


class _FakeSyncCall:
    def __init__(self, driver: _FakeDriver) -> None:
        self._driver = driver

    def start(self) -> _FakePlaywright:
        return _FakePlaywright(self._driver)


class _FakeDriver:
    """Fake Playwright root + persistent-context factory."""

    def __init__(self, page: _FakePage) -> None:
        self.page = page
        self.starts = 0
        self.stops = 0
        self.launch_kwargs: dict[str, object] | None = None
        self.context: _FakeContext | None = None

    def sync_playwright(self) -> _FakeSyncCall:
        return _FakeSyncCall(self)


def _install(monkeypatch: pytest.MonkeyPatch, driver: _FakeDriver) -> None:
    monkeypatch.setattr(context_mod, "is_windows", lambda: True)
    monkeypatch.setattr(context_mod, "_import_playwright", lambda: driver.sync_playwright)


def _started(monkeypatch: pytest.MonkeyPatch, page: _FakePage | None = None):
    driver = _FakeDriver(page or _FakePage())
    _install(monkeypatch, driver)
    browser = PlaywrightBrowser(settings=Settings(), profile="general")
    browser.start()
    return browser, driver


# ── lifecycle ──────────────────────────────────────────────────────────────


def test_construction_never_launches() -> None:
    browser = PlaywrightBrowser(settings=Settings())
    assert browser.started is False
    with pytest.raises(ValueError):
        PlaywrightBrowser(settings=Settings(), profile="owner")  # unknown profile


def test_requires_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(context_mod, "is_windows", lambda: False)
    browser = PlaywrightBrowser(settings=Settings())
    with pytest.raises(BrowserError, match="Windows"):
        browser.start()


def test_missing_playwright_is_a_clean_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom() -> object:
        raise BrowserError("Playwright is not installed")

    monkeypatch.setattr(context_mod, "is_windows", lambda: True)
    monkeypatch.setattr(context_mod, "_import_playwright", _boom)
    browser = PlaywrightBrowser(settings=Settings())
    with pytest.raises(BrowserError, match="not installed"):
        browser.start()
    # A failed start must release the profile lock (retryable).
    assert profile_lock("general").acquire(blocking=False) is True
    profile_lock("general").release()


def test_launch_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    browser, driver = _started(monkeypatch)
    try:
        assert driver.starts == 1
        assert driver.launch_kwargs is not None
        assert driver.launch_kwargs["channel"] == "chrome"
        assert driver.launch_kwargs["headless"] is False
        user_data_dir = str(driver.launch_kwargs["user_data_dir"])
        assert user_data_dir == str(browser_profile_dir("general"))
        assert "whatsapp" not in user_data_dir.lower()
        assert browser.started is True
    finally:
        browser.close()


def test_holds_profile_lock_for_session(monkeypatch: pytest.MonkeyPatch) -> None:
    browser, _ = _started(monkeypatch)
    try:
        # While the session is live the shared profile lock is held.
        assert profile_lock("general").acquire(blocking=False) is False
    finally:
        browser.close()
    # Closing releases it for the next task.
    assert profile_lock("general").acquire(blocking=False) is True
    profile_lock("general").release()


def test_profile_lock_serializes(monkeypatch: pytest.MonkeyPatch) -> None:
    assert profile_lock("general") is profile_lock("general")
    lock = profile_lock("general")
    assert lock.acquire(blocking=False) is True
    outcomes: list[bool] = []

    def _try() -> None:
        outcomes.append(lock.acquire(timeout=0.05))

    thread = threading.Thread(target=_try)
    thread.start()
    thread.join()
    assert outcomes == [False]
    lock.release()


def test_close_then_restart(monkeypatch: pytest.MonkeyPatch) -> None:
    browser, driver = _started(monkeypatch)
    browser.close()
    assert browser.started is False
    assert driver.stops >= 1
    browser.start()
    try:
        assert driver.starts == 2
    finally:
        browser.close()


# ── goto ───────────────────────────────────────────────────────────────────


def test_goto_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    page = _FakePage(url="about:blank")
    browser, _ = _started(monkeypatch, page)
    try:
        data = browser.goto("https://example.com/path")
        assert data["url"] == "https://example.com/path"
        assert data["origin"] == "https://example.com"
        assert page.goto_calls == ["https://example.com/path"]
    finally:
        browser.close()


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "file:///etc/passwd",
        "ms-settings:",
        "https://user:pass@example.com/",
        "not a url",
        "http:///nohost",
    ],
)
def test_goto_rejects_unsafe_urls(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    page = _FakePage(url="about:blank")
    browser, _ = _started(monkeypatch, page)
    try:
        with pytest.raises(BrowserError):
            browser.goto(url)
        assert page.goto_calls == []  # never navigated
    finally:
        browser.close()


def test_goto_redirect_to_allowed_origin_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings()
    settings.browser.allowed_domains = ["login.example.com"]
    page = _FakePage(url="about:blank")
    page.redirect_to = "https://login.example.com/home"
    driver = _FakeDriver(page)
    _install(monkeypatch, driver)
    browser = PlaywrightBrowser(settings=settings)
    browser.start()
    try:
        data = browser.goto("https://shop.example.com/")
        assert data["origin"] == "https://login.example.com"
    finally:
        browser.close()


def test_goto_redirect_unauthorized_blocks_further_interaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = _FakePage(url="about:blank")
    page.redirect_to = "https://evil.example.net/landing"
    browser, _ = _started(monkeypatch, page)
    try:
        with pytest.raises(BrowserError, match="unauthorized origin"):
            browser.goto("https://shop.example.com/")
        # Interactions on the wrong page are refused until a valid goto clears it.
        with pytest.raises(BrowserError, match="blocked"):
            browser.read()
        with pytest.raises(BrowserError, match="blocked"):
            browser.click("e1")
        with pytest.raises(BrowserError, match="blocked"):
            browser.type_text("e1", "hi")
        assert browser._redirect_blocked is True
        # A new, non-redirecting navigation is authorized afresh and clears it.
        page.redirect_to = None
        browser.goto("https://example.com/ok")
        assert browser._redirect_blocked is False
    finally:
        browser.close()


def test_goto_same_origin_redirect_not_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    page = _FakePage(url="about:blank")
    page.redirect_to = "https://example.com/other"
    browser, _ = _started(monkeypatch, page)
    try:
        data = browser.goto("https://example.com/start")
        assert data["origin"] == "https://example.com"
        assert browser._redirect_blocked is False
    finally:
        browser.close()


# ── read / references ──────────────────────────────────────────────────────


def _elements(n: int) -> list[_FakeElement]:
    return [_FakeElement("button", f"Button {i}") for i in range(n)]


def test_read_bounds_and_references(monkeypatch: pytest.MonkeyPatch) -> None:
    page = _FakePage(_elements(200))
    browser, _ = _started(monkeypatch, page)
    try:
        observation = browser.read(max_elements=5)
        assert len(observation.elements) == 5
        refs = [e.ref for e in observation.elements]
        assert refs == ["e1", "e2", "e3", "e4", "e5"]
        assert observation.origin == "https://example.com"
        assert observation.elements[0].role == "button"
        assert observation.elements[0].name == "Button 0"
    finally:
        browser.close()


def test_read_skips_unnamed_and_flags_password(monkeypatch: pytest.MonkeyPatch) -> None:
    page = _FakePage(
        [
            _FakeElement("button"),  # unnamed -> skipped
            _FakeElement("input", "Email", input_type="email"),
            _FakeElement("input", "Password", input_type="password"),
            _FakeElement("a", "Home"),
        ]
    )
    browser, _ = _started(monkeypatch, page)
    try:
        observation = browser.read()
        names = [e.name for e in observation.elements]
        assert names == ["Email", "Password", "Home"]
        password = next(e for e in observation.elements if e.name == "Password")
        assert password.is_password is True
        assert password.input_type == "password"
        email = next(e for e in observation.elements if e.name == "Email")
        assert email.is_password is False
    finally:
        browser.close()


def test_references_are_transient_across_navigation(monkeypatch: pytest.MonkeyPatch) -> None:
    page = _FakePage([_FakeElement("button", "Send")], url="https://example.com/a")
    browser, _ = _started(monkeypatch, page)
    try:
        browser.read()
        assert browser.resolve("e1") is not None
        browser.goto("https://example.com/b")
        assert browser.resolve("e1") is None  # refs cleared on navigation
    finally:
        browser.close()


def test_resolve_unknown_ref(monkeypatch: pytest.MonkeyPatch) -> None:
    browser, _ = _started(monkeypatch, _FakePage([_FakeElement("button", "Send")]))
    try:
        assert browser.resolve("e999") is None
    finally:
        browser.close()


# ── click ──────────────────────────────────────────────────────────────────


def _one_button(monkeypatch: pytest.MonkeyPatch, element: _FakeElement | None = None):
    page = _FakePage([element or _FakeElement("button", "Send")])
    browser, _ = _started(monkeypatch, page)
    browser.read()
    return browser, page


def test_click_revalidates_and_clicks(monkeypatch: pytest.MonkeyPatch) -> None:
    browser, page = _one_button(monkeypatch)
    try:
        data = browser.click("e1")
        assert data["role"] == "button"
        assert data["name"] == "Send"
        assert page._elements[0].clicks == 1
    finally:
        browser.close()


def test_click_unknown_ref_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    browser, _ = _one_button(monkeypatch)
    try:
        with pytest.raises(BrowserError, match="unknown or stale"):
            browser.click("e42")
    finally:
        browser.close()


def test_click_changed_name_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    element = _FakeElement("button", "Send")
    browser, _ = _one_button(monkeypatch, element)
    try:
        element.name = "Send money"  # label changed since observation
        with pytest.raises(BrowserError, match="identity changed|no longer present"):
            browser.click("e1")
        assert element.clicks == 0
    finally:
        browser.close()


def test_click_ambiguous_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    element = _FakeElement("button", "Send")
    page = _FakePage([element])
    browser, _ = _started(monkeypatch, page)
    browser.read()
    try:
        page._elements.append(_FakeElement("button", "Send"))  # ambiguity appears
        with pytest.raises(BrowserError, match="ambiguous or changed"):
            browser.click("e1")
        assert element.clicks == 0
    finally:
        browser.close()


def test_click_after_page_navigation_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    browser, page = _one_button(monkeypatch)
    try:
        page.url = "https://other.example.org/page"
        with pytest.raises(BrowserError, match="navigated"):
            browser.click("e1")
    finally:
        browser.close()


# ── type ───────────────────────────────────────────────────────────────────


def test_type_fills_field(monkeypatch: pytest.MonkeyPatch) -> None:
    element = _FakeElement("input", "Search", input_type="search")
    browser, _ = _one_button(monkeypatch, element)
    try:
        data = browser.type_text("e1", "hello world")
        assert element.filled == "hello world"
        assert data["characters"] == 11
        assert "text" not in data
    finally:
        browser.close()


def test_type_refuses_password_field(monkeypatch: pytest.MonkeyPatch) -> None:
    element = _FakeElement("input", "Password", input_type="password")
    browser, _ = _one_button(monkeypatch, element)
    try:
        with pytest.raises(BrowserError, match="credential"):
            browser.type_text("e1", "hunter2")
        assert element.filled is None
    finally:
        browser.close()


def test_type_refuses_non_input_target(monkeypatch: pytest.MonkeyPatch) -> None:
    browser, page = _one_button(monkeypatch)  # a <button>, not a text input
    try:
        with pytest.raises(BrowserError, match="not a text input"):
            browser.type_text("e1", "hi")
        assert page._elements[0].filled is None
    finally:
        browser.close()


def test_type_rejects_empty_or_oversized(monkeypatch: pytest.MonkeyPatch) -> None:
    browser, _ = _one_button(monkeypatch, _FakeElement("input", "Search"))
    try:
        with pytest.raises(BrowserError, match="empty or oversized"):
            browser.type_text("e1", "")
        with pytest.raises(BrowserError, match="empty or oversized"):
            browser.type_text("e1", "x" * 5000)
    finally:
        browser.close()


# ── WhatsApp page accessor (Stage 8, Slice 8.4b) ───────────────────────────


def _started_whatsapp(monkeypatch: pytest.MonkeyPatch):
    driver = _FakeDriver(_FakePage())
    _install(monkeypatch, driver)
    browser = PlaywrightBrowser(settings=Settings(), profile="whatsapp")
    browser.start()
    return browser, driver


def test_whatsapp_page_is_only_for_the_whatsapp_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    browser, _ = _started(monkeypatch)
    try:
        with pytest.raises(BrowserError, match="whatsapp profile"):
            browser.whatsapp_page()
    finally:
        browser.close()


def test_whatsapp_page_requires_a_started_session(monkeypatch: pytest.MonkeyPatch) -> None:
    browser = PlaywrightBrowser(settings=Settings(), profile="whatsapp")
    with pytest.raises(BrowserError, match="not started"):
        browser.whatsapp_page()


def test_whatsapp_page_returns_the_narrow_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    from jarvis.browser.whatsapp_page import PlaywrightWhatsAppPage

    browser, _ = _started_whatsapp(monkeypatch)
    try:
        page = browser.whatsapp_page()
        assert isinstance(page, PlaywrightWhatsAppPage)
        # The narrow surface has no generic DOM handle.
        assert not hasattr(page, "evaluate")
        assert not hasattr(page, "locator")
    finally:
        browser.close()


def test_the_whatsapp_accessor_is_not_part_of_the_session_protocol() -> None:
    from jarvis.browser.context import BrowserSession

    assert "whatsapp_page" not in getattr(BrowserSession, "__protocol_attrs__", set())
