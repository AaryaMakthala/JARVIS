"""Managed-browser driver tests (Stage 7, Slice 6).

Playwright is never imported and Chrome is never launched: every test injects a
fake Playwright/``sync_api`` object, a fake persistent context and fake pages/
locators.  The tests assert the driver's own logic (lazy start, launch
arguments, per-profile serialization, redirect fail-closed, bounded reads,
transient references, identity re-validation and credential refusal).
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from jarvis.browser import context as context_mod
from jarvis.browser.context import BrowserError, PlaywrightBrowser, profile_lock
from jarvis.browser.find import match_identity
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
        self.click_url: str | None = None

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
    def __init__(self, element: _FakeElement, page: _FakePage | None = None) -> None:
        self._element = element
        self._page = page

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
        if self._page is not None and self._element.click_url is not None:
            self._page.url = self._element.click_url

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


class _FakeKeyboard:
    def __init__(self, page: _FakePage) -> None:
        self._page = page

    def press(self, key: str) -> None:
        self._page.key_calls.append(key)
        if self._page.expect_depth and key == self._page.navigate_on_key:
            self._page.url = self._page.navigation_url
            self._page.navigation_waited = True


class _FakeExpectNavigation:
    def __init__(self, page: _FakePage) -> None:
        self._page = page

    def __enter__(self) -> None:
        self._page.expect_depth += 1

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._page.expect_depth -= 1


class _FakeMouse:
    def __init__(self) -> None:
        self.wheel_calls: list[tuple[int, int]] = []

    def wheel(self, dx: int, dy: int) -> None:
        self.wheel_calls.append((dx, dy))


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
        self.keyboard = _FakeKeyboard(self)
        self.key_calls: list[str] = []
        self.mouse = _FakeMouse()
        self.title_calls = 0
        self.screenshot_calls: list[str | None] = []
        self.screenshot_result = b"\x89PNG\r\n\x1a\nfake"
        self.screenshot_navigation_url: str | None = None
        self.wait_calls: list[tuple[str, int | None]] = []
        self.go_back_calls = 0
        self.go_forward_calls = 0
        self.bring_to_front_calls = 0
        self.close_calls = 0
        self.enter_can_navigate = False
        self.navigate_on_key = ""
        self.navigation_url = url
        self.navigation_waited = False
        self.expect_navigation_calls = 0
        self.expect_depth = 0

    def goto(self, url: str, wait_until: str | None = None, timeout: int | None = None) -> None:
        del wait_until, timeout
        self.goto_calls.append(url)
        self.url = self.redirect_to or url

    def title(self) -> str:
        self.title_calls += 1
        return self._title

    def screenshot(self, path: str | None = None) -> bytes:
        self.screenshot_calls.append(path)
        if self.screenshot_navigation_url is not None:
            self.url = self.screenshot_navigation_url
        return self.screenshot_result

    def wait_for_load_state(self, state: str, timeout: int | None = None) -> None:
        self.wait_calls.append((state, timeout))

    def go_back(self, wait_until: str | None = None, timeout: int | None = None) -> None:
        del wait_until, timeout
        self.go_back_calls += 1

    def go_forward(self, wait_until: str | None = None, timeout: int | None = None) -> None:
        del wait_until, timeout
        self.go_forward_calls += 1

    def bring_to_front(self) -> None:
        self.bring_to_front_calls += 1

    def close(self) -> None:
        self.close_calls += 1

    def evaluate(self, script: str) -> bool:
        if "document.activeElement" in script:
            return self.enter_can_navigate
        raise AssertionError(f"unexpected script: {script}")

    def expect_navigation(
        self, *, wait_until: str | None = None, timeout: int | None = None
    ) -> _FakeExpectNavigation:
        del wait_until, timeout
        self.expect_navigation_calls += 1
        return _FakeExpectNavigation(self)

    def locator(self, _selector: str) -> _FakeListLocator:
        return _FakeListLocator([_FakeLocator(e, self) for e in self._elements])

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
            matches.append(_FakeLocator(e, self))
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


def test_click_cross_origin_blocks_follow_up_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    element = _FakeElement("button", "Send")
    element.click_url = "https://evil.example.net/landing"
    page = _FakePage([element])
    driver = _FakeDriver(page)
    _install(monkeypatch, driver)
    browser = PlaywrightBrowser(settings=Settings())
    browser.start()
    browser.read()
    try:
        with pytest.raises(BrowserError, match="disallowed domain"):
            browser.click("e1")
        assert browser._redirect_blocked is True
        with pytest.raises(BrowserError, match="blocked"):
            browser.read()
        with pytest.raises(BrowserError, match="blocked"):
            browser.press_key("Tab")
        assert page.key_calls == []
    finally:
        browser.close()


def test_blocked_session_refuses_reads_and_actions_before_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = _FakePage([_FakeElement("button", "Send")])
    browser, _ = _started(monkeypatch, page)
    browser.read()
    browser._redirect_blocked = True
    operations = [
        ("scroll", lambda: browser.scroll("down")),
        ("wait", lambda: browser.wait(1, load_state="load")),
        ("back", browser.back),
        ("forward", browser.forward),
        ("switch_tab", lambda: browser.switch_tab(0)),
        ("close_extra_tab", browser.close_extra_tab),
        ("resolve", lambda: browser.resolve("e1")),
        ("current_url", browser.current_url),
    ]
    try:
        for _name, operation in operations:
            with pytest.raises(BrowserError, match="blocked"):
                operation()
        assert page.mouse.wheel_calls == []
        assert page.wait_calls == []
        assert page.go_back_calls == 0
        assert page.go_forward_calls == 0
        assert page.bring_to_front_calls == 0
        assert page.close_calls == 0
    finally:
        browser.close()


def test_blocked_session_refuses_to_list_tabs(monkeypatch: pytest.MonkeyPatch) -> None:
    page = _FakePage(title="Secret tab")
    browser, _ = _started(monkeypatch, page)
    browser._redirect_blocked = True
    try:
        with pytest.raises(BrowserError, match="blocked"):
            browser.list_tabs()
        assert page.title_calls == 0
    finally:
        browser.close()


def test_blocked_session_refuses_screenshot_before_file_creation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    page = _FakePage()
    browser, _ = _started(monkeypatch, page)
    browser._redirect_blocked = True
    target = tmp_path / "nested" / "shot.png"
    try:
        with pytest.raises(BrowserError, match="blocked"):
            browser.screenshot(str(target))
        assert page.screenshot_calls == []
        assert not target.exists()
        assert not target.parent.exists()
    finally:
        browser.close()


def test_screenshot_refuses_disallowed_origin_before_capture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = Settings()
    settings.browser.allowed_domains = ["example.com"]
    page = _FakePage(url="https://evil.example.net/landing")
    driver = _FakeDriver(page)
    _install(monkeypatch, driver)
    browser = PlaywrightBrowser(settings=settings)
    browser.start()
    target = tmp_path / "nested" / "shot.png"
    try:
        with pytest.raises(BrowserError, match="disallowed domain"):
            browser.screenshot(str(target))
        assert browser._redirect_blocked is True
        assert page.screenshot_calls == []
        assert not target.parent.exists()
    finally:
        browser.close()


def test_screenshot_refuses_publication_when_origin_changes_during_capture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = Settings()
    settings.browser.allowed_domains = ["example.com"]
    page = _FakePage()
    page.screenshot_navigation_url = "https://evil.example.net/landing"
    driver = _FakeDriver(page)
    _install(monkeypatch, driver)
    browser = PlaywrightBrowser(settings=settings)
    browser.start()
    target = tmp_path / "nested" / "shot.png"
    try:
        with pytest.raises(BrowserError, match="disallowed domain"):
            browser.screenshot(str(target))
        assert page.screenshot_calls == [None]
        assert browser._redirect_blocked is True
        assert not target.exists()
        assert not target.parent.exists()
        with pytest.raises(BrowserError, match="blocked"):
            browser.read()
    finally:
        browser.close()


def test_screenshot_allowed_origin_captures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = Settings()
    settings.browser.allowed_domains = ["example.com"]
    page = _FakePage()
    driver = _FakeDriver(page)
    _install(monkeypatch, driver)
    browser = PlaywrightBrowser(settings=settings)
    browser.start()
    target = tmp_path / "shot.png"
    try:
        data = browser.screenshot(str(target))
        assert data["path"] == str(target)
        assert page.screenshot_calls == [None]
        assert target.read_bytes() == page.screenshot_result
        assert browser._redirect_blocked is False
    finally:
        browser.close()


# ── keyboard navigation ────────────────────────────────────────────────────


def test_press_enter_waits_for_navigation(monkeypatch: pytest.MonkeyPatch) -> None:
    page = _FakePage(url="https://example.com/search")
    page.enter_can_navigate = True
    page.navigate_on_key = "Enter"
    page.navigation_url = "https://example.com/results?q=jarvis"
    browser, _ = _started(monkeypatch, page)
    try:
        data = browser.press_key("Enter")
        assert data == {"key": "Enter"}
        assert page.expect_navigation_calls == 1
        assert page.navigation_waited is True
        assert browser.current_url() == "https://example.com/results?q=jarvis"
    finally:
        browser.close()


def test_press_enter_without_navigable_control_does_not_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = _FakePage(url="https://example.com/search")
    page.enter_can_navigate = False
    browser, _ = _started(monkeypatch, page)
    try:
        browser.press_key("Enter")
        assert page.key_calls == ["Enter"]
        assert page.expect_navigation_calls == 0
    finally:
        browser.close()


def test_press_non_navigation_key_does_not_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    page = _FakePage(url="https://example.com/search")
    page.enter_can_navigate = True
    browser, _ = _started(monkeypatch, page)
    try:
        browser.press_key("Tab")
        assert page.key_calls == ["Tab"]
        assert page.expect_navigation_calls == 0
    finally:
        browser.close()


# ── find / open_match (Stage 8.6, slice 8.6.2) ───────────────────────────


def _result_page() -> _FakePage:
    return _FakePage(
        [
            _FakeElement("a", "Python Tutorial - Part 1"),
            _FakeElement("a", "Python Tutorial - Part 2"),
            _FakeElement("a", "Java Basics"),
            _FakeElement("button", "Search"),
        ]
    )


def test_find_lists_the_matching_elements(monkeypatch: pytest.MonkeyPatch) -> None:
    browser, _ = _started(monkeypatch, _result_page())
    try:
        matches = browser.find("Python")
        assert [m.name for m in matches] == ["Python Tutorial - Part 1", "Python Tutorial - Part 2"]
        assert all(m.origin == "https://example.com" for m in matches)
    finally:
        browser.close()


def test_find_honours_the_role_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    browser, _ = _started(monkeypatch, _result_page())
    try:
        assert [m.name for m in browser.find("Search", role="button")] == ["Search"]
        assert browser.find("Search", role="link") == []
    finally:
        browser.close()


def test_find_reports_no_match(monkeypatch: pytest.MonkeyPatch) -> None:
    browser, _ = _started(monkeypatch, _result_page())
    try:
        assert browser.find("nothing like this") == []
    finally:
        browser.close()


def test_find_keeps_its_refs_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    """A new observation invalidates the previous one (unchanged contract)."""
    browser, _ = _started(monkeypatch, _result_page())
    try:
        browser.read()
        assert browser.resolve("e1") is not None
        browser.find("Java")
        assert browser.resolve("e1") is None
    finally:
        browser.close()


def test_open_match_clicks_the_match(monkeypatch: pytest.MonkeyPatch) -> None:
    page = _result_page()
    browser, _ = _started(monkeypatch, page)
    try:
        data = browser.open_match("Java Basics")
        assert data["name"] == "Java Basics"
        assert data["position"] == 1
        assert data["matches"] == 1
        assert page._elements[2].clicks == 1
    finally:
        browser.close()


def test_open_match_refuses_an_ambiguous_match(monkeypatch: pytest.MonkeyPatch) -> None:
    page = _result_page()
    browser, _ = _started(monkeypatch, page)
    try:
        with pytest.raises(BrowserError, match="elements match"):
            browser.open_match("Python")
        assert all(e.clicks == 0 for e in page._elements)
    finally:
        browser.close()


def test_open_match_opens_the_chosen_match(monkeypatch: pytest.MonkeyPatch) -> None:
    page = _result_page()
    browser, _ = _started(monkeypatch, page)
    try:
        data = browser.open_match("Python", index=2)
        assert data["name"] == "Python Tutorial - Part 2"
        assert data["position"] == 2
        assert page._elements[1].clicks == 1
        assert page._elements[0].clicks == 0
    finally:
        browser.close()


def test_open_match_refuses_an_out_of_range_index(monkeypatch: pytest.MonkeyPatch) -> None:
    page = _result_page()
    browser, _ = _started(monkeypatch, page)
    try:
        with pytest.raises(BrowserError, match="out of range"):
            browser.open_match("Python", index=9)
        assert all(e.clicks == 0 for e in page._elements)
    finally:
        browser.close()


def test_open_match_refuses_a_changed_element(monkeypatch: pytest.MonkeyPatch) -> None:
    """The confirmed identity no longer holds: nothing is clicked."""
    page = _result_page()
    browser, _ = _started(monkeypatch, page)
    try:
        approved = match_identity("link", "Java Basics", 1)
        page._elements[2].name = "Java Basics 2024"
        with pytest.raises(BrowserError, match="changed after you confirmed"):
            browser.open_match("Java", expect_identity=approved)
        assert all(e.clicks == 0 for e in page._elements)
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


def test_blocked_whatsapp_session_refuses_page_accessor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    browser, _ = _started_whatsapp(monkeypatch)
    browser._redirect_blocked = True
    try:
        with pytest.raises(BrowserError, match="blocked"):
            browser.whatsapp_page()
    finally:
        browser.close()


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
