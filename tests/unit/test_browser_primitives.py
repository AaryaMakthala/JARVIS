from jarvis.browser.context import BrowserError, KEY_ALLOWLIST
from jarvis.browser.tools import (
    BrowserPressKeyArgs,
    BrowserScrollArgs,
    BrowserTabsArgs,
    BrowserScreenshotArgs,
    BrowserSearchArgs,
    make_browser_press_key_spec,
    make_browser_screenshot_spec,
    make_browser_scroll_spec,
    make_browser_search_spec,
    make_browser_tabs_spec,
)
from jarvis.config import Settings, SiteSettings
from jarvis.tools.base import ToolContext


class FakeSession:
    def __init__(self):
        self.calls = []
        self.url = "https://example.com/"
        self.origin = "example.com"
        self.tabs = [{"index": 0, "url": self.url, "title": "t"}]

    def goto(self, url: str):
        self.calls.append(("goto", url))
        self.url = url
        self.origin = "example.com"
        return {"url": url, "origin": self.origin, "title": "t"}

    def current_url(self):
        return self.url

    def session_id(self):
        return "fake-1"

    def press_key(self, key: str):
        if key not in KEY_ALLOWLIST:
            raise BrowserError(f"key {key!r} not allowed")
        self.calls.append(("press", key))
        return {"key": key}

    def scroll(self, direction: str, amount: int = 1):
        amt = max(1, min(int(amount), 10))
        if direction not in ("up", "down"):
            raise BrowserError("bad")
        self.calls.append(("scroll", direction, amt))
        return {"direction": direction, "amount": amt}

    def wait(self, seconds: float = 1.0, *, load_state: str = None):
        secs = max(0.0, min(float(seconds), 10.0))
        self.calls.append(("wait", secs))
        return {"seconds": secs}

    def back(self):
        self.calls.append(("back",))
        self.url = "https://example.com/prev"
        return {"url": self.url}

    def forward(self):
        self.calls.append(("forward",))
        self.url = "https://example.com/next"
        return {"url": self.url}

    def list_tabs(self):
        return self.tabs

    def switch_tab(self, index: int):
        if index < 0 or index >= len(self.tabs):
            raise BrowserError("tab index out of range")
        self.calls.append(("switch", index))
        self.url = self.tabs[index]["url"]
        return {"index": index, "url": self.url}

    def close_extra_tab(self):
        self.calls.append(("close_extra",))
        return {"closed": 0}

    def screenshot(self, path: str = None):
        self.calls.append(("screenshot", path))
        p = "/tmp/screenshot-test.png"
        return {"path": p}

    def find(self, query: str, **kwargs):
        from jarvis.browser.context import ElementRef

        return [ElementRef(ref="e1", role="link", name="Result One", origin="example.com")]


def test_press_key_allowlist():
    s = FakeSession()
    ctx = ToolContext(settings=Settings(), browser=s)
    spec = make_browser_press_key_spec()
    r = spec.execute(BrowserPressKeyArgs(key="Enter"), ctx)
    assert r.ok


def test_scroll_caps():
    s = FakeSession()
    ctx = ToolContext(settings=Settings(), browser=s)
    spec = make_browser_scroll_spec()
    r = spec.execute(BrowserScrollArgs(direction="down", amount=10), ctx)
    assert r.ok


def test_tabs_bounds():
    s = FakeSession()
    ctx = ToolContext(settings=Settings(), browser=s)
    spec = make_browser_tabs_spec()
    r = spec.execute(BrowserTabsArgs(action="switch", index=5), ctx)
    assert not r.ok


def test_screenshot_artifact():
    s = FakeSession()
    ctx = ToolContext(settings=Settings(), browser=s)
    spec = make_browser_screenshot_spec()
    r = spec.execute(BrowserScreenshotArgs(), ctx)
    assert r.ok
    assert r.data.get("artifact", {}).get("type") == "screenshot"


def test_browser_search_url_built_in_code():
    s = FakeSession()
    ctx = ToolContext(
        settings=Settings(sites=[SiteSettings(name="YouTube", url="https://www.youtube.com", aliases=["yt"])]),
        browser=s,
    )
    spec = make_browser_search_spec()
    r = spec.execute(BrowserSearchArgs(site="YouTube", query="cats"), ctx)
    assert r.ok
    assert any("results?search_query=" in str(c) for c in s.calls if c[0] == "goto")


def test_browser_search_no_template_message():
    s = FakeSession()
    ctx = ToolContext(settings=Settings(sites=[SiteSettings(name="Gmail", url="https://mail.google.com")]), browser=s)
    spec = make_browser_search_spec()
    r = spec.execute(BrowserSearchArgs(site="Gmail", query="test"), ctx)
    assert not r.ok
    assert "browser_open_site" in r.error or "browser_type" in r.error


def test_browser_search_records_result_set():
    s = FakeSession()
    ctx = ToolContext(
        settings=Settings(sites=[SiteSettings(name="YouTube", url="https://www.youtube.com", aliases=["yt"])]),
        browser=s,
    )
    spec = make_browser_search_spec()
    r = spec.execute(BrowserSearchArgs(site="YouTube", query="cats"), ctx)
    assert r.ok
    assert "results" in r.data
    assert len(r.data.get("results", [])) >= 1
