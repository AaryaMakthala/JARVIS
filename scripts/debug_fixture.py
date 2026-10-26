"""One-off diagnostic for the browser fixture search/Enter path.

Run on Windows from the repo root:
    .venv\\Scripts\\python.exe scripts/debug_fixture.py

It starts the offline fixture site, opens the managed Playwright browser,
types into the search box and presses Enter. It prints the URL and the
document.activeElement tag after every step. It makes no network call.
"""

from __future__ import annotations

import http.server
import os
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

from jarvis.browser.context import BrowserError, PlaywrightBrowser
from jarvis.config import BrowserSettings, Settings

SITE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "site"


class _Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SITE), **kwargs)

    def translate_path(self, path):
        translated = Path(super().translate_path(path))
        if not translated.exists() and translated.suffix == "":
            html = translated.with_suffix(".html")
            if html.is_file():
                return str(html)
        return str(translated)

    def log_message(self, *args):
        return


def _active(page) -> str:
    try:
        return str(
            page.evaluate(
                "() => { const el = document.activeElement;"
                " return el ? el.tagName.toLowerCase() : null; }"
            )
        )
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        return f"<error: {exc}>"


def _show(step: str, session: PlaywrightBrowser, page) -> None:
    url = session.current_url()
    parts = urlsplit(url)
    print(f"{step}: path={parts.path!r} query={bool(parts.query)} active={_active(page)}")


def main() -> int:
    if os.name != "nt":
        print("Windows-only: the managed browser uses Chrome on Windows.")
        return 2
    if not SITE.is_dir():
        print(f"fixture site not found: {SITE}")
        return 2

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    settings = Settings(browser=BrowserSettings(allowed_domains=["127.0.0.1"]))
    session = PlaywrightBrowser(settings, profile="general")
    try:
        session.start()
        page = getattr(session, "_page", None)
        if page is None:
            print("no active page after start")
            return 1
        print(f"base={base}")

        session.goto(f"{base}/search")
        _show("goto", session, page)

        observation = session.read()
        _show("read", session, page)
        search = next(
            (e for e in observation.elements if e.role == "textbox" and e.name == "Search"),
            None,
        )
        if search is None:
            seen = [(e.role, e.name) for e in observation.elements]
            print(f"no textbox named Search; observed={seen}")
            return 1
        print(f"target: ref={search.ref} role={search.role} name={search.name!r}")

        form = page.evaluate(
            "() => { const f = document.querySelector('form');"
            " return f ? {action: f.getAttribute('action'), method: f.method} : null; }"
        )
        print(f"form={form}")

        session.type_text(search.ref, "jarvis")
        _show("type", session, page)

        events: list[str] = []

        def _on_navigated(frame) -> None:
            events.append(f"framenavigated {frame.url}")

        def _on_request(request) -> None:
            events.append(f"request {request.method} {request.url}")

        def _on_response(response) -> None:
            try:
                status = response.status
            except Exception as exc:  # noqa: BLE001 - diagnostic only
                status = f"<error: {exc}>"
            events.append(f"response {status} {response.url}")

        page.on("framenavigated", _on_navigated)
        page.on("request", _on_request)
        page.on("response", _on_response)

        try:
            session.press_key("Enter")
        except BrowserError as exc:
            print(f"enter error: {exc}")
        _show("enter", session, page)

        time.sleep(2.0)
        print(f"url: page.url={page.url!r} current_url={session.current_url()!r}")
        try:
            location = page.evaluate("() => document.location.href")
        except Exception as exc:  # noqa: BLE001 - context may be destroyed
            location = f"<error: {exc}>"
        print("events since Enter:")
        for event in events:
            print(f"  {event}")
        print(f"location.href={location!r}")
        print(f"redirect_blocked={getattr(session, '_redirect_blocked', None)}")

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if urlsplit(session.current_url()).path == "/results":
                break
            page.wait_for_timeout(100)
        _show("after-wait", session, page)
        if urlsplit(session.current_url()).path != "/results":
            print("RESULT: Enter did not reach /results")
            return 1
        print("RESULT: Enter reached /results")
        return 0
    except BrowserError as exc:
        print(f"browser error: {exc}")
        return 1
    finally:
        try:
            session.close()
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
