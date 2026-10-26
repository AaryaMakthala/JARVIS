import http.server
import threading
from pathlib import Path

import pytest


@pytest.fixture
def fixture_server(tmp_path=None):
    here = Path(__file__).parent / "site"

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(here), **kwargs)

        def translate_path(self, path):
            """Serve ``/name`` from ``name.html`` as well as the exact file."""
            translated = Path(super().translate_path(path))
            if not translated.exists() and translated.suffix == "":
                html = translated.with_suffix(".html")
                if html.is_file():
                    return str(html)
            return str(translated)

        def log_message(self, *args):
            return

    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler) as httpd:
        port = httpd.server_address[1]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        yield f"http://127.0.0.1:{port}"
        httpd.shutdown()
