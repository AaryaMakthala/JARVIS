import http.server
import os
import socketserver
import threading
from pathlib import Path

import pytest


@pytest.fixture
def fixture_server(tmp_path=None):
    here = Path(__file__).parent / "site"
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(here), **kwargs)
        def log_message(self, *args):
            return
    with socketserver.TCPServer(("127.0.0.1", 0), Handler) as httpd:
        port = httpd.server_address[1]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        yield f"http://127.0.0.1:{port}"
        httpd.shutdown()
