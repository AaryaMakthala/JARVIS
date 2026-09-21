"""Integration test for IPC authentication over a real TCP socket.

Spins up a :class:`DaemonServer` with a fake context (no LLM, no LangGraph)
and verifies that:
- Unauthenticated connections are rejected
- Wrong tokens are rejected
- Correct token grants access and can send/receive messages
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from jarvis.config import Settings
from jarvis.daemon.protocol import PROTOCOL_VERSION, AuthMessage, StatusRequest
from jarvis.daemon.server import DaemonServer

# ── Helpers ──────────────────────────────────────────────────────────────

TEST_TOKEN = "test-ipc-token-abcdef123456"


class _FakeStore:
    """Minimal SecretStore that returns the test token."""

    def __init__(self, token: str = TEST_TOKEN) -> None:
        self._token = token
        self._kv: dict[str, str] = {"ipc_token": token}

    def get(self, name: str) -> str | None:
        return self._kv.get(name)

    def set(self, name: str, value: str) -> None:
        self._kv[name] = value

    def has(self, name: str) -> bool:
        return name in self._kv

    def check_store_access(self) -> str | None:
        return "FakeStore"


def _make_server() -> DaemonServer:
    """Create a DaemonServer with fake dependencies."""
    settings = Settings()
    store = _FakeStore()
    ctx = MagicMock()  # fake AppContext — no real agent needed
    return DaemonServer(settings=settings, store=store, ctx=ctx)


class _ServerHarness:
    """Manages a daemon server in a background thread with its own event loop."""

    def __init__(self) -> None:
        self.server = _make_server()
        self.port: int = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._started = threading.Event()

    def start(self) -> None:
        def _run() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            async def _serve() -> None:
                srv = await asyncio.start_server(self.server._handle_client, "127.0.0.1", 0)
                self.port = srv.sockets[0].getsockname()[1]
                async with srv:
                    await self.server._shutdown_event.wait()

            self._loop.run_until_complete(_serve())

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        time.sleep(0.5)  # let the server bind

    def stop(self) -> None:
        # The event loop is blocked on _shutdown_event.wait() in another
        # thread — a direct Event.set() from here would never wake its
        # select() sleep.  call_soon_threadsafe is the thread-safe path.
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self.server._shutdown_event.set)
        else:  # pragma: no cover - loop always created in start()
            self.server._shutdown_event.set()
        if self._thread:
            self._thread.join(timeout=3)

    def connect(self, token: str = TEST_TOKEN) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        loop = asyncio.new_event_loop()
        reader, writer = loop.run_until_complete(asyncio.open_connection("127.0.0.1", self.port))
        # Store the loop on the writer so we can use it later
        writer._test_loop = loop  # type: ignore[attr-defined]
        return reader, writer

    def send_and_recv(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, msg: Any
    ) -> dict[str, Any]:
        loop = writer._test_loop  # type: ignore[attr-defined]
        line = msg.model_dump_json() + "\n"
        writer.write(line.encode("utf-8"))
        loop.run_until_complete(writer.drain())
        raw = loop.run_until_complete(asyncio.wait_for(reader.readline(), timeout=5))
        return json.loads(raw.decode("utf-8").strip())

    def close(self, writer: asyncio.StreamWriter) -> None:
        loop = writer._test_loop  # type: ignore[attr-defined]
        writer.close()
        loop.run_until_complete(asyncio.sleep(0.05))
        loop.close()


# ── Tests ────────────────────────────────────────────────────────────────


class TestIPCAuth:
    """Verify authentication over a real TCP socket."""

    @pytest.fixture(autouse=True)
    def _harness(self) -> Any:
        self.harness = _ServerHarness()
        self.harness.start()
        yield
        self.harness.stop()

    def test_correct_token_grants_access(self) -> None:
        reader, writer = self.harness.connect()
        try:
            resp = self.harness.send_and_recv(reader, writer, AuthMessage(token=TEST_TOKEN))
            assert resp["type"] == "auth_ok"
            assert resp["version"] == PROTOCOL_VERSION
        finally:
            self.harness.close(writer)

    def test_wrong_token_rejected(self) -> None:
        reader, writer = self.harness.connect()
        try:
            # Server delays 0.5s then closes without response on wrong token
            loop = writer._test_loop  # type: ignore[attr-defined]
            line = AuthMessage(token="wrong-token").model_dump_json() + "\n"
            writer.write(line.encode())
            loop.run_until_complete(writer.drain())
            raw = loop.run_until_complete(asyncio.wait_for(reader.readline(), timeout=3))
            assert raw == b""  # connection closed without auth_ok
        finally:
            self.harness.close(writer)

    def test_missing_auth_rejected(self) -> None:
        reader, writer = self.harness.connect()
        try:
            # Send a non-auth message as first message
            loop = writer._test_loop  # type: ignore[attr-defined]
            line = StatusRequest().model_dump_json() + "\n"
            writer.write(line.encode())
            loop.run_until_complete(writer.drain())
            # Connection should be closed (EOF)
            raw = loop.run_until_complete(asyncio.wait_for(reader.readline(), timeout=2))
            assert raw == b""
        finally:
            self.harness.close(writer)

    def test_authenticated_can_get_status(self) -> None:
        reader, writer = self.harness.connect()
        try:
            # Auth first
            resp = self.harness.send_and_recv(reader, writer, AuthMessage(token=TEST_TOKEN))
            assert resp["type"] == "auth_ok"
            # Then status
            resp = self.harness.send_and_recv(reader, writer, StatusRequest())
            assert resp["type"] == "status_response"
            assert resp["daemon"] == "running"
        finally:
            self.harness.close(writer)


class TestIPCAuthNoDaemon:
    """Test client error handling when no daemon is running."""

    def test_client_raises_when_daemon_not_running(self) -> None:
        from jarvis.daemon.client import DaemonClient, DaemonError

        client = DaemonClient(port=19999, token="some-token")  # unlikely to be running
        with pytest.raises(DaemonError, match="cannot connect to daemon"):
            client.connect()
