"""C2: LLM requests are bounded, and a cancel only frees the slot when the
worker has actually stopped.

No network: the HTTP layer is exercised with a stubbed transport and a stub
client, and the daemon cancel path with the existing in-memory server fixture.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx
import pytest

from jarvis.config import Settings, VoiceSettings
from jarvis.llm.abort import cancelled_by
from jarvis.llm.client import bounded_http_timeout, post_with_deadline
from jarvis.llm.openai_compat import OpenAICompatClient
from jarvis.tools.base import Cancelled


class _FakeStore:
    def get(self, name: str) -> str | None:
        return None


def _server() -> Any:
    from jarvis.daemon.server import DaemonServer

    return DaemonServer(settings=Settings(voice=VoiceSettings(enabled=False)), store=_FakeStore())


# ── the overall HTTP bound ────────────────────────────────────────────────


class _SlowBody:
    status_code = 200

    def iter_bytes(self) -> Any:
        for _ in range(500):
            time.sleep(0.01)
            yield b"x"


class _StreamCtx:
    def __enter__(self) -> _SlowBody:
        return _SlowBody()

    def __exit__(self, *exc: object) -> None:
        return None


class _TrickleClient:
    def stream(self, *args: Any, **kwargs: Any) -> _StreamCtx:
        return _StreamCtx()


def test_a_trickling_body_cannot_outlive_the_overall_bound() -> None:
    started = time.monotonic()
    with pytest.raises(httpx.TimeoutException):
        post_with_deadline(
            _TrickleClient(), "chat/completions", body={}, timeout_s=0.1, deadline_at=None
        )
    assert time.monotonic() - started < 1.0


def test_a_cancelled_request_stops_before_it_is_sent() -> None:
    sent = False

    class _RecordingClient:
        def stream(self, *args: Any, **kwargs: Any) -> _StreamCtx:
            nonlocal sent
            sent = True
            return _StreamCtx()

    with cancelled_by(lambda: True), pytest.raises(Cancelled):
        post_with_deadline(
            _RecordingClient(), "chat/completions", body={}, timeout_s=30.0, deadline_at=None
        )
    assert sent is False


def test_all_http_components_are_bounded() -> None:
    timeout = bounded_http_timeout(7.0)
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.connect == timeout.read == timeout.write == timeout.pool == 7.0


def test_the_client_applies_the_component_timeout() -> None:
    client = OpenAICompatClient(
        "key",
        provider="openrouter",
        base_url="https://example.invalid",
        planner_model="m",
        fast_model="m",
        timeout=5.0,
    )
    try:
        assert client._http.timeout.connect == 5.0
        assert client._http.timeout.read == 5.0
    finally:
        client.close()


# ── the provider client still works over the streaming path ───────────────


def _client(handler: Any) -> OpenAICompatClient:
    return OpenAICompatClient(
        "key",
        provider="openrouter",
        base_url="https://example.invalid",
        planner_model="m",
        fast_model="m",
        http_client=httpx.Client(
            base_url="https://example.invalid",
            transport=httpx.MockTransport(handler),
        ),
    )


def test_a_normal_completion_is_parsed_over_streaming() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "hi"}}], "usage": {}},
            request=request,
        )

    text, _usage = _client(handler).text(system="s", user="u", model_role="planner")
    assert text == "hi"


def test_an_unreadable_200_body_is_a_transient_failure() -> None:
    from jarvis.llm.client import LLMTransientError

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json", request=request)

    with pytest.raises(LLMTransientError):
        _client(handler).text(system="s", user="u", model_role="planner")


# ── cancellation only frees the slot when the worker stopped ──────────────


def test_cancel_ack_does_not_mark_the_slot_free() -> None:
    from jarvis.daemon.protocol import CancelMessage
    from jarvis.daemon.server import ClientConnection, TaskSlot

    server = _server()
    slot = TaskSlot(task_id="t1", text="long job", source="terminal", owner_id="oid")
    server._active = slot

    class _Writer:
        def write(self, data: bytes) -> None:
            pass

        async def drain(self) -> None:
            pass

    conn = ClientConnection(writer=_Writer(), reader=object())
    asyncio.run(server._handle_cancel(CancelMessage(task_id="t1"), conn))

    assert slot.cancel.cancelled is True
    assert slot.cancelled is True
    # The worker, not the canceller, decides when the task is done.
    assert slot.done is False
    assert slot.error is None


def test_the_worker_records_exactly_one_cancelled_turn() -> None:
    from jarvis.agent.conversation import ConversationManager
    from jarvis.daemon.server import TaskSlot

    manager = ConversationManager()
    server = _server()
    server._ctx = type("Ctx", (), {"conversations": manager})()
    server._active = TaskSlot(
        task_id="t1",
        text="long job",
        source="terminal",
        owner_id="oid",
        conversation_id="c1",
    )
    server._active.cancel.cancel()
    server._active.cancelled = True

    server._finish_cancelled(server._active)

    assert server._active.done is True
    turns = manager.recent_turns("c1")
    assert [(t.request, t.status) for t in turns] == [("long job", "cancelled")]


def test_daemon_server_cancel_ack_is_json() -> None:
    # Guard the protocol shape the CLI waits for (regression for the reword).
    from jarvis.daemon.protocol import CancelMessage
    from jarvis.daemon.server import ClientConnection, TaskSlot

    server = _server()
    slot = TaskSlot(task_id="t1", text="x", source="terminal", owner_id="oid")
    server._active = slot
    written: list[bytes] = []

    class _Writer:
        def write(self, data: bytes) -> None:
            written.append(data)

        async def drain(self) -> None:
            pass

    conn = ClientConnection(writer=_Writer(), reader=object())
    asyncio.run(server._handle_cancel(CancelMessage(task_id="t1"), conn))
    payload = json.loads(written[0].decode("utf-8"))
    assert payload["kind"] == "ack"
    assert payload["data"]["message"] == "cancelled"
