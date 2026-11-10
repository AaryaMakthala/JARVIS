"""C5: printing an answer must never crash, and a finished session must exit.

The live acceptance run hit ``UnicodeEncodeError: 'charmap' codec`` on the
non-breaking hyphen (U+2011) in a web answer, and a piped ``jarvis run`` hung
because the client's event loop was never closed.  These tests pin both.
"""

from __future__ import annotations

import asyncio
import io
from typing import Any

import pytest
from rich.console import Console

from jarvis import cli
from jarvis.agent.runner import TaskOutcome
from jarvis.daemon import client as client_module
from jarvis.daemon.client import DaemonClient

NON_BREAKING_HYPHEN = "\u2011"


class _Stream:
    """Minimal text stream that records a ``reconfigure`` call."""

    def __init__(self, *, can_reconfigure: bool = True, fail: bool = False) -> None:
        self.can_reconfigure = can_reconfigure
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    def reconfigure(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)
        if self.fail:
            raise ValueError("cannot reconfigure")


def test_utf8_stdout_switches_encoding_and_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = _Stream()
    monkeypatch.setattr(cli.sys, "stdout", stream)

    assert cli._utf8_stdout() is stream
    assert stream.calls == [{"encoding": "utf-8", "errors": "replace"}]


def test_utf8_stdout_survives_a_stream_that_cannot_reconfigure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plain = _Stream(can_reconfigure=False)
    monkeypatch.setattr(cli.sys, "stdout", plain)
    assert cli._utf8_stdout() is plain

    failing = _Stream(fail=True)
    monkeypatch.setattr(cli.sys, "stdout", failing)
    assert cli._utf8_stdout() is failing


def test_a_non_breaking_hyphen_prints_on_a_narrow_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buffer = io.BytesIO()
    raw = io.TextIOWrapper(buffer, encoding="cp1252")
    monkeypatch.setattr(cli.sys, "stdout", raw)

    stream = cli._utf8_stdout()
    console = Console(file=stream, force_terminal=False, width=120)
    outcome = TaskOutcome(
        task_id="t1",
        interrupted=False,
        confirmation=None,
        final_answer="Open the control panel" + NON_BREAKING_HYPHEN + "style app.",
        error=None,
        halted_reason=None,
        decisions={},
        results=[],
    )

    console.print(outcome.final_answer)

    raw.flush()
    written = buffer.getvalue().decode("utf-8")
    assert "control panel" in written


def test_close_shuts_the_owned_loop() -> None:
    client_module._close_owned_loop()
    client = DaemonClient(port=1, token="t")
    loop = client_module._get_or_create_loop()
    assert client_module._OWNED_LOOP is loop

    client.close()

    assert loop.is_closed()
    assert client_module._OWNED_LOOP is None


def test_close_leaves_a_borrowed_loop_alone() -> None:
    client_module._close_owned_loop()
    borrowed = asyncio.new_event_loop()
    asyncio.set_event_loop(borrowed)
    try:
        client = DaemonClient(port=1, token="t")
        client.close()
        assert not borrowed.is_closed()
    finally:
        asyncio.set_event_loop(None)
        borrowed.close()
        client_module._OWNED_LOOP = None


def test_close_is_safe_twice_and_without_a_connection() -> None:
    client = DaemonClient(port=1, token="t")
    client.close()
    client.close()
