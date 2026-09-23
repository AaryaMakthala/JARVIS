"""Unit tests for daemon/task_runtime.py: the single user-answer window.

Covers ``timeout_answer`` and ``TaskRuntime.wait_for_response`` (the only place
the worker blocks on a slot's event) plus the wait-only ``TerminalSink``, so a
regression in the D1 timeout hand-off is caught here rather than in the daemon.
"""

from __future__ import annotations

import time
from threading import Event
from types import SimpleNamespace
from typing import Any

from jarvis.daemon.server import TaskSlot
from jarvis.daemon.task_runtime import (
    CONFIRMATION_TIMEOUT_S,
    TaskRuntime,
    TerminalSink,
    timeout_answer,
)


def _slot(resume_answer: Any = None, done: bool = False) -> TaskSlot:
    slot = TaskSlot(task_id="t1", text="hi", source="terminal")
    slot.resume_answer = resume_answer
    slot.done = done
    return slot


# ── timeout_answer ───────────────────────────────────────────────────────


class TestTimeoutAnswer:
    def test_carries_action_hash_back_for_a_confirmation(self) -> None:
        answer = timeout_answer({"action_hash": "a" * 64})
        assert answer["approved"] is False
        assert answer["timed_out"] is True
        assert answer["action_hash"] == "a" * 64

    def test_clarification_payload_gets_an_empty_hash(self) -> None:
        answer = timeout_answer({"question": "which file?", "type": "clarification"})
        assert answer["approved"] is False
        assert answer["timed_out"] is True
        assert answer["action_hash"] == ""

    def test_missing_hash_defaults_to_empty(self) -> None:
        answer = timeout_answer({})
        assert answer["action_hash"] == ""


# ── TaskRuntime.wait_for_response — the single window ────────────────────


class TestWaitForResponse:
    def test_default_window_is_30_seconds_single_sourced(self) -> None:
        assert CONFIRMATION_TIMEOUT_S == 30.0
        assert TaskRuntime().confirm_timeout_s == CONFIRMATION_TIMEOUT_S

    def test_returns_stashed_answer_once(self) -> None:
        slot = _slot(resume_answer={"approved": True, "action_hash": "h"})
        slot.event.set()
        runtime = TaskRuntime(confirm_timeout_s=5.0)
        answer = runtime.wait_for_response(slot, {"action_hash": "h"})
        assert answer == {"approved": True, "action_hash": "h"}
        assert slot.resume_answer is None  # consumed exactly once

    def test_timeout_returns_timeout_answer_and_keeps_fail_closed(self) -> None:
        slot = _slot()
        # 0-length window: never blocks, deterministic in CI.
        runtime = TaskRuntime(confirm_timeout_s=0.0)
        answer = runtime.wait_for_response(slot, {"action_hash": "h"})
        assert answer["timed_out"] is True
        assert answer["approved"] is False
        assert slot.resume_answer is None

    def test_wake_with_no_stashed_answer_is_a_timeout_not_approval(self) -> None:
        slot = _slot()
        slot.event.set()  # woken by something else; nothing was stashed
        runtime = TaskRuntime(confirm_timeout_s=0.0)
        answer = runtime.wait_for_response(slot, {"action_hash": "h"})
        assert answer["timed_out"] is True
        assert answer["approved"] is False

    def test_real_event_wake_round_trip(self) -> None:
        """A responder stashing the answer and setting the event is honoured."""
        slot = _slot()
        runtime = TaskRuntime(confirm_timeout_s=2.0)

        def responder() -> None:
            time.sleep(0.01)
            slot.resume_answer = {"approved": True, "action_hash": "abc"}
            slot.event.set()

        import threading

        thread = threading.Thread(target=responder)
        thread.start()
        answer = runtime.wait_for_response(slot, {"action_hash": "abc"})
        thread.join(timeout=5)
        assert answer == {"approved": True, "action_hash": "abc"}

    def test_minimum_window_clamped_to_zero(self) -> None:
        assert TaskRuntime(confirm_timeout_s=-5).confirm_timeout_s == 0.0


# ── TerminalSink is wait-only (publishing is the dispatch loop's job) ────


class TestTerminalSink:
    def test_confirms_through_the_runtime_window(self) -> None:
        slot = _slot(resume_answer={"approved": False, "action_hash": "h"})
        slot.event.set()
        sink = TerminalSink(TaskRuntime(confirm_timeout_s=5.0))
        assert sink.confirm(slot, {"action_hash": "h"}) == {
            "approved": False,
            "action_hash": "h",
        }

    def test_clarifies_through_the_runtime_window(self) -> None:
        slot = _slot(resume_answer="the report")
        slot.event.set()
        sink = TerminalSink(TaskRuntime(confirm_timeout_s=5.0))
        assert sink.clarify(slot, {"question": "which?"}) == "the report"

    def test_constructs_with_runtime_only(self) -> None:
        # Regression: _sink_for builds TerminalSink(runtime=...) and never a
        # publish/conn pair; anything else is a signature mismatch.
        sink = TerminalSink(TaskRuntime())
        assert sink._runtime is not None


# ── slot event lifecycle: wait_for_response never clears the event ───────


class TestEventNotClearedByRuntime:
    def test_answering_without_clearing_keeps_a_second_wait_live(self) -> None:
        """wait_for_response must not clear slot.event (the worker clears it
        BEFORE signalling dispatch, so an early answer can never be lost)."""
        event = Event()
        event.set()
        slot = SimpleNamespace(event=event, resume_answer={"approved": True, "action_hash": "h"})
        runtime = TaskRuntime(confirm_timeout_s=5.0)
        first = runtime.wait_for_response(slot, {"action_hash": "h"})
        assert first["approved"] is True
        # The event is untouched by wait_for_response; the worker's own clear
        # is the only consumer.
        assert event.is_set() is True
