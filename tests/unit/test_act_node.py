"""Tests for the act node's exactly-once side-effect guard.

A non-idempotent step (``whatsapp_send``, ``file_delete``, ``type_text``) must
not run twice just because the graph re-entered the node: a resumed checkpoint,
a replan that proposes the same ``step_id``, or a provider fallback that re-asks
the LLM can all re-present a step that already had its effect.

The guard must also stay out of the way of the *designed* retry path: a step
whose verification failed is retryable, because a failed verify means the
post-condition did not hold, not that the action must never be attempted again.
"""

from __future__ import annotations

from typing import Any

import pytest

from jarvis.agent.context import make_app_context
from jarvis.agent.nodes.act import _recorded_result, act
from jarvis.agent.state import Plan, Step, StepResult
from jarvis.config import Settings
from support import make_spec, registry_with


def _ctx(specs: list[Any]) -> Any:
    return make_app_context(Settings(), registry=registry_with(*specs))


def _state(step_id: str = "s1", results: list[StepResult] | None = None) -> dict[str, Any]:
    return {
        "plan": Plan(
            goal="x", steps=[Step(id=step_id, tool="fake_echo", args={"text": "A"}, rationale="r")]
        ),
        "step_index": 0,
        "results": list(results or []),
    }


def _success(step_id: str = "s1", **kwargs: Any) -> StepResult:
    return StepResult(step_id=step_id, ok=True, output="sent", **kwargs)


class TestRecordedResultLookup:
    def test_returns_none_when_absent(self) -> None:
        assert _recorded_result({"results": []}, "s1") is None

    def test_returns_none_when_there_are_no_results(self) -> None:
        assert _recorded_result({}, "s1") is None

    def test_finds_the_matching_result(self) -> None:
        want = _success()
        assert _recorded_result({"results": [want]}, "s1") is want

    def test_takes_the_latest_match(self) -> None:
        """A later retry must not be masked by an earlier attempt."""
        first = StepResult(step_id="s1", ok=False, error="boom")
        second = _success()
        found = _recorded_result({"results": [first, second]}, "s1")
        assert found is second

    def test_ignores_other_step_ids(self) -> None:
        assert _recorded_result({"results": [_success("s2")]}, "s1") is None


class TestExactlyOnce:
    def test_a_successful_step_is_not_executed_again(self) -> None:
        record: list[tuple[str, dict[str, Any]]] = []
        ctx = _ctx([make_spec("fake_echo", base_tier=0, record=record)])

        first = act(_state(), ctx)
        assert record, "the first visit must execute the tool"
        assert len(first["results"]) == 1
        assert first["results"][0].ok is True

        # The node is re-entered with the same step_index and the recorded result.
        state = _state(results=first["results"])
        second = act(state, ctx)

        assert len(record) == 1, "the tool must not run a second time"
        assert "results" not in second, "no duplicate result may be appended"
        assert second["step_index"] == 1, "the step must be advanced past"
        assert second["retry_count"] == 0
        assert second["retry_now"] is False
        assert second["replan_now"] is False

    def test_three_re_entries_execute_once(self) -> None:
        record: list[tuple[str, dict[str, Any]]] = []
        ctx = _ctx([make_spec("fake_echo", base_tier=0, record=record)])
        state = _state()
        for _ in range(3):
            update = act(state, ctx)
            state = {**state, **update}
        assert len(record) == 1

    def test_a_different_step_id_is_executed(self) -> None:
        record: list[tuple[str, dict[str, Any]]] = []
        ctx = _ctx([make_spec("fake_echo", base_tier=0, record=record)])
        act(_state(results=[_success("s9")]), ctx)
        assert len(record) == 1


class TestFailuresStayRetryable:
    def test_a_failed_step_is_retried(self) -> None:
        """The guard must not swallow the designed retry path."""
        record: list[tuple[str, dict[str, Any]]] = []
        ctx = _ctx([make_spec("fake_echo", base_tier=0, record=record, run_ok=False)])

        first = act(_state(), ctx)
        assert first["results"][0].ok is False

        state = _state(results=first["results"])
        act(state, ctx)
        assert len(record) == 2, "a failed step must be retried"

    def test_act_does_not_run_verification_itself(self) -> None:
        """``ToolSpec.execute`` runs the action; ``run_verified`` verifies.

        The guard therefore reads ``verified`` from results written by a later
        node, and must treat the ``None`` produced here as "not yet verified"
        rather than as a failure.
        """
        record: list[tuple[str, dict[str, Any]]] = []
        ctx = _ctx([make_spec("fake_echo", base_tier=0, record=record, verify_ok=False)])
        update = act(_state(), ctx)
        assert update["results"][0].verified is None
        assert update["results"][0].ok is True
        # ...and the unverified result still short-circuits a re-entry.
        act(_state(results=update["results"]), ctx)
        assert len(record) == 1

    @pytest.mark.parametrize(
        "recorded",
        [
            StepResult(step_id="s1", ok=True, output="x", verified=False),
            StepResult(step_id="s1", ok=False, error="boom"),
        ],
    )
    def test_unsafe_recorded_results_do_not_short_circuit(self, recorded: StepResult) -> None:
        record: list[tuple[str, dict[str, Any]]] = []
        ctx = _ctx([make_spec("fake_echo", base_tier=0, record=record)])
        update = act(_state(results=[recorded]), ctx)
        assert len(record) == 1
        assert update.get("step_index") is None, "the guard must not have advanced"


class TestGuardIsNotAPolicyBypass:
    def test_policy_still_runs_for_a_fresh_step(self) -> None:
        """The guard is a de-duplicator, not a shortcut around the engine."""
        record: list[tuple[str, dict[str, Any]]] = []
        ctx = _ctx([make_spec("fake_echo", base_tier=3, record=record)])
        update = act(_state(), ctx)
        assert record == []
        assert update["results"][0].error == "blocked by policy"
        assert "halted_reason" in update

    def test_unknown_tool_is_still_rejected(self) -> None:
        ctx = _ctx([])
        state: dict[str, Any] = {
            "plan": Plan(goal="x", steps=[Step(id="s1", tool="nope", args={}, rationale="r")]),
            "step_index": 0,
            "results": [],
        }
        update = act(state, ctx)
        assert update["results"][0].error == "unknown tool"
