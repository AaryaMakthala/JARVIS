"""4.17c: focus failures are non-retryable; other failures still retry.

Re-running ``type_text`` with the same target/window fails the same way, so
``failed to focus`` / ``focus verification failed`` skip ``retry_now`` and go
straight to the existing replan/fail-closed branch.  Unrelated failures keep
retrying up to ``max_retries_per_step`` exactly as before.
"""

from __future__ import annotations

from typing import Any

from jarvis.agent.context import make_app_context
from jarvis.agent.nodes.verify import verify
from jarvis.agent.state import StepResult
from jarvis.config import AgentSettings, Settings
from support import echo_plan, make_spec, registry_with

_FOCUS_ERRORS = (
    "failed to focus 'notepad': found_index is specified as 0, but 0 window/s found",
    "focus verification failed: foreground is 'Calculator', expected 'notepad'",
)


def _ctx(max_retries: int = 2, max_replans: int = 1) -> Any:
    record: list[tuple[str, dict[str, Any]]] = []
    spec = make_spec("fake_echo", base_tier=0, record=record, run_ok=False)
    return make_app_context(
        Settings(agent=AgentSettings(max_retries_per_step=max_retries, max_replans=max_replans)),
        llm=None,
        registry=registry_with(spec),
    )


def _state(error: str, retry_count: int = 0, replan_count: int = 0) -> dict[str, Any]:
    return {
        "plan": echo_plan(),
        "step_index": 0,
        "results": [StepResult(step_id="s1", ok=False, error=error, verified=True)],
        "retry_count": retry_count,
        "replan_count": replan_count,
    }


# ── focus failures are NOT retried ───────────────────────────────────────


def test_focus_error_skips_retry_and_goes_straight_to_replan() -> None:
    out = verify(_state(_FOCUS_ERRORS[0]), _ctx())
    assert out["retry_now"] is False, "re-running type_text would fail the same way"
    assert out["replan_now"] is True


def test_focus_verification_error_skips_retry() -> None:
    out = verify(_state(_FOCUS_ERRORS[1]), _ctx())
    assert out["retry_now"] is False
    assert out["replan_now"] is True


def test_focus_error_fails_closed_when_replan_budget_gone() -> None:
    out = verify(_state(_FOCUS_ERRORS[0], replan_count=1), _ctx(max_replans=1))
    assert out["retry_now"] is False
    assert out["replan_now"] is False, "budget gone -> fail closed, never a blind retry"


# ── unrelated failures still retry up to max_retries_per_step ────────────


def test_unrelated_failure_retries_up_to_max_then_replans() -> None:
    ctx = _ctx(max_retries=2, max_replans=1)
    out0 = verify(_state("boom"), ctx)
    assert out0["retry_now"] is True and out0["retry_count"] == 1
    out1 = verify(_state("boom", retry_count=1), ctx)
    assert out1["retry_now"] is True and out1["retry_count"] == 2
    out2 = verify(_state("boom", retry_count=2), ctx)
    assert out2["retry_now"] is False and out2["replan_now"] is True
