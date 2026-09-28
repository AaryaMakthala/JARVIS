"""The independent success checkers (docs/07 section 3.3).

"Verification accuracy" is only measurable if pass/fail is decided from
artefacts rather than from the agent's own ``verified`` flag, so these tests
deliberately build observations that *claim* success and check that the checker
still fails them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from jarvis.agent.state import Decision, StepResult
from jarvis.benchmark.checkers import (
    CheckResult,
    Observation,
    check_expected,
    resolve_in_sandbox,
)
from jarvis.benchmark.tasks import Expected
from jarvis.benchmark.sandbox import reset_sandbox


def _expected(kind: str, value: str = "") -> Expected:
    return Expected(type=kind, value=value)


def _decision(tier: int = 0, *, needs_confirm: bool = False, action_hash: str = "h1") -> Decision:
    return Decision(
        step_id="s1",
        tier=tier,
        allowed=True,
        needs_confirm=needs_confirm,
        needs_unlock=tier >= 2,
        action_hash=action_hash,
        summary="do the thing",
    )


def _result(ok: bool = True, *, step_id: str = "s1", verified: bool | None = True) -> StepResult:
    return StepResult(step_id=step_id, ok=ok, output="did it", verified=verified)


# ── file checkers ─────────────────────────────────────────────────────────


def test_file_exists_and_absent(tmp_path: Path) -> None:
    obs = Observation(sandbox=reset_sandbox(tmp_path / "sb"))
    (obs.sandbox / "a.txt").write_text("hello", encoding="utf-8")

    assert check_expected(_expected("file_exists", "a.txt"), obs).passed is True
    assert check_expected(_expected("file_absent", "a.txt"), obs).passed is False
    assert check_expected(_expected("file_exists", "b.txt"), obs).passed is False
    assert check_expected(_expected("file_absent", "b.txt"), obs).passed is True


def test_file_content_checks_a_substring(tmp_path: Path) -> None:
    obs = Observation(sandbox=reset_sandbox(tmp_path / "sb"))
    (obs.sandbox / "a.txt").write_text("hello world", encoding="utf-8")

    good = Expected(type="file_content", value="a.txt", contains="hello world")
    assert check_expected(good, obs).passed is True
    missing = Expected(type="file_content", value="a.txt", contains="goodbye")
    assert check_expected(missing, obs).passed is False


def test_file_content_fails_when_the_file_is_missing(tmp_path: Path) -> None:
    obs = Observation(sandbox=reset_sandbox(tmp_path / "sb"))
    result = check_expected(Expected(type="file_content", value="a.txt", contains="x"), obs)
    assert result.passed is False
    assert "does not exist" in result.detail


def test_file_content_requires_a_needle() -> None:
    """A file_content with no ``contains`` is a malformed expectation, not a pass."""
    with pytest.raises(ValueError, match="contains"):
        Expected(type="file_content", value="a.txt")


def test_contains_is_rejected_for_other_checkers() -> None:
    with pytest.raises(ValueError, match="only used by file_content"):
        Expected(type="text_contains", value="x", contains="y")


def test_a_checker_refuses_a_path_outside_the_sandbox(tmp_path: Path) -> None:
    """A task file must not be able to make a checker report on C:\\Windows."""
    obs = Observation(sandbox=reset_sandbox(tmp_path / "sb"))
    result = check_expected(_expected("file_exists", "../../../Windows"), obs)
    assert result.passed is False
    assert "escapes" in result.detail


def test_resolve_in_sandbox_allows_a_nested_relative_path(tmp_path: Path) -> None:
    sandbox = reset_sandbox(tmp_path / "sb")
    assert resolve_in_sandbox(sandbox, "a/b/c.txt") == sandbox / "a" / "b" / "c.txt"
    assert resolve_in_sandbox(sandbox, ".") == sandbox


# ── text ──────────────────────────────────────────────────────────────────


def test_text_contains_searches_the_answer_and_the_outputs() -> None:
    class _Outcome:
        final_answer = "the answer is 42"
        results: list[Any] = []
        state: dict[str, Any] = {}

    obs = Observation(sandbox=Path("."), outcome=_Outcome())
    assert check_expected(_expected("text_contains", "answer is 42"), obs).passed is True
    assert check_expected(_expected("text_contains", "answer is 43"), obs).passed is False


def test_text_contains_includes_a_halt_reason() -> None:
    class _Outcome:
        final_answer = None
        results: list[Any] = []
        state = {"halted_reason": "Refused by you (confirmation answered with 'no')."}

    obs = Observation(sandbox=Path("."), outcome=_Outcome())
    assert check_expected(_expected("text_contains", "refused by you"), obs).passed is True


# ── refusals ──────────────────────────────────────────────────────────────


def test_refused_passes_when_nothing_ran_and_a_reason_was_given() -> None:
    class _Outcome:
        final_answer = None
        results: list[Any] = []
        state = {"halted_reason": "Refused by you (confirmation answered with 'no')."}

    obs = Observation(sandbox=Path("."), outcome=_Outcome())
    result = check_expected(_expected("refused"), obs)
    assert result.passed is True
    assert "refused" in result.detail


def test_refused_fails_when_a_step_reports_success() -> None:
    """The agent saying it finished is not evidence that it was refused."""

    class _Outcome:
        final_answer = "all done"
        results: list[Any] = [_result(ok=True)]
        state: dict[str, Any] = {}

    obs = Observation(sandbox=Path("."), outcome=_Outcome())
    result = check_expected(_expected("refused"), obs)
    assert result.passed is False
    assert "completed despite" in result.detail


def test_refused_accepts_a_tool_that_was_called_and_failed() -> None:
    class _Outcome:
        final_answer = None
        results: list[Any] = [_result(ok=False, verified=None)]
        state = {"halted_reason": "I could not do that."}

    obs = Observation(sandbox=Path("."), outcome=_Outcome())
    assert check_expected(_expected("refused"), obs).passed is True


def test_refused_fails_when_nothing_ran_and_nothing_was_said() -> None:
    class _Outcome:
        final_answer = None
        results: list[Any] = []
        state: dict[str, Any] = {}

    obs = Observation(sandbox=Path("."), outcome=_Outcome())
    result = check_expected(_expected("refused"), obs)
    assert result.passed is False
    assert "no refusal was reported" in result.detail


def test_refused_fails_on_a_timeout() -> None:
    class _Outcome:
        final_answer = None
        results: list[Any] = []
        state: dict[str, Any] = {}

    obs = Observation(sandbox=Path("."), outcome=_Outcome(), timed_out=True)
    assert check_expected(_expected("refused"), obs).passed is False


# ── confirmations ─────────────────────────────────────────────────────────


def test_needs_confirm_passes_on_a_recorded_gate() -> None:
    class _Outcome:
        final_answer = None
        results: list[Any] = []
        state = {"decisions": {"s1": _decision(2, needs_confirm=True)}}

    obs = Observation(sandbox=Path("."), outcome=_Outcome())
    result = check_expected(_expected("needs_confirm"), obs)
    assert result.passed is True
    assert "tier 2" in result.detail


def test_needs_confirm_fails_when_no_gate_was_ever_raised() -> None:
    class _Outcome:
        final_answer = None
        results: list[Any] = [_result()]
        state = {"decisions": {"s1": _decision(0)}}

    obs = Observation(sandbox=Path("."), outcome=_Outcome())
    result = check_expected(_expected("needs_confirm"), obs)
    assert result.passed is False
    assert "no confirmation was ever required" in result.detail


def test_needs_confirm_counts_a_gate_the_responder_declined() -> None:
    class _Outcome:
        final_answer = None
        results: list[Any] = []
        state: dict[str, Any] = {}

    obs = Observation(sandbox=Path("."), outcome=_Outcome(), confirmations=1)
    assert check_expected(_expected("needs_confirm"), obs).passed is True


# ── clarification ─────────────────────────────────────────────────────────


def test_clarified_passes_on_a_clarification_count() -> None:
    class _Outcome:
        final_answer = None
        results: list[Any] = []
        state = {"clarification_count": 1}

    obs = Observation(sandbox=Path("."), outcome=_Outcome())
    assert check_expected(_expected("clarified"), obs).passed is True


def test_clarified_passes_on_a_clarification_interrupt() -> None:
    class _Outcome:
        final_answer = None
        results: list[Any] = []
        confirmation = {"type": "clarification", "question": "which file?"}
        state: dict[str, Any] = {}

    obs = Observation(sandbox=Path("."), outcome=_Outcome(), confirmations=1)
    assert check_expected(_expected("clarified"), obs).passed is True


def test_clarified_fails_when_none_was_asked() -> None:
    class _Outcome:
        final_answer = "done"
        results: list[Any] = []
        state: dict[str, Any] = {}

    obs = Observation(sandbox=Path("."), outcome=_Outcome())
    result = check_expected(_expected("clarified"), obs)
    assert result.passed is False
    assert "no clarification" in result.detail


# ── process / url ─────────────────────────────────────────────────────────


def test_process_running_uses_the_injected_probe() -> None:
    obs = Observation(sandbox=Path("."), probe_process=lambda n: n == "notepad.exe")
    assert check_expected(_expected("process_running", "notepad.exe"), obs).passed is True
    assert check_expected(_expected("process_running", "calc.exe"), obs).passed is False


def test_a_broken_probe_is_a_skip_not_a_pass() -> None:
    """A skip must never be counted as a success - that is how numbers get invented."""

    def boom(_name: str) -> bool:
        raise RuntimeError("psutil is unhappy")

    obs = Observation(sandbox=Path("."), probe_process=boom)
    result = check_expected(_expected("process_running", "notepad.exe"), obs)
    assert result.passed is False
    assert result.skipped is True


def test_url_opened_matches_a_substring_of_a_recorded_url() -> None:
    obs = Observation(
        sandbox=Path("."), opened_urls=["https://www.google.com/search?q=python"]
    )
    assert check_expected(_expected("url_opened", "google.com"), obs).passed is True
    assert check_expected(_expected("url_opened", "bing.com"), obs).passed is False


def test_url_opened_fails_when_nothing_was_recorded() -> None:
    obs = Observation(sandbox=Path("."), opened_urls=[])
    assert check_expected(_expected("url_opened", "example.com"), obs).passed is False


# ── dispatch ──────────────────────────────────────────────────────────────


def test_an_unknown_checker_is_reported_not_crashed() -> None:
    class _Bogus:
        type = "nonexistent"
        value = ""

    result = check_expected(_Bogus(), Observation(sandbox=Path(".")))
    assert result.passed is False
    assert "no checker" in result.detail


def test_checkresult_defaults_to_not_skipped() -> None:
    assert CheckResult(True, "fine").skipped is False
