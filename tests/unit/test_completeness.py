"""Stage 5a completeness check (ROADMAP V2.2): graph-level behaviour tests.

One brain call decides goal / what is provided / what is missing / executability
(D2: no extra LLM request).  These tests drive the real graph + real SQLite
checkpointer with a scripted FakeLLM:

* a fully specified request asks nothing and runs;
* an optional gap proceeds only with a stated assumption that reaches the final
  answer (carried as ``Plan.assumption``, surfaced by ``respond``);
* a gap with no stated assumption is rejected fail-closed (silent invention);
* the clarification budget is capped at two rounds and the stop message names
  the still-missing items;
* a perception-derived step keeps its taint flag through projection and marks
  the policy decision (never silently becomes a clean value);
* the transient ``BrainDecision.assumption`` / ``missing`` fields never appear in
  the checkpointed state.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from jarvis.agent.context import make_app_context
from jarvis.agent.graph import open_sqlite_checkpointer
from jarvis.agent.runner import resume_task, run_task
from jarvis.agent.schemas import ActionIntent, BrainDecision
from jarvis.config import Settings
from jarvis.llm.client import FakeLLM
from support import make_spec, registry_with


@pytest.fixture
def saver(tmp_path: Path) -> Any:
    conn = sqlite3.connect(tmp_path / "c.db", check_same_thread=False)
    yield open_sqlite_checkpointer(str(tmp_path / "c.db"))
    conn.close()


def _ctx(record: list[tuple[str, dict[str, Any]]], *decisions: Any) -> Any:
    return make_app_context(
        Settings(),
        llm=FakeLLM(list(decisions)),
        registry=registry_with(make_spec("fake_echo", base_tier=0, record=record)),
    )


def _action(
    assumption: str = "",
    missing: list[str] | None = None,
    *,
    untrusted: bool = False,
) -> BrainDecision:
    return BrainDecision(
        request_type="action",
        goal="act on the request",
        assumption=assumption,
        missing=list(missing or []),
        actions=[
            ActionIntent(
                tool="fake_echo",
                args={"text": "hi"},
                rationale="echo it",
                depends_on_untrusted=untrusted,
            )
        ],
    )


def _clarify(question: str, missing: list[str] | None = None) -> BrainDecision:
    return BrainDecision(
        request_type="clarification",
        clarification_question=question,
        missing=list(missing or []),
    )


def test_fully_specified_action_asks_nothing_and_runs(saver: Any) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    out = run_task(_ctx(record, _action()), saver, "echo hi")
    assert out.interrupted is False
    assert record == [("fake_echo", {"text": "hi"})]
    assert "Assumption:" not in (out.final_answer or "")


def test_optional_gap_proceeds_and_the_assumption_reaches_the_user(saver: Any) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    out = run_task(
        _ctx(
            record, _action(assumption="the file is in the Downloads folder", missing=["location"])
        ),
        saver,
        "open the saved report",
    )
    assert record == [("fake_echo", {"text": "hi"})]
    assert out.final_answer == ("Assumption: the file is in the Downloads folder. fake_echo ran hi")
    plan = out.state["plan"]
    assert plan is not None and plan.assumption == "the file is in the Downloads folder"


def test_gap_without_a_stated_assumption_is_rejected_fail_closed(saver: Any) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    out = run_task(_ctx(record, _action(missing=["the file path"])), saver, "open the saved report")
    assert record == []
    assert out.interrupted is False
    assert out.state["request_kind"] == "unsupported"
    assert "no assumption was stated" in (out.final_answer or "")


def test_required_gap_triggers_a_clarification_and_runs_nothing(saver: Any) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    out = run_task(_ctx(record, _clarify("Which report did you mean?")), saver, "open the report")
    assert record == []
    assert out.interrupted is True
    assert out.interrupt_kind == "clarification"
    assert (
        out.confirmation is not None
        and out.confirmation["question"] == "Which report did you mean?"
    )


def test_never_more_than_two_rounds_and_the_stop_names_missing(saver: Any) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    ctx = _ctx(
        record,
        _clarify("Which folder?", ["the exact folder"]),
        _clarify("Which folder?", ["the exact folder"]),
        _clarify("Which folder?", ["the exact folder"]),
    )
    first = run_task(ctx, saver, "empty the folder")
    assert first.interrupted is True  # round 1
    second = resume_task(ctx, saver, first.task_id, "the reports folder")
    assert second.interrupted is True  # round 2
    third = resume_task(ctx, saver, second.task_id, "the reports folder")
    assert third.interrupted is False  # budget exhausted -> deterministic stop
    assert record == []
    assert third.state["clarification_count"] == 2
    assert "Still missing: the exact folder." in (third.final_answer or "")
    assert "2 question(s)" in (third.final_answer or "")


def test_perception_flag_survives_projection_and_marks_the_decision(saver: Any) -> None:
    """A value derived from page/screen text stays flagged (I1), deterministically."""
    record: list[tuple[str, dict[str, Any]]] = []
    out = run_task(_ctx(record, _action(untrusted=True)), saver, "use what the page says")
    step = out.state["plan"].steps[0]  # type: ignore[union-attr]
    assert step.depends_on_untrusted is True
    assert out.decisions["s1"].warn_untrusted is True


def test_brain_decision_not_checkpointed_and_state_has_no_assumption_or_missing(
    saver: Any,
) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    out = run_task(
        _ctx(
            record, _action(assumption="the file is in the Downloads folder", missing=["location"])
        ),
        saver,
        "open the saved report",
    )
    persisted = out.state
    assert record == [("fake_echo", {"text": "hi"})]
    # the transient fields never become state keys / new state types
    assert "assumption" not in persisted
    assert "missing" not in persisted
    # no BrainDecision object was ever checkpointed
    assert not any(isinstance(v, BrainDecision) for v in persisted.values())
    # the only place the assumption text lives is the allowlisted Plan field
    assert persisted["plan"].assumption == "the file is in the Downloads folder"
    assert type(persisted["plan"]).__name__ == "Plan"
