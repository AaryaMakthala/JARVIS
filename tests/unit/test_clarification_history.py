"""Stage 8.4e: every answered clarification round survives the next question.

Before this slice the graph kept only the *newest* question/answer pair, so a
second clarification erased what the human had already told the brain: the
re-plan then started from the original request alone ("send it to him" ->
"which file?" -> "the report" -> "which contact?" -> "Rahul" lost "the report").

These tests pin the three moving parts:

* :func:`jarvis.agent.state.clarification_rounds` normalises the raw history
  and drops malformed entries instead of raising (the history comes back from
  a checkpoint, so it is untrusted state).
* the ``clarify`` node appends every answered round, oldest first.
* the brain prompt renders *all* rounds and tells the model not to ask again
  about something the user already answered.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from jarvis.agent.context import make_app_context
from jarvis.agent.graph import open_sqlite_checkpointer
from jarvis.agent.prompts import brain_user_prompt
from jarvis.agent.runner import resume_task, run_task
from jarvis.agent.schemas import BrainDecision
from jarvis.agent.state import clarification_rounds
from jarvis.config import Settings
from jarvis.llm.client import Usage
from support import brain_action, brain_clarify, make_spec, registry_with


@pytest.fixture
def saver(tmp_path: Path) -> Any:
    conn = sqlite3.connect(tmp_path / "c.db", check_same_thread=False)
    yield open_sqlite_checkpointer(str(tmp_path / "c.db"))
    conn.close()


class _RecordingLLM:
    """A scripted LLM that keeps every user prompt it was handed."""

    def __init__(self, responses: list[BrainDecision]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    def structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[Any],
        model_role: str = "planner",
        temperature: float = 0.0,
        max_retries: int | None = None,
        timeout_s: float | None = None,
    ) -> tuple[BrainDecision, Usage]:
        assert issubclass(schema, BrainDecision)
        self.prompts.append(user)
        item = self._responses.pop(0)
        return item, Usage(calls=1, prompt_tokens=2, completion_tokens=2)


# ── the normaliser ─────────────────────────────────────────────────────────


def test_clarification_rounds_keeps_order_and_strips_whitespace() -> None:
    rounds = clarification_rounds(
        [
            {"question": "Which file?", "answer": " the report "},
            {"question": "Which contact?", "answer": "Rahul"},
        ]
    )
    assert rounds == [("Which file?", "the report"), ("Which contact?", "Rahul")]


def test_clarification_rounds_drops_malformed_entries() -> None:
    rounds = clarification_rounds(
        [
            {"question": "ok?", "answer": "yes"},
            {"question": "", "answer": "yes"},  # blank question
            {"question": "q", "answer": "  "},  # blank answer
            {"question": 7, "answer": "yes"},  # wrong type
            "not a dict",
            {"question": "q2", "answer": "a2"},
        ]
    )
    assert rounds == [("ok?", "yes"), ("q2", "a2")]


def test_clarification_rounds_tolerates_a_non_list_history() -> None:
    assert clarification_rounds(None) == []
    assert clarification_rounds("nope") == []
    assert clarification_rounds({"question": "q", "answer": "a"}) == []


# ── the prompt ─────────────────────────────────────────────────────────────


def test_prompt_renders_every_round_oldest_first() -> None:
    prompt = brain_user_prompt(
        "send it to him",
        clarification_history=[
            {"question": "Which file?", "answer": "the report"},
            {"question": "Which contact?", "answer": "Rahul"},
        ],
        clarification_count=2,
    )
    assert prompt.index("'the report'") < prompt.index("'Rahul'")
    assert "Which file?" in prompt and "Which contact?" in prompt
    # the single-pair fallback is not used when a history is present
    assert "You asked the user:\n" not in prompt


def test_prompt_forbids_re_asking_an_answered_question() -> None:
    prompt = brain_user_prompt(
        "send it to him",
        clarification_history=[{"question": "Which file?", "answer": "the report"}],
    )
    assert "Do not ask again about anything the user has already answered" in prompt


def test_prompt_without_history_keeps_the_single_pair_fallback() -> None:
    prompt = brain_user_prompt(
        "open the file",
        clarification_question="Which file did you mean?",
        clarification_answer="the report",
    )
    assert "You asked the user:" in prompt
    assert "the report" in prompt


# ── the graph ──────────────────────────────────────────────────────────────


def test_two_clarification_rounds_keep_both_answers(saver: Any) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    llm = _RecordingLLM(
        [
            brain_clarify("Which file did you mean?"),
            brain_clarify("Which contact did you mean?"),
            brain_action("fake_echo", {"text": "hi"}),
        ]
    )
    ctx = make_app_context(
        Settings(),
        llm=llm,
        registry=registry_with(make_spec("fake_echo", base_tier=0, record=record)),
    )

    first = run_task(ctx, saver, "send it to him")
    assert first.interrupt_kind == "clarification"

    second = resume_task(ctx, saver, first.task_id, "the report")
    assert second.interrupt_kind == "clarification"

    third = resume_task(ctx, saver, second.task_id, "Rahul")
    assert third.interrupted is False

    history = third.state.get("clarification_history")
    assert history == [
        {"question": "Which file did you mean?", "answer": "the report"},
        {"question": "Which contact did you mean?", "answer": "Rahul"},
    ]
    # the newest pair is still cleared once the brain has consumed it
    assert third.state["clarification_count"] == 2
    assert third.state["clarification_answer"] is None


def test_the_brain_sees_both_answers_on_the_second_round(saver: Any) -> None:
    llm = _RecordingLLM(
        [
            brain_clarify("Which file did you mean?"),
            brain_clarify("Which contact did you mean?"),
            brain_action("fake_echo", {"text": "hi"}),
        ]
    )
    ctx = make_app_context(
        Settings(),
        llm=llm,
        registry=registry_with(make_spec("fake_echo", base_tier=0, record=[])),
    )
    first = run_task(ctx, saver, "send it to him")
    second = resume_task(ctx, saver, first.task_id, "the report")
    resume_task(ctx, saver, second.task_id, "Rahul")

    # the last brain call is the one that planned the action: it must carry
    # both answers, not just the newest one
    final_prompt = llm.prompts[-1]
    assert "the report" in final_prompt
    assert "Rahul" in final_prompt
    assert "Do not ask again about anything the user has already answered" in final_prompt


def test_a_cancelled_clarification_records_no_round(saver: Any) -> None:
    ctx = make_app_context(
        Settings(),
        llm=_RecordingLLM([brain_clarify("Which file did you mean?")]),
        registry=registry_with(make_spec("fake_echo", base_tier=0, record=[])),
    )
    first = run_task(ctx, saver, "open the file")
    cancelled = resume_task(ctx, saver, first.task_id, "cancel")
    assert cancelled.state.get("clarification_history") == []
