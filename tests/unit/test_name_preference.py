"""Phase 5B: the user's preferred name is stated, learned, used and forgettable.

The detector is deterministic and conservative.  These tests cover the
persistent path as well as the parser: a wrong name must not come from reported,
quoted, hypothetical or negated text, and forgetting one preference must never
wipe the rest of memory.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from jarvis import memory_cli
from jarvis.agent import compose_rules
from jarvis.agent.compose_rules import spoken_answer
from jarvis.agent.context import make_app_context
from jarvis.agent.nodes.compose import compose
from jarvis.agent.nodes.memory_save import detect_name_statement, memory_save
from jarvis.agent.state import Plan, Step, StepResult
from jarvis.config import Settings
from jarvis.memory import SqliteMemory, open_db, open_memory
from jarvis.memory_cli import forget_preference

runner = CliRunner()


@pytest.fixture
def memory() -> SqliteMemory:
    store = SqliteMemory(open_db(":memory:"), settings=Settings())
    try:
        yield store
    finally:
        store.close()


def _ctx(store: Any, **settings_kwargs: Any):
    return make_app_context(Settings(**settings_kwargs), memory=store)


def _conversation(text: str, **extra: Any) -> dict[str, Any]:
    plan = Plan(kind="conversation", goal=text, steps=[])
    return {"user_input": text, "plan": plan, "results": [], **extra}


def _success_state(task_id: str = "1") -> dict[str, Any]:
    plan = Plan(
        kind="tool",
        goal="do the thing",
        steps=[Step(id="s1", tool="fake_echo", args={}, rationale="r", expect="e")],
    )
    return {
        "task_id": task_id,
        "plan": plan,
        "results": [
            StepResult(step_id="s1", ok=True, verified=True, output="did the thing")
        ],
    }


# ----------------------------------------------------------------- detection


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("My name is Aarya.", ("Aarya", True)),
        ("My name is Arya.", ("Arya", True)),
        ("Actually, call me Aarya instead.", ("Aarya", True)),
        ("Call me Bob.", ("Bob", True)),
        ("Call me Bob, please.", ("Bob", True)),
        ("Call me Arya.", ("Arya", True)),
        ("Call me, Anne-Marie.", ("Anne-Marie", True)),
        ("Call me Mary Jane.", ("Mary Jane", True)),
        ("my name is Mary Jane", ("Mary Jane", True)),
        ("my name is mary jane", ("mary jane", True)),
        ("My name is mary jane.", ("mary jane", True)),
        ("my name is O'Brien", ("O'Brien", True)),
        ("my name is Aarya for short", ("Aarya", True)),
        ("You can call me Bob", ("Bob", False)),
        ("I'm called Bob", ("Bob", False)),
        ("I\u2019m called \u00c9lodie", ("\u00c9lodie", False)),
        ('My name is "Bob".', ("Bob", True)),
    ],
)
def test_explicit_name_statements(text: str, expected: tuple[str, bool]) -> None:
    assert detect_name_statement(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        'He said "my name is Bob"',
        "He said my name is Bob",
        "The note says 'call me Bob'",
        "Suppose my name is Bob",
        "If my name is Bob, that would be odd",
        "What if I said, 'call me Bob'?",
        "Maybe my name is Bob",
        "Whether my name is Bob is unclear",
        "Don't call me Bob",
        "My name is not Bob",
        "Call me a taxi",
        "Call me example.com",
        "My name is Bob123",
        "He told me to call me Bob",
        "Call me in the morning",
        "Call me on my phone",
        "Call me after lunch",
        "Call me tonight",
        "Call me soon",
        "Call me around noon",
        "Call me before the meeting",
        "Call me when I get home",
        "Call me at 5",
        "Call me during the meeting",
        "Call me until tomorrow",
        "Call me again",
        "Call me quickly",
        "Call me yesterday",
        "Call me Bob again.",
        "Call me Bob quickly.",
        "Call me like Alice",
        "Call me promptly",
        "Call me Bob promptly",
        "Call me Bob, again.",
        "Call me Bob; quickly.",
        "Call me Bob - again.",
        "Call me Bob \u2014 again.",
        "Call me Bob, yesterday.",
        "Call me Bob, like Alice.",
        "Call me Bob (again).",
        "Call me Bob, promptly.",
        "My name is Mary Jane Watson",
    ],
)
def test_non_declarations_are_not_learned(text: str) -> None:
    assert detect_name_statement(text) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Call me Bob. You can call me Bobby.", ("Bob", True)),
        ("You can call me Bob, but my name is Alice.", ("Bob", False)),
        ("You can call me Bob. Call me Bobby later.", ("Bob", False)),
    ],
)
def test_earliest_statement_wins_regardless_of_pattern_order(
    text: str, expected: tuple[str, bool]
) -> None:
    assert detect_name_statement(text) == expected


# --------------------------------------------------------------- integration


def test_conversation_statement_stores_the_name(memory: SqliteMemory) -> None:
    out = memory_save(_conversation("Actually, call me Aarya instead."), _ctx(memory))
    assert out["memory_saved_reason"] == "name_saved"
    assert memory.preferences.get("user_name") == "Aarya"


def test_strong_statement_replaces_an_existing_name(memory: SqliteMemory) -> None:
    memory.preferences.set("user_name", "Aarya")
    out = memory_save(_conversation("My name is Bob"), _ctx(memory))
    assert out["memory_saved_reason"] == "name_saved"
    assert memory.preferences.get("user_name") == "Bob"


def test_weak_statement_cannot_replace_an_existing_name(memory: SqliteMemory) -> None:
    memory.preferences.set("user_name", "Aarya")
    out = memory_save(_conversation("You can call me Bob"), _ctx(memory))
    assert out["memory_saved_reason"] == "skipped_conversation"
    assert memory.preferences.get("user_name") == "Aarya"


def test_weak_statement_fills_an_empty_name_slot(memory: SqliteMemory) -> None:
    out = memory_save(_conversation("You can call me Bob"), _ctx(memory))
    assert out["memory_saved_reason"] == "name_saved"
    assert memory.preferences.get("user_name") == "Bob"


@pytest.mark.parametrize("initial", [None, "Aarya"])
def test_sensitive_turn_never_learns_or_overwrites_a_name(
    memory: SqliteMemory, monkeypatch: pytest.MonkeyPatch, initial: str | None
) -> None:
    """A credential-looking turn stores nothing, even with a valid name.

    Regression: ``_learn_name`` used to run after the sensitive outcome and
    could create or replace ``user_name``.  The recording wrapper proves the
    preference write method is not called, not merely that its value is benign.
    """
    if initial is not None:
        memory.preferences.set("user_name", initial)
    writes: list[tuple[str, str]] = []
    original_set = memory.preferences.set

    def recording_set(key: str, value: str) -> Any:
        writes.append((key, value))
        return original_set(key, value)

    monkeypatch.setattr(memory.preferences, "set", recording_set)
    text = "My name is Bob. password=hunter2"
    assert detect_name_statement(text) == ("Bob", True)

    out = memory_save(_conversation(text), _ctx(memory))

    assert out["memory_saved"] is False
    assert out["memory_saved_reason"] == "skipped_sensitive"
    assert out["memory_failure_logged"] is False
    assert memory.preferences.get("user_name") == initial
    assert writes == []


@pytest.mark.parametrize(
    "text",
    [
        "Call me in the morning",
        "Call me on my phone",
        "Call me after lunch",
        "Call me tonight",
        "Call me soon",
        "Call me at 5",
        "Call me during the meeting",
        "Call me until tomorrow",
        "Call me again",
        "Call me quickly",
        "Call me yesterday",
        "Call me Bob again.",
        "Call me Bob quickly.",
        "Call me like Alice",
        "Call me promptly",
        "Call me Bob promptly",
        "Call me Bob, again.",
        "Call me Bob; quickly.",
        "Call me Bob - again.",
        "Call me Bob \u2014 again.",
        "Call me Bob, yesterday.",
        "Call me Bob, like Alice.",
        "Call me Bob (again).",
        "Call me Bob, promptly.",
    ],
)
def test_ambiguous_statement_cannot_replace_an_existing_name(
    memory: SqliteMemory, text: str
) -> None:
    memory.preferences.set("user_name", "Aarya")
    out = memory_save(_conversation(text), _ctx(memory))
    stored = memory.preferences.get("user_name")
    assert out["memory_saved_reason"] == "skipped_conversation"
    assert stored == "Aarya"
    assert stored not in {"again", "quickly", "yesterday", "Bob", "Bob again", "Bob quickly"}


def test_three_word_name_is_rejected_instead_of_truncated(memory: SqliteMemory) -> None:
    memory.preferences.set("user_name", "Aarya")
    out = memory_save(_conversation("My name is Mary Jane Watson"), _ctx(memory))
    assert out["memory_saved_reason"] == "skipped_conversation"
    assert memory.preferences.get("user_name") == "Aarya"


def test_same_name_is_not_rewritten(memory: SqliteMemory) -> None:
    memory.preferences.set("user_name", "Aarya")
    out = memory_save(_conversation("My name is Aarya"), _ctx(memory))
    assert out["memory_saved_reason"] == "skipped_conversation"


def test_learn_name_can_be_switched_off(memory: SqliteMemory) -> None:
    ctx = _ctx(memory, memory={"learn_name": False})
    out = memory_save(_conversation("My name is Aarya"), ctx)
    assert out["memory_saved_reason"] == "skipped_conversation"
    assert memory.preferences.get("user_name") is None


def test_dry_run_does_not_learn_a_name(memory: SqliteMemory) -> None:
    ctx = _ctx(memory)
    ctx.dry_run = True
    memory_save(_conversation("My name is Aarya"), ctx)
    assert memory.preferences.get("user_name") is None


@pytest.mark.parametrize("field", ["cancelled", "halted_reason"])
def test_stopped_turns_do_not_learn_a_name(memory: SqliteMemory, field: str) -> None:
    value: Any = True if field == "cancelled" else "halted by policy"
    memory_save(_conversation("My name is Aarya", **{field: value}), _ctx(memory))
    assert memory.preferences.get("user_name") is None


def test_tool_turns_do_not_learn_a_name(memory: SqliteMemory) -> None:
    tool_plan = Plan(
        kind="tool",
        goal="open notepad",
        steps=[Step(id="s1", tool="open_app", args={}, rationale="r", expect="e")],
    )
    state = {
        "user_input": "My name is Aarya",
        "plan": tool_plan,
        "results": [],
    }
    memory_save(state, _ctx(memory))
    assert memory.preferences.get("user_name") is None


def test_name_survives_reopening_the_database(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    first = open_memory(path, settings=Settings())
    try:
        memory_save(_conversation("My name is Aarya"), _ctx(first))
    finally:
        first.close()

    again = open_memory(path, settings=Settings())
    try:
        assert again.preferences.get("user_name") == "Aarya"
    finally:
        again.close()


# --------------------------------------------------------------------- CLI


@contextmanager
def _yield(memory: Any) -> Iterator[Any]:
    yield memory


def _patch_memory_cli(monkeypatch: pytest.MonkeyPatch, memory: Any) -> None:
    monkeypatch.setattr(memory_cli, "_open_store", lambda: _yield(memory))


class _FailingPreferences:
    def delete(self, key: str) -> bool:
        del key
        raise RuntimeError("database is locked")


class _FailingMemory:
    skills = object()
    preferences = _FailingPreferences()


def test_forget_pref_removes_only_the_named_preference(
    memory: SqliteMemory, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory.preferences.set("user_name", "Aarya")
    memory.preferences.set("units", "metric")
    _patch_memory_cli(monkeypatch, memory)

    result = runner.invoke(memory_cli.memory_app, ["forget-pref", "user_name"])
    assert result.exit_code == 0, result.output
    assert memory.preferences.get("user_name") is None
    assert memory.preferences.get("units") == "metric"


def test_forget_pref_unknown_key_exits_1(
    memory: SqliteMemory, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_memory_cli(monkeypatch, memory)
    result = runner.invoke(memory_cli.memory_app, ["forget-pref", "missing"])
    assert result.exit_code == 1
    assert "no preference" in result.output


def test_forget_pref_invalid_key_reports_invalid_input(
    memory: SqliteMemory, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_memory_cli(monkeypatch, memory)
    result = runner.invoke(memory_cli.memory_app, ["forget-pref", "Bad/Key"])
    assert result.exit_code == 1
    assert "invalid preference key" in result.output
    assert "no preference" not in result.output


def test_forget_pref_storage_failure_exits_1_without_claiming_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_memory_cli(monkeypatch, _FailingMemory())
    result = runner.invoke(memory_cli.memory_app, ["forget-pref", "user_name"])
    assert result.exit_code == 1
    assert "could not delete preference" in result.output
    assert "deleted preference" not in result.output


def test_forget_preference_helper_reports_the_result(memory: SqliteMemory) -> None:
    memory.preferences.set("user_name", "Aarya")
    assert forget_preference(memory, "User_Name")[0] is True
    assert forget_preference(memory, "user_name")[0] is False


# ------------------------------------------------------------------- compose


def test_success_speech_can_use_the_stored_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(compose_rules, "_pick", lambda options, task_id: options[1])
    answer = spoken_answer({"task_id": "1"}, "success", address=", Aarya")
    assert answer == "That's done, Aarya."


def test_compose_uses_a_name_learned_on_the_previous_turn(
    memory: SqliteMemory, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _ctx(memory)
    memory_save(_conversation("My name is Aarya"), ctx)
    monkeypatch.setattr(compose_rules, "_pick", lambda options, task_id: options[1])

    out = compose(_success_state("task-2"), ctx)
    assert out["spoken_answer"] == "That's done, Aarya."
    assert out["screen_answer"] == "did the thing"


def test_compose_keeps_plain_speech_without_a_stored_name(
    memory: SqliteMemory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(compose_rules, "_pick", lambda options, task_id: options[1])
    out = compose(_success_state(), _ctx(memory))
    assert out["spoken_answer"] == "That's done."
    assert out["screen_answer"] == "did the thing"


def test_compose_ignores_a_credential_looking_stored_name(
    memory: SqliteMemory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        memory.preferences,
        "get",
        lambda key, default=None: "password=hunter2",
    )
    monkeypatch.setattr(compose_rules, "_pick", lambda options, task_id: options[1])
    out = compose(_success_state(), _ctx(memory))
    assert out["spoken_answer"] == "That's done."
    assert "password" not in out["spoken_answer"]


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (
            {
                "task_id": "failure",
                "plan": Plan(
                    kind="tool",
                    goal="do the thing",
                    steps=[Step(id="s1", tool="fake_echo", args={}, rationale="r")],
                ),
                "results": [StepResult(step_id="s1", ok=False, error="boom")],
            },
            "Could not complete the task: boom.",
        ),
        (
            {"task_id": "refusal", "halted_reason": "I did not perform the action."},
            "I did not perform the action.",
        ),
    ],
)
def test_compose_failure_and_refusal_stay_factual_and_unaddressed(
    memory: SqliteMemory, state: dict[str, Any], expected: str
) -> None:
    memory.preferences.set("user_name", "Aarya")
    out = compose(state, _ctx(memory))
    assert out["spoken_answer"] == expected
    assert out["screen_answer"] == expected
    assert "Aarya" not in out["spoken_answer"]


def test_compose_cancellation_and_clarification_remain_truthful(
    memory: SqliteMemory, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory.preferences.set("user_name", "Aarya")
    ctx = _ctx(memory)
    monkeypatch.setattr(compose_rules, "_pick", lambda options, task_id: options[1])

    cancelled = compose({"task_id": "cancel", "cancelled": True}, ctx)
    assert cancelled["spoken_answer"] == "Stopped, Aarya."
    assert cancelled["screen_answer"] == "Cancelled."

    clarify_plan = Plan(
        kind="tool",
        goal="open a file",
        steps=[],
        needs_clarification=True,
        clarification_question="Which file?",
    )
    clarified = compose(
        {"task_id": "clarify", "plan": clarify_plan, "results": []}, ctx
    )
    assert clarified["spoken_answer"] == "I need one more detail, Aarya: Which file?"
    assert clarified["screen_answer"] == "Clarification needed: Which file?"
