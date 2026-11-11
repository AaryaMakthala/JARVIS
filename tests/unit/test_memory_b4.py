"""B4: explicit, deterministic remember/forget memory commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from jarvis.agent.context import make_app_context
from jarvis.agent.graph import open_sqlite_checkpointer
from jarvis.agent.runner import run_task
from jarvis.config import Settings
from jarvis.llm.client import FakeLLM
from jarvis.memory import SqliteMemory, open_db, open_memory
from jarvis.memory.intent import detect_memory_intent
from support import brain_conversation, registry_with


def _close(saver: Any) -> None:
    closer = getattr(saver, "conn", None)
    if closer is not None:
        closer.close()


def _ctx(llm: Any, memory: Any, *, dry_run: bool = False):
    return make_app_context(
        Settings(),
        llm=llm,
        registry=registry_with(),
        memory=memory,
        dry_run=dry_run,
    )


@pytest.fixture
def store() -> SqliteMemory:
    memory = SqliteMemory(open_db(":memory:"), settings=Settings())
    try:
        yield memory
    finally:
        memory.close()


def _remember(store: SqliteMemory, text: str, *, dry_run: bool = False):
    llm = FakeLLM([])
    saver = open_sqlite_checkpointer(":memory:")
    try:
        outcome = run_task(_ctx(llm, store, dry_run=dry_run), saver, text)
    finally:
        _close(saver)
    return outcome, llm


# ---------------------------------------------------------------- parsing


def test_parser_recognises_clear_commands_and_ignores_questions() -> None:
    intent = detect_memory_intent("Please remember that my preferred coding language is Python.")
    assert intent is not None
    assert intent.valid
    assert intent.action == "remember"
    assert intent.content == "my preferred coding language is Python"
    assert intent.category == "preference"
    assert intent.key.startswith("fact.preference.")

    assert detect_memory_intent("What programming language do I prefer?") is None
    assert detect_memory_intent("Do you remember my birthday?") is None
    assert detect_memory_intent("I remember the Alamo.") is None


def test_forget_uses_the_same_key_as_remember() -> None:
    remembered = detect_memory_intent("Remember my preferred coding language is Python.")
    forgotten = detect_memory_intent("Forget my preferred coding language.")
    assert remembered is not None and forgotten is not None
    assert forgotten.action == "forget"
    assert forgotten.key == remembered.key


def test_parser_refuses_secrets_and_oversized_facts() -> None:
    secret = detect_memory_intent("Remember that my password is hunter2.")
    assert secret is not None
    assert secret.error == "sensitive"

    long = detect_memory_intent("Remember that " + "x" * 401)
    assert long is not None
    assert long.error == "too_long"


def test_a_slug_that_merely_contains_a_credential_hint_is_not_refused(
    store: SqliteMemory,
) -> None:
    outcome, _ = _remember(store, "Remember that my favorite shopping site is Amazon.")
    assert outcome.state["memory_saved_reason"] == "fact_saved"
    assert store.counts()["preferences"] == 1
    assert store.preferences.all()[0].key.startswith("fact.preference.fact_")


# ------------------------------------------------------------------- graph


def test_remember_stores_a_fact_without_calling_the_llm(store: SqliteMemory) -> None:
    outcome, llm = _remember(store, "Remember that my preferred coding language is Python.")

    assert llm.calls == []
    assert outcome.final_answer == "Noted, sir. I'll remember that."
    assert outcome.state["memory_saved"] is True
    assert outcome.state["memory_saved_reason"] == "fact_saved"
    assert store.counts()["preferences"] == 1
    stored = store.preferences.all()[0]
    assert stored.value == "my preferred coding language is Python"
    assert stored.category == "preference"
    assert stored.provenance == "user_explicit"


def test_a_later_question_retrieves_the_fact(store: SqliteMemory) -> None:
    _remember(store, "Remember that my preferred coding language is Python.")

    prompts: list[str] = []

    class SpyLLM(FakeLLM):
        def structured(self, *, system: str, user: str, **kwargs: Any) -> tuple[Any, Any]:
            prompts.append(user)
            return super().structured(system=system, user=user, **kwargs)

    saver = open_sqlite_checkpointer(":memory:")
    try:
        outcome = run_task(
            _ctx(SpyLLM([brain_conversation("You prefer Python.")]), store),
            saver,
            "What programming language do I prefer?",
        )
    finally:
        _close(saver)

    assert outcome.final_answer == "You prefer Python."
    assert prompts
    assert "Remembered fact (preference): my preferred coding language is Python" in prompts[0]


def test_update_overwrites_and_forget_deletes(store: SqliteMemory) -> None:
    _remember(store, "Remember that my preferred coding language is Python.")
    _remember(store, "Remember that my preferred coding language is Rust.")

    assert store.counts()["preferences"] == 1
    assert store.preferences.all()[0].value == "my preferred coding language is Rust"

    outcome, llm = _remember(store, "Forget that my preferred coding language is Python.")
    assert llm.calls == []
    assert outcome.final_answer == "I'll forget that, sir."
    assert outcome.state["memory_saved_reason"] == "fact_forgotten"
    assert store.counts()["preferences"] == 0

    outcome, _ = _remember(store, "Forget that my preferred coding language is Python.")
    assert outcome.final_answer == "I don't have that stored, sir."
    assert outcome.state["memory_saved_reason"] == "fact_not_found"


def test_a_pending_clarification_is_not_hijacked(store: SqliteMemory) -> None:
    from jarvis.agent.nodes.memory_save import memory_save
    from jarvis.agent.state import Plan

    state = {
        "user_input": "Remember that my preferred coding language is Python.",
        "plan": Plan(kind="conversation", goal="answer", steps=[]),
        "clarification_question": "Which project?",
    }
    out = memory_save(state, _ctx(FakeLLM([]), store))
    assert out["memory_saved_reason"] == "skipped_conversation"
    assert store.counts()["preferences"] == 0


def test_sensitive_and_oversized_facts_are_not_stored(store: SqliteMemory) -> None:
    outcome, llm = _remember(store, "Remember that my password is hunter2.")
    assert llm.calls == []
    assert "won't remember" in outcome.final_answer
    assert outcome.state["memory_saved_reason"] == "skipped_sensitive"
    assert store.counts()["preferences"] == 0

    outcome, llm = _remember(store, "Remember that " + "x" * 401)
    assert llm.calls == []
    assert "short facts" in outcome.final_answer
    assert outcome.state["memory_saved_reason"] == "skipped_too_long"
    assert store.counts()["preferences"] == 0


def test_dry_run_and_missing_backend_answer_honestly(store: SqliteMemory) -> None:
    outcome, _ = _remember(store, "Remember that my preferred editor is vim.", dry_run=True)
    assert "dry run" in outcome.final_answer
    assert store.counts()["preferences"] == 0

    saver = open_sqlite_checkpointer(":memory:")
    try:
        outcome = run_task(
            _ctx(FakeLLM([]), None),
            saver,
            "Remember that my preferred editor is vim.",
        )
    finally:
        _close(saver)
    assert "isn't available" in outcome.final_answer
    assert outcome.state["memory_saved_reason"] == "error"


def test_facts_survive_a_database_restart(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    first = open_memory(db_path, settings=Settings())
    assert isinstance(first, SqliteMemory)
    try:
        _remember(first, "Remember that my preferred coding language is Python.")
    finally:
        first.close()

    second = open_memory(db_path, settings=Settings())
    assert isinstance(second, SqliteMemory)
    try:
        intent = detect_memory_intent("Remember that my preferred coding language is Python.")
        assert intent is not None
        assert second.preferences.get(intent.key) == "my preferred coding language is Python"
    finally:
        second.close()


# --------------------------------------------------------------- retrieval


def test_an_ordinary_preference_does_not_reserve_a_skill_slot(store: SqliteMemory) -> None:
    store.preferences.set("units", "metric")
    store.skills.save("echo hello", plan={"kind": "tool", "goal": "echo hello", "steps": []})

    records = store.retrieve("echo hello", limit=1)
    assert records[0].meta.get("skill_id") is not None


def test_retrieval_reserves_a_slot_for_an_explicit_fact(store: SqliteMemory) -> None:
    intent = detect_memory_intent("Remember that my preferred coding language is Python.")
    assert intent is not None
    store.preferences.set(intent.key, intent.content)
    for i in range(5):
        store.skills.save(
            f"echo hello {i}",
            plan={"kind": "tool", "goal": f"echo hello {i}", "steps": []},
        )

    # B9: an explicit fact is still reserved, but only for a query that is
    # actually about it.  The skills match the same query, so the reservation
    # is what keeps the fact in the bounded result.
    records = store.retrieve("echo hello and my preferred coding language", limit=3)
    facts = [record for record in records if record.meta.get("fact")]
    assert len(facts) == 1
    assert facts[0].text == "Remembered fact (preference): my preferred coding language is Python"
    assert facts[0].meta["category"] == "preference"
    assert facts[0].meta["provenance"] == "user_explicit"
