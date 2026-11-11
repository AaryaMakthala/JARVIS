"""B4 acceptance: the canonical remember/ask/update/forget conversation pair.

Run with an isolated temporary database (never the developer's real
``memory.db``).  Each ``run_task`` call is a separate graph thread, so the
question genuinely arrives in a separate "conversation" from the remember
command, and the restart case re-opens the database from disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jarvis.agent.context import make_app_context
from jarvis.agent.graph import open_sqlite_checkpointer
from jarvis.agent.runner import run_task
from jarvis.config import Settings
from jarvis.llm.client import FakeLLM
from jarvis.memory import SqliteMemory, open_memory
from jarvis.memory.intent import detect_memory_intent
from support import brain_conversation, registry_with


def _ctx(llm: Any, memory: Any):
    return make_app_context(Settings(), llm=llm, registry=registry_with(), memory=memory)


def _turn(memory: SqliteMemory, text: str, llm: FakeLLM | None = None):
    saver = open_sqlite_checkpointer(":memory:")
    try:
        return run_task(_ctx(llm or FakeLLM([]), memory), saver, text)
    finally:
        closer = getattr(saver, "conn", None)
        if closer is not None:
            closer.close()


def _memory(path: Path) -> SqliteMemory:
    memory = open_memory(path, settings=Settings())
    assert isinstance(memory, SqliteMemory)
    return memory


def test_b4_acceptance_remember_ask_update_forget(tmp_path: Path) -> None:
    db = tmp_path / "acceptance-memory.db"
    memory = _memory(db)
    try:
        # Conversation A: remember the preference (no LLM call).
        llm = FakeLLM([])
        outcome = _turn(memory, "Remember that my preferred coding language is Python.", llm)
        assert llm.calls == []
        assert outcome.final_answer == "Noted, sir. I'll remember that."
        assert outcome.state["memory_saved_reason"] == "fact_saved"

        # Conversation B (a new graph thread): ask for it.
        prompts: list[str] = []

        class Spy(FakeLLM):
            def structured(self, *, system: str, user: str, **kwargs: Any) -> tuple[Any, Any]:
                prompts.append(user)
                return super().structured(system=system, user=user, **kwargs)

        outcome = _turn(
            memory,
            "What programming language do I prefer?",
            Spy([brain_conversation("You prefer Python.")]),
        )
        assert outcome.final_answer == "You prefer Python."
        assert "Remembered fact (preference): my preferred coding language is Python" in prompts[0]
        assert "<untrusted_data>" in prompts[0]

        # Update: same subject, new value - exactly one row, new value wins.
        _turn(memory, "Remember that my preferred coding language is Java.")
        assert memory.counts()["preferences"] == 1
        stored = memory.preferences.all()[0]
        assert stored.value == "my preferred coding language is Java"

        # The update is what a new conversation retrieves.
        outcome = _turn(
            memory,
            "What programming language do I prefer?",
            Spy([brain_conversation("You prefer Java.")]),
        )
        assert "my preferred coding language is Java" in prompts[-1]

        # Forget: removed from future retrieval.
        llm = FakeLLM([])
        outcome = _turn(memory, "Forget my preferred coding language.", llm)
        assert llm.calls == []
        assert outcome.final_answer == "I'll forget that, sir."
        assert memory.counts()["preferences"] == 0
        records = memory.retrieve("What programming language do I prefer?", limit=5)
        assert not [r for r in records if r.meta.get("fact")]

        # Forgetting again is honest, not an error.
        outcome = _turn(memory, "Forget my preferred coding language.", FakeLLM([]))
        assert outcome.final_answer == "I don't have that stored, sir."
    finally:
        memory.close()

    # Still gone after a genuine database close/reopen (daemon restart).
    restarted = _memory(db)
    try:
        intent = detect_memory_intent("Remember that my preferred coding language is Python.")
        assert intent is not None
        assert restarted.preferences.get(intent.key) is None
    finally:
        restarted.close()


def test_b4_acceptance_unrelated_preferences_are_not_returned_because_they_are_recent(
    tmp_path: Path,
) -> None:
    memory = _memory(tmp_path / "acceptance-memory.db")
    try:
        intent = detect_memory_intent("Remember that my preferred coding language is Python.")
        assert intent is not None
        memory.preferences.set(intent.key, intent.content)
        # A recent ordinary preference about something else entirely.
        memory.preferences.set("units", "metric")

        records = memory.retrieve("What programming language do I prefer?", limit=5)
        facts = [r for r in records if r.meta.get("fact")]
        assert [r.text for r in facts] == [
            "Remembered fact (preference): my preferred coding language is Python"
        ]
    finally:
        memory.close()


def test_b4_acceptance_ordinary_conversation_creates_no_preference(tmp_path: Path) -> None:
    memory = _memory(tmp_path / "acceptance-memory.db")
    try:
        llm = FakeLLM([brain_conversation("I like talking about programming languages.")])
        outcome = _turn(memory, "I enjoy programming languages a lot.", llm)
        assert outcome.final_answer == "I like talking about programming languages."
        assert memory.counts()["preferences"] == 0
        assert outcome.state["memory_saved_reason"] != "fact_saved"
    finally:
        memory.close()
