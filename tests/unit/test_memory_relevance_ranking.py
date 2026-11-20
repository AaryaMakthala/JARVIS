"""P6: relevance-ranked preference/fact recall in the real memory flow."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from jarvis.agent.context import make_app_context
from jarvis.agent.graph import open_sqlite_checkpointer
from jarvis.agent.nodes.memory_retrieve import _select
from jarvis.agent.runner import run_task
from jarvis.config import Settings
from jarvis.llm.client import FakeLLM
from jarvis.memory import SqliteMemory, open_memory
from jarvis.memory.base import MemoryRecord
from jarvis.memory.intent import detect_memory_intent
from support import brain_conversation, registry_with

_PYTHON_FACT = "Remembered fact (preference): my preferred coding language is Python"
_JAVA_FACT = "Remembered fact (personal): my coding language is Java"


def _memory(path: Path) -> SqliteMemory:
    memory = open_memory(path, settings=Settings())
    assert isinstance(memory, SqliteMemory)
    return memory


def _remember(memory: SqliteMemory, statement: str) -> None:
    intent = detect_memory_intent(statement)
    assert intent is not None
    memory.preferences.set(intent.key, intent.content)


def test_best_match_outranks_a_newer_partial_match_after_restart(tmp_path: Path) -> None:
    db = tmp_path / "ranked-memory.db"
    first = _memory(db)
    try:
        _remember(first, "Remember that my preferred coding language is Python.")
        _remember(first, "Remember that my coding language is Java.")
    finally:
        first.close()

    second = _memory(db)
    try:
        records = second.retrieve("What is my preferred coding language?", limit=2)
        assert [record.text for record in records] == [_PYTHON_FACT, _JAVA_FACT]
        assert [record.meta["relevance"] for record in records] == [1.0, 0.667]
    finally:
        second.close()


def test_verbose_request_does_not_dilute_an_exact_fact_match(tmp_path: Path) -> None:
    memory = _memory(tmp_path / "verbose-memory.db")
    try:
        _remember(memory, "Remember that my preferred coding language is Python.")
        records = memory.retrieve(
            "Please tell me what my preferred coding language is today and then open the browser",
            limit=3,
        )
        assert [record.text for record in records] == [_PYTHON_FACT]
        assert records[0].meta["fact"] is True
    finally:
        memory.close()


def test_unrelated_request_receives_no_preference(tmp_path: Path) -> None:
    memory = _memory(tmp_path / "unrelated-memory.db")
    try:
        memory.preferences.set("units", "metric")
        memory.preferences.set("search_engine", "duckduckgo")
        assert memory.retrieve("What is the weather in London?", limit=5) == []
    finally:
        memory.close()


def test_updated_preference_replaces_the_old_value_in_ranked_retrieval(tmp_path: Path) -> None:
    memory = _memory(tmp_path / "update-memory.db")
    try:
        intent = detect_memory_intent("Remember that my preferred coding language is Python.")
        assert intent is not None
        memory.preferences.set(intent.key, intent.content)
        memory.preferences.set(intent.key, "my preferred coding language is Rust")

        records = memory.retrieve("What is my preferred coding language?", limit=5)
        assert [record.text for record in records] == [
            "Remembered fact (preference): my preferred coding language is Rust"
        ]
        assert memory.counts()["preferences"] == 1
    finally:
        memory.close()


def test_retrieval_stays_within_the_configured_limit(tmp_path: Path) -> None:
    memory = _memory(tmp_path / "bounded-memory.db")
    try:
        for index in range(8):
            memory.preferences.set(f"pref_{index}", f"python coding language {index}")
        records = memory.retrieve("What is my preferred coding language?", limit=3)
        assert len(records) == 3
        assert all(record.meta["relevance"] >= 0.5 for record in records)
    finally:
        memory.close()


def test_sensitive_fact_is_still_refused_by_the_persistence_path(tmp_path: Path) -> None:
    memory = _memory(tmp_path / "private-memory.db")
    try:
        with pytest.raises(ValueError, match="credential"):
            memory.preferences.set("api_key", "sk-live-abc123")
        assert memory.counts()["preferences"] == 0
    finally:
        memory.close()


def test_oversized_item_does_not_hide_a_smaller_relevant_item() -> None:
    settings = SimpleNamespace(item_max_chars=80, context_max_chars=90, failure_max_age_days=30)
    records = [
        MemoryRecord(kind="preference", text="x" * 80),
        MemoryRecord(kind="preference", text="usable", meta={"relevance": 0.9}),
    ]
    assert [item["text"] for item in _select(records, settings)] == ["usable"]


def test_ranked_fact_reaches_a_new_conversation_prompt(tmp_path: Path) -> None:
    memory = _memory(tmp_path / "conversation-memory.db")
    saver = open_sqlite_checkpointer(":memory:")
    prompts: list[str] = []

    class Spy(FakeLLM):
        def structured(self, *, system: str, user: str, **kwargs: Any) -> tuple[Any, Any]:
            prompts.append(user)
            return super().structured(system=system, user=user, **kwargs)

    try:
        _remember(memory, "Remember that my preferred coding language is Python.")
        outcome = run_task(
            make_app_context(
                Settings(),
                llm=Spy([brain_conversation("You prefer Python.")]),
                registry=registry_with(),
                memory=memory,
            ),
            saver,
            "Please tell me what my preferred coding language is today and then open the browser",
        )
    finally:
        closer = getattr(saver, "conn", None)
        if closer is not None:
            closer.close()
        memory.close()

    assert outcome.final_answer == "You prefer Python."
    assert prompts
    assert _PYTHON_FACT in prompts[0]
    assert "<untrusted_data>" in prompts[0]
