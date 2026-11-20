"""P7: alias-aware fact identity and conservative supersession."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jarvis.agent.context import make_app_context
from jarvis.agent.graph import open_sqlite_checkpointer
from jarvis.agent.runner import run_task
from jarvis.config import Settings
from jarvis.llm.client import FakeLLM
from jarvis.memory import SqliteMemory, open_memory
from jarvis.memory.aliases import fact_key_for_topic, topic_from_text, topic_from_value
from jarvis.memory.correction import detect_correction
from jarvis.memory.intent import detect_memory_intent
from jarvis.memory.prefs import SUPERSEDED_PREFIX, is_superseded
from support import brain_conversation, registry_with

_PYTHON_FACT = "Remembered fact (preference): my preferred coding language is Python"
_CANONICAL_PYTHON_FACT = "Remembered fact (preference): my preferred programming language is Python"
_JAVA_FACT = "Remembered fact (personal): my coding language is Java"


def _memory(path: Path) -> SqliteMemory:
    memory = open_memory(path, settings=Settings())
    assert isinstance(memory, SqliteMemory)
    return memory


def _turn(memory: SqliteMemory, text: str, llm: FakeLLM | None = None) -> Any:
    saver = open_sqlite_checkpointer(":memory:")
    try:
        return run_task(
            make_app_context(
                Settings(),
                llm=llm or FakeLLM([]),
                registry=registry_with(),
                memory=memory,
            ),
            saver,
            text,
        )
    finally:
        closer = getattr(saver, "conn", None)
        if closer is not None:
            closer.close()


def _remember(memory: SqliteMemory, statement: str) -> None:
    intent = detect_memory_intent(statement)
    assert intent is not None
    memory.preferences.set(intent.key, intent.content)


def _correct(memory: SqliteMemory, statement: str) -> Any:
    return _turn(memory, statement, FakeLLM([brain_conversation("Understood.")]))


def _active_values(memory: SqliteMemory) -> list[str]:
    return [pref.value for pref in memory.preferences.active(limit=50)]


def test_aliases_and_known_values_resolve_without_keyword_guessing() -> None:
    assert topic_from_text("my preferred coding language") == "programming_language"
    assert topic_from_text("programming language I prefer") == "programming_language"
    assert topic_from_text("current operating system") == "operating_system"
    assert topic_from_text("preferred response language") == "response_language"
    assert topic_from_value("PyCharm") == "code_editor"
    assert topic_from_value("Java") == "programming_language"
    assert topic_from_text("response language and code editor") == "response_language"
    assert topic_from_value("a long sentence about many things") is None


def test_correction_parser_ignores_additional_and_vague_statements() -> None:
    assert detect_correction("I also use Linux at home.") is None
    assert detect_correction("I prefer Python.") is None
    vague = detect_correction("My preference has changed.")
    assert vague is not None and vague.error == "ambiguous"
    explicit = detect_correction("Actually, I prefer Python now.")
    assert explicit is not None and explicit.value == "Python" and explicit.valid


def test_rephrased_correction_supersedes_the_old_active_value(tmp_path: Path) -> None:
    memory = _memory(tmp_path / "language.db")
    try:
        _remember(memory, "Remember that my preferred coding language is Java.")
        outcome = _correct(memory, "Actually, I prefer Python now.")

        assert outcome.final_answer == "Understood."
        assert outcome.state["memory_saved_reason"] == "fact_superseded"
        assert _active_values(memory) == ["my preferred programming language is Python"]
        assert (
            memory.preferences.get(fact_key_for_topic("programming_language"))
            == "my preferred programming language is Python"
        )

        history = [p for p in memory.preferences.all() if is_superseded(p.key)]
        assert [p.value for p in history] == ["my preferred coding language is Java"]

        records = memory.retrieve("What is my preferred coding language?", limit=5)
        assert [r.text for r in records] == [_CANONICAL_PYTHON_FACT]
    finally:
        memory.close()


def test_switched_editor_supersedes_an_existing_editor_fact(tmp_path: Path) -> None:
    memory = _memory(tmp_path / "editor.db")
    try:
        _remember(memory, "Remember that I use VS Code.")
        outcome = _correct(memory, "I have switched to PyCharm.")

        assert outcome.state["memory_saved_reason"] == "fact_superseded"
        assert _active_values(memory) == ["my preferred code editor is PyCharm"]
        assert (
            memory.preferences.get(fact_key_for_topic("code_editor"))
            == "my preferred code editor is PyCharm"
        )
    finally:
        memory.close()


def test_response_language_correction_uses_the_explicit_alias(tmp_path: Path) -> None:
    memory = _memory(tmp_path / "response-language.db")
    try:
        _remember(memory, "Remember that my preferred response language is English.")
        outcome = _correct(memory, "I want responses in Hindi now.")

        assert outcome.state["memory_saved_reason"] == "fact_superseded"
        assert _active_values(memory) == ["my preferred response language is Hindi"]
        assert (
            memory.preferences.get(fact_key_for_topic("response_language"))
            == "my preferred response language is Hindi"
        )
    finally:
        memory.close()


def test_unrelated_fact_does_not_replace_an_existing_preference(tmp_path: Path) -> None:
    memory = _memory(tmp_path / "unrelated.db")
    try:
        _remember(memory, "Remember that my preferred coding language is Java.")
        outcome = _correct(memory, "I use Windows now.")

        assert outcome.state["memory_saved_reason"] == "fact_unchanged"
        assert _active_values(memory) == ["my preferred coding language is Java"]
    finally:
        memory.close()


def test_additional_preference_does_not_supersede_the_existing_one(tmp_path: Path) -> None:
    memory = _memory(tmp_path / "additional.db")
    try:
        _remember(memory, "Remember that my preferred coding language is Java.")
        outcome = _correct(memory, "I also use Linux at home.")
        _remember(memory, "Remember that I also speak French.")

        assert outcome.state["memory_saved_reason"] != "fact_superseded"
        assert "my preferred coding language is Java" in _active_values(memory)
        assert "I also speak French" in _active_values(memory)
    finally:
        memory.close()


def test_rephrased_duplicates_collapse_to_one_active_fact(tmp_path: Path) -> None:
    memory = _memory(tmp_path / "duplicates.db")
    try:
        _remember(memory, "Remember that my preferred coding language is Java.")
        _remember(memory, "Remember that my coding language is Java.")
        assert len(_active_values(memory)) == 2

        outcome = _correct(memory, "Actually, I prefer Python now.")
        assert outcome.state["memory_saved_reason"] == "fact_superseded"
        assert _active_values(memory) == ["my preferred programming language is Python"]
        archived = [p for p in memory.preferences.all() if p.key.startswith(SUPERSEDED_PREFIX)]
        assert len(archived) == 2
    finally:
        memory.close()


def test_removal_correction_archives_without_a_replacement(tmp_path: Path) -> None:
    memory = _memory(tmp_path / "removal.db")
    try:
        _remember(memory, "Remember that my preferred coding language is Java.")
        outcome = _correct(memory, "I no longer use Java.")

        assert outcome.state["memory_saved_reason"] == "fact_superseded"
        assert _active_values(memory) == []
        assert memory.retrieve("What is my preferred coding language?", limit=5) == []
        assert len([p for p in memory.preferences.all() if is_superseded(p.key)]) == 1
    finally:
        memory.close()


def test_ambiguous_correction_is_not_written(tmp_path: Path) -> None:
    memory = _memory(tmp_path / "ambiguous.db")
    try:
        _remember(memory, "Remember that my preferred coding language is Java.")
        outcome = _correct(memory, "My preference has changed.")

        assert outcome.state["memory_saved_reason"] == "fact_ambiguous"
        assert _active_values(memory) == ["my preferred coding language is Java"]
    finally:
        memory.close()


def test_sensitive_correction_is_refused_by_the_shared_guard(tmp_path: Path) -> None:
    memory = _memory(tmp_path / "sensitive.db")
    try:
        _remember(memory, "Remember that my preferred coding language is Java.")
        outcome = _correct(memory, "Actually, I prefer sk-live-abc123 now.")

        assert outcome.state["memory_saved_reason"] == "skipped_sensitive"
        assert _active_values(memory) == ["my preferred coding language is Java"]
    finally:
        memory.close()


def test_supersession_and_history_survive_reopen(tmp_path: Path) -> None:
    db = tmp_path / "restart.db"
    first = _memory(db)
    try:
        _remember(first, "Remember that my preferred coding language is Java.")
        _correct(first, "Actually, I prefer Python now.")
    finally:
        first.close()

    second = _memory(db)
    try:
        assert _active_values(second) == ["my preferred programming language is Python"]
        assert [r.text for r in second.retrieve("What language do I prefer?", limit=5)] == [
            _CANONICAL_PYTHON_FACT
        ]
        assert len([p for p in second.preferences.all() if is_superseded(p.key)]) == 1
    finally:
        second.close()


def test_p6_relevance_ranking_remains_intact(tmp_path: Path) -> None:
    memory = _memory(tmp_path / "ranking.db")
    try:
        _remember(memory, "Remember that my preferred coding language is Python.")
        _remember(memory, "Remember that my coding language is Java.")
        records = memory.retrieve("What is my preferred coding language?", limit=2)

        assert [r.text for r in records] == [_PYTHON_FACT, _JAVA_FACT]
        assert [r.meta["relevance"] for r in records] == [1.0, 0.667]
    finally:
        memory.close()
