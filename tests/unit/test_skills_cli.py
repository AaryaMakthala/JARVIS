"""`jarvis skills list|show|delete|clear` (Phase 8).

Hermetic: the memory store is pinned to a temp SQLite file by monkeypatching
``cli._open_memory_store``, so no test touches the real ``memory.db``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from jarvis import cli
from jarvis.config import Settings
from jarvis.memory import SqliteMemory, open_db

runner = CliRunner()


@pytest.fixture
def memory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A real SQLite memory store for assertions, plus a per-command factory.

    The CLI opens and closes its **own** connection per invocation (that is
    what the real product does, and what ``TestConnectionsAreClosed`` checks), so
    the fixture cannot hand the same live handle to both the test and the
    command.  Both point at the same temp file, so rows written by one are
    visible to the other.
    """
    db = tmp_path / "memory.db"
    store = SqliteMemory(open_db(db), settings=Settings())
    handed_out: list[SqliteMemory] = []

    def factory(*_args: object, **_kwargs: object) -> SqliteMemory:
        backend = SqliteMemory(open_db(db), settings=Settings())
        handed_out.append(backend)
        return backend

    monkeypatch.setattr(cli, "_open_memory_store", factory)
    try:
        yield store
    finally:
        store.close()
        for backend in handed_out:
            backend.close()


def _skill(store: SqliteMemory, goal: str = "delete the downloads folder") -> int:
    saved = store.skills.save(
        goal,
        plan={
            "kind": "tool",
            "goal": goal,
            "steps": [
                {"id": "s1", "tool": "delete_path", "args": {"paths": ["C:/x"]}, "expect": "gone"}
            ],
        },
        tools_used=["delete_path"],
    )
    assert saved.saved
    return saved.skill.id


class TestSkillsList:
    def test_empty_store_says_so(self, memory: SqliteMemory) -> None:
        result = runner.invoke(cli.app, ["skills", "list"])
        assert result.exit_code == 0, result.output
        assert "no skills yet" in result.output

    def test_lists_a_stored_skill(self, memory: SqliteMemory) -> None:
        _skill(memory)
        result = runner.invoke(cli.app, ["skills", "list"])
        assert result.exit_code == 0, result.output
        assert "delete the downloads folder" in result.output
        assert "delete_path" in result.output
        assert "1 skills" in result.output

    def test_marks_a_skill_that_fails_more_than_it_succeeds(self, memory: SqliteMemory) -> None:
        skill_id = _skill(memory)
        memory.skills.record_failure(skill_id)
        memory.skills.record_failure(skill_id)
        result = runner.invoke(cli.app, ["skills", "list"])
        assert "fails more often" in result.output

    def test_respects_the_limit(self, memory: SqliteMemory) -> None:
        for i in range(4):
            _skill(memory, f"open browser page number {i} and summarise the contents")
        result = runner.invoke(cli.app, ["skills", "list", "--limit", "2"])
        assert result.exit_code == 0, result.output
        assert result.output.count("delete_path") + result.output.count("page number") <= 2


class TestSkillsShow:
    def test_shows_one_skill(self, memory: SqliteMemory) -> None:
        skill_id = _skill(memory)
        result = runner.invoke(cli.app, ["skills", "show", str(skill_id)])
        assert result.exit_code == 0, result.output
        assert "delete_path" in result.output
        assert "paths=" in result.output

    def test_unknown_id_exits_1(self, memory: SqliteMemory) -> None:
        result = runner.invoke(cli.app, ["skills", "show", "999"])
        assert result.exit_code == 1
        assert "no skill with id 999" in result.output


class TestSkillsDelete:
    def test_deletes_a_skill(self, memory: SqliteMemory) -> None:
        skill_id = _skill(memory)
        result = runner.invoke(cli.app, ["skills", "delete", str(skill_id)])
        assert result.exit_code == 0, result.output
        assert "deleted skill" in result.output
        assert memory.skills.count() == 0

    def test_unknown_id_exits_1(self, memory: SqliteMemory) -> None:
        result = runner.invoke(cli.app, ["skills", "delete", "999"])
        assert result.exit_code == 1
        assert memory.skills.count() == 0


class TestSkillsClear:
    def test_clear_asks_first(self, memory: SqliteMemory) -> None:
        _skill(memory)
        result = runner.invoke(cli.app, ["skills", "clear"], input="n\n")
        assert result.exit_code == 0, result.output
        assert memory.skills.count() == 1

    def test_clear_removes_everything_after_confirmation(self, memory: SqliteMemory) -> None:
        _skill(memory)
        memory.failures.record("a goal", {"tool": "x"}, "denied")
        memory.preferences.set("units", "metric")
        result = runner.invoke(cli.app, ["skills", "clear"], input="y\n")
        assert result.exit_code == 0, result.output
        assert "cleared 3 memory rows" in result.output
        assert memory.counts() == {"skills": 0, "failures": 0, "preferences": 0}


def test_secrets_are_never_shown_by_the_cli(memory: SqliteMemory) -> None:
    saved = memory.skills.save(
        "unlock the vault",
        plan={
            "kind": "tool",
            "goal": "unlock the vault",
            "steps": [
                {"id": "s1", "tool": "unlock", "args": {"password": "hunter2"}, "expect": "e"}
            ],
        },
        tools_used=["unlock"],
    )
    assert saved.saved
    listing = runner.invoke(cli.app, ["skills", "list"])
    assert "hunter2" not in listing.output
    detail = runner.invoke(cli.app, ["skills", "show", str(saved.skill.id)])
    assert "hunter2" not in detail.output
    assert "[redacted]" in detail.output


class TestDisabledMemory:
    """`jarvis skills ...` must explain itself, not crash, when memory is off."""

    @pytest.fixture(autouse=True)
    def _disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = Settings()
        settings.memory.enabled = False
        # the real seam, so this exercises exactly the path `jarvis` takes
        monkeypatch.setattr(cli.config, "load_settings", lambda: settings)

    @pytest.mark.parametrize(
        "args", [["skills", "list"], ["skills", "show", "1"], ["skills", "delete", "1"]]
    )
    def test_commands_report_that_memory_is_disabled(self, args: list[str]) -> None:
        result = runner.invoke(cli.app, args)
        assert result.exit_code == 1
        assert "memory is disabled" in result.output
        assert "AttributeError" not in result.output


class TestConnectionsAreClosed:
    """Every command must release the SQLite handle, including on `typer.Exit`."""

    @pytest.fixture
    def tracked(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[SqliteMemory]:
        opened: list[SqliteMemory] = []
        real_open = cli.open_memory

        def spy(*args: object, **kwargs: object) -> SqliteMemory:
            backend = real_open(*args, **kwargs)
            if isinstance(backend, SqliteMemory):
                opened.append(backend)
            return backend

        monkeypatch.setattr(cli.config, "load_settings", lambda: Settings())
        monkeypatch.setattr(cli, "open_memory", spy)
        monkeypatch.setattr(
            cli,
            "_open_memory_store",
            lambda *_a, **_k: spy(settings=Settings(), path=tmp_path / "m.db"),
        )
        return opened

    @pytest.mark.parametrize(
        ("args", "code"),
        [
            (["skills", "show", "999"], 1),
            (["skills", "delete", "999"], 1),
            (["skills", "list"], 0),
        ],
    )
    def test_store_is_closed_whatever_the_exit_path(
        self, tracked: list[SqliteMemory], args: list[str], code: int
    ) -> None:
        assert runner.invoke(cli.app, args).exit_code == code
        assert tracked, "expected a real SqliteMemory backend"
        # a closed connection refuses any query - proof it was released
        with pytest.raises(sqlite3.ProgrammingError):
            tracked[-1].counts()
