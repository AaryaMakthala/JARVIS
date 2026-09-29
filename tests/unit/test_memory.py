"""Phase 8 memory store: skills, failures, preferences, retrieval, safety."""

from __future__ import annotations

import sqlite3

import pytest

from jarvis.config import Settings
from jarvis.memory import (
    FailureStore,
    MemoryRecord,
    NullMemory,
    PreferenceStore,
    SkillStore,
    SqliteMemory,
    open_db,
    open_memory,
)
from jarvis.memory import db as db_module
from jarvis.memory.embeddings import (
    DEFAULT_SIMILARITY,
    cosine,
    embed_text,
    from_bytes,
    to_bytes,
)


@pytest.fixture
def conn() -> sqlite3.Connection:
    connection = open_db(":memory:")
    try:
        yield connection
    finally:
        connection.close()


@pytest.fixture
def memory(conn: sqlite3.Connection) -> SqliteMemory:
    return SqliteMemory(conn, settings=Settings())


def _plan(goal: str = "delete the downloads folder") -> dict:
    return {
        "kind": "tool",
        "goal": goal,
        "steps": [
            {
                "id": "s1",
                "tool": "delete_path",
                "args": {"path": "C:/Users/a/Downloads"},
                "expect": "folder gone",
            }
        ],
    }


# ---------------------------------------------------------------- embeddings


def test_embed_is_deterministic_across_calls() -> None:
    assert embed_text("delete the downloads folder") == embed_text("delete the downloads folder")


def test_embed_is_process_stable() -> None:
    """A vector written by one process must still match in the next one.

    Guards the reason :func:`embed_text` hashes with BLAKE2b instead of
    ``hash()`` (which is salted per process).
    """
    import subprocess
    import sys

    code = (
        "from jarvis.memory.embeddings import embed_text;"
        "v=embed_text('delete the downloads folder');"
        "print(round(sum(x*x for x in v), 6))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=60
    )
    ours = round(sum(x * x for x in embed_text("delete the downloads folder")), 6)
    assert float(out.stdout.strip()) == pytest.approx(ours, abs=1e-6)


def test_embed_blank_text_never_matches() -> None:
    assert cosine(embed_text(""), embed_text("delete the downloads")) == 0.0


def test_similar_command_scores_above_threshold() -> None:
    score = cosine(
        embed_text("delete the downloads folder"), embed_text("Delete the Downloads folder!")
    )
    assert score >= DEFAULT_SIMILARITY


def test_unrelated_command_scores_below_threshold() -> None:
    score = cosine(
        embed_text("delete the downloads folder"), embed_text("who won the cricket final")
    )
    assert score < DEFAULT_SIMILARITY


def test_bytes_round_trip() -> None:
    vector = embed_text("open spotify")
    assert from_bytes(to_bytes(vector)) == pytest.approx(vector, abs=1e-6)


def test_corrupt_blob_degrades_to_empty() -> None:
    assert from_bytes(b"") == []
    assert from_bytes(b"\x00\x01") == []


# -------------------------------------------------------------------- skills


def test_save_then_find_retrieves_the_skill(memory: SqliteMemory) -> None:
    result = memory.skills.save(
        "delete the downloads folder", plan=_plan(), tools_used=["delete_path"]
    )
    assert result.saved and result.created

    found = memory.skills.find("delete the downloads folder")
    assert [s.goal_text for s in found] == ["delete the downloads folder"]


def test_repeating_a_goal_reinforces_instead_of_duplicating(memory: SqliteMemory) -> None:
    memory.skills.save("delete the downloads folder", plan=_plan(), tools_used=["delete_path"])
    again = memory.skills.save(
        "Delete the downloads folder!", plan=_plan(), tools_used=["delete_path"]
    )
    assert again.created is False
    assert again.reason == "reinforced"
    assert memory.skills.count() == 1
    assert memory.skills.all()[0].success_count == 2


def test_unrelated_goal_creates_a_second_skill(memory: SqliteMemory) -> None:
    memory.skills.save("delete the downloads folder", plan=_plan())
    memory.skills.save("open spotify and play jazz", plan=_plan("open spotify"))
    assert memory.skills.count() == 2


def test_skill_that_fails_more_than_it_succeeds_is_ignored(memory: SqliteMemory) -> None:
    saved = memory.skills.save("delete the downloads folder", plan=_plan())
    skill_id = saved.skill.id
    memory.skills.record_failure(skill_id)
    memory.skills.record_failure(skill_id)

    assert memory.skills.find("delete the downloads folder") == []
    assert memory.skills.get(skill_id).trusted is False
    assert [s.id for s in memory.skills.all()] == [skill_id]
    assert [s.id for s in memory.skills.all(trusted_only=True)] == []


def test_save_refuses_a_goal_that_carries_a_secret(memory: SqliteMemory) -> None:
    result = memory.skills.save("unlock with password=hunter2", plan=_plan())
    assert result.saved is False
    assert "secret" in result.reason
    assert memory.skills.count() == 0


def test_save_redacts_a_credential_nested_in_a_list_argument(memory: SqliteMemory) -> None:
    """``steps`` is a *list*: a naive dict-only scrub would let hunter2 through."""
    plan = _plan("update the saved profile")
    plan["steps"][0]["args"] = {"targets": [{"name": "x", "password": "hunter2"}]}
    result = memory.skills.save("update the saved profile", plan=plan)
    assert result.saved is True
    assert "hunter2" not in str(result.skill.plan)
    assert result.skill.plan["steps"][0]["args"]["targets"][0]["password"] == "[redacted]"


def test_save_redacts_secret_named_args_with_no_value_in_it(memory: SqliteMemory) -> None:
    """A secret-*named* field with nothing secret in it is stored, redacted."""
    plan = _plan("refresh the app token from the vault")
    plan["steps"][0]["args"] = {"api_key": "[redacted]", "path": "C:/x"}
    result = memory.skills.save("refresh the app token from the vault", plan=plan)
    assert result.saved is True
    assert result.skill.plan["steps"][0]["args"]["api_key"] == "[redacted]"
    assert result.skill.plan["steps"][0]["args"]["path"] == "C:/x"


def test_save_rejects_blank_goal(memory: SqliteMemory) -> None:
    assert memory.skills.save("   ").saved is False


def test_skill_text_is_prose_not_json(memory: SqliteMemory) -> None:
    saved = memory.skills.save(
        "delete the downloads folder", plan=_plan(), tools_used=["delete_path"]
    )
    text = saved.skill.as_memory_text()
    assert "delete_path" in text
    assert "{" not in text and "}" not in text


def test_delete_and_clear(memory: SqliteMemory) -> None:
    first = memory.skills.save("delete the downloads folder", plan=_plan()).skill
    memory.skills.save("open spotify", plan=_plan("open spotify"))
    assert memory.skills.delete(first.id) is True
    assert memory.skills.delete(first.id) is False
    assert memory.skills.clear() == 1


def test_max_rows_prunes_least_recently_used(
    memory: SqliteMemory, conn: sqlite3.Connection
) -> None:
    store = SkillStore(conn, max_rows=2)
    for i in range(4):
        store.save(f"open browser page number {i} and summarise it", plan=_plan())
    assert store.count() <= 2


# ------------------------------------------------------------------ failures


def test_failure_is_recorded_and_reported(memory: SqliteMemory) -> None:
    failure = memory.failures.record(
        "delete the downloads folder", {"tool": "delete_path"}, "Access denied: file in use"
    )
    assert failure is not None
    assert failure.error == "Access denied: file in use"
    assert memory.failures.recent()[0].step["tool"] == "delete_path"


def test_failure_skips_empty_error(memory: SqliteMemory) -> None:
    assert memory.failures.record("a goal", {"tool": "x"}, "  ") is None
    assert memory.failures.count() == 0


def test_repeated_tools_names_what_already_failed(memory: SqliteMemory) -> None:
    memory.failures.record("delete the downloads folder", {"tool": "web_answer"}, "no result")
    memory.failures.record("delete the downloads folder", {"tool": "delete_path"}, "denied")
    # most recent first, de-duplicated
    assert memory.failures.repeated_tools("delete the downloads") == ["delete_path", "web_answer"]


def test_failure_text_is_cautionary_not_a_veto(memory: SqliteMemory) -> None:
    """A remembered failure must advise, not forbid.

    The old wording ended in "Do not repeat that approach", and the planner
    obeyed it as a standing refusal: a stale row made "lock my computer" come
    back as `unsupported` for a week.  The hint still tells the planner to try
    something else, but the system prompt - not the hint - now says a past
    failure is never a reason to refuse the current request.
    """
    memory.failures.record("delete the downloads folder", {"tool": "delete_path"}, "denied")
    text = memory.failures.recent()[0].as_memory_text()
    assert "Prefer a different approach" in text
    assert "Do not repeat" not in text
    assert "delete the downloads folder" in text and "denied" in text


def test_failure_goal_with_a_secret_is_redacted(memory: SqliteMemory) -> None:
    memory.failures.record("unlock using password=hunter2", {"tool": "unlock"}, "denied")
    assert memory.failures.recent()[0].goal_text == "[redacted goal]"


def test_failure_step_drops_secret_looking_pairs(memory: SqliteMemory) -> None:
    memory.failures.record("a goal", {"tool": "x", "password": "hunter2"}, "denied")
    assert "hunter2" not in str(memory.failures.recent()[0].step)


def test_failure_error_text_is_redacted_before_it_is_stored(
    memory: SqliteMemory,
) -> None:
    """A tool error is untrusted third-party text and may echo a credential."""
    memory.failures.record(
        "a goal", {"tool": "x"}, "login failed: user bob, password=hunter2, try again"
    )
    stored = memory.failures.recent()[0].error
    assert "hunter2" not in stored
    assert "password=[redacted]" in stored
    assert "login failed" in stored  # the useful part survives


def test_failure_error_text_is_redacted_in_the_nested_step(
    memory: SqliteMemory,
) -> None:
    memory.failures.record("a goal", {"tool": "x", "args": {"api_key": "sk-live-1"}}, "denied")
    assert "sk-live-1" not in str(memory.failures.recent()[0].step)


def test_redact_secrets_keeps_ordinary_prose() -> None:
    redact = db_module.redact_secrets
    assert redact("no secrets here") == "no secrets here"
    assert redact("") == ""
    assert redact("token: abc123") == "token: [redacted]"
    # idempotent: an already-redacted value is not re-redacted or mangled
    once = redact("token: abc123")
    assert redact(once) == once


# --------------------------------------------------------------- preferences


def test_preference_round_trip(memory: SqliteMemory) -> None:
    memory.preferences.set("search_engine", "duckduckgo")
    assert memory.preferences.get("search_engine") == "duckduckgo"
    assert memory.preferences.get("Search_Engine") == "duckduckgo"
    assert memory.preferences.get("missing", "fallback") == "fallback"


def test_preference_set_overwrites(memory: SqliteMemory) -> None:
    memory.preferences.set("units", "metric")
    memory.preferences.set("units", "imperial")
    assert memory.preferences.count() == 1
    assert memory.preferences.get("units") == "imperial"


@pytest.mark.parametrize("key", ["", "  ", "with space", "a" * 80, "password"])
def test_preference_rejects_unusable_keys(memory: SqliteMemory, key: str) -> None:
    with pytest.raises(ValueError):
        memory.preferences.set(key, "x")


def test_preference_rejects_secret_values(memory: SqliteMemory) -> None:
    with pytest.raises(ValueError, match="credential"):
        memory.preferences.set("login", "password=hunter2")


def test_preference_delete(memory: SqliteMemory) -> None:
    memory.preferences.set("units", "metric")
    assert memory.preferences.delete("units") is True
    assert memory.preferences.delete("units") is False


# ----------------------------------------------------------------- retrieval


def test_retrieve_returns_skills_first_then_preferences_then_failures(
    memory: SqliteMemory,
) -> None:
    memory.skills.save("delete the downloads folder", plan=_plan(), tools_used=["delete_path"])
    memory.preferences.set("units", "metric")
    memory.failures.record("delete the downloads folder", {"tool": "delete_path"}, "denied")

    records = memory.retrieve("delete the downloads folder", limit=3)
    kinds = [r.meta.get("skill_id") is not None for r in records]
    assert kinds[0] is True  # a verified approach outranks everything else
    assert any(r.kind == "preference" for r in records)
    assert any("Prefer a different approach" in r.text for r in records)


def test_retrieve_respects_the_limit(memory: SqliteMemory) -> None:
    for i in range(5):
        memory.preferences.set(f"pref_{i}", str(i))
    assert len(memory.retrieve("anything", limit=2)) == 2


def test_retrieve_ignores_untrusted_skill(memory: SqliteMemory) -> None:
    saved = memory.skills.save("delete the downloads folder", plan=_plan()).skill
    memory.skills.record_failure(saved.id)
    memory.skills.record_failure(saved.id)
    assert memory.retrieve("delete the downloads folder") == []


def test_retrieve_blank_query_returns_nothing(memory: SqliteMemory) -> None:
    memory.skills.save("delete the downloads folder", plan=_plan())
    assert memory.retrieve("   ") == []
    assert memory.retrieve("delete the downloads", limit=0) == []


def test_retrieve_never_exceeds_a_corrupt_row(
    memory: SqliteMemory, conn: sqlite3.Connection
) -> None:
    conn.execute(
        "INSERT INTO skills (goal_text, plan_json, embedding, tools_used, success_count,"
        " fail_count, created_at, last_used_at) VALUES ('broken', '{}', ?, '[]', 1, 0, '', '')",
        (b"\x00\x01\x02",),
    )
    memory.skills.save("delete the downloads folder", plan=_plan())
    assert [r.text for r in memory.retrieve("delete the downloads folder")] == [
        'Earlier successful approach for "delete the downloads folder" used: no tools.'
    ]


# ------------------------------------------------------------------ remember


def test_remember_routes_by_kind(memory: SqliteMemory) -> None:
    memory.remember(MemoryRecord(kind="session", text="hello"))
    memory.remember(
        MemoryRecord(
            kind="episodic",
            text="delete the downloads folder",
            meta={"goal_text": "delete the downloads folder"},
        )
    )
    memory.remember(
        MemoryRecord(kind="preference", text="x", meta={"key": "units", "value": "metric"})
    )
    assert memory.counts() == {"skills": 1, "failures": 0, "preferences": 1}


def test_remember_does_not_persist_session_records(memory: SqliteMemory) -> None:
    memory.remember(MemoryRecord(kind="session", text="a passing thought"))
    assert memory.counts()["skills"] == 0


def test_remember_round_trip_of_a_failure_stays_a_failure(memory: SqliteMemory) -> None:
    memory.failures.record("delete the downloads folder", {"tool": "delete_path"}, "denied")
    record = memory.retrieve("delete the downloads folder")[0]
    assert record.meta.get("failure_id") is not None
    memory.remember(record)
    # Re-offering a "this failed before" hint must never mint a reusable skill.
    assert memory.counts()["skills"] == 0
    assert memory.counts()["failures"] == 2


def test_remember_never_raises_on_a_broken_backend(memory: SqliteMemory) -> None:
    class Exploding:
        def remember(self, record: MemoryRecord) -> None:
            raise RuntimeError("disk on fire")

    from jarvis.agent.nodes.memory_save import _write

    outcome = _write(Exploding(), MemoryRecord(text="x"), "")
    assert outcome.record is None
    assert outcome.reason == "memory write failed"


# ------------------------------------------------------------------ factory


def test_open_memory_returns_null_when_disabled(tmp_path) -> None:
    settings = Settings(memory={"enabled": False})
    assert isinstance(open_memory(tmp_path / "m.db", settings=settings), NullMemory)


def test_open_memory_returns_null_in_dry_run(tmp_path) -> None:
    path = tmp_path / "m.db"
    assert isinstance(open_memory(path, settings=Settings(), dry_run=True), NullMemory)
    assert not path.exists()


def test_open_memory_degrades_when_the_store_cannot_be_opened(tmp_path) -> None:
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("i am a file", encoding="utf-8")
    assert isinstance(open_memory(blocker / "m.db", settings=Settings()), NullMemory)


def test_open_memory_creates_a_real_store(tmp_path) -> None:
    store = open_memory(tmp_path / "m.db", settings=Settings())
    assert isinstance(store, SqliteMemory)
    try:
        assert store.counts() == {"skills": 0, "failures": 0, "preferences": 0}
    finally:
        store.close()


def test_stores_share_one_connection(conn: sqlite3.Connection) -> None:
    assert isinstance(SkillStore(conn), SkillStore)
    assert isinstance(FailureStore(conn), FailureStore)
    assert isinstance(PreferenceStore(conn), PreferenceStore)
    assert sqlite3.Connection is conn.__class__


# ------------------------------------------------- unverifiable-success rows
#
# The live bug: a build before a3cdf21 recorded "succeeded but could not be
# verified" as a failure, so a later "lock my computer" was refused with "a
# previous attempt failed" although the lock had worked.  The row is still in
# the user's memory.db, so the migration has to neutralise it *without*
# throwing away genuine history.

#: The full schema-v1 database (identical to the current one except the
#: ``failures`` columns added by the migration below).
_V1_SCHEMA = """
CREATE TABLE skills (
  id INTEGER PRIMARY KEY,
  goal_text TEXT NOT NULL,
  plan_json TEXT NOT NULL,
  embedding BLOB NOT NULL,
  tools_used TEXT NOT NULL,
  success_count INTEGER DEFAULT 1,
  fail_count INTEGER DEFAULT 0,
  created_at TEXT NOT NULL,
  last_used_at TEXT NOT NULL
);
CREATE TABLE failures (
  id INTEGER PRIMARY KEY,
  goal_text TEXT,
  step_json TEXT,
  error TEXT,
  created_at TEXT NOT NULL
);
CREATE TABLE preferences (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE task_log (
  task_id TEXT PRIMARY KEY,
  source TEXT,
  user_input TEXT,
  status TEXT,
  api_calls INTEGER,
  tokens INTEGER,
  steps INTEGER,
  replans INTEGER,
  started_at TEXT,
  finished_at TEXT,
  duration_ms INTEGER,
  peak_rss_mb REAL
);
"""


def _v1_db(tmp_path, name: str = "v1.db") -> sqlite3.Connection:
    """A schema-v1 database holding the false row from the live incident."""
    connection = db_module.connect(tmp_path / name)
    connection.executescript(_V1_SCHEMA)
    connection.execute("PRAGMA user_version = 1")
    connection.execute(
        "INSERT INTO failures (goal_text, step_json, error, created_at) VALUES (?, ?, ?, ?)",
        (
            "Lock the computer",
            '{"tool": "lock_computer", "id": "s1", "tainted": false}',
            "step s1 did not complete",
            "2026-09-28T11:53:39+00:00",
        ),
    )
    return connection


def test_migration_quarantines_the_false_unverifiable_success_row(tmp_path) -> None:
    connection = _v1_db(tmp_path)
    try:
        assert db_module.migrate_failures(connection) == 1
        row = connection.execute("SELECT quarantined, step_state FROM failures").fetchone()
        assert row["quarantined"] == 1
        assert row["step_state"] is None  # history preserved, nothing rewritten
    finally:
        connection.close()


def test_quarantined_row_is_never_offered_to_the_planner(tmp_path) -> None:
    """The end-to-end regression: the same goal must retrieve no failure."""
    store = open_memory(tmp_path / "current.db", settings=Settings())
    try:
        store.failures.record(
            "Lock the computer",
            {"tool": "lock_computer", "id": "s1", "tainted": False},
            "step s1 did not complete",
        )  # a current build: recorded WITH provenance
        assert store.failures.relevant("lock my computer", limit=3) != []
    finally:
        store.close()

    legacy = _v1_db(tmp_path)
    try:
        db_module.migrate_failures(legacy)
        store = SqliteMemory(legacy, settings=Settings())
        assert store.failures.relevant("lock my computer", limit=3) == []
        assert store.counts()["failures"] == 0
        assert store.quarantined_failures() == 1
    finally:
        legacy.close()


def test_quarantined_row_is_kept_and_counted_not_deleted(tmp_path) -> None:
    legacy = _v1_db(tmp_path)
    try:
        db_module.migrate_failures(legacy)
        rows = legacy.execute("SELECT COUNT(*) FROM failures").fetchone()[0]
        assert rows == 1
        store = SqliteMemory(legacy, settings=Settings())
        # Still on disk and still listable, so the history stays auditable.
        assert store.failures.quarantined_count() == 1
        assert store.failures.recent(include_quarantined=True)[0].goal_text == "Lock the computer"
        assert store.failures.recent() == []
    finally:
        legacy.close()


def test_migration_keeps_a_genuine_legacy_failure(tmp_path) -> None:
    """A real tool error is never quarantined, however old the row is."""
    legacy = _v1_db(tmp_path)
    try:
        legacy.execute(
            "INSERT INTO failures (goal_text, step_json, error, created_at) VALUES (?, ?, ?, ?)",
            (
                "delete the downloads folder",
                '{"tool": "delete_path", "id": "s2", "tainted": false}',
                "Access denied: file in use",
                "2026-09-01T09:00:00+00:00",
            ),
        )
        assert db_module.migrate_failures(legacy) == 1  # only the false one
        store = SqliteMemory(legacy, settings=Settings())
        assert store.counts()["failures"] == 1
        assert store.quarantined_failures() == 1
        assert [
            f.error for f in store.failures.relevant("delete the downloads folder", limit=3)
        ] == ["Access denied: file in use"]
    finally:
        legacy.close()


def test_migration_keeps_a_tainted_legacy_row(tmp_path) -> None:
    """Placeholder wording alone is not enough: a tainted step is a real failure."""
    legacy = _v1_db(tmp_path)
    try:
        legacy.execute("DELETE FROM failures")
        legacy.execute(
            "INSERT INTO failures (goal_text, step_json, error, created_at) VALUES (?, ?, ?, ?)",
            (
                "open a page",
                '{"tool": "web_answer", "id": "s1", "tainted": true}',
                "step s1 did not complete",
                "2026-09-01T09:00:00+00:00",
            ),
        )
        assert db_module.migrate_failures(legacy) == 0
        assert SqliteMemory(legacy, settings=Settings()).counts()["failures"] == 1
    finally:
        legacy.close()


def test_migration_is_idempotent(tmp_path) -> None:
    legacy = _v1_db(tmp_path)
    try:
        assert db_module.migrate_failures(legacy) == 1
        assert db_module.migrate_failures(legacy) == 0
        assert db_module.migrate_failures(legacy) == 0
    finally:
        legacy.close()


def test_current_build_never_quarantines_its_own_rows(memory: SqliteMemory) -> None:
    """A current ``verified=False`` with no error text has the same sentence as
    the old bug, but it carries provenance - so it must survive the migration."""
    memory.failures.record(
        "lock my computer",
        {"tool": "lock_computer", "id": "s1", "tainted": False},
        "step s1 did not complete",
        state={"ok": True, "verified": False, "tainted": False},
    )
    assert memory.failures.relevant("lock my computer", limit=3) != []
    assert memory.quarantined_failures() == 0


def test_opening_an_upgraded_database_stamps_the_version(tmp_path) -> None:
    store = open_memory(tmp_path / "m.db", settings=Settings())
    conn = store._conn
    try:
        assert db_module.user_version(conn) == db_module.SCHEMA_VERSION
    finally:
        store.close()


# ------------------------------------------------------ failure retrieval


def test_failure_retrieval_is_scoped_to_the_request(memory: SqliteMemory) -> None:
    """The general bug: a remembered failure used to reach *every* prompt.

    With ``retrieval_limit`` 3 and no skills, an unrelated old failure was
    injected into the planner's context for any command at all - which is how
    one row refused a lock request the user had never failed at.
    """
    memory.failures.record("delete the downloads folder", {"tool": "delete_path"}, "denied")
    assert memory.retrieve("lock my computer", limit=3) == []
    assert memory.retrieve("what is the capital of France", limit=3) == []
    assert [r.text for r in memory.retrieve("delete the downloads folder", limit=3)]


def test_failure_retrieval_still_matches_a_rephrased_goal(memory: SqliteMemory) -> None:
    """Scoping must not swallow a genuine failure: the floor is 0.45, not 0.75."""
    memory.failures.record("open notepad", {"tool": "type_text"}, "denied")
    hits = memory.failures.relevant("open the notepad app", limit=3)
    assert [f.step["tool"] for f in hits] == ["type_text"]


def test_failure_retrieval_skips_quarantined_and_blank_queries(memory: SqliteMemory) -> None:
    memory.failures.record("open notepad", {"tool": "type_text"}, "denied")
    assert memory.failures.relevant("open notepad", limit=0) == []
    assert memory.failures.relevant("   ", limit=3) == []


def test_failure_provenance_is_recorded_and_read_back(memory: SqliteMemory) -> None:
    memory.failures.record(
        "delete the downloads folder",
        {"tool": "delete_path", "id": "s1"},
        "denied",
        state={"ok": False, "verified": None, "tainted": False},
    )
    stored = memory.failures.recent()[0]
    assert stored.state == {"ok": False, "verified": None, "tainted": False}
    assert "failed at delete_path" in stored.as_memory_text()

    memory.failures.record(
        "open a page",
        {"tool": "web_answer", "id": "s2"},
        "denied",
        state={"ok": True, "verified": False, "tainted": False},
    )
    assert "did not verify at web_answer" in memory.failures.recent()[0].as_memory_text()

    memory.failures.record(
        "summarise a page",
        {"tool": "web_answer", "id": "s3"},
        "the page asked me to run a command",
        state={"ok": True, "verified": True, "tainted": True},
    )
    assert "returned untrusted external text" in memory.failures.recent()[0].as_memory_text()


def test_failure_provenance_is_whitelisted(memory: SqliteMemory) -> None:
    """Only the three known fields are stored: a caller cannot smuggle text in."""
    memory.failures.record(
        "a goal",
        {"tool": "x"},
        "denied",
        state={"ok": False, "note": "password=hunter2", "verified": "maybe"},
    )
    assert memory.failures.recent()[0].state == {
        "ok": False,
        "verified": None,
        "tainted": False,
    }


def test_config_and_store_agree_on_the_failure_similarity_floor() -> None:
    """``config`` may not import the memory layer, so the default is pinned here."""
    from jarvis.memory.failures import FAILURE_SIMILARITY

    assert Settings().memory.failure_similarity_threshold == pytest.approx(FAILURE_SIMILARITY)
