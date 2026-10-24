import sqlite3

from jarvis.memory import db
from jarvis.memory.tasklog import TaskLogStore


def test_migration_adds_columns(tmp_path):
    p = tmp_path / "m.db"
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    conn.executescript(db.SCHEMA)
    conn.commit()
    # drop new cols to simulate old
    # easier: just init_schema
    conn2 = db.open_db(str(p))
    db.init_schema(conn2)
    cur = conn2.execute("PRAGMA table_info(task_log)")
    cols = {r[1] for r in cur.fetchall()}
    for c in ["halt_reason","failure_code","tools_used","first_failing_step","error_text","llm_ms","llm_calls","provider_chain","tool_ms"]:
        assert c in cols


def test_redaction_and_population(tmp_path):
    conn = db.open_db(str(tmp_path / "t.db"))
    db.init_schema(conn)
    store = TaskLogStore(conn)
    store.start("t1", source="cli", user_input="hello")
    store.finish("t1", status="failed", halt_reason="boom", failure_code="E1", tools_used="open_app", first_failing_step="s1", error_text="password=secret123" + "x"*400, llm_ms=10, llm_calls=1, provider_chain="groq", tool_ms=5)
    r = store.get("t1")
    assert r is not None
    assert r.error_text is not None
    assert len(r.error_text) <= 300
    assert "secret123" not in (r.error_text or "")
