"""The persistence boundary must reject bare vendor token shapes.

The live acceptance run found that ``PreferenceStore.set`` only rejected
``name=value`` credential pairs, so a raw ``sk-...``/``AKIA...``/``ghp_...``
value could be stored under a harmless key.  These tests pin the corrected
behaviour at both layers:

* the store (``PreferenceStore`` / ``SqliteMemory.remember``), and
* the conversational guard (``detect_memory_intent``),

using the same shared detector, so the two cannot diverge again.  Every token
below is synthetic and was never issued by any provider.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pytest

from jarvis.memory import SqliteMemory, open_memory
from jarvis.memory.base import MemoryRecord
from jarvis.memory.db import looks_like_token, looks_secret, redact_secrets
from jarvis.memory.intent import detect_memory_intent

# Synthetic, non-functional values.  They match the shape of the vendor
# formats but are deliberately invalid (and appear nowhere else).
FAKE_OPENAI = "sk-proj-SYNTHETIC000000000000000000000000000000000000"
FAKE_AWS = "AKIAIOSFODNN7SYNTHETIC"
FAKE_GITHUB_CLASSIC = "ghp_SYNTHETIC0000000000000000000000000000"
FAKE_GITHUB_PAT = "github_pat_11SYNTHETIC0000000000_SYNTHETIC000000000000000000000000000000"


@pytest.fixture
def memory(tmp_path: Path) -> SqliteMemory:
    store = open_memory(tmp_path / "memory.db")
    assert isinstance(store, SqliteMemory)
    yield store
    store.close()


def _rows(db_path: Path) -> list[tuple[str, str]]:
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        return [tuple(row) for row in conn.execute("SELECT key, value FROM preferences")]
    finally:
        conn.close()


# ----------------------------------------------------------------- detector


@pytest.mark.parametrize(
    "token",
    [FAKE_OPENAI, FAKE_AWS, FAKE_GITHUB_CLASSIC, FAKE_GITHUB_PAT],
)
def test_detector_recognises_vendor_shapes(token: str) -> None:
    assert looks_like_token(token) is True
    assert looks_secret(token) is True


@pytest.mark.parametrize(
    "text",
    [
        "Python",
        "VS Code",
        "Git Bash",
        "4 spaces",
        "dark",
        "scikit-learn",
        "task-queue",
        "print('hello world')",
        "https://github.com/aarya/jarvis",
        "C:/Users/aarya/dev/jarvis",
        "Windows 11 + CPython 3.11",
        "prefer focused pytest runs over the full suite",
    ],
)
def test_detector_keeps_ordinary_text(text: str) -> None:
    """No blanket rejection of words, paths, code or identifiers."""
    assert looks_like_token(text) is False
    assert looks_secret(text) is False


# -------------------------------------------------------------------- store


@pytest.mark.parametrize("token", [FAKE_OPENAI, FAKE_AWS, FAKE_GITHUB_CLASSIC, FAKE_GITHUB_PAT])
def test_set_rejects_token_values(memory: SqliteMemory, token: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError) as excinfo:
        memory.preferences.set("favourite_note", token)
    # The message is constant: the supplied value never appears in it.
    assert token not in str(excinfo.value)
    assert memory.preferences.count() == 0
    assert _rows(tmp_path / "memory.db") == []


def test_set_rejects_token_shaped_key(memory: SqliteMemory) -> None:
    with pytest.raises(ValueError):
        memory.preferences.set("sk-proj-SYNTHETIC0000000000000000000000", "anything")
    assert memory.preferences.count() == 0


def test_rejected_update_keeps_previous_value(memory: SqliteMemory) -> None:
    memory.preferences.set("editor", "VS Code")
    for token in (FAKE_OPENAI, FAKE_AWS, FAKE_GITHUB_CLASSIC, FAKE_GITHUB_PAT):
        with pytest.raises(ValueError):
            memory.preferences.set("editor", token)
        assert memory.preferences.get("editor") == "VS Code"
    assert memory.preferences.count() == 1


def test_remember_rejects_token_without_writing(memory: SqliteMemory, tmp_path: Path) -> None:
    """The graph path (``SqliteMemory.remember``) rejects the whole record."""
    memory.remember(
        MemoryRecord(
            kind="preference",
            text="remember a note",
            meta={"preferences": {"harmless_key": FAKE_OPENAI}},
        )
    )
    assert memory.preferences.count() == 0
    assert _rows(tmp_path / "memory.db") == []


def test_ordinary_preferences_still_work(memory: SqliteMemory) -> None:
    memory.preferences.set("editor", "VS Code")
    memory.preferences.set("indent", "4 spaces")
    memory.preferences.set("python", "CPython 3.11")
    assert memory.preferences.get("editor") == "VS Code"
    assert memory.preferences.get("indent") == "4 spaces"
    assert memory.preferences.count() == 3


# ------------------------------------------------------- chat/store parity


@pytest.mark.parametrize("token", [FAKE_OPENAI, FAKE_AWS, FAKE_GITHUB_CLASSIC, FAKE_GITHUB_PAT])
def test_chat_and_store_reject_the_same_token(memory: SqliteMemory, token: str) -> None:
    intent = detect_memory_intent(f"remember my note is {token}")
    assert intent is not None
    assert intent.error == "sensitive"
    with pytest.raises(ValueError):
        memory.preferences.set("my_note", token)


def test_chat_accepts_ordinary_fact_that_store_accepts(memory: SqliteMemory) -> None:
    intent = detect_memory_intent("remember that my preferred editor is VS Code")
    assert intent is not None
    assert intent.valid
    memory.preferences.set(intent.key, "VS Code")
    assert memory.preferences.get(intent.key) == "VS Code"


# ------------------------------------------------------------- no leakage


@pytest.mark.parametrize("token", [FAKE_OPENAI, FAKE_GITHUB_PAT])
def test_rejection_does_not_leak_or_log_the_token(
    memory: SqliteMemory, token: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(ValueError) as excinfo:
            memory.preferences.set("favourite_note", token)
        # The same path the graph uses, which logs the failure reason.
        memory.remember(
            MemoryRecord(
                kind="preference",
                text="x",
                meta={"preferences": {"favourite_note": token}},
            )
        )
    assert token not in str(excinfo.value)
    assert token not in caplog.text
    assert token not in repr(memory.preferences.all())


def test_redaction_removes_bare_tokens_but_keeps_prose() -> None:
    text = f"the key {FAKE_OPENAI} was rejected"
    assert redact_secrets(text) == "the key [redacted] was rejected"
    assert redact_secrets("nothing sensitive here") == "nothing sensitive here"
    assert redact_secrets("password=hunter2") == "password=[redacted]"
