from __future__ import annotations

import ast
import inspect
from datetime import UTC, datetime
from pathlib import Path

import pytest

from jarvis.taskstore import (
    TaskEventType,
    TaskStatus,
    TaskStoreUnavailable,
    TaskStoreVocabulary,
    open_task_store,
    schema,
)
from jarvis.taskstore.privacy import (
    redact_secrets,
    safe_failure_code,
    safe_model,
    safe_node,
    safe_owner_kind,
    safe_provider,
    safe_source,
    safe_task_id,
    safe_tool,
    sanitize_detail,
)

TASK_1 = "00000000-0000-4000-8000-000000000001"
STEP_1 = "00000000-0000-4000-8000-000000000101"
STEP_2 = "00000000-0000-4000-8000-000000000102"

FAILURE_CATEGORIES = (
    "policy",
    "validation",
    "tool",
    "llm",
    "timeout",
    "cancelled",
    "verification",
    "internal",
    "unknown",
)


def _vocab() -> TaskStoreVocabulary:
    providers = ("groq", "openrouter", "nvidia", "gemini")
    return TaskStoreVocabulary(
        providers=frozenset(providers),
        models=frozenset((provider, "openai/gpt-oss-120b") for provider in providers),
        tools=frozenset({"browser_goto", "whatsapp_read_messages"}),
        nodes=frozenset({"planner"}),
        failure_codes=frozenset({"E_TOOL", "E_OTHER", "E_HALT"}),
        failure_categories=frozenset(FAILURE_CATEGORIES),
    )


VOCAB = _vocab()


@pytest.fixture
def store(tmp_path: Path):
    opened = open_task_store(tmp_path / "task_events.db", create=True, vocabulary=VOCAB)
    try:
        yield opened
    finally:
        opened.close()


def _event(result):
    assert result.persisted is True
    assert result.value is not None
    return result.value


def test_taskstore_source_does_not_import_memory() -> None:
    package = Path(__file__).resolve().parents[2] / "src" / "jarvis" / "taskstore"
    for source in package.glob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any(
                name == "jarvis.memory" or name.startswith("jarvis.memory.") for name in names
            )
            assert "MemoryBackend" not in {alias.name for alias in getattr(node, "names", [])}


def test_schema_has_no_raw_conversation_or_tool_columns(store) -> None:
    forbidden = {
        "user_input",
        "prompt",
        "request",
        "message",
        "conversation",
        "error_text",
        "tool_args",
        "tool_output",
        "file_path",
        "window_text",
        "page_text",
        "screenshot",
        "password",
        "token",
    }
    for table in schema.REQUIRED_TABLES:
        assert forbidden.isdisjoint(schema.columns(store._conn, table))


def test_detail_is_allow_listed_and_free_text_is_dropped(store) -> None:
    store.ensure_task(TASK_1)
    event = _event(
        store.append_event(
            TASK_1,
            TaskEventType.NOTE,
            detail={
                "event_code": "progress",
                "count": 2,
                "node": "planner",
                "note": "the user said something private",
                "error_text": "raw failure text",
                "password": "hunter2",
                "tool_name": "unapproved-key",
            },
        )
    )
    assert event.detail == {"event_code": "progress", "count": 2, "node": "planner"}
    assert store.events(TASK_1)[-1].detail == event.detail


def test_secret_looking_optional_values_are_dropped(store) -> None:
    store.ensure_task(TASK_1)
    event = _event(
        store.append_event(
            TASK_1,
            TaskEventType.NOTE,
            detail={
                "event_code": "progress",
                "count": 1,
                "node": "api_key=sk-live-secret",
                "message_kind": "my_password_is_hunter2",
            },
        )
    )
    assert event.detail == {"event_code": "progress", "count": 1}


def test_detail_values_are_bounded_and_invalid_types_are_dropped() -> None:
    detail = sanitize_detail(
        TaskEventType.NOTE,
        {
            "event_code": "progress",
            "node": "x" * 300,
            "attempt": 1_000_000_001,
            "count": -1,
            "duration_ms": True,
            "retry_count": [1],
        },
        VOCAB,
    )
    assert detail == {"event_code": "progress"}


def test_source_and_failure_fields_are_sanitised(store, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        store.ensure_task(TASK_1, source="password=hunter2")
    assert store.get(TASK_1) is None

    store.ensure_task(TASK_1, source="voice")
    snapshot = store.get(TASK_1)
    assert snapshot is not None
    assert snapshot.source == "voice"

    store.update_status(TASK_1, TaskStatus.RUNNING)
    with pytest.raises(ValueError):
        store.finish(
            TASK_1,
            TaskStatus.FAILED,
            failure_category="the user said something private",
            failure_code="token=sk-live-secret",
        )
    assert store.get(TASK_1).status is TaskStatus.RUNNING

    store.close()
    database = (tmp_path / "task_events.db").read_bytes()
    assert b"hunter2" not in database
    assert b"sk-live-secret" not in database
    assert b"the user said something private" not in database


def test_review_sensitive_labels_are_not_persisted(store, tmp_path: Path) -> None:
    store.ensure_task(TASK_1, source="voice")

    with pytest.raises(ValueError):
        store.append_event(
            TASK_1,
            TaskEventType.FAILURE,
            detail={
                "failure_code": "C:/Users/aarya/secret.txt",
                "failure_category": "tool",
                "node": "planner",
                "tool": "browser_goto",
            },
        )
    with pytest.raises(ValueError):
        store.append_event(
            TASK_1,
            TaskEventType.TOOL_STARTED,
            detail={
                "step_id": "sk-live-secret",
                "tool": "browser_goto",
            },
        )
    note_event = _event(
        store.append_event(
            TASK_1,
            TaskEventType.NOTE,
            detail={
                "event_code": "progress",
                "count": 2,
                "node": "evil_node",
                "message_kind": "MY_PRIVATE_NOTE",
            },
        )
    )

    assert note_event.detail == {"event_code": "progress", "count": 2}

    store.close()
    database = (tmp_path / "task_events.db").read_bytes()
    for value in (
        "C:/Users/aarya/secret.txt",
        "555-1234",
        "sk-live-secret",
        "ghp_ABCDEFG",
        "evil_node",
        "MY_PRIVATE_NOTE",
    ):
        assert value.encode() not in database


def test_legitimate_labels_are_accepted_and_persisted(store, tmp_path: Path) -> None:
    store.ensure_task(TASK_1, source="terminal")
    for provider in ("groq", "openrouter", "nvidia", "gemini"):
        event = _event(
            store.append_event(
                TASK_1,
                TaskEventType.LLM_CALL,
                detail={
                    "provider": provider,
                    "model": "openai/gpt-oss-120b",
                    "role": "planner",
                },
            )
        )
        assert event.detail["provider"] == provider
        assert event.detail["model"] == "openai/gpt-oss-120b"

    first_tool = _event(
        store.append_event(
            TASK_1,
            TaskEventType.TOOL_STARTED,
            detail={"tool": "browser_goto", "step_id": STEP_1},
        )
    )
    second_tool = _event(
        store.append_event(
            TASK_1,
            TaskEventType.TOOL_STARTED,
            detail={"tool": "whatsapp_read_messages", "step_id": STEP_2},
        )
    )
    assert first_tool.detail == {"step_id": STEP_1, "tool": "browser_goto"}
    assert second_tool.detail == {"step_id": STEP_2, "tool": "whatsapp_read_messages"}

    store.close()
    database = (tmp_path / "task_events.db").read_bytes()
    for value in (
        "terminal",
        "groq",
        "openrouter",
        "nvidia",
        "gemini",
        "openai/gpt-oss-120b",
        "browser_goto",
        "whatsapp_read_messages",
    ):
        assert value.encode() in database


def test_source_and_owner_vocabularies_are_closed() -> None:
    for value in ("terminal", "voice", "benchmark"):
        assert safe_source(value) == value
    for value in ("cli", "TEST", "C:/Users/aarya/secret.txt", "555-1234"):
        assert safe_source(value) is None

    for value in ("daemon", "cli", "benchmark"):
        assert safe_owner_kind(value) == value
    for value in ("terminal", "voice", "tool"):
        assert safe_owner_kind(value) is None


def test_sensitive_and_legitimate_field_values() -> None:
    assert safe_model("openai/gpt-oss-120b", "groq", VOCAB) == "openai/gpt-oss-120b"
    assert safe_provider("groq", VOCAB) == "groq"
    assert safe_provider("openrouter", VOCAB) == "openrouter"
    assert safe_provider("nvidia", VOCAB) == "nvidia"
    assert safe_provider("gemini", VOCAB) == "gemini"
    assert safe_tool("browser_goto", VOCAB) == "browser_goto"
    assert safe_tool("whatsapp_read_messages", VOCAB) == "whatsapp_read_messages"
    assert safe_node("planner", VOCAB) == "planner"
    assert safe_failure_code("E_TOOL", VOCAB) == "E_TOOL"

    for value in ("C:/Users/aarya/secret.txt", "home/aarya/secret.txt", "../secret"):
        assert safe_model(value, "groq", VOCAB) is None
    for value in ("sk-live-secret", "ghp_ABCDEFG", "555-1234"):
        assert safe_model(value, "groq", VOCAB) is None
    assert safe_provider("ghp_ABCDEFG", VOCAB) is None
    assert safe_provider("openai", VOCAB) is None
    assert safe_tool("C:/Users/aarya/secret.txt", VOCAB) is None
    assert safe_tool("sk-live-secret", VOCAB) is None
    assert safe_tool("555-1234", VOCAB) is None
    assert safe_node("sk-live-secret", VOCAB) is None
    assert safe_failure_code("C:/Users/aarya/secret.txt", VOCAB) is None
    assert safe_failure_code("555-1234", VOCAB) is None
    assert safe_failure_code("sk-live-secret", VOCAB) is None


def test_event_payload_allowlists_are_event_specific(store) -> None:
    created = sanitize_detail(
        TaskEventType.TASK_CREATED,
        {
            "owner_kind": "cli",
            "provider": "groq",
            "source": "voice",
            "tool": "browser_goto",
        },
        VOCAB,
    )
    started = sanitize_detail(
        TaskEventType.TOOL_STARTED,
        {
            "provider": "groq",
            "source": "voice",
            "step_id": STEP_1,
            "tool": "browser_goto",
        },
        VOCAB,
    )
    finished = sanitize_detail(
        TaskEventType.FINISHED,
        {
            "count": 1,
            "duration_ms": 5,
            "failure_code": "E_TOOL",
            "outcome": "failed",
            "provider": "groq",
        },
        VOCAB,
    )

    assert created == {"owner_kind": "cli", "source": "voice"}
    assert started == {"step_id": STEP_1, "tool": "browser_goto"}
    assert finished == {
        "count": 1,
        "duration_ms": 5,
        "failure_code": "E_TOOL",
        "outcome": "failed",
    }
    assert sanitize_detail("unknown_event", {"tool": "browser_goto"}, VOCAB) == {}

    store.ensure_task(TASK_1)
    with pytest.raises(ValueError):
        store.append_event(TASK_1, "unknown_event", detail={"tool": "browser_goto"})


def test_required_payload_fields_reject_arbitrary_labels(store) -> None:
    store.ensure_task(TASK_1)
    with pytest.raises(ValueError):
        store.append_event(
            TASK_1,
            TaskEventType.NOTE,
            detail={"event_code": "MY_PRIVATE_REASON"},
        )
    with pytest.raises(ValueError):
        store.append_event(
            TASK_1,
            TaskEventType.TOOL_STARTED,
            detail={"tool": "unregistered_tool", "step_id": STEP_1},
        )
    with pytest.raises(ValueError):
        store.append_event(
            TASK_1,
            TaskEventType.FAILURE,
            detail={"failure_code": "sk-live-secret", "failure_category": "tool"},
        )
    with pytest.raises(ValueError):
        store.append_event(
            TASK_1,
            TaskEventType.WARNING,
            detail={"severity": "MY_PRIVATE_NOTE", "reason_code": "none"},
        )


def test_llm_call_requires_a_configured_provider_and_model_pair(store) -> None:
    store.ensure_task(TASK_1)
    with pytest.raises(ValueError):
        store.append_event(
            TASK_1,
            TaskEventType.LLM_CALL,
            detail={
                "provider": "madeupprovider",
                "model": "openai/gpt-oss-120b",
                "role": "planner",
            },
        )
    with pytest.raises(ValueError):
        store.append_event(
            TASK_1,
            TaskEventType.LLM_CALL,
            detail={
                "provider": "openai",
                "model": "openai/gpt-oss-120b",
                "role": "planner",
            },
        )
    with pytest.raises(ValueError):
        store.append_event(
            TASK_1,
            TaskEventType.LLM_CALL,
            detail={"provider": "groq", "model": "evil/model", "role": "planner"},
        )


def test_public_write_api_has_no_raw_error_text_parameter(store) -> None:
    forbidden = {"error", "error_text", "message", "prompt", "user_input"}
    for method in (type(store).append_event, type(store).finish):
        assert forbidden.isdisjoint(inspect.signature(method).parameters)


def test_task_id_is_bounded_and_closed() -> None:
    assert safe_task_id(TASK_1) == TASK_1
    with pytest.raises(ValueError):
        safe_task_id("bad task id")
    with pytest.raises(ValueError):
        safe_task_id("task-1")
    assert redact_secrets("password=hunter2") == "password=[redacted]"
    assert redact_secrets("nothing sensitive") == "nothing sensitive"


def test_unavailable_store_reads_do_not_look_empty() -> None:
    from jarvis.taskstore import NullTaskEventStore

    null = NullTaskEventStore("database_unavailable", vocabulary=VOCAB)
    with pytest.raises(TaskStoreUnavailable):
        null.get(TASK_1)
    with pytest.raises(TaskStoreUnavailable):
        null.events(TASK_1)
    with pytest.raises(TaskStoreUnavailable):
        null.recent(0)


def test_created_event_has_no_start_time_for_queued_task(store) -> None:
    when = datetime(2026, 1, 1, tzinfo=UTC)
    snapshot = store.ensure_task(TASK_1, created_at=when).value
    assert snapshot is not None
    assert snapshot.status is TaskStatus.QUEUED
    assert snapshot.started_at is None
    assert store.events(TASK_1)[0].created_at == when
