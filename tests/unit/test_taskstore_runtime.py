from __future__ import annotations

from pathlib import Path

import pytest

from jarvis import config
from jarvis.agent.context import make_app_context
from jarvis.taskstore import NullTaskEventStore, TaskFailureCategory, open_task_store
from jarvis.taskstore.runtime import build_task_vocabulary, open_runtime_task_store
from jarvis.tools.registry import ToolRegistry


class _Registry:
    def names(self) -> list[str]:
        return ["Browser_Goto", "whatsapp_send"]


def _settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> config.Settings:
    monkeypatch.setenv("JARVIS_DATA_DIR", str(tmp_path))
    settings = config.load_settings()
    settings.llm.provider_order = ["groq", "gemini"]
    settings.llm.models = {}
    settings.llm.planner_model = "planner-model"
    settings.llm.fast_model = "fast-model"
    settings.llm.vision_model = ""
    return settings


def test_vocabulary_comes_from_live_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(monkeypatch, tmp_path)

    vocabulary = build_task_vocabulary(settings, _Registry())

    assert vocabulary.providers == frozenset({"groq", "gemini"})
    assert vocabulary.models == frozenset(
        {
            ("groq", "planner-model"),
            ("groq", "fast-model"),
            ("gemini", "planner-model"),
            ("gemini", "fast-model"),
        }
    )
    assert vocabulary.tools == frozenset({"browser_goto", "whatsapp_send"})
    assert {"intake", "brain", "act", "respond"} <= vocabulary.nodes
    assert "cancelled" in vocabulary.failure_codes
    assert vocabulary.failure_categories == frozenset(
        category.value for category in TaskFailureCategory
    )


def test_read_only_status_check_needs_no_tool_registry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(monkeypatch, tmp_path)

    vocabulary = build_task_vocabulary(settings)

    assert vocabulary.tools == frozenset()


def test_open_runtime_task_store_creates_and_reports_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(monkeypatch, tmp_path)
    path = tmp_path / "task_events.db"

    store = open_runtime_task_store(settings, _Registry(), path=path)
    try:
        assert store.available
        assert store.status().available
        assert path.is_file()
    finally:
        store.close()
    assert not store.available


def test_open_runtime_task_store_is_fail_soft_when_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(monkeypatch, tmp_path)
    path = tmp_path / "missing.db"

    store = open_runtime_task_store(settings, _Registry(), path=path, create=False)

    assert isinstance(store, NullTaskEventStore)
    assert store.status().reason == "task_store_unavailable"
    assert not path.exists()


def test_app_context_carries_the_task_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(monkeypatch, tmp_path)
    path = tmp_path / "task_events.db"
    store = open_task_store(
        path,
        create=True,
        vocabulary=build_task_vocabulary(settings, _Registry()),
    )
    try:
        ctx = make_app_context(settings, registry=ToolRegistry(), tasks=store)
        assert ctx.tasks is store
        assert not hasattr(ctx.tool_context(), "tasks")
    finally:
        store.close()


def test_doctor_reports_missing_store_as_warning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from jarvis.cli import _task_store_check

    settings = _settings(monkeypatch, tmp_path)

    check = _task_store_check(settings)

    assert check.status == "WARN"
    assert "not initialized" in check.detail


def test_doctor_reports_readable_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from jarvis.cli import _task_store_check

    settings = _settings(monkeypatch, tmp_path)
    store = open_task_store(
        config.task_events_db(),
        create=True,
        vocabulary=build_task_vocabulary(settings, _Registry()),
    )
    store.close()

    check = _task_store_check(settings)

    assert check.status == "PASS"
    assert "schema v" in check.detail
