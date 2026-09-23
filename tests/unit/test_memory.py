"""Phase 2 memory layer: protocol, NullMemory, config cap, node behaviour."""

from __future__ import annotations

from jarvis.agent.context import make_app_context
from jarvis.agent.nodes.memory_retrieve import memory_retrieve
from jarvis.config import Settings
from jarvis.memory import MemoryBackend, MemoryRecord, NullMemory


def test_null_memory_is_deterministic_and_never_leaks() -> None:
    mem = NullMemory()
    assert mem.retrieve("anything", limit=99) == []
    mem.remember(MemoryRecord(kind="session", text="secret?"))
    assert mem.retrieve("anything", limit=99) == []


def test_memory_record_is_strict() -> None:
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        MemoryRecord(kind="system", text="x")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        MemoryRecord(kind="session", text="")


def test_memory_is_a_runtime_protocol() -> None:
    assert isinstance(NullMemory(), MemoryBackend)


def test_config_memory_caps_retrieval() -> None:
    assert Settings().memory.retrieval_limit == 3
    assert Settings().model_dump()["memory"]["retrieval_limit"] == 3


class _StubMemory:
    """Records the query it received and returns a fixed set of records."""

    def __init__(self, records: list[MemoryRecord]) -> None:
        self._records = list(records)
        self.limit_seen: int | None = None

    def retrieve(self, query: str, limit: int = 3) -> list[MemoryRecord]:
        self.limit_seen = limit
        return self._records[:limit]

    def remember(self, record: MemoryRecord) -> None:
        del record


def test_no_backend_means_empty_context() -> None:
    ctx = make_app_context(Settings())  # memory defaults to None
    out = memory_retrieve({"user_input": "hi"}, ctx)
    assert out == {"memory_context": []}


def test_backend_records_are_capped_by_config_check() -> None:
    records = [
        MemoryRecord(kind=k, text=f"item {k}") for k in ("session", "episodic", "preference")
    ]
    stub = _StubMemory(records)
    ctx = make_app_context(Settings(), memory=stub)  # type: ignore[arg-type]
    out = memory_retrieve({"user_input": "find it"}, ctx)
    assert stub.limit_seen == 3
    assert [item["text"] for item in out["memory_context"]] == [
        "item session",
        "item episodic",
        "item preference",
    ]
    assert any("kind" in item for item in out["memory_context"])


def test_verbose_records_are_truncated_for_the_prompt() -> None:
    long = MemoryRecord(kind="session", text="x" * 5000)
    stub = _StubMemory([long])
    ctx = make_app_context(Settings(), memory=stub)  # type: ignore[arg-type]
    out = memory_retrieve({"user_input": "q"}, ctx)
    assert len(out["memory_context"][0]["text"]) < 5000


def test_backend_failure_never_blocks_a_task() -> None:
    class Broken:
        def retrieve(self, query: str, limit: int = 3) -> list[MemoryRecord]:
            raise RuntimeError("db down")

    ctx = make_app_context(Settings(), memory=Broken())  # type: ignore[arg-type]
    out = memory_retrieve({"user_input": "hi"}, ctx)
    assert out == {"memory_context": []}


def test_catalogue_with_policy_exposes_tiers_internally() -> None:
    from jarvis.tools.registry import build_default_registry

    rows = build_default_registry(Settings()).catalogue_with_policy()
    assert rows  # at least the toolset is registered
    for row in rows:
        assert set(row) == {"name", "description", "base_tier", "windows_only", "timeout_s"}
        assert isinstance(row["base_tier"], int)


def test_catalogue_with_policy_has_no_secrets() -> None:
    from jarvis.tools.registry import build_default_registry

    registry = build_default_registry(Settings())
    # Descriptions legitimately mention "password"/"Tier-2" as *words*; the
    # invariant is that no secret VALUES or API-key-holding fields appear.
    blob = repr(registry.catalogue_with_policy())
    assert "hunter2" not in blob and "sk-" not in blob and "api_key" not in blob
    # No execution-capable fields the LLM could abuse via the plain catalogue.
    plain = registry.catalogue_for_llm()
    for row in plain:
        assert set(row) == {"name", "description", "args_schema"}
        assert "command" not in row and "tier" not in row and "base_tier" not in row
