"""B7 backend acceptance: the foundations working together.

These are the end-to-end checks the backend roadmap was for: one request must
produce an observable lifecycle in the task-event store, and conversation
context must stay bounded and per-conversation.  Everything here is
deterministic (FakeLLM + fake tools + a temporary SQLite file); the existing
suites cover the individual pieces (routing, memory, voice, invariants).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver

from jarvis.agent.context import make_app_context
from jarvis.agent.conversation import ConversationManager
from jarvis.agent.graph import build_secure_serde
from jarvis.agent.runner import resume_task, run_task
from jarvis.config import Settings
from jarvis.llm.client import FakeLLM
from jarvis.taskstore import TaskStatus
from jarvis.taskstore.runtime import open_runtime_task_store
from support import approve, brain_action, make_spec, registry_with


def _registry(record: list[tuple[str, dict[str, Any]]]) -> Any:
    return registry_with(make_spec("fake_echo", base_tier=1, record=record))


def test_tier1_task_lifecycle_is_observable_end_to_end(tmp_path: Path) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    registry = _registry(record)
    settings = Settings()
    store = open_runtime_task_store(settings, registry, path=tmp_path / "tasks.db")
    conn = sqlite3.connect(tmp_path / "checkpoints.db", check_same_thread=False)
    saver = SqliteSaver(conn, serde=build_secure_serde())
    try:
        ctx = make_app_context(
            settings,
            llm=FakeLLM([brain_action("fake_echo", {"text": "secret request body"})]),
            registry=registry,
            tasks=store,
            conversations=ConversationManager(),
        )

        paused = run_task(ctx, saver, "echo the secret request body", conversation_id="c1")
        assert paused.confirmation is not None
        assert record == []  # nothing executes before approval
        snapshot = store.get(paused.task_id)
        assert snapshot is not None
        assert snapshot.status == TaskStatus.AWAITING_CONFIRMATION

        done = resume_task(ctx, saver, paused.task_id, approve(paused.confirmation))
        assert done.final_answer
        assert record == [("fake_echo", {"text": "secret request body"})]

        snapshot = store.get(done.task_id)
        assert snapshot is not None
        assert snapshot.status == TaskStatus.COMPLETED
        assert snapshot.stage.value == "done"

        events = store.events(done.task_id)
        kinds = [event.event_type.value for event in events]
        assert "task_created" in kinds
        assert "task_started" in kinds
        assert "task_finished" in kinds
        # B1 privacy invariant: the request/tool text never reaches the store.
        rendered = json.dumps([event.detail for event in events])
        assert "secret request body" not in rendered
        assert "echo the secret request body" not in rendered
    finally:
        conn.close()
        store.close()


def test_conversations_do_not_share_context_or_need_a_restart(tmp_path: Path) -> None:
    manager = ConversationManager()
    manager.record("alpha", "t1", "open notepad", "notepad is open", "completed")
    manager.record("beta", "t2", "what is python?", "a language", "answered")

    alpha_block = manager.render_history("alpha") or ""
    assert "notepad" in alpha_block
    assert "python" not in alpha_block
    assert "notepad" not in (manager.render_history("beta") or "")

    # A fresh manager is what a daemon restart looks like: nothing survives.
    restarted = ConversationManager()
    assert restarted.recent_turns("alpha") == []
    assert restarted.render_history("alpha") is None


class _RecorderLLM:
    """Records every brain user prompt so isolation is proven on the real prompt."""

    def __init__(self, decisions: list[Any]) -> None:
        self._decisions = list(decisions)
        self.users: list[str] = []

    def structured(self, *, system: str, user: str, schema: type[Any], **kwargs: Any) -> Any:
        from jarvis.llm.client import Usage

        self.users.append(user)
        return self._decisions.pop(0), Usage(calls=1)


def test_follow_up_uses_only_its_own_conversation_history() -> None:
    manager = ConversationManager()
    manager.record("c1", "t1", "remember Python", "noted", "completed")
    manager.record("c2", "t2", "remember Rust", "noted", "completed")

    from support import brain_conversation

    recorder = _RecorderLLM([brain_conversation("Python"), brain_conversation("Rust")])
    ctx = make_app_context(
        Settings(),
        llm=recorder,
        registry=registry_with(),
        conversations=manager,
    )

    run_task(ctx, None, "which language did I mention?", conversation_id="c1")
    run_task(ctx, None, "which language did I mention?", conversation_id="c2")

    assert len(recorder.users) == 2
    first, second = recorder.users
    assert "Python" in first and "Rust" not in first
    assert "Rust" in second and "Python" not in second
