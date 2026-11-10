"""C1: each channel must deliver the answer it actually asked for.

``compose`` writes a short spoken courtesy ("Done.") for the audio device.  A
terminal user asked a question, so the daemon must deliver the factual answer
there - never the courtesy - and the conversation history must record exactly
what was delivered.  These tests pin the pure selector plus the daemon wiring.
"""

from __future__ import annotations

from typing import Any

from jarvis.agent.answer import NO_ANSWER_MESSAGE, delivered_answer
from jarvis.agent.conversation import ConversationManager
from jarvis.agent.runner import TaskOutcome
from jarvis.agent.state import Plan, StepResult
from jarvis.config import Settings, VoiceSettings
from jarvis.daemon.server import DaemonServer, TaskSlot

# ── the pure selector ─────────────────────────────────────────────────────


def test_terminal_gets_the_factual_answer_not_the_spoken_courtesy() -> None:
    assert (
        delivered_answer(
            voice=False,
            final_answer="Today is Tuesday, 10 November 2026.",
            spoken_answer="Done.",
        )
        == "Today is Tuesday, 10 November 2026."
    )


def test_terminal_keeps_a_unix_style_dash_and_screen_text_intact() -> None:
    # The live defect: U+2011 in a web answer must survive verbatim, and the
    # screen answer is preferred when there is no final answer.
    screen = "Gro\u2011the best answer"
    assert delivered_answer(voice=False, final_answer=None, screen_answer=screen) == screen


def test_a_courtesy_cannot_overwrite_a_factual_answer() -> None:
    delivered = delivered_answer(
        voice=False,
        final_answer="1000",
        screen_answer="125 * 8 = 1000",
        spoken_answer="All set, sir.",
    )
    assert delivered == "1000"


def test_voice_still_speaks_the_short_spoken_answer() -> None:
    assert (
        delivered_answer(
            voice=True,
            final_answer="Notepad is open and ready for you.",
            spoken_answer="Done.",
        )
        == "Done."
    )


def test_voice_falls_back_to_the_final_answer_then_the_safe_refusal() -> None:
    assert delivered_answer(voice=True, final_answer="Notepad is open.") == "Notepad is open."
    assert delivered_answer(voice=True, final_answer=None) == NO_ANSWER_MESSAGE
    assert delivered_answer(voice=False, final_answer="   ") == NO_ANSWER_MESSAGE


# ── daemon wiring ─────────────────────────────────────────────────────────


class _FakeSaver:
    def __init__(self) -> None:
        self.conn = self

    def close(self) -> None:
        pass


def _outcome() -> TaskOutcome:
    return TaskOutcome(
        task_id="t1",
        interrupted=False,
        confirmation=None,
        final_answer="Today is Tuesday, 10 November 2026.",
        error=None,
        halted_reason=None,
        decisions={},
        results=[StepResult(step_id="s1", ok=True, output="2026-11-10", verified=None)],
        spoken_answer="Done.",
        screen_answer="Today is Tuesday, 10 November 2026.",
        state={
            "task_id": "t1",
            "plan": Plan(kind="conversation", goal="date", steps=[]),
            "results": [],
            "final_answer": "Today is Tuesday, 10 November 2026.",
        },
    )


class _FakeStore:
    def get(self, name: str) -> str | None:
        return None


def _run_worker(monkeypatch: Any, source: str) -> tuple[DaemonServer, TaskSlot]:
    import jarvis.agent as agent_mod
    import jarvis.config as config_mod

    manager = ConversationManager()
    server = DaemonServer(settings=Settings(voice=VoiceSettings(enabled=False)), store=_FakeStore())
    server._ctx = type("Ctx", (), {"conversations": manager})()
    monkeypatch.setattr(agent_mod, "open_sqlite_checkpointer", lambda _path: _FakeSaver())
    monkeypatch.setattr(agent_mod, "run_task", lambda *a, **k: _outcome())
    monkeypatch.setattr(config_mod, "checkpoints_db", lambda: ":memory:")

    slot = TaskSlot(
        task_id="t1",
        text="what is today's date",
        source=source,
        owner_id="client-1",
        conversation_id="c1",
    )
    server._worker_run(slot)
    return server, slot


def test_terminal_worker_delivers_the_date_and_records_it(monkeypatch: Any) -> None:
    server, slot = _run_worker(monkeypatch, source="terminal")

    assert slot.result_text == "Today is Tuesday, 10 November 2026."
    turns = server._ctx.conversations.recent_turns("c1")
    assert [(turn.answer, turn.status) for turn in turns] == [
        ("Today is Tuesday, 10 November 2026.", "answered")
    ]


def test_voice_worker_delivers_the_spoken_courtesy(monkeypatch: Any) -> None:
    _, slot = _run_worker(monkeypatch, source="voice")

    assert slot.result_text == "Done."
