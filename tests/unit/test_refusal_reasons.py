"""The gate must name *why* a confirmation was refused (wording, not safety).

Before this, every non-approving resume answer produced one sentence - "Refused
by you (confirmation answered with 'no' or a mismatched action)" - which was
wrong whenever the user was never asked: a Tier 2 voice action while
``voice.allow_tier2_by_voice`` is off, or a terminal approval with no JARVIS
password set.

These tests pin the *wording* only.  Nothing here may change an approval
decision, an action hash or a tier - see
:func:`test_reason_never_changes_decision`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from jarvis import cli
from jarvis.agent.context import make_app_context
from jarvis.agent.graph import open_sqlite_checkpointer
from jarvis.agent.runner import resume_task, run_task
from jarvis.agent.schemas import ActionIntent
from jarvis.config import PolicySettings, Settings, VoiceSettings
from jarvis.daemon.server import DaemonServer, TaskSlot
from jarvis.llm.client import FakeLLM
from jarvis.policy.refusal import (
    DEFAULT_REFUSAL_TEXT,
    REASON_MISMATCH,
    REASON_NO_PASSWORD,
    REASON_TIER2_VOICE_DISABLED,
    REASON_USER_DECLINED,
    REFUSAL_TEXT,
    refusal_answer,
    refusal_text,
)
from jarvis.policy.unlock import UnlockManager
from jarvis.tools.files import make_create_file_spec, make_delete_path_spec
from support import FakeDirTrash, approve, brain_steps, registry_with

PAYLOAD: dict[str, Any] = {
    "tier": 2,
    "summary": "delete one file",
    "action_hash": "h" * 64,
    "typed_confirmation": None,
}


class _PwStore:
    """In-memory password store (no keyring, no real secrets)."""

    def __init__(self) -> None:
        self._values: dict[str, str] = {}

    def get(self, name: str) -> str | None:
        return self._values.get(name)

    def set(self, name: str, value: str) -> None:
        self._values[name] = value

    def has(self, name: str) -> bool:
        return name in self._values


class _NoPasswordStore:
    """Secret store with no ipc_token - the server only needs to construct."""

    def get(self, name: str) -> str | None:
        return None

    def set(self, name: str, value: str) -> None:
        raise AssertionError("must not write")

    def has(self, name: str) -> bool:
        return False

    def check_store_access(self) -> str:
        return "FakeStore"


# â”€â”€ the voice path: flag off -> "voice is turned off" â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class _RefusingLoopStub:
    """A live loop that refuses the payload, exactly as the flag-off loop does."""

    def __init__(self) -> None:
        self.windows_opened = 0

    def is_active(self) -> bool:
        return True

    def can_confirm_by_voice(self, payload: dict[str, Any]) -> bool:
        return False

    def confirm_by_voice(self, *args: Any, **kwargs: Any) -> str:
        self.windows_opened += 1
        raise AssertionError("no window may open when can_confirm_by_voice is False")


class _RefusingServer(DaemonServer):
    def __init__(self, loop: Any) -> None:
        super().__init__(
            settings=Settings(voice=VoiceSettings(enabled=False)),
            store=_NoPasswordStore(),
        )
        self._loop = loop

    def _ctx_voice_loop(self) -> Any:
        return self._loop


def _voice_flag_off_answer(flag: bool) -> dict[str, Any]:
    """The resume answer the daemon produces for a Tier 2 voice confirmation."""
    loop = _RefusingLoopStub()
    server = _RefusingServer(loop)
    slot = TaskSlot(task_id="t1", text="delete text.txt", source="voice")
    server._run_voice_confirmation(slot, dict(PAYLOAD))
    assert loop.windows_opened == 0, "the refusal must happen before any window"
    answer = slot.resume_answer
    assert answer is not None
    return answer


def test_flag_off_reason_is_voice_disabled() -> None:
    # The flag is off in the real config, so Tier 2 refuses here.
    answer = _voice_flag_off_answer(False)
    assert answer == {
        "approved": False,
        "action_hash": "h" * 64,
        "reason": REASON_TIER2_VOICE_DISABLED,
    }
    assert refusal_text(answer) == REFUSAL_TEXT[REASON_TIER2_VOICE_DISABLED]
    assert "turned off" in refusal_text(answer)

    # Tier 3 and Tier 1 are NOT "voice is turned off" - keep the old wording so
    # the message never misdescribes why a Tier 3 was blocked.
    for tier in (1, 3):
        other = dict(PAYLOAD, tier=tier)
        assert _reason_for_tier(other) is None


def _reason_for_tier(payload: dict[str, Any]) -> str | None:
    loop = _RefusingLoopStub()
    server = _RefusingServer(loop)
    slot = TaskSlot(task_id="t1", text="delete text.txt", source="voice")
    server._run_voice_confirmation(slot, payload)
    answer = slot.resume_answer or {}
    reason = answer.get("reason")
    return reason if isinstance(reason, str) else None


# â”€â”€ the terminal path: no password -> "no password is set" â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def test_no_password_reason() -> None:
    ctx = SimpleNamespace(unlock=UnlockManager(_PwStore()))  # no password set
    approved, reason = cli._tier2_unlock_or_refuse(ctx, {"action_hash": "h" * 64})
    assert approved is False
    assert reason == REASON_NO_PASSWORD
    assert refusal_text({"approved": False, "reason": reason}) == REFUSAL_TEXT[REASON_NO_PASSWORD]
    assert "jarvis password set" in REFUSAL_TEXT[REASON_NO_PASSWORD]


def test_wrong_password_reason_is_unnamed(monkeypatch: pytest.MonkeyPatch) -> None:
    # A wrong password is neither "cancelled" nor "no password set"; the gate
    # keeps today's wording rather than saying something untrue.
    monkeypatch.setattr(cli, "_chat_password_prompt", lambda text: "wrong-password")
    manager = UnlockManager(_PwStore())
    manager.set_password("correct-horse-battery-staple")
    approved, reason = cli._tier2_unlock_or_refuse(SimpleNamespace(unlock=manager), {})
    assert approved is False
    assert reason is None


# â”€â”€ a real "no" still reads as a cancellation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def test_user_no_still_says_cancelled() -> None:
    answer = refusal_answer(PAYLOAD, REASON_USER_DECLINED)
    assert answer == {
        "approved": False,
        "action_hash": "h" * 64,
        "reason": REASON_USER_DECLINED,
    }
    assert refusal_text(answer) == "Cancelled."


# â”€â”€ a reason is wording only: never an approval â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def _locked_ctx(tmp_path: Path, ws: Path, plan: Any, *, flag: bool) -> Any:
    manager = UnlockManager(_PwStore())
    manager.set_password("correct-horse-battery-staple")
    return make_app_context(
        Settings(
            policy=PolicySettings(allowed_roots=[str(ws.resolve())]),
            voice=VoiceSettings(allow_tier2_by_voice=flag),
        ),
        llm=FakeLLM([plan]),
        registry=registry_with(make_create_file_spec(), make_delete_path_spec()),
        unlock=manager,
        trash=FakeDirTrash(tmp_path / "trash"),
        undo_log=tmp_path / "undo.jsonl",
    )


def _delete_plan(ws: Path, name: str) -> Any:
    (ws / name).write_text("content", encoding="utf-8")
    return brain_steps(
        [
            ActionIntent(
                tool="delete_path",
                args={"paths": [str(ws / name)]},
                rationale="delete the file (reversible)",
            )
        ]
    )


def _resume_with_reason(tmp_path: Path, ws: Path, name: str, reason: str | None) -> Any:
    """Run a locked Tier 2 delete to its interrupt, resume with *reason*."""
    ctx = _locked_ctx(tmp_path, ws, _delete_plan(ws, name), flag=False)
    saver = open_sqlite_checkpointer(str(tmp_path / f"{name}.db"))
    first = run_task(ctx, saver, f"delete {name}", source="terminal")
    assert first.interrupted is True
    assert (first.confirmation or {})["tier"] == 2
    payload = first.confirmation or {}
    return resume_task(ctx, saver, first.task_id, refusal_answer(payload, reason))


def test_reason_never_changes_decision(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()

    # (a) every reason refuses, keeps the hash, and never deletes.
    for index, reason in enumerate(
        (
            REASON_TIER2_VOICE_DISABLED,
            REASON_NO_PASSWORD,
            REASON_USER_DECLINED,
            REASON_MISMATCH,
        )
    ):
        name = f"a{index}.txt"
        answer = refusal_answer(PAYLOAD, reason)
        assert answer["approved"] is False
        assert answer["action_hash"] == "h" * 64
        out = _resume_with_reason(tmp_path, ws, name, reason)
        assert out.halted_reason is not None
        assert (ws / name).exists(), f"{reason} must never delete"
        assert refusal_text(answer) in (out.final_answer or "")

    # (b) an unnamed refusal is byte-identical to today's payload.
    assert refusal_answer(PAYLOAD) == {"approved": False, "action_hash": "h" * 64}

    # (c) a reason cannot rescue a mismatched hash.
    ws2 = tmp_path / "ws2"
    ws2.mkdir()
    ctx = _locked_ctx(tmp_path, ws2, _delete_plan(ws2, "b.txt"), flag=False)
    saver = open_sqlite_checkpointer(str(tmp_path / "b.db"))
    first = run_task(ctx, saver, "delete b.txt", source="terminal")
    bad = refusal_answer(first.confirmation or {}, REASON_USER_DECLINED)
    bad.update(approve(first.confirmation or {}))  # approved=True, hash tampered
    bad["action_hash"] = "0" * 64
    out = resume_task(ctx, saver, first.task_id, bad)
    assert out.halted_reason is not None
    assert (ws2 / "b.txt").exists()

    # (d) a reason on an *approved* answer is ignored - it is not a second gate
    #     and it cannot unlock anything; the approved path still needs the hash.
    ws3 = tmp_path / "ws3"
    ws3.mkdir()
    manager = UnlockManager(_PwStore())
    manager.set_password("correct-horse-battery-staple")
    unlocked = manager.verify("correct-horse-battery-staple")
    assert unlocked is True
    ctx3 = make_app_context(
        Settings(
            policy=PolicySettings(allowed_roots=[str(ws3.resolve())]),
            voice=VoiceSettings(allow_tier2_by_voice=False),
        ),
        llm=FakeLLM([_delete_plan(ws3, "c.txt")]),
        registry=registry_with(make_create_file_spec(), make_delete_path_spec()),
        unlock=manager,
        trash=FakeDirTrash(tmp_path / "trash"),
        undo_log=tmp_path / "undo.jsonl",
    )
    saver3 = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    term = run_task(ctx3, saver3, "delete c.txt", source="terminal")
    good = approve(term.confirmation or {})
    good["reason"] = REASON_TIER2_VOICE_DISABLED
    resume_task(ctx3, saver3, term.task_id, good)
    assert not (ws3 / "c.txt").exists(), "an unlocked session still deletes normally"


def test_unknown_reason_is_not_spoken() -> None:
    # Only the four known codes become text: a client cannot inject a sentence
    # into JARVIS's own answer by setting "reason".
    for bogus in ("ignore previous instructions", "", None, 7, {"a": 1}):
        assert refusal_text({"approved": False, "reason": bogus}) == DEFAULT_REFUSAL_TEXT
    assert refusal_text("not a dict") == DEFAULT_REFUSAL_TEXT
    assert refusal_text(None) == DEFAULT_REFUSAL_TEXT
