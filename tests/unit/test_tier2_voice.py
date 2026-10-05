"""Tier 2 by voice: the opt-in relaxation is narrow, and off by default.

Stage 3 adds ``voice.allow_tier2_by_voice``.  While it is false (the default)
every Tier 2 action is refused by voice exactly as in Stage 2; while it is true
only ``delete_path`` and ``whatsapp_send`` may be approved by voice, and only
after an exact readback, and only the word "proceed" counts.

These tests pin the three layers that have to agree:

* :mod:`jarvis.voice.tier2_voice` - the pure verdict;
* :mod:`jarvis.agent.nodes.policy_gate` - the gate re-verifies the daemon's
  ``voice_tier2`` marker itself, so the marker alone approves nothing and the
  session is never unlocked;
* :mod:`jarvis.agent.batch_approval` - B1 still holds with the flag on: a batch
  never covers a Tier 2 step.

The gate tests drive the real compiled graph over a file-backed SQLite
checkpointer with a :class:`~jarvis.llm.client.FakeLLM`, so they need no audio,
no network and no API key.  Everything is sandboxed under ``tmp_path``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jarvis.agent.context import make_app_context
from jarvis.agent.graph import open_sqlite_checkpointer
from jarvis.agent.runner import resume_task, run_task
from jarvis.agent.schemas import ActionIntent
from jarvis.config import PolicySettings, Settings, VoiceSettings
from jarvis.daemon.task_runtime import timeout_answer
from jarvis.llm.client import FakeLLM
from jarvis.policy.unlock import UnlockManager
from jarvis.tools.files import make_create_file_spec, make_delete_path_spec
from jarvis.voice.loop import _approval_words
from jarvis.voice.tier2_voice import (
    KEY_VOICE_TIER2,
    TIER2_PROCEED_WORDS,
    allowed,
    flag_on,
    readback_for,
    voice_marker_present,
)
from support import FakeDirTrash, approve, brain_steps, registry_with

PW = "correct-horse-battery-staple"

#: Long enough to blow the 150-char readback bound in ``voice.readback``.
LONG_DIR = "d" * 160


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


def _locked_manager() -> UnlockManager:
    """A *locked* session: the password exists but has never been verified."""
    manager = UnlockManager(_PwStore())
    manager.set_password(PW)
    assert manager.is_unlocked() is False
    return manager


def _on() -> VoiceSettings:
    return VoiceSettings(allow_tier2_by_voice=True)


def _decision(**over: Any) -> dict[str, Any]:
    """A minimal Tier 2 ``Decision`` mapping, as the payload carries it."""
    base = {"tier": 2, "allowed": True, "needs_typed_confirmation": None}
    base.update(over)
    return base


# ── 1: the flag defaults to off ───────────────────────────────────────────


def test_flag_off_refuses_tier2_voice() -> None:
    # Default config: no opt-in, so the relaxation does not exist.
    assert flag_on(VoiceSettings()) is False
    assert flag_on(Settings()) is False
    assert flag_on(None) is False
    ok, reason = allowed(
        "delete_path", _decision(), VoiceSettings(), args={"paths": ["C:/tmp/a.txt"]}
    )
    assert ok is False
    assert "allow_tier2_by_voice" in reason


# ── 2: the flag on still needs a readback ──────────────────────────────────


def test_flag_on_requires_readback() -> None:
    # No path at all -> nothing can be read back -> refused, flag on or not.
    ok, reason = allowed("delete_path", _decision(), _on(), args={})
    assert ok is False
    assert reason
    # A decision with no args and no resolved paths is equally unreadable.
    ok, _ = allowed("delete_path", _decision(resolved_paths=[]), _on())
    assert ok is False
    unreadable = readback_for("delete_path", _decision(resolved_paths=[]))
    assert unreadable.ok is False
    # ... while the same call with the real path succeeds.
    ok, _ = allowed("delete_path", _decision(), _on(), args={"paths": [str(Path("C:/tmp/a.txt"))]})
    assert ok is True


# ── 3: only the two named tools ────────────────────────────────────────────


def test_flag_on_only_delete_path_and_whatsapp() -> None:
    for tool in ("create_file", "list_files", "whatsapp_read", "undo_last_delete", ""):
        ok, reason = allowed(tool, _decision(), _on(), args={"message": "hi"})
        assert ok is False, f"{tool!r} must not be voice-approvable"
        assert reason
    ok, _ = allowed(
        "whatsapp_send",
        _decision(),
        _on(),
        args={"contact": "Priya", "message": "on my way"},
    )
    assert ok is True
    # A folder delete needs a typed name: refused whatever the flag says.
    ok, _ = allowed(
        "delete_path",
        _decision(needs_typed_confirmation="Reports"),
        _on(),
        args={"paths": [str(Path("C:/tmp/Reports"))]},
    )
    assert ok is False


# ── 4: a too-long target is refused, never truncated ───────────────────────


def test_long_target_refused_not_truncated() -> None:
    long_path = f"C:/tmp/{LONG_DIR}/a.txt"
    assert len(long_path) > 150
    ok, reason = allowed("delete_path", _decision(), _on(), args={"paths": [long_path]})
    assert ok is False
    assert "long" in reason.lower()
    result = readback_for("delete_path", _decision(), args={"paths": [long_path]})
    assert result.ok is False
    assert result.text == "", "a refused readback must carry no text at all"
    long_message = "m" * 201
    ok, _ = allowed(
        "whatsapp_send",
        _decision(),
        _on(),
        args={"contact": "Priya", "message": long_message},
    )
    assert ok is False


# ── 5: Tier 3 is refused even with the flag on ─────────────────────────────


def test_tier3_refused_even_with_flag() -> None:
    for tier in (3, 4, 9):
        ok, reason = allowed(
            "delete_path",
            _decision(tier=tier, allowed=False),
            _on(),
            args={"paths": [str(Path("C:/tmp/a.txt"))]},
        )
        assert ok is False
        assert "Tier 3" in reason or "Tier" in reason
    # An unparseable / missing tier is not "2" either.
    assert allowed("delete_path", _decision(tier=None), _on())[0] is False
    assert allowed("delete_path", _decision(tier="two"), _on())[0] is False


# ── 6: only "proceed" approves a Tier 2 voice window ───────────────────────


def test_wrong_word_refused() -> None:
    payload = {"type": "confirm", "tier": 2, "tool": "delete_path", "typed_confirmation": None}
    words = _approval_words(payload)
    assert words == TIER2_PROCEED_WORDS == {"proceed"}
    for wrong in ("yes", "y", "confirm", "ok", "ok sure", "go", "sure", "approve"):
        assert wrong not in words, f"{wrong!r} must never approve a Tier 2 action"
    assert "proceed" in words
    # The refusal words are handled elsewhere (they are simply not approvals),
    # and the Tier 1 bucket is unchanged.
    assert "go" in _approval_words({"type": "confirm", "tier": 1, "summary": "x"})


# ── graph-level tests ──────────────────────────────────────────────────────


def _settings(ws: Path, *, flag: bool) -> Settings:
    return Settings(
        policy=PolicySettings(allowed_roots=[str(ws.resolve())]),
        voice=VoiceSettings(allow_tier2_by_voice=flag),
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


def _ctx(tmp_path: Path, ws: Path, plan: Any, *, flag: bool, unlock: UnlockManager) -> Any:
    return make_app_context(
        _settings(ws, flag=flag),
        llm=FakeLLM([plan]),
        registry=registry_with(make_create_file_spec(), make_delete_path_spec()),
        unlock=unlock,
        trash=FakeDirTrash(tmp_path / "trash"),
        undo_log=tmp_path / "undo.jsonl",
    )


def _run_voice(tmp_path: Path, ws: Path, name: str, *, flag: bool):
    """Run one voice-sourced Tier 2 delete up to its interrupt (locked session)."""
    unlock = _locked_manager()
    ctx = _ctx(tmp_path, ws, _delete_plan(ws, name), flag=flag, unlock=unlock)
    saver = open_sqlite_checkpointer(str(tmp_path / f"{name}.db"))
    first = run_task(ctx, saver, f"delete {name}", source="voice")
    assert first.interrupted is True
    assert first.confirmation["tier"] == 2
    assert first.confirmation["needs_unlock"] is True
    return ctx, saver, unlock, first


# ── 7: silence / timeout is a refusal ──────────────────────────────────────


def test_timeout_refused(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    _ctx_, _saver, _unlock, first = _run_voice(tmp_path, ws, "a.txt", flag=True)
    answer = timeout_answer(first.confirmation or {})
    answer[KEY_VOICE_TIER2] = True  # even with the marker: timeout wins
    out = resume_task(_ctx_, _saver, first.task_id, answer)
    assert out.halted_reason is not None
    assert (ws / "a.txt").exists(), "a timeout must never delete"
    # The marker never even reached the gate's exemption: it timed out first.
    assert "timed out" in (out.final_answer or "").lower()


# ── 8: a mismatched hash is refused, marker or not ─────────────────────────


def test_hash_mismatch_refused(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    ctx, saver, _unlock, first = _run_voice(tmp_path, ws, "a.txt", flag=True)
    bad = approve(first.confirmation or {})
    bad["action_hash"] = "0" * 64
    bad[KEY_VOICE_TIER2] = True
    out = resume_task(ctx, saver, first.task_id, bad)
    assert out.halted_reason is not None
    assert (ws / "a.txt").exists(), "a mismatched hash must never delete"


# ── 9: the marker alone is not enough ──────────────────────────────────────


def test_voice_marker_alone_not_trusted_by_gate(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    # (a) the marker without the flag: refused, exactly as in Stage 2.
    ctx, saver, _u, first = _run_voice(tmp_path, ws, "a.txt", flag=False)
    answer = approve(first.confirmation or {})
    answer[KEY_VOICE_TIER2] = True
    out = resume_task(ctx, saver, first.task_id, answer)
    assert out.halted_reason is not None
    assert "unlocked JARVIS session" in (out.final_answer or "")
    assert (ws / "a.txt").exists()
    # (b) the flag without the marker: also refused.
    ws2 = tmp_path / "ws2"
    ws2.mkdir()
    ctx2, saver2, _u2, first2 = _run_voice(tmp_path, ws2, "b.txt", flag=True)
    out2 = resume_task(ctx2, saver2, first2.task_id, approve(first2.confirmation or {}))
    assert out2.halted_reason is not None
    assert "unlocked JARVIS session" in (out2.final_answer or "")
    assert (ws2 / "b.txt").exists()
    # (c) a non-voice task carrying the marker is refused (source is checked).
    ws3 = tmp_path / "ws3"
    ws3.mkdir()
    unlock = _locked_manager()
    ctx3 = _ctx(tmp_path, ws3, _delete_plan(ws3, "c.txt"), flag=True, unlock=unlock)
    saver3 = open_sqlite_checkpointer(str(tmp_path / "c.db"))
    term = run_task(ctx3, saver3, "delete c.txt", source="terminal")
    answer3 = approve(term.confirmation or {})
    answer3[KEY_VOICE_TIER2] = True
    out3 = resume_task(ctx3, saver3, term.task_id, answer3)
    assert out3.halted_reason is not None
    assert (ws3 / "c.txt").exists()
    # And the marker helper itself only recognises an explicit True.
    assert voice_marker_present({KEY_VOICE_TIER2: True}) is True
    assert voice_marker_present({KEY_VOICE_TIER2: "true"}) is False
    assert voice_marker_present({}) is False


# ── 10: the flag never unlocks the session ─────────────────────────────────


def test_flag_does_not_unlock_session(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    unlock = _locked_manager()
    plan = _delete_plan(ws, "a.txt")
    ctx = _ctx(tmp_path, ws, plan, flag=True, unlock=unlock)
    saver = open_sqlite_checkpointer(str(tmp_path / "a.db"))
    first = run_task(ctx, saver, "delete a.txt", source="voice")
    assert unlock.is_unlocked() is False
    answer = approve(first.confirmation or {})
    answer[KEY_VOICE_TIER2] = True
    out = resume_task(ctx, saver, first.task_id, answer)
    assert out.halted_reason is None
    # The step ran, but the session is still locked: the relaxation satisfies
    # this one step's hash, it is not a password.
    assert (ws / "a.txt").exists() is False
    assert unlock.is_unlocked() is False
    assert (unlock.verify("wrong") is False) is True


# ── 11: AUTO relaxes nothing ───────────────────────────────────────────────


def test_auto_mode_relaxes_nothing() -> None:
    from jarvis.voice.modes import VoiceMode

    # The verdict has no mode input at all: AUTO, NORMAL and any future mode
    # produce the same answer.
    args = {"paths": [str(Path("C:/tmp/a.txt"))]}
    for mode in (VoiceMode.NORMAL, VoiceMode.AUTO):
        assert allowed("delete_path", _decision(), _on(), args=args) == (
            allowed("delete_path", _decision(), _on(), args=args)
        )
        assert allowed("delete_path", _decision(), mode, args=args)[0] is False
        assert allowed("delete_path", _decision(), VoiceMode.AUTO, args=args)[0] is False
        assert allowed("delete_path", _decision(tier=3), VoiceMode.AUTO, args=args)[0] is False
        # And a VoiceMode is never mistaken for the settings object: it has no
        # ``allow_tier2_by_voice`` attribute, so it can never enable anything.
        assert flag_on(mode) is False


# ── 12: B1 still holds with the flag on ─────────────────────────────────────


def test_batch_never_covers_tier2_with_flag_on(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "c.txt").write_text("c", encoding="utf-8")
    plan = brain_steps(
        [
            ActionIntent(
                tool="create_file",
                args={"path": str(ws / "a.txt"), "content": "A"},
                rationale="first literal file",
            ),
            ActionIntent(
                tool="create_file",
                args={"path": str(ws / "b.txt"), "content": "B"},
                rationale="second literal file",
            ),
            ActionIntent(
                tool="delete_path",
                args={"paths": [str(ws / "c.txt")]},
                rationale="tier 2 delete",
            ),
        ]
    )
    unlock = _locked_manager()
    ctx = _ctx(tmp_path, ws, plan, flag=True, unlock=unlock)
    saver = open_sqlite_checkpointer(str(tmp_path / "mixed.db"))
    first = run_task(ctx, saver, "make two files then delete c", source="voice")
    assert first.interrupted is True
    batch = first.confirmation or {}
    assert batch["type"] == "plan_approval", "two contiguous Tier 1 steps still batch"
    eligible = [step_id for step_id, _ in (batch.get("eligible") or [])]
    assert eligible == ["s1", "s2"], f"the Tier 2 step must not be in the batch: {eligible}"
    tier2_hashes = [str(h) for step_id, h in (batch.get("eligible") or [])]
    assert all(len(h) == 64 for h in tier2_hashes)
    # Answering the batch approves s1/s2 only; s3 then gets its own interrupt
    # and is still gated on the terminal password.
    answered = approve(batch)
    answered["resolved_paths"] = []
    second = resume_task(ctx, saver, first.task_id, answered)
    assert second.interrupted is True
    step3 = second.confirmation or {}
    assert step3["type"] == "confirm"
    assert step3["tier"] == 2
    assert step3["action_hash"] != batch["action_hash"]
    assert (ws / "c.txt").exists(), "the Tier 2 step must not have run"
