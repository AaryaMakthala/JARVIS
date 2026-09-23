"""Deterministic fake tools for Phase 1 agent tests (no Windows, no network).

The fake tools only append to a shared ``record`` list; asserting the list is
empty (or has the exact expected entries) is how tests prove a side effect did
or did not happen.  ``verify_ok`` lets a test simulate a post-condition check
that fails (verification failure) while the action itself "succeeded".
"""

from __future__ import annotations

import secrets
import shutil
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from jarvis.agent.schemas import ActionIntent, BrainDecision, ReplanDecision
from jarvis.agent.state import Plan, Step
from jarvis.tools.base import ToolContext, ToolResult, ToolSpec
from jarvis.tools.files import TrashRecord
from jarvis.tools.registry import ToolRegistry


class TextArgs(BaseModel):
    text: str = Field(min_length=1, max_length=64)


def make_spec(
    name: str,
    *,
    base_tier: int,
    record: list[tuple[str, dict[str, Any]]],
    verify_ok: bool = True,
    run_ok: bool = True,
) -> ToolSpec:
    """Build a fake tool that records every call into ``record``.

    ``run_ok=False`` simulates a tool whose *action* fails (the replan path);
    ``verify_ok=False`` simulates a post-condition check that fails while the
    action itself "succeeded".
    """

    def run(args: TextArgs, ctx: ToolContext) -> ToolResult:
        record.append((name, args.model_dump()))
        if not run_ok:
            return ToolResult(ok=False, error=f"{name} could not run {args.text}")
        return ToolResult(ok=True, output=f"{name} ran {args.text}", data={"text": args.text})

    def verify_always(args: TextArgs, result: ToolResult, ctx: ToolContext) -> ToolResult:
        return result.model_copy(update={"verified": verify_ok})

    return ToolSpec(
        name=name,
        description=f"fake tool {name} for tests",
        args_model=TextArgs,
        base_tier=base_tier,
        run=run,
        verify=verify_always,
        describe=lambda args: f"{name} text={args.text!r}",
    )


def registry_with(*specs: ToolSpec) -> ToolRegistry:
    """A fresh registry containing exactly the given specs."""
    registry = ToolRegistry()
    for spec in specs:
        registry.register(spec)
    return registry


def echo_plan(text: str = "hi") -> Plan:
    """A single-step plan that calls the fake Tier-1 tool ``fake_echo``."""
    return Plan(
        goal=f"echo {text}",
        steps=[Step(id="s1", tool="fake_echo", args={"text": text}, rationale="echo it")],
    )


def conversation_plan(answer: str = "That is a great question!") -> Plan:
    """A plan classified as ``conversation`` (no tools, nothing to gate)."""
    return Plan(
        kind="conversation",
        goal="answer the user directly",
        steps=[],
        dialog_answer=answer,
    )


# ── Brain-decision helpers ──────────────────────────────────────────────
#
# The graph now classifies via one structured ``brain`` call; these helpers
# build the transient :class:`BrainDecision` the FakeLLM script returns.
# ``brain_action`` targets a single tool; ``brain_steps`` lets a test hand a
# batch of intents back; the *_variances produce the other request types.


def brain_action(
    tool: str,
    args: dict[str, Any],
    *,
    rationale: str = "do it",
    expect: str = "",
    depends_on_untrusted: bool = False,
) -> BrainDecision:
    """A brain decision that proposes a single-tool action plan."""
    return BrainDecision(
        request_type="action",
        goal=f"act on the request with {tool}",
        actions=[
            ActionIntent(
                tool=tool,
                args=args,
                rationale=rationale,
                expect=expect,
                depends_on_untrusted=depends_on_untrusted,
            )
        ],
    )


def brain_steps(actions: list[ActionIntent]) -> BrainDecision:
    """A brain decision whose plan is exactly the given action intents."""
    return BrainDecision(request_type="action", goal="act on the request", actions=actions)


def brain_conversation(response: str = "That is a great question!") -> BrainDecision:
    """A brain decision that answers the user directly (no tools)."""
    return BrainDecision(request_type="conversation", response_text=response)


def brain_clarify(question: str) -> BrainDecision:
    """A brain decision that must ask the human ``question`` first."""
    return BrainDecision(request_type="clarification", clarification_question=question)


def brain_unsupported(message: str = "I can't do that.") -> BrainDecision:
    """A brain decision that halts: the request is out of scope."""
    return BrainDecision(request_type="unsupported", response_text=message)


def brain_delete(paths: list[str]) -> BrainDecision:
    """An action decision that calls the real ``delete_path`` tool."""
    return BrainDecision(
        request_type="action",
        goal="delete the given paths to the Recycle Bin",
        actions=[
            ActionIntent(
                tool="delete_path",
                args={"paths": list(paths)},
                rationale="delete the given paths (reversible)",
            )
        ],
    )


def replan_continue(plan: Plan) -> ReplanDecision:
    """A replanner decision that continues with a replacement plan."""
    return ReplanDecision(action="continue", plan=plan)


def replan_stop(message: str = "the approach is not working") -> ReplanDecision:
    """A replanner decision that stops with an honest message."""
    return ReplanDecision(action="stop", message=message)


def approve(confirmation: dict[str, Any]) -> dict[str, Any]:
    """A user answer that approves the exact confirmation payload."""
    return {
        "approved": True,
        "action_hash": confirmation["action_hash"],
        "resolved_paths": list(confirmation.get("resolved_paths", [])),
    }


def deny(confirmation: dict[str, Any]) -> dict[str, Any]:
    """A user answer that says no (keeps the hash, but disapproved)."""
    return {"approved": False, "action_hash": confirmation["action_hash"]}


def tamper(confirmation: dict[str, Any]) -> dict[str, Any]:
    """A user answer whose action_hash does not match the payload."""
    return {"approved": True, "action_hash": "0" * 64}


def approve_typed(confirmation: dict[str, Any], typed: str) -> dict[str, Any]:
    """An approval that also types the required folder-name confirmation."""
    return approve(confirmation) | {"typed_confirmation": typed}


def delete_plan(step_id: str = "s1", paths: list[str] | None = None) -> Plan:
    """A single-step plan that calls the real ``delete_path`` tool."""
    return Plan(
        goal="delete the given paths to the Recycle Bin",
        steps=[
            Step(
                id=step_id,
                tool="delete_path",
                args={"paths": list(paths or [])},
                rationale="delete the given paths (reversible)",
            )
        ],
    )


class FakeDirTrash:
    """TrashService that keeps items in a test folder instead of the OS Bin.

    Implements the same :class:`TrashService` contract (``send`` / ``restore``)
    as the real recycle bin so the delete/undo logic can be tested without ever
    touching ``$Recycle.Bin``.  ``send`` records the sha256 of the moved copy,
    exactly like the real implementation on Windows.
    """

    def __init__(self, root: Path) -> None:
        from jarvis.tools.files import _describe_target

        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._describe_target = _describe_target

    def send(self, path: Path) -> TrashRecord:
        dest = self.root / secrets.token_hex(8)
        try:
            shutil.move(str(path), str(dest))
        except OSError as exc:
            return TrashRecord(
                original_path=str(path), kind="file", size=0, item_count=0, error=str(exc)
            )
        kind, size, count, digest = self._describe_target(dest)
        return TrashRecord(
            original_path=str(path.resolve(strict=False)),
            kind=kind,
            size=size,
            item_count=count,
            sha256=digest,
            recycled_path=str(dest),
        )

    def restore(self, record: TrashRecord) -> bool:
        src = Path(record.recycled_path) if record.recycled_path else None
        if src is None or not src.exists():
            return False
        try:
            target = Path(record.original_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(target))
        except OSError:
            return False
        return Path(record.original_path).exists()


# ── Voice helpers ───────────────────────────────────────────────────────


def voice_silence(duration_s: float = 1.0, sample_rate: int = 16_000):
    """Create a silence AudioSegment for voice tests."""
    from jarvis.voice.interfaces import AudioSegment

    n = int(duration_s * sample_rate)
    return AudioSegment(samples=[0.0] * n, sample_rate=sample_rate)


def voice_speech(duration_s: float = 2.0, sample_rate: int = 16_000):
    """Create a fake speech AudioSegment for voice tests."""
    from jarvis.voice.interfaces import AudioSegment

    n = int(duration_s * sample_rate)
    return AudioSegment(samples=[0.5] * n, sample_rate=sample_rate)


def make_voice_loop(
    wake_detected: bool = True,
    transcript: str = "open notepad",
    response: str | None = None,
):
    """Build a VoiceLoop with all fakes for testing."""
    from jarvis.voice.fakes import (
        FakeAudioInput,
        FakeSTT,
        FakeTTS,
        FakeWakeWord,
    )
    from jarvis.voice.loop import VoiceLoop

    wake_results = []
    if wake_detected:
        from jarvis.voice.interfaces import WakeWordResult

        wake_results.append(WakeWordResult(detected=True, confidence=0.9))

    audio = FakeAudioInput(
        segments=[voice_speech(0.5)] * 50,  # plenty of speech segments
    )
    wake = FakeWakeWord(results=wake_results)
    from jarvis.voice.interfaces import STTResult

    stt = FakeSTT(results=[STTResult(text=transcript)])
    tts = FakeTTS()

    loop = VoiceLoop(
        audio=audio,
        wake_detector=wake,
        stt=stt,
        tts=tts,
    )
    return loop, audio, wake, stt, tts
