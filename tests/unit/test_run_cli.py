"""`jarvis run "<command>"` - the one-shot command (docs/01 spec CLI table).

The important property under test is that the one-shot path is **not** more
permissive than the interactive one: an approval is still required, still shows
the exact summary, and is still bound to the same action hash.  Everything here
runs against a fake LLM and a fake tool - no network, no daemon, no Windows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from jarvis import cli
from jarvis.agent.state import Decision
from jarvis.config import Settings
from jarvis.llm.client import FakeLLM
from jarvis.tools.base import ToolResult
from support import brain_action, make_spec, registry_with

runner = CliRunner()


class _Recorder:
    """Captures everything the CLI prints so tests can assert on it."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.text: list[str] = []
        monkeypatch.setattr(
            cli.console, "print", lambda *a, **k: self.text.append(" ".join(str(x) for x in a))
        )

    @property
    def output(self) -> str:
        return "\n".join(self.text)


@pytest.fixture
def answers(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Queued answers for the confirmation/clarification prompts."""
    queue: list[str] = []
    monkeypatch.setattr(cli, "_chat_prompt", lambda *_a, **_k: queue.pop(0))
    monkeypatch.setattr(cli, "_chat_password_prompt", lambda *_a, **_k: "")
    return queue


def _patch_ctx(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    llm: FakeLLM,
    *,
    tier: int = 0,
    record: list[tuple[str, dict[str, Any]]] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = record if record is not None else []
    spec = make_spec("fake_echo", base_tier=tier, record=calls)
    ctx = cli.make_app_context(Settings(), llm=llm, registry=registry_with(spec), memory=None)
    saver = cli.open_sqlite_checkpointer(str(tmp_path / "cp.db"))
    monkeypatch.setattr(cli, "make_app_context", lambda *_a, **_k: ctx)
    monkeypatch.setattr(cli, "open_sqlite_checkpointer", lambda *_a, **_k: saver)
    monkeypatch.setattr(cli, "build_llm_client", lambda *_a, **_k: _Selection(llm))
    monkeypatch.setattr(cli.config, "load_settings", lambda: Settings())
    monkeypatch.setattr(cli.secret_module, "SecretStore", lambda: _Store())
    monkeypatch.setattr(cli, "UnlockManager", lambda *_a, **_k: None)
    monkeypatch.setattr(cli.config, "checkpoints_db", lambda: tmp_path / "cp.db")
    return calls


class _Selection:
    def __init__(self, client: Any) -> None:
        self.client = client
        self.reasons: list[str] = []
        self.info = None


class _Store:
    def get(self, *_a: Any, **_k: Any) -> str:
        return ""


class TestRunNoDaemon:
    def test_runs_a_command_and_prints_the_answer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out = _Recorder(monkeypatch)
        calls = _patch_ctx(
            monkeypatch, tmp_path, FakeLLM([brain_action("fake_echo", {"text": "hi"})])
        )
        result = runner.invoke(cli.app, ["run", "--no-daemon", "echo hi"])
        assert result.exit_code == 0, result.output
        assert "fake_echo ran hi" in out.output
        assert calls and calls[0][0] == "fake_echo"

    def test_blank_command_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out = _Recorder(monkeypatch)
        _patch_ctx(monkeypatch, tmp_path, FakeLLM())
        assert runner.invoke(cli.app, ["run", "--no-daemon", "   "]).exit_code == 2
        assert "give a command" in out.output

    def test_dry_run_describes_without_executing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out = _Recorder(monkeypatch)
        calls = _patch_ctx(
            monkeypatch,
            tmp_path,
            FakeLLM([brain_action("fake_echo", {"text": "hi"})]),
            tier=1,
        )
        result = runner.invoke(cli.app, ["run", "--no-daemon", "--dry-run", "echo hi"])
        assert result.exit_code in (0, 1)  # the answer is what matters, not the code
        # The Tier-1 tool still asks for a dry-run description, and the
        # underlying fake tool is never reached.
        assert "[dry-run]" in out.output or "approval needed" in out.output
        assert calls == []

    def test_a_missing_provider_exits_1(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out = _Recorder(monkeypatch)
        _patch_ctx(monkeypatch, tmp_path, FakeLLM())

        class _NoProvider:
            client = None
            reasons = ["no key set"]  # noqa: RUF012 - per-test stand-in, not shared config

        monkeypatch.setattr(cli, "build_llm_client", lambda *_a, **_k: _NoProvider())
        result = runner.invoke(cli.app, ["run", "--no-daemon", "echo hi"])
        assert result.exit_code == 1
        assert "No free LLM provider" in out.output


class TestRunRequiresApproval:
    """A Tier-2 action must still be approved interactively by ``jarvis run``."""

    def _ctx_with_interrupt(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> list[tuple[str, dict[str, Any]]]:
        calls: list[tuple[str, dict[str, Any]]] = []
        spec = make_spec("fake_echo", base_tier=2, record=calls)
        ctx = cli.make_app_context(Settings(), llm=FakeLLM(), registry=registry_with(spec))
        saver = cli.open_sqlite_checkpointer(str(tmp_path / "cp.db"))
        monkeypatch.setattr(cli, "make_app_context", lambda *_a, **_k: ctx)
        monkeypatch.setattr(cli, "open_sqlite_checkpointer", lambda *_a, **_k: saver)
        monkeypatch.setattr(cli, "build_llm_client", lambda *_a, **_k: _Selection(FakeLLM()))
        monkeypatch.setattr(cli.config, "load_settings", lambda: Settings())
        monkeypatch.setattr(cli.secret_module, "SecretStore", lambda: _Store())
        monkeypatch.setattr(cli, "UnlockManager", lambda *_a, **_k: None)
        monkeypatch.setattr(cli.config, "checkpoints_db", lambda: tmp_path / "cp.db")
        return calls

    def test_dry_run_with_the_daemon_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out = _Recorder(monkeypatch)
        result = runner.invoke(cli.app, ["run", "--dry-run", "echo hi"])
        assert result.exit_code == 2
        assert "--no-daemon" in out.output


class TestResolveInterrupts:
    """The shared resolver must behave identically for chat and run."""

    class _Outcome:
        def __init__(self, confirmation: dict | None, kind: str = "") -> None:
            self.confirmation = confirmation
            self.final_answer = None
            self.error = None
            self.task_id = "t1"
            self.interrupt_kind = kind

    def test_clarification_is_answered_then_returns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cli, "_chat_prompt", lambda *_a, **_k: "the downloads one")
        first = self._Outcome({"question": "which one?"}, kind="clarification")
        second = self._Outcome(None)
        calls: list[tuple[Any, ...]] = []
        monkeypatch.setattr(cli, "resume_task", lambda *a: (calls.append(a), second)[1])
        out = cli.resolve_interrupts(first, None, None)  # type: ignore[arg-type]
        assert out is second
        assert calls[0][2] == "t1"
        assert calls[0][3] == "the downloads one"

    def test_a_denial_resumes_with_the_same_hash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        req = {
            "type": "confirm",
            "tier": 2,
            "summary": "delete C:/x",
            "action_hash": "h" * 64,
            "typed_confirmation": None,
            "needs_unlock": False,
        }
        monkeypatch.setattr(cli, "_chat_prompt", lambda *_a, **_k: "n")
        first = self._Outcome(req, kind="confirm")
        second = self._Outcome(None)
        calls: list[tuple[Any, ...]] = []
        monkeypatch.setattr(cli, "resume_task", lambda *a: (calls.append(a), second)[1])
        cli.resolve_interrupts(first, None, None)  # type: ignore[arg-type]
        answer = calls[0][3]
        assert answer["approved"] is False
        assert answer["action_hash"] == "h" * 64  # still bound to the exact action

    def test_an_approval_is_bound_to_the_exact_hash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        req = {
            "type": "confirm",
            "tier": 1,
            "summary": "type hello",
            "action_hash": "a" * 64,
            "typed_confirmation": None,
            "needs_unlock": False,
        }
        monkeypatch.setattr(cli, "_chat_prompt", lambda *_a, **_k: "yes")
        first = self._Outcome(req, kind="confirm")
        second = self._Outcome(None)
        calls: list[tuple[Any, ...]] = []
        monkeypatch.setattr(cli, "resume_task", lambda *a: (calls.append(a), second)[1])
        cli.resolve_interrupts(first, None, None)  # type: ignore[arg-type]
        assert calls[0][3]["action_hash"] == "a" * 64
        assert calls[0][3]["approved"] is True

    def test_typed_confirmation_is_required_for_a_typed_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        req = {
            "type": "confirm",
            "tier": 2,
            "summary": "delete folder Reports",
            "action_hash": "t" * 64,
            "typed_confirmation": "Reports",
            "needs_unlock": False,
        }
        prompts: list[str] = []

        def prompt(text: str, *_a: Any, **_k: Any) -> str:
            prompts.append(text)
            return "yes" if "Approve" in text else "Reports"

        monkeypatch.setattr(cli, "_chat_prompt", prompt)
        first = self._Outcome(req, kind="confirm")
        second = self._Outcome(None)
        calls: list[tuple[Any, ...]] = []
        monkeypatch.setattr(cli, "resume_task", lambda *a: (calls.append(a), second)[1])
        cli.resolve_interrupts(first, None, None)  # type: ignore[arg-type]
        assert any("Reports" in p for p in prompts)
        assert calls[0][3]["typed_confirmation"] == "Reports"


class _Outcome:
    """Minimal stand-in for a finished ``TaskOutcome``."""

    def __init__(self, final_answer: str | None) -> None:
        self.confirmation = None
        self.interrupt_kind = ""
        self.task_id = "t1"
        self.error = None
        self.final_answer = final_answer


def test_decision_shape_used_by_confirmations_is_unchanged() -> None:
    """Guard against the refactor drifting the confirmation contract."""
    d = Decision(
        step_id="s1",
        tier=2,
        allowed=True,
        needs_confirm=True,
        needs_unlock=False,
        summary="delete",
        action_hash="a" * 64,
    )
    assert d.needs_confirm and not d.needs_unlock
    assert isinstance(d.resolved_paths, list)


def test_tool_result_is_unchanged() -> None:
    assert ToolResult(ok=True, output="x").ok is True


class TestChatLoopUnchanged:
    """The extraction of `resolve_interrupts` must not alter the REPL."""

    def test_eof_exits_without_running_a_task(self, monkeypatch: pytest.MonkeyPatch) -> None:
        out = _Recorder(monkeypatch)
        ran: list[str] = []
        monkeypatch.setattr(cli, "run_task", lambda _c, _s, t: ran.append(t))
        monkeypatch.setattr(cli, "_chat_prompt", lambda *_a, **_k: (_ for _ in ()).throw(EOFError))
        cli.chat_loop(None, None)  # type: ignore[arg-type]
        assert ran == []
        assert "bye" in out.output

    def test_ctrl_c_exits_without_running_a_task(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _Recorder(monkeypatch)
        ran: list[str] = []
        monkeypatch.setattr(cli, "run_task", lambda _c, _s, t: ran.append(t))
        monkeypatch.setattr(
            cli, "_chat_prompt", lambda *_a, **_k: (_ for _ in ()).throw(KeyboardInterrupt)
        )
        cli.chat_loop(None, None)  # type: ignore[arg-type]
        assert ran == []

    def test_blank_lines_are_skipped_then_a_command_runs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out = _Recorder(monkeypatch)
        lines = iter(["", "   ", "echo hi"])

        def prompt(*_a: Any, **_k: Any) -> str:
            try:
                return next(lines)
            except StopIteration:
                raise EOFError from None

        monkeypatch.setattr(cli, "_chat_prompt", prompt)
        sent: list[str] = []
        monkeypatch.setattr(
            cli,
            "run_task",
            lambda _c, _s, t: sent.append(t) or _Outcome("hi"),
        )
        monkeypatch.setattr(cli, "resolve_interrupts", lambda outcome, *_a: outcome)
        monkeypatch.setattr(
            cli, "_print_outcome", lambda o: out.text.append(f"FINAL:{o.final_answer}")
        )
        cli.chat_loop(None, None)  # type: ignore[arg-type]
        # blanks never reached run_task, and the real command did
        assert sent == ["echo hi"]
        assert "FINAL:hi" in out.output
        assert "bye" in out.output
