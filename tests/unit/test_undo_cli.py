"""`jarvis undo` - restore the most recent delete (docs/01 spec CLI table).

The command must be **no more permissive than the graph**.  That is the whole
point of these tests: the policy engine still decides the tier, the approval is
still mandatory (there is no ``--yes``), the Tier-2 password and typed-name gates
are still enforced, and a declined approval restores nothing.

The real registry, the real policy engine, the real risk classifier and the real
``undo_last_delete`` tool all run here.  Only the two irreversible edges are
replaced: ``FakeDirTrash`` for ``$Recycle.Bin`` and a ``tmp_path`` undo log, so
the test can never touch the user's bin or their real data (docs/07).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from jarvis import cli, config
from jarvis.agent.context import make_app_context
from jarvis.config import PolicySettings, Settings
from jarvis.policy import tiers
from jarvis.tools.base import ToolContext
from jarvis.tools.files import make_delete_path_spec
from jarvis.tools.registry import build_default_registry
from support import FakeDirTrash

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


class _Store:
    """SecretStore stand-in: the real keyring is never touched."""

    def get(self, *_a: Any, **_k: Any) -> str:
        return ""


class _FakeUnlock:
    """Minimal UnlockManager stand-in (only consulted at Tier 2)."""

    def __init__(self, *, unlocked: bool = True, password_ok: bool = True) -> None:
        self._unlocked = unlocked
        self._password_ok = password_ok
        self.attempted: list[str] = []

    def has_password(self) -> bool:
        return True

    def is_unlocked(self) -> bool:
        return self._unlocked

    def verify(self, password: str) -> bool:
        self.attempted.append(password)
        return self._password_ok


@pytest.fixture
def answers(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Queued answers for the approval prompt (and any typed-name prompt)."""
    queue: list[str] = []
    monkeypatch.setattr(cli, "_chat_prompt", lambda *_a, **_k: queue.pop(0))
    monkeypatch.setattr(cli, "_chat_password_prompt", lambda *_a, **_k: "")
    return queue


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    """A temp allowed root + undo log + fake bin, injected into the real command.

    ``cli.make_app_context`` is the only thing replaced, and only to hand over
    ``trash`` / ``undo_log``; the registry, classifier, engine and tool the
    command builds for itself are the production ones.
    """
    ws = tmp_path / "ws"
    ws.mkdir()
    settings = Settings(policy=PolicySettings(allowed_roots=[str(ws.resolve())]))
    state: dict[str, Any] = {
        "settings": settings,
        "ws": ws,
        "log": tmp_path / "undo.jsonl",
        "trash": FakeDirTrash(tmp_path / "trash"),
        "tier": None,
        "typed": None,
        "unlock": None,
    }
    monkeypatch.setattr(cli.config, "load_settings", lambda: settings)
    monkeypatch.setattr(cli.secret_module, "SecretStore", lambda: _Store())
    monkeypatch.setattr(
        cli,
        "make_app_context",
        lambda settings=None, *, dry_run=False, **_kw: _context(state, dry_run=dry_run),
    )
    return state


def _context(state: dict[str, Any], *, dry_run: bool = False) -> Any:
    """Build the real context, with the two irreversible edges replaced."""
    settings: Settings = state["settings"]
    registry = build_default_registry(settings)
    real = make_app_context(settings, registry=registry)
    ctx = make_app_context(
        settings,
        registry=registry,
        classifier=cli.build_classifier(settings, registry),
        unlock=state["unlock"],
        trash=state["trash"],
        undo_log=state["log"],
        dry_run=dry_run,
    )
    tier = state["tier"]
    if tier is None:
        return ctx
    engine = real.engine

    class _Engine:
        """Real decisions, but with the tier forced to what a test needs."""

        def decide(self, step: Any, pctx: Any) -> Any:
            decision = engine.decide(step, pctx)
            return decision.model_copy(
                update={
                    "tier": tier,
                    "allowed": tier < tiers.TIER_BLOCKED,
                    "needs_confirm": tier >= tiers.TIER_CONFIRM,
                    "needs_unlock": tier >= tiers.TIER_CONFIRM_UNLOCK,
                    "needs_typed_confirmation": state["typed"],
                }
            )

    ctx.engine = _Engine()
    return ctx


def _delete(state: dict[str, Any], name: str, content: str = "data") -> Path:
    """Run the real delete tool so a genuine undo record exists."""
    target = state["ws"] / name
    target.write_text(content, encoding="utf-8")
    ctx = ToolContext(
        settings=state["settings"],
        trash=state["trash"],
        undo_log=state["log"],
    )
    spec = make_delete_path_spec()
    result = spec.run(spec.args_model(paths=[str(target)]), ctx)
    assert result.ok is True, result.error
    return target


def _undo(*args: str) -> Any:
    return runner.invoke(cli.app, ["undo", *args])


def _restored_marker(state: dict[str, Any]) -> dict[str, Any]:
    lines = state["log"].read_text(encoding="utf-8").strip().splitlines()
    return json.loads(lines[-1])


# ---------------------------------------------------------------------------
# the approval is mandatory, and there is no bypass
# ---------------------------------------------------------------------------


def test_undo_offers_no_yes_flag() -> None:
    """A `--yes` shortcut would let a script skip the confirmation (inv. 7)."""
    assert _undo("--yes").exit_code == 2
    assert _undo("-y").exit_code == 2
    help_text = _undo("--help").output.lower()
    assert "undo" in help_text
    assert "--yes" not in help_text
    assert "-y " not in help_text


def test_declining_the_prompt_restores_nothing(
    env: dict[str, Any], answers: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    printed = _Recorder(monkeypatch)
    target = _delete(env, "keep.txt")
    answers.append("n")
    result = _undo()
    assert result.exit_code == 1
    assert "cancelled" in printed.output
    assert target.exists() is False  # still in the bin
    assert "restored" not in _restored_marker(env)


def test_approving_restores_and_verifies(
    env: dict[str, Any], answers: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    printed = _Recorder(monkeypatch)
    target = _delete(env, "back.txt", "hello")
    answers.append("yes")
    result = _undo()
    assert result.exit_code == 0, result.output
    assert target.exists() is True
    assert target.read_text(encoding="utf-8") == "hello"
    assert "restored" in printed.output
    assert "could not be verified" not in printed.output
    assert _restored_marker(env)["restored"] is True


def test_the_prompt_shows_the_tier_and_the_action(
    env: dict[str, Any], answers: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    printed = _Recorder(monkeypatch)
    _delete(env, "shown.txt")
    answers.append("n")
    _undo()
    assert "approval needed" in printed.output
    assert tiers.tier_label(1) in printed.output
    assert "Undo the most recent" in printed.output


def test_the_action_is_confirmed_only_once_per_run(
    env: dict[str, Any], answers: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """One restore, one prompt - not a loop that could repeat the side effect."""
    printed = _Recorder(monkeypatch)
    target = _delete(env, "once.txt")
    answers.append("yes")
    assert _undo().exit_code == 0
    assert printed.output.count("approval needed") == 1
    assert target.exists() is True
    # A second run finds nothing left to undo rather than restoring again.
    answers.append("yes")
    assert _undo().exit_code == 0


# ---------------------------------------------------------------------------
# Tier 2 and the typed-name confirmation are the graph's gates
# ---------------------------------------------------------------------------


def test_tier2_refuses_a_wrong_password(
    env: dict[str, Any], answers: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    printed = _Recorder(monkeypatch)
    target = _delete(env, "tier2.txt")
    unlock = _FakeUnlock(unlocked=False, password_ok=False)
    env["unlock"] = unlock
    env["tier"] = tiers.TIER_CONFIRM_UNLOCK
    answers.append("yes")
    result = _undo()
    assert result.exit_code == 1
    assert "Wrong password" in printed.output
    assert unlock.attempted == [""]
    assert target.exists() is False


def test_tier2_with_the_right_password_restores(
    env: dict[str, Any], answers: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    unlock = _FakeUnlock(unlocked=False, password_ok=True)
    env["unlock"] = unlock
    env["tier"] = tiers.TIER_CONFIRM_UNLOCK
    target = _delete(env, "tier2-ok.txt")
    answers.append("yes")
    monkeypatch.setattr(cli, "_chat_password_prompt", lambda *_a, **_k: "hunter2")
    assert _undo().exit_code == 0
    assert unlock.attempted == ["hunter2"]
    assert target.exists() is True


def test_tier2_without_a_password_manager_fails_closed(
    env: dict[str, Any], answers: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    printed = _Recorder(monkeypatch)
    target = _delete(env, "no-unlock.txt")
    env["tier"] = tiers.TIER_CONFIRM_UNLOCK
    answers.append("yes")
    result = _undo()
    assert result.exit_code == 1
    assert "refusing" in printed.output
    assert target.exists() is False


def test_a_typed_confirmation_must_match(
    env: dict[str, Any], answers: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    printed = _Recorder(monkeypatch)
    target = _delete(env, "typed.txt")
    env["tier"] = tiers.TIER_CONFIRM
    env["typed"] = "JarvisWorkspace"
    answers.extend(["yes", "wrong-name"])
    result = _undo()
    assert result.exit_code == 1
    assert "cancelled" in printed.output
    assert target.exists() is False


def test_a_blocked_decision_never_runs_the_tool(
    env: dict[str, Any], answers: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tier 3 must stop the command before it even asks anything."""
    printed = _Recorder(monkeypatch)
    target = _delete(env, "blocked.txt")
    env["tier"] = tiers.TIER_BLOCKED
    result = _undo()
    assert result.exit_code == 1
    assert "blocked" in printed.output
    assert answers == []  # never prompted
    assert target.exists() is False


# ---------------------------------------------------------------------------
# safe failure paths
# ---------------------------------------------------------------------------


def test_nothing_to_undo_is_not_an_error(
    env: dict[str, Any], answers: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    printed = _Recorder(monkeypatch)
    answers.append("yes")
    result = _undo()
    assert result.exit_code == 0, result.output
    assert "nothing to undo" in printed.output
    assert "could not be verified" not in printed.output


def test_a_restore_the_bin_cannot_do_exits_non_zero(
    env: dict[str, Any], answers: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    printed = _Recorder(monkeypatch)
    target = _delete(env, "orphan.txt")

    class DeadBin:
        def send(self, path: Path) -> Any:  # pragma: no cover - unused here
            raise NotImplementedError

        def restore(self, record: Any) -> bool:
            return False

    env["trash"] = DeadBin()
    answers.append("yes")
    result = _undo()
    assert result.exit_code == 1
    assert "Recycle Bin" in printed.output
    assert target.exists() is False


def test_a_target_outside_the_allowed_roots_is_refused(
    env: dict[str, Any], answers: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tool's own fail-closed check still applies through the CLI."""
    printed = _Recorder(monkeypatch)
    target = _delete(env, "moved.txt")
    record = json.loads(env["log"].read_text(encoding="utf-8").splitlines()[-1])
    record["original_path"] = str((env["ws"].parent / "elsewhere.txt").resolve())
    env["log"].write_text(json.dumps(record) + "\n", encoding="utf-8")
    answers.append("yes")
    result = _undo()
    assert result.exit_code == 1
    assert "cannot auto-restore" in printed.output
    assert target.exists() is False


def test_dry_run_restores_nothing(
    env: dict[str, Any], answers: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    printed = _Recorder(monkeypatch)
    target = _delete(env, "preview.txt")
    answers.append("yes")
    result = _undo("--dry-run")
    assert result.exit_code == 0, result.output
    assert target.exists() is False
    assert "[dry-run]" in printed.output


# ---------------------------------------------------------------------------
# wiring: the real engine, the real classifier, no hidden state
# ---------------------------------------------------------------------------


def test_the_shipped_registry_and_engine_decide_tier_one(env: dict[str, Any]) -> None:
    """No test double here: the production stack must yield the documented tier."""
    settings: Settings = env["settings"]
    registry = build_default_registry(settings)
    assert registry.get(cli.UNDO_TOOL).base_tier == tiers.TIER_CONFIRM

    ctx = make_app_context(
        settings,
        registry=registry,
        classifier=cli.build_classifier(settings, registry),
    )
    from jarvis.agent.nodes.util import policy_context_for

    policy_ctx = policy_context_for(
        {"user_input": cli.UNDO_CLASSIFIER_INPUT, "results": []}, ctx.policy_ctx
    )
    assert policy_ctx is not None
    decision = ctx.engine.decide(cli._undo_step(), policy_ctx)
    assert decision.allowed is True
    assert decision.tier == 1
    assert decision.needs_confirm is True
    assert decision.needs_unlock is False


def test_the_approved_action_is_the_one_that_runs(env: dict[str, Any]) -> None:
    """The action hash is what gets approved (docs/03 invariant 7)."""
    from jarvis.policy import rules

    settings: Settings = env["settings"]
    registry = build_default_registry(settings)
    spec = registry.get(cli.UNDO_TOOL)
    expected = rules.action_hash(spec, spec.args_model.model_validate({}))

    ctx = make_app_context(
        settings, registry=registry, classifier=cli.build_classifier(settings, registry)
    )
    from jarvis.agent.nodes.util import policy_context_for

    policy_ctx = policy_context_for(
        {"user_input": cli.UNDO_CLASSIFIER_INPUT, "results": []}, ctx.policy_ctx
    )
    assert policy_ctx is not None
    assert ctx.engine.decide(cli._undo_step(), policy_ctx).action_hash == expected


def test_the_command_opens_no_database_or_checkpointer(
    env: dict[str, Any], answers: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`undo` is one tool call: no memory, no checkpoints, nothing to leak."""
    opened: list[str] = []
    monkeypatch.setattr(cli, "open_memory", lambda *a, **k: opened.append("memory"))
    monkeypatch.setattr(cli, "open_sqlite_checkpointer", lambda *a, **k: opened.append("cp"))
    monkeypatch.setattr(cli, "resume_task", lambda *a, **k: opened.append("resume"))
    _delete(env, "quiet.txt")
    answers.append("yes")
    assert _undo().exit_code == 0
    assert opened == []


def test_the_risk_layer_comes_from_config(env: dict[str, Any]) -> None:
    """The classifier the command uses is the configured one, not a hard-coded one."""
    assert config.Settings().risk.enabled is True
    settings: Settings = env["settings"]
    registry = build_default_registry(settings)
    classifier = cli.build_classifier(settings, registry)
    assert classifier is not None and classifier.active is True
