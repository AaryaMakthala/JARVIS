"""validate-node tests: default/alias location resolution before the hash (D1b)."""

from __future__ import annotations

from pathlib import Path

from jarvis.agent.context import make_app_context
from jarvis.agent.nodes.validate import validate
from jarvis.agent.state import Plan, Step
from jarvis.config import PathSettings, PolicySettings, Settings
from jarvis.policy import rules
from jarvis.tools.files import make_create_file_spec, make_delete_path_spec
from support import registry_with


def _ctx(tmp_path: Path, default: Path | None = None, roots: list[Path] | None = None):
    settings = Settings(
        policy=PolicySettings(allowed_roots=[str(r) for r in (roots or [tmp_path])]),
        paths=PathSettings(default_save_dir=str(default or tmp_path)),
    )
    registry = registry_with(make_create_file_spec(), make_delete_path_spec())
    return make_app_context(settings, registry=registry)


def _step(tool: str, args: dict) -> Step:
    return Step(id="s1", tool=tool, args=args, rationale="r")


def _state(step: Step, **extra) -> dict:
    return {"plan": Plan(goal="g", steps=[step]), **extra}


def test_bare_filename_resolved_before_hash(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    state = _state(_step("create_file", {"path": "notes.txt", "content": "hi"}))

    out = validate(state, ctx)

    assert out["plan"].steps[0].args["path"] == str(tmp_path / "notes.txt")
    assert out["gated_resolved_paths"]["s1"] == [str(tmp_path / "notes.txt")]


def test_hash_is_over_resolved_path(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    state = _state(_step("create_file", {"path": "notes.txt", "content": "hi"}))

    out = validate(state, ctx)
    step = out["plan"].steps[0]
    decision = ctx.engine.decide(step, ctx.policy_ctx)

    spec = ctx.registry.get("create_file")
    expected = rules.action_hash(
        spec, spec.args_model.model_validate({"path": str(tmp_path / "notes.txt"), "content": "hi"})
    )
    raw = rules.action_hash_raw("create_file", {"path": "notes.txt", "content": "hi"})
    assert decision.action_hash == expected
    assert decision.action_hash != raw


def test_unknown_folder_sets_clarification(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    state = _state(_step("create_file", {"path": "Reports/notes.txt", "content": "hi"}))

    out = validate(state, ctx)
    plan = out["plan"]

    assert plan.needs_clarification is True
    assert plan.clarification_question == "Which folder should I use?"
    assert plan.steps == []


def test_absolute_path_unchanged(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    target = tmp_path / "notes.txt"
    state = _state(_step("create_file", {"path": str(target), "content": "hi"}))

    out = validate(state, ctx)

    assert out["plan"].steps[0].args["path"] == str(target)


def test_replan_also_resolved(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    state = _state(
        _step("delete_path", {"paths": ["notes.txt"]}),
        pending_replan=True,
        results=[],
        approved_hashes=[],
    )

    out = validate(state, ctx)

    assert out["plan"].steps[0].args["paths"] == [str(tmp_path / "notes.txt")]


def test_resolved_path_outside_allowed_roots_refused(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    ctx = _ctx(tmp_path, default=ws, roots=[ws])
    outside = tmp_path / "outside"
    step = _step("create_file", {"path": str(outside / "notes.txt"), "content": "x"})

    out = validate(_state(step), ctx)
    decision = ctx.engine.decide(out["plan"].steps[0], ctx.policy_ctx)

    assert decision.allowed is False
    assert any("outside the allowed folders" in reason for reason in decision.reasons)


def test_never_auto_adds_root(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    ctx = _ctx(tmp_path, default=outside, roots=[ws])
    before = list(ctx.settings.policy.allowed_roots)

    out = validate(_state(_step("create_file", {"path": "notes.txt", "content": "x"})), ctx)

    assert out["plan"].steps[0].args["path"] == str(outside / "notes.txt")
    assert list(ctx.settings.policy.allowed_roots) == before
    decision = ctx.engine.decide(out["plan"].steps[0], ctx.policy_ctx)
    assert decision.allowed is False
