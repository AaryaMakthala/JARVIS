"""open_in_app tool tests (Tier 1, allowlisted app, resolved safe path).

Nothing here launches a process: ``subprocess.Popen`` is replaced by a
recorder everywhere.  The deny-list is pinned by digest (owner decision 3:
tested module constant, never config-driven); the readable list lives in
``jarvis/tools/open_in_app.py``.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from jarvis.agent.context import make_app_context
from jarvis.agent.nodes.validate import _LOCATION_ARGS
from jarvis.agent.nodes.validate import validate as validate_fn
from jarvis.agent.state import Plan, Step
from jarvis.config import AppSettings, PathSettings, PolicySettings, Settings
from jarvis.policy import engine as engine_module
from jarvis.policy import paths, rules, tiers
from jarvis.policy.engine import PolicyContext, PolicyEngine
from jarvis.tools.base import ToolContext, ToolResult
from jarvis.tools.open_in_app import DENIED_FILE_SUFFIXES, OpenInAppArgs, make_open_in_app_spec
from jarvis.tools.registry import _FORBIDDEN_RE, build_default_registry
from support import registry_with

#: sha256 over the sorted owner deny-list (decision 3): any edit must be deliberate.
DENIED_PIN = "e2f4ffcb9c2f068e5982f2103d264640f8ae9226c99464f4f11935cbf8b5e3a8"

#: Roadmap flows must keep working (create -> open python/text files).
ALLOWED_SUFFIXES = (".txt", ".md", ".py", ".pyw", ".json", ".csv", ".png")


def _ctx(tmp_path: Path, *, dry_run: bool = False, command: str | None = None) -> ToolContext:
    exe = tmp_path / "fakeditor.exe"
    exe.write_text("", encoding="utf-8")
    settings = Settings(
        apps=AppSettings(fakeditor=str(command or exe)),
        policy=PolicySettings(allowed_roots=[str(tmp_path)]),
    )
    return ToolContext(settings=settings, dry_run=dry_run)


def _touch(tmp_path: Path, name: str) -> Path:
    target = tmp_path / name
    target.write_text("hi", encoding="utf-8")
    return target


def _run(path: str, ctx: ToolContext, *, app: str = "fakeditor") -> Any:
    return make_open_in_app_spec().run_verified(OpenInAppArgs(app=app, path=path), ctx)


class _Recorder:
    """Stand-in for ``subprocess.Popen`` recording argv and kwargs."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.kwargs: list[dict[str, Any]] = []

    def __call__(self, argv: list[str], **kwargs: Any) -> Any:
        self.calls.append(list(argv))
        self.kwargs.append(dict(kwargs))
        return type("Proc", (), {"pid": 4242})()


def _recorder(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    recorder = _Recorder()
    monkeypatch.setattr(subprocess, "Popen", recorder)
    return recorder


class TestSpec:
    def test_contract(self) -> None:
        spec = make_open_in_app_spec()
        assert spec.name == "open_in_app"
        assert spec.base_tier == tiers.TIER_CONFIRM
        assert spec.path_args == ("path",)
        assert spec.windows_only is False
        assert rules.matches_blocked(spec, OpenInAppArgs(app="notepad", path="x")) is False
        assert _FORBIDDEN_RE.search(f"{spec.name} {spec.description}") is None

    def test_args_reject_extras_and_bad_app_names(self) -> None:
        with pytest.raises(ValidationError):
            OpenInAppArgs(app="notepad", path="x", extra="y")  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            OpenInAppArgs(app="", path="x")
        with pytest.raises(ValidationError):
            OpenInAppArgs(app="a" * 65, path="x")

    def test_registered_in_default_registry(self) -> None:
        registry = build_default_registry(Settings())
        assert "open_in_app" in registry.names()
        spec = registry.get("open_in_app")
        assert spec.base_tier == tiers.TIER_CONFIRM
        assert spec.path_args == ("path",)


class TestEngineDecide:
    @pytest.fixture
    def ws(self, tmp_path: Path) -> Path:
        root = tmp_path / "ws"
        root.mkdir()
        return root

    def _decide(self, ws: Path, path: str) -> Any:
        settings = Settings(policy=PolicySettings(allowed_roots=[str(ws)]))
        pctx = PolicyContext(registry=build_default_registry(settings), settings=settings)
        step = Step(
            id="s1", tool="open_in_app", args={"app": "notepad", "path": path}, rationale="r"
        )
        return PolicyEngine().decide(step, pctx)

    def test_inside_root_is_tier1_and_records_resolved_path(self, ws: Path) -> None:
        target = ws / "notes.txt"
        decision = self._decide(ws, str(target))

        assert decision.allowed is True
        assert decision.tier == tiers.TIER_CONFIRM
        assert decision.needs_confirm is True
        assert decision.reasons == []
        assert decision.resolved_paths == [str(target.resolve(strict=False))]

    def test_outside_roots_is_blocked(self, ws: Path, tmp_path: Path) -> None:
        decision = self._decide(ws, str(tmp_path / "outside.txt"))

        assert decision.allowed is False
        assert decision.tier == tiers.TIER_BLOCKED
        assert any("outside the allowed folders" in r for r in decision.reasons)

    def test_protected_path_is_blocked(self, ws: Path) -> None:
        protected = Path(os.path.expanduser("~/.ssh")) / "id_rsa"
        decision = self._decide(ws, str(protected))

        assert decision.allowed is False
        assert any("path is protected" in r for r in decision.reasons)

    def test_not_refused_by_the_windows_gate_off_windows(
        self, ws: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "is_windows", lambda: False)
        decision = self._decide(ws, str(ws / "notes.txt"))

        assert decision.allowed is True
        assert decision.tier == tiers.TIER_CONFIRM
        assert decision.reasons == []


class TestRunLaunch:
    @pytest.mark.parametrize("name", ["notes.txt", "my notes.txt", "-rf.txt"])
    def test_argv_is_exactly_exe_plus_one_absolute_path(
        self, name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = _recorder(monkeypatch)
        target = _touch(tmp_path, name)
        result = _run(str(target), _ctx(tmp_path))

        expected = [str(tmp_path / "fakeditor.exe"), str(paths.resolve_safe(str(target)))]
        assert result.ok is True
        assert recorder.calls == [expected]
        assert len(recorder.calls[0]) == 2
        assert "shell" not in recorder.kwargs[0]
        assert recorder.kwargs[0]["close_fds"] is True
        assert not recorder.calls[0][1].startswith("-")
        assert result.data["file"] == expected[1]
        assert result.data["pid"] == 4242

    def test_unknown_app_lists_the_allowlist(self, tmp_path: Path) -> None:
        result = _run("x", _ctx(tmp_path), app="spotify")

        assert result.ok is False
        assert "unknown app" in (result.error or "")
        assert "notepad" in result.data["known"]
        assert "fakeditor" in result.data["known"]

    @pytest.mark.parametrize("shim", ["codish.cmd", "codish.bat"])
    def test_batch_shim_exe_is_refused(
        self, shim: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = _recorder(monkeypatch)
        script = tmp_path / shim
        script.write_text("", encoding="utf-8")
        target = _touch(tmp_path, "notes.txt")
        result = _run(str(target), _ctx(tmp_path, command=str(script)))

        assert result.ok is False
        assert "absolute path in [apps]" in (result.error or "")
        assert recorder.calls == []

    def test_dry_run_launches_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = _recorder(monkeypatch)
        result = _run(str(_touch(tmp_path, "notes.txt")), _ctx(tmp_path, dry_run=True))

        assert recorder.calls == []
        assert result.ok is True
        assert "[dry-run]" in result.output

    def test_outside_roots_is_refused_at_run_time(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = _recorder(monkeypatch)
        outside = tmp_path.parent / f"outside_{tmp_path.name}.txt"
        outside.write_text("hi", encoding="utf-8")
        result = _run(str(outside), _ctx(tmp_path))

        assert result.ok is False
        assert "outside the allowed folders" in (result.error or "")
        assert recorder.calls == []

    def test_interpreter_entry_is_refused_and_never_launches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = _recorder(monkeypatch)
        python = tmp_path / "python.exe"
        python.write_text("", encoding="utf-8")
        target = _touch(tmp_path, "notes.txt")
        result = _run(str(target), _ctx(tmp_path, command=str(python)))

        assert result.ok is False
        assert "is an interpreter" in (result.error or "")
        assert recorder.calls == []


class TestTargetRules:
    def test_owner_denial_list_is_pinned(self) -> None:
        digest = hashlib.sha256("\n".join(sorted(DENIED_FILE_SUFFIXES)).encode()).hexdigest()
        assert digest == DENIED_PIN

    def test_roadmap_file_types_stay_allowed(self) -> None:
        for suffix in ALLOWED_SUFFIXES:
            assert suffix not in DENIED_FILE_SUFFIXES, suffix

    @pytest.mark.parametrize(
        "name",
        [
            "evil.exe",
            "EVIL.EXE",
            "evil.exe.",
            "evil.exe ",
            "script.bat",
            "script.cmd",
            "run.ps1",
            "note.js",
            "installer.msi",
            "page.html",
            "site.url",
            "lib.dll",
            "link.lnk",
            "keys.reg",
        ],
    )
    def test_denied_suffix_refuses_before_launch(
        self, name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = _recorder(monkeypatch)
        result = _run(str(tmp_path / name), _ctx(tmp_path))

        assert result.ok is False
        assert "refusing to open this file type" in (result.error or "")
        assert recorder.calls == []

    def test_alternate_data_stream_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = _recorder(monkeypatch)
        result = _run(str(tmp_path / "x.txt:evil.exe"), _ctx(tmp_path))

        assert result.ok is False
        assert recorder.calls == []

    def test_missing_file_is_refused(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _recorder(monkeypatch)
        result = _run(str(tmp_path / "ghost.txt"), _ctx(tmp_path))

        assert result.ok is False
        assert "does not exist" in (result.error or "")
        assert recorder.calls == []

    def test_directory_is_refused(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _recorder(monkeypatch)
        (tmp_path / "folder").mkdir()
        result = _run(str(tmp_path / "folder"), _ctx(tmp_path))

        assert result.ok is False
        assert "folder, not a file" in (result.error or "")
        assert recorder.calls == []

    def test_relative_path_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = _recorder(monkeypatch)
        result = _run("notes.txt", _ctx(tmp_path))

        assert result.ok is False
        assert "must be absolute" in (result.error or "")
        assert recorder.calls == []

    def test_symlink_target_is_what_gets_checked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The deny-list runs on the RESOLVED path, never on the link name."""
        recorder = _recorder(monkeypatch)
        _touch(tmp_path, "evil.exe")
        link = tmp_path / "notes.txt"
        try:
            os.symlink(tmp_path / "evil.exe", link)
        except OSError as exc:
            pytest.skip(f"symlink creation not permitted on this machine: {exc}")

        result = _run(str(link), _ctx(tmp_path))

        assert result.ok is False
        assert "file type" in (result.error or "")
        assert recorder.calls == []


class TestVerifyDescribe:
    def test_verify_is_none_with_note_naming_app_and_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _recorder(monkeypatch)
        result = _run(str(_touch(tmp_path, "notes.txt")), _ctx(tmp_path))

        assert result.ok is True
        assert result.verified is None
        assert "fakeditor" in result.verify_note
        assert "notes.txt" in result.verify_note

    def test_verify_of_a_failed_run_is_false(self, tmp_path: Path) -> None:
        spec = make_open_in_app_spec()
        failed = ToolResult(ok=False, error="nope")

        verified = spec.verify(OpenInAppArgs(app="fakeditor", path="x"), failed, _ctx(tmp_path))

        assert verified is not None
        assert verified.verified is False

    def test_describe_names_resolved_path_and_app(self, tmp_path: Path) -> None:
        target = _touch(tmp_path, "notes.txt")
        args = OpenInAppArgs(app="notepad", path=str(target))
        expected = f"Open {paths.resolve_safe(str(target))} in notepad"

        assert make_open_in_app_spec().describe(args) == expected

    def test_describe_falls_back_to_raw_path_on_path_error(self) -> None:
        args = OpenInAppArgs(app="notepad", path="bad\x00.txt")

        assert make_open_in_app_spec().describe(args) == "Open bad\x00.txt in notepad"


class TestLocationRewrite:
    def test_open_in_app_is_a_location_arg(self) -> None:
        assert _LOCATION_ARGS["open_in_app"] == ("path",)

    def test_bare_name_resolved_to_absolute_before_hash(self, tmp_path: Path) -> None:
        settings = Settings(
            policy=PolicySettings(allowed_roots=[str(tmp_path)]),
            paths=PathSettings(default_save_dir=str(tmp_path)),
        )
        ctx = make_app_context(settings, registry=registry_with(make_open_in_app_spec()))
        step_args = {"app": "notepad", "path": "notes.txt"}
        step = Step(id="s1", tool="open_in_app", args=step_args, rationale="r")
        out = validate_fn({"plan": Plan(goal="g", steps=[step])}, ctx)

        resolved = str(tmp_path / "notes.txt")
        assert out["plan"].steps[0].args["path"] == resolved
        decision = ctx.engine.decide(out["plan"].steps[0], ctx.policy_ctx)
        spec = ctx.registry.get("open_in_app")
        model = spec.args_model.model_validate({"app": "notepad", "path": resolved})
        expected = rules.action_hash(spec, model)
        raw = rules.action_hash_raw("open_in_app", {"app": "notepad", "path": "notes.txt"})
        assert decision.action_hash == expected
        assert decision.action_hash != raw
