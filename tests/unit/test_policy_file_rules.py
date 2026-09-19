"""Policy-engine tests for Phase 2 file rules (delete/typed-confirm/paths).

The engine is deterministic and pure: we drive ``decide()`` directly and check
the resulting :class:`Decision` - tiers, typed folder-name confirmations,
resolved paths and hard blocks (docs/03_SECURITY_AND_POLICY.md sections 3-4).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from jarvis.agent.context import make_app_context
from jarvis.agent.state import Step
from jarvis.config import PolicySettings, Settings
from jarvis.policy import tiers
from jarvis.tools.files import (
    make_append_file_spec,
    make_create_file_spec,
    make_delete_path_spec,
    make_list_dir_spec,
    make_read_file_spec,
    make_undo_last_delete_spec,
)
from jarvis.tools.system import make_lock_jarvis_spec
from support import make_spec, registry_with


def _settings(ws: Path, *, typed: bool = True) -> Settings:
    return Settings(
        policy=PolicySettings(
            allowed_roots=[str(ws.resolve())],
            typed_confirmation_for_folders=typed,
        )
    )


def _ctx(ws: Path, *, typed: bool = True) -> Any:
    settings = _settings(ws, typed=typed)
    registry = registry_with(
        make_delete_path_spec(),
        make_undo_last_delete_spec(),
        make_create_file_spec(),
        make_append_file_spec(),
        make_list_dir_spec(),
        make_read_file_spec(),
        make_lock_jarvis_spec(),
    )
    return make_app_context(settings, registry=registry)


def _ctx_for_spec(spec: Any, ws: Path) -> Any:
    return make_app_context(_settings(ws), registry=registry_with(spec))


def _step(tool: str, args: dict[str, Any]) -> Step:
    return Step(id="s1", tool=tool, args=args, rationale="r")


def _decide(ctx: Any, tool: str, args: dict[str, Any]) -> Any:
    return ctx.engine.decide(_step(tool, args), ctx.policy_ctx)


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    return root


# ---------------------------------------------------------------------------
# delete_path tiering + typed confirmation
# ---------------------------------------------------------------------------


def test_delete_file_is_tier2_unlock_without_typed_confirmation(ws: Path) -> None:
    target = ws / "a.txt"
    decision = _decide(_ctx(ws), "delete_path", {"paths": [str(target)]})
    assert decision.allowed is True
    assert decision.tier == tiers.TIER_CONFIRM_UNLOCK
    assert decision.needs_confirm is True
    assert decision.needs_unlock is True
    assert decision.needs_typed_confirmation is None
    assert decision.resolved_paths == [str(target.resolve(strict=False))]


def test_delete_folder_requires_typed_folder_name(ws: Path) -> None:
    folder = ws / "Reports"
    folder.mkdir()
    decision = _decide(_ctx(ws), "delete_path", {"paths": [str(folder)]})
    assert decision.allowed is True
    assert decision.needs_typed_confirmation == "Reports"


def test_delete_of_two_folders_joins_sorted_names(ws: Path) -> None:
    (ws / "Beta").mkdir()
    (ws / "Alpha").mkdir()
    decision = _decide(_ctx(ws), "delete_path", {"paths": [str(ws / "Beta"), str(ws / "Alpha")]})
    assert decision.needs_typed_confirmation == "Alpha, Beta"


def test_typed_confirmation_disabled_via_settings_falls_back_to_plain(ws: Path) -> None:
    folder = ws / "Docs"
    folder.mkdir()
    decision = _decide(_ctx(ws, typed=False), "delete_path", {"paths": [str(folder)]})
    assert decision.needs_typed_confirmation is None
    assert decision.tier == tiers.TIER_CONFIRM_UNLOCK


def test_delete_many_files_records_every_resolved_path(ws: Path) -> None:
    targets = [ws / f"f{i}.txt" for i in range(3)]
    decision = _decide(_ctx(ws), "delete_path", {"paths": [str(p) for p in targets]})
    assert decision.allowed is True
    assert decision.resolved_paths == [str(p.resolve(strict=False)) for p in targets]


# ---------------------------------------------------------------------------
# hard path blocks (invariant 4 / T4)
# ---------------------------------------------------------------------------


def test_delete_of_an_allowed_root_itself_is_refused(ws: Path) -> None:
    decision = _decide(_ctx(ws), "delete_path", {"paths": [str(ws.resolve())]})
    assert decision.allowed is False
    assert decision.tier == tiers.TIER_BLOCKED
    assert "allowed folder" in "; ".join(decision.reasons)


def test_delete_of_an_ancestor_of_an_allowed_root_is_refused(ws: Path) -> None:
    decision = _decide(_ctx(ws), "delete_path", {"paths": [str(ws.parent)]})
    assert decision.allowed is False
    assert "allowed folder" in "; ".join(decision.reasons)


def test_delete_outside_allowed_roots_is_refused(ws: Path, tmp_path: Path) -> None:
    outside = tmp_path / "elsewhere"
    decision = _decide(_ctx(ws), "delete_path", {"paths": [str(outside)]})
    assert decision.allowed is False
    assert "outside the allowed folders" in "; ".join(decision.reasons)


def test_delete_of_protected_windows_path_is_refused(ws: Path) -> None:
    if os.name != "nt":
        pytest.skip("protected OS paths only exist on Windows")
    protected = Path(os.environ["SystemRoot"]) / "System32"
    decision = _decide(_ctx(ws), "delete_path", {"paths": [str(protected)]})
    assert decision.allowed is False
    assert "protected" in "; ".join(decision.reasons)


def test_bad_path_string_is_refused_before_any_tier(ws: Path) -> None:
    decision = _decide(_ctx(ws), "delete_path", {"paths": [r"\\server\share\x"]})
    assert decision.allowed is False
    assert "bad path" in "; ".join(decision.reasons)


def test_empty_delete_list_is_invalid_arguments(ws: Path) -> None:
    decision = _decide(_ctx(ws), "delete_path", {"paths": []})
    assert decision.allowed is False
    assert "invalid arguments" in "; ".join(decision.reasons)


# ---------------------------------------------------------------------------
# create_file / append_file path-tier rules
# ---------------------------------------------------------------------------


def test_create_file_overwrite_keeps_tier1_and_warns(ws: Path) -> None:
    target = ws / "existing.txt"
    decision = _decide(
        _ctx(ws), "create_file", {"path": str(target), "content": "x", "overwrite": True}
    )
    assert decision.allowed is True
    assert decision.tier == tiers.TIER_CONFIRM
    assert decision.needs_unlock is False
    assert "OVERWRITES" in decision.summary


def test_create_file_inside_root_is_tier1(ws: Path) -> None:
    target = ws / "new.txt"
    decision = _decide(_ctx(ws), "create_file", {"path": str(target), "content": "x"})
    assert decision.tier == tiers.TIER_CONFIRM
    assert decision.resolved_paths == [str(target.resolve(strict=False))]


def test_read_only_tools_are_tier0(ws: Path) -> None:
    target = ws / "notes.txt"
    for tool, args in (
        ("list_dir", {"path": str(ws)}),
        ("read_file", {"path": str(target)}),
    ):
        decision = _decide(_ctx(ws), tool, args)
        assert decision.tier == tiers.TIER_SAFE
        assert decision.needs_confirm is False


def test_lock_jarvis_is_tier0_and_harmless(ws: Path) -> None:
    decision = _decide(_ctx(ws), "lock_jarvis", {})
    assert decision.allowed is True
    assert decision.tier == tiers.TIER_SAFE
    assert decision.needs_confirm is False


def test_undo_last_delete_is_tier1(ws: Path) -> None:
    decision = _decide(_ctx(ws), "undo_last_delete", {})
    assert decision.allowed is True
    assert decision.tier == tiers.TIER_CONFIRM
    assert decision.needs_typed_confirmation is None


# ---------------------------------------------------------------------------
# name-level hard block keeps the sanctioned exemption narrow
# ---------------------------------------------------------------------------


def test_unsanctioned_delete_names_are_still_hard_blocked(ws: Path) -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    for name in ("delete_notes", "rm_backup", "delete_all"):
        spec = make_spec(name, base_tier=0, record=record)
        decision = _decide(_ctx_for_spec(spec, ws), name, {"text": "x"})
        assert decision.allowed is False
        assert "hard-blocked" in "; ".join(decision.reasons)


def test_sanctioned_delete_tools_are_never_name_blocked(ws: Path) -> None:
    context = _ctx(ws)
    decision = _decide(context, "delete_path", {"paths": [str(ws / "x.txt")]})
    assert decision.allowed is True
    undo = _decide(context, "undo_last_delete", {})
    assert undo.allowed is True
