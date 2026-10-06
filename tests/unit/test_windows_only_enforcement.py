"""``ToolSpec.windows_only`` is enforced by the policy engine (Stage 4.2).

The engine is the single decision path every step takes, so the platform check
lives there: ``decide`` refuses a ``windows_only`` tool on a host that is not
Windows, with a reason that names the real cause.  On Windows the check must be
a no-op, and ``register()``/``build_default_registry()`` must stay usable
everywhere (the refusal is a *decision*, not a registration error).

The platform function is the only fake here: every ``ToolSpec``/``ToolRegistry``
is the real class, and the platform is patched where ``engine.py`` looks it up.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from jarvis.agent.state import Step
from jarvis.config import Settings
from jarvis.policy import engine as engine_module
from jarvis.policy import tiers
from jarvis.policy.engine import _WINDOWS_ONLY_REASON, PolicyContext, PolicyEngine
from jarvis.tools.base import ToolContext, ToolResult, ToolSpec
from jarvis.tools.registry import ToolRegistry, build_default_registry

#: The generic Tier 3 text a hard-blocked tool records.  The platform refusal
#: must never be mistaken for it.
_GENERIC_BLOCK_REASON = "this tool/action is hard-blocked by policy"


class _Args(BaseModel):
    """Minimal args model so a step validates without a real tool's schema."""

    model_config = ConfigDict(extra="forbid")

    text: str = "x"


def _spec(name: str, *, tier: int, windows_only: bool, calls: list[str]) -> ToolSpec:
    """A real ``ToolSpec`` whose run records the call, so we can prove it never runs."""

    def _run(args: _Args, ctx: ToolContext) -> ToolResult:
        calls.append(name)
        return ToolResult(ok=True, output=f"{name} ran")

    return ToolSpec(
        name=name,
        description="test-only tool",
        args_model=_Args,
        base_tier=tier,
        windows_only=windows_only,
        run=_run,
    )


def _decide(spec: ToolSpec) -> Any:
    """Run the real engine over a registry holding exactly ``spec``."""
    registry = ToolRegistry()
    registry.register(spec)
    pctx = PolicyContext(registry=registry, settings=Settings())
    step = Step(id="s1", tool=spec.name, args={}, rationale="r")
    return PolicyEngine().decide(step, pctx)


def _patch_platform(monkeypatch: pytest.MonkeyPatch, *, windows: bool) -> None:
    """Patch the platform function where ``engine.py`` actually calls it."""
    monkeypatch.setattr(engine_module, "is_windows", lambda: windows)


class TestRefusedOffWindows:
    def test_windows_only_tool_is_refused_and_never_runs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_platform(monkeypatch, windows=False)
        calls: list[str] = []
        decision = _decide(_spec("fake_win_tool", tier=1, windows_only=True, calls=calls))

        assert decision.allowed is False
        assert calls == [], "decide() must never execute a tool"
        assert decision.reasons == [_WINDOWS_ONLY_REASON]
        # No confirmation is offered for something that cannot run here.
        assert decision.needs_confirm is False
        assert decision.needs_unlock is False
        assert decision.needs_typed_confirmation is None
        # allowed=False stays tier 3 in this codebase (see _blocked), never a
        # fake tier below the tool's own base_tier.
        assert decision.tier == tiers.TIER_BLOCKED

    def test_reason_names_the_platform_not_a_generic_block(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_platform(monkeypatch, windows=False)
        calls: list[str] = []
        decision = _decide(_spec("fake_win_tool", tier=2, windows_only=True, calls=calls))

        assert decision.reasons != [_GENERIC_BLOCK_REASON]
        assert "hard-blocked" not in decision.reasons[0]
        assert "windows" in decision.reasons[0].lower()
        assert decision.summary.endswith("(Windows only)")

    def test_a_tier_two_tool_does_not_fake_a_lower_tier(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_platform(monkeypatch, windows=False)
        calls: list[str] = []
        decision = _decide(_spec("fake_tier2_win_tool", tier=2, windows_only=True, calls=calls))
        assert decision.tier >= 2


class TestNoOpOnWindows:
    def test_windows_only_tool_is_untouched_on_windows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_platform(monkeypatch, windows=True)
        calls: list[str] = []
        decision = _decide(_spec("fake_win_tool", tier=1, windows_only=True, calls=calls))

        assert decision.allowed is True
        assert decision.tier == 1
        assert decision.needs_confirm is True
        assert decision.reasons == []

    def test_decision_matches_the_same_tool_without_the_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On Windows the flag changes nothing about the decision."""
        _patch_platform(monkeypatch, windows=True)
        calls: list[str] = []
        flagged = _decide(_spec("fake_win_tool", tier=1, windows_only=True, calls=calls))
        plain = _decide(_spec("fake_plain_tool", tier=1, windows_only=False, calls=calls))

        assert (flagged.tier, flagged.allowed) == (plain.tier, plain.allowed)
        assert flagged.needs_confirm == plain.needs_confirm
        assert flagged.needs_unlock == plain.needs_unlock
        assert flagged.reasons == plain.reasons

    def test_non_windows_only_tool_is_unaffected_off_windows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_platform(monkeypatch, windows=False)
        calls: list[str] = []
        decision = _decide(_spec("fake_plain_tool", tier=0, windows_only=False, calls=calls))

        assert decision.allowed is True
        assert decision.tier == 0
        assert decision.reasons == []


def _decide_real(tool: str) -> Any:
    """Run the real engine over the real registry for a shipped tool, no args."""
    registry = build_default_registry(Settings())
    pctx = PolicyContext(registry=registry, settings=Settings())
    return PolicyEngine().decide(Step(id="s1", tool=tool, args={}, rationale="r"), pctx)


class TestShippedTool:
    def test_lock_computer_is_refused_off_windows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A real windows_only tool, real registry, no fakes but the platform."""
        _patch_platform(monkeypatch, windows=False)
        decision = _decide_real("lock_computer")

        assert decision.allowed is False
        assert decision.reasons == [_WINDOWS_ONLY_REASON]
        assert decision.needs_confirm is False
        assert decision.tier == tiers.TIER_BLOCKED

    def test_lock_computer_is_unchanged_on_windows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_platform(monkeypatch, windows=True)
        decision = _decide_real("lock_computer")

        assert decision.allowed is True
        assert decision.tier == 1
        assert decision.needs_confirm is True
        assert _WINDOWS_ONLY_REASON not in decision.reasons


class TestRegistryIsUnaffected:
    def test_registry_still_builds_off_windows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``register()`` must never refuse a windows_only tool on non-Windows."""
        _patch_platform(monkeypatch, windows=False)
        registry = build_default_registry(Settings())
        names = registry.names()
        assert "type_text" in names
        assert "lock_computer" in names
        assert registry.get("type_text").windows_only is True

    def test_catalogue_with_policy_still_reports_the_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_platform(monkeypatch, windows=False)
        entries = {e["name"]: e for e in build_default_registry(Settings()).catalogue_with_policy()}
        assert entries["type_text"]["windows_only"] is True
        assert entries["open_app"]["windows_only"] is False

    @pytest.mark.parametrize("windows", [False, True])
    def test_no_registered_tool_decides_below_its_base_tier(
        self, monkeypatch: pytest.MonkeyPatch, windows: bool
    ) -> None:
        """The registry-wide no-lower property, over every shipped tool."""
        _patch_platform(monkeypatch, windows=windows)
        registry = build_default_registry(Settings())
        pctx = PolicyContext(registry=registry, settings=Settings())
        engine = PolicyEngine()

        for spec in registry.iter_all():
            step = Step(id="s1", tool=spec.name, args={}, rationale="r")
            decision = engine.decide(step, pctx)
            assert decision.tier >= spec.base_tier, spec.name
            if windows and spec.windows_only:
                assert _WINDOWS_ONLY_REASON not in decision.reasons, spec.name
