"""lock_computer tool tests (docs/04 §2.4: Tier 1, LockWorkStation, best effort).

The real tool calls ``user32.LockWorkStation`` via ``ctypes.windll``.  Those
tests monkeypatch ``ctypes.windll`` so they run on any OS; the only test that
would need a real desktop session is marked ``windows_only`` and skipped here.
"""

from __future__ import annotations

import ctypes
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from jarvis.agent.state import Step  # import first: engine imports it back
from jarvis.config import Settings
from jarvis.ml import serialize as ml_serialize
from jarvis.ml import signals as ml_signals
from jarvis.policy import tiers
from jarvis.policy.engine import PolicyContext, PolicyEngine
from jarvis.tools.base import ToolContext
from jarvis.tools.registry import build_default_registry
from jarvis.tools.system import LockComputerArgs, make_lock_computer_spec


def _ctx(dry_run: bool = False) -> ToolContext:
    return ToolContext(settings=Settings(), dry_run=dry_run)


def _install_windll(monkeypatch: pytest.MonkeyPatch, result_code: int) -> list[int]:
    """Replace ``ctypes.windll.user32.LockWorkStation`` with a recording fake."""
    calls: list[int] = []
    user32 = SimpleNamespace(LockWorkStation=lambda: calls.append(result_code) or result_code)
    windll = SimpleNamespace(user32=user32)
    monkeypatch.setattr(ctypes, "windll", windll, raising=False)
    return calls


def _decide(args: dict[str, object] | None = None, user_input: str = ""):
    """Run the real engine over the real registry for ``lock_computer``."""
    engine = PolicyEngine()
    pctx = PolicyContext(
        registry=build_default_registry(Settings()),
        settings=Settings(),
        user_input=user_input,
    )
    return engine.decide(
        Step(id="s1", tool="lock_computer", args=dict(args or {}), rationale="r"), pctx
    )


# ── Spec contract (docs/04 §2.4) ────────────────────────────────────────


class TestLockComputerSpec:
    def test_spec_contract(self) -> None:
        spec = make_lock_computer_spec()
        assert spec.name == "lock_computer"
        assert spec.base_tier == 1
        assert spec.windows_only is True
        assert spec.timeout_s == 10
        assert spec.verify is not None
        assert spec.describe(LockComputerArgs()) == "Lock the Windows computer"

    def test_args_reject_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            LockComputerArgs(whatever=1)  # type: ignore[call-arg]

    def test_dry_run_has_no_side_effect(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = _install_windll(monkeypatch, 1)
        spec = make_lock_computer_spec()
        result = spec.execute(LockComputerArgs(), _ctx(dry_run=True))
        assert result.ok
        assert "[dry-run]" in result.output
        assert calls == []  # Win32 was never touched


# ── Run / verify ────────────────────────────────────────────────────────


class TestLockComputerRun:
    def test_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = _install_windll(monkeypatch, 1)
        spec = make_lock_computer_spec()
        result = spec.run(LockComputerArgs(), _ctx())
        assert result.ok, result.error
        assert calls == [1]
        assert result.data.get("locked") is True

    def test_zero_return_reports_failure_honestly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_windll(monkeypatch, 0)
        spec = make_lock_computer_spec()
        result = spec.run(LockComputerArgs(), _ctx())
        assert not result.ok
        assert "could not lock" in (result.error or "")

    def test_access_denied_is_explained(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-interactive sessions get access denied; the message says so."""
        _install_windll(monkeypatch, 0)
        monkeypatch.setattr(ctypes, "get_last_error", lambda: 5, raising=False)
        spec = make_lock_computer_spec()
        result = spec.run(LockComputerArgs(), _ctx())
        assert not result.ok
        assert "access denied" in (result.error or "")

    def test_windll_missing_never_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No ``ctypes.windll`` (non-Windows): honest ToolResult, no exception."""

        class _NoWindll:
            def __getattr__(self, name: str) -> None:
                raise AttributeError(name)

        monkeypatch.setattr(ctypes, "windll", _NoWindll(), raising=False)
        spec = make_lock_computer_spec()
        result = spec.run(LockComputerArgs(), _ctx())
        assert not result.ok
        assert "LockWorkStation failed" in (result.error or "")

    def test_verify_is_best_effort_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """docs/04: verification is 'best effort' -> verified=None, not a claim."""
        _install_windll(monkeypatch, 1)
        spec = make_lock_computer_spec()
        raw = spec.run(LockComputerArgs(), _ctx())
        verified = spec.apply_verify(LockComputerArgs(), raw, _ctx())
        assert verified.verified is None


# ── Registry + policy integration ───────────────────────────────────────


class TestLockComputerIntegration:
    def test_registered(self) -> None:
        registry = build_default_registry()
        spec = registry.get("lock_computer")
        assert spec is not None
        assert spec.base_tier == 1

    def test_policy_tier1_needs_confirm(self) -> None:
        """The engine maps the declared Tier 1 to needs_confirm (docs/04)."""
        decision = _decide()
        assert decision.allowed is True
        assert decision.tier == tiers.TIER_CONFIRM
        assert decision.needs_confirm is True
        assert decision.needs_unlock is False

    def test_classifier_cannot_lower_the_tier(self) -> None:
        """Even with a 'safe'-leaning wording, the tier never drops below 1."""
        decision = _decide(user_input="please stay awake")
        assert decision.tier >= tiers.TIER_CONFIRM

        # The classifier alone (safe prior, mild wording) may want Tier 0,
        # but the declared base tier is a floor the engine always keeps.
        outcome = ml_signals.assess(
            ml_serialize.serialize_action("lock_computer", {}, "please stay awake"),
            prior_label=ml_signals.SAFE,
        )
        assert outcome.min_tier >= 0
        assert (
            decision.tier >= ml_signals.LABEL_MIN_TIER[outcome.label]
            or decision.tier >= tiers.TIER_CONFIRM
        )
