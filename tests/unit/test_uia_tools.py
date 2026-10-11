"""Native UIA tool tests (Stage 7, Slice 7).

No Windows desktop, no pywinauto and no live UIA session: every test injects a
fake provider/window/control through the module-level ``_build_provider`` hook
(the same pattern ``whatsapp.py`` uses).  The tests assert the tools' own logic:
platform refusal, bounded enumeration, secure-control exclusion, taint,
unique/stale/ambiguous/changed resolution, credential refusal, no coordinate
fallback and fail-closed error handling.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from jarvis.config import Settings
from jarvis.policy.targets import ResolvedTarget
from jarvis.tools.base import ToolContext, ToolResult, resolve_target
from jarvis.uia import tools as uia_tools
from jarvis.uia.tools import (
    ControlIdentity,
    ControlLookup,
    PywinautoUiaProvider,
    UiaClickArgs,
    UiaListControlsArgs,
    _PywinautoControl,
    control_ref,
    make_ui_click_spec,
    make_ui_list_controls_spec,
)

# ── fakes ──────────────────────────────────────────────────────────────────


class FakeControl:
    def __init__(self, name: str, role: str = "Button", *, is_password: bool = False) -> None:
        self._identity = ControlIdentity(
            name=name, role=role, ref=control_ref(role, name), is_password=is_password
        )
        self.clicks = 0

    def identity(self) -> ControlIdentity:
        return self._identity

    def click(self) -> None:
        self.clicks += 1


class FakeWindow:
    def __init__(self, controls: list[FakeControl], *, list_error: str | None = None) -> None:
        self.controls = controls
        self.list_error = list_error
        self.listed_max: int | None = None

    def list_controls(self, max_elements: int) -> list[FakeControl]:
        self.listed_max = max_elements
        if self.list_error:
            raise RuntimeError(self.list_error)
        return self.controls[:max_elements]

    def find_control(self, *, ref: str, name: str, role: str) -> ControlLookup:
        del ref
        wanted = name.casefold()
        matches = [c for c in self.controls if c.identity().name.casefold() == wanted]
        if role:
            matches = [c for c in matches if c.identity().role.casefold() == role.casefold()]
        if not matches:
            return ControlLookup(reason="missing")
        if len(matches) > 1:
            return ControlLookup(reason="ambiguous")
        return ControlLookup(control=matches[0])


class ScriptedWindow:
    """Returns a scripted sequence of lookups (for stale/changed tests)."""

    def __init__(self, lookups: list[ControlLookup]) -> None:
        self._lookups = list(lookups)
        self._index = 0

    def list_controls(self, max_elements: int) -> list[object]:
        del max_elements
        return []

    def find_control(self, *, ref: str, name: str, role: str) -> ControlLookup:
        del ref, name, role
        if not self._lookups:
            return ControlLookup(reason="missing")
        item = self._lookups[min(self._index, len(self._lookups) - 1)]
        self._index += 1
        return item


class FakeProvider:
    def __init__(self, window: object | None, *, error: str | None = None) -> None:
        self.window = window
        self.error = error
        self.find_calls = 0

    def find_window(self, window: str) -> object | None:
        del window
        self.find_calls += 1
        if self.error:
            raise RuntimeError(self.error)
        return self.window


def _ctx(*, dry_run: bool = False) -> ToolContext:
    return ToolContext(settings=Settings(), dry_run=dry_run)


def _patch(monkeypatch: pytest.MonkeyPatch, provider: object) -> None:
    monkeypatch.setattr(uia_tools, "_build_provider", lambda ctx: provider)


# ── import safety / laziness ───────────────────────────────────────────────


def test_import_does_not_import_pywinauto() -> None:
    import jarvis.uia  # noqa: F401

    assert not hasattr(uia_tools, "pywinauto")


def test_specs_never_build_a_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(ctx: object) -> object:
        raise AssertionError("provider must not be built by registration")

    monkeypatch.setattr(uia_tools, "_build_provider", _boom)
    assert make_ui_list_controls_spec().name == "ui_list_controls"
    assert make_ui_click_spec().name == "ui_click"


def test_missing_pywinauto_degrades_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pywinauto", None)
    provider = PywinautoUiaProvider()
    assert provider.find_window("Notepad") is None  # no desktop, no raise


# ── spec contract ──────────────────────────────────────────────────────────


def test_spec_contract() -> None:
    listing = make_ui_list_controls_spec()
    click = make_ui_click_spec()
    assert (listing.name, listing.base_tier, listing.windows_only) == ("ui_list_controls", 0, True)
    assert (click.name, click.base_tier, click.windows_only) == ("ui_click", 1, True)
    assert listing.resolve is None
    assert click.resolve is not None
    # Slice 9: a state-changing UIA click costs one task-budget unit.
    assert listing.budget_cost == 0
    assert click.budget_cost == 1
    with pytest.raises(ValidationError):
        UiaListControlsArgs(window="")
    with pytest.raises(ValidationError):
        UiaClickArgs(window="Notepad")  # control_name required
    with pytest.raises(ValidationError):
        UiaClickArgs(window="Notepad", control_name="Save", extra="nope")


def test_not_windows_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(uia_tools, "is_windows", lambda: False)
    listing = make_ui_list_controls_spec().execute(UiaListControlsArgs(window="Notepad"), _ctx())
    click = make_ui_click_spec().execute(
        UiaClickArgs(window="Notepad", control_name="Save"), _ctx()
    )
    assert listing.ok is False and "Windows" in (listing.error or "")
    assert click.ok is False and "Windows" in (click.error or "")


# ── ui_list_controls ───────────────────────────────────────────────────────


def test_list_controls_ok_bounded_and_tainted(monkeypatch: pytest.MonkeyPatch) -> None:
    window = FakeWindow(
        [
            FakeControl("Save"),
            FakeControl("Password", role="Edit", is_password=True),  # skipped
            FakeControl("", role="Text"),  # unnamed, skipped
            FakeControl("Cancel"),
        ]
    )
    _patch(monkeypatch, FakeProvider(window))
    result = make_ui_list_controls_spec().execute(UiaListControlsArgs(window="Notepad"), _ctx())
    assert result.ok is True
    assert result.tainted is True  # control names are untrusted
    assert [c["name"] for c in result.data["controls"]] == ["Save", "Cancel"]
    assert result.data["secure_controls_skipped"] == 1
    assert all(c["ref"].startswith("uia:") for c in result.data["controls"])
    assert "Button" in result.output


def test_list_controls_bounds_count(monkeypatch: pytest.MonkeyPatch) -> None:
    window = FakeWindow([FakeControl(f"Control {i}") for i in range(100)])
    _patch(monkeypatch, FakeProvider(window))
    result = make_ui_list_controls_spec().execute(UiaListControlsArgs(window="Big"), _ctx())
    assert result.ok is True
    assert len(result.data["controls"]) <= 60
    assert window.listed_max == 60


def test_list_controls_window_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, FakeProvider(None))
    result = make_ui_list_controls_spec().execute(UiaListControlsArgs(window="Ghost"), _ctx())
    assert result.ok is False
    assert "no visible window" in (result.error or "")


def test_list_controls_provider_error_is_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, FakeProvider(None, error="desktop exploded"))
    result = make_ui_list_controls_spec().execute(UiaListControlsArgs(window="X"), _ctx())
    assert result.ok is False and result.error


def test_list_controls_enumeration_error_is_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, FakeProvider(FakeWindow([], list_error="uia read failed")))
    result = make_ui_list_controls_spec().execute(UiaListControlsArgs(window="X"), _ctx())
    assert result.ok is False
    assert "could not read controls" in (result.error or "")


def test_list_controls_dry_run_skips_desktop(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(ctx: object) -> object:
        raise AssertionError("dry run must not touch the desktop")

    monkeypatch.setattr(uia_tools, "_build_provider", _boom)
    result = make_ui_list_controls_spec().execute(
        UiaListControlsArgs(window="Notepad"), _ctx(dry_run=True)
    )
    assert result.ok is True and "[dry-run]" in result.output


def test_list_controls_verify_none() -> None:
    spec = make_ui_list_controls_spec()
    assert spec.verify is not None
    out = spec.verify(UiaListControlsArgs(window="X"), ToolResult(ok=True), _ctx())
    assert out.verified is None


# ── ui_click ───────────────────────────────────────────────────────────────


def test_click_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    control = FakeControl("Save")
    _patch(monkeypatch, FakeProvider(FakeWindow([control])))
    result = make_ui_click_spec().execute(
        UiaClickArgs(window="Notepad", control_name="Save"), _ctx()
    )
    assert result.ok is True
    assert result.tainted is True
    assert result.data["name"] == "Save"
    assert result.data["ref"].startswith("uia:")
    assert control.clicks == 1


def test_click_missing_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, FakeProvider(FakeWindow([])))
    result = make_ui_click_spec().execute(
        UiaClickArgs(window="Notepad", control_name="Save"), _ctx()
    )
    assert result.ok is False
    assert "no unique control" in (result.error or "")


def test_click_ambiguous_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    first = FakeControl("Save")
    second = FakeControl("Save")
    _patch(monkeypatch, FakeProvider(FakeWindow([first, second])))
    result = make_ui_click_spec().execute(
        UiaClickArgs(window="Notepad", control_name="Save"), _ctx()
    )
    assert result.ok is False
    assert "ambiguous" in (result.error or "")
    assert first.clicks == 0 and second.clicks == 0


def test_click_stale_disappeared_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    control = FakeControl("Save")
    window = ScriptedWindow([ControlLookup(control=control), ControlLookup(reason="missing")])
    _patch(monkeypatch, FakeProvider(window))
    result = make_ui_click_spec().execute(
        UiaClickArgs(window="Notepad", control_name="Save"), _ctx()
    )
    assert result.ok is False
    assert "changed or disappeared" in (result.error or "")
    assert control.clicks == 0


def test_click_changed_identity_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    first = FakeControl("Save")
    second = FakeControl("Save", role="MenuItem")  # identity changed between lookups
    window = ScriptedWindow([ControlLookup(control=first), ControlLookup(control=second)])
    _patch(monkeypatch, FakeProvider(window))
    result = make_ui_click_spec().execute(
        UiaClickArgs(window="Notepad", control_name="Save"), _ctx()
    )
    assert result.ok is False
    assert "identity changed" in (result.error or "")
    assert first.clicks == 0 and second.clicks == 0


def test_click_password_control_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    control = FakeControl("Password", role="Edit", is_password=True)
    _patch(monkeypatch, FakeProvider(FakeWindow([control])))
    result = make_ui_click_spec().execute(
        UiaClickArgs(window="Login", control_name="Password"), _ctx()
    )
    assert result.ok is False
    assert "credential" in (result.error or "")
    assert control.clicks == 0


def test_click_window_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, FakeProvider(None))
    result = make_ui_click_spec().execute(UiaClickArgs(window="Ghost", control_name="Save"), _ctx())
    assert result.ok is False and "no visible window" in (result.error or "")


def test_click_action_failure_is_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Failing(FakeControl):
        def click(self) -> None:
            raise RuntimeError("invoke pattern unavailable")

    _patch(monkeypatch, FakeProvider(FakeWindow([_Failing("Save")])))
    result = make_ui_click_spec().execute(
        UiaClickArgs(window="Notepad", control_name="Save"), _ctx()
    )
    assert result.ok is False
    assert (result.error or "").startswith("ui_click failed:")


def test_click_dry_run_skips_desktop(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(ctx: object) -> object:
        raise AssertionError("dry run must not touch the desktop")

    monkeypatch.setattr(uia_tools, "_build_provider", _boom)
    result = make_ui_click_spec().execute(
        UiaClickArgs(window="Notepad", control_name="Save"), _ctx(dry_run=True)
    )
    assert result.ok is True and "[dry-run]" in result.output


def test_click_verify_none() -> None:
    spec = make_ui_click_spec()
    assert spec.verify is not None
    out = spec.verify(UiaClickArgs(window="X", control_name="Y"), ToolResult(ok=True), _ctx())
    assert out.verified is None


# ── resolver hook ──────────────────────────────────────────────────────────


def test_resolver_builds_uia_target(monkeypatch: pytest.MonkeyPatch) -> None:
    control = FakeControl("Save")
    _patch(monkeypatch, FakeProvider(FakeWindow([control])))
    spec = make_ui_click_spec()
    args = UiaClickArgs(window="Notepad", control_name="Save")
    target = spec.resolve(args, _ctx())
    assert isinstance(target, ResolvedTarget)
    assert target.kind == "uia"
    assert target.origin == ""
    assert (target.name, target.role) == ("Save", "Button")
    assert target.fingerprint() is not None
    assert resolve_target(spec, args, _ctx()) is not None
    assert control.clicks == 0  # resolution never authorizes or acts


def test_resolver_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = make_ui_click_spec()
    args = UiaClickArgs(window="Notepad", control_name="Save")
    # No window.
    _patch(monkeypatch, FakeProvider(None))
    assert spec.resolve(args, _ctx()) is None
    # Provider raises.
    _patch(monkeypatch, FakeProvider(None, error="boom"))
    assert spec.resolve(args, _ctx()) is None
    # Ambiguous control.
    _patch(monkeypatch, FakeProvider(FakeWindow([FakeControl("Save"), FakeControl("Save")])))
    assert spec.resolve(args, _ctx()) is None
    # Unsupported platform.
    monkeypatch.setattr(uia_tools, "is_windows", lambda: False)
    assert spec.resolve(args, _ctx()) is None
    assert resolve_target(spec, args, _ctx()) is None


# ── real adapter: no coordinate fallback, secure handling ──────────────────


def _element(name: str = "Save", role: str = "Button", *, invoke: bool = True) -> object:
    element = SimpleNamespace(element_info=SimpleNamespace(name=name, control_type=role))
    if invoke:

        def _invoke() -> None:
            element.invoked = True  # type: ignore[attr-defined]

        element.invoke = _invoke
    return element


def test_pywinauto_control_uses_invoke_not_coordinates() -> None:
    element = _element()
    control = _PywinautoControl(element, ControlIdentity("Save", "Button"))
    control.click()
    assert element.invoked is True  # type: ignore[attr-defined]


def test_pywinauto_control_refuses_without_invoke_pattern() -> None:
    control = _PywinautoControl(_element(invoke=False), ControlIdentity("Save", "Button"))
    with pytest.raises(TypeError, match="invoke"):
        control.click()  # never a coordinate/mouse fallback


def test_pywinauto_control_identity_requires_name_and_role() -> None:
    blank = SimpleNamespace(element_info=SimpleNamespace(name="", control_type="Button"))
    assert _PywinautoControl(blank, ControlIdentity("", "Button")).identity() is None
    assert _PywinautoControl(_element(), ControlIdentity("x", "y")).identity() is not None


def test_control_ref_is_deterministic_and_identity_sensitive() -> None:
    assert control_ref("Button", "Save") == control_ref("Button", "Save")
    assert control_ref("Button", "Save") != control_ref("Button", "Cancel")
    assert control_ref("Button", "Save") != control_ref("MenuItem", "Save")
