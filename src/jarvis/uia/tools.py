"""Native Windows UI Automation tools (Stage 7, Slice 7).

``ui_list_controls`` enumerates the identifiable controls of a named native
window; ``ui_click`` clicks exactly one uniquely identifiable control.  These
tools operate on *native* application controls through UI Automation only - no
screen coordinates, no generic mouse clicks, no OCR guessing (docs/04 §2.3).

Design rules
------------
* pywinauto is imported lazily inside the real provider's methods; importing
  ``jarvis.uia`` (or building the specs) never imports it and never touches a
  desktop, so JARVIS stays importable on Linux and where the optional Windows
  automation stack is absent.
* The desktop is reached only through :class:`UiaProvider`, built by
  :func:`_build_provider` (monkeypatched in tests).
* A control's identity is ``(name, role, transient ref)`` plus password
  metadata.  ``ui_click`` re-locates *and* re-validates the identity
  immediately before clicking and refuses missing, ambiguous, stale or changed
  targets.  It never falls back to coordinates.
* Secure/password controls are never read or clicked.
* Window-derived text (control names) is untrusted: every observation is
  ``tainted=True``; the tools never infer an objective from control text.
* A listed control is merely *identifiable* - authorizing a risky control
  (send/pay/confirm/delete) is the PolicyEngine's job in a later slice.  The
  ``resolve`` hook only *observes* a target; it never authorizes.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from jarvis.platform_guard import is_windows
from jarvis.policy.targets import ResolvedTarget
from jarvis.tools.base import ToolContext, ToolResult, ToolSpec

logger = logging.getLogger("jarvis.uia.tools")

__all__ = [
    "ControlIdentity",
    "ControlLookup",
    "PywinautoUiaProvider",
    "UiaClickArgs",
    "UiaControl",
    "UiaListControlsArgs",
    "UiaProvider",
    "UiaWindow",
    "make_ui_click_spec",
    "make_ui_list_controls_spec",
]

#: Hard bounds: a huge window must not stall JARVIS or flood the answer.
_MAX_CONTROLS = 60
_MAX_NAME_CHARS = 120
_MAX_OUTPUT_CHARS = 8000
_REF_MAX = 64


def control_ref(role: str, name: str) -> str:
    """Deterministic, opaque, transient reference for ``(role, name)``.

    Recomputable from a fresh observation, so the policy gate and the act node
    derive the *same* ref for the *same* control without ever storing it.
    """
    payload = f"{str(role or '').strip().lower()}\x1f{str(name or '').strip()}"
    return "uia:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True)
class ControlIdentity:
    """Minimum identity metadata for one native control (never a value)."""

    name: str
    role: str
    ref: str = ""
    is_password: bool = False

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe metadata: name/role/ref and the password flag only."""
        return {
            "name": self.name,
            "role": self.role,
            "ref": self.ref,
            "is_password": self.is_password,
        }

    def to_target(self) -> ResolvedTarget | None:
        """Build a UIA :class:`ResolvedTarget`, or ``None`` if incomplete.

        ``ref`` is required for a stable fingerprint; an identity without a
        (recomputable) ref is treated as unreliable and refused.
        """
        ref = self.ref or control_ref(self.role, self.name)
        if not self.name or not self.role or not ref:
            return None
        return ResolvedTarget(
            kind="uia",
            ref=ref,
            name=self.name,
            role=self.role,
            origin="",  # UIA targets belong to a window, not a web origin
            is_password=self.is_password,
        )


@dataclass(frozen=True)
class ControlLookup:
    """Result of relocating a control: the control, or a refusal reason."""

    control: UiaControl | None = None
    reason: str = ""

    @property
    def ok(self) -> bool:
        """Whether exactly one reliably identifiable control was found."""
        return self.control is not None and not self.reason


@runtime_checkable
class UiaControl(Protocol):
    """A single native control that can be observed and clicked."""

    def identity(self) -> ControlIdentity | None:
        """Return the control's current identity, or ``None`` if unreadable."""

    def click(self) -> None:
        """Invoke the control's default UIA action (never a coordinate click)."""


@runtime_checkable
class UiaWindow(Protocol):
    """A native window whose controls can be enumerated and relocated."""

    def list_controls(self, max_elements: int) -> list[UiaControl]:
        """Visible, identifiable, non-secure controls, bounded by ``max_elements``."""

    def find_control(self, *, ref: str, name: str, role: str) -> ControlLookup:
        """Relocate exactly one control by identity; fail closed on ambiguity."""


@runtime_checkable
class UiaProvider(Protocol):
    """Finds a native window by name/title (tests inject a fake)."""

    def find_window(self, window: str) -> UiaWindow | None:
        """Return the named window, or ``None`` when it cannot be found."""


def _build_provider(ctx: ToolContext) -> UiaProvider:
    """Build the real UIA provider (monkeypatched in tests; no import here)."""
    del ctx
    return PywinautoUiaProvider()


# ── argument models ────────────────────────────────────────────────────────


class UiaListControlsArgs(BaseModel):
    """Arguments for ``ui_list_controls``."""

    model_config = ConfigDict(extra="forbid")

    window: str = Field(
        min_length=1,
        max_length=100,
        description="Title/name of the native window to enumerate.",
    )


class UiaClickArgs(BaseModel):
    """Arguments for ``ui_click``."""

    model_config = ConfigDict(extra="forbid")

    window: str = Field(
        min_length=1, max_length=100, description="Title/name of the target window."
    )
    control_name: str = Field(
        min_length=1,
        max_length=_MAX_NAME_CHARS,
        description="Exact accessible name of the control to click.",
    )


# ── shared helpers ─────────────────────────────────────────────────────────


def _not_windows(verb: str) -> ToolResult:
    return ToolResult(ok=False, error=f"{verb} native UI controls is only supported on Windows")


def _bounded_name(name: str) -> str:
    return " ".join(str(name or "").split())[:_MAX_NAME_CHARS]


def _locate_window(provider: UiaProvider, window: str) -> UiaWindow | None:
    """Find a window, treating any provider failure as 'not found' (fail closed)."""
    try:
        return provider.find_window(window)
    except Exception as exc:  # noqa: BLE001 - a UIA failure must never raise to the graph
        logger.warning("ui window lookup failed: %s", type(exc).__name__)
        return None


def _lookup(window: UiaWindow, *, ref: str, name: str, role: str) -> ControlLookup:
    """Relocate a control, treating any provider failure as a refusal."""
    try:
        return window.find_control(ref=ref, name=name, role=role)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ui control lookup failed: %s", type(exc).__name__)
        return ControlLookup(reason="unreadable")


# ── ui_list_controls ───────────────────────────────────────────────────────


def _run_ui_list_controls(args: UiaListControlsArgs, ctx: ToolContext) -> ToolResult:
    if ctx.dry_run:
        return ToolResult(ok=True, output=f"[dry-run] would list controls of {args.window!r}")
    if not is_windows():
        return _not_windows("listing")
    provider = _build_provider(ctx)
    window = _locate_window(provider, args.window)
    if window is None:
        return ToolResult(ok=False, error=f"no visible window matching {args.window!r}")
    try:
        controls = window.list_controls(_MAX_CONTROLS)
    except Exception as exc:  # noqa: BLE001 - enumeration failure is a refusal
        return ToolResult(ok=False, error=f"could not read controls: {type(exc).__name__}")

    lines: list[str] = []
    data: list[dict[str, Any]] = []
    secure_skipped = 0
    for control in controls:
        if len(data) >= _MAX_CONTROLS:
            break
        try:
            identity = control.identity()
        except Exception as exc:  # noqa: BLE001 - an unreadable control is skipped
            logger.debug("skipping unreadable control: %s", type(exc).__name__)
            continue
        if identity is None:
            continue
        if identity.is_password:
            secure_skipped += 1  # never expose secure controls
            continue
        name = _bounded_name(identity.name)
        role = str(identity.role or "").strip()
        if not name or not role:
            continue
        ref = identity.ref or control_ref(role, name)
        lines.append(f"[{ref}] {role} — {name!r}")
        data.append({"name": name, "role": role, "ref": ref})

    block = "\n".join(lines)
    if len(block) > _MAX_OUTPUT_CHARS:
        block = block[:_MAX_OUTPUT_CHARS] + "\n[truncated]"
    output = (
        f"Controls in {args.window!r} ({len(data)} shown):\n{block}"
        if data
        else (f"No identifiable controls found in {args.window!r}.")
    )
    return ToolResult(
        ok=True,
        output=output,
        data={"window": args.window, "controls": data, "secure_controls_skipped": secure_skipped},
        verified=None,
        tainted=True,  # control names are untrusted window-derived text
    )


def _verify_list_controls(args: BaseModel, result: ToolResult, ctx: ToolContext) -> ToolResult:
    del args, ctx
    return result.model_copy(update={"verified": None})


# ── ui_click ───────────────────────────────────────────────────────────────


def _resolve_uia_control(args: BaseModel, ctx: ToolContext) -> ResolvedTarget | None:
    """Resolve a ``(window, control_name)`` pair to one UIA target (Slice 5 hook).

    Returning ``None`` means "no single reliably identifiable control"; the
    policy gate treats that as a refusal.  This never raises for ordinary
    non-resolution and never authorizes - the PolicyEngine still decides.
    """
    if not is_windows():
        return None
    name = _bounded_name(getattr(args, "control_name", ""))
    window_name = str(getattr(args, "window", "") or "")
    if not name or not window_name:
        return None
    try:
        provider = _build_provider(ctx)
        window = _locate_window(provider, window_name)
        if window is None:
            return None
        lookup = _lookup(window, ref="", name=name, role="")
        if not lookup.ok or lookup.control is None:
            return None
        identity = lookup.control.identity()
    except Exception:  # noqa: BLE001 - fail closed per the resolver contract
        return None
    if identity is None:
        return None
    return identity.to_target()


def _run_ui_click(args: UiaClickArgs, ctx: ToolContext) -> ToolResult:
    if ctx.dry_run:
        return ToolResult(
            ok=True, output=f"[dry-run] would click {args.control_name!r} in {args.window!r}"
        )
    if not is_windows():
        return _not_windows("clicking")
    provider = _build_provider(ctx)
    window = _locate_window(provider, args.window)
    if window is None:
        return ToolResult(ok=False, error=f"no visible window matching {args.window!r}")

    name = _bounded_name(args.control_name)
    first = _lookup(window, ref="", name=name, role="")
    if not first.ok or first.control is None:
        reason = first.reason or "missing"
        return ToolResult(ok=False, error=f"no unique control named {name!r}: {reason}")
    try:
        identity = first.control.identity()
    except Exception as exc:  # noqa: BLE001
        return ToolResult(ok=False, error=f"could not read control identity: {type(exc).__name__}")
    if identity is None:
        return ToolResult(ok=False, error=f"could not read control identity for {name!r}")
    if identity.is_password:
        return ToolResult(ok=False, error=f"refusing to click a credential control: {name!r}")

    # Re-locate and re-validate the identity immediately before the action.
    second = _lookup(window, ref=identity.ref, name=identity.name, role=identity.role)
    if not second.ok or second.control is None:
        return ToolResult(
            ok=False, error=f"control {name!r} changed or disappeared before clicking"
        )
    try:
        recheck = second.control.identity()
    except Exception as exc:  # noqa: BLE001
        return ToolResult(
            ok=False, error=f"could not re-read control identity: {type(exc).__name__}"
        )
    if recheck != identity or recheck is None or recheck.is_password:
        return ToolResult(ok=False, error=f"control {name!r} identity changed before clicking")

    try:
        second.control.click()
    except Exception as exc:  # noqa: BLE001 - a failed action is never a success
        return ToolResult(ok=False, error=f"ui_click failed: {exc}")
    return ToolResult(
        ok=True,
        output=f"clicked control {identity.name!r} ({identity.role}) in {args.window!r}",
        data={"window": args.window, **identity.as_dict()},
        verified=None,  # the click's effect is not reliably observable here
        tainted=True,  # control metadata is untrusted window-derived text
    )


def _verify_ui_click(args: BaseModel, result: ToolResult, ctx: ToolContext) -> ToolResult:
    del args, ctx
    return result.model_copy(update={"verified": None})


# ── real pywinauto provider (lazy; never imported at module load) ───────────


def _is_secure(element_info: object) -> bool:
    """Whether a UIA element is a password/secure control.

    Reuses the Stage-6 perception implementation so the privacy rule has a
    single home (``jarvis.perception.ui_text``).
    """
    from jarvis.perception import ui_text

    return bool(ui_text._is_password(element_info))


def _is_visible(element_info: object) -> bool:
    """Whether a UIA element is visible; hidden controls are skipped."""
    from jarvis.perception import ui_text

    return bool(ui_text._is_visible(element_info))


class _PywinautoControl:
    """Fail-closed adapter around one pywinauto control."""

    def __init__(self, element: object, identity: ControlIdentity) -> None:
        self._element = element
        self._identity = identity

    def identity(self) -> ControlIdentity | None:
        try:
            info = getattr(self._element, "element_info", None)
            if info is None or not _is_visible(info):
                return None
            name = _bounded_name(str(getattr(info, "name", "") or ""))
            role = str(getattr(info, "control_type", "") or "").strip()
            if not name or not role:
                return None
            return ControlIdentity(
                name=name,
                role=role,
                ref=control_ref(role, name),
                is_password=_is_secure(info),
            )
        except Exception:  # noqa: BLE001 - unreadable identity means "cannot use"
            return None

    def click(self) -> None:
        """Invoke the control's UIA action; never a coordinate/mouse fallback."""
        invoke = getattr(self._element, "invoke", None)
        if not callable(invoke):
            raise TypeError("control does not expose a UI Automation invoke action")
        invoke()


class _PywinautoWindow:
    """Enumerate/relocate controls of one pywinauto window (UIA backend)."""

    def __init__(self, window: object) -> None:
        self._window = window

    def _iter_controls(self) -> list[object]:
        return list(self._window.descendants())

    def list_controls(self, max_elements: int) -> list[UiaControl]:
        controls: list[UiaControl] = []
        seen: set[str] = set()
        for element in self._iter_controls():
            if len(controls) >= max_elements:
                break
            info = getattr(element, "element_info", None)
            if info is None or not _is_visible(info) or _is_secure(info):
                continue
            name = _bounded_name(str(getattr(info, "name", "") or ""))
            role = str(getattr(info, "control_type", "") or "").strip()
            if not name or not role:
                continue
            key = f"{role.casefold()}\x1f{name.casefold()}"
            if key in seen:
                continue
            seen.add(key)
            controls.append(
                _PywinautoControl(
                    element,
                    ControlIdentity(name=name, role=role, ref=control_ref(role, name)),
                )
            )
        return controls

    def find_control(self, *, ref: str, name: str, role: str) -> ControlLookup:
        del ref  # pywinauto exposes no stable cross-call handle; match by name+role
        wanted = str(name or "").strip().casefold()
        wanted_role = str(role or "").strip().casefold()
        matches: list[object] = []
        for element in self._iter_controls():
            info = getattr(element, "element_info", None)
            if info is None:
                continue
            cname = str(getattr(info, "name", "") or "").strip().casefold()
            crole = str(getattr(info, "control_type", "") or "").strip().casefold()
            if cname != wanted:
                continue
            if wanted_role and crole != wanted_role:
                continue
            matches.append(element)
        if not matches:
            return ControlLookup(reason="missing")
        if len(matches) > 1:
            return ControlLookup(reason="ambiguous")
        element = matches[0]
        info = getattr(element, "element_info", None)
        if info is None or _is_secure(info):
            return ControlLookup(reason="secure")
        control = _PywinautoControl(element, ControlIdentity(name=name, role=role))
        if control.identity() is None:
            return ControlLookup(reason="unreadable")
        return ControlLookup(control=control)


class PywinautoUiaProvider:
    """Find a native window by title (lazy pywinauto import, fail closed)."""

    def find_window(self, window: str) -> UiaWindow | None:
        try:
            from pywinauto import Desktop
        except Exception:
            logger.debug("pywinauto unavailable; ui tools cannot run", exc_info=True)
            return None
        try:
            visible = [w for w in Desktop(backend="uia").windows() if w.is_visible()]
        except Exception:
            logger.warning("ui desktop enumeration failed", exc_info=True)
            return None
        pattern = re.compile(f".*{re.escape(window)}.*", re.IGNORECASE)
        for candidate in visible:
            try:
                if pattern.match(candidate.window_text() or ""):
                    return _PywinautoWindow(candidate)
            except Exception as exc:  # noqa: BLE001 - a vanished window is skipped
                logger.debug("skipping vanishing window: %s", type(exc).__name__)
                continue
        return None


# ── specs ──────────────────────────────────────────────────────────────────


def make_ui_list_controls_spec() -> ToolSpec:
    """Build the ``ui_list_controls`` tool (Tier 0, Windows only, tainted)."""
    return ToolSpec(
        name="ui_list_controls",
        description=(
            "List the visible, identifiable controls (name and control type) of "
            "a named native window using UI Automation. Read only; secure "
            "controls are skipped and the names are untrusted data."
        ),
        args_model=UiaListControlsArgs,
        base_tier=0,
        windows_only=True,
        timeout_s=15,
        run=_run_ui_list_controls,
        verify=_verify_list_controls,
        describe=lambda args: f"list controls of {args.window!r}",
        resolve=None,
    )


def make_ui_click_spec() -> ToolSpec:
    """Build the ``ui_click`` tool (Tier 1, Windows only, tainted)."""
    return ToolSpec(
        name="ui_click",
        description=(
            "Click exactly one named control of a native window through UI "
            "Automation. Refuses missing, ambiguous, stale or changed controls, "
            "and never falls back to screen coordinates."
        ),
        args_model=UiaClickArgs,
        base_tier=1,
        windows_only=True,
        timeout_s=15,
        budget_cost=1,
        run=_run_ui_click,
        verify=_verify_ui_click,
        describe=lambda args: f"click {args.control_name!r} in {args.window!r}",
        resolve=_resolve_uia_control,
    )
