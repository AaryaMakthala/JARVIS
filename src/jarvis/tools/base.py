"""Tool framework: ToolResult, ToolContext, CancelToken and ToolSpec.

Every tool returns a :class:`ToolResult` and never raises to the graph
(docs/04_TOOLS_SPEC.md section 1).  A :class:`ToolSpec` declares its Pydantic
args model, base tier, verification function and a canonical ``describe()``.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from pydantic import BaseModel, Field

from jarvis.config import Settings

if TYPE_CHECKING:
    from jarvis.policy.targets import ResolvedTarget

T = TypeVar("T")


class CancelToken:
    """Simple cancellation flag shared with long-running tools."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        """Request cancellation."""
        self._event.set()

    @property
    def cancelled(self) -> bool:
        """Return ``True`` once :meth:`cancel` has been requested."""
        return self._event.is_set()

    def check(self) -> None:
        """Raise :class:`Cancelled` when cancellation was requested."""
        if self._event.is_set():
            raise Cancelled()


class Cancelled(RuntimeError):
    """Raised by tools that observe a cancelled run."""


class ToolResult(BaseModel):
    """Everything a tool returns; never raised as an exception."""

    ok: bool
    output: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    tainted: bool = False
    verified: bool | None = None
    #: Human-facing explanation of the ``verified`` status, set by ``verify()``
    #: when ``verified is None`` for a step whose completion cannot be observed
    #: (a *best-effort* action, e.g. ``lock_computer``: the Windows lock screen
    #: is not visible to a user-mode process).  It is the only caveat shown for
    #: such a step in the final answer (the generic wording was removed in
    #: 4.14), and is never a claim of verification.
    verify_note: str = ""


@dataclass
class ToolContext:
    """Dependency container handed to every tool invocation."""

    settings: Settings
    dry_run: bool = False
    cancel: CancelToken | None = None
    llm: Any | None = None  # LLMClient | None (imported lazily to avoid cycles)
    memory: Any | None = None
    logger: logging.Logger | None = None
    scratch_dir: Path | None = None
    trash: Any | None = None  # TrashService | None (defaults to the real recycle bin)
    unlock: Any | None = None  # UnlockManager | None (shared with the policy engine)
    undo_log: Path | None = None  # override for the undo log path (tests)
    voice: Any | None = None  # VoiceToolsFacade | None (live voice loop for dictation tools)
    browser: Any | None = (
        None  # BrowserSession | None (managed Chrome, Stage 7; never started here)
    )
    whatsapp_browser: Any | None = (
        None  # LazyProfileBrowser | None (WhatsApp profile, Stage 8; never started here)
    )
    #: The sha256 digests the policy gate bound this dispatch's approval to.
    #: Set by ``act`` only for a ``ToolSpec.bind_content_hash`` tool, and only
    #: after act has re-hashed the same resolved paths itself (Stage 8.4d).
    #: Never derived from planner output.  A tool must treat ``None``/empty as
    #: "no approval binding" and refuse (fail closed).
    approved_content_hashes: tuple[str, ...] | None = None
    #: The stable identity of the interaction target the policy gate bound this
    #: dispatch's approval to.  Set by ``act`` only for a
    #: ``ToolSpec.bind_target_identity`` tool, and only after act has
    #: re-resolved the target itself.  A tool must treat ``None``/empty as
    #: "no approval binding" and refuse (fail closed).
    approved_target_identity: str | None = None
    #: The shared, in-memory browser result sets (Stage 8, P2).  A
    #: ``jarvis.browser.results.BrowserResultStore`` or ``None`` when the caller
    #: wired none; ``None`` simply disables position/title selection against a
    #: result list (the tool then refuses instead of guessing).
    browser_results: Any | None = None


ArgsModel = type[BaseModel]


class TargetResolver(Protocol):
    """Optional ``resolve(args, ctx)`` hook that identifies an action's target.

    Implementations come from the Stage-7 browser/UIA backends (Slices 6-7).
    A resolver only *observes* the runtime (the current page or the active
    window) and returns identity metadata as a :class:`ResolvedTarget`, or
    ``None`` when no element is involved or none can be reliably observed.
    It must never make an authorization decision - the PolicyEngine is the
    sole authority, and it decides *after* this observation.
    """

    def __call__(self, args: BaseModel, ctx: ToolContext) -> ResolvedTarget | None: ...


@dataclass
class ToolSpec:
    """The declarative contract for one executable tool."""

    name: str
    description: str
    args_model: ArgsModel
    base_tier: int
    windows_only: bool = False
    timeout_s: int = 30
    #: Names of args model fields that are filesystem paths (policy/paths).
    path_args: tuple[str, ...] = ()
    #: Task-scope action-budget units this tool costs per dispatch (Stage 7,
    #: Slice 9).  ``0`` (the default, so every existing tool is unaffected) means
    #: the tool never consumes budget; state-changing browser/UIA actions declare
    #: ``1`` and the act node reserves it atomically immediately before dispatch.
    budget_cost: int = 0
    #: When true, the policy gate binds the approval to the sha256 *content* of
    #: every resolved path argument, and the act node recomputes it from a fresh
    #: read immediately before dispatch, refusing on any change, missing file or
    #: over-cap file (Stage 8.4d).  Default off, so no existing tool's approval
    #: or hashing is affected.  Use only for tools whose action must be bound to
    #: the exact bytes of a file, not just its path (``whatsapp_send_file``).
    bind_content_hash: bool = False
    #: When true, the act node hands the tool the *stable identity* of the
    #: target the gate authorized (``ResolvedTarget.ref``) and the tool must
    #: re-derive that identity from a fresh observation immediately before it
    #: acts, refusing when it no longer holds (Stage 8.6).  Default off, so no
    #: existing tool is affected.  Use only for tools whose target is named by
    #: a query rather than by a handle the backend issued (``browser_open_result``).
    bind_target_identity: bool = False
    #: Optional callables; defaults keep unknown tools safe.
    run: Callable[[BaseModel, ToolContext], ToolResult] = field(
        default=lambda args, ctx: ToolResult(ok=False, error="tool has no run implementation")
    )
    #: Optional read-only precondition check: ``precheck(args, ctx)`` returns an
    #: error string when the action must not even be offered for confirmation,
    #: or ``None`` when it passes.  It must have no side effects and must never
    #: authorize anything - a failure can only block.  The policy gate runs it
    #: *before* ``interrupt()`` (so an invalid target is not confirmed or
    #: unlocked) and ``act`` runs it again immediately before dispatch.  An
    #: exception is treated as a failure (fail closed).
    precheck: Callable[[BaseModel, ToolContext], str | None] | None = None
    verify: Callable[[BaseModel, ToolResult, ToolContext], ToolResult] | None = None
    describe: Callable[[BaseModel], str] = field(
        default=lambda args: f"{args.__class__.__name__}({_brief_args(args)})"
    )
    #: Optional runtime target resolver: ``resolve(args, ctx)`` returns the
    #: ``ResolvedTarget`` the action would operate on, or ``None`` when no
    #: element is involved or none can be reliably observed.  Deliberately
    #: separate from ``execute()``: registration and ordinary execution never
    #: call it - the policy gate resolves and authorizes *before* act()
    #: dispatches (docs/03 §5 T4).  A resolver must not raise for ordinary
    #: non-resolution; if one does, ``resolve_target()`` fails closed.
    resolve: Callable[[BaseModel, ToolContext], ResolvedTarget | None] | None = None

    def execute(self, args: BaseModel, ctx: ToolContext) -> ToolResult:
        """Run the tool honouring dry-run, cancellation and timeouts.

        Never raises: exceptions become ``ToolResult(ok=False, ...)``.
        """
        logger = ctx.logger or logging.getLogger(f"jarvis.tools.{self.name}")
        started = time.perf_counter()
        logger.info("tool start name=%s", self.name)
        try:
            if ctx.cancel is not None:
                ctx.cancel.check()
            if ctx.dry_run:
                return self._dry_run_result(args)
            result = self.run(args, ctx)
        except Cancelled:
            result = ToolResult(ok=False, error="cancelled")
        except Exception as exc:  # noqa: BLE001 - tools never raise to the graph
            result = ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}")
        result.data.setdefault("duration_ms", _elapsed_ms(started))
        err = (result.error or "")[:200]
        logger.info(
            "tool end %s ok=%s verified=%s error=%s", self.name, result.ok, result.verified, err
        )
        return result

    def _dry_run_result(self, args: BaseModel) -> ToolResult:
        """Produce a side-effect-free result for dry runs."""
        return ToolResult(ok=True, output=f"[dry-run] would run {self.name}: {_brief_args(args)}")

    def run_verified(
        self, args: BaseModel, ctx: ToolContext, _result: ToolResult | None = None
    ) -> ToolResult:
        """Run the tool and then apply :meth:`verify` when present."""
        result = self.execute(args, ctx)
        return self.apply_verify(args, result, ctx)

    def apply_verify(self, args: BaseModel, result: ToolResult, ctx: ToolContext) -> ToolResult:
        """Attach a ``verified`` status to ``result`` via this tool's verify()."""
        if result.ok and self.verify is not None:
            try:
                verified = self.verify(args, result, ctx)
            except Exception:  # noqa: BLE001 - verification must not raise
                return result.model_copy(update={"verified": False})
            return result.model_copy(
                update={"verified": verified.verified, "verify_note": verified.verify_note}
            )
        return result

    def path_arg_values(self, args: BaseModel) -> list[tuple[str, str]]:
        """Return (name, raw string) for every declared path argument.

        A field may be a single path string or a list of them (delete_path);
        every element is yielded so the engine checks all of them.
        """
        values: list[tuple[str, str]] = []
        for name in self.path_args:
            value = getattr(args, name, None)
            if isinstance(value, str):
                values.append((name, value))
            elif isinstance(value, list):
                values.extend((name, item) for item in value if isinstance(item, str))
        return values


def resolve_target(spec: ToolSpec, args: BaseModel, ctx: ToolContext) -> ResolvedTarget | None:
    """Resolve the interaction target for ``spec``, failing closed.

    Invokes ``spec.resolve`` when it is set and returns its result.  A resolver
    that returns ``None`` or raises means "no target could be reliably
    observed"; both become ``None`` here so a broken resolver can never crash
    the graph.  Returning ``None`` never approves anything - the caller (the
    policy gate) treats an unresolvable target as a refusal, because the
    *decision* to authorize belongs solely to the PolicyEngine.
    """
    if spec.resolve is None:
        return None
    try:
        return spec.resolve(args, ctx)
    except Exception:  # noqa: BLE001 - a resolver must never crash the pipeline
        return None


def _brief_args(args: BaseModel) -> str:
    """Compact, deterministic rendering of args for dry-run summaries."""
    import json

    return json.dumps(args.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def ensure_bool(value: Any, default: bool = False) -> bool:
    """Coerce an untrusted value to a bool without raising."""
    if isinstance(value, bool):
        return value
    return default
