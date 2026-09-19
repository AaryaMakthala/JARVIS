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
from typing import Any, TypeVar

from pydantic import BaseModel, Field

from jarvis.config import Settings

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


ArgsModel = type[BaseModel]


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
    #: Optional callables; defaults keep unknown tools safe.
    run: Callable[[BaseModel, ToolContext], ToolResult] = field(
        default=lambda args, ctx: ToolResult(ok=False, error="tool has no run implementation")
    )
    verify: Callable[[BaseModel, ToolResult, ToolContext], ToolResult] | None = None
    describe: Callable[[BaseModel], str] = field(
        default=lambda args: f"{args.__class__.__name__}({_brief_args(args)})"
    )

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
        logger.info("tool end name=%s ok=%s verified=%s", self.name, result.ok, result.verified)
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
            return result.model_copy(update={"verified": verified.verified})
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
