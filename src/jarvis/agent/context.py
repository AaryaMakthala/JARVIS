"""Application context: the dependency container threaded through the graph.

An :class:`AppContext` bundles everything a node needs (settings, registry,
LLM client, policy engine) so the graph stays free of global state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jarvis.config import Settings, load_settings
from jarvis.llm.client import LLMClient
from jarvis.logging_setup import get_logger
from jarvis.policy.engine import PolicyContext, PolicyEngine, RiskClassifier, UnlockManager
from jarvis.tools.base import CancelToken, ToolContext
from jarvis.tools.registry import ToolRegistry, build_default_registry


@dataclass
class AppContext:
    """Everything the agent nodes need.  Read-only after construction."""

    settings: Settings
    registry: ToolRegistry
    llm: LLMClient | None = None
    engine: PolicyEngine = field(default_factory=PolicyEngine)
    policy_ctx: PolicyContext | None = None
    unlock: UnlockManager | None = None
    classifier: RiskClassifier | None = None
    dry_run: bool = False
    cancel: CancelToken = field(default_factory=CancelToken)
    logger: logging.Logger = field(default_factory=lambda: get_logger("agent"))
    trash: Any | None = None  # TrashService | None (defaults to the real recycle bin)
    undo_log: Path | None = None  # override for the undo log path (tests)

    def tool_context(self) -> ToolContext:
        """Build a ToolContext for the current step (no secret leaks)."""
        return ToolContext(
            settings=self.settings,
            dry_run=self.dry_run,
            cancel=self.cancel,
            llm=self.llm,
            memory=None,
            logger=self.logger,
            trash=self.trash,
            unlock=self.unlock,
            undo_log=self.undo_log,
        )


def make_app_context(
    settings: Settings | None = None,
    *,
    llm: LLMClient | None = None,
    registry: ToolRegistry | None = None,
    unlock: UnlockManager | None = None,
    classifier: RiskClassifier | None = None,
    dry_run: bool = False,
    trash: Any | None = None,
    undo_log: Path | None = None,
) -> AppContext:
    """Build a ready-to-use :class:`AppContext` with defaults."""
    settings = settings or load_settings()
    registry = registry or build_default_registry(settings)
    engine = PolicyEngine()
    policy_ctx = PolicyContext(
        registry=registry, settings=settings, classifier=classifier, unlock=unlock
    )
    return AppContext(
        settings=settings,
        registry=registry,
        llm=llm,
        engine=engine,
        policy_ctx=policy_ctx,
        unlock=unlock,
        classifier=classifier,
        dry_run=dry_run,
        trash=trash,
        undo_log=undo_log,
    )
