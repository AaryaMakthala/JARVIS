"""Application context: the dependency container threaded through the graph.

An :class:`AppContext` bundles everything a node needs (settings, registry,
LLM client, policy engine) so the graph stays free of global state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jarvis.agent.session_context import SessionContext
from jarvis.agent.target_binding import TargetBindingStore
from jarvis.browser.results import BrowserResultStore
from jarvis.config import Settings, load_settings
from jarvis.llm.client import LLMClient
from jarvis.logging_setup import get_logger
from jarvis.memory import MemoryBackend
from jarvis.policy.engine import PolicyContext, PolicyEngine, RiskClassifier, UnlockManager
from jarvis.policy.scope import ScopeRegistry
from jarvis.tools.base import CancelToken, ToolContext
from jarvis.tools.registry import ToolRegistry, build_default_registry


@dataclass
class VoiceToolsFacade:
    """Bridge from agent tools to the running voice loop.

    The daemon fills :attr:`service` after it starts :class:`VoiceService`; the
    ``loop`` fallback lets tests and other embedders hand over a live loop
    directly.  Tools read :attr:`active_loop` and must behave honestly when
    nothing is running.
    """

    service: Any | None = None
    loop: Any | None = None

    @property
    def active_loop(self) -> Any | None:
        """The live :class:`VoiceLoop`, or ``None`` when not active."""
        if self.service is not None:
            try:
                if self.service.is_active() and self.service.loop is not None:
                    return self.service.loop
            except Exception:  # noqa: BLE001
                return None
        return self.loop


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
    voice: VoiceToolsFacade | None = None  # live voice loop for dictation tools
    memory: MemoryBackend | None = None  # bounded, secret-free memory store
    #: Managed Chrome session (Stage 7).  A passed-in ``BrowserSession`` is
    #: owned by the daemon and threaded to tools; construction never starts a
    #: browser, and tool calls fail cleanly when it is ``None``.
    browser: Any | None = None
    #: Managed Chrome session on the dedicated WhatsApp profile (Stage 8).  A
    #: second, independent profile - never the general one - so the
    #: authenticated WhatsApp Web session is isolated (ROADMAP_V2.1 B6).  It is
    #: a lazy holder (see ``jarvis.browser.lazy``): construction never starts
    #: Chrome, and tools fail cleanly when it is ``None`` or cannot start.  It
    #: deliberately does not grant the managed-browser app token for generic
    #: ``type_text``; the WhatsApp tool drives this profile internally.
    whatsapp_browser: Any | None = None
    #: Transient, task-scoped target-bindings store (Stage 7).  In-memory only:
    #: bindings, scope grants and resolved targets are never written to
    #: ``AgentState`` or the checkpoint store, so a daemon restart loses them
    #: and requires fresh authorization (fail closed).  The PolicyEngine remains
    #: the sole authorization authority; this store holds no decisions.
    target_bindings: TargetBindingStore = field(default_factory=TargetBindingStore)
    #: Transient, task-scoped browser/UIA scope-grant registry (Stage 7, Slice 8).
    #: Holds approved domain scopes and their atomic action budgets; like
    #: ``target_bindings`` it is in-memory only and never checkpointed, so a
    #: daemon restart requires fresh authorization.
    scope_registry: ScopeRegistry = field(default_factory=ScopeRegistry)
    #: Task-session artifact context (Stage 8, D19).  Holds the artifacts a
    #: tool just produced (e.g. the photo ``take_photo`` saved) so a follow-up
    #: like "delete it" can resolve to a real path.  IN-MEMORY ONLY and never
    #: written to ``AgentState`` or the checkpoint store, so a daemon restart
    #: forgets every artifact; a resolved path is only a proposal and is
    #: re-checked like any other path before a tool acts on it.
    session_context: SessionContext = field(default_factory=SessionContext)
    #: Bounded, in-memory result sets of the managed browser (Stage 8, P2).  The
    #: entries ``browser_find`` listed are remembered per browser session so a
    #: later turn can say "open the first link".  IN-MEMORY ONLY and never
    #: written to ``AgentState`` or the checkpoint store.  A selection is a
    #: proposal: the policy gate and the target binding still authorize the
    #: action, and a stale, expired or navigated-away set is refused.
    browser_results: BrowserResultStore = field(default_factory=BrowserResultStore)

    def tool_context(self) -> ToolContext:
        """Build a ToolContext for the current step (no secret leaks)."""
        return ToolContext(
            settings=self.settings,
            dry_run=self.dry_run,
            cancel=self.cancel,
            llm=self.llm,
            memory=self.memory,
            logger=self.logger,
            trash=self.trash,
            unlock=self.unlock,
            undo_log=self.undo_log,
            voice=self.voice,
            browser=self.browser,
            whatsapp_browser=self.whatsapp_browser,
            browser_results=self.browser_results,
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
    voice: VoiceToolsFacade | None = None,
    memory: MemoryBackend | None = None,
    target_bindings: TargetBindingStore | None = None,
    browser: Any | None = None,
    whatsapp_browser: Any | None = None,
    scope_registry: ScopeRegistry | None = None,
    session_context: SessionContext | None = None,
    browser_results: BrowserResultStore | None = None,
) -> AppContext:
    """Build a ready-to-use :class:`AppContext` with defaults."""
    settings = settings or load_settings()
    registry = registry or build_default_registry(settings)
    engine = PolicyEngine()
    # The managed browser's app token gates generic ``type_text`` (Slice 9):
    # only exposed to the engine when a browser session is actually wired up.
    from jarvis.browser.context import MANAGED_BROWSER_CHANNEL

    policy_ctx = PolicyContext(
        registry=registry,
        settings=settings,
        classifier=classifier,
        unlock=unlock,
        managed_browser_app=MANAGED_BROWSER_CHANNEL if browser is not None else "",
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
        voice=voice,
        memory=memory,
        target_bindings=target_bindings or TargetBindingStore(),
        browser=browser,
        whatsapp_browser=whatsapp_browser,
        scope_registry=scope_registry
        if scope_registry is not None
        else ScopeRegistry(default_max_actions=int(settings.browser.task_scope_max_actions)),
        session_context=session_context
        if session_context is not None
        else SessionContext(
            max_artifacts=int(settings.session.max_artifacts),
            ttl_seconds=float(settings.session.artifact_ttl_seconds),
        ),
        browser_results=browser_results if browser_results is not None else BrowserResultStore(),
    )
