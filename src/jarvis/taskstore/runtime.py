"""Composition helpers for the application-owned task event store.

The task store itself is deliberately independent of the memory backend.  This
module is the one place that knows how to derive its *trusted* vocabulary from
the live configuration (providers, models, tools, graph nodes and failure
codes), so callers never have to invent allow-list entries.  Opening is
fail-soft: an unusable file yields a :class:`NullTaskEventStore` and the rest
of the agent keeps working.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from jarvis import config
from jarvis.taskstore.models import TaskOwner, TaskStoreError
from jarvis.taskstore.schema import SchemaError
from jarvis.taskstore.store import NullTaskEventStore, TaskStore, open_task_store
from jarvis.taskstore.vocabulary import TaskFailureCategory, TaskStoreVocabulary

__all__ = ["build_task_vocabulary", "open_runtime_task_store"]

_LOG = logging.getLogger(__name__)

_MODEL_ROLES = ("planner", "fast", "vision")


def _configured_models(settings: config.Settings) -> set[tuple[str, str]]:
    """Every (provider, model) pair the configuration can select."""
    pairs: set[tuple[str, str]] = set()
    for provider in settings.llm.provider_order:
        name = provider.strip().lower()
        if not name:
            continue
        for role in _MODEL_ROLES:
            model = settings.llm.model_for(provider, role).strip()
            if model:
                pairs.add((name, model))
    return pairs


def build_task_vocabulary(
    settings: config.Settings | None = None,
    registry: object | None = None,
) -> TaskStoreVocabulary:
    """Build the immutable allow-list from the trusted application state.

    ``registry`` only needs a ``names()`` method.  When it is omitted the tool
    set is empty, which is sufficient for read-only status checks.
    """
    settings = settings or config.load_settings()
    from jarvis.agent.graph import GRAPH_NODE_NAMES
    from jarvis.agent.state import FailureCode

    tools = registry.names() if registry is not None else ()
    return TaskStoreVocabulary(
        providers=frozenset(settings.llm.provider_order),
        models=frozenset(_configured_models(settings)),
        tools=frozenset(tools),
        nodes=GRAPH_NODE_NAMES,
        failure_codes=frozenset(code.value for code in FailureCode),
        failure_categories=frozenset(category.value for category in TaskFailureCategory),
    )


def open_runtime_task_store(
    settings: config.Settings | None = None,
    registry: object | None = None,
    *,
    path: str | Path | None = None,
    create: bool = True,
    read_only: bool = False,
    owner: TaskOwner | None = None,
    logger: logging.Logger | None = None,
) -> TaskStore:
    """Open the task store, falling back to a no-op store on any failure.

    ``create=True`` initializes a missing database; ``read_only=True`` never
    writes or migrates.  A failure is reported once and then the agent keeps
    running without durable task history, exactly like a missing memory store.
    """
    settings = settings or config.load_settings()
    log = logger or _LOG
    vocabulary = build_task_vocabulary(settings, registry)
    try:
        return open_task_store(
            path,
            create=create,
            read_only=read_only,
            vocabulary=vocabulary,
            owner=owner,
        )
    except (TaskStoreError, SchemaError, sqlite3.Error, OSError) as exc:
        log.warning("task store unavailable: %s", exc)
        return NullTaskEventStore(vocabulary=vocabulary)
