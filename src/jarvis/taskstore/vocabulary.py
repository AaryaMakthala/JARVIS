"""Immutable operational vocabularies and closed taskstore enums."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class TaskSource(str, Enum):
    """Allowed task sources."""

    TERMINAL = "terminal"
    VOICE = "voice"
    BENCHMARK = "benchmark"


class TaskOwnerKind(str, Enum):
    """Allowed task owner kinds."""

    DAEMON = "daemon"
    CLI = "cli"
    BENCHMARK = "benchmark"


class TaskModelRole(str, Enum):
    """Allowed model roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    PLANNER = "planner"


class TaskApprovalState(str, Enum):
    """Allowed approval states."""

    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class TaskOutcome(str, Enum):
    """Allowed terminal task outcomes."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class TaskSeverity(str, Enum):
    """Allowed warning severities."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class TaskReasonCode(str, Enum):
    """Allowed operational reason codes."""

    NONE = "none"
    POLICY_DENIED = "policy_denied"
    VALIDATION_FAILED = "validation_failed"
    TOOL_ERROR = "tool_error"
    LLM_ERROR = "llm_error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    VERIFICATION_FAILED = "verification_failed"
    INTERNAL_ERROR = "internal_error"
    UNKNOWN = "unknown"


class TaskMessageKind(str, Enum):
    """Allowed message kinds."""

    STATUS = "status"
    PROGRESS = "progress"
    TOOL = "tool"
    APPROVAL = "approval"
    CLARIFICATION = "clarification"
    WARNING = "warning"
    FAILURE = "failure"


class TaskEventCode(str, Enum):
    """Allowed operational note/event codes."""

    STARTED = "started"
    STAGE = "stage"
    TOOL = "tool"
    RETRY = "retry"
    FALLBACK = "fallback"
    REPLAN = "replan"
    PROGRESS = "progress"
    HEARTBEAT = "heartbeat"
    CHECKPOINT = "checkpoint"


class TaskFailureCategory(str, Enum):
    """Allowed failure categories."""

    POLICY = "policy"
    VALIDATION = "validation"
    TOOL = "tool"
    LLM = "llm"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    VERIFICATION = "verification"
    INTERNAL = "internal"
    UNKNOWN = "unknown"


def _text_set(values: Iterable[str], *, name: str, lower: bool) -> frozenset[str]:
    if isinstance(values, str):
        raise TypeError(f"{name} must be a collection of strings")
    result: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise TypeError(f"{name} values must be strings")
        text = value.strip()
        if not text:
            raise ValueError(f"{name} values must be non-empty")
        result.add(text.lower() if lower else text)
    return frozenset(result)


@dataclass(frozen=True, slots=True)
class TaskStoreVocabulary:
    """Trusted, immutable operational vocabulary supplied by the application."""

    providers: frozenset[str]
    models: frozenset[tuple[str, str]]
    tools: frozenset[str]
    nodes: frozenset[str]
    failure_codes: frozenset[str]
    failure_categories: frozenset[str]

    def __post_init__(self) -> None:
        providers = _text_set(self.providers, name="providers", lower=True)
        tools = _text_set(self.tools, name="tools", lower=True)
        nodes = _text_set(self.nodes, name="nodes", lower=False)
        failure_codes = _text_set(self.failure_codes, name="failure_codes", lower=False)
        categories = _text_set(self.failure_categories, name="failure_categories", lower=True)
        allowed_categories = frozenset(category.value for category in TaskFailureCategory)
        if not categories <= allowed_categories:
            raise ValueError("failure_categories contains an unknown category")

        models: set[tuple[str, str]] = set()
        for pair in self.models:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise TypeError("models must contain (provider, model) pairs")
            provider, model = pair
            if not isinstance(provider, str) or not isinstance(model, str):
                raise TypeError("model pairs must contain strings")
            provider = provider.strip().lower()
            model = model.strip()
            if not provider or not model:
                raise ValueError("model pairs must be non-empty")
            if provider not in providers:
                raise ValueError("model pair provider is not configured")
            models.add((provider, model))

        object.__setattr__(self, "providers", providers)
        object.__setattr__(self, "models", frozenset(models))
        object.__setattr__(self, "tools", tools)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "failure_codes", failure_codes)
        object.__setattr__(self, "failure_categories", categories)
