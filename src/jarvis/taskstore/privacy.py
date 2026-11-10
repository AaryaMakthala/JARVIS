"""Privacy guards for the independent task event store.

Task history is operational data, not a conversation archive.  This module
keeps that boundary explicit: identifiers and labels are field-specific,
event payloads are allow-listed per event type, and free-text request/error
fields are never accepted.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Mapping
from enum import Enum
from typing import Any

from .vocabulary import (
    TaskApprovalState,
    TaskEventCode,
    TaskFailureCategory,
    TaskMessageKind,
    TaskModelRole,
    TaskOutcome,
    TaskOwnerKind,
    TaskReasonCode,
    TaskRequestClass,
    TaskRouteReason,
    TaskSeverity,
    TaskSource,
    TaskStoreVocabulary,
)

MAX_TASK_ID_CHARS = 36
MAX_MODEL_CHARS = 128
MAX_DETAIL_ITEMS = 16
MAX_COUNT = 1_000_000_000
MAX_DURATION_MS = 86_400_000

SOURCE_VALUES = frozenset(source.value for source in TaskSource)
OWNER_KIND_VALUES = frozenset(kind.value for kind in TaskOwnerKind)
ROLE_VALUES = frozenset(role.value for role in TaskModelRole)
FAILURE_CATEGORIES = frozenset(category.value for category in TaskFailureCategory)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
_PROVIDER_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,127}$")
_TOOL_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_FAILURE_CODE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
_OWNER_INSTANCE_RE = re.compile(r"^[a-f0-9]{32}$")
_DRIVE_PATH_RE = re.compile(r"^[A-Za-z]:")
_TRAVERSAL_RE = re.compile(r"(?:^|[\\/])\.\.(?:[\\/]|$)")
_URI_PATH_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_PHONE_SEPARATORS_RE = re.compile(r"[\s().+-]")
_PATH_EXTENSIONS = frozenset(
    {
        ".bat",
        ".cmd",
        ".csv",
        ".db",
        ".dll",
        ".doc",
        ".docx",
        ".exe",
        ".ini",
        ".jpeg",
        ".jpg",
        ".json",
        ".key",
        ".log",
        ".md",
        ".pdf",
        ".pem",
        ".png",
        ".ps1",
        ".py",
        ".sqlite",
        ".txt",
        ".xls",
        ".xlsx",
        ".xml",
        ".yaml",
        ".yml",
        ".zip",
    }
)

SECRET_KEY_HINTS = (
    "password",
    "passwd",
    "passcode",
    "secret",
    "api_key",
    "apikey",
    "token",
    "credential",
    "pin",
    "otp",
    "authorization",
    "cookie",
)

_REDACTED = "[redacted]"
_SECRET_ASSIGNMENT = re.compile(
    r"[a-z0-9_.-]*(?:" + "|".join(SECRET_KEY_HINTS) + r")[a-z0-9_.-]*"
    r"[\"']?\s*(?P<sep>[:=])\s*[\"']?(?P<value>[^\"',;}\s]*)",
    re.IGNORECASE,
)
_TOKEN_PREFIX_PATTERNS = (
    re.compile(r"(?i)(?:^|[_-])sk[-_][A-Za-z0-9_-]{8,}"),
    re.compile(r"(?i)^(?:ghp|gho|ghu|ghs)_[A-Za-z0-9]{4,}"),
    re.compile(r"(?i)^github_pat_[A-Za-z0-9_]{8,}"),
    re.compile(r"(?i)^xox[baprs]-[A-Za-z0-9-]{8,}"),
    re.compile(r"^AKIA[0-9A-Z]{12,}"),
    re.compile(r"^ASIA[0-9A-Z]{12,}"),
    re.compile(r"^AIza[0-9A-Za-z_-]{20,}"),
    re.compile(r"^eyJ[A-Za-z0-9_-]{10,}\."),
    re.compile(r"(?i)^bearer[_-][A-Za-z0-9._-]{8,}"),
)

_DETAIL_POLICIES: dict[str, dict[str, str]] = {
    "task_created": {
        "owner_kind": "owner_kind",
        "source": "source",
    },
    "task_started": {
        "attempt": "count",
        "owner_kind": "owner_kind",
    },
    "stage_changed": {
        "stage": "stage",
    },
    "status_changed": {
        "reason_code": "reason_code",
        "status": "status",
    },
    "plan_created": {
        "step_count": "count",
    },
    "route": {
        "model_role": "role",
        "reason_code": "route_reason",
        "task_class": "request_class",
    },
    "llm_call": {
        "duration_ms": "duration",
        "failure_category": "failure_category",
        "model": "model",
        "provider": "provider",
        "role": "role",
    },
    "note": {
        "attempt": "count",
        "count": "count",
        "duration_ms": "duration",
        "event_code": "event_code",
        "message_kind": "message_kind",
        "node": "node",
        "retry_count": "count",
    },
    "warning": {
        "attempt": "count",
        "message_kind": "message_kind",
        "node": "node",
        "reason_code": "reason_code",
        "severity": "severity",
    },
    "failure": {
        "attempt": "count",
        "failure_category": "failure_category",
        "failure_code": "failure_code",
        "node": "node",
        "reason_code": "reason_code",
        "tool": "tool",
    },
    "confirmation_requested": {
        "approval_state": "approval_state",
        "step_id": "step_id",
        "tool": "tool",
    },
    "clarification_requested": {
        "node": "node",
        "reason_code": "reason_code",
        "step_id": "step_id",
    },
    "tool_started": {
        "attempt": "count",
        "step_id": "step_id",
        "tool": "tool",
    },
    "tool_finished": {
        "attempt": "count",
        "duration_ms": "duration",
        "ok": "bool",
        "step_id": "step_id",
        "tool": "tool",
        "verified": "bool",
    },
    "task_finished": {
        "count": "count",
        "duration_ms": "duration",
        "event_count": "count",
        "failure_category": "failure_category",
        "failure_code": "failure_code",
        "outcome": "outcome",
        "retry_count": "count",
    },
}

_REQUIRED_DETAIL_KEYS: dict[str, frozenset[str]] = {
    "task_created": frozenset({"source"}),
    "stage_changed": frozenset({"stage"}),
    "status_changed": frozenset({"status"}),
    "plan_created": frozenset({"step_count"}),
    "route": frozenset({"task_class", "model_role", "reason_code"}),
    "llm_call": frozenset({"provider", "model", "role"}),
    "note": frozenset({"event_code"}),
    "warning": frozenset({"severity", "reason_code"}),
    "failure": frozenset({"failure_code", "failure_category"}),
    "confirmation_requested": frozenset({"approval_state", "step_id"}),
    "clarification_requested": frozenset({"step_id"}),
    "tool_started": frozenset({"tool", "step_id"}),
    "tool_finished": frozenset({"tool", "step_id", "ok", "verified", "duration_ms"}),
    "task_finished": frozenset({"outcome"}),
}


def redact_secrets(text: str) -> str:
    """Replace credential-looking assignments in free text."""
    if not text:
        return ""

    def _sub(match: re.Match[str]) -> str:
        value = match.group("value")
        if not value or value == _REDACTED:
            return match.group(0)
        return f"{match.group(0)[: match.start('value') - match.start(0)]}{_REDACTED}"

    return _SECRET_ASSIGNMENT.sub(_sub, text)


def looks_secret(text: str) -> bool:
    """True when ``text`` contains assignment-style or standalone credentials."""
    if not text:
        return False
    for match in _SECRET_ASSIGNMENT.finditer(text):
        value = match.group("value")
        if value and value != _REDACTED:
            return True
    return any(pattern.search(text) for pattern in _TOKEN_PREFIX_PATTERNS)


def looks_phone(text: str) -> bool:
    """True when ``text`` is a phone-number-shaped value."""
    compact = _PHONE_SEPARATORS_RE.sub("", text)
    return compact.isdigit() and 7 <= len(compact) <= 15


def looks_like_path(text: str) -> bool:
    """True when ``text`` looks like an absolute, traversal, or file path."""
    if not text:
        return False
    if "\\" in text or text.startswith("/") or _URI_PATH_RE.match(text):
        return True
    if _DRIVE_PATH_RE.match(text) or _TRAVERSAL_RE.search(text):
        return True
    if "/" not in text:
        return False
    name = text.rsplit("/", 1)[-1]
    dot = name.rfind(".")
    return dot > 0 and name[dot:].lower() in _PATH_EXTENSIONS


def _unsafe_value(text: str) -> bool:
    return looks_secret(text) or looks_phone(text) or looks_like_path(text)


def _canonical_uuid(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    text = value.strip()
    try:
        parsed = uuid.UUID(text)
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"{field} must be a canonical UUID") from exc
    canonical = str(parsed)
    if text.lower() != canonical:
        raise ValueError(f"{field} must be a canonical UUID")
    return canonical


def safe_task_id(value: str) -> str:
    """Validate a caller-supplied task identifier as a canonical UUID."""
    return _canonical_uuid(value, field="task_id")


def safe_step_id(value: str) -> str:
    """Validate a caller-supplied step identifier as a canonical UUID."""
    return _canonical_uuid(value, field="step_id")


def _enum_value(value: Any, enum_type: type[Enum]) -> str | None:
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    allowed = {member.value for member in enum_type}
    return text if text in allowed else None


def safe_source(value: Any) -> str | None:
    """Return a supported task source, or ``None``."""
    return _enum_value(value, TaskSource)


def safe_owner_kind(value: Any) -> str | None:
    """Return a supported owner kind, or ``None``."""
    return _enum_value(value, TaskOwnerKind)


def safe_role(value: Any) -> str | None:
    """Return a supported conversation role, or ``None``."""
    return _enum_value(value, TaskModelRole)


def safe_approval_state(value: Any) -> str | None:
    """Return a supported approval state, or ``None``."""
    return _enum_value(value, TaskApprovalState)


def safe_outcome(value: Any) -> str | None:
    """Return a supported terminal outcome, or ``None``."""
    return _enum_value(value, TaskOutcome)


def safe_severity(value: Any) -> str | None:
    """Return a supported severity, or ``None``."""
    return _enum_value(value, TaskSeverity)


def safe_reason_code(value: Any) -> str | None:
    """Return a supported operational reason code, or ``None``."""
    return _enum_value(value, TaskReasonCode)


def safe_message_kind(value: Any) -> str | None:
    """Return a supported message kind, or ``None``."""
    return _enum_value(value, TaskMessageKind)


def safe_event_code(value: Any) -> str | None:
    """Return a supported event code, or ``None``."""
    return _enum_value(value, TaskEventCode)


def safe_request_class(value: Any) -> str | None:
    """Return a supported deterministic request class, or ``None``."""
    return _enum_value(value, TaskRequestClass)


def safe_route_reason(value: Any) -> str | None:
    """Return a supported deterministic routing reason, or ``None``."""
    return _enum_value(value, TaskRouteReason)


def safe_provider(value: Any, vocabulary: TaskStoreVocabulary) -> str | None:
    """Return a configured provider identifier, or ``None``."""
    if not isinstance(vocabulary, TaskStoreVocabulary):
        return None
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if not _PROVIDER_RE.fullmatch(text) or _unsafe_value(text):
        return None
    return text if text in vocabulary.providers else None


def safe_model(value: Any, provider: str | None, vocabulary: TaskStoreVocabulary) -> str | None:
    """Return a configured model for ``provider``, or ``None``."""
    if not isinstance(vocabulary, TaskStoreVocabulary) or not isinstance(provider, str):
        return None
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or len(text) > MAX_MODEL_CHARS:
        return None
    if not _MODEL_RE.fullmatch(text) or _unsafe_value(text):
        return None
    return text if (provider, text) in vocabulary.models else None


def safe_node(value: Any, vocabulary: TaskStoreVocabulary) -> str | None:
    """Return a registered graph node identifier, or ``None``."""
    if not isinstance(vocabulary, TaskStoreVocabulary):
        return None
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not _IDENTIFIER_RE.fullmatch(text) or _unsafe_value(text):
        return None
    return text if text in vocabulary.nodes else None


def safe_tool(value: Any, vocabulary: TaskStoreVocabulary) -> str | None:
    """Return a registered tool identifier, or ``None``."""
    if not isinstance(vocabulary, TaskStoreVocabulary):
        return None
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if not _TOOL_RE.fullmatch(text) or _unsafe_value(text):
        return None
    return text if text in vocabulary.tools else None


def safe_owner_instance(value: Any) -> str | None:
    """Return a generated daemon-instance identifier, or ``None``."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text if _OWNER_INSTANCE_RE.fullmatch(text) else None


def safe_failure_code(value: Any, vocabulary: TaskStoreVocabulary) -> str | None:
    """Return a known failure code, or ``None``."""
    if not isinstance(vocabulary, TaskStoreVocabulary):
        return None
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not _FAILURE_CODE_RE.fullmatch(text) or _unsafe_value(text):
        return None
    return text if text in vocabulary.failure_codes else None


def safe_failure_category(value: Any, vocabulary: TaskStoreVocabulary) -> str | None:
    """Return a configured failure category, or ``None``."""
    if not isinstance(vocabulary, TaskStoreVocabulary):
        return None
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str):
        return None
    category = value.strip().lower()
    return category if category in vocabulary.failure_categories else None


def safe_stage(value: Any) -> str | None:
    """Return a supported task stage, or ``None``."""
    from .models import TaskStage

    return _enum_value(value, TaskStage)


def safe_status(value: Any) -> str | None:
    """Return a supported task status, or ``None``."""
    from .models import TaskStatus

    return _enum_value(value, TaskStatus)


def _safe_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _safe_count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if 0 <= value <= MAX_COUNT else None


def _safe_duration(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if 0 <= value <= MAX_DURATION_MS else None


def _validate_detail_value(
    kind: str,
    value: Any,
    provider: str | None,
    vocabulary: TaskStoreVocabulary,
) -> Any:
    if kind == "bool":
        return _safe_bool(value)
    if kind == "count":
        return _safe_count(value)
    if kind == "duration":
        return _safe_duration(value)
    if kind == "stage":
        return safe_stage(value)
    if kind == "status":
        return safe_status(value)
    if kind == "source":
        return safe_source(value)
    if kind == "owner_kind":
        return safe_owner_kind(value)
    if kind == "role":
        return safe_role(value)
    if kind == "approval_state":
        return safe_approval_state(value)
    if kind == "outcome":
        return safe_outcome(value)
    if kind == "severity":
        return safe_severity(value)
    if kind == "reason_code":
        return safe_reason_code(value)
    if kind == "message_kind":
        return safe_message_kind(value)
    if kind == "event_code":
        return safe_event_code(value)
    if kind == "request_class":
        return safe_request_class(value)
    if kind == "route_reason":
        return safe_route_reason(value)
    if kind == "step_id":
        try:
            return safe_step_id(value)
        except (TypeError, ValueError):
            return None
    if kind == "provider":
        return safe_provider(value, vocabulary)
    if kind == "model":
        return safe_model(value, provider, vocabulary)
    if kind == "tool":
        return safe_tool(value, vocabulary)
    if kind == "node":
        return safe_node(value, vocabulary)
    if kind == "failure_code":
        return safe_failure_code(value, vocabulary)
    if kind == "failure_category":
        return safe_failure_category(value, vocabulary)
    return None


def sanitize_detail(
    event_type: str | Enum,
    detail: Mapping[str, Any] | None,
    vocabulary: TaskStoreVocabulary,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Allow-list and bound event detail for one validated event type.

    ``strict`` is used when reading persisted rows: malformed values are rejected
    instead of being silently dropped.
    """
    if isinstance(event_type, Enum):
        event_type = event_type.value
    if not isinstance(event_type, str):
        if strict:
            raise ValueError("event_type must be a supported string")
        return {}
    event_name = event_type.strip().lower()
    policy = _DETAIL_POLICIES.get(event_name)
    if policy is None:
        if strict:
            raise ValueError("event_type has no detail policy")
        return {}
    if detail is None:
        raw_detail: Mapping[str, Any] = {}
    elif isinstance(detail, Mapping):
        raw_detail = detail
    elif strict:
        raise ValueError("detail must be a mapping")
    else:
        raw_detail = {}
    if strict and len(raw_detail) > MAX_DETAIL_ITEMS:
        raise ValueError("detail has too many fields")

    provider = (
        safe_provider(raw_detail.get("provider"), vocabulary) if "provider" in policy else None
    )
    clean: dict[str, Any] = {}
    for key, value in list(raw_detail.items())[:MAX_DETAIL_ITEMS]:
        if not isinstance(key, str):
            if strict:
                raise ValueError("detail keys must be strings")
            continue
        kind = policy.get(key)
        if kind is None:
            if strict:
                raise ValueError(f"detail field {key} is not allowed")
            continue
        if value is None:
            clean[key] = None
            continue
        parsed = _validate_detail_value(kind, value, provider, vocabulary)
        if parsed is None:
            if strict:
                raise ValueError(f"detail field {key} is invalid")
            continue
        if strict and parsed != value:
            raise ValueError(f"detail field {key} is not normalised")
        clean[key] = parsed

    required = _REQUIRED_DETAIL_KEYS.get(event_name, frozenset())
    for key in required:
        if key not in clean or (clean[key] is None and key != "verified"):
            raise ValueError(f"{event_name} detail is missing required field {key}")
    return clean
