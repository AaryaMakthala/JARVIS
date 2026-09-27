"""Explicit LLM failure categories and their routing consequences.

Before this module a provider failure surfaced as one of five exception
classes, and a ``503`` was indistinguishable from a malformed request.  The
composite client could therefore not decide *whether to retry the same
provider, how long to cool it down, or whether to abandon it for this call*.

Every failure is now reduced to exactly one :class:`FailureCategory`, and the
category alone decides the three questions that matter:

* :func:`should_retry_same_provider` — bounded retry on the same provider
  (429 / 5xx / transport), or fail straight through (auth, unknown model,
  malformed request: retrying cannot help).
* :func:`cooldown_seconds` — how long to skip this provider (and this role)
  before trying it again.  ``0`` means "retry on the very next call".
* :func:`is_permanent_for_process` — whether the failure can never succeed
  for the life of the process (a rejected credential, a retired model).  These
  still do **not** blacklist anything permanently: the cooldown is finite and
  every provider is retried on a later interaction, so a rotated key or a
  re-published model recovers by itself.

Classification is deliberately *structural*: it inspects the exception's class
names across its MRO and the message, so it works for the Groq SDK, for
``httpx`` (OpenRouter / NVIDIA), and for the Gemini client without importing
any of them.  The transport modules are optional dependencies.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum

logger = logging.getLogger(__name__)


class FailureCategory(StrEnum):
    """The single, explicit reason an LLM call failed."""

    NONE = "none"
    #: Credential rejected (HTTP 401/403).  Never retried; long cooldown.
    AUTH = "auth"
    #: Provider throttled us (HTTP 429).  Bounded retry, then cool down.
    RATE_LIMIT = "rate_limit"
    #: Provider-side failure (HTTP 5xx).  Bounded retry, then cool down.
    SERVER_ERROR = "server_error"
    #: Socket/TLS/HTTP transport failure after connect.
    CONNECTION = "connection"
    #: Hostname could not be resolved.
    DNS = "dns"
    #: TLS handshake / certificate failure.
    TLS = "tls"
    #: The request exceeded the configured timeout.
    TIMEOUT = "timeout"
    #: The provider rejected the request shape itself (HTTP 400/422).
    MALFORMED_REQUEST = "malformed_request"
    #: The configured model is unknown, retired, or not served (HTTP 404).
    UNSUPPORTED_MODEL = "unsupported_model"
    #: The model cannot do the requested capability (e.g. no JSON schema).
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    #: The model answered, but not in the required schema (twice).
    INVALID_STRUCTURED_OUTPUT = "invalid_structured_output"
    #: The prompt does not fit the model's context window.
    CONTEXT_LENGTH = "context_length"
    #: Anything not recognised — treated as transient but never trusted.
    UNEXPECTED = "unexpected"


#: Categories worth a bounded retry against the *same* provider.  A 429 or a
#: 503 frequently succeeds on the next attempt; a rejected key or a malformed
#: request never will.
RETRYABLE: frozenset[FailureCategory] = frozenset(
    {
        FailureCategory.RATE_LIMIT,
        FailureCategory.SERVER_ERROR,
        FailureCategory.CONNECTION,
        FailureCategory.DNS,
        FailureCategory.TLS,
        FailureCategory.TIMEOUT,
        FailureCategory.CONTEXT_LENGTH,
    }
)

#: Default cooldown before a provider is skipped for the failed role.  Finite
#: on purpose: every provider is retried on a later interaction, so a rotated
#: credential or a re-published model recovers without restarting the daemon.
DEFAULT_COOLDOWN_S: dict[FailureCategory, float] = {
    FailureCategory.RATE_LIMIT: 60.0,
    FailureCategory.SERVER_ERROR: 45.0,
    FailureCategory.CONNECTION: 30.0,
    FailureCategory.DNS: 30.0,
    FailureCategory.TLS: 60.0,
    FailureCategory.TIMEOUT: 30.0,
    FailureCategory.CONTEXT_LENGTH: 300.0,
    FailureCategory.AUTH: 300.0,
    FailureCategory.UNSUPPORTED_MODEL: 300.0,
    FailureCategory.MALFORMED_REQUEST: 120.0,
    FailureCategory.UNSUPPORTED_CAPABILITY: 3600.0,
    FailureCategory.INVALID_STRUCTURED_OUTPUT: 60.0,
    FailureCategory.UNEXPECTED: 30.0,
    FailureCategory.NONE: 0.0,
}

#: Substrings that identify a context-window overflow across providers.
_CONTEXT_MARKERS = (
    "context length",
    "context_length",
    "maximum context",
    "max context",
    "too many tokens",
    "reduce the length",
    "input is too long",
    "prompt is too long",
)

#: Substrings that identify a structured-output / schema rejection.
_SCHEMA_MARKERS = (
    "json_schema",
    "response_format",
    "structured output",
    "does not support",
    "tool call",
    "function calling",
)

_TRANSPORT_MARKERS = {
    "dns": FailureCategory.DNS,
    "getaddrinfo": FailureCategory.DNS,
    "name or service not known": FailureCategory.DNS,
    "nodename nor servname": FailureCategory.DNS,
    "temporary failure in name resolution": FailureCategory.DNS,
    "no address associated with hostname": FailureCategory.DNS,
    "ssl": FailureCategory.TLS,
    "tls": FailureCategory.TLS,
    "certificate": FailureCategory.TLS,
    "certificate_verify": FailureCategory.TLS,
}


def _safe_status(exc: BaseException) -> int | None:
    """The exception's HTTP status, or ``None`` — reading it cannot raise.

    ``status_code`` is sometimes a property that touches a response object which
    may already be closed, so both the attribute reads and the ``int()``
    conversion are guarded.
    """
    raw: object = None
    for attribute in ("status_code", "status"):
        try:
            raw = getattr(exc, attribute, None)
        except Exception as inner:  # noqa: BLE001 - a hostile attribute is "unknown"
            logger.debug("failure status attribute %s unreadable: %s", attribute, inner)
            continue
        if raw is not None:
            break
    if raw is None:
        return None
    try:
        return int(raw)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return None


def _safe_text(exc: BaseException) -> str:
    """``str(exc).lower()`` that cannot raise.

    A provider SDK exception with a broken ``__str__`` (it re-raises, or reads a
    response body that is already closed) would otherwise propagate out of
    :func:`classify` and take down the fallback logic that is supposed to be
    catching the failure in the first place.  ``getattr`` is guarded for the
    same reason: a property-style ``status_code`` can raise too.
    """
    try:
        return (str(exc) or "").lower()
    except Exception as inner:  # noqa: BLE001 - failing to stringify is the signal
        logger.debug("failure message unstringifiable for %s: %s", type(exc).__name__, inner)
        return ""


def classify(exc: BaseException) -> FailureCategory:
    """Reduce ``exc`` to exactly one :class:`FailureCategory`.

    Prefers an explicit ``category`` attribute set by the raising client
    (authoritative), then the exception's own class names across its MRO,
    then message markers.  Never raises.
    """
    explicit = getattr(exc, "category", None)
    if isinstance(explicit, FailureCategory):
        return explicit

    status = _safe_status(exc)

    names = {cls.__name__.lower() for cls in type(exc).__mro__}
    text = _safe_text(exc)

    if status is not None:
        from_status = _from_status(status, text)
        if from_status is not None:
            return from_status

    if "ratelimiterror" in names or "toomanyrequests" in names:
        return FailureCategory.RATE_LIMIT
    if names & {"authenticationerror", "permissiondeniederror"}:
        return FailureCategory.AUTH
    if "notfounderror" in names:
        return FailureCategory.UNSUPPORTED_MODEL
    if "internalservererror" in names or "serviceunavailable" in names:
        return FailureCategory.SERVER_ERROR
    if "apiconnectionerror" in names or "apierror" in names:
        return FailureCategory.CONNECTION
    if "connecttimeout" in names or "readtimeout" in names or "timeouterror" in names:
        return FailureCategory.TIMEOUT
    if "timeouterror" in names or "timeouterror" in text:
        return FailureCategory.TIMEOUT
    if "connectionerror" in names or "networkerror" in names or "transporterror" in names:
        return _transport_category(text)
    if "connecterror" in names:
        return _transport_category(text)

    for marker in _CONTEXT_MARKERS:
        if marker in text:
            return FailureCategory.CONTEXT_LENGTH
    for marker in _SCHEMA_MARKERS:
        if marker in text:
            return FailureCategory.UNSUPPORTED_CAPABILITY
    if "could not be repaired" in text or "unexpected structure" in text:
        return FailureCategory.INVALID_STRUCTURED_OUTPUT
    if "authentication failed" in text or "rejected the api key" in text:
        return FailureCategory.AUTH
    if "is not available" in text or "not found" in text:
        return FailureCategory.UNSUPPORTED_MODEL
    if "rate limit" in text or "http 429" in text:
        return FailureCategory.RATE_LIMIT
    if "timed out" in text or "timeout" in text:
        return FailureCategory.TIMEOUT

    from jarvis.llm.client import (
        LLMAuthError,
        LLMCapabilityError,
        LLMError,
        LLMModelError,
        LLMTransientError,
    )

    if isinstance(exc, LLMCapabilityError):
        return FailureCategory.UNSUPPORTED_CAPABILITY
    if isinstance(exc, LLMAuthError):
        return FailureCategory.AUTH
    if isinstance(exc, LLMModelError):
        return FailureCategory.UNSUPPORTED_MODEL
    if isinstance(exc, LLMTransientError):
        return FailureCategory.SERVER_ERROR
    if isinstance(exc, LLMError):
        return FailureCategory.UNEXPECTED
    return FailureCategory.UNEXPECTED


def _from_status(status: int, text: str) -> FailureCategory | None:
    """Map an HTTP status to a category (``None`` when unrecognised)."""
    if status in (401, 403):
        return FailureCategory.AUTH
    if status == 404:
        return FailureCategory.UNSUPPORTED_MODEL
    if status == 429:
        return FailureCategory.RATE_LIMIT
    if status in (408, 504):
        return FailureCategory.TIMEOUT
    if status == 413:
        return FailureCategory.CONTEXT_LENGTH
    if status in (400, 422):
        for marker in _CONTEXT_MARKERS:
            if marker in text:
                return FailureCategory.CONTEXT_LENGTH
        for marker in _SCHEMA_MARKERS:
            if marker in text:
                return FailureCategory.UNSUPPORTED_CAPABILITY
        return FailureCategory.MALFORMED_REQUEST
    if status >= 500:
        return FailureCategory.SERVER_ERROR
    return None


def _transport_category(text: str) -> FailureCategory:
    """Separate DNS / TLS from a generic connection failure by message."""
    for marker, category in _TRANSPORT_MARKERS.items():
        if marker in text:
            return category
    return FailureCategory.CONNECTION


def should_retry_same_provider(category: FailureCategory) -> bool:
    """Whether a bounded retry against the same provider can help."""
    return category in RETRYABLE


def cooldown_seconds(category: FailureCategory) -> float:
    """How long to skip this provider/role before trying it again."""
    return DEFAULT_COOLDOWN_S.get(category, 30.0)


def is_retryable_exception(exc: BaseException) -> bool:
    """Convenience wrapper: retry the same provider for ``exc``?"""
    return should_retry_same_provider(classify(exc))


@dataclass(frozen=True)
class LLMCallFailure:
    """One provider/role failure, reduced to safe, loggable fields.

    Carries no credential, no prompt, and no raw provider body: only the
    provider label, the model label, the role, an HTTP status, a category, a
    latency, and the client's own already-sanitised message.
    """

    provider: str
    model: str
    role: str
    category: FailureCategory = FailureCategory.UNEXPECTED
    http_status: int | None = None
    latency_ms: int = 0
    message: str = ""
    at: float = field(default_factory=time.monotonic)

    @property
    def retryable(self) -> bool:
        """Whether a bounded retry on the same provider could succeed."""
        return should_retry_same_provider(self.category)

    @property
    def cooldown_s(self) -> float:
        """Seconds to skip this provider/role for."""
        return cooldown_seconds(self.category)

    def label(self) -> str:
        """``provider/model`` for a fallback message."""
        return f"{self.provider}/{self.model}" if self.model else self.provider

    def safe_summary(self) -> str:
        """A one-line, secret-free description for logs and terminal output."""
        status = f" HTTP {self.http_status}" if self.http_status else ""
        return f"{self.label()} [{self.role}]{status} {self.category.value}"


def failure_from_exception(
    exc: BaseException,
    *,
    provider: str,
    model: str,
    role: str,
    latency_ms: int = 0,
) -> LLMCallFailure:
    """Build a :class:`LLMCallFailure` from a raised transport/client error."""
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    try:
        http_status = int(status) if status is not None else None
    except (TypeError, ValueError):
        http_status = None
    return LLMCallFailure(
        provider=provider,
        model=model,
        role=role,
        category=classify(exc),
        http_status=http_status,
        latency_ms=latency_ms,
        message=str(exc),
    )
