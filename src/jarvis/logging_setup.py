"""JSON-lines file logging with a secret-redaction filter.

Every log record emitted through the root logger is serialised to a single JSON
line and passed through :func:`redact` so secrets (API keys, passwords, phone
numbers, registered secret values) never reach the log file. Library code must
use :func:`get_logger`; only the CLI writes to the console (via ``rich``).
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import re
import threading
from datetime import UTC, datetime
from pathlib import Path

from jarvis import config

_LOGGER_NAME = "jarvis"

_redact_lock = threading.Lock()
_registered_secrets: list[str] = []

#: Known API-key prefixes. Prefix-based matching catches keys before their exact
#: value can be registered.
_KEY_PATTERNS = (
    re.compile(r"gsk_[A-Za-z0-9]{16,}"),
    re.compile(r"AIza[A-Za-z0-9_\-]{16,}"),
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"tvly-[A-Za-z0-9]{16,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]{16,}"),
)

#: Sensitive field names whose values are scrubbed whenever they appear in a
#: message as ``name: value`` or ``name=value``.
_SENSITIVE_FIELD = re.compile(
    r"(password|passwd|api[_-]?key|authorization|token|secret|ipc_token)"
    r"\s*[:=]\s*[\"']?[^\"'\s,}\]]+",
    re.IGNORECASE,
)

#: International phone numbers such as ``+91 12345 67890`` (kept last 2 digits).
_PHONE = re.compile(r"\+[0-9][\d\s\-()]{7,20}[0-9]")

_REDACTED = "<redacted>"


def _mask_phone(match: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", match.group(0))
    return f"+{'•' * (len(digits) - 2)}{digits[-2:]}"


def register_redact_value(value: str) -> None:
    """Register an exact secret value to be masked everywhere in logs.

    Safe to call repeatedly; short/trivial values are ignored. Values recorded
    here are held in memory only and are never logged themselves.
    """
    if not value or len(value) < 8:
        return
    with _redact_lock:
        if value not in _registered_secrets:
            _registered_secrets.append(value)


def redact(text: str) -> str:
    """Redact registered secrets, known key prefixes, sensitive fields, and phone numbers."""
    if not text:
        return text
    with _redact_lock:
        secrets = tuple(_registered_secrets)
    for secret in secrets:
        text = text.replace(secret, _REDACTED)
    for pattern in _KEY_PATTERNS:
        text = pattern.sub(_REDACTED, text)
    text = _SENSITIVE_FIELD.sub(lambda m: m.group(1) + "=" + _REDACTED, text)
    return _PHONE.sub(_mask_phone, text)


class JsonLinesFormatter(logging.Formatter):
    """Format a record as a single redacted JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, object] = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)
        return redact(json.dumps(data, ensure_ascii=False, default=str))


def configure_logging(
    log_dir: Path | None = None,
    level: int = logging.INFO,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    """Attach a rotating redacted JSONL file handler to the *root* logger.

    Idempotent per log file: subsequent calls do not add duplicate handlers.
    All loggers (including third-party ones) therefore flow through the
    redaction formatter.
    """
    target = log_dir if log_dir is not None else config.logs_dir()
    target.mkdir(parents=True, exist_ok=True)
    log_path = target / "jarvis.jsonl"

    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        if isinstance(handler, logging.FileHandler) and getattr(
            handler, "baseFilename", None
        ) == str(log_path):
            root.info(
                "logging configured: file=%s level=%s",
                log_path,
                logging.getLevelName(level),
            )
            return root

    handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(JsonLinesFormatter())
    handler.setLevel(level)
    root.addHandler(handler)
    root.info(
        "logging configured: file=%s level=%s handler=%s",
        log_path,
        logging.getLevelName(level),
        type(handler).__name__,
    )
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a child logger of the ``jarvis`` namespace."""
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")
