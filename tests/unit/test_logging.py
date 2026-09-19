"""Logging redaction and the JSON-lines formatter."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from jarvis import logging_setup


def test_redact_registered_secret_exact_value() -> None:
    logging_setup.register_redact_value("gsk-super_secret_test_key_123456")
    out = logging_setup.redact("the key is gsk-super_secret_test_key_123456 here")
    assert "<redacted>" in out
    assert "super_secret_test_key" not in out


def test_redact_key_prefix_matches_before_registration() -> None:
    out = logging_setup.redact("request api_key=gsk_ABCDEFGHIJKLMNOPQRSTUVWX123456")
    assert "gsk_" not in out
    assert "<redacted>" in out


def test_redact_sensitive_field() -> None:
    out = logging_setup.redact("password=SuperSecret123 README")
    assert "SuperSecret123" not in out
    assert "password=<redacted>" in out


def test_redact_phone_keeps_last_two_digits() -> None:
    out = logging_setup.redact("call +91 98765 43210 right now")
    assert "98765" not in out
    assert "+" in out
    assert out.endswith("10 right now")


def test_redact_ignores_short_or_empty() -> None:
    logging_setup.register_redact_value("short")  # registered but <8 chars -> ignored at register
    assert logging_setup.redact("") == ""
    assert "short" in logging_setup.redact("value is short here")


def _log_record(msg: str, name: str = "jarvis.test") -> logging.LogRecord:
    return logging.LogRecord(
        name=name, level=logging.INFO, pathname=__file__, lineno=1, msg=msg, args=(), exc_info=None
    )


def test_json_lines_formatter_redacts() -> None:
    logging_setup.register_redact_value("gsk_redact_me_please_123456")
    fmt = logging_setup.JsonLinesFormatter()
    line = fmt.format(_log_record("secret gsk_redact_me_please_123456 logged"))
    data = json.loads(line)
    assert data["level"] == "INFO"
    assert data["logger"] == "jarvis.test"
    assert "redact_me_please" not in line
    assert "<redacted>" in data["message"]


def test_configure_logging_writes_redacted_jsonl(tmp_path: Path) -> None:
    logging_setup.register_redact_value("gsk_file_probe_key_12345678")
    logging_setup.configure_logging(log_dir=tmp_path)
    logger = logging_setup.get_logger("probe")
    logger.info("key gsk_file_probe_key_12345678 in file")
    log_path = tmp_path / "jarvis.jsonl"
    assert log_path.is_file()
    content = log_path.read_text(encoding="utf-8")
    assert "file_probe_key" not in content
    assert "<redacted>" in content
    record = json.loads(content.strip().splitlines()[-1])
    assert record["message"].startswith("key")


def test_configure_logging_is_idempotent(tmp_path: Path) -> None:
    logging_setup.configure_logging(log_dir=tmp_path)
    logging_setup.configure_logging(log_dir=tmp_path)
    path = str(tmp_path / "jarvis.jsonl")
    matching = [
        h
        for h in logging.getLogger().handlers
        if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == path
    ]
    assert len(matching) == 1
