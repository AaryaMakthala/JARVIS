"""Config: defaults, TOML loading, env overrides, and path helpers."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

import pytest

from jarvis import config


@pytest.fixture
def isolated_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("JARVIS_DATA_DIR", str(tmp_path))
    return tmp_path


def test_defaults_when_no_config(isolated_data_dir: Path) -> None:
    settings = config.load_settings()
    assert settings.llm.planner_model == ""
    assert settings.llm.provider_order == ["groq", "gemini"]
    assert settings.agent.max_steps == 12
    assert settings.policy.unlock_ttl_seconds == 300
    assert settings.daemon.host == "127.0.0.1"
    assert settings.voice.enabled is False


def test_default_config_toml_parses() -> None:
    data = tomllib.loads(config.DEFAULT_CONFIG_TOML)
    assert data["config_version"] == 1
    llm = data["llm"]
    assert llm["planner_model"] == ""
    assert llm["max_retries"] == 3
    assert data["agent"]["max_steps"] == 12
    assert data["policy"]["unlock_ttl_seconds"] == 300


def test_load_from_toml_file(tmp_path: Path) -> None:
    toml = tmp_path / "config.toml"
    toml.write_text(config.DEFAULT_CONFIG_TOML, encoding="utf-8")
    settings = config.load_settings(toml)
    assert settings.llm.timeout_seconds == 30.0
    assert settings.agent.max_steps == 12


def test_env_overrides_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    toml = tmp_path / "config.toml"
    toml.write_text(config.DEFAULT_CONFIG_TOML, encoding="utf-8")
    monkeypatch.setenv("JARVIS_LLM__PLANNER_MODEL", "groq/llama-3.3-70b-versatile")
    settings = config.load_settings(toml)
    assert settings.llm.planner_model == "groq/llama-3.3-70b-versatile"
    assert settings.llm.fast_model == ""


def test_missing_toml_returns_defaults(tmp_path: Path) -> None:
    settings = config.load_settings(tmp_path / "nope.toml")
    assert settings.llm.planner_model == ""


def test_user_data_dir_override(isolated_data_dir: Path) -> None:
    assert config.user_data_dir() == isolated_data_dir
    assert config.user_data_dir("a", "b") == isolated_data_dir / "a" / "b"


def test_path_helpers_live_under_data_dir(isolated_data_dir: Path) -> None:
    assert config.config_file() == isolated_data_dir / "config.toml"
    assert config.log_file() == isolated_data_dir / "logs" / "jarvis.jsonl"
    assert config.memory_db() == isolated_data_dir / "memory.db"
    assert config.checkpoints_db() == isolated_data_dir / "checkpoints.db"
    assert config.undo_log() == isolated_data_dir / "undo_log.jsonl"
    assert config.contacts_file() == isolated_data_dir / "contacts.json"
    assert config.reports_dir() == isolated_data_dir / "reports"
    assert config.daemon_runtime_file() == isolated_data_dir / "daemon.json"


def test_user_data_dir_unset_uses_platform_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JARVIS_DATA_DIR", raising=False)
    assert os.environ.get("JARVIS_DATA_DIR") is None
    assert config.user_data_dir().is_absolute()
