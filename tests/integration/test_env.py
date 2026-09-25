"""Environment-override integration checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis import config


def test_env_overrides_flow_into_settings_without_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JARVIS_DATA_DIR", str(tmp_path / "fake"))
    monkeypatch.setenv("JARVIS_AGENT__MAX_STEPS", "7")
    monkeypatch.setenv("JARVIS_LLM__TIMEOUT_SECONDS", "12.5")
    settings = config.load_settings(tmp_path / "missing.toml")
    assert settings.agent.max_steps == 7
    assert settings.llm.timeout_seconds == 12.5


def test_full_init_flow_then_reload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Writes a config via init, then loads it back with file as the source."""
    monkeypatch.setenv("JARVIS_DATA_DIR", str(tmp_path / "data"))
    config_path = config.config_file()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config.DEFAULT_CONFIG_TOML, encoding="utf-8")
    settings = config.load_settings()
    assert settings.llm.provider_order == ["groq", "openrouter", "nvidia", "gemini"]
    assert settings.llm.free_only is True
    assert settings.llm.strict_zero_cost is True
    assert config.log_file().parent.name == "logs"
