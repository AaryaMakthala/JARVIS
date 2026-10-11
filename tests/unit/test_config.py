"""Config: defaults, TOML loading, env overrides, and path helpers."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from jarvis import config


@pytest.fixture
def isolated_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("JARVIS_DATA_DIR", str(tmp_path))
    return tmp_path


def test_defaults_when_no_config(isolated_data_dir: Path) -> None:
    settings = config.load_settings()
    assert settings.llm.planner_model == ""
    assert settings.llm.provider_order == ["groq", "openrouter", "nvidia", "gemini"]
    assert settings.llm.free_only is True
    assert settings.llm.strict_zero_cost is True
    assert settings.agent.max_steps == 12
    assert settings.policy.unlock_ttl_seconds == 300
    assert settings.daemon.host == "127.0.0.1"
    assert settings.voice.enabled is False


def test_default_config_toml_parses() -> None:
    data = tomllib.loads(config.DEFAULT_CONFIG_TOML)
    assert data["config_version"] == 1
    llm = data["llm"]
    assert llm["provider_order"] == ["groq", "openrouter", "nvidia", "gemini"]
    assert llm["free_only"] is True
    assert llm["strict_zero_cost"] is True
    assert llm["max_retries"] == 1
    assert llm["models"] == {
        "groq": {
            "planner": "openai/gpt-oss-120b",
            "fast": "openai/gpt-oss-20b",
        },
        "openrouter": {"planner": "openrouter/free", "fast": "openrouter/free"},
        "nvidia": {
            "planner": "nvidia/nemotron-3-super-120b-a12b",
            "fast": "nvidia/nemotron-3.5-lightning-30b-a3b",
        },
        "gemini": {
            "planner": "gemini-3.8-flash",
            "fast": "gemini-3.7-flash",
        },
    }
    assert data["agent"]["max_steps"] == 12
    assert data["policy"]["unlock_ttl_seconds"] == 300


def test_llm_key_slot_and_vision_defaults() -> None:
    """Stage 5a config primitives: key slots, rotation flag, vision role."""
    llm = config.LLMSettings()
    assert llm.key_slots == {}
    assert llm.rotate_on_rate_limit is False
    assert config.ProviderModels().vision == ""
    parsed = tomllib.loads(config.DEFAULT_CONFIG_TOML)
    assert parsed["llm"]["rotate_on_rate_limit"] is False
    # key_slots ships as a documented comment: `keys set --slot 2` prints the
    # exact line the owner adds to config.toml (decision: no auto-edit).
    assert "key_slots" not in parsed["llm"]


def test_key_slots_and_rotation_load_from_toml(tmp_path: Path) -> None:
    toml = tmp_path / "config.toml"
    toml.write_text(
        '[llm]\nkey_slots = { groq = ["groq_api_key", "groq_2_api_key"] }\n'
        "rotate_on_rate_limit = true\n",
        encoding="utf-8",
    )
    settings = config.load_settings(toml)
    assert settings.llm.key_slots == {"groq": ["groq_api_key", "groq_2_api_key"]}
    assert settings.llm.rotate_on_rate_limit is True


def test_vision_model_resolution() -> None:
    """Per-provider ``vision`` role wins, then the legacy ``vision_model``."""
    llm = config.LLMSettings(models={"gemini": config.ProviderModels(vision="gemini-3.8-flash")})
    assert llm.model_for("gemini", "vision") == "gemini-3.8-flash"
    assert llm.model_for("groq", "vision") == ""
    legacy = config.LLMSettings(
        vision_model="legacy-vision",
        models={"gemini": config.ProviderModels(planner="p")},
    )
    assert legacy.model_for("gemini", "vision") == "legacy-vision"


def test_strict_zero_cost_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    toml = tmp_path / "config.toml"
    toml.write_text(config.DEFAULT_CONFIG_TOML, encoding="utf-8")
    monkeypatch.setenv("JARVIS_LLM__STRICT_ZERO_COST", "false")
    settings = config.load_settings(toml)
    assert settings.llm.strict_zero_cost is False
    assert settings.llm.free_only is True


def test_load_from_toml_file(tmp_path: Path) -> None:
    toml = tmp_path / "config.toml"
    toml.write_text(config.DEFAULT_CONFIG_TOML, encoding="utf-8")
    settings = config.load_settings(toml)
    assert settings.llm.timeout_seconds == 30.0
    assert settings.llm.models["groq"].planner == "openai/gpt-oss-120b"
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


def test_voice_enabled_true_loaded_from_toml(tmp_path: Path) -> None:
    toml = tmp_path / "config.toml"
    toml.write_text('[voice]\nenabled = true\nwake_word = "zzz_wake"\n', encoding="utf-8")
    settings = config.load_settings(toml)
    assert settings.voice.enabled is True
    assert settings.voice.wake_word == "zzz_wake"
    assert settings.llm.planner_model == ""


def test_voice_enabled_false_loaded_from_toml(tmp_path: Path) -> None:
    toml = tmp_path / "config.toml"
    toml.write_text("[voice]\nenabled = false\n", encoding="utf-8")
    settings = config.load_settings(toml)
    assert settings.voice.enabled is False


def test_bare_stop_while_busy_defaults_to_true() -> None:
    """D22: a bare "stop" cancels while speaking unless the owner opts out.

    Pinned in both places the default lives, so the shipped ``config.toml`` and
    the model cannot disagree about what a fresh install does.
    """
    assert config.VoiceSettings().bare_stop_while_busy is True
    voice = tomllib.loads(config.DEFAULT_CONFIG_TOML)["voice"]
    assert voice["bare_stop_while_busy"] is True


def test_bare_stop_while_busy_can_be_turned_off(tmp_path: Path) -> None:
    """``false`` restores wake-word-only barge-in while speaking."""
    toml = tmp_path / "config.toml"
    toml.write_text("[voice]\nbare_stop_while_busy = false\n", encoding="utf-8")
    assert config.load_settings(toml).voice.bare_stop_while_busy is False


def test_env_override_wins_over_toml_true_to_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    toml = tmp_path / "config.toml"
    toml.write_text("[voice]\nenabled = true\n", encoding="utf-8")
    monkeypatch.setenv("JARVIS_VOICE__ENABLED", "false")
    settings = config.load_settings(toml)
    assert settings.voice.enabled is False


def test_env_override_wins_over_toml_false_to_true(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    toml = tmp_path / "config.toml"
    toml.write_text("[voice]\nenabled = false\n", encoding="utf-8")
    monkeypatch.setenv("JARVIS_VOICE__ENABLED", "true")
    settings = config.load_settings(toml)
    assert settings.voice.enabled is True


def test_cli_and_daemon_resolve_same_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both `jarvis on` (CLI) and the daemon load via the same loader/--file."""
    monkeypatch.setenv("JARVIS_DATA_DIR", str(tmp_path))
    config_path = config.config_file()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        '[voice]\nenabled = true\nwake_word = "cli_daemon_shared"\n', encoding="utf-8"
    )

    from jarvis.daemon.server import DaemonServer

    class _FakeStore:
        def get(self, name: str) -> str:
            return "t"

    server = DaemonServer(settings=None, store=_FakeStore())
    cli_settings = config.load_settings()
    assert server._settings.voice.enabled is True
    assert server._settings.voice.wake_word == "cli_daemon_shared"
    assert cli_settings.voice.enabled is True
    assert server._settings.voice == cli_settings.voice


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


def test_browser_defaults() -> None:
    """Stage 7 browser scope defaults (D12/I4)."""
    browser = config.BrowserSettings()
    assert browser.allowed_domains == []
    assert browser.task_scope_max_actions == 8
    assert browser.scope_expiry_s == 120.0


def test_browser_default_config_toml_parses() -> None:
    parsed = tomllib.loads(config.DEFAULT_CONFIG_TOML)
    assert parsed["browser"]["allowed_domains"] == []
    assert parsed["browser"]["task_scope_max_actions"] == 8
    assert parsed["browser"]["scope_expiry_s"] == 120.0


def test_browser_allowed_domains_normalised() -> None:
    browser = config.BrowserSettings(
        allowed_domains=["  YouTube.COM.", "spotify.com", "youtube.com", ""]
    )
    assert browser.allowed_domains == ["youtube.com", "spotify.com"]


def test_browser_allowed_domains_rejects_non_host() -> None:
    """A scheme or path is a config error, never silently accepted."""
    with pytest.raises(ValidationError):
        config.BrowserSettings(allowed_domains=["https://youtube.com"])
    with pytest.raises(ValidationError):
        config.BrowserSettings(allowed_domains=["youtube.com/watch"])


def test_browser_profile_dirs_are_separate(isolated_data_dir: Path) -> None:
    """B6: whatsapp and general never share a directory."""
    whatsapp = config.browser_profile_dir("whatsapp")
    general = config.browser_profile_dir("general")
    assert whatsapp != general
    assert whatsapp == isolated_data_dir / "browser_profile" / "whatsapp"
    assert general == isolated_data_dir / "browser_profile" / "general"
    assert config.browser_profile_dir() == general


def test_browser_profile_dir_rejects_unknown(isolated_data_dir: Path) -> None:
    with pytest.raises(ValueError):
        config.browser_profile_dir("rogue")
