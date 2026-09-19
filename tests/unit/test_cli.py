"""CLI: init wizard, doctor output, stubs, and --version."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from jarvis import __version__, cli


class FakeStore:
    """In-memory SecretStore used by unit tests."""

    def __init__(self, **present: str) -> None:
        self._values = dict(present)
        self._backend = "FakeBackend"

    def get(self, name: str) -> str | None:
        return self._values.get(name)

    def set(self, name: str, value: str) -> None:
        self._values[name] = value

    def delete(self, name: str) -> None:
        self._values.pop(name, None)

    def has(self, name: str) -> bool:
        return name in self._values

    def check_store_access(self) -> str | None:
        return self._backend


runner = CliRunner()


def test_version_flag() -> None:
    result = runner.invoke(cli.app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_no_args_shows_help() -> None:
    result = runner.invoke(cli.app, [])
    assert "init" in result.output


def test_status_stub() -> None:
    result = runner.invoke(cli.app, ["status"])
    assert result.exit_code == 0
    assert "Phase 3" in result.output


def test_password_set_cli_non_interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    store = FakeStore()
    monkeypatch.setattr(cli.secret_module, "SecretStore", lambda: store)
    monkeypatch.setattr(cli.config, "load_settings", lambda: cli.config.Settings())
    result = runner.invoke(
        cli.app,
        [
            "password",
            "set",
            "--password",
            "s3cure-pass-9",
            "--confirm-password",
            "s3cure-pass-9",
            "--non-interactive",
        ],
    )
    assert result.exit_code == 0
    assert "password set" in result.output.lower()
    assert "s3cure-pass-9" not in result.output  # never echoed
    stored = store.get("password_hash")
    assert stored and stored.startswith("$argon2id$")


def test_password_set_cli_requires_current_when_hash_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeStore()
    manager = cli.UnlockManager(store)
    manager.set_password("old-pass-1234")
    monkeypatch.setattr(cli.secret_module, "SecretStore", lambda: store)
    monkeypatch.setattr(cli.config, "load_settings", lambda: cli.config.Settings())
    result = runner.invoke(
        cli.app,
        [
            "password",
            "set",
            "--password",
            "new-pass-5678",
            "--confirm-password",
            "new-pass-5678",
            "--non-interactive",
        ],
    )
    assert result.exit_code == 1
    assert "current password" in result.output.lower()
    assert manager.verify("old-pass-1234") is True  # old hash untouched


def test_password_change_cli_non_interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    store = FakeStore()
    manager = cli.UnlockManager(store)
    manager.set_password("old-pass-1234")
    monkeypatch.setattr(cli.secret_module, "SecretStore", lambda: store)
    monkeypatch.setattr(cli.config, "load_settings", lambda: cli.config.Settings())
    result = runner.invoke(
        cli.app,
        [
            "password",
            "change",
            "--current-password",
            "old-pass-1234",
            "--password",
            "new-pass-5678",
            "--confirm-password",
            "new-pass-5678",
            "--non-interactive",
        ],
    )
    assert result.exit_code == 0
    assert manager.verify("new-pass-5678") is True
    assert manager.verify("old-pass-1234") is False


def test_lock_and_unlock_command_contract() -> None:
    manager = cli.UnlockManager(FakeStore())
    manager.set_password("long-pass-9000")
    assert cli.lock_command(manager) == "JARVIS session locked"
    assert manager.is_unlocked() is False
    message = cli.unlock_command(manager, password="long-pass-9000", interactive=False)
    assert message.startswith("JARVIS session unlocked")
    assert manager.is_unlocked() is True
    with pytest.raises(ValueError):
        cli.unlock_command(manager, password="wrong", interactive=False)


def test_confirmation_answer_never_carries_a_password() -> None:
    answer = cli._confirmation_answer(True, {"action_hash": "h" * 64, "tier": 2}, "Reports")
    assert answer == {
        "approved": True,
        "action_hash": "h" * 64,
        "typed_confirmation": "Reports",
    }
    assert "password" not in answer
    assert set(cli._confirmation_answer(True, {"action_hash": "x"}, None)) == {
        "approved",
        "action_hash",
    }


def test_doctor_full_pass_directory() -> None:
    store = FakeStore(
        groq_api_key="gsk-key-1234", gemini_api_key="AIza-key", tavily_api_key="tvly-key"
    )
    checks = {c.name: c for c in cli.run_doctor(settings=cli.config.load_settings(), store=store)}
    assert checks["keyring"].status == "PASS"
    assert checks["groq_api_key"].status == "PASS"
    assert checks["gemini_api_key"].status == "PASS"
    assert checks["platform"].status in ("PASS", "WARN")


def test_doctor_reports_missing_key_and_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("JARVIS_DATA_DIR", str(tmp_path))
    store = FakeStore()
    checks = {c.name: c for c in cli.run_doctor(settings=cli.config.load_settings(), store=store)}
    assert checks["groq_api_key"].status == "FAIL"
    assert checks["gemini_api_key"].status == "WARN"
    assert checks["config"].status == "WARN"
    assert checks["models"].status == "WARN"


def test_doctor_pass_with_configured_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("JARVIS_DATA_DIR", str(tmp_path))
    settings = cli.config.load_settings()
    settings.llm.planner_model = "p"
    settings.llm.fast_model = "f"
    settings.llm.vision_model = "v"
    store = FakeStore(groq_api_key="gsk-key-1234")
    checks = {c.name: c for c in cli.run_doctor(settings=settings, store=store)}
    assert checks["models"].status == "PASS"


def test_doctor_live_requires_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("JARVIS_DATA_DIR", str(tmp_path))
    checks = {
        c.name: c
        for c in cli.run_doctor(settings=cli.config.load_settings(), store=FakeStore(), live=True)
    }
    assert checks["live_groq"].status == "FAIL"


def test_init_config_writes_config_and_stores_key(tmp_path: Path) -> None:
    store = FakeStore()
    data_dir = tmp_path / "data"
    config_path = tmp_path / "config.toml"
    messages = cli.init_config(
        store,
        provided_keys={"groq": "gsk-test-123"},
        interactive=False,
        data_dir=data_dir,
        config_path=config_path,
    )
    assert data_dir.is_dir()
    assert config_path.is_file()
    parsed = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert parsed["llm"]["planner_model"] == ""
    assert store.get("groq_api_key") == "gsk-test-123"
    assert "groq" in " ".join(messages)


def test_init_config_non_interactive_skips_missing_keys(tmp_path: Path) -> None:
    store = FakeStore()
    messages = cli.init_config(
        store,
        interactive=False,
        data_dir=tmp_path / "data",
        config_path=tmp_path / "config.toml",
    )
    assert store.has("gemini_api_key") is False
    assert "skipped" in " ".join(messages)


def test_init_config_generates_ipc_token(tmp_path: Path) -> None:
    store = FakeStore()
    cli.init_config(
        store, interactive=False, data_dir=tmp_path / "data", config_path=tmp_path / "config.toml"
    )
    token = store.get("ipc_token")
    assert token and len(token) >= 32


def test_init_config_keeps_existing_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("# custom\n", encoding="utf-8")
    messages = cli.init_config(
        FakeStore(),
        interactive=False,
        data_dir=tmp_path / "data",
        config_path=config_path,
    )
    assert any("left untouched" in m for m in messages)
    assert "# custom" in config_path.read_text(encoding="utf-8")
