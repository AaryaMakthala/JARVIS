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


def test_password_set_stub_reports_phase_2() -> None:
    result = runner.invoke(cli.app, ["password", "set"])
    assert result.exit_code == 2
    assert "Phase 2" in result.output


def test_password_change_stub_reports_phase_2() -> None:
    result = runner.invoke(cli.app, ["password", "change"])
    assert result.exit_code == 2
    assert "Phase 2" in result.output


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
