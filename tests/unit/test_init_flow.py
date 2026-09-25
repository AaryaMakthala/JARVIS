"""``jarvis init`` wizard: one prompt per provider, no infinite loop.

Regression tests for the Gemini infinite-prompt bug: ``typer.prompt`` used to
re-prompt forever on an empty Enter.  The fix prompts each provider exactly
once via ``getpass.getpass`` and treats an empty answer as "skip".
"""

from __future__ import annotations

from pathlib import Path

from jarvis import cli


class FakeStore:
    def __init__(self) -> None:
        self._values: dict[str, str] = {}

    def get(self, name: str) -> str | None:
        return self._values.get(name)

    def set(self, name: str, value: str) -> None:
        self._values[name] = value

    def has(self, name: str) -> bool:
        return name in self._values


def _noop_prompt(_msg: str) -> str:
    return ""  # Enter on every prompt


def test_enter_skips_every_optional_provider_in_one_round(tmp_path: Path) -> None:
    store = FakeStore()
    prompts: list[str] = []
    messages = cli.init_config(
        store,
        interactive=True,
        prompt_fn=lambda m: (prompts.append(m), "")[1],
        data_dir=tmp_path / "data",
        config_path=tmp_path / "config.toml",
    )
    # exactly one prompt per credential provider - never a second round
    assert [p for p in prompts if "API key" in p] == [
        "Groq API key [optional]: ",
        "OpenRouter API key [optional]: ",
        "NVIDIA API key [optional]: ",
        "Gemini API key [optional]: ",
        "Tavily API key [optional]: ",
    ]
    assert len(prompts) == 5
    joined = "\n".join(messages)
    assert "Groq: no" in joined
    assert "LLM free-only mode: enabled" in joined
    assert "skipped" in joined


def test_provided_key_is_stored_once_without_prompting(tmp_path: Path) -> None:
    store = FakeStore()
    messages = cli.init_config(
        store,
        interactive=False,
        provided_keys={"groq": "  gsk-test-123  "},
        data_dir=tmp_path / "data",
        config_path=tmp_path / "config.toml",
    )
    assert store.get("groq_api_key") == "gsk-test-123"
    joined = "\n".join(messages)
    assert "Groq: yes" in joined
    assert "Gemini: no" in joined


def test_config_written_once_and_reported() -> None:
    store = FakeStore()
    cfg = Path("/tmp/init_cfg_test.toml")
    cfg.unlink(missing_ok=True)
    messages = cli.init_config(
        store,
        interactive=False,
        provided_keys={},
        data_dir=Path("/tmp/init_data_test"),
        config_path=cfg,
    )
    assert cfg.is_file()
    assert any("wrote default config" in m for m in messages)


def test_config_already_exists_is_left_untouched(tmp_path: Path) -> None:
    store = FakeStore()
    cfg = tmp_path / "config.toml"
    cfg.write_text("original", encoding="utf-8")
    messages = cli.init_config(
        store,
        interactive=False,
        provided_keys={},
        data_dir=tmp_path / "data",
        config_path=cfg,
    )
    assert cfg.read_text(encoding="utf-8") == "original"
    assert any("already exists" in m for m in messages)


def test_ipc_token_is_generated_iff_missing(tmp_path: Path) -> None:
    store_no_token = FakeStore()
    messages = cli.init_config(
        store_no_token,
        interactive=False,
        provided_keys={},
        data_dir=tmp_path / "a",
        config_path=tmp_path / "a.toml",
    )
    assert any("ipc_token: generated" in m for m in messages)


def test_key_values_never_written_to_config_or_messages(tmp_path: Path) -> None:
    store = FakeStore()
    secret = "gsk-hunter2-noecho"
    messages = cli.init_config(
        store,
        interactive=False,
        provided_keys={"groq": secret},
        data_dir=tmp_path / "d",
        config_path=tmp_path / "c.toml",
    )
    assert secret not in "\n".join(messages)
    assert secret not in (tmp_path / "c.toml").read_text(encoding="utf-8")
    assert store.get("groq_api_key") == secret
