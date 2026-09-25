"""``jarvis keys`` sub-app: set/paste/clear/status (secret-safe).

Asserts the contract: hidden input is the default, whitespace stripped,
explicit safe receipt feedback followed by a masked preview (first/last 4
characters only), empty values rejected without storage, clipboard import
never echoes the value, ``status`` shows only
``<provider>: configured|not configured``, and unknown providers are rejected.
The complete secret must never appear in any output or log.

Visible entry exists only behind the explicit ``--visible`` opt-in flag; these
tests pin that opt-in down (warning shown, full key in the receipt) *and* that
neither mode ever writes the full key to logs.
"""

from __future__ import annotations

import logging
import typing

import pytest
from typer.testing import CliRunner

from jarvis import cli

CREDENTIAL_PROVIDERS = cli.CREDENTIAL_PROVIDERS
runner = CliRunner()


class FakeStore:
    def __init__(self, **present: str) -> None:
        self._values = dict(present)

    def get(self, name: str) -> str | None:
        return self._values.get(name)

    def set(self, name: str, value: str) -> None:
        self._values[name] = value

    def delete(self, name: str) -> None:
        self._values.pop(name, None)

    def has(self, name: str) -> bool:
        return name in self._values


class FailingStore(FakeStore):
    def set(self, name: str, value: str) -> None:
        raise cli.secret_module.SecretStoreError("simulated safe storage failure")


# ── set ───────────────────────────────────────────────────────────────────


def test_set_stores_trimmed_value() -> None:
    store = FakeStore()
    out = cli.keys_set_command(store, "groq", value="  test-secret  ", interactive=False)
    receipt, preview_line = out.splitlines()
    assert receipt == "✓ API key received and saved securely."
    assert preview_line == "Key: test•••cret"  # 11 chars -> first4 + 3 bullets + last4
    assert store.get("groq_api_key") == "test-secret"
    assert "test-secret" not in out


def test_set_prompt_used_when_interactive() -> None:
    store = FakeStore()
    calls: list[str] = []
    out = cli.keys_set_command(
        store,
        "gemini",
        interactive=True,
        prompt_fn=lambda p: (calls.append(p), "test-interactive-secret")[1],
    )
    assert out.splitlines()[0] == "✓ API key received and saved securely."
    assert store.get("gemini_api_key") == "test-interactive-secret"
    assert calls == ["Gemini API key: "]
    assert "test-interactive-secret" not in out


def test_set_accepts_windows_pasted_input() -> None:
    store = FakeStore()
    out = cli.keys_set_command(
        store,
        "openrouter",
        interactive=True,
        prompt_fn=lambda _prompt: "  test-pasted-secret\r\n",
    )
    assert out.splitlines()[0] == "✓ API key received and saved securely."
    assert out.splitlines()[1].startswith("Key: ")
    assert store.get("openrouter_api_key") == "test-pasted-secret"
    assert "test-pasted-secret" not in out


# ── masked preview (shown only after successful storage) ────────────────

_PREVIEW_SECRET = "gsk_1234567890ABCDEFGHIJKXYZ"


def test_preview_displays_first_four_characters() -> None:
    assert cli.masked_key_preview(_PREVIEW_SECRET).startswith("gsk_")


def test_preview_displays_last_four_characters() -> None:
    assert cli.masked_key_preview(_PREVIEW_SECRET).endswith("KXYZ")


def test_preview_masks_every_middle_character() -> None:
    preview = cli.masked_key_preview(_PREVIEW_SECRET)
    assert preview[4:-4] == "•" * (len(_PREVIEW_SECRET) - 8)
    assert "1234567890" not in preview
    assert "EFGHIJ" not in preview


def test_preview_never_contains_the_complete_secret() -> None:
    assert _PREVIEW_SECRET not in cli.masked_key_preview(_PREVIEW_SECRET)


def test_preview_short_key_is_fully_masked() -> None:
    # 8 characters or fewer would be fully described by first-4 + last-4.
    assert cli.masked_key_preview("sup3rs3k") == "•" * 8
    assert cli.masked_key_preview("abc") == "•••"
    assert "sup3" not in cli.masked_key_preview("sup3rs3k")
    assert "s3k" not in cli.masked_key_preview("sup3rs3k")


def test_preview_shortest_safe_key_keeps_one_masked_character() -> None:
    assert cli.masked_key_preview("abcdefghi") == "abcd•fghi"


def test_set_returns_receipt_followed_by_masked_preview() -> None:
    store = FakeStore()
    out = cli.keys_set_command(store, "groq", value=_PREVIEW_SECRET, interactive=False)
    receipt, preview_line = out.splitlines()
    assert receipt == "✓ API key received and saved securely."
    assert preview_line == "Key: gsk_" + "•" * (len(_PREVIEW_SECRET) - 8) + "KXYZ"
    assert _PREVIEW_SECRET not in out
    assert store.get("groq_api_key") == _PREVIEW_SECRET  # stored exactly, unchanged


def test_set_short_key_stored_with_fully_masked_preview() -> None:
    store = FakeStore()
    out = cli.keys_set_command(store, "groq", value="ab12cd", interactive=False)
    assert out.splitlines()[0] == "✓ API key received and saved securely."
    assert out.splitlines()[1] == "Key: ••••••"
    assert "ab12cd" not in out
    assert store.get("groq_api_key") == "ab12cd"


def test_set_rejects_empty_value() -> None:
    store = FakeStore()
    with pytest.raises(ValueError, match=r"✗ No API key entered\. Nothing was saved\."):
        cli.keys_set_command(store, "groq", value="   ", interactive=False)
    assert store.get("groq_api_key") is None


def test_set_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="unknown provider"):
        cli.keys_set_command(FakeStore(), "mystery", value="x", interactive=False)


def test_keys_set_cli_uses_hidden_prompt_and_never_echoes_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeStore()
    secret = "test-cli-secret"
    prompts: list[str] = []

    def hidden_prompt(prompt: str) -> str:
        prompts.append(prompt)
        return f"  {secret}\r\n"

    monkeypatch.setattr(cli.secret_module, "SecretStore", lambda: store)
    monkeypatch.setattr(cli.getpass, "getpass", hidden_prompt)

    result = runner.invoke(cli.app, ["keys", "set", "groq"])

    assert result.exit_code == 0
    assert prompts == ["Groq API key: "]  # input itself stays hidden (getpass)
    assert "✓ API key received and saved securely." in result.output
    assert f"Key: {cli.masked_key_preview(secret)}" in result.output
    assert secret not in result.output
    assert store.get("groq_api_key") == secret


def test_keys_set_cli_shows_only_first_and_last_four_of_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeStore()
    secret = "gsk_1234567890ABCDEFGHIJKXYZ"
    monkeypatch.setattr(cli.secret_module, "SecretStore", lambda: store)
    monkeypatch.setattr(cli.getpass, "getpass", lambda _prompt: f"  {secret}\r\n")

    result = runner.invoke(cli.app, ["keys", "set", "groq"])

    assert result.exit_code == 0
    assert "✓ API key received and saved securely." in result.output
    assert "Key: gsk_" in result.output
    assert "KXYZ" in result.output
    assert "•" * (len(secret) - 8) in result.output
    assert secret not in result.output
    assert store.get("groq_api_key") == secret  # Windows/CRLF paste stored exactly


def test_keys_set_cli_rejects_empty_hidden_input(monkeypatch: pytest.MonkeyPatch) -> None:
    store = FakeStore()
    monkeypatch.setattr(cli.secret_module, "SecretStore", lambda: store)
    monkeypatch.setattr(cli.getpass, "getpass", lambda _prompt: " \r\n ")

    result = runner.invoke(cli.app, ["keys", "set", "groq"])

    assert result.exit_code == 1
    assert "✗ No API key entered. Nothing was saved." in result.output
    assert "Key:" not in result.output  # no preview without a stored value
    assert store.get("groq_api_key") is None


def test_keys_set_cli_does_not_claim_success_when_storage_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "test-storage-failure-secret"
    monkeypatch.setattr(cli.secret_module, "SecretStore", FailingStore)
    monkeypatch.setattr(cli.getpass, "getpass", lambda _prompt: secret)

    result = runner.invoke(cli.app, ["keys", "set", "groq"])

    assert result.exit_code == 1
    assert "✗ API key received, but secure storage failed." in result.output
    assert "API key received and saved securely" not in result.output
    assert "Key:" not in result.output  # preview is built only after storage succeeds
    assert secret not in result.output
    assert cli.masked_key_preview(secret) not in result.output


# ── paste ────────────────────────────────────────────────────────────────


def test_paste_imports_from_clipboard_and_clears_it() -> None:
    store = FakeStore()
    out = cli.keys_paste_command(
        store,
        "nvidia",
        clipboard_fn=lambda: "  nvapi-123  ",
        clear_clipboard_fn=lambda: None,
    )
    assert out == "saved"
    assert store.get("nvidia_api_key") == "nvapi-123"


def test_paste_rejects_empty_clipboard() -> None:
    with pytest.raises(ValueError, match="clipboard was empty"):
        cli.keys_paste_command(FakeStore(), "groq", clipboard_fn=lambda: " \n ")


def test_paste_does_not_echo_the_key() -> None:
    store = FakeStore()
    out = cli.keys_paste_command(
        store, "groq", clipboard_fn=lambda: "gsk-super-secret", clear_clipboard_fn=lambda: None
    )
    assert out == "saved"  # return value never carries the value


# ── status (env fallback) ─────────────────────────────────────────────────


def test_keys_status_cli_reports_only_configured_state(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "test-status-secret"
    store = FakeStore(groq_api_key=secret)
    for env_name in (
        "GROQ_API_KEY",
        "OPENROUTER_API_KEY",
        "NVIDIA_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "TAVILY_API_KEY",
    ):
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setattr(cli.secret_module, "SecretStore", lambda: store)

    result = runner.invoke(cli.app, ["keys", "status"])

    expected = [
        f"{short}: {'configured' if short == 'groq' else 'not configured'}"
        for short in CREDENTIAL_PROVIDERS
    ]
    assert result.exit_code == 0
    assert result.output.strip().splitlines() == expected
    assert secret not in result.output
    assert "length=" not in result.output


def test_status_lists_every_provider_once(monkeypatch: typing.Any) -> None:
    store = FakeStore(groq_api_key="x")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    lines = cli.keys_status_lines(store)
    assert [l.split(":")[0] for l in lines] == list(CREDENTIAL_PROVIDERS)
    assert "groq: configured" in lines
    for short in ("openrouter", "gemini", "nvidia", "tavily"):
        assert f"{short}: not configured" in lines


def test_status_reflects_environment(monkeypatch: typing.Any) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-from-env")
    lines = cli.keys_status_lines(FakeStore())
    assert "gemini: configured" in lines


def test_status_reflects_google_env_fallback(monkeypatch: typing.Any) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "AIza-google-env")
    lines = cli.keys_status_lines(FakeStore())
    assert "gemini: configured" in lines


def test_status_never_leaks_value_shape(monkeypatch: typing.Any) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    lines = cli.keys_status_lines(FakeStore(groq_api_key="gsk-mega-secret-value-123"))
    assert "gsk-mega-secret-value-123" not in " ".join(lines)
    assert "mega" not in " ".join(lines)


# ── clear ─────────────────────────────────────────────────────────────────


def test_clear_removes_key() -> None:
    store = FakeStore(groq_api_key="x")
    out = cli.keys_clear_command(store, "groq")
    assert out == "cleared"
    assert store.get("groq_api_key") is None


def test_clear_unknown_provider_rejected() -> None:
    with pytest.raises(ValueError, match="unknown provider"):
        cli.keys_clear_command(FakeStore(), "mystery")


# ── visible mode (explicit --visible opt-in; intentionally insecure) ────

_VISIBLE_SECRET = "nvapi-visible-test-key-12345678"
_VISIBLE_WARNING = "WARNING: API key visibility is enabled."


def test_visible_mode_shows_full_key_in_receipt() -> None:
    store = FakeStore()
    out = cli.keys_set_command(
        store,
        "nvidia",
        interactive=True,
        visible=True,
        prompt_fn=lambda _prompt: f"  {_VISIBLE_SECRET}\r\n",
    )
    receipt, key_line = out.splitlines()
    assert receipt == "✓ API key received and saved securely."
    assert key_line == f"Key: {_VISIBLE_SECRET}"  # explicit opt-in shows the whole key
    assert store.get("nvidia_api_key") == _VISIBLE_SECRET


def test_visible_mode_default_reader_is_terminal_input(monkeypatch: pytest.MonkeyPatch) -> None:
    """Visible mode reads via input(): the terminal echoes, edits and accepts paste."""
    store = FakeStore()
    prompts: list[str] = []

    def fake_input(prompt: str) -> str:
        prompts.append(prompt)
        return _VISIBLE_SECRET

    monkeypatch.setattr("builtins.input", fake_input)
    out = cli.keys_set_command(store, "nvidia", interactive=True, visible=True)
    assert prompts == ["NVIDIA API key: "]
    assert store.get("nvidia_api_key") == _VISIBLE_SECRET
    assert out.splitlines()[1] == f"Key: {_VISIBLE_SECRET}"


def test_visible_mode_paste_in_single_read_is_stored_and_shown() -> None:
    store = FakeStore()
    pasted = f"  {_VISIBLE_SECRET}\r\n"  # a Windows paste arrives as one burst
    calls: list[str] = []
    out = cli.keys_set_command(
        store,
        "nvidia",
        interactive=True,
        visible=True,
        prompt_fn=lambda prompt: (calls.append(prompt), pasted)[1],
    )
    assert len(calls) == 1  # one read consumed the whole pasted line
    assert store.get("nvidia_api_key") == _VISIBLE_SECRET
    assert out.splitlines()[1] == f"Key: {_VISIBLE_SECRET}"


def test_visible_mode_applies_backspace_edits() -> None:
    store = FakeStore()
    typed = "nvapi-abc\x08def"  # typed abc, one backspace, then def
    out = cli.keys_set_command(
        store, "nvidia", interactive=True, visible=True, prompt_fn=lambda _p: typed
    )
    assert store.get("nvidia_api_key") == "nvapi-abdef"
    assert out.splitlines()[1] == "Key: nvapi-abdef"
    assert "nvapi-abcdef" not in out  # the unedited line was never accepted


def test_apply_backspace_handles_leading_and_consecutive_backspaces() -> None:
    assert cli._apply_backspace("\x08\x08ab\x08c") == "ac"
    assert cli._apply_backspace("ab\x7f") == "a"
    assert cli._apply_backspace("plain-line") == "plain-line"  # cooked terminals: no-op


def test_visible_mode_empty_input_is_rejected_without_storing() -> None:
    store = FakeStore()
    with pytest.raises(ValueError, match=r"✗ No API key entered\. Nothing was saved\."):
        cli.keys_set_command(
            store, "nvidia", interactive=True, visible=True, prompt_fn=lambda _p: " \r\n "
        )
    assert store.get("nvidia_api_key") is None


def test_keys_set_cli_visible_flag_warns_and_prints_full_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeStore()
    monkeypatch.setattr(cli.secret_module, "SecretStore", lambda: store)

    result = runner.invoke(
        cli.app, ["keys", "set", "nvidia", "--visible"], input=f"{_VISIBLE_SECRET}\n"
    )

    assert result.exit_code == 0
    assert _VISIBLE_WARNING in result.output  # labelled as intentionally insecure
    assert "✓ API key received and saved securely." in result.output
    assert f"Key: {_VISIBLE_SECRET}" in result.output
    assert store.get("nvidia_api_key") == _VISIBLE_SECRET


def test_keys_set_cli_visible_empty_input_exits_without_storing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeStore()
    monkeypatch.setattr(cli.secret_module, "SecretStore", lambda: store)

    result = runner.invoke(cli.app, ["keys", "set", "nvidia", "--visible"], input="\n")

    assert result.exit_code == 1
    assert "✗ No API key entered. Nothing was saved." in result.output
    assert "Key:" not in result.output  # no receipt without a stored value
    assert store.get("nvidia_api_key") is None


def test_keys_set_cli_visible_storage_failure_never_prints_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli.secret_module, "SecretStore", FailingStore)

    result = runner.invoke(
        cli.app, ["keys", "set", "nvidia", "--visible"], input=f"{_VISIBLE_SECRET}\n"
    )

    assert result.exit_code == 1
    assert "✗ API key received, but secure storage failed." in result.output
    assert "API key received and saved securely" not in result.output
    assert "Key:" not in result.output  # receipt is built only after storage succeeds
    assert _VISIBLE_SECRET not in result.output


def test_keys_set_cli_default_is_hidden_and_unwarned(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default invocation must stay on getpass with no warning and no full key."""
    store = FakeStore()
    secret = "test-default-hidden-secret"
    prompts: list[str] = []
    monkeypatch.setattr(cli.secret_module, "SecretStore", lambda: store)
    monkeypatch.setattr(cli.getpass, "getpass", lambda p: (prompts.append(p), secret)[1])

    result = runner.invoke(cli.app, ["keys", "set", "groq"])

    assert result.exit_code == 0
    assert prompts == ["Groq API key: "]  # hidden reader, no --visible involved
    assert _VISIBLE_WARNING not in result.output
    assert secret not in result.output
    assert f"Key: {cli.masked_key_preview(secret)}" in result.output
    assert store.get("groq_api_key") == secret


def test_keys_set_command_defaults_to_hidden_receipt() -> None:
    """Without visible=True the receipt must be masked even when value is given."""
    store = FakeStore()
    out = cli.keys_set_command(store, "groq", value=_VISIBLE_SECRET, interactive=False)
    assert out.splitlines()[1] == f"Key: {cli.masked_key_preview(_VISIBLE_SECRET)}"
    assert _VISIBLE_SECRET not in out


# ── log hygiene: the full key never reaches logs in either mode ──────────


def test_hidden_mode_never_writes_key_to_logs(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FakeStore()
    secret = "test-log-leak-hidden-secret-1234567890"
    monkeypatch.setattr(cli.secret_module, "SecretStore", lambda: store)
    monkeypatch.setattr(cli.getpass, "getpass", lambda _p: f"  {secret}\r\n")

    with caplog.at_level(logging.DEBUG):
        result = runner.invoke(cli.app, ["keys", "set", "groq"])

    assert result.exit_code == 0
    assert secret not in result.output
    assert secret not in caplog.text
    assert secret not in caplog.text.lower()


def test_visible_mode_shows_key_only_in_receipt_never_in_logs(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FakeStore()
    secret = "test-log-leak-visible-secret-1234567890"
    monkeypatch.setattr(cli.secret_module, "SecretStore", lambda: store)

    with caplog.at_level(logging.DEBUG):
        result = runner.invoke(cli.app, ["keys", "set", "nvidia", "--visible"], input=f"{secret}\n")

    assert result.exit_code == 0
    assert f"Key: {secret}" in result.output  # explicit --visible session only
    assert secret not in caplog.text
