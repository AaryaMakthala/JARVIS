"""JARVIS command line interface.

Commands implemented in Phase 0: ``init`` (writes config + stores API keys in
the OS credential store), ``doctor`` (environment checks), ``status`` and
``password`` are stubs until later phases. Rendering lives only here; the
logic behind each command is a plain function in :func:`init_config` and
:func:`run_doctor` so tests can call it without a console.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import tomli_w
import typer
from rich.console import Console
from rich.table import Table

from jarvis import __version__, config, logging_setup
from jarvis import secrets as secret_module
from jarvis.llm.client import GroqClient, LLMError
from jarvis.platform_guard import is_64bit, is_python_supported, is_windows

app = typer.Typer(
    name="jarvis",
    help="JARVIS - your safe Windows desktop AI agent.",
    add_completion=False,
    no_args_is_help=True,
)
password_app = typer.Typer(help="Unlock-password management (Phase 2).", no_args_is_help=True)
app.add_typer(password_app, name="password")

console = Console()
logger = logging.getLogger(__name__)

_CHECK_STATUS = ("PASS", "WARN", "FAIL")


@dataclass(frozen=True)
class Check:
    """Single doctor check result."""

    name: str
    status: str  # one of _CHECK_STATUS
    detail: str


def _check(name: str, status: str, detail: str) -> Check:
    if status not in _CHECK_STATUS:
        raise ValueError(f"unknown check status {status!r}")
    return Check(name, status, detail)


def init_config(
    store: secret_module.SecretStore,
    *,
    provided_keys: dict[str, str] | None = None,
    interactive: bool = True,
    data_dir: Path | None = None,
    config_path: Path | None = None,
) -> list[str]:
    """Create the data directory, write ``config.toml``, store API keys.

    Returns a list of human-readable status lines. Never prints secrets.
    """
    messages: list[str] = []
    data_dir = data_dir or config.user_data_dir()
    config_path = config_path or config.config_file()
    store = store or secret_module.SecretStore()

    data_dir.mkdir(parents=True, exist_ok=True)
    if config_path.is_file():
        messages.append(f"config.toml already exists - left untouched: {config_path}")
    else:
        import tomllib

        with config_path.open("wb") as fh:
            tomli_w.dump(tomllib.loads(config.DEFAULT_CONFIG_TOML), fh)
        messages.append(f"wrote default config: {config_path}")

    provided = provided_keys or {}
    for short, secret_name in (
        ("groq", "groq_api_key"),
        ("gemini", "gemini_api_key"),
        ("tavily", "tavily_api_key"),
    ):
        if provided.get(short):
            store.set(secret_name, provided[short])
            messages.append(f"{short}: API key stored in the OS credential store")
        elif interactive:
            try:
                value = typer.prompt(f"{secret_module.SECRET_NAMES[secret_name]}", hide_input=True)
            except (typer.Abort, typer.BadParameter):
                raise typer.Exit(code=1)
            if value:
                store.set(secret_name, value)
                messages.append(f"{short}: API key stored in the OS credential store")
        else:
            messages.append(f"{short}: key not provided (skipped in non-interactive mode)")

    token = store.get("ipc_token")
    if not token:
        import secrets as stdlib_secrets

        store.set("ipc_token", stdlib_secrets.token_urlsafe(32))
        messages.append("ipc_token: generated and stored (daemon auth, Phase 3)")

    return messages


def _live_groq_check(settings: config.Settings, store: secret_module.SecretStore) -> Check:
    key = None
    try:
        key = store.get("groq_api_key")
    except secret_module.SecretStoreError as exc:
        return _check("live_groq", "FAIL", str(exc))
    if not key:
        return _check("live_groq", "FAIL", "groq_api_key is missing - run `jarvis init`")
    if not settings.llm.fast_model:
        return _check("live_groq", "WARN", "fast_model not configured - set it in config.toml")
    try:
        client = GroqClient(key, settings)
        client.text(system="You are a connectivity probe.", user="Reply with the single word: ok")
    except LLMError as exc:
        return _check("live_groq", "FAIL", str(exc))
    return _check("live_groq", "PASS", "Groq API reachable")


def run_doctor(
    *,
    settings: config.Settings | None = None,
    store: secret_module.SecretStore | None = None,
    live: bool = False,
) -> list[Check]:
    """Produce environment checks without touching the console."""
    settings = settings or config.load_settings()
    store = store or secret_module.SecretStore()
    checks: list[Check] = []

    if is_windows():
        checks.append(_check("platform", "PASS", "Windows"))
    else:
        checks.append(_check("platform", "WARN", "v1 targets Windows 10/11"))

    if is_python_supported():
        checks.append(_check("python", "PASS", "Python 3.11 or 3.12"))
    else:
        checks.append(_check("python", "FAIL", "need Python 3.11 or 3.12 (64-bit)"))

    checks.append(_check("64bit", "PASS" if is_64bit() else "FAIL", "64-bit interpreter"))

    config_toml = config.config_file()
    if config_toml.is_file():
        checks.append(_check("config", "PASS", f"found {config_toml}"))
    else:
        checks.append(_check("config", "WARN", "no config.toml - run `jarvis init`"))

    missing_models = [
        role for role in ("planner", "fast", "vision") if not getattr(settings.llm, f"{role}_model")
    ]
    if missing_models:
        checks.append(
            _check(
                "models",
                "WARN",
                "unset model role(s): "
                + ", ".join(missing_models)
                + " (choose from provider docs, set in config.toml)",
            )
        )
    else:
        checks.append(_check("models", "PASS", "planner/fast/vision models are configured"))

    backend = store.check_store_access()
    if backend:
        checks.append(_check("keyring", "PASS", f"credential store reachable ({backend})"))
    else:
        checks.append(
            _check("keyring", "FAIL", "credential store unreachable - secrets cannot be protected")
        )

    for name, status_if_missing in (
        ("groq_api_key", "FAIL"),
        ("gemini_api_key", "WARN"),
        ("tavily_api_key", "WARN"),
    ):
        if store.has(name):
            checks.append(_check(name, "PASS", "present"))
        else:
            checks.append(
                _check(name, status_if_missing, f"{name} missing - provide it with `jarvis init`")
            )

    if live:
        checks.append(_live_groq_check(settings, store))
    return checks


def _print_version(value: bool) -> bool:
    """Eager ``--version`` callback: prints and exits before command resolution."""
    if value:
        console.print(f"jarvis {__version__}")
        raise typer.Exit()
    return value


@app.callback()
def _main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show the jarvis version.",
            is_eager=True,
            callback=_print_version,
        ),
    ] = False,
) -> None:
    pass  # option is handled by the eager callback above


@app.command()
def init(
    groq_api_key: Annotated[
        str | None, typer.Option("--groq-api-key", help="Groq API key (omit to be prompted).")
    ] = None,
    gemini_api_key: Annotated[
        str | None, typer.Option("--gemini-api-key", help="Gemini API key (optional).")
    ] = None,
    tavily_api_key: Annotated[
        str | None, typer.Option("--tavily-api-key", help="Tavily API key (optional).")
    ] = None,
    non_interactive: Annotated[
        bool, typer.Option("--non-interactive", help="Never prompt; use flags/env only.")
    ] = False,
) -> None:
    """Initialise JARVIS: config + store API keys in the OS credential store."""
    provided: dict[str, str] = {}
    if groq_api_key:
        provided["groq"] = groq_api_key
    if gemini_api_key:
        provided["gemini"] = gemini_api_key
    if tavily_api_key:
        provided["tavily"] = tavily_api_key
    for line in init_config(
        secret_module.SecretStore(),
        provided_keys=provided,
        interactive=not non_interactive,
    ):
        console.print(line)


@app.command()
def doctor(
    live: Annotated[
        bool, typer.Option("--live", help="Also make a real Groq API call with a configured key.")
    ] = False,
) -> None:
    """Verify the JARVIS environment."""
    checks = run_doctor(live=live)
    table = Table(show_header=False, box=None)
    for check in checks:
        table.add_row(check.status, check.name, check.detail)
    console.print(table)
    failures = [c for c in checks if c.status == "FAIL"]
    if failures:
        raise typer.Exit(code=1)


@app.command()
def status() -> None:
    """Show daemon status (Phase 3)."""
    console.print("[yellow]daemon is not implemented yet (planned for Phase 3)[/yellow]")


@password_app.command("set")
def password_set() -> None:
    """Set the JARVIS unlock password (Phase 2)."""
    console.print(
        "[yellow]password management is not implemented yet (planned for Phase 2)[/yellow]"
    )
    raise typer.Exit(code=2)


@password_app.command("change")
def password_change() -> None:
    """Change the JARVIS unlock password (Phase 2)."""
    console.print(
        "[yellow]password management is not implemented yet (planned for Phase 2)[/yellow]"
    )
    raise typer.Exit(code=2)


def main() -> None:
    """Console entry point (also used by python -m jarvis)."""
    logging_setup.configure_logging(config.logs_dir())
    app()
