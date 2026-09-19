"""JARVIS command line interface.

Commands implemented in Phase 0: ``init`` (writes config + stores API keys in
the OS credential store), ``doctor`` (environment checks) and the ``status`` /
``password`` stubs.  Phase 1 adds ``chat --no-daemon``: an in-process REPL that
runs the LangGraph agent and blocks on rich confirmations. Rendering lives only
here; the logic behind each command is a plain function (``init_config``,
``run_doctor``, ``chat_loop``) so tests can call it without a console.
"""

from __future__ import annotations

import getpass
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import tomli_w
import typer
from langgraph.checkpoint.sqlite import SqliteSaver
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from jarvis import __version__, config, logging_setup
from jarvis import secrets as secret_module
from jarvis.agent import (
    AppContext,
    make_app_context,
    open_sqlite_checkpointer,
    resume_task,
    run_task,
)
from jarvis.llm.client import GroqClient, LLMError
from jarvis.platform_guard import is_64bit, is_python_supported, is_windows
from jarvis.policy import tiers
from jarvis.policy.unlock import UnlockManager

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


def _chat_prompt(text: str) -> str:
    """Prompt helper; separate for monkeypatching in tests."""
    return Prompt.ask(text)


def _chat_password_prompt(text: str) -> str:
    """Prompt helper for the JARVIS password; separate for monkeypatching."""
    return getpass.getpass(text)


def _confirmation_answer(agreed: bool, req: dict[str, Any], typed: str | None) -> dict[str, Any]:
    """Build the resume payload from a chat answer.

    The only keys ever sent are ``approved``, ``action_hash`` and - when a
    typed folder-name confirmation is required - ``typed_confirmation``.
    The password is never part of a resume payload (docs/03 invariant 8).
    """
    answer: dict[str, Any] = {"approved": agreed, "action_hash": req.get("action_hash")}
    if typed is not None:
        answer["typed_confirmation"] = typed
    return answer


def _tier2_unlock_or_refuse(ctx: AppContext, req: dict[str, Any]) -> bool:
    """Gate a Tier-2 answer behind a live unlocked session (password entry)."""
    manager = ctx.unlock
    if manager is None or not hasattr(manager, "has_password"):
        console.print(
            "[red]Tier 2 action needs an unlocked JARVIS session, but no unlock "
            "manager is available in this session - refusing.[/red]"
        )
        return False
    if not manager.has_password():
        console.print(
            "[red]No JARVIS password is set. Run `jarvis password set` first - "
            "refusing the Tier 2 action.[/red]"
        )
        return False
    if manager.is_unlocked():
        return True
    password = _chat_password_prompt("JARVIS password: ")
    if manager.verify(password):
        return True
    console.print(
        "[red]Wrong password or the session is locked out - refusing the Tier 2 action.[/red]"
    )
    return False


def chat_loop(ctx: AppContext, saver: SqliteSaver) -> None:
    """One interactive session over the Phase 1 agent (no daemon)."""
    while True:
        try:
            line = _chat_prompt("[bold cyan]you>[/]")
        except (KeyboardInterrupt, EOFError):
            console.print("[dim]bye[/dim]")
            break
        line = (line or "").strip()
        if not line:
            continue

        outcome = run_task(ctx, saver, line)
        while outcome.confirmation:
            req = outcome.confirmation
            console.print(
                "".join(
                    (
                        "[yellow]approval needed[/yellow] (",
                        tiers.tier_label(req.get("tier")),
                        ")",
                    )
                )
            )
            console.print(req.get("summary") or "(no summary)")
            if req.get("untrusted"):
                console.print(
                    "[red]NOTE: this action was derived from untrusted content "
                    "(web/file text).[/red]"
                )
            answer = _chat_prompt("Approve?")
            agreed = (answer or "").strip().lower() in ("y", "yes")
            typed: str | None = None
            if agreed and req.get("typed_confirmation"):
                typed = _chat_prompt(
                    f"Type this exactly to confirm the delete: {req.get('typed_confirmation')}"
                )
            if agreed and req.get("needs_unlock"):
                agreed = _tier2_unlock_or_refuse(ctx, req)
            outcome = resume_task(
                ctx,
                saver,
                outcome.task_id,
                _confirmation_answer(agreed, req, typed),
            )

        if outcome.error:
            console.print(f"[red]{outcome.error}[/red]")
        elif outcome.final_answer:
            console.print(f"[green]{outcome.final_answer}[/green]")
        else:
            console.print("[dim](no answer)[/dim]")


@app.command()
def chat(
    no_daemon: Annotated[
        bool, typer.Option("--no-daemon", help="Run the agent in-process (Phase 1).")
    ] = False,
) -> None:
    """Talk to JARVIS (interactive REPL)."""
    if not no_daemon:
        console.print(
            "[yellow]daemon mode is not implemented yet (planned for Phase 3); "
            "use `jarvis chat --no-daemon`.[/yellow]"
        )
        raise typer.Exit(code=2)

    settings = config.load_settings()
    store = secret_module.SecretStore()
    groq_key = store.get("groq_api_key")
    if not groq_key:
        console.print("[red]no Groq API key found - run `jarvis init` first.[/red]")
        raise typer.Exit(code=1)
    if not settings.llm.planner_model:
        console.print(
            "[red]llm.planner_model is unset - pick a model and set it in config.toml.[/red]"
        )
        raise typer.Exit(code=1)

    try:
        llm = GroqClient(groq_key, settings)
        unlock_manager = UnlockManager(store, settings=settings)
        ctx = make_app_context(settings, llm=llm, unlock=unlock_manager)
        saver = open_sqlite_checkpointer(str(config.checkpoints_db()))
    except LLMError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    console.print("[dim]JARVIS chat (Ctrl+C to exit)[/dim]")
    chat_loop(ctx, saver)


@password_app.command("set")
def password_set(
    new_password: Annotated[
        str | None,
        typer.Option("--password", help="New JARVIS password (non-interactive).", hide_input=True),
    ] = None,
    confirm_password: Annotated[
        str | None, typer.Option("--confirm-password", help="Repeat the password.", hide_input=True)
    ] = None,
    current_password: Annotated[
        str | None,
        typer.Option(
            "--current-password", help="Current password (if one is set).", hide_input=True
        ),
    ] = None,
    non_interactive: Annotated[
        bool, typer.Option("--non-interactive", help="Never prompt; use flags only.")
    ] = False,
) -> None:
    """Set the JARVIS unlock password (stored as an Argon2id hash)."""
    try:
        message = password_set_command(
            UnlockManager(secret_module.SecretStore(), settings=config.load_settings()),
            password=new_password,
            confirm_password=confirm_password,
            current_password=current_password,
            interactive=not non_interactive,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]{message}[/green]")


@password_app.command("change")
def password_change(
    current_password: Annotated[
        str | None,
        typer.Option("--current-password", help="Current JARVIS password.", hide_input=True),
    ] = None,
    new_password: Annotated[
        str | None, typer.Option("--password", help="New JARVIS password.", hide_input=True)
    ] = None,
    confirm_password: Annotated[
        str | None,
        typer.Option("--confirm-password", help="Repeat the new password.", hide_input=True),
    ] = None,
    non_interactive: Annotated[
        bool, typer.Option("--non-interactive", help="Never prompt; use flags only.")
    ] = False,
) -> None:
    """Change the JARVIS unlock password (requires the current one)."""
    try:
        message = password_change_command(
            UnlockManager(secret_module.SecretStore(), settings=config.load_settings()),
            current_password=current_password,
            new_password=new_password,
            confirm_password=confirm_password,
            interactive=not non_interactive,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]{message}[/green]")


@app.command()
def lock() -> None:
    """Lock the JARVIS session (Tier-2 actions are refused until unlock)."""
    manager = UnlockManager(secret_module.SecretStore(), settings=config.load_settings())
    console.print(f"[green]{lock_command(manager)}[/green]")


@app.command()
def unlock(
    password: Annotated[
        str | None,
        typer.Option("--password", help="JARVIS password (non-interactive).", hide_input=True),
    ] = None,
    non_interactive: Annotated[
        bool, typer.Option("--non-interactive", help="Never prompt; use flags only.")
    ] = False,
) -> None:
    """Unlock the JARVIS session for Tier-2 actions."""
    manager = UnlockManager(secret_module.SecretStore(), settings=config.load_settings())
    try:
        message = unlock_command(manager, password=password, interactive=not non_interactive)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]{message}[/green]")


def _apply_new_password(manager: UnlockManager, new: str | None, confirm: str | None) -> str:
    """Validate + store a new password; returns a success message."""
    if new is None or confirm is None or new != confirm:
        raise ValueError("the two password entries do not match")
    manager.set_password(new)  # raises ValueError when too short
    return "JARVIS password set (Argon2id hash stored in the OS credential store)"


def password_set_command(
    manager: UnlockManager,
    *,
    password: str | None = None,
    confirm_password: str | None = None,
    current_password: str | None = None,
    interactive: bool = True,
) -> str:
    """Plain-function password set/change logic (readable + testable).

    When an existing password is present, the current one must be supplied and
    verified before the hash may be replaced (fail closed: an unlocked-but-casual
    CLI session cannot silently rewrite the password).
    """
    if interactive:
        if manager.has_password():
            current = getpass.getpass("Current JARVIS password: ")
            if not manager.verify(current):
                raise ValueError("current password is wrong (or the session is locked out)")
        password = getpass.getpass("New JARVIS password: ")
        confirm_password = getpass.getpass("Repeat new JARVIS password: ")
        return _apply_new_password(manager, password, confirm_password)
    if not password:
        raise ValueError("a password is required (use --password in non-interactive mode)")
    if manager.has_password() and not (current_password and manager.verify(current_password)):
        raise ValueError("current password is required (and must be correct) to overwrite it")
    return _apply_new_password(manager, password, confirm_password)


def password_change_command(
    manager: UnlockManager,
    *,
    current_password: str | None = None,
    new_password: str | None = None,
    confirm_password: str | None = None,
    interactive: bool = True,
) -> str:
    """Plain-function password-change logic (readable + testable)."""
    if interactive:
        current = getpass.getpass("Current JARVIS password: ")
        if not manager.verify(current):
            raise ValueError("current password is wrong (or the session is locked out)")
        new = getpass.getpass("New JARVIS password: ")
        confirm = getpass.getpass("Repeat new JARVIS password: ")
        return _apply_new_password(manager, new, confirm)
    if not current_password or not new_password:
        raise ValueError("current_password and new_password are required in non-interactive mode")
    if not manager.verify(current_password):
        raise ValueError("current password is wrong (or the session is locked out)")
    return _apply_new_password(manager, new_password, confirm_password)


def lock_command(manager: UnlockManager) -> str:
    """Lock the session; returns a status message."""
    manager.lock()
    return "JARVIS session locked"


def unlock_command(
    manager: UnlockManager,
    *,
    password: str | None = None,
    interactive: bool = True,
) -> str:
    """Unlock the session for Tier-2 actions; returns a status message."""
    if interactive:
        password = getpass.getpass("JARVIS password: ")
    if password is None:
        raise ValueError("a password is required to unlock")
    if manager.verify(password):
        return f"JARVIS session unlocked for {manager.remaining_seconds():.0f}s"
    raise ValueError("wrong password (or the session is locked out)")


def main() -> None:
    """Console entry point (also used by python -m jarvis)."""
    logging_setup.configure_logging(config.logs_dir())
    app()
