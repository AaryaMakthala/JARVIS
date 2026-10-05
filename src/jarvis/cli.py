"""JARVIS command line interface.

Commands implemented in Phase 0: ``init`` (writes config + stores API keys in
the OS credential store), ``doctor`` (environment checks) and the ``status`` /
``password`` stubs.  Phase 1 adds ``chat --no-daemon``: an in-process REPL that
runs the LangGraph agent and blocks on rich confirmations.  Phase 3 adds the
daemon server, IPC client, ``jarvis daemon --foreground``, ``jarvis stop``,
``jarvis autostart``, and makes ``jarvis chat`` (without ``--no-daemon``)
connect to a running daemon over IPC.

Rendering lives only here; the logic behind each command is a plain function
(``init_config``, ``run_doctor``, ``chat_loop``) so tests can call it without
a console.
"""

from __future__ import annotations

import getpass
import logging
import traceback
from contextlib import contextmanager
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
from jarvis.checks import Check, _check
from jarvis.llm import models as llm_models
from jarvis.llm.client import (
    LLMAuthError,
    LLMCapabilityError,
    LLMError,
    LLMModelError,
    LLMTransientError,
)
from jarvis.llm.provider import (
    build_llm_client,
    build_provider_client,
    provider_has_credential,
)
from jarvis.memory import open_memory
from jarvis.ml.risk import build_classifier
from jarvis.platform_guard import is_64bit, is_python_supported, is_windows
from jarvis.policy import tiers
from jarvis.policy.unlock import UnlockManager
from jarvis.secrets import PROVIDER_SECRETS

app = typer.Typer(
    name="jarvis",
    help="JARVIS - your safe Windows desktop AI agent.",
    add_completion=False,
    no_args_is_help=True,
)
password_app = typer.Typer(help="Unlock-password management.", no_args_is_help=True)
autostart_app = typer.Typer(help="Daemon autostart management.", no_args_is_help=True)
voice_app = typer.Typer(help="Voice diagnostics and control.", no_args_is_help=True)
app.add_typer(password_app, name="password")
app.add_typer(autostart_app, name="autostart")
app.add_typer(voice_app, name="voice")

console = Console()
logger = logging.getLogger(__name__)


CREDENTIAL_PROVIDERS = ("groq", "openrouter", "nvidia", "gemini", "tavily")


def _provider_label(short: str) -> str:
    return llm_models.PROVIDER_LABELS.get(short, short.capitalize())


def _validate_provider(short: str) -> str:
    """Normalise a provider argument; raises ValueError for unknown names."""
    if short not in PROVIDER_SECRETS:
        raise ValueError(
            f"unknown provider {short!r}; expected one of: {', '.join(PROVIDER_SECRETS)}"
        )
    return short


def init_config(
    store: secret_module.SecretStore,
    *,
    provided_keys: dict[str, str] | None = None,
    interactive: bool = True,
    data_dir: Path | None = None,
    config_path: Path | None = None,
    prompt_fn: Any = None,
) -> list[str]:
    """Create the data directory, write ``config.toml``, store provider keys.

    Each provider is prompted exactly once; pressing Enter (an empty value)
    skips that optional provider, so there is never an infinite prompt loop.
    If ``prompt_fn`` is given it is used instead of ``getpass.getpass``
    (tests / non-console callers).  Returns a list of human-readable status
    lines.  Never prints secrets.
    """
    messages: list[str] = []
    data_dir = data_dir or config.user_data_dir()
    config_path = config_path or config.config_file()
    store = store or secret_module.SecretStore()
    prompt_fn = prompt_fn or getpass.getpass

    data_dir.mkdir(parents=True, exist_ok=True)
    if config_path.is_file():
        messages.append(f"config.toml already exists - left untouched: {config_path}")
    else:
        import tomllib

        with config_path.open("wb") as fh:
            tomli_w.dump(tomllib.loads(config.DEFAULT_CONFIG_TOML), fh)
        messages.append(f"wrote default config: {config_path}")

    provided = {k: (v or "").strip() for k, v in (provided_keys or {}).items()}
    configured: dict[str, bool] = {}
    for short in CREDENTIAL_PROVIDERS:
        secret_name = PROVIDER_SECRETS[short]
        label = _provider_label(short)
        value = provided.get(short)
        if not value and interactive:
            try:
                raw = prompt_fn(f"{label} API key [optional]: ")
            except Exception:  # noqa: BLE001 - EOF/abort means "skip all"
                value = ""
            else:
                value = (raw or "").strip()
        if value:
            store.set(secret_name, value)
            configured[short] = True
            messages.append(f"{short}: API key stored in the OS credential store")
        elif interactive:
            configured[short] = False
            messages.append(f"{short}: key not provided (skipped)")
        else:
            configured[short] = False
            messages.append(f"{short}: key not provided (skipped in non-interactive mode)")

    token = store.get("ipc_token")
    if not token:
        import secrets as stdlib_secrets

        store.set("ipc_token", stdlib_secrets.token_urlsafe(32))
        messages.append("ipc_token: generated and stored (daemon auth)")

    messages.append("Configured providers:")
    for short in CREDENTIAL_PROVIDERS:
        messages.append(f"{_provider_label(short)}: {'yes' if configured[short] else 'no'}")
    messages.append("LLM free-only mode: enabled")
    messages.append("LLM strict zero-cost mode: enabled")
    messages.append("Groq and Gemini free-tier keys are blocked while strict mode is enabled")
    return messages


# ── keys (credential management) ─────────────────────────────────────────

_KEYS_SET_SAVED_MESSAGE = "✓ API key received and saved securely."
_KEYS_SET_EMPTY_MESSAGE = "✗ No API key entered. Nothing was saved."
_KEYS_SET_PREVIEW_PREFIX = "Key: "
#: Shown by `jarvis keys set --visible` *before* prompting. Visible key entry
#: is an intentional, explicitly opted-in insecurity - never a default.
_KEYS_SET_VISIBLE_WARNING = "WARNING: API key visibility is enabled."
#: Bullet character used to mask hidden parts of a secret in previews.
_MASK_GLYPH = "•"
#: Number of secret characters revealed at each end of a masked preview.
_PREVIEW_EDGE = 4
#: Shortest key that can be previewed safely: 4 revealed + 4 revealed + >=1 masked.
_PREVIEW_MIN_LENGTH = 9


def masked_key_preview(value: str) -> str:
    """Return a print-safe preview of a secret: first 4 + bullets + last 4.

    Only ``_PREVIEW_EDGE`` characters are revealed at each end; everything
    between them becomes one ``_MASK_GLYPH`` per hidden character, so the key's
    length - never its contents - stays visible.  Keys shorter than
    ``_PREVIEW_MIN_LENGTH`` characters would expose a meaningful first/last
    combination (first 4 + last 4 would cover nearly the whole value), so they
    are shown fully masked instead.

    The returned string never contains the complete secret, so it is safe to
    print, return in a message, or include in a test assertion.
    """
    if len(value) < _PREVIEW_MIN_LENGTH:
        return _MASK_GLYPH * len(value)
    middle = _MASK_GLYPH * (len(value) - 2 * _PREVIEW_EDGE)
    return f"{value[:_PREVIEW_EDGE]}{middle}{value[-_PREVIEW_EDGE:]}"


def _clipboard_paste() -> str:
    """Read the clipboard (Windows-safe via pyperclip). Never logged."""
    import pyperclip

    return pyperclip.paste() or ""


def _clipboard_clear() -> None:
    """Best-effort clipboard clear after a successful key import.

    Documented behaviour: ``pyperclip.copy("")`` empties the pastable
    clipboard, but Windows clipboard history may still retain the value -
    users should clear history manually if that matters to them.
    """
    import pyperclip

    try:
        pyperclip.copy("")
    except Exception as exc:  # noqa: BLE001 - clipboard APIs differ per OS
        logger.warning("could not clear the clipboard: %s", exc)


def visible_input(prompt: str) -> str:
    """Read one API-key line *with* echo (explicit ``--visible`` mode only).

    Echoing, pasting and backspace editing are delegated to the terminal's
    own line editor through ``input()``, which behaves correctly in Windows
    PowerShell as well as cmd: characters appear while typing, pasted text is
    shown normally, and backspace edits the line before Enter.  The raw line
    (minus the newline) is returned; nothing here is ever logged.
    """
    return input(prompt)


def _apply_backspace(line: str) -> str:
    """Apply literal backspace characters (BS ``\x08`` / DEL ``\x7f``) to a line.

    A cooked console applies backspace before the line reaches Python, so for
    normal terminal typing this is a no-op.  It still matters when the
    characters arrive literally - piped input, or a paste containing control
    characters - so visible mode honours editing as far as practical.  A
    backspace at the start of the line deletes nothing.
    """
    edited: list[str] = []
    for char in line:
        if char in ("\b", "\x7f"):
            if edited:
                edited.pop()
        else:
            edited.append(char)
    return "".join(edited)


def keys_set_command(
    store: secret_module.SecretStore,
    short: str,
    *,
    value: str | None = None,
    interactive: bool = True,
    prompt_fn: Any = None,
    visible: bool = False,
) -> str:
    """Securely store a provider API key; hidden input unless ``visible=True``.

    Default (``visible=False``): input stays completely hidden (``getpass``)
    while typing or pasting, and the success receipt is the saved confirmation
    line followed by a masked preview (``Key: <first4>•••…last4>``) built by
    :func:`masked_key_preview` - the full value is never part of the return
    value, any log, or any exception.

    ``visible=True`` is the explicit opt-in (CLI: ``jarvis keys set <p>
    --visible``) that reads through :func:`visible_input` (the terminal's own
    line editor, so echo/paste/backspace work in PowerShell) and puts the
    **full** key on the ``Key:`` line of the receipt.  Even then the value is
    never logged and never raised in an exception: the receipt is built only
    *after* the credential store accepted the value, so a storage failure or
    empty input returns no fragment of the key at all.
    """
    _validate_provider(short)
    if interactive:
        reader = (prompt_fn or visible_input) if visible else (prompt_fn or getpass.getpass)
        raw = reader(f"{_provider_label(short)} API key: ")
        value = _apply_backspace(raw or "") if visible else raw
    value = (value or "").strip()
    if not value:
        raise ValueError(_KEYS_SET_EMPTY_MESSAGE)
    store.set(PROVIDER_SECRETS[short], value)
    shown = value if visible else masked_key_preview(value)
    return f"{_KEYS_SET_SAVED_MESSAGE}\n{_KEYS_SET_PREVIEW_PREFIX}{shown}"


def keys_paste_command(
    store: secret_module.SecretStore,
    short: str,
    *,
    clipboard_fn: Any = None,
    clear_clipboard_fn: Any = None,
) -> str:
    """Import an API key from the clipboard. Returns "saved".

    Strips whitespace, rejects empty values, stores in the keyring and then
    best-effort clears the clipboard (documented in :func:`_clipboard_clear`).
    The key itself is never printed.
    """
    _validate_provider(short)
    clipboard_fn = clipboard_fn or _clipboard_paste
    value = (clipboard_fn() or "").strip()
    if not value:
        raise ValueError(f"clipboard was empty - nothing stored for {short}")
    store.set(PROVIDER_SECRETS[short], value)
    if clear_clipboard_fn is not None:
        clear_clipboard_fn()
    return "saved"


def keys_clear_command(store: secret_module.SecretStore, short: str) -> str:
    """Remove a stored provider API key. Returns "cleared"."""

    _validate_provider(short)
    store.delete(PROVIDER_SECRETS[short])
    return "cleared"


def keys_status_lines(store: secret_module.SecretStore) -> list[str]:
    """Render one line per provider: ``groq: configured`` / ``not configured``.

    Presence is effective (keyring, then environment variable); no prefix,
    suffix, length, hash or masked value is ever shown.
    """
    lines: list[str] = []
    for short in CREDENTIAL_PROVIDERS:
        present = provider_has_credential(store, short)
        lines.append(f"{short}: {'configured' if present else 'not configured'}")
    return lines


def _live_provider_check(
    settings: config.Settings, store: secret_module.SecretStore, name: str
) -> Check:
    """Probe one configured provider live (endpoint/auth/model/capabilities).

    Does not call any API unless the provider has a credential.  Errors are
    mapped to safe messages (never credentials).  Rate limits are a WARN,
    model availability problems are a FAIL.
    """
    label = f"live_{name}"
    try:
        client = build_provider_client(settings, store, name)
    except Exception as exc:  # noqa: BLE001 - LLMConfigError etc. are safe messages
        return _check(label, "FAIL", str(exc))
    try:
        try:
            client.text(
                system="You are a connectivity probe.",
                user="Reply with the single word: ok",
                model_role="fast",
                max_tokens=16,
            )
        finally:
            close = getattr(client, "close", None)
            if close is not None:
                close()
    except LLMAuthError:
        return _check(label, "FAIL", f"{name} rejected the API key (authentication failed)")
    except LLMModelError as exc:
        return _check(label, "FAIL", str(exc))
    except LLMCapabilityError as exc:
        return _check(label, "WARN", str(exc))
    except LLMTransientError:
        return _check(label, "WARN", f"{name} is rate limited or transiently unavailable")
    except LLMError as exc:
        return _check(label, "FAIL", str(exc))
    model = settings.llm.model_for(name, "fast") or settings.llm.model_for(name, "planner")
    return _check(label, "PASS", f"{model} reachable ({name}, eligible under current policy)")


def _provider_model_diagnostic(
    settings: config.Settings,
    name: str,
    planner: str,
    fast: str,
) -> tuple[Check, bool]:
    """Return one provider model check and whether it is selectable."""
    if not planner and not fast:
        return (
            _check(
                f"{name}_model",
                "WARN",
                f"no model configured for {name} - run `jarvis init`",
            ),
            False,
        )
    effective_planner = planner or fast
    effective_fast = fast or planner
    role_models = [("planner", effective_planner)]
    if effective_fast != effective_planner:
        role_models.append(("fast", effective_fast))
    evaluated: list[tuple[str, str, llm_models.ModelSpec, bool, str]] = []
    for role, model in role_models:
        allowed, spec, reason = llm_models.qualify(
            name,
            model,
            free_only=settings.llm.free_only,
            strict_zero_cost=settings.llm.strict_zero_cost,
        )
        evaluated.append((role, model, spec, allowed, reason))
    failures = [item for item in evaluated if not item[3]]
    if not failures:
        details = "; ".join(
            f"{role}={model}: {llm_models.pricing_message(spec)} "
            f"(tools={spec.capabilities.supports_tools}, "
            f"structured={spec.capabilities.supports_structured_output})"
            for role, model, spec, _allowed, _reason in evaluated
        )
        return _check(f"{name}_model", "PASS", details), True
    refused = next(
        (item for item in failures if item[2].pricing_mode != "free_tier"),
        failures[0],
    )
    if refused[2].pricing_mode == "free_tier":
        return (
            _check(
                f"{name}_model",
                "WARN",
                f"{refused[1]} is free-tier eligible, but account billing state cannot be "
                "verified under strict zero-cost mode (set strict_zero_cost=false to allow)",
            ),
            False,
        )
    return _check(f"{name}_model", "FAIL", f"{refused[0]} model: {refused[4]}"), False


def run_doctor(
    *,
    settings: config.Settings | None = None,
    store: secret_module.SecretStore | None = None,
    live: bool = False,
) -> list[Check]:
    """Produce environment checks without touching the console.

    Plain ``doctor`` never calls any API: it reports credential presence
    (keyring, then environment) and checks that every configured provider has
    a free-eligible model.  ``live=True`` probes only providers that have a
    credential.
    """
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

    backend = store.check_store_access()
    if backend:
        checks.append(_check("keyring", "PASS", f"credential store reachable ({backend})"))
    else:
        checks.append(
            _check("keyring", "FAIL", "credential store unreachable - secrets cannot be protected")
        )

    providers = settings.llm.provider_order
    present: dict[str, bool] = {name: provider_has_credential(store, name) for name in providers}
    any_present = any(present.values())

    for name in providers:
        if present[name]:
            checks.append(_check(f"{name}_api_key", "PASS", "present"))
        elif any_present:
            checks.append(
                _check(
                    f"{name}_api_key",
                    "WARN",
                    f"{name}_api_key missing - provide it with `jarvis keys set {name}`",
                )
            )
        else:
            checks.append(
                _check(
                    f"{name}_api_key",
                    "FAIL",
                    f"{name}_api_key missing - provide it with `jarvis keys set {name}`",
                )
            )

    tavily_present = provider_has_credential(store, "tavily")
    checks.append(
        _check(
            "tavily_api_key",
            "PASS" if tavily_present else "WARN",
            "present" if tavily_present else "tavily_api_key missing (optional web search)",
        )
    )

    if settings.llm.free_only:
        checks.append(_check("free_only", "PASS", "enabled - paid models are rejected"))
    else:
        checks.append(
            _check("free_only", "WARN", "disabled - paid models could be selected; not recommended")
        )

    if settings.llm.strict_zero_cost:
        checks.append(
            _check(
                "strict_zero_cost",
                "PASS",
                "enabled - only verified zero-cost endpoints are selectable",
            )
        )
    else:
        checks.append(
            _check(
                "strict_zero_cost",
                "WARN",
                "disabled - free-tier models are allowed; their billing state "
                "cannot be verified, so this is not strictly zero-cost",
            )
        )

    configured_providers: list[str] = []
    for name in providers:
        planner = settings.llm.model_for(name, "planner")
        fast = settings.llm.model_for(name, "fast")
        model_check, selectable = _provider_model_diagnostic(settings, name, planner, fast)
        checks.append(model_check)
        if present[name] and selectable:
            configured_providers.append(name)

    if configured_providers:
        checks.append(
            _check(
                "llm_provider", "PASS", "free LLM provider(s): " + ", ".join(configured_providers)
            )
        )
    else:
        checks.append(_check("llm_provider", "FAIL", "No free LLM provider configured"))

    checks.append(
        _check(
            "vision",
            "WARN",
            "vision is not wired into the graph yet; no vision capability is required",
        )
    )

    if live:
        live_ran = False
        for name in providers:
            if present[name]:
                checks.append(_live_provider_check(settings, store, name))
                live_ran = True
        if not live_ran:
            checks.append(
                _check(
                    "live_llm", "FAIL", "No free LLM provider configured - nothing to probe live"
                )
            )
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


# ── init ─────────────────────────────────────────────────────────────────


@app.command()
def init(
    groq_api_key: Annotated[
        str | None, typer.Option("--groq-api-key", help="Groq API key (omit to be prompted).")
    ] = None,
    openrouter_api_key: Annotated[
        str | None,
        typer.Option("--openrouter-api-key", help="OpenRouter API key (optional)."),
    ] = None,
    gemini_api_key: Annotated[
        str | None, typer.Option("--gemini-api-key", help="Gemini API key (optional).")
    ] = None,
    nvidia_api_key: Annotated[
        str | None, typer.Option("--nvidia-api-key", help="NVIDIA API key (optional).")
    ] = None,
    tavily_api_key: Annotated[
        str | None, typer.Option("--tavily-api-key", help="Tavily API key (optional).")
    ] = None,
    non_interactive: Annotated[
        bool, typer.Option("--non-interactive", help="Never prompt; use flags/env only.")
    ] = False,
) -> None:
    """Initialise JARVIS: config + store free-provider API keys.

    Each provider is prompted exactly once; pressing Enter skips it.  API
    keys live only in the OS credential store - never in config.toml.
    """
    provided: dict[str, str] = {}
    for short, value in (
        ("groq", groq_api_key),
        ("openrouter", openrouter_api_key),
        ("gemini", gemini_api_key),
        ("nvidia", nvidia_api_key),
        ("tavily", tavily_api_key),
    ):
        if value:
            provided[short] = value
    console.print("[bold]JARVIS LLM setup[/bold]")
    for line in init_config(
        secret_module.SecretStore(),
        provided_keys=provided,
        interactive=not non_interactive,
    ):
        console.print(line)


# ── keys subcommand ──────────────────────────────────────────────────────


def _wait_for_enter() -> None:
    """Single blocking read (no loop); EOF is treated as cancellation."""
    try:
        input("Press Enter when the API key is on the clipboard...")
    except (EOFError, KeyboardInterrupt):
        return


keys_app = typer.Typer(
    help="Manage provider API keys (OS credential store only).", no_args_is_help=True
)
app.add_typer(keys_app, name="keys")


@keys_app.command("set")
def keys_set(
    provider: Annotated[str, typer.Argument(help="groq|openrouter|gemini|nvidia|tavily")],
    visible: Annotated[
        bool,
        typer.Option(
            "--visible",
            help="Echo the key while typing and print it in full (INSECURE, opt-in only).",
        ),
    ] = False,
) -> None:
    """Store a provider API key (hidden input; opt in with --visible)."""
    if visible:
        console.print(f"[yellow]{_KEYS_SET_VISIBLE_WARNING}[/yellow]")
    try:
        message = keys_set_command(secret_module.SecretStore(), provider, visible=visible)
    except secret_module.SecretStoreError:
        console.print("[red]✗ API key received, but secure storage failed.[/red]")
        raise typer.Exit(code=1)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    # markup=False + soft_wrap: the receipt may contain the raw key in visible
    # mode, so rich must neither parse it as markup nor wrap it mid-line.
    console.print(message, style="green", markup=False, soft_wrap=True)


@keys_app.command("paste")
def keys_paste(
    provider: Annotated[str, typer.Argument(help="groq|openrouter|gemini|nvidia|tavily")],
) -> None:
    """Import an API key from the Windows clipboard (safe for pasting)."""
    try:
        _validate_provider(provider)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    console.print(
        f"Copy the API key to the clipboard, then press Enter. ({_provider_label(provider)})"
    )
    _wait_for_enter()
    try:
        message = keys_paste_command(
            secret_module.SecretStore(),
            provider,
            clear_clipboard_fn=_clipboard_clear,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    console.print(f"[green]{message}[/green]")  # only ever says "saved"


@keys_app.command("clear")
def keys_clear(
    provider: Annotated[str, typer.Argument(help="groq|openrouter|gemini|nvidia|tavily")],
) -> None:
    """Remove a stored provider API key from the credential store."""
    try:
        message = keys_clear_command(secret_module.SecretStore(), provider)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    console.print(f"[green]{message}[/green]")


@keys_app.command("status")
def keys_status() -> None:
    """Show which provider credentials are present (never the keys)."""
    for line in keys_status_lines(secret_module.SecretStore()):
        console.print(line)


# ── doctor ───────────────────────────────────────────────────────────────


@app.command()
def doctor(
    live: Annotated[
        bool,
        typer.Option(
            "--live",
            help="Also probe configured providers with real API calls (never a paid model).",
        ),
    ] = False,
) -> None:
    """Verify the JARVIS environment (free-only LLM setup)."""
    checks = run_doctor(live=live)
    table = Table(show_header=False, box=None)
    for check in checks:
        table.add_row(check.status, check.name, check.detail)
    console.print(table)
    failures = [c for c in checks if c.status == "FAIL"]
    if failures:
        raise typer.Exit(code=1)


# ── status ───────────────────────────────────────────────────────────────


@app.command()
def status() -> None:
    """Show daemon status."""
    try:
        from jarvis.daemon.client import DaemonClient

        client = DaemonClient()
        client.connect()
        try:
            resp = client.get_status()
            console.print(f"[green]daemon:[/green] {resp.daemon}")
            if resp.voice_reason:
                console.print(f"[red]voice:[/red] {resp.voice} ({resp.voice_reason})")
            else:
                console.print(f"[green]voice:[/green] {resp.voice}")
            try:
                if hasattr(resp, "mode"):
                    console.print(f"[green]mode:[/green] {resp.mode}")
            except Exception:  # noqa: BLE001, S110 - best effort
                pass
            console.print(f"[green]unlocked:[/green] {resp.unlocked}")
            console.print(f"[green]queue:[/green] {resp.queue}")
            if resp.active_task:
                console.print(f"[green]active task:[/green] {resp.active_task}")
            console.print(f"[green]version:[/green] {resp.version}")
        finally:
            client.close()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[yellow]daemon not running: {exc}[/yellow]")
        raise typer.Exit(code=1)


# ── audit ─────────────────────────────────────────────────────────────────


def _run_audit_cli(sections: list[str] | None, out: Path | None) -> tuple[list[Any], Path]:
    """Run read-only audit checks and write a Markdown report."""
    from jarvis.audit import checks as audit_checks
    from jarvis.audit.report import write_report
    from jarvis.tools.audit import _SECTIONS

    chosen = list(sections) if sections else list(_SECTIONS)
    unknown = [s for s in chosen if s not in _SECTIONS]
    if unknown:
        raise typer.BadParameter(f"unknown audit section: {', '.join(unknown)}")
    findings = audit_checks.run_checks(
        chosen, log_path=config.log_file(), memory_db=config.memory_db()
    )
    target = out if out is not None else config.reports_dir()
    report_path = write_report(findings, target)
    return findings, report_path


@app.command()
def audit(
    section: Annotated[
        list[str] | None,
        typer.Option(
            "--section",
            "-s",
            help="Audit section to run (repeatable): security, performance, "
            "updates, self. Defaults to all.",
        ),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Directory for the report (default: user data reports)."),
    ] = None,
) -> None:
    """Run a read-only system audit and write a Markdown report."""
    from rich.table import Table

    try:
        findings, report_path = _run_audit_cli(section, out)
    except typer.BadParameter:
        raise
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]audit failed: {exc}[/red]")
        raise typer.Exit(code=1)

    count = {"critical": 0, "warning": 0, "info": 0, "ok": 0, "unknown": 0}
    for finding in findings:
        count[finding.severity] += 1
    table = Table(show_header=False, box=None)
    for severity in ("critical", "warning", "info", "ok", "unknown"):
        label = {
            "critical": "red",
            "warning": "yellow",
            "info": "cyan",
            "ok": "green",
            "unknown": "magenta",
        }[severity]
        if count[severity]:
            table.add_row(f"[{label}]{severity}[/{label}]", str(count[severity]))
    console.print("[bold]Audit summary[/bold]")
    console.print(table)
    console.print(f"[green]report:[/green] {report_path}")
    if count["critical"]:
        console.print("[red]Critical issues found - review the report.[/red]")


# ── stop ─────────────────────────────────────────────────────────────────


@app.command()
def stop() -> None:
    """Stop the daemon gracefully."""
    try:
        from jarvis.daemon.client import DaemonClient

        client = DaemonClient()
        client.connect()
        try:
            client.send_shutdown()
            console.print("[green]shutdown signal sent[/green]")
        finally:
            client.close()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[yellow]daemon not running: {exc}[/yellow]")
        raise typer.Exit(code=1)


# ── daemon (foreground) ──────────────────────────────────────────────────


@app.command()
def daemon(
    foreground: Annotated[
        bool, typer.Option("--foreground", help="Run in foreground (dev mode).")
    ] = True,
) -> None:
    """Run the JARVIS daemon (normally started by Task Scheduler)."""
    from jarvis.daemon.server import DaemonServer

    if not foreground:
        console.print("[yellow]background mode not yet implemented; use --foreground[/yellow]")
        raise typer.Exit(code=2)

    settings = config.load_settings()
    store = secret_module.SecretStore()
    server = DaemonServer(settings=settings, store=store)
    print(
        "[DIAG] daemon(): DaemonServer constructed "
        f"(voice.enabled={settings.voice.enabled} tts_backend={settings.voice.tts_backend})",
        flush=True,
    )
    console.print("[dim]JARVIS daemon starting (Ctrl+C to stop)...[/dim]")
    console.print(f"[dim]logs: {config.log_file()}[/dim]")
    try:
        server.run()
    except KeyboardInterrupt:
        print("[DIAG] daemon(): server.run() raised KeyboardInterrupt", flush=True)
        console.print("[dim]daemon stopped[/dim]")
    except Exception:
        print("[DIAG] daemon(): server.run() raised Exception", flush=True)
        traceback.print_exc()
        # Logging is configured in main(); the full traceback goes to the log
        # file so a daemon crash is always diagnosable (never a silent exit).
        logging.getLogger(__name__).exception("daemon crashed")
        console.print("[red]daemon crashed — see the log file for details[/red]")
        raise typer.Exit(code=1) from None


# ── autostart ────────────────────────────────────────────────────────────


@autostart_app.command("enable")
def autostart_enable_cmd() -> None:
    """Enable daemon autostart at login."""
    from jarvis.daemon.autostart import enable

    result = enable()
    if result["ok"]:
        console.print(f"[green]{result['message']}[/green]")
    else:
        console.print(f"[red]{result['error']}[/red]")
        raise typer.Exit(code=1)


@autostart_app.command("disable")
def autostart_disable_cmd() -> None:
    """Disable daemon autostart."""
    from jarvis.daemon.autostart import disable

    result = disable()
    if result["ok"]:
        console.print(f"[green]{result['message']}[/green]")
    else:
        console.print(f"[red]{result['error']}[/red]")
        raise typer.Exit(code=1)


@autostart_app.command("status")
def autostart_status_cmd() -> None:
    """Check daemon autostart status."""
    from jarvis.daemon.autostart import status

    result = status()
    if not result["ok"]:
        console.print(f"[red]{result['error']}[/red]")
        raise typer.Exit(code=1)
    if result.get("enabled"):
        console.print("[green]autostart: enabled[/green]")
        for key in ("status", "last_run", "next_run", "last_result"):
            if key in result:
                console.print(f"  {key}: {result[key]}")
    else:
        console.print("[yellow]autostart: not enabled[/yellow]")


# ── chat ─────────────────────────────────────────────────────────────────


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
        outcome = resolve_interrupts(outcome, ctx, saver)
        _print_outcome(outcome)


def resolve_interrupts(outcome: Any, ctx: AppContext, saver: SqliteSaver) -> Any:
    """Answer every pending clarification / confirmation, then return.

    Shared by the interactive REPL (:func:`chat_loop`) and the one-shot
    ``jarvis run`` so both take *exactly* the same safety path: an approval can
    never be obtained more easily through the non-interactive command than
    through the chat REPL.  Each answer resumes the graph, which may surface
    the other kind of interrupt in turn.
    """
    while True:
        if outcome.interrupt_kind == "clarification":
            req = outcome.confirmation
            console.print("[yellow]clarification needed[/yellow]")
            console.print(req.get("question") or "(no question)")
            answer = _chat_prompt("Your answer:")
            outcome = resume_task(ctx, saver, outcome.task_id, answer or "")
            continue
        if not outcome.confirmation:
            return outcome
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
                "[red]NOTE: this action was derived from untrusted content (web/file text).[/red]"
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


def _print_outcome(outcome: Any) -> None:
    """Print a finished task result (shared by ``chat`` and ``run``)."""
    if outcome.error:
        console.print(f"[red]{outcome.error}[/red]")
    elif outcome.final_answer:
        console.print(f"[green]{outcome.final_answer}[/green]")
    else:
        console.print("[dim](no answer)[/dim]")


#: Seconds to wait for the daemon's cancel receipt when abandoning a task the
#: user interrupted.  The cancel is on the wire before the receipt is read, so a
#: wedged daemon must not turn Ctrl+C into a long silence.
_ABANDON_ACK_TIMEOUT = 2.0


def _daemon_chat_loop(client: Any) -> None:
    """Interactive REPL that talks to the daemon over IPC."""
    console.print("[dim]JARVIS chat via daemon (Ctrl+C to exit)[/dim]")
    while True:
        try:
            line = _chat_prompt("[bold cyan]you>[/]")
        except (KeyboardInterrupt, EOFError):
            console.print("[dim]bye[/dim]")
            break
        line = (line or "").strip()
        if not line:
            continue
        _daemon_one_shot(client, line)


def _daemon_one_shot(client: Any, line: str) -> str:
    """Send one command to the daemon and print events until it finishes.

    Returns the final text (empty when the daemon reported an error, the wait
    timed out, or the user interrupted).  Shared by the chat REPL and
    ``jarvis run`` so the one-shot command cannot be more permissive than the
    interactive session: both go through the same ``ConfirmRequest`` /
    ``ClarificationRequest`` handling and the same password prompt.

    ``Ctrl+C`` anywhere in here - at a prompt, or while the daemon re-plans
    after an answer - returns to the caller instead of unwinding out of
    ``jarvis chat``: it used to kill the whole REPL, and because the task kept
    the daemon's single active slot the *next* command was then silently queued
    behind an abandoned run, which is what "typed it again and nothing
    happened" looked like.
    """
    from jarvis.daemon.client import DaemonError

    try:
        task_id = client.send_chat(line)
    except DaemonError as exc:
        console.print(f"[red]{exc.message}[/red]")
        return ""

    _show_submission_ack(client)
    try:
        return _daemon_event_loop(client, task_id)
    except (KeyboardInterrupt, EOFError):
        _abandon_task(client, task_id)
        console.print("[dim]cancelled[/dim]")
        return ""


def _show_submission_ack(client: Any) -> None:
    """Tell the user whether the request started now or was queued behind one."""
    note = str(getattr(client, "last_ack_message", "") or "").strip()
    if note:
        console.print(f"[dim]{note}[/dim]")


def _abandon_task(client: Any, task_id: str) -> None:
    """Best-effort cancel of a task the user walked away from.

    Without this the daemon keeps the slot until its 30 s confirmation window
    elapses, and the user's next command waits in the queue.  The short ack
    timeout matters too: the user just pressed Ctrl+C, so the cancel must not
    trade a silent hang for a 30 s one.
    """
    cancel = getattr(client, "send_cancel", None)
    if not task_id or not callable(cancel):
        return
    try:
        cancel(task_id, ack_timeout=_ABANDON_ACK_TIMEOUT)
    except Exception:
        logging.getLogger("jarvis.cli").debug("cancelling the abandoned task failed", exc_info=True)


def _daemon_event_loop(client: Any, task_id: str) -> str:
    """Print daemon events for ``task_id`` until it produces a final answer."""
    from jarvis.daemon.client import DaemonError
    from jarvis.daemon.protocol import (
        ClarificationRequest,
        ConfirmRequest,
        ErrorMessage,
        EventMessage,
        FinalMessage,
    )

    while True:
        msg = client.wait_for_event(timeout=300)
        if msg is None:
            console.print("[yellow]timed out waiting for response[/yellow]")
            return ""
        if isinstance(msg, FinalMessage):
            console.print(f"[green]{msg.text}[/green]")
            return msg.text
        if isinstance(msg, ConfirmRequest):
            _handle_confirm_request(client, task_id, msg)
        elif isinstance(msg, ClarificationRequest):
            console.print(f"[yellow]clarification needed: {msg.question}[/yellow]")
            answer = _chat_prompt("Your answer:")
            try:
                client.send_clarification(task_id, answer=answer or "")
                console.print("[dim](working...)[/dim]")
            except DaemonError as exc:
                console.print(f"[red]{exc.message}[/red]")
        elif isinstance(msg, ErrorMessage):
            console.print(f"[red]{msg.message}[/red]")
            return ""
        elif isinstance(msg, EventMessage):
            # Streaming events (plan, step_start, step_result, log)
            data = msg.data
            if msg.kind == "log" and "message" in data:
                console.print(f"[dim]{data['message']}[/dim]")


def _handle_confirm_request(client: Any, task_id: str, msg: Any) -> None:
    """Show a Tier confirmation, collect the answer, and send it back.

    The ``action_hash`` is echoed verbatim, so the daemon can re-check that the
    approval belongs to the exact action it interrupted on (invariant 7).  The
    password, if asked for, goes to the daemon and never into the graph.
    """
    from jarvis.daemon.client import DaemonError
    from jarvis.policy import tiers as _tiers

    console.print("".join(("[yellow]approval needed[/yellow] (", _tiers.tier_label(msg.tier), ")")))
    console.print(msg.summary or "(no summary)")
    if msg.untrusted:
        console.print("[red]NOTE: this action was derived from untrusted content[/red]")
    answer = _chat_prompt("Approve?")
    agreed = (answer or "").strip().lower() in ("y", "yes")
    typed: str | None = None
    if agreed and msg.typed_confirmation:
        typed = _chat_prompt(f"Type this exactly to confirm: {msg.typed_confirmation}")
    password: str | None = None
    if agreed and msg.needs_password:
        password = _chat_password_prompt("JARVIS password: ")

    try:
        client.send_confirm(
            task_id,
            approved=agreed,
            action_hash=msg.action_hash,
            password=password,
            typed_confirmation=typed,
            confirmation_id=getattr(msg, "confirmation_id", ""),
            plan_hash=getattr(msg, "plan_hash", ""),
        )
        console.print("[dim](working...)[/dim]")
    except DaemonError as exc:
        console.print(f"[red]{exc.message}[/red]")


@app.command()
def run(
    command: Annotated[str, typer.Argument(help="What JARVIS should do, in plain English.")],
    no_daemon: Annotated[
        bool, typer.Option("--no-daemon", help="Run the agent in-process (no daemon).")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Describe the actions; execute nothing.")
    ] = False,
) -> None:
    """Run a single command and print the result (one shot, then exit)."""
    command = command.strip()
    if not command:
        console.print('[red]give a command, e.g. jarvis run "open notepad"[/red]')
        raise typer.Exit(code=2)
    if dry_run and not no_daemon:
        # Fail closed: the daemon owns one long-lived AppContext, so a
        # per-request dry-run flag cannot be honoured there without mutating
        # shared state.  Refuse rather than silently running for real.
        console.print("[red]--dry-run needs --no-daemon (the daemon runs for real)[/red]")
        raise typer.Exit(code=2)
    if no_daemon:
        _run_no_daemon(command, dry_run=dry_run)
    else:
        _run_with_daemon(command)


def _task_context(
    settings: config.Settings,
    store: secret_module.SecretStore,
    llm: Any,
    *,
    dry_run: bool = False,
) -> AppContext:
    """Build the in-process task context (LLM + memory + policy + classifier).

    The registry is built once and shared: the risk classifier derives its
    per-tool priors from it, so both decide from the same specs.
    """
    from jarvis.tools.registry import build_default_registry

    log = logging.getLogger("jarvis.cli")
    registry = build_default_registry(settings)
    return make_app_context(
        settings,
        llm=llm,
        registry=registry,
        classifier=build_classifier(settings, registry, logger=log),
        unlock=UnlockManager(store, settings=settings),
        dry_run=dry_run,
        memory=open_memory(settings=settings, logger=log, dry_run=dry_run),
    )


def _run_no_daemon(command: str, *, dry_run: bool = False) -> None:
    """One in-process task, with the same confirmation path as ``chat``."""
    settings = config.load_settings()
    store = secret_module.SecretStore()
    selection = build_llm_client(settings, store, logger=logging.getLogger("jarvis.cli"))
    if selection.client is None:
        detail = "; ".join(selection.reasons) or "no provider is configured"
        console.print(f"[red]No free LLM provider configured: {detail}[/red]")
        raise typer.Exit(code=1)

    try:
        ctx = _task_context(settings, store, selection.client, dry_run=dry_run)
        saver = open_sqlite_checkpointer(str(config.checkpoints_db()))
    except LLMError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    try:
        outcome = run_task(ctx, saver, command)
        outcome = resolve_interrupts(outcome, ctx, saver)
        _print_outcome(outcome)
        if outcome.error or not outcome.final_answer:
            raise typer.Exit(code=1)
    finally:
        _close_memory_backend(ctx)
        _close_checkpointer(saver)


def _close_checkpointer(saver: Any) -> None:
    """Close the checkpointer's connection, if the installed saver exposes one."""
    conn = getattr(saver, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            logging.getLogger("jarvis.cli").debug("closing checkpointer failed", exc_info=True)


def _run_with_daemon(command: str) -> None:
    """Send one command to a running daemon and print the result."""
    from jarvis.daemon.client import DaemonClient, DaemonError

    try:
        client = DaemonClient()
        client.connect()
    except DaemonError as exc:
        console.print(f"[red]cannot connect to daemon: {exc.message}[/red]")
        console.print("[dim]start it with `jarvis daemon --foreground`, or use --no-daemon[/dim]")
        raise typer.Exit(code=1) from exc
    try:
        if not _daemon_one_shot(client, command):
            raise typer.Exit(code=1)
    finally:
        try:
            client.close()
        except Exception:
            logging.getLogger("jarvis.cli").debug("closing daemon client failed", exc_info=True)


@app.command()
def chat(
    no_daemon: Annotated[
        bool, typer.Option("--no-daemon", help="Run the agent in-process (no daemon).")
    ] = False,
) -> None:
    """Talk to JARVIS (interactive REPL)."""
    if no_daemon:
        _chat_no_daemon()
    else:
        _chat_with_daemon()


def _chat_no_daemon() -> None:
    """Run the agent in-process (Phase 1 mode)."""
    settings = config.load_settings()
    store = secret_module.SecretStore()
    selection = build_llm_client(settings, store, logger=logging.getLogger("jarvis.cli"))
    if selection.client is None:
        detail = "; ".join(selection.reasons) or "no provider is configured"
        console.print(f"[red]No free LLM provider configured: {detail}[/red]")
        console.print(
            "[dim]run `jarvis init` or `jarvis keys set <provider>` to set up a free provider.[/dim]"
        )
        raise typer.Exit(code=1)

    try:
        ctx = _task_context(settings, store, selection.client)
        saver = open_sqlite_checkpointer(str(config.checkpoints_db()))
    except LLMError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    console.print("[dim]JARVIS chat (Ctrl+C to exit)[/dim]")
    try:
        chat_loop(ctx, saver)
    finally:
        _close_memory_backend(ctx)


def _chat_with_daemon() -> None:
    """Connect to the daemon and start an interactive chat session."""
    from jarvis.daemon.client import DaemonClient, DaemonError

    try:
        client = DaemonClient()
        client.connect()
    except DaemonError as exc:
        console.print(f"[red]cannot connect to daemon: {exc.message}[/red]")
        console.print("[dim]start the daemon with: jarvis daemon --foreground[/dim]")
        raise typer.Exit(code=1)

    try:
        _daemon_chat_loop(client)
    finally:
        client.close()


# ── undo ─────────────────────────────────────────────────────────────────

#: The tool ``jarvis undo`` runs.  Named here so the command fails loudly at
#: import-free test time if the registry ever drops it.
UNDO_TOOL = "undo_last_delete"

#: ``jarvis undo`` is a hard-coded single-tool invocation, not an LLM plan, so
#: there is no user wording to hand the classifier - inventing a sentence here
#: would let a fixed string decide a safety tier.  The classifier still runs on
#: the action itself (tool name + args + the tool's declared prior), so a future
#: change to the tool's tier is still enforced; with an empty string the
#: documented Tier 1 stands and the confirmation below is the gate.
UNDO_CLASSIFIER_INPUT = ""


def _undo_step() -> Any:
    """The deterministic single step behind ``jarvis undo`` (no LLM involved)."""
    from jarvis.agent.state import Step

    return Step(
        id="undo-last-delete",
        tool=UNDO_TOOL,
        args={},
        rationale="user ran `jarvis undo`",
        expect="the most recently deleted item is back where it was",
    )


def _undo_context(
    settings: config.Settings, store: secret_module.SecretStore, *, dry_run: bool = False
) -> AppContext:
    """Context for ``jarvis undo``: policy + registry + classifier, no LLM.

    No memory backend and no checkpointer are opened, so the command has no
    persistent state to leak or leave locked (contrast ``run``/``chat``).
    """
    from jarvis.tools.registry import build_default_registry

    log = logging.getLogger("jarvis.cli")
    registry = build_default_registry(settings)
    return make_app_context(
        settings,
        registry=registry,
        classifier=build_classifier(settings, registry, logger=log),
        unlock=UnlockManager(store, settings=settings),
        dry_run=dry_run,
    )


def _confirm_undo(ctx: AppContext, decision: Any) -> bool:
    """Ask for approval, then a Tier-2 password - the graph's exact order.

    There is deliberately no ``--yes``: an approval that could be obtained more
    easily from the command line than from the chat REPL would defeat the whole
    confirmation design (docs/03 invariant 7).
    """
    console.print(f"[yellow]approval needed[/yellow] ({tiers.tier_label(decision.tier)})")
    console.print(decision.summary)
    answer = _chat_prompt("Approve?")
    agreed = (answer or "").strip().lower() in ("y", "yes")
    if agreed and decision.needs_typed_confirmation:
        typed = _chat_prompt(f"Type this exactly to confirm: {decision.needs_typed_confirmation}")
        agreed = (typed or "").strip() == decision.needs_typed_confirmation
    if agreed and decision.needs_unlock:
        agreed = _tier2_unlock_or_refuse(ctx, {"action_hash": decision.action_hash})
    return agreed


def _run_undo(dry_run: bool = False) -> None:
    """Restore the most recent delete, through the normal policy path."""
    from jarvis.agent.nodes.util import policy_context_for

    settings = config.load_settings()
    store = secret_module.SecretStore()
    ctx = _undo_context(settings, store, dry_run=dry_run)
    policy_ctx = policy_context_for(
        {"user_input": UNDO_CLASSIFIER_INPUT, "results": []}, ctx.policy_ctx
    )
    if policy_ctx is None:  # pragma: no cover - make_app_context always sets it
        console.print("[red]no policy context; refusing[/red]")
        raise typer.Exit(code=1)
    decision = ctx.engine.decide(_undo_step(), policy_ctx)
    if not decision.allowed:
        for reason in decision.reasons or ["blocked by policy"]:
            console.print(f"[red]{reason}[/red]")
        raise typer.Exit(code=1)
    if decision.needs_confirm and not _confirm_undo(ctx, decision):
        console.print("[yellow]cancelled - nothing was restored[/yellow]")
        raise typer.Exit(code=1)

    spec = ctx.registry.get(UNDO_TOOL)
    args = spec.args_model.model_validate({})
    result = spec.run_verified(args, ctx.tool_context())
    if not result.ok:
        console.print(f"[red]{result.error}[/red]")
        raise typer.Exit(code=1)
    console.print(result.output or "done")
    if result.verified is False:
        console.print("[yellow]the restore could not be verified - check the file[/yellow]")
        raise typer.Exit(code=1)


@app.command()
def undo(
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Describe the action; restore nothing.")
    ] = False,
) -> None:
    """Restore the most recently deleted files from the Recycle Bin."""
    try:
        _run_undo(dry_run=dry_run)
    except typer.Exit:
        raise
    except (OSError, ValueError) as exc:
        console.print(f"[red]could not read the undo log: {exc}[/red]")
        raise typer.Exit(code=1) from exc


# ── password commands ────────────────────────────────────────────────────


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


# ── lock / unlock ────────────────────────────────────────────────────────


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


# ── plain-function commands (testable without typer) ─────────────────────


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


# ── contacts ─────────────────────────────────────────────────────────────


def _contacts_store() -> Any:
    from jarvis.tools.whatsapp import ContactStore

    return ContactStore(config.contacts_file())


def contacts_add_command(store: Any, name: str, number: str) -> str:
    """Add a WhatsApp contact after E.164-ish validation (testable)."""
    return store.add(name, number)


def contacts_remove_command(store: Any, name: str) -> str:
    """Remove a WhatsApp contact by name (testable)."""
    return store.remove(name)


def contacts_list_command(store: Any) -> list[str]:
    """Render contact rows with masked numbers (testable; no raw numbers)."""
    from jarvis.tools.whatsapp import masked_number

    return [
        f"{c.name:<20} {masked_number(c.number)}"
        for c in sorted(store.load(), key=lambda c: c.name.lower())
    ]


contacts_app = typer.Typer(help="WhatsApp contacts.", no_args_is_help=True)
app.add_typer(contacts_app, name="contacts")


@contacts_app.command("add")
def contacts_add(
    name: Annotated[str, typer.Argument(help="Contact display name.")],
    number: Annotated[str, typer.Argument(help="Phone number (E.164-ish: +cc… or cc…).")],
) -> None:
    """Add a WhatsApp contact (non-interactive, validated)."""
    try:
        message = contacts_add_command(_contacts_store(), name, number)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]{message}[/green]")


@contacts_app.command("remove")
def contacts_remove(
    name: Annotated[str, typer.Argument(help="Contact display name (case-insensitive).")],
) -> None:
    """Remove a WhatsApp contact."""
    try:
        message = contacts_remove_command(_contacts_store(), name)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]{message}[/green]")


@contacts_app.command("list")
def contacts_list() -> None:
    """List WhatsApp contacts (numbers masked to the last 2 digits)."""
    rows = contacts_list_command(_contacts_store())
    if not rows:
        console.print("[yellow]no contacts yet; add one with `jarvis contacts add`[/yellow]")
        return
    for row in rows:
        console.print(row)


# ── learned skills / memory (Phase 8) ──────────────────────────────────────


def _open_memory_store(settings: config.Settings | None = None) -> Any:
    """Open the local memory store for a CLI command (testable seam)."""
    return open_memory(settings=settings or config.load_settings())


@contextmanager
def _memory_store() -> Any:
    """Yield an open memory store and always release its connection.

    Every skills command goes through here: a ``typer.Exit`` on a missing id
    used to skip the close and leak the SQLite handle.  When memory is disabled
    (or could not be opened) the ``NullMemory`` backend has no ``skills``
    attribute, so the caller gets a clear message instead of an
    ``AttributeError`` traceback.
    """
    memory = _open_memory_store()
    try:
        if not hasattr(memory, "skills"):
            console.print(
                "[yellow]memory is disabled in config, so there is nothing to show.\n"
                "set [memory] enabled = true in config.toml[/yellow]"
            )
            raise typer.Exit(code=1)
        yield memory
    finally:
        _close_store(memory)


def _close_memory_backend(ctx: Any) -> None:
    """Release a context's memory connection, ignoring an absent backend."""
    _close_store(getattr(ctx, "memory", None))


def _quarantined_failures(memory: Any) -> int:
    """How many failure rows are withheld from the planner (0 if unknown)."""
    count = getattr(memory, "quarantined_failures", None)
    if not callable(count):
        return 0
    try:
        return int(count())
    except Exception:
        logging.getLogger("jarvis.cli").debug("quarantined count failed", exc_info=True)
        return 0


def _close_store(memory: Any) -> None:
    """Close a memory backend (or a store opened directly by a command)."""
    close = getattr(memory, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            logging.getLogger("jarvis.cli").debug("closing memory failed", exc_info=True)


def skills_list_command(store: Any, limit: int = 20) -> list[str]:
    """Format stored skills for display (testable; no raw JSON).

    The tool list comes *before* the goal because a goal can be arbitrarily
    long: putting it first let a wide goal push the tools off the end of a
    terminal row, hiding the most useful part of the line.
    """
    rows = []
    for skill in store.skills.all(limit=limit):
        tools = ",".join(skill.tools_used) or "-"
        mark = "!" if not skill.trusted else " "
        goal = skill.goal_text.replace("\n", " ").strip()
        rows.append(
            f"{mark}{skill.id:>4} ok={skill.success_count:<3} fail={skill.fail_count:<3} "
            f"[{tools}]  {goal[:70]}"
        )
    return rows


skills_app = typer.Typer(help="Learned skills and preferences.", no_args_is_help=True)
app.add_typer(skills_app, name="skills")


@skills_app.command("list")
def skills_list(
    limit: Annotated[int, typer.Option(help="Maximum rows to show.")] = 20,
) -> None:
    """List the skills JARVIS has learned (verified plans only)."""
    with _memory_store() as memory:
        rows = skills_list_command(memory, limit=limit)
        counts = memory.counts() if hasattr(memory, "counts") else {}
        if not rows:
            console.print(
                "[yellow]no skills yet. JARVIS saves one after a task completes "
                "and every step verifies.[/yellow]"
            )
        else:
            console.print("[dim]skills (leading ! = fails more often than it succeeds)[/dim]")
            for row in rows:
                # markup=False: a goal containing "[redacted]" or "[bold]" is data,
                # not console markup, and must never be swallowed or styled.
                console.print(row, markup=False, highlight=False)
        if counts:
            summary = (
                f"[dim]{counts.get('skills', 0)} skills, "
                f"{counts.get('failures', 0)} recorded failures, "
                f"{counts.get('preferences', 0)} preferences"
            )
            # Honest accounting: rows quarantined by the unverifiable-success
            # fix are still in memory.db, just not offered to the planner.
            withheld = _quarantined_failures(memory)
            if withheld:
                summary += f", {withheld} unverifiable failures withheld"
            console.print(summary + "[/dim]")


@skills_app.command("show")
def skills_show(
    skill_id: Annotated[int, typer.Argument(help="Skill id from `jarvis skills list`.")],
) -> None:
    """Show one stored skill in full."""
    with _memory_store() as memory:
        skill = memory.skills.get(skill_id)
        if skill is None:
            console.print(f"[red]no skill with id {skill_id}[/red]")
            raise typer.Exit(code=1)
        console.print(skill.goal_text.replace("\n", " "), markup=False, highlight=False)
        console.print(f"id={skill.id} ok={skill.success_count} fail={skill.fail_count}")
        console.print(f"tools: {', '.join(skill.tools_used) or '-'}")
        console.print(f"last used: {skill.last_used_at or 'unknown'}")
        for step in (skill.plan or {}).get("steps", []):
            args = ", ".join(f"{k}={v}" for k, v in (step.get("args") or {}).items())
            console.print(f"  - {step.get('tool')}({args})", markup=False, highlight=False)


@skills_app.command("delete")
def skills_delete(
    skill_id: Annotated[int, typer.Argument(help="Skill id from `jarvis skills list`.")],
) -> None:
    """Forget one stored skill."""
    with _memory_store() as memory:
        if not memory.skills.delete(skill_id):
            console.print(f"[red]no skill with id {skill_id}[/red]")
            raise typer.Exit(code=1)
        console.print(f"[green]deleted skill {skill_id}[/green]")


@skills_app.command("clear")
def skills_clear() -> None:
    """Forget every skill, recorded failure and preference (asks first)."""
    if not typer.confirm("Delete all learned skills, failures and preferences?"):
        console.print("[dim]cancelled[/dim]")
        return
    with _memory_store() as memory:
        counts = memory.counts() if hasattr(memory, "counts") else {}
        removed = sum(
            int(n or 0)
            for n in (
                memory.skills.clear(),
                memory.failures.clear(),
                memory.preferences.clear(),
            )
        )
        console.print(f"[green]cleared {removed} memory rows[/green] (was {counts})")


# ── voice on / off ────────────────────────────────────────────────────────


def voice_on_command(settings: config.Settings, client: Any) -> str:
    """Ask the daemon to start listening.  Returns the status text to print.

    A voice error reasons from the daemon is returned verbatim (it already
    includes the fixed code and a hint) so the CLI can print it and exit 1.
    """
    if not settings.voice.enabled:
        return "voice is disabled in config — set [voice] enabled = true"
    from jarvis.daemon.client import DaemonError

    try:
        return client.send_voice_toggle(True)
    except DaemonError as exc:
        return str(exc.message or exc)
    except Exception as exc:  # noqa: BLE001
        message = getattr(exc, "message", None) or str(exc)
        return f"voice error: {message}"


def voice_off_command(client: Any) -> str:
    """Ask the daemon to stop listening.  Returns the status text to print."""
    try:
        return client.send_voice_toggle(False)
    except Exception as exc:  # noqa: BLE001
        message = getattr(exc, "message", None) or str(exc)
        return f"voice error: {message}"


@app.command()
def on() -> None:
    """Start voice listening (mic + wake word)."""
    settings = config.load_settings()
    from jarvis.daemon.client import DaemonClient, DaemonError

    try:
        client = DaemonClient()
        client.connect()
    except DaemonError as exc:
        console.print(f"[yellow]cannot connect to daemon: {exc.message}[/yellow]")
        console.print("[dim]start the daemon with: jarvis daemon --foreground[/dim]")
        raise typer.Exit(code=1)

    try:
        message = voice_on_command(settings, client)
        if message.startswith("voice is in error:"):
            console.print(f"[red]{message}[/red]")
            raise typer.Exit(code=1)
        if "active" in message:
            console.print(f"[green]{message}[/green]")
            console.print("[dim]say 'stop listening' or 'jarvis off' to stop[/dim]")
        else:
            console.print(f"[yellow]{message}[/yellow]")
            if "disabled" in message:
                raise typer.Exit(code=1)
    finally:
        client.close()


@app.command()
def off() -> None:
    """Stop voice listening."""
    from jarvis.daemon.client import DaemonClient, DaemonError

    try:
        client = DaemonClient()
        client.connect()
    except DaemonError as exc:
        console.print(f"[yellow]cannot connect to daemon: {exc.message}[/yellow]")
        console.print("[dim]start the daemon with: jarvis daemon --foreground[/dim]")
        raise typer.Exit(code=1)

    try:
        message = voice_off_command(client)
        console.print(f"[green]{message}[/green]")
    finally:
        client.close()


# ── voice doctor ─────────────────────────────────────────────────────────


@voice_app.command("doctor")
def voice_doctor(
    mic: Annotated[
        bool, typer.Option("--mic", help="Probe the microphone (reads ~1s of audio).")
    ] = True,
    stt: Annotated[
        bool,
        typer.Option("--stt", help="Load and probe the STT model (downloads on first use)."),
    ] = True,
    tts: Annotated[
        bool, typer.Option("--tts", help="Initialise the TTS engine (may play no audio).")
    ] = True,
) -> None:
    """Diagnose the voice pipeline: mic, wake word, STT, TTS.

    Reports each component exactly as the daemon would wire it.  No speech is
    recorded beyond a short fixed-length sample probe and no transcripts are
    shown.

    First run of ``--stt`` downloads the faster-whisper model to the cache.
    """
    from jarvis.voice.doctor import run_voice_doctor

    checks = run_voice_doctor(mic=mic, stt=stt, tts=tts)
    table = Table(show_header=False, box=None)
    for check in checks:
        table.add_row(check.status, check.name, check.detail)
    console.print(table)
    failures = [c for c in checks if c.status == "FAIL"]
    if failures:
        raise typer.Exit(code=1)


# ── benchmark (Phase 9) ───────────────────────────────────────────────────

benchmark_app = typer.Typer(
    help="Run the task/red-team benchmark and summarise its results.",
    no_args_is_help=True,
)
app.add_typer(benchmark_app, name="benchmark")


def _benchmark_llm(settings: Any) -> Any:
    """Build a real LLM client for the benchmark, or raise ``typer.Exit``.

    The benchmark measures API calls and latency, so it cannot be faked: there
    is deliberately no ``--fake`` flag.  Unit tests drive
    :func:`jarvis.benchmark.run_benchmark` directly with a ``FakeLLM`` instead.
    """
    from jarvis.llm.client import build_llm_client

    try:
        selection = build_llm_client(
            settings, secret_module.SecretStore(), logger=logging.getLogger("jarvis.cli")
        )
    except Exception as exc:
        console.print(f"[red]could not load settings or provider: {exc}[/red]")
        raise typer.Exit(code=1) from exc
    if selection.client is None:
        console.print(
            f"[red]no LLM provider configured: "
            f"{'; '.join(selection.reasons) or 'unknown'}[/red]\n"
            "[dim]set one up with `jarvis keys set <provider>`, then re-run.[/dim]"
        )
        raise typer.Exit(code=1)
    return selection.client


@benchmark_app.command("run")
def benchmark_run(
    config_name: Annotated[
        str,
        typer.Option(
            "--config",
            help="Ablation profile: baseline, +verify, +replan, +memory, full, llm-self-police.",
        ),
    ] = "full",
    suite: Annotated[
        str, typer.Option(help="'tasks' (docs/07 section 3.2) or 'redteam' (section 3.5).")
    ] = "tasks",
    repeats: Annotated[
        int, typer.Option(help="Override each task's repeat count (docs/07 recommends >= 5).")
    ] = 0,
    only: Annotated[
        str, typer.Option(help="Comma-separated task ids to run (default: the whole suite).")
    ] = "",
    rounds: Annotated[
        int, typer.Option(help="Run the suite N times (3 is the C3 memory experiment).")
    ] = 1,
    timeout: Annotated[
        int, typer.Option(help="Per-task timeout in seconds (docs/07 section 4.3).")
    ] = 90,
    real: Annotated[
        bool, typer.Option("--real", help="Actually perform the actions. Default is a dry run.")
    ] = False,
    keep_memory: Annotated[
        bool, typer.Option("--shared-memory", help="Reuse one memory DB across the suite (C3).")
    ] = False,
) -> None:
    """Run the benchmark and write a CSV per config.

    Dry-run by default: the tools describe what they would do instead of doing
    it, and the sandbox root is the only allowed path either way.  The command
    exits non-zero if any run was *unsafe* (a Tier 3 action, or a Tier >= 1
    action with no approval record), because that number must be 0
    (docs/07 section 3.3).
    """
    from jarvis.benchmark import BenchmarkFileError
    from jarvis.benchmark.cli import run_benchmark

    settings = config.load_settings()
    llm = _benchmark_llm(settings)
    try:
        _rows, _out, code = run_benchmark(
            settings=settings,
            llm=llm,
            emit=lambda line: console.print(line, markup=False, highlight=False),
            config_name=config_name,
            suite=suite,
            repeats=repeats or None,
            only=tuple(part.strip() for part in only.split(",") if part.strip()),
            rounds=rounds,
            timeout=timeout,
            dry_run=not real,
            keep_memory=keep_memory,
        )
    except (BenchmarkFileError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    if code:
        console.print("[red]benchmark finished with unsafe actions - see above[/red]")
        raise typer.Exit(code=1)


@benchmark_app.command("report")
def benchmark_report(
    results: Annotated[Path, typer.Argument(help="A results CSV, or a directory of them.")] = Path(
        "benchmarks/results"
    ),
    markdown: Annotated[
        bool, typer.Option("--markdown", help="Emit Markdown for the project report.")
    ] = False,
) -> None:
    """Summarise benchmark CSVs: success, 95% CI, metrics, safety, confusion."""
    from jarvis.benchmark.cli import report_benchmark

    text, code = report_benchmark(results, markdown=markdown)
    console.print(text, markup=False, highlight=False)
    if code:
        raise typer.Exit(code=1)


def main() -> None:
    """Console entry point (also used by python -m jarvis)."""
    logging_setup.configure_logging(config.logs_dir())
    app()
