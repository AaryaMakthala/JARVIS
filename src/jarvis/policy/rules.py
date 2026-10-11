"""Declarative policy rules: tool/arg patterns that raise or lower tiers.

Only the deterministic PolicyEngine reads these numbers.  Rules may only raise
a step's tier (or block it outright); nothing here can be influenced by LLM
output or untrusted content (docs/03_SECURITY_AND_POLICY.md sections 3-4).

Phase 1 scope: hard-block names, URL-scheme allowlist, unknown app names, and
an overwrite-warning rule for ``create_file``.

Phase 2 adds: an exact-name exemption so the sanctioned ``delete_path`` and
``undo_last_delete`` tools are not caught by the ``delete``/``rm`` name
fragments below, and the typed folder-name confirmation rule for deletes.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from collections.abc import Generator, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from jarvis.config import Settings
from jarvis.policy import tiers
from jarvis.tools.base import ToolResult, ToolSpec
from jarvis.tools.web import is_allowed_http_url

#: Tool *names* (or plain-English misspellings of them) that are always
#: blocked even if someone tried to register them under a variation.
BLOCKED_NAME_FRAGMENTS = (
    "disable_defender",
    "defender",
    "firewall",
    "bitlocker",
    "disable_",
    "enable_",
    "uninstall_",
    "run_powershell",
    "shell",
    "powershell",
    "registry",
    "manage_password",
    "browser_password",
    "cookie",
    "payment",
    "transfer",
    "purchase",
    "shutdown",
    "delete",
    "rm",
)

#: The ONLY tool names allowed to contain a ``BLOCKED_NAME_FRAGMENT``.  These
#: are the deliberately destructive tools, each exhaustively hardened and
#: tested (Recycle-Bin only + undo log + Tier 2 + typed confirmation for
#: folders).  Any other future tool named ``*delete*`` stays hard-blocked.
_SANCTIONED_TOOLS = frozenset({"delete_path", "undo_last_delete"})

_URL_MAX = 2083


def matches_blocked(spec: ToolSpec, args: Any) -> bool:
    """Return ``True`` when this exact (tool, args) pair is hard-blocked.

    A tool name containing a blocked fragment is refused regardless of args -
    except for the few names in :data:`_SANCTIONED_TOOLS`.
    """
    name = spec.name.lower()
    if name in _SANCTIONED_TOOLS:
        return False
    return any(fragment in name for fragment in BLOCKED_NAME_FRAGMENTS)


def url_is_blocked(url: str) -> str | None:
    """Return a reason when ``url`` must not be opened (else ``None``)."""
    if not url:
        return "empty URL"
    if len(url) > _URL_MAX:
        return f"URL longer than {_URL_MAX} chars"
    if not is_allowed_http_url(url):
        return "only http:// and https:// URLs may be opened"
    return None


# ── Stage 7 browser domain / origin primitives (D18 / B4 / I5) ────────────
#
# Pure, side-effect-free helpers shared by the policy engine and the browser
# tools.  They never read settings, user input, the network or the DOM; the
# engine supplies the inputs.  ``url_origin`` is the single canonicalizer used
# everywhere an origin is compared, so a redirect cannot escape the approved
# scope by spelling the same host differently (I5).

_DEFAULT_PORTS = {"http": 80, "https": 443}
_ORIGIN_SCHEMES = frozenset(_DEFAULT_PORTS)


def _canonical_host(raw_host: str | None) -> str | None:
    """Return a canonical host (lower, no trailing dot, IDNA/IP-normalised).

    ``None`` means the host cannot be trusted for an origin comparison.  Pure
    and stdlib-only; a unicode name is encoded to its IDNA A-label so
    ``bücher.example`` and ``xn--bcher-kva.example`` compare equal.
    """
    if not raw_host:
        return None
    host = raw_host.strip().lower().rstrip(".")
    if not host:
        return None
    if any(ch.isspace() for ch in host) or any(ch in host for ch in "/\\@?#"):
        return None
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        return str(ip)
    try:
        ascii_host = host.encode("idna").decode("ascii")
    except (UnicodeError, ValueError):
        return None
    return ascii_host or None


def _canonical_origin(url: str) -> tuple[str, str] | None:
    """Return ``(origin, host)`` for an http/https URL, else ``None``.

    Rejects userinfo, malformed URLs, invalid/zero ports and any scheme other
    than http/https.  The origin omits the port when it is the scheme default.
    """
    if not isinstance(url, str) or not url:
        return None
    if any(ch in url for ch in " \t\r\n\\"):
        return None
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    if scheme not in _ORIGIN_SCHEMES:
        return None
    netloc = parts.netloc
    if netloc.endswith(":"):
        return None
    if "@" in netloc:
        return None
    host = _canonical_host(parts.hostname)
    if host is None:
        return None
    try:
        port = parts.port
    except ValueError:
        return None
    if port is not None and not 0 < port < 65536:
        return None
    default = _DEFAULT_PORTS[scheme]
    host_display = f"[{host}]" if ":" in host else host
    if port is None or port == default:
        return f"{scheme}://{host_display}", host
    return f"{scheme}://{host_display}:{port}", host


def url_origin(url: str) -> str | None:
    """Return the stable canonical origin of an http/https URL, else ``None``.

    Normalises hostname case, trailing dots, default ports and IDNs (stdlib
    IDNA).  Rejects userinfo, malformed URLs, invalid ports and unsupported
    schemes.  Two spellings of the same origin always yield one string.
    """
    result = _canonical_origin(url)
    return result[0] if result is not None else None


def domain_allowed(url: str, allowed_domains: Sequence[str]) -> bool:
    """Whether ``url``'s host is explicitly in ``allowed_domains`` (B4/D18).

    Exact host match only - a subdomain or sibling is **not** implicitly
    allowed, and an invalid URL or empty allowlist fails closed.  Entries are
    canonicalised the same way as the URL host, so case/IDN/trailing-dot
    variations match.
    """
    result = _canonical_origin(url)
    if result is None:
        return False
    _, host = result
    wanted = {canon for entry in allowed_domains if (canon := _canonical_host(entry))}
    if not wanted:
        return False
    return host in wanted


# ── Stage 7 risky-control classification (D12 / I8) ───────────────────────
#
# Token-aware, never substring: "sender" / "allowance" / "confirmation" are not
# the risky controls, while "Send message" / "Allow" / "Buy now" are.  Payment
# and purchase controls are a distinct, stricter class (Tier 3) from the
# controls that merely re-gate with a fresh confirmation.

CONTROL_FRESH_CONFIRM = "confirm"
CONTROL_PAYMENT = "payment"

_RISKY_PAYMENT_TOKENS = frozenset({"buy", "purchase", "pay", "payment", "payments"})
_RISKY_CONFIRM_TOKENS = frozenset({"send", "submit", "delete", "confirm", "allow"})
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def risky_control(name: str, role: str = "") -> str | None:
    """Classify an accessible control name/role for Stage 7 re-gating.

    Returns :data:`CONTROL_PAYMENT` for payment/purchase controls (Tier 3),
    :data:`CONTROL_FRESH_CONFIRM` for controls that must re-gate individually
    even inside an approved scope, and ``None`` when the control is not risky.
    Matching is token-based (case/whitespace/punctuation insensitive) so it
    does not fire on words that merely contain a risky fragment.
    """
    tokens = set(_TOKEN_RE.findall(f"{name or ''} {role or ''}".lower()))
    if tokens & _RISKY_PAYMENT_TOKENS:
        return CONTROL_PAYMENT
    if tokens & _RISKY_CONFIRM_TOKENS:
        return CONTROL_FRESH_CONFIRM
    return None


def _field_getter(target: Any) -> Any:
    """Return ``get(name)`` for a mapping or ``getattr(target, name, None)``."""
    if isinstance(target, Mapping):
        return lambda name: target.get(name)
    return lambda name: getattr(target, name, None)


def _norm_field(value: Any) -> str:
    return str(value or "").strip().lower()


def is_password_target(target: Any) -> bool:
    """Whether an interaction target is a password field (must never be typed).

    Recognises DOM ``input[type=password]`` (``input_type``/``role``) and UIA
    ``IsPassword`` controls (``is_password``/``control_type``).  Fails closed:
    an uninspectable target, or one that exposes no recognised descriptor, is
    treated as a password field.  Only type metadata is read - never a value.
    """
    if target is None:
        return True
    get = _field_getter(target)
    explicit = get("is_password")
    input_type = _norm_field(get("input_type"))
    role = _norm_field(get("role"))
    control_type = _norm_field(get("control_type"))
    if explicit is True:
        return True
    if input_type in {"password", "passwd"}:
        return True
    if role in {"password", "passwordfield", "passwordtextbox"}:
        return True
    if control_type in {"password", "passwordedit", "editpassword", "passwordbox"}:
        return True
    inspected = explicit is not None or bool(input_type or role or control_type)
    return not inspected


def args_overlap_taint(args: Any, fragments: tuple[str, ...], min_len: int = 12) -> bool:
    """Deterministic taint check: do any arg values overlap with tainted text?

    Per docs/03_SECURITY_AND_POLICY.md §8, a step is tainted when its args
    contain substrings (≥ ``min_len`` chars) of text from a previous tainted
    result.  This check is independent of ``step.depends_on_untrusted`` (which
    is LLM-controlled and therefore untrusted itself).  The engine uses this
    to *raise* taint even when the LLM omits the flag.
    """
    if not fragments:
        return False
    for val in _iter_arg_strings(args):
        for frag in fragments:
            if len(frag) >= min_len and val in frag:
                return True
    return False


def _iter_arg_strings(args: Any) -> Generator[str]:
    """Yield every string leaf value from an args object or dict."""
    if isinstance(args, dict):
        items = args.values()
    elif hasattr(args, "model_dump"):
        items = args.model_dump(mode="json").values()
    else:
        return
    for v in items:
        if isinstance(v, str):
            yield v
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, str):
                    yield item


def tool_tier(spec: ToolSpec, args: Any) -> int:
    """Return the extra tier a tool-specific rule imposes (may raise only)."""
    if spec.name == "open_app":
        # The allowlist lives in settings.apps; an unknown app is refused in
        # run() but the tier itself stays at base (0) - harmless either way.
        return tiers.TIER_SAFE
    if spec.name in ("open_url", "google_search", "web_answer"):
        return tiers.TIER_SAFE
    return tiers.TIER_SAFE


def path_tier(spec: ToolSpec, resolved_path: str, args: Any) -> int:
    """Return any extra tier for a resolved path."""
    if spec.name == "create_file" and getattr(args, "overwrite", False):
        return tiers.TIER_CONFIRM  # overwrite only ever needs Tier 1 here
    if spec.name == "delete_path":
        # Deleting a folder is Tier 2 regardless (base) and the engine adds a
        # typed folder-name confirmation (typed_confirmation()).  A plain file
        # delete stays at base_tier 2.
        return tiers.TIER_CONFIRM
    return tiers.TIER_SAFE


def typed_confirmation(spec: ToolSpec, resolved_paths: list[str], settings: Settings) -> str | None:
    """Return the text the user must type to confirm folder deletes, else None.

    Folder deletion is the least reversible action we offer, so when enabled
    the user must type the folder names(s) from the summary instead of a bare
    yes/no (docs/04_TOOLS_SPEC.md ``delete_path``).  The confirmation is bound
    to the exact folder names, re-checked on resume by the policy gate.
    """
    if spec.name != "delete_path":
        return None
    if settings is not None and not settings.policy.typed_confirmation_for_folders:
        return None
    names = sorted(Path(p).name for p in resolved_paths if Path(p).is_dir())
    return ", ".join(names) if names else None


def overwrite_warning(spec: ToolSpec, args: Any) -> str | None:
    """Return a warning suffix for ``create_file`` overwriting an existing file."""
    if spec.name != "create_file":
        return None
    if getattr(args, "overwrite", False):
        return "OVERWRITES any existing file at this path"
    return None


def canonical_args(args: Any) -> str:
    """Stable, deterministic JSON rendering of ``args`` used for hashing.

    ``sort_keys=True`` + ``separators`` keeps the hash independent of key
    order and whitespace; ``default=str`` survives values pydantic leaves as
    non-JSON primitives.
    """
    if isinstance(args, ToolResult):
        raw: dict[str, Any] = json.loads(args.model_dump_json())
    elif isinstance(args, dict):
        raw = dict(args)
    else:
        raw = args.model_dump(mode="json")
    return json.dumps(raw, sort_keys=True, separators=(",", ":"), default=str)


def action_hash(spec: ToolSpec, args: Any) -> str:
    """sha256 of (tool name + canonical args); binds approvals to this action."""
    return action_hash_raw(spec.name, args)


def action_hash_raw(tool_name: str, raw_args: Any) -> str:
    """Hash a tool name + raw argument mapping (used for refused/unknown steps)."""
    payload = f"{tool_name}\n{canonical_args(raw_args)}".encode()
    return hashlib.sha256(payload).hexdigest()


def app_commands(settings: Settings) -> dict[str, str]:
    """Return the ``name -> command`` allowlist from settings."""
    data = settings.apps.model_dump()
    commands: dict[str, str] = {}
    for name, command in data.items():
        if isinstance(command, str) and command:
            commands[str(name)] = command
    return commands
