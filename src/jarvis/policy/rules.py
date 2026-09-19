"""Declarative policy rules: tool/arg patterns that raise or lower tiers.

Only the deterministic PolicyEngine reads these numbers.  Rules may only raise
a step's tier (or block it outright); nothing here can be influenced by LLM
output or untrusted content (docs/03_SECURITY_AND_POLICY.md sections 3-4).

Phase 1 scope: hard-block names, URL-scheme allowlist, unknown app names, and
an overwrite-warning rule for ``create_file``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

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

_URL_MAX = 2083


def matches_blocked(spec: ToolSpec, args: Any) -> bool:
    """Return ``True`` when this exact (tool, args) pair is hard-blocked.

    A tool name containing a blocked fragment is refused regardless of args.
    """
    name = spec.name.lower()
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


def tool_tier(spec: ToolSpec, args: Any) -> int:
    """Return the extra tier a tool-specific rule imposes (may raise only)."""
    if spec.name == "open_app":
        # The allowlist lives in settings.apps; an unknown app is refused in
        # run() but the tier itself stays at base (0) - harmless either way.
        return tiers.TIER_SAFE
    if spec.name in ("open_url", "google_search"):
        return tiers.TIER_SAFE
    return tiers.TIER_SAFE


def path_tier(spec: ToolSpec, resolved_path: str, args: Any) -> int:
    """Return any extra tier for a resolved path (v1: overwrite rule)."""
    if spec.name == "create_file" and getattr(args, "overwrite", False):
        return tiers.TIER_CONFIRM  # overwrite only ever needs Tier 1 here
    return tiers.TIER_SAFE


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
