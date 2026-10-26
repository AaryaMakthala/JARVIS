"""Tool groups: send only the tools a request can plausibly need.

The brain prompt is dominated by the tool catalogue - measured on this registry
it is ~31 KB of a ~35 KB prompt for the request "open notepad", while the
instructions are ~3.5 KB.  Every tool's JSON schema is paid for on every call,
including the seventeen browser tools for a request that never touches a
browser.

This module groups the closed registry and picks the groups a request needs
**deterministically**, from the user's own words and from state facts the graph
already computed.  It is prompt shaping only:

* it never sees, reads or encodes a tier, a policy decision or a tool result;
* it cannot grant anything, because the registry stays the closed allowlist and
  ``validate`` / ``policy_gate`` / ``act`` still judge every step;
* it **fails open**: an unassigned tool, or a request that matches no group
  beyond ``core``, sends the full catalogue - exactly today's behaviour.

The selector is only used when ``agent.prompt_trim_enabled`` is true (default
false), so an install that does not opt in builds a byte-identical prompt.
"""

from __future__ import annotations

import urllib.parse
from typing import TYPE_CHECKING, Any

from jarvis.browser.sites import SiteDirectory, render_site_catalogue

if TYPE_CHECKING:  # pragma: no cover - typing only
    from jarvis.browser.sites import Site
    from jarvis.tools.registry import ToolRegistry

__all__ = [
    "ALWAYS_GROUPS",
    "DEFAULT_SITE_NAMES",
    "GROUPS",
    "MAX_TRIMMED_SITES",
    "groups_for",
    "select_catalogue",
    "tool_group_map",
    "trim_site_catalogue",
]

#: Sent on every request regardless of triggers (cheap, broadly useful tools).
ALWAYS_GROUPS: tuple[str, ...] = ("core",)

#: Every registered tool in exactly one group.  ``core`` is always sent; the
#: others join it when a trigger below matches.  Groups may be empty.
GROUPS: dict[str, tuple[str, ...]] = {
    "core": (
        "get_time",
        "system_info",
        "audit_run",
        "defender_status",
        "defender_quick_scan",
        "web_answer",
    ),
    "files": (
        "append_file",
        "create_file",
        "delete_path",
        "file_search",
        "list_dir",
        "read_file",
        "undo_last_delete",
    ),
    "apps_windows": (
        "open_app",
        "open_in_app",
        "open_path",
        "type_text",
        "list_windows",
        "lock_computer",
        "lock_jarvis",
    ),
    "browser": (
        "browser_back",
        "browser_click",
        "browser_find",
        "browser_forward",
        "browser_goto",
        "browser_open_result",
        "browser_open_site",
        "browser_press_key",
        "browser_read",
        "browser_screenshot",
        "browser_scroll",
        "browser_search",
        "browser_tabs",
        "browser_type",
        "browser_wait",
        "google_search",
        "open_url",
    ),
    "messaging": (
        "whatsapp_open_chat",
        "whatsapp_send",
        "whatsapp_send_file",
    ),
    "media": (
        "start_dictation",
        "stop_dictation",
        "save_dictation",
        "take_photo",
    ),
    "perception": (
        "read_screen",
        "describe_screen",
        "ui_list_controls",
        "ui_click",
    ),
    "capture": (
        "take_screenshot",
        "show_file",
    ),
    "system": (
        "process_list",
        "volume_control",
    ),
    "memory": (),
}

#: Lower-case substrings that pull a group in from the user's own words.  Plain
#: words only - no tiers, no policy terms.
_GROUP_KEYWORDS: dict[str, tuple[str, ...]] = {
    "files": (
        "file",
        "folder",
        "directory",
        "drive",
        ".txt",
        ".pdf",
        ".doc",
        ".csv",
        ".png",
        ".jpg",
        ".json",
        "save as",
        "delete",
        "rename",
        "find",
        "path",
    ),
    "apps_windows": (
        "open",
        "launch",
        "start ",
        "app",
        "window",
        "notepad",
        "calculator",
        "explorer",
        "paint",
        "type ",
        "press",
        "key",
        "lock",
        "close",
        "minimi",
        "maximi",
        "focus",
    ),
    "browser": (
        "browser",
        "web",
        "website",
        "site",
        "http",
        ".com",
        ".org",
        ".net",
        "google",
        "search",
        "page",
        "tab",
        "link",
        "youtube",
        "wikipedia",
        "github",
        "reddit",
        "amazon",
    ),
    "messaging": (
        "whatsapp",
        "message",
        "text ",
        "chat",
        "send",
        "contact",
    ),
    "media": (
        "photo",
        "picture",
        "camera",
        "webcam",
        "selfie",
        "dictation",
        "dictate",
        "transcribe",
        "voice note",
    ),
    "perception": (
        "screen",
        "look",
        "see",
        "visible",
        "click",
        "button",
        "control",
        "interface",
        "window",
    ),
    "capture": (
        "screenshot",
        "show ",
        "display ",
        "view ",
    ),
    "system": (
        "volume",
        "mute",
        "unmute",
        "sound",
        "process",
        "processes",
        "task manager",
        "cpu",
        "memory",
    ),
}

#: Sites always kept in the trimmed site catalogue (a short, useful default).
DEFAULT_SITE_NAMES: tuple[str, ...] = ("Google", "YouTube", "Wikipedia", "GitHub")

#: Hard cap on sites rendered into a trimmed prompt.
MAX_TRIMMED_SITES = 10

#: Longest stretch of the user's text scanned for keywords.
_MAX_TEXT_CHARS = 500


def tool_group_map() -> dict[str, str]:
    """``tool name -> group`` for every assignment (the reverse of :data:`GROUPS`)."""
    return {name: group for group, names in GROUPS.items() for name in names}


def groups_for(state: dict[str, Any]) -> set[str]:
    """The group names to send for this request (always includes ``core``)."""
    text = _request_text(state).casefold()
    selected: set[str] = set(ALWAYS_GROUPS)
    for group, keywords in _GROUP_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            selected.add(group)

    # State facts the graph already computed - no guessing from prose.
    refs = state.get("resolved_refs")
    if isinstance(refs, dict) and (refs.get("has_page") or refs.get("has_result_set")):
        selected.add("browser")
    if str(state.get("specialist_goal") or "").strip():
        selected.add("browser")
    if str(state.get("request_kind") or "") == "browse":
        selected.add("browser")
    return selected


def select_catalogue(
    registry: ToolRegistry, state: dict[str, Any]
) -> list[dict[str, object]]:
    """The tool catalogue for this request, or the full one (fail open).

    Full catalogue when a registered tool belongs to no group (the registry grew
    and this table did not), when nothing beyond ``core`` was triggered, or when
    the filter would leave nothing.
    """
    full = registry.catalogue_for_llm()
    assignment = tool_group_map()
    names = [str(entry.get("name") or "") for entry in full]
    if any(name and name not in assignment for name in names):
        return full

    groups = groups_for(state)
    if groups <= set(ALWAYS_GROUPS):
        return full

    keep = {name for group in groups for name in GROUPS.get(group, ())}
    selected = [entry for entry in full if str(entry.get("name") or "") in keep]
    return selected or full


def trim_site_catalogue(directory: SiteDirectory, state: dict[str, Any]) -> str:
    """Render only the sites this request can use, plus a short default list.

    A site is kept when the user's words name it (or one of its aliases) or when
    it is the host of the page already open; :data:`DEFAULT_SITE_NAMES` fills the
    rest up to :data:`MAX_TRIMMED_SITES`.  Rendering stays
    :func:`~jarvis.browser.sites.render_site_catalogue` - names and hosts only,
    so the model still copies no URL.
    """
    text = _request_text(state).casefold()
    host = _active_page_host(state)
    kept: list[Site] = []
    seen: set[str] = set()

    def _add(site: Site | None) -> None:
        if site is None or site.name in seen or len(kept) >= MAX_TRIMMED_SITES:
            return
        kept.append(site)
        seen.add(site.name)

    for site in directory.sites:
        keys = (site.name, *site.aliases)
        if (host and site.host == host) or any(key.casefold() in text for key in keys):
            _add(site)
    for name in DEFAULT_SITE_NAMES:
        _add(directory.resolve(name))
    return render_site_catalogue(SiteDirectory(kept))


def _request_text(state: dict[str, Any]) -> str:
    """The user's words for this request (input plus any clarification answer)."""
    parts = [str(state.get("user_input") or ""), str(state.get("clarification_answer") or "")]
    return " ".join(part for part in parts if part)[:_MAX_TEXT_CHARS]


def _active_page_host(state: dict[str, Any]) -> str:
    """Host of the page already open, or ``""`` (plain state read, no I/O)."""
    refs = state.get("resolved_refs")
    if not isinstance(refs, dict):
        return ""
    url = refs.get("page_url")
    if not isinstance(url, str) or not url:
        return ""
    try:
        return (urllib.parse.urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""
