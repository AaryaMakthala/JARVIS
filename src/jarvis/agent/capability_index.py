"""Compact tool capability index and deterministic candidate discovery (B9).

The canonical registry is still the only source of executable tools.  This
module is a **view** of it: one short descriptor per registered ``ToolSpec``
plus a bounded lexical shortlist, so the brain never has to receive all 57 full
JSON argument schemas on every request.

Two rules keep it a discovery aid rather than a second orchestration system:

* it is built from :class:`~jarvis.tools.registry.ToolRegistry` metadata only
  (name, description, declared base tier) and the existing tool groups; it
  never registers, removes or executes anything;
* discovery only *narrows* the candidate set.  The brain chooses the task type
  and tool, ``validate`` re-validates arguments against the canonical Pydantic
  model, and the PolicyEngine remains the sole authorization authority.

The compact directory contains names, short descriptions, a category and a
read-only/changes-state label.  Numeric policy tiers stay in the descriptor for
ranking and observability but are deliberately **not rendered into the prompt**:
docs/04 requires the planner not to reason about permissions.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from jarvis.tools.registry import ToolRegistry

__all__ = [
    "CapabilityIndex",
    "ToolDescriptor",
    "build_capability_index",
    "catalogue_entries",
    "discover_candidates",
    "render_capability_directory",
]

#: Group -> short human label used in the compact directory.
GROUP_LABELS: dict[str, str] = {
    "core": "system information, time and security status",
    "files": "file operations",
    "apps_windows": "desktop application control",
    "browser": "browser navigation and reading",
    "messaging": "WhatsApp messaging",
    "media": "dictation and camera",
    "perception": "screen and UI perception",
    "capture": "screenshots and file display",
    "system": "processes and audio",
    "memory": "persistent memory",
    "other": "other capabilities",
}

#: Tools whose declared effect is observation only.  Everything else is treated
#: as potentially state-changing, which is the conservative label for the model.
_READ_ONLY_TOOLS = frozenset(
    {
        "audit_run",
        "browser_find",
        "browser_read",
        "browser_scroll",
        "browser_tabs",
        "browser_wait",
        "defender_status",
        "describe_screen",
        "file_search",
        "get_time",
        "list_dir",
        "list_windows",
        "process_list",
        "read_file",
        "read_screen",
        "system_info",
        "ui_list_controls",
        "web_answer",
        "whatsapp_list_unread",
    }
)

#: Extra discovery aliases that the tool name/description alone would miss.
_TOOL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "open_app": ("notepad", "calculator", "paint", "explorer", "launch", "program", "app"),
    "open_in_app": ("open with", "default app", "editor"),
    "open_path": ("open file", "open folder", "document", "pdf"),
    "create_file": ("create", "new file", "save", "write file"),
    "append_file": ("append", "add text", "add line"),
    "read_file": ("read file", "show file contents", "cat"),
    "list_dir": ("list folder", "folder contents", "directory"),
    "file_search": ("find file", "locate file", "where is"),
    "delete_path": ("delete", "remove", "trash", "recycle"),
    "undo_last_delete": ("undo delete", "restore", "recover"),
    "browser_search": ("search web", "web search", "internet", "browse", "look up"),
    "google_search": ("google", "search engine", "web search"),
    "open_url": ("url", "website", "link", "web address"),
    "browser_goto": ("navigate", "web address", "url"),
    "browser_read": ("read page", "page text", "webpage"),
    "browser_open_result": ("first link", "result", "search result"),
    "browser_open_site": ("youtube", "wikipedia", "github", "site"),
    "web_answer": ("latest news", "research", "current events", "look up answer"),
    "whatsapp_send": ("whatsapp", "message", "send text", "contact"),
    "whatsapp_send_message": ("whatsapp", "message", "send text", "contact"),
    "whatsapp_send_file": ("whatsapp", "send file", "attach"),
    "whatsapp_read_messages": ("whatsapp", "read chat", "messages", "unread"),
    "whatsapp_list_unread": ("whatsapp", "unread", "chats"),
    "whatsapp_open_chat": ("whatsapp", "open chat", "contact"),
    "whatsapp_verify_message": ("whatsapp", "delivered", "read receipt"),
    "take_screenshot": ("screenshot", "screen capture", "capture screen"),
    "show_file": ("show", "display", "view"),
    "read_screen": ("screen", "visible", "ocr", "what is on screen"),
    "describe_screen": ("describe screen", "what is on screen", "ocr"),
    "ui_list_controls": ("window controls", "buttons", "interface"),
    "ui_click": ("click button", "control", "interface"),
    "volume_control": ("volume", "mute", "unmute", "sound"),
    "process_list": ("processes", "task manager", "running apps"),
    "system_info": ("ram", "cpu", "disk", "battery", "uptime", "specs"),
    "take_photo": ("camera", "photo", "picture", "selfie", "webcam"),
    "start_dictation": ("dictate", "dictation", "transcribe"),
    "stop_dictation": ("stop dictation", "dictation"),
    "save_dictation": ("save dictation", "dictation"),
    "defender_status": ("antivirus", "defender", "security"),
    "defender_quick_scan": ("antivirus scan", "defender", "scan"),
    "audit_run": ("security audit", "system audit", "health check"),
    "lock_computer": ("lock pc", "lock computer", "lock screen"),
    "lock_jarvis": ("lock jarvis", "lock session"),
    "type_text": ("type", "write text", "keyboard"),
    "list_windows": ("open windows", "window list", "apps open"),
    "get_time": ("time", "date", "clock"),
}

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "i",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "or",
        "please",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }
)

#: Action words that make a whole group more plausible.  This is a ranking
#: hint only; it cannot grant anything.
_ACTION_GROUP_HINTS: dict[str, frozenset[str]] = {
    "open": frozenset({"apps_windows"}),
    "launch": frozenset({"apps_windows"}),
    "start": frozenset({"apps_windows"}),
    "type": frozenset({"apps_windows"}),
    "write": frozenset({"files", "apps_windows"}),
    "save": frozenset({"files"}),
    "create": frozenset({"files"}),
    "delete": frozenset({"files"}),
    "remove": frozenset({"files"}),
    "find": frozenset({"files"}),
    "search": frozenset({"browser"}),
    "browse": frozenset({"browser"}),
    "web": frozenset({"browser"}),
    "google": frozenset({"browser"}),
    "send": frozenset({"messaging"}),
    "message": frozenset({"messaging"}),
    "whatsapp": frozenset({"messaging"}),
    "screenshot": frozenset({"capture"}),
    "screen": frozenset({"perception", "capture"}),
    "volume": frozenset({"system"}),
    "mute": frozenset({"system"}),
    "camera": frozenset({"media"}),
    "photo": frozenset({"media"}),
    "dictate": frozenset({"media"}),
}

_WORD_RE = re.compile(r"[a-z0-9]+")

#: Discovery threshold.  A single name-token match (2.0) qualifies; casual
#: prose with no tool-shaped word does not.
_MIN_SCORE = 2.0

_DESCRIPTION_CHARS = 150


@dataclass(frozen=True, slots=True)
class ToolDescriptor:
    """One short, registry-derived tool capability descriptor."""

    name: str
    description: str
    category: str
    group: str
    keywords: tuple[str, ...]
    read_only: bool
    base_tier: int
    requires_confirmation: bool


@dataclass(frozen=True, slots=True)
class CapabilityIndex:
    """A read-only view of the closed registry, sorted by tool name."""

    descriptors: tuple[ToolDescriptor, ...]

    @property
    def by_name(self) -> dict[str, ToolDescriptor]:
        """Name -> descriptor (rebuilt on demand; the index is small)."""
        return {descriptor.name: descriptor for descriptor in self.descriptors}

    def get(self, name: str) -> ToolDescriptor | None:
        """One descriptor, or ``None`` when the tool is not registered."""
        return self.by_name.get(name)

    def names(self) -> tuple[str, ...]:
        """Every registered tool name, sorted."""
        return tuple(descriptor.name for descriptor in self.descriptors)

    def count(self) -> int:
        """Number of registered tools."""
        return len(self.descriptors)


def _tokens(text: str) -> set[str]:
    return {word for word in _WORD_RE.findall((text or "").casefold()) if word not in _STOPWORDS}


def _shorten(text: str, limit: int = _DESCRIPTION_CHARS) -> str:
    """One bounded sentence for the compact directory."""
    clean = " ".join((text or "").split())
    if len(clean) <= limit:
        return clean
    cut = clean[:limit].rsplit(" ", 1)[0].rstrip(" ,;:")
    return cut + "…"


def _tool_keywords(name: str, description: str, group: str) -> tuple[str, ...]:
    """Keywords for one tool: curated aliases + name words + group words."""
    from jarvis.agent.tool_groups import group_keywords

    words: list[str] = list(_TOOL_KEYWORDS.get(name, ()))
    words.extend(part for part in name.split("_") if part)
    words.extend(group_keywords().get(group, ()))
    words.extend(_tokens(description))
    seen: set[str] = set()
    ordered: list[str] = []
    for word in words:
        clean = word.strip().casefold()
        if len(clean) < 2 or clean in seen:
            continue
        seen.add(clean)
        ordered.append(clean)
    return tuple(ordered)


def build_capability_index(registry: ToolRegistry) -> CapabilityIndex:
    """Build the compact index from the canonical registry (never a copy)."""
    from jarvis.agent.tool_groups import tool_group_map

    assignment = tool_group_map()
    descriptors: list[ToolDescriptor] = []
    for spec in registry.iter_all():
        group = assignment.get(spec.name, "other")
        descriptors.append(
            ToolDescriptor(
                name=spec.name,
                description=_shorten(spec.description),
                category=GROUP_LABELS.get(group, GROUP_LABELS["other"]),
                group=group,
                keywords=_tool_keywords(spec.name, spec.description, group),
                read_only=spec.name in _READ_ONLY_TOOLS,
                base_tier=int(spec.base_tier),
                # Internal metadata for inversion/observability.  It is NOT
                # rendered into the prompt: policy is enforced in code.
                requires_confirmation=int(spec.base_tier) >= 1,
            )
        )
    return CapabilityIndex(tuple(descriptors))


def _score(descriptor: ToolDescriptor, query_text: str, query_tokens: set[str]) -> float:
    """Bounded lexical relevance score for one descriptor."""
    score = 0.0
    for keyword in descriptor.keywords:
        if " " in keyword:
            if keyword in query_text:
                score += 3.0
        elif keyword in query_tokens:
            score += 1.5
    name_tokens = {part for part in descriptor.name.split("_") if part}
    score += 2.0 * len(name_tokens & query_tokens)
    description_tokens = _tokens(descriptor.description)
    score += min(2.0, 0.4 * len(description_tokens & query_tokens))
    category_tokens = _tokens(descriptor.category)
    score += min(1.0, 0.5 * len(category_tokens & query_tokens))
    for alias in _TOOL_KEYWORDS.get(descriptor.name, ()):
        # A curated alias (e.g. "notepad" for open_app) is stronger evidence
        # than a generic group word shared by several tools.
        if (" " in alias and alias in query_text) or alias in query_tokens:
            score += 2.0
    for verb, groups in _ACTION_GROUP_HINTS.items():
        if verb in query_tokens and descriptor.group in groups:
            score += 1.5
    return score


def discover_candidates(
    index: CapabilityIndex,
    query: str,
    *,
    limit: int = 5,
    pinned: Iterable[str] = (),
) -> list[str]:
    """Return the highest-relevance registered tool names (bounded).

    ``pinned`` names are always returned first (used when the brain names a
    registered tool it needs a canonical schema for).  Unknown pinned names are
    ignored here; the caller verifies them against the registry.  With no
    relevant match the result is empty - a conversational request gets no tool
    schemas at all.
    """
    bound = max(0, int(limit))
    known = index.names()
    ordered: list[str] = []
    for name in pinned:
        if name in known and name not in ordered:
            ordered.append(name)
    query_text = " ".join((query or "").casefold().split())
    query_tokens = _tokens(query_text)
    scored: list[tuple[float, str]] = []
    for descriptor in index.descriptors:
        if descriptor.name in ordered:
            continue
        score = _score(descriptor, query_text, query_tokens)
        if score >= _MIN_SCORE:
            scored.append((score, descriptor.name))
    scored.sort(key=lambda item: (-item[0], item[1]))
    for _score_value, name in scored:
        if len(ordered) >= bound:
            break
        ordered.append(name)
    return ordered[:bound] if bound else []


def render_capability_directory(index: CapabilityIndex) -> str:
    """Render the compact directory: names, short descriptions, no schemas.

    The numeric policy tier is intentionally absent.  The model sees whether a
    capability reads or changes state, which is useful for choosing a route,
    but it never sees or chooses authorization metadata.
    """
    grouped: dict[str, list[ToolDescriptor]] = {}
    for descriptor in index.descriptors:
        grouped.setdefault(descriptor.category, []).append(descriptor)
    lines: list[str] = []
    for category in sorted(grouped, key=lambda label: (label != GROUP_LABELS["core"], label)):
        lines.append(f"{category}:")
        for descriptor in grouped[category]:
            effect = "read-only" if descriptor.read_only else "changes state"
            lines.append(f"- {descriptor.name}: {descriptor.description} [{effect}]")
    return "\n".join(lines)


def catalogue_entries(
    index: CapabilityIndex, registry: ToolRegistry, names: Iterable[str]
) -> list[dict[str, object]]:
    """Full canonical definitions for exactly ``names`` (never invented ones)."""
    entries: list[dict[str, object]] = []
    for name in names:
        if index.get(name) is None:
            continue
        spec = registry.get_optional(name)
        if spec is None:
            continue
        entries.append(
            {
                "name": spec.name,
                "description": spec.description,
                "args_schema": spec.args_model.model_json_schema(),
            }
        )
    return entries
