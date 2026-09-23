"""Tool registry: the only way the planner can reach executable tools.

Rules from docs/04_TOOLS_SPEC.md section 1:

* ``register`` fails on duplicate names and on names/descriptions matching a
  forbidden pattern (shell, powershell, cmd, exec, eval);
* ``catalogue_for_llm()`` returns schemas + descriptions and **no tiers** (the
  planner must not reason about permissions);
* ``get`` raises :class:`UnknownTool` for anything not registered.

The planner's tool set is therefore a closed allowlist: an unknown tool name
in a plan is rejected before any code can run.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from jarvis.tools.base import ToolSpec

if TYPE_CHECKING:
    from jarvis.config import Settings

# Word-bounded so ordinary words ('command', 'evaluate') never match.
_FORBIDDEN_RE = re.compile(r"\b(shell|powershell|cmd|exec|eval)\b", re.IGNORECASE)


class UnknownTool(KeyError):
    """Raised when a planner references a tool that does not exist."""


class ToolRegistry:
    """Closed set of executable tools (name -> ToolSpec)."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        """Add a tool, enforcing uniqueness and the forbidden-name rule."""
        name = spec.name
        if name in self._tools:
            raise ValueError(f"duplicate tool registration: {name!r}")
        forbidden = _FORBIDDEN_RE.search(f"{name} {spec.description}")
        if forbidden:
            raise ValueError(
                f"tool {name!r} contains forbidden term {forbidden.group(0)!r} "
                "in its name/description"
            )
        self._tools[name] = spec

    def get(self, name: str) -> ToolSpec:
        """Return the spec for ``name`` or raise :class:`UnknownTool`."""
        try:
            return self._tools[name]
        except KeyError as exc:
            raise UnknownTool(f"tool {name!r} is not registered") from exc

    def get_optional(self, name: str) -> ToolSpec | None:
        """Return the spec or ``None`` (engines use this for safe lookups)."""
        return self._tools.get(name)

    def names(self) -> list[str]:
        """Registered tool names, sorted for stable output."""
        return sorted(self._tools)

    def count(self) -> int:
        """Number of registered tools."""
        return len(self._tools)

    def iter_all(self) -> list[ToolSpec]:
        """All specs, sorted by name."""
        return [self._tools[name] for name in self.names()]

    def catalogue_for_llm(self) -> list[dict[str, object]]:
        """JSON-able tool catalogue for the planner prompt (no tiers)."""
        entries: list[dict[str, object]] = []
        for spec in self.iter_all():
            entries.append(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "args_schema": spec.args_model.model_json_schema(),
                }
            )
        return entries

    def catalogue_with_policy(self) -> list[dict[str, object]]:
        """Internal observability catalogue (name, tier, windows-only).

        Never sent to the LLM (docs/04_TOOLS_SPEC.md section 1: the planner
        must not reason about permissions).  Used for logs, auditing and the
        report tooling; tiers here are the *declared* base tiers.
        """
        entries: list[dict[str, object]] = []
        for spec in self.iter_all():
            entries.append(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "base_tier": spec.base_tier,
                    "windows_only": spec.windows_only,
                    "timeout_s": spec.timeout_s,
                }
            )
        return entries


def build_default_registry(settings: Settings | None = None) -> ToolRegistry:
    """Register the Phase 2 minimum tool set.

    ``settings`` is accepted for forward compatibility (aliases, platform
    choices) but not currently read.
    """
    _ = settings
    from jarvis.tools.apps import make_open_app_spec
    from jarvis.tools.dictation import (
        make_save_dictation_spec,
        make_start_dictation_spec,
        make_stop_dictation_spec,
    )
    from jarvis.tools.files import (
        make_append_file_spec,
        make_create_file_spec,
        make_delete_path_spec,
        make_list_dir_spec,
        make_read_file_spec,
        make_undo_last_delete_spec,
    )
    from jarvis.tools.keyboard import make_type_text_spec
    from jarvis.tools.system import make_lock_jarvis_spec, make_system_info_spec
    from jarvis.tools.web import make_google_search_spec, make_open_url_spec, make_web_answer_spec
    from jarvis.tools.whatsapp import make_whatsapp_send_spec

    registry = ToolRegistry()
    for spec in (
        make_open_app_spec(),
        make_open_url_spec(),
        make_google_search_spec(),
        make_web_answer_spec(),
        make_system_info_spec(),
        make_lock_jarvis_spec(),
        make_create_file_spec(),
        make_append_file_spec(),
        make_list_dir_spec(),
        make_read_file_spec(),
        make_delete_path_spec(),
        make_undo_last_delete_spec(),
        make_type_text_spec(),
        make_start_dictation_spec(),
        make_stop_dictation_spec(),
        make_save_dictation_spec(),
        make_whatsapp_send_spec(),
    ):
        registry.register(spec)
    return registry
