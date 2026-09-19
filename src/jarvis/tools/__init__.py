"""Tool definitions and the tool registry.

Every tool returns ToolResult, declares a base_tier, and provides verify() and
a dry_run behaviour (docs/04_TOOLS_SPEC.md).  The planner may only call tools
registered in :class:`~jarvis.tools.registry.ToolRegistry`.
"""

__all__ = ["apps", "base", "files", "registry", "system", "web"]
