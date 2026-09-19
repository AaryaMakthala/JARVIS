"""System and user prompt templates for the LLM client.

Prompts are intentionally not defined until the planner is implemented
(Phase 1). They will follow the rules in docs/04_TOOLS_SPEC.md section 3:
tools-only planning, no reasoning about permissions, and untrusted content
wrapped in ``<untrusted_data>`` delimiters.
"""

from __future__ import annotations

__all__: list[str] = []
