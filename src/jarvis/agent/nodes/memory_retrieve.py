"""memory_retrieve node: fetch relevant past examples (no-op in Phase 1).

Phase 8 implements embedding retrieval.  For now the node is a pure pass
through so the graph topology matches docs/02_ARCHITECTURE.md and the planner
prompt can forward examples when memory exists.
"""

from __future__ import annotations

from typing import Any

from jarvis.agent.context import AppContext


def memory_retrieve(state: dict[str, Any], ctx: AppContext) -> dict[str, Any]:
    """Return an empty memory context until Phase 8."""
    del state, ctx
    return {"memory_context": []}
