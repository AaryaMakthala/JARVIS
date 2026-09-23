"""memory_retrieve node: fetch relevant past examples (bounded).

Uses the context's :class:`MemoryBackend` when one is configured (daemon /
CLI); falls back to an empty context otherwise.  Retrieval is always capped by
``settings.memory.retrieval_limit`` so a prompt can never be flooded, and the
records are converted to plain dicts for the planner prompt.
"""

from __future__ import annotations

from typing import Any

from jarvis.agent.context import AppContext
from jarvis.agent.nodes.util import truncate


def memory_retrieve(state: dict[str, Any], ctx: AppContext) -> dict[str, Any]:
    """Return up to ``retrieval_limit`` memory records for the user input."""
    if ctx.memory is None:
        return {"memory_context": []}

    query = state.get("user_input") or ""
    limit = max(1, int(ctx.settings.memory.retrieval_limit or 1))
    try:
        records = ctx.memory.retrieve(query, limit=limit)
    except Exception:  # noqa: BLE001 - a memory failure must never block a task
        ctx.logger.warning("memory retrieval failed; continuing without it")
        return {"memory_context": []}

    context = []
    for record in records:
        text = truncate(getattr(record, "text", "") or "")
        if not text:
            continue
        item = {"kind": getattr(record, "kind", "session"), "text": text}
        meta = getattr(record, "meta", None)
        if isinstance(meta, dict) and meta:
            item["meta"] = meta
        context.append(item)
    return {"memory_context": context}
