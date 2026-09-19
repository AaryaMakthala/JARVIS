"""Small helpers shared by agent nodes."""

from __future__ import annotations

_SNIPPET = 4000


def truncate(text: str, limit: int = _SNIPPET) -> str:
    """Truncate text to ``limit`` chars for storage in state (docs 4 KB rule)."""
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def new_task_id() -> str:
    """Compact, collision-resistant task id for checkpointer thread_id."""
    import secrets

    return secrets.token_hex(6)
