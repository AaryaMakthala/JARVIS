"""Small helpers shared by agent nodes."""

from __future__ import annotations

import dataclasses
import secrets
from typing import Any

from jarvis.agent.state import StepResult

_SNIPPET = 4000


def truncate(text: str, limit: int = _SNIPPET) -> str:
    """Truncate text to ``limit`` chars for storage in state (docs 4 KB rule)."""
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def new_task_id() -> str:
    """Compact, collision-resistant task id for checkpointer thread_id."""
    return secrets.token_hex(6)


def tainted_fragments(state: dict[str, Any]) -> tuple[str, ...]:
    """Output text of every previous step that was flagged as untrusted.

    Only ``StepResult`` objects with ``tainted=True`` contribute.  This is our
    own data (tool output), not LLM text, which is why the engine trusts it for
    the independent taint check in docs/03 section 8.
    """
    return tuple(
        r.output
        for r in state.get("results") or []
        if isinstance(r, StepResult) and r.tainted and r.output
    )


def policy_context_for(state: dict[str, Any], policy_ctx: Any) -> Any:
    """Return a PolicyContext enriched with the state the engine cannot see.

    Two things are added, both read-only and both from engine-owned data:

    * ``tainted_fragments`` - so the taint check does not rely on the LLM having
      set ``depends_on_untrusted``;
    * ``user_input`` - the original request, which the Phase 8 risk classifier
      needs (docs/08 section 3) and which no rule ever reads.

    The original context is returned unchanged when there is nothing to add, so
    the common path allocates nothing.
    """
    if policy_ctx is None:
        return None
    tainted = tainted_fragments(state)
    user_input = str(state.get("user_input") or "")
    if not tainted and not user_input:
        return policy_ctx
    return dataclasses.replace(policy_ctx, tainted_fragments=tainted, user_input=user_input)
