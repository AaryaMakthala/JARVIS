"""Helpers for interpreting a tool execution history with retries.

``AgentState.results`` is an append-only audit trail: a failed attempt stays in
the list even after the same step succeeded on retry.  Terminal wording, task
status, skill recording and replanning need the *effective* outcome instead:
the latest attempt for each step id, in first-attempt order.
"""

from __future__ import annotations

from typing import Any

from jarvis.agent.state import StepResult

__all__ = ["effective_results"]


def effective_results(state: dict[str, Any]) -> list[StepResult]:
    """Latest recorded outcome per step id, preserving first-seen order."""
    latest: dict[str, StepResult] = {}
    order: list[str] = []
    for position, raw in enumerate(state.get("results") or []):
        if not isinstance(raw, StepResult):
            continue
        step_id = str(raw.step_id or "")
        key = step_id or f"__unnamed_{position}"
        if key not in latest:
            order.append(key)
        latest[key] = raw
    return [latest[key] for key in order]
