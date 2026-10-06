"""Deterministic plan summary for the Stage 3 plan announcement.

Pure and side-effect free: no I/O, no LLM.  The summary is built **only** from
each step's canonical ``ToolSpec.describe(args)`` output, which the policy
engine already computed into ``Decision.summary`` (``policy/engine.py``).  The
planner's own ``rationale`` text is never read, so a prompt injection cannot
put words into JARVIS's mouth about what it is about to do.

The result is bounded for TTS by truncating at a whole-step boundary -- a step
is either spoken in full or not at all, and dropped steps are counted
(``"and N more steps"``).  The text is passed through :func:`redact` as B9
requires.  It is *not* spoken here; the caller sends it through
:func:`jarvis.agent.answer.spoken_answer`.
"""

from __future__ import annotations

from typing import Any

from jarvis.logging_setup import redact

__all__ = ["DEFAULT_PLAN_BUDGET", "summarise"]

#: Longest announcement, in characters, before steps start being dropped.
DEFAULT_PLAN_BUDGET = 600


def summarise(
    plan: Any, decisions: dict[str, Any] | None, *, budget: int = DEFAULT_PLAN_BUDGET
) -> str:
    """Return a bounded, deterministic, redacted plan announcement.

    ``plan`` is a :class:`~jarvis.agent.state.Plan`; ``decisions`` maps
    ``step.id`` to the engine's :class:`~jarvis.agent.state.Decision`.  Each
    step contributes its ``Decision.summary`` (built from ``ToolSpec.describe``)
    when present, joined in plan order with ``"then"``.  A missing decision
    falls back to the tool name -- never to LLM text.

    Returns ``""`` for an empty plan.  Never raises for odd input.
    """
    steps = list(getattr(plan, "steps", None) or [])
    texts = [t for t in (redact(_step_text(step, decisions or {})) for step in steps) if t]
    if not texts:
        return ""

    limit = max(int(budget), 0)
    total = len(texts)
    chosen: list[str] = []
    for index, text in enumerate(texts):
        remaining = total - index - 1
        if len(_format(chosen + [text], remaining)) <= limit:
            chosen.append(text)
        else:
            break

    if not chosen:
        # Budget smaller than a single step describe: fall back to a count-only
        # line rather than truncating a step mid-sentence.
        return redact(f"I will perform {total} steps.")
    return redact(_format(chosen, total - len(chosen)))


def _step_text(step: Any, decisions: dict[str, Any]) -> str:
    """Canonical text for one step: its engine summary, else its tool name."""
    decision = decisions.get(str(getattr(step, "id", "") or ""))
    summary = getattr(decision, "summary", "") if decision is not None else ""
    if isinstance(summary, str) and summary.strip():
        return summary.strip()
    tool = str(getattr(step, "tool", "") or "").strip()
    return f"use {tool}" if tool else ""


def _format(chosen: list[str], dropped: int) -> str:
    """Render ``chosen`` step texts, optionally counting the dropped ones."""
    body = ", then ".join(chosen)
    if dropped > 0:
        noun = "step" if dropped == 1 else "steps"
        return f"I will {body}, and {dropped} more {noun}."
    return f"I will {body}."
