"""compose node: the short spoken answer and the full screen answer.

Runs on every terminal path, between the routers and ``respond``.  It is the
deterministic bridge that lets one task speak differently to a voice channel
than to the terminal, without a second LLM call and without touching the plan.

What it may write: exactly two string keys, ``spoken_answer`` and
``screen_answer``.  It never reads or writes ``plan``, ``decisions``,
``approved_hashes``, ``results``, ``halted_reason`` or ``step_index``, never
calls a tool, and never approves anything.

``screen_answer`` is what ``respond`` has always shown (it asks ``respond`` for
it), so the terminal is unchanged.  ``spoken_answer`` is at most two sentences,
built by :mod:`jarvis.agent.compose_rules` from what the state proves.

Failure is soft: with ``compose.enabled = false``, or if anything here raises,
the node returns no update at all and ``respond`` falls back to its own text -
exactly the behaviour before this node existed.
"""

from __future__ import annotations

from typing import Any

from jarvis.agent.compose_rules import address_phrase, classify_outcome, spoken_answer
from jarvis.agent.context import AppContext
from jarvis.agent.nodes.respond import base_answer
from jarvis.memory.db import looks_secret


def compose(state: dict[str, Any], ctx: AppContext) -> dict[str, Any]:
    """Write ``spoken_answer`` / ``screen_answer`` for this terminal state."""
    if not _enabled(ctx):
        return {}
    try:
        outcome = classify_outcome(state)
        address = address_phrase(_user_name(ctx))
        return {
            "spoken_answer": spoken_answer(state, outcome, address=address),
            "screen_answer": base_answer(state),
        }
    except Exception as exc:  # noqa: BLE001 - compose must never break a task
        ctx.logger.warning("compose skipped: %s", exc)
        return {}


def _enabled(ctx: AppContext) -> bool:
    """``compose.enabled`` (default true when the section is absent)."""
    settings = getattr(getattr(ctx, "settings", None), "compose", None)
    return bool(getattr(settings, "enabled", True))


def _user_name(ctx: AppContext) -> str:
    """The stored ``user_name`` preference, or ``""`` (read-only, bounded).

    Read through the existing :class:`~jarvis.memory.prefs.PreferenceStore`; no
    write path is added anywhere.  Any backend without preferences, a locked
    database, a blank value, an over-long value or a value that still looks like
    a credential all yield ``""`` - the plain wording is always safe.
    """
    store = getattr(getattr(ctx, "memory", None), "preferences", None)
    getter = getattr(store, "get", None)
    if not callable(getter):
        return ""
    try:
        value = getter("user_name")
    except Exception as exc:  # noqa: BLE001 - a broken store only loses the name
        ctx.logger.debug("user_name preference unreadable: %s", exc)
        return ""
    if not isinstance(value, str):
        return ""
    name = " ".join(value.split())[:40].strip()
    if not name or looks_secret(name):
        return ""
    return name
