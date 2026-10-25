"""memory_retrieve node: fetch a small, safe set of past examples.

Memory is **advisory**.  What lands here is reference text for the planner; the
planner still has to produce a plan that goes through ``validate`` and
``policy_gate``, so nothing retrieved here can be executed or replayed.  The
prompt wraps this block in ``<untrusted_data>`` for the same reason it wraps a
web page.

The node is therefore deliberately paranoid about its own input - a stored
example is third-party text that may quote a page, a tool error or an old goal:

* **bounded**: each item is cut to ``memory.item_max_chars`` (300) and the whole
  block to ``memory.context_max_chars`` (1200), so retrieval can never flood a
  prompt;
* **filtered**: an item that trips the existing secret check is dropped, and a
  failure example older than ``memory.failure_max_age_days`` (30) is dropped -
  a stale failure is the one most likely to be wrong about today's world;
* **deduplicated**: near-identical items (same words once punctuation and case
  are ignored) are sent once;
* **time-bounded**: the store call runs with
  ``memory.retrieve_timeout_seconds`` (2.0).  The SQLite connection is opened
  with ``check_same_thread=False`` and every store call takes the same
  ``RLock``, so a worker thread is safe; if it does not answer in time the task
  continues with **empty** memory context (fail open for retrieval) and one
  warning line that carries no content.

``memory_context`` stays a ``list[dict]`` of plain primitives (``kind``,
``text``, optional ``meta``) so the checkpoint serializer needs no new type.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from typing import Any

from jarvis.agent.context import AppContext
from jarvis.memory.base import MemoryRecord
from jarvis.memory.db import looks_secret

__all__ = ["memory_retrieve"]

#: Fallbacks when an older settings object has no M2 keys at all.
_DEFAULT_ITEM_CHARS = 300
_DEFAULT_CONTEXT_CHARS = 1200
_DEFAULT_MAX_AGE_DAYS = 30
_DEFAULT_TIMEOUT_S = 2.0
#: Room reserved for the wrapper keys of one item (``kind``/``meta`` punctuation).
_ITEM_OVERHEAD_CHARS = 40


def memory_retrieve(state: dict[str, Any], ctx: AppContext) -> dict[str, Any]:
    """Return the bounded, filtered memory context for this task."""
    if ctx.memory is None:
        return {"memory_context": []}

    settings = ctx.settings.memory
    query = state.get("user_input") or ""
    limit = max(1, int(getattr(settings, "retrieval_limit", 3) or 3))
    timeout_s = float(getattr(settings, "retrieve_timeout_seconds", _DEFAULT_TIMEOUT_S) or 0.0)

    try:
        records = _retrieve_with_timeout(ctx.memory, query, limit, timeout_s)
    except Exception:  # noqa: BLE001 - a memory failure must never block a task
        # No content in the log line: the query (user text) stays out of it.
        ctx.logger.warning("memory retrieval unavailable this turn; continuing without it")
        return {"memory_context": []}

    context = _select(records, settings)
    if not context:
        ctx.logger.debug("memory: nothing usable for this request")
    return {"memory_context": context}


def _retrieve_with_timeout(
    memory: Any, query: str, limit: int, timeout_s: float
) -> list[MemoryRecord]:
    """Call ``memory.retrieve`` in a worker, bounded by ``timeout_s``.

    The thread is abandoned on timeout (a running SQLite call cannot be
    interrupted); it holds the store's lock only for the duration of that one
    query, and the next task's own timeout still applies.  ``timeout_s <= 0``
    disables the bound.
    """
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jarvis-memory")
    try:
        future = executor.submit(memory.retrieve, query, limit)
        return list(future.result(timeout=timeout_s if timeout_s > 0 else None))
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _select(records: list[MemoryRecord], settings: Any) -> list[dict[str, Any]]:
    """Filter, dedupe and bound the retrieved records (order preserved)."""
    item_cap = max(int(getattr(settings, "item_max_chars", _DEFAULT_ITEM_CHARS) or 0), 1)
    total_cap = max(int(getattr(settings, "context_max_chars", _DEFAULT_CONTEXT_CHARS) or 0), 1)
    max_age_days = int(getattr(settings, "failure_max_age_days", _DEFAULT_MAX_AGE_DAYS) or 0)

    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()
    used = 0
    for record in records:
        text = " ".join(str(getattr(record, "text", "") or "").split())
        if not text:
            continue
        meta = getattr(record, "meta", None)
        if not isinstance(meta, dict):
            meta = None
        if _is_secret(text, meta):
            continue
        if _is_stale_failure(meta, max_age_days):
            continue
        key = _dedupe_key(text)
        if key in seen:
            continue
        item: dict[str, Any] = {
            "kind": str(getattr(record, "kind", "session") or "session"),
            "text": text[:item_cap],
        }
        if meta:
            item["meta"] = meta
        cost = len(item["text"]) + _ITEM_OVERHEAD_CHARS
        if meta:
            cost += len(json.dumps(meta, default=str))
        if used + cost > total_cap:
            break
        seen.add(key)
        used += cost
        chosen.append(item)
    return chosen


def _is_secret(text: str, meta: dict[str, Any] | None) -> bool:
    """True when the item pairs a credential-ish name with a real value."""
    if looks_secret(text):
        return True
    return bool(meta) and looks_secret(json.dumps(meta, default=str))


def _is_stale_failure(meta: dict[str, Any] | None, max_age_days: int) -> bool:
    """True for a failure example older than ``max_age_days`` (0 = keep all).

    Only failure rows carry ``failure_id``; skills and preferences have no
    timestamp here, so they are unaffected.  An unparseable date fails open
    (the row is kept) - dropping examples because of a formatting quirk would
    hide real history.
    """
    if not meta or max_age_days <= 0 or "failure_id" not in meta:
        return False
    created = meta.get("created_at")
    if not isinstance(created, str) or not created:
        return False
    try:
        stamp = datetime.fromisoformat(created)
    except ValueError:
        return False
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    return stamp < datetime.now(UTC) - timedelta(days=max_age_days)


def _dedupe_key(text: str) -> str:
    """Normalised form used to detect a near-identical item (case/punctuation)."""
    return " ".join("".join(ch if ch.isalnum() else " " for ch in text.casefold()).split())
