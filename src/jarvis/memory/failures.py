"""Failure memory: what went wrong, so retries and replans stop repeating it.

Phase 8 of ``docs/05_BUILD_PLAN.md`` asks for a failure log that informs retry
decisions.  Records are **examples of past failures**, surfaced to the planner
alongside skills but always labelled as failures, so the model can avoid the
approach that did not work.  They are never instructions and never executed.

Two rules keep that log from poisoning unrelated work, both learned the hard
way on a live machine (a "lock my computer" request was refused for a week
because of one row, see ``PROGRESS.md``):

* **Provenance.** Every row records *why* the step was bad (``ok``/``verified``/
  ``tainted``) in ``step_state``, so the sentence handed to the planner is
  honest and a future reader can audit the row.  Rows written by a build older
  than the ``verified=None`` fix have no provenance; the false ones among them
  are quarantined by :func:`jarvis.memory.db.migrate_failures`.
* **Relevance.** A remembered failure is only surfaced for a *related* request
  (:meth:`FailureStore.relevant`).  Before that, the most recent N failures
  were injected into every single prompt, so one bad row could refuse any
  command at all.

The same anti-secret guard as :mod:`jarvis.memory.skills` applies: a failure
message that looks like it contains a credential is stored with the offending
text replaced rather than persisted verbatim.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from jarvis.memory.db import (
    looks_secret,
    now_iso,
    prune_failures,
    redact_secrets,
    safe_plan,
)
from jarvis.memory.embeddings import cosine, embed_text

__all__ = ["FAILURE_SIMILARITY", "MAX_ERROR_CHARS", "Failure", "FailureStore"]

#: Failures come from tool error strings, which can be arbitrarily long.  A
#: remembered failure is a hint, not an archive.
MAX_ERROR_CHARS = 300

#: Cosine floor for showing a remembered failure for a given request.  Lower
#: than :data:`~jarvis.memory.embeddings.DEFAULT_SIMILARITY` (0.75) for skills
#: on purpose: with the lexical encoder a genuine rephrasing ("open notepad" vs
#: "open the notepad app") scores ~0.56, and **missing** a real failure is worse
#: than showing a slightly loose one - an unrelated goal still scores < 0.2, so
#: the two are well separated.  Mirrored by
#: ``config.MemorySettings.failure_similarity_threshold`` (kept in sync by a
#: test, since ``config`` must not import the memory layer).
FAILURE_SIMILARITY = 0.45


@dataclass(frozen=True)
class Failure:
    """One recorded failed step."""

    id: int
    goal_text: str
    step: dict[str, Any]
    error: str
    created_at: str = ""
    state: dict[str, Any] = field(default_factory=dict)
    quarantined: bool = False

    def as_memory_text(self) -> str:
        """Bounded, honest sentence for the planner prompt.

        The clause is chosen from the recorded provenance so the planner is told
        *which* kind of failure this was.  The caution is phrased as advice, not
        as a veto: a remembered failure must not become a standing refusal to
        do what the user just asked for (the planner system prompt carries the
        general rule).
        """
        tool = str(self.step.get("tool") or "a tool")
        goal = self.goal_text or "a similar request"
        error = (self.error or "it failed").strip()[:MAX_ERROR_CHARS]
        return (
            f'Earlier attempt to "{goal}" {self._outcome_clause()} at {tool}: {error}. '
            f"Prefer a different approach, or check the state first before repeating it."
        )

    def _outcome_clause(self) -> str:
        """How this step went wrong, from its recorded provenance."""
        if self.state.get("ok") is False:
            return "failed"
        if self.state.get("verified") is False:
            return "did not verify"
        if self.state.get("tainted"):
            return "returned untrusted external text"
        return "failed"  # legacy row: no provenance, so the cautious wording


class FailureStore:
    """Append-only log of failed steps, bounded by an LRU-ish cap."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        max_rows: int = 200,
        similarity_threshold: float = FAILURE_SIMILARITY,
    ) -> None:
        self._conn = conn
        self._max_rows = int(max_rows)
        self._threshold = float(similarity_threshold)

    def record(
        self,
        goal_text: str,
        step: dict[str, Any] | None = None,
        error: str = "",
        state: dict[str, Any] | None = None,
    ) -> Failure | None:
        """Append one failure; ``None`` when the entry is skipped.

        ``state`` is the failing step's outcome (``ok``/``verified``/``tainted``).
        It is stored as the row's provenance so the planner sees an honest
        sentence and a later reader can tell a real failure from a bug artefact.

        Two entries are refused, both returning ``None``:

        * an empty or secret-only message, and
        * a step whose outcome says it *succeeded* - see below.
        """
        # Truncate first (the text is unbounded), then redact.  A tool error is
        # untrusted third-party text: "login failed for hunter2, password
        # hunter2" must not become a durable row on disk.
        message = redact_secrets((error or "").strip())[:MAX_ERROR_CHARS]
        if not message:
            return None
        if _is_unverified_success(state):
            # Defence in depth.  "The step succeeded but nothing could confirm
            # it" is not a failure, and recording it as one is what refused a
            # live "lock my computer" for a week.  The caller in
            # ``memory_save._failure_record`` already filters this out; refusing
            # it here means no future caller can reintroduce the bug.  A step that
            # *failed* verification keeps its row - that is a real failure.
            return None

        safe_step = safe_plan(step or {})
        goal = (goal_text or "").strip()[:200]
        if looks_secret(goal):
            goal = "[redacted goal]"
        now = now_iso()
        cursor = self._conn.execute(
            "INSERT INTO failures (goal_text, step_json, error, created_at, step_state)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                goal,
                json.dumps(safe_step, ensure_ascii=False),
                message,
                now,
                json.dumps(_clean_state(state), ensure_ascii=False),
            ),
        )
        prune_failures(self._conn, self._max_rows)
        return Failure(
            id=int(cursor.lastrowid or 0),
            goal_text=goal,
            step=safe_step,
            error=message,
            created_at=now,
            state=_clean_state(state),
        )

    def count(self) -> int:
        """Failures the planner can still see (quarantined rows excluded)."""
        return int(
            self._conn.execute(
                "SELECT COUNT(*) FROM failures WHERE COALESCE(quarantined, 0) = 0"
            ).fetchone()[0]
        )

    def quarantined_count(self) -> int:
        """Rows withheld from the planner by the unverifiable-success fix."""
        return int(
            self._conn.execute(
                "SELECT COUNT(*) FROM failures WHERE COALESCE(quarantined, 0) = 1"
            ).fetchone()[0]
        )

    def recent(
        self,
        limit: int = 3,
        goal_text: str | None = None,
        *,
        include_quarantined: bool = False,
    ) -> list[Failure]:
        """Most recent failures, optionally restricted to one goal substring.

        Quarantined rows are excluded unless asked for explicitly, so a stale
        false failure cannot reach the planner through this path either.
        """
        if limit <= 0:
            return []
        where = [] if include_quarantined else ["COALESCE(quarantined, 0) = 0"]
        params: list[Any] = []
        if goal_text:
            where.append("goal_text LIKE ?")
            params.append(f"%{goal_text}%")
        clause = f" WHERE {' AND '.join(where)}" if where else ""
        params.append(limit)
        rows = self._conn.execute(
            f"SELECT * FROM failures{clause} ORDER BY id DESC LIMIT ?", params
        )
        return [self._to_failure(row) for row in rows]

    def relevant(
        self,
        query: str,
        limit: int = 3,
        *,
        floor: float | None = None,
    ) -> list[Failure]:
        """Failures whose goal is similar enough to ``query``, best first.

        The retrieval path for the planner.  Scoring mirrors
        :meth:`~jarvis.memory.skills.SkillStore.find` - the same deterministic
        offline embedding over the stored ``goal_text`` - run on read: with the
        table capped at ``max_failures`` a full scan is cheap, and it means no
        extra column and no backfill for rows written by an older build.
        """
        if limit <= 0 or not (query or "").strip():
            return []
        threshold = self._threshold if floor is None else float(floor)
        query_vector = embed_text(query)
        scored: list[tuple[float, int, Failure]] = []
        for row in self._conn.execute("SELECT * FROM failures WHERE COALESCE(quarantined, 0) = 0"):
            failure = self._to_failure(row)
            score = cosine(query_vector, embed_text(failure.goal_text))
            if score >= threshold:
                scored.append((score, failure.id, failure))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [failure for _score, _id, failure in scored[:limit]]

    def repeated_tools(self, goal_text: str, limit: int = 5) -> list[str]:
        """Tool names that already failed for ``goal_text`` (most recent first).

        This is the signal a replan wants: "you already tried ``web_answer`` on
        this goal and it did not verify".
        """
        seen: list[str] = []
        for failure in self.recent(limit=limit * 4, goal_text=goal_text):
            tool = str(failure.step.get("tool") or "").strip()
            if tool and tool not in seen:
                seen.append(tool)
            if len(seen) >= limit:
                break
        return seen

    def clear(self) -> int:
        """Delete every failure (used by ``jarvis reset --memory``).

        Includes quarantined rows: an explicit reset is a request to forget.
        """
        cur = self._conn.execute("DELETE FROM failures")
        return int(cur.rowcount or 0)

    @staticmethod
    def _to_failure(row: sqlite3.Row) -> Failure:
        # ``sqlite3.Row`` has no ``__contains__``, so the column set is read
        # explicitly: a row read from a not-yet-migrated database simply has no
        # provenance and is not quarantined.
        columns = set(row.keys())
        return Failure(
            id=int(row["id"]),
            goal_text=str(row["goal_text"] or ""),
            step=_loads(row["step_json"]),
            error=str(row["error"] or ""),
            created_at=str(row["created_at"] or ""),
            state=_loads(row["step_state"]) if "step_state" in columns else {},
            quarantined=bool(row["quarantined"]) if "quarantined" in columns else False,
        )


def _is_unverified_success(state: dict[str, Any] | None) -> bool:
    """True when a step's outcome is "succeeded, and nothing could confirm it".

    This is the exact negation of the "bad step" test in
    :func:`jarvis.agent.nodes.memory_save._failure_record`
    (``not ok or verified is False or tainted``), kept here as the store's own
    backstop.  Both must agree, so the two definitions are written as one: a row
    is refused only when the step reported success, verification was not a
    failure, and the output was not tainted.  ``verified=False`` is a genuine
    failure and is always kept.
    """
    outcome = _clean_state(state)
    return outcome["ok"] is True and outcome["verified"] is not False and not outcome["tainted"]


def _clean_state(state: dict[str, Any] | None) -> dict[str, Any]:
    """Only the three provenance fields, so a caller cannot smuggle anything in.

    ``None`` is a legitimate input: a row written before provenance existed (or
    by a caller that has none) is stored with all three fields unknown rather
    than being rejected.
    """
    data = state or {}
    return {
        "ok": data.get("ok") if isinstance(data.get("ok"), bool) else None,
        "verified": data.get("verified") if data.get("verified") in (True, False) else None,
        "tainted": bool(data.get("tainted")),
    }


def _loads(raw: Any) -> dict[str, Any]:
    """Tolerant JSON object load: a corrupt row degrades to ``{}``."""
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}
