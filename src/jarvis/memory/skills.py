"""Learned skills: verified plans JARVIS may re-use as *examples* (Phase 8).

A "skill" is a goal sentence plus the plan that worked, with success/failure
counters.  Safety rules from ``docs/05_BUILD_PLAN.md`` (Phase 8) and
``docs/03_SECURITY_AND_POLICY.md``:

* **A skill is never executed.**  It is only ever returned to the planner as
  text inside ``memory_context``, and the planner still has to produce a new
  plan that goes through ``validate`` + ``policy_gate``.  Nothing in this module
  can run a tool, touch a file, or bypass the policy engine.
* **Only successful, fully verified, non-tainted plans are saved.**  The gate
  lives in the ``memory_save`` node; :meth:`SkillStore.save` additionally
  refuses rows that look like they carry a credential (fail closed).
* **A skill that fails more often than it succeeds is ignored** at retrieval
  time (``fail_count > success_count``), so one bad plan does not poison future
  runs.
* Repeating a goal updates the existing row (dedup by cosine similarity above
  :data:`~jarvis.memory.embeddings.DEFAULT_SIMILARITY`) instead of piling up
  near-duplicates.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from jarvis.memory.db import now_iso, prune_skills, safe_plan, secret_free_rows
from jarvis.memory.embeddings import DEFAULT_SIMILARITY, cosine, embed_text, from_bytes, to_bytes

__all__ = ["Skill", "SkillSaveResult", "SkillStore"]


@dataclass(frozen=True)
class Skill:
    """One remembered, verified plan."""

    id: int
    goal_text: str
    plan: dict[str, Any] = field(default_factory=dict)
    tools_used: list[str] = field(default_factory=list)
    success_count: int = 1
    fail_count: int = 0
    created_at: str = ""
    last_used_at: str = ""

    @property
    def trusted(self) -> bool:
        """True when this skill has a positive success history."""
        return self.fail_count <= self.success_count

    def as_memory_text(self) -> str:
        """The bounded, human-readable form handed to the planner prompt.

        Deliberately *not* the JSON plan: the planner sees prose it can reuse
        the reasoning from, and the machine-readable plan never widens the
        untrusted-data surface of the prompt.
        """
        tools = ", ".join(self.tools_used) or "no tools"
        return f'Earlier successful approach for "{self.goal_text}" used: {tools}.'


@dataclass(frozen=True)
class SkillSaveResult:
    """Outcome of :meth:`SkillStore.save`."""

    skill: Skill | None
    created: bool = False
    reason: str = ""

    @property
    def saved(self) -> bool:
        return self.skill is not None


class SkillStore:
    """CRUD + retrieval for the ``skills`` table."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        similarity_threshold: float = DEFAULT_SIMILARITY,
        max_rows: int = 500,
    ) -> None:
        self._conn = conn
        self._threshold = float(similarity_threshold)
        self._max_rows = int(max_rows)

    # ------------------------------------------------------------------ read

    def count(self) -> int:
        """Number of stored skills (any trust level)."""
        return int(self._conn.execute("SELECT COUNT(*) FROM skills").fetchone()[0])

    def all(self, *, trusted_only: bool = False, limit: int = 200) -> list[Skill]:
        """List skills, most recently used first."""
        if limit <= 0:
            return []
        sql = "SELECT * FROM skills"
        if trusted_only:
            sql += " WHERE fail_count <= success_count"
        sql += " ORDER BY last_used_at DESC, id DESC LIMIT ?"
        return [self._to_skill(row) for row in self._conn.execute(sql, (limit,))]

    def get(self, skill_id: int) -> Skill | None:
        """One skill by id, or ``None``."""
        row = self._conn.execute("SELECT * FROM skills WHERE id = ?", (skill_id,)).fetchone()
        return self._to_skill(row) if row else None

    def find(self, query: str, limit: int = 3) -> list[Skill]:
        """Skills relevant to ``query``, best first, above the threshold.

        Scans the (small, LRU-bounded) table and scores with the deterministic
        lexical embedding.  Untrusted skills are filtered *before* scoring so a
        poisoned plan cannot even be matched, let alone surfaced.
        """
        if limit <= 0 or not (query or "").strip():
            return []
        query_vector = embed_text(query)
        scored: list[tuple[float, int, Skill]] = []
        for row in self._conn.execute("SELECT * FROM skills"):
            skill = self._to_skill(row)
            if not skill.trusted:
                continue
            score = cosine(query_vector, from_bytes(row["embedding"]))
            if score >= self._threshold:
                scored.append((score, skill.id, skill))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [skill for _score, _id, skill in scored[:limit]]

    # ----------------------------------------------------------------- write

    def save(
        self,
        goal_text: str,
        plan: dict[str, Any] | None = None,
        tools_used: list[str] | None = None,
    ) -> SkillSaveResult:
        """Store a verified plan, or fold it into a near-identical existing row.

        Returns a :class:`SkillSaveResult` with ``reason`` set on refusal; the
        caller reports that honestly instead of pretending the skill was kept.
        """
        goal = (goal_text or "").strip()
        if not goal:
            return SkillSaveResult(skill=None, reason="empty goal text")

        plan_copy = safe_plan(plan or {})
        if not secret_free_rows(goal, json.dumps(plan_copy, ensure_ascii=False)):
            return SkillSaveResult(skill=None, reason="refused: looks like it carries a secret")

        tools = [str(name) for name in (tools_used or []) if str(name).strip()]
        now = now_iso()

        existing = self._match_for_update(goal)
        if existing is not None:
            self._conn.execute(
                "UPDATE skills SET success_count = success_count + 1, last_used_at = ? WHERE id = ?",
                (now, existing.id),
            )
            refreshed = self.get(existing.id)
            return SkillSaveResult(skill=refreshed, created=False, reason="reinforced")

        cursor = self._conn.execute(
            """
            INSERT INTO skills (goal_text, plan_json, embedding, tools_used,
                                success_count, fail_count, created_at, last_used_at)
            VALUES (?, ?, ?, ?, 1, 0, ?, ?)
            """,
            (
                goal,
                json.dumps(plan_copy, ensure_ascii=False),
                to_bytes(embed_text(goal)),
                json.dumps(tools),
                now,
                now,
            ),
        )
        prune_skills(self._conn, self._max_rows)
        skill = self.get(int(cursor.lastrowid or 0))
        return SkillSaveResult(skill=skill, created=True)

    def record_failure(self, skill_id: int) -> bool:
        """Increment ``fail_count`` for a skill that turned out to be a dead end."""
        cur = self._conn.execute(
            "UPDATE skills SET fail_count = fail_count + 1, last_used_at = ? WHERE id = ?",
            (now_iso(), skill_id),
        )
        return bool(cur.rowcount)

    def record_use(self, skill_id: int) -> bool:
        """Mark a retrieved skill as used (refreshes the LRU order)."""
        cur = self._conn.execute(
            "UPDATE skills SET last_used_at = ? WHERE id = ?", (now_iso(), skill_id)
        )
        return bool(cur.rowcount)

    def delete(self, skill_id: int) -> bool:
        """Forget one skill; ``True`` when a row was removed."""
        cur = self._conn.execute("DELETE FROM skills WHERE id = ?", (skill_id,))
        return bool(cur.rowcount)

    def clear(self) -> int:
        """Delete every skill (used by ``jarvis reset --memory``)."""
        cur = self._conn.execute("DELETE FROM skills")
        return int(cur.rowcount or 0)

    # --------------------------------------------------------------- helpers

    def _match_for_update(self, goal: str) -> Skill | None:
        """The trusted skill this goal folds into, if any (dedup)."""
        matches = self.find(goal, limit=1)
        return matches[0] if matches else None

    @staticmethod
    def _to_skill(row: sqlite3.Row) -> Skill:
        return Skill(
            id=int(row["id"]),
            goal_text=str(row["goal_text"]),
            plan=_loads(row["plan_json"]),
            tools_used=_loads(row["tools_used"], default=[]),
            success_count=int(row["success_count"] or 0),
            fail_count=int(row["fail_count"] or 0),
            created_at=str(row["created_at"] or ""),
            last_used_at=str(row["last_used_at"] or ""),
        )


def _loads(raw: Any, default: Any = None) -> Any:
    """Tolerant JSON load: a corrupt row degrades to ``default``."""
    if not raw:
        return {} if default is None else default
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return {} if default is None else default
    return value
