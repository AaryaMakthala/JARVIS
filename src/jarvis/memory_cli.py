"""``jarvis memory``: inspect and forget what JARVIS learned (offline, no LLM).

The commands here read and write **the same stores the agent uses**
(:class:`~jarvis.memory.store.SqliteMemory`: ``skills``, ``failures``,
``preferences``).  There is no second database and no separate cache, so a row
deleted here is really gone and a row listed here is really eligible to be
retrieved.

Rules this module keeps:

* **Plain text, no markup surprises.** Every value printed is bounded and
  printed with ``markup=False`` - a stored goal containing ``[redacted]`` is
  data, not Rich markup.
* **No secrets, no plan arguments.** A preference value that still looks like a
  credential is shown as ``[hidden]``; plan step arguments are never printed
  (``jarvis skills show`` already owns that and redacts at write time).
* **Bounded output.** ``list`` shows at most :data:`MAX_LIST_ROWS` rows and one
  short line each, so it stays usable on a small terminal.
* **Explicit deletion.** ``clear`` needs the word ``CLEAR`` (or an interactive
  yes), and ``forget`` refuses an id that exists in both tables unless a kind is
  named.

Task-log rows (per-task counters) are deliberately *not* touched: they are
telemetry for the benchmark, not learned memory.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Annotated, Any

import typer
from rich.console import Console

from jarvis import config
from jarvis.memory.db import looks_secret
from jarvis.memory.store import open_memory

__all__ = [
    "MAX_LIST_ROWS",
    "MAX_TEXT_CHARS",
    "MAX_VALUE_CHARS",
    "clear_memory",
    "forget_memory",
    "format_list_rows",
    "format_pref_rows",
    "memory_app",
]

console = Console()

#: Rows shown by ``jarvis memory list``.
MAX_LIST_ROWS = 50
#: Stored text shown per row (the goal/error text is only a label here).
MAX_TEXT_CHARS = 60
#: Preference value shown per row before truncation.
MAX_VALUE_CHARS = 40
#: Kinds a row id can name.
KINDS = ("skill", "failure")

memory_app = typer.Typer(
    help="Inspect and forget what JARVIS has learned (no LLM, works offline).",
    no_args_is_help=True,
)


# ------------------------------------------------------------------ formatting


def format_list_rows(memory: Any, limit: int = MAX_LIST_ROWS) -> list[str]:
    """One bounded line per stored skill and failure, newest first.

    Columns: ``kind  id  date  text``.  A quarantined failure is marked ``!`` so
    it is clear the planner is not being shown that row (the unverifiable-success
    fix withholds it rather than deleting it).
    """
    rows: list[tuple[str, int, str, str]] = []
    for skill in memory.skills.all(limit=max(limit, 1)):
        rows.append(("skill", int(skill.id), str(skill.last_used_at or skill.created_at), skill.goal_text))
    for failure in memory.failures.recent(limit=max(limit, 1), include_quarantined=True):
        text = f"{failure.goal_text} - {failure.error}" if failure.goal_text else failure.error
        mark = "!" if failure.quarantined else ""
        rows.append((f"failure{mark}", int(failure.id), str(failure.created_at), text))
    rows.sort(key=lambda row: (row[2], row[0], row[1]), reverse=True)
    lines = []
    for kind, row_id, date, text in rows[: max(limit, 1)]:
        label = " ".join(str(text or "").split())[:MAX_TEXT_CHARS]
        lines.append(f"{kind:<8} {row_id:>4}  {date[:10]:<10}  {label}")
    return lines


def format_pref_rows(memory: Any, limit: int = MAX_LIST_ROWS) -> list[str]:
    """One line per stored preference; credential-looking values are hidden."""
    lines = []
    for pref in memory.preferences.all(limit=max(limit, 1)):
        value = " ".join(str(pref.value or "").split())
        shown = "[hidden]" if looks_secret(value) else value[:MAX_VALUE_CHARS]
        lines.append(f"{pref.key} = {shown}")
    return lines


# ------------------------------------------------------------------ mutations


def forget_memory(memory: Any, row_id: int, kind: str = "auto") -> tuple[bool, str]:
    """Delete one skill/failure row; returns ``(deleted, message)``.

    ``kind="auto"`` deletes only when the id exists in exactly one table: an id
    that is both a skill and a failure is ambiguous and refused, because deleting
    the wrong learned row is not undoable.
    """
    wanted = (kind or "auto").strip().lower()
    if wanted not in (*KINDS, "auto"):
        return False, f"unknown kind {kind!r}; use auto, skill or failure"
    in_skills = memory.skills.get(int(row_id)) is not None
    in_failures = any(f.id == int(row_id) for f in memory.failures.recent(limit=500, include_quarantined=True))
    if not in_skills and not in_failures:
        return False, f"no skill or failure with id {row_id}"
    if wanted == "auto":
        if in_skills and in_failures:
            return False, f"id {row_id} is both a skill and a failure; pass --kind"
        wanted = "skill" if in_skills else "failure"
    if wanted == "skill":
        return (True, f"deleted skill {row_id}") if memory.skills.delete(int(row_id)) else (
            False,
            f"no skill with id {row_id}",
        )
    return (True, f"deleted failure {row_id}") if memory.failures.delete(int(row_id)) else (
        False,
        f"no failure with id {row_id}",
    )


def clear_memory(memory: Any) -> int:
    """Delete every learned skill, failure and preference; returns the count."""
    return int(memory.skills.clear() or 0) + int(memory.failures.clear() or 0) + int(
        memory.preferences.clear() or 0
    )


# --------------------------------------------------------------------- store


@contextmanager
def _open_store() -> Iterator[Any]:
    """Yield the agent's memory store, or exit with a clear message."""
    memory = open_memory(settings=config.load_settings())
    try:
        if not hasattr(memory, "skills"):
            console.print(
                "memory is disabled in config, so there is nothing to show.\n"
                "set [memory] enabled = true in config.toml"
            )
            raise typer.Exit(code=1)
        yield memory
    finally:
        close = getattr(memory, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # closing must never mask the result
                logging.getLogger("jarvis.memory_cli").debug("closing memory failed", exc_info=True)


# ------------------------------------------------------------------ commands


@memory_app.command("list")
def memory_list(
    limit: Annotated[int, typer.Option(help="Maximum rows to show.")] = MAX_LIST_ROWS,
) -> None:
    """List learned skills and recorded failures (newest first)."""
    with _open_store() as memory:
        rows = format_list_rows(memory, limit=min(max(limit, 1), MAX_LIST_ROWS))
        if not rows:
            console.print("nothing stored yet.")
            return
        for row in rows:
            console.print(row, markup=False, highlight=False)


@memory_app.command("forget")
def memory_forget(
    row_id: Annotated[int, typer.Argument(help="Id from `jarvis memory list`.")],
    kind: Annotated[str, typer.Option(help="auto, skill or failure.")] = "auto",
) -> None:
    """Delete one stored skill or failure."""
    with _open_store() as memory:
        deleted, message = forget_memory(memory, row_id, kind)
        console.print(message, markup=False)
        if not deleted:
            raise typer.Exit(code=1)


@memory_app.command("clear")
def memory_clear(
    confirm: Annotated[str, typer.Argument(help="Type CLEAR to confirm.")] = "",
) -> None:
    """Delete every learned skill, failure and preference (task log is kept)."""
    typed = (confirm or "").strip()
    if typed and typed != "CLEAR":
        console.print("not cleared: pass the word CLEAR to confirm.", markup=False)
        raise typer.Exit(code=1)
    if not typed and not typer.confirm(
        "Delete every learned skill, recorded failure and preference?"
    ):
        console.print("cancelled")
        return
    with _open_store() as memory:
        removed = clear_memory(memory)
        console.print(f"cleared {removed} memory rows", markup=False)


@memory_app.command("prefs")
def memory_prefs() -> None:
    """Show stored preference keys (values truncated, secrets hidden)."""
    with _open_store() as memory:
        rows = format_pref_rows(memory)
        if not rows:
            console.print("no preferences stored.")
            return
        for row in rows:
            console.print(row, markup=False, highlight=False)
