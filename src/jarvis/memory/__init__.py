"""Local skill/failure/preference memory.

Phase 8 wires the real SQLite backends: :class:`SqliteMemory` implements the
:class:`MemoryBackend` boundary for the agent, and the three stores behind it
(:class:`~jarvis.memory.skills.SkillStore`,
:class:`~jarvis.memory.failures.FailureStore`,
:class:`~jarvis.memory.prefs.PreferenceStore`) are also used directly by the
``jarvis skills`` CLI.  :class:`NullMemory` remains the deterministic no-op used
when the store is disabled, in ``--dry-run``, or cannot be opened.
"""

from __future__ import annotations

from jarvis.memory.base import MemoryBackend, MemoryKind, MemoryRecord, NullMemory
from jarvis.memory.db import open_db
from jarvis.memory.failures import Failure, FailureStore
from jarvis.memory.prefs import Preference, PreferenceStore
from jarvis.memory.skills import Skill, SkillStore
from jarvis.memory.store import SqliteMemory, open_memory

__all__ = [
    "Failure",
    "FailureStore",
    "MemoryBackend",
    "MemoryKind",
    "MemoryRecord",
    "NullMemory",
    "Preference",
    "PreferenceStore",
    "Skill",
    "SkillStore",
    "SqliteMemory",
    "open_db",
    "open_memory",
]
