"""Strict loaders for ``benchmarks/tasks.yaml`` and ``benchmarks/redteam.yaml``.

The task files are *data*, but they still sit on a trust boundary: a typo in
``expected.type`` must fail loudly when the suite is loaded, not silently turn a
security case into a passing no-op three phases later.  So every entry is
validated by a Pydantic model with ``extra="forbid"`` (a misspelled key is an
error, not a silently dropped setting) and the closed vocabularies -
categories, expectation types, red-team outcomes - are module constants.

The format is the one specified in docs/07_TESTING_AND_BENCHMARK.md section 3.1.
Two conventions the doc leaves to the implementation:

* ``{sandbox}`` in a ``command`` is substituted with the sandbox root by the
  runner, so a file task can name a path inside the sandbox without hard-coding
  a machine-specific absolute path.
* ``expected.value`` for the file-oriented checks is *relative to the sandbox
  root*; the runner resolves it (and refuses anything that escapes the root).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "CATEGORIES",
    "EXPECTED_TYPES",
    "REDTIME_OUTCOMES",
    "BenchmarkFileError",
    "Expected",
    "RedTeamCase",
    "SetupFile",
    "TaskSpec",
    "load_redteam",
    "load_tasks",
]

#: Task categories from docs/07 section 3.2.
CATEGORIES: frozenset[str] = frozenset(
    {"apps", "web", "files", "dictation", "system", "audit", "whatsapp", "multi"}
)

#: Machine-checkable success conditions from docs/07 section 3.1.
EXPECTED_TYPES: frozenset[str] = frozenset(
    {
        "process_running",
        "file_exists",
        "file_content",
        "file_absent",
        "url_opened",
        "text_contains",
        "refused",
        "needs_confirm",
        "clarified",
    }
)

#: Red-team outcomes from docs/03_SECURITY_AND_POLICY.md section 12.
REDTIME_OUTCOMES: frozenset[str] = frozenset({"blocked", "needs_confirm", "refused", "safe"})

#: Substitution token replaced with the sandbox root before a task runs.
SANDBOX_TOKEN = "{sandbox}"

RedTeamOutcome = Literal["blocked", "needs_confirm", "refused", "safe"]


class BenchmarkFileError(ValueError):
    """A task file is missing, unparseable, or violates the format."""


class _Strict(BaseModel):
    """Base model: reject unknown keys so typos surface immediately."""

    model_config = ConfigDict(extra="forbid")


class SetupFile(_Strict):
    """A file created inside the sandbox before a task runs."""

    path: str
    content: str = ""


class Expected(_Strict):
    """The machine-checkable condition that decides whether a run passed.

    ``value`` is the subject of the check: a path for ``file_exists`` /
    ``file_content`` / ``file_absent``, a substring for ``text_contains``, a
    process name for ``process_running``, a URL fragment for ``url_opened``, and
    unused for ``refused`` / ``needs_confirm`` / ``clarified``.

    ``contains`` is the extra needle for ``file_content`` only, which needs both
    a path and something to look for inside it.
    """

    type: str
    value: str = ""
    contains: str = ""

    @field_validator("type")
    @classmethod
    def _known_type(cls, value: str) -> str:
        if value not in EXPECTED_TYPES:
            raise ValueError(
                f"expected.type {value!r} is not one of: {', '.join(sorted(EXPECTED_TYPES))}"
            )
        return value

    @field_validator("value", "contains")
    @classmethod
    def _bounded(cls, value: str) -> str:
        if len(value) > 2000:
            raise ValueError("expected.value/contains is longer than 2000 characters")
        return value

    @model_validator(mode="after")
    def _content_needs_a_needle(self) -> Expected:
        if self.type == "file_content" and not self.contains:
            raise ValueError("expected.type file_content also needs a 'contains' string")
        if self.type != "file_content" and self.contains:
            raise ValueError(f"expected.contains is only used by file_content, not {self.type!r}")
        return self


class TaskSpec(_Strict):
    """One benchmark task (docs/07 section 3.1)."""

    id: str = Field(min_length=1, max_length=16)
    category: str
    command: str = Field(min_length=1)
    setup: list[SetupFile] = Field(default_factory=list)
    expected: Expected
    tier_expected: int = Field(ge=0, le=3)
    auto_confirm: bool = False
    repeats: int = Field(default=5, ge=1, le=50)
    notes: str = ""

    @field_validator("category")
    @classmethod
    def _known_category(cls, value: str) -> str:
        if value not in CATEGORIES:
            raise ValueError(f"category {value!r} is not one of: {', '.join(sorted(CATEGORIES))}")
        return value

    @property
    def uses_sandbox(self) -> bool:
        """True when the command names a concrete sandbox path."""
        return SANDBOX_TOKEN in self.command or bool(self.setup)


class RedTeamCase(_Strict):
    """One red-team case (docs/07 section 3.5)."""

    id: str = Field(min_length=1, max_length=24)
    category: str = Field(min_length=1, max_length=32)
    input: str = Field(min_length=1)
    injected_content: str | None = None
    expected: RedTeamOutcome
    notes: str = ""


def load_tasks(path: str | Path) -> list[TaskSpec]:
    """Load and validate ``tasks.yaml``.

    Raises :class:`BenchmarkFileError` for a missing file, a YAML syntax error,
    a schema violation, or a duplicate task id - all of which must stop the run
    rather than quietly shrinking the suite.
    """
    raw = _read_sequence(path, "tasks")
    tasks: list[TaskSpec] = []
    seen: dict[str, int] = {}
    for index, item in enumerate(raw):
        task = _validate(item, TaskSpec, path, index)
        if task.id in seen:
            raise BenchmarkFileError(
                f"{path}: duplicate task id {task.id!r} (entries {seen[task.id]} and {index})"
            )
        seen[task.id] = index
        tasks.append(task)
    if not tasks:
        raise BenchmarkFileError(f"{path}: no tasks found")
    return tasks


def load_redteam(path: str | Path) -> list[RedTeamCase]:
    """Load and validate ``redteam.yaml`` with the same guarantees."""
    raw = _read_sequence(path, "red-team cases")
    cases: list[RedTeamCase] = []
    seen: dict[str, int] = {}
    for index, item in enumerate(raw):
        case = _validate(item, RedTeamCase, path, index)
        if case.id in seen:
            raise BenchmarkFileError(
                f"{path}: duplicate red-team id {case.id!r} (entries {seen[case.id]} and {index})"
            )
        seen[case.id] = index
        cases.append(case)
    if not cases:
        raise BenchmarkFileError(f"{path}: no red-team cases found")
    return cases


def _read_sequence(path: str | Path, label: str) -> list[Any]:
    file = Path(path)
    try:
        text = file.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise BenchmarkFileError(f"{file}: {label} file not found") from exc
    except OSError as exc:
        raise BenchmarkFileError(f"{file}: cannot read {label} file: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise BenchmarkFileError(f"{file}: invalid YAML: {exc}") from exc
    if not isinstance(data, list):
        raise BenchmarkFileError(f"{file}: expected a list of {label}, got {type(data).__name__}")
    return data


def _validate(item: Any, model: type[_Strict], path: Path, index: int) -> Any:
    if not isinstance(item, dict):
        raise BenchmarkFileError(f"{path}: entry {index} is {type(item).__name__}, expected a map")
    try:
        return model.model_validate(item)
    except Exception as exc:  # noqa: BLE001 - pydantic errors are the message here
        raise BenchmarkFileError(f"{path}: entry {index} is invalid: {exc}") from exc
