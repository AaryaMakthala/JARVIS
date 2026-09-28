"""Phase 9 benchmark harness (docs/07_TESTING_AND_BENCHMARK.md).

The package form of ``benchmarks/runner.py`` and ``benchmarks/analyze.py``: the
logic lives inside the ``jarvis`` distribution so ``jarvis benchmark run`` is a
real command on a real install, and so the harness is unit-testable with a
``FakeLLM`` instead of needing a provider key.  The two files in ``benchmarks/``
are thin shims onto this package.
"""

from __future__ import annotations

from jarvis.benchmark.analyze import (
    ConfusionTable,
    Summary,
    analyze,
    ci95,
    load_rows,
    render_markdown,
    render_text,
)
from jarvis.benchmark.checkers import CheckResult, Observation, check_expected
from jarvis.benchmark.profiles import (
    PROFILES,
    AblationProfile,
    ProfileError,
    apply_profile,
    get_profile,
    profile_names,
)
from jarvis.benchmark.runner import (
    CSV_COLUMNS,
    RunOptions,
    RunRecord,
    run_suite,
    scripted_answer,
    write_csv,
)
from jarvis.benchmark.sandbox import SandboxTrash, reset_sandbox
from jarvis.benchmark.tasks import (
    BenchmarkFileError,
    RedTeamCase,
    TaskSpec,
    load_redteam,
    load_tasks,
)

__all__ = [
    "CSV_COLUMNS",
    "PROFILES",
    "AblationProfile",
    "BenchmarkFileError",
    "CheckResult",
    "ConfusionTable",
    "Observation",
    "ProfileError",
    "RedTeamCase",
    "RunOptions",
    "RunRecord",
    "SandboxTrash",
    "Summary",
    "TaskSpec",
    "analyze",
    "apply_profile",
    "check_expected",
    "ci95",
    "get_profile",
    "load_redteam",
    "load_rows",
    "load_tasks",
    "profile_names",
    "render_markdown",
    "render_text",
    "reset_sandbox",
    "run_suite",
    "scripted_answer",
    "write_csv",
]
