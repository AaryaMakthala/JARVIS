"""Documented entry point for the benchmark runner (docs/07 section 4).

The implementation lives in :mod:`jarvis.benchmark.runner` so that
``jarvis benchmark run`` works from an installed wheel and so the harness is
unit-testable with a ``FakeLLM``.  This file exists because
docs/05_BUILD_PLAN.md and docs/07 name ``benchmarks/runner.py`` as the runner's
home, and both paths now run the same code.

    python benchmarks/runner.py --config full --repeats 5
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:  # run from a checkout without installing
    sys.path.insert(0, str(REPO_ROOT / "src"))

from jarvis.benchmark.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
