"""Documented entry point for benchmark analysis (docs/07 section 4.4).

The implementation lives in :mod:`jarvis.benchmark.analyze` so that
``jarvis benchmark report`` works from an installed wheel.  This file exists
because docs/07 names ``benchmarks/analyze.py`` as the analyzer's home, and both
paths now run the same code.

    python benchmarks/analyze.py benchmarks/results
    python benchmarks/analyze.py benchmarks/results --markdown
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:  # run from a checkout without installing
    sys.path.insert(0, str(REPO_ROOT / "src"))

from jarvis.benchmark.cli import main  # noqa: E402

if __name__ == "__main__":
    # Force the `report` subcommand so this file behaves like docs/07 describes.
    raise SystemExit(main(["report", *sys.argv[1:]]))
