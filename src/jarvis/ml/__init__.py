"""Local risk classification for planned actions (Phase 8).

Layout:

``serialize``  one action -> one deterministic string (docs/08 section 3)
``signals``    deterministic lexical evidence and the label arithmetic
``inference``  the ONNX path for the trained model (docs/08 sections 5-6)
``risk``       :class:`RiskClassifier`, the escalate-only integration point

The policy engine only ever calls ``RiskClassifier.min_tier()`` and takes a
``max()`` with its own tier (docs/03 section 4, step 5), so the backend can be
replaced without touching the security properties.
"""

from __future__ import annotations

from jarvis.ml.risk import (
    BACKEND_AUTO,
    BACKEND_LEXICAL,
    BACKEND_ONNX,
    DEFAULT_MODEL_DIR,
    DEFAULT_THRESHOLD,
    Prediction,
    RiskClassifier,
    build_classifier,
    model_dir,
)
from jarvis.ml.signals import DANGEROUS, LABEL_MIN_TIER, SAFE, SENSITIVE, Label, more_severe

__all__ = [
    "BACKEND_AUTO",
    "BACKEND_LEXICAL",
    "BACKEND_ONNX",
    "DANGEROUS",
    "DEFAULT_MODEL_DIR",
    "DEFAULT_THRESHOLD",
    "LABEL_MIN_TIER",
    "SAFE",
    "SENSITIVE",
    "Label",
    "Prediction",
    "RiskClassifier",
    "build_classifier",
    "model_dir",
    "more_severe",
]
