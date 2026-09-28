"""ONNX inference for the trained risk classifier (docs/08 sections 5-6).

This is the path the human's Colab-trained model takes: ``onnxruntime`` on the
CPU execution provider plus the ``tokenizers`` tokenizer, and **never** ``torch``
at runtime.  Everything is optional - the ``[ml]`` extra is the only thing that
brings these packages in, and a missing package, a missing model file or a
corrupt model is reported as :class:`ClassifierUnavailable` rather than raised,
so the agent keeps working on the rules alone (docs/08 section 6).

The expected layout under ``ml/models/risk_classifier/`` is::

    model.onnx  tokenizer.json  labels.json  threshold.json

``labels.json`` is ``["safe", "sensitive", "dangerous"]`` (the model output
index -> label) and ``threshold.json`` is ``{"threshold": 0.62}``.  Anything
unreadable falls back to the defaults declared here, so a hand-copied model
folder still runs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_LABELS",
    "DEFAULT_THRESHOLD",
    "ClassifierUnavailable",
    "OnnxPredictor",
    "load_threshold",
    "read_labels",
]

#: Model output index -> label.  Matches docs/08 section 2.
DEFAULT_LABELS: tuple[str, ...] = ("safe", "sensitive", "dangerous")

#: Used when ``threshold.json`` is absent.  docs/08 section 2 asks for a
#: threshold tuned on the validation set; 0.5 is the neutral choice until the
#: trained model supplies its own.
DEFAULT_THRESHOLD = 0.5

logger = logging.getLogger("jarvis.ml")


class ClassifierUnavailable(RuntimeError):
    """The ONNX model cannot be used; the caller must fall back to the rules."""


def _read_json(path: Path) -> Any:
    """Read a small JSON file, or raise :class:`ClassifierUnavailable`."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ClassifierUnavailable(f"cannot read {path.name}: {exc}") from exc


def read_labels(model_dir: Path) -> tuple[str, ...]:
    """Return the model's label order, or the documented default."""
    path = model_dir / "labels.json"
    if not path.is_file():
        return DEFAULT_LABELS
    data = _read_json(path)
    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise ClassifierUnavailable(f"{path.name} must be a list of label strings")
    return tuple(data)


def load_threshold(model_dir: Path) -> float:
    """Return the model's tuned decision threshold, or the default."""
    path = model_dir / "threshold.json"
    if not path.is_file():
        return DEFAULT_THRESHOLD
    data = _read_json(path)
    value = data.get("threshold") if isinstance(data, dict) else data
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ClassifierUnavailable(f"{path.name} must hold a number")
    return float(value)


@dataclass
class OnnxPredictor:
    """A loaded, ready-to-call ONNX classifier.

    Build one with :meth:`load`; constructing the dataclass directly with
    ``None`` fields is only useful for tests that inject a fake session.
    """

    session: Any
    tokenizer: Any
    labels: tuple[str, ...]
    max_length: int = 128

    @classmethod
    def load(cls, model_dir: Path) -> OnnxPredictor:
        """Load the tokenizer and the CPU ONNX session from ``model_dir``.

        Raises :class:`ClassifierUnavailable` for every foreseeable problem
        (optional dependency absent, file missing, session refused) so the
        caller degrades instead of crashing the agent.
        """
        model_path = model_dir / "model.onnx"
        tokenizer_path = model_dir / "tokenizer.json"
        if not model_path.is_file():
            raise ClassifierUnavailable(f"no model.onnx in {model_dir}")
        if not tokenizer_path.is_file():
            raise ClassifierUnavailable(f"no tokenizer.json in {model_dir}")
        try:
            import onnxruntime  # optional [ml] extra, imported lazily by design
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise ClassifierUnavailable(
                f"install the [ml] extra to use the ONNX classifier ({exc})"
            ) from exc
        try:
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            session = onnxruntime.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
        except Exception as exc:  # any load failure degrades to the lexical backend
            raise ClassifierUnavailable(f"could not load {model_path.name}: {exc}") from exc
        logger.info("risk classifier ONNX model loaded from %s", model_dir)
        return cls(session=session, tokenizer=tokenizer, labels=read_labels(model_dir))

    def predict(self, text: str) -> tuple[str, float]:
        """Return ``(label, confidence)`` for one serialised action.

        ``confidence`` is the max softmax probability, i.e. exactly the quantity
        docs/08 section 2 compares against the threshold.
        """
        import numpy as np  # only needed for the tensor conversion

        encoded = self.tokenizer.encode(text)
        ids = encoded.ids[: self.max_length]
        mask = encoded.attention_mask[: self.max_length]
        inputs = {
            "input_ids": np.array([ids], dtype=np.int64),
            "attention_mask": np.array([mask], dtype=np.int64),
        }
        feeds = {
            spec.name: inputs[spec.name]
            for spec in self.session.get_inputs()
            if spec.name in inputs
        }
        logits = self.session.run(None, feeds)[0]
        probabilities = _softmax(np.asarray(logits, dtype="float64").reshape(-1))
        best = int(probabilities.argmax())
        label = self.labels[best] if best < len(self.labels) else DEFAULT_LABELS[-1]
        return label, round(float(probabilities[best]), 4)


def _softmax(logits: Any) -> Any:
    """Numerically stable softmax over a 1-D logit vector."""
    import numpy as np

    shifted = logits - logits.max()
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum()
