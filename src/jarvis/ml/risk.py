"""The risk classifier JARVIS runs on every planned action (Phase 8).

``docs/08_RISK_CLASSIFIER.md`` is the design; this module is the integration
point named in its section 6.  The safety contract, restated because it is the
only part that matters:

* **The classifier can only raise a tier.**  :meth:`RiskClassifier.min_tier`
    returns a value the engine feeds to ``max()`` (docs/03 section 4, step 5).
    There is no code path by which it lowers a tier, and it can never return 3 -
    Tier 3 hard blocks stay in the rules and are never delegated to a model.
* **It never crashes the agent.**  :meth:`min_tier` catches everything and
    answers ``0`` (i.e. "no opinion") on failure; a missing optional dependency
    or a broken model file downgrades the backend at construction time.
* **It never weakens confirmation.**  Escalating a tier *adds* a confirmation,
    an unlock requirement or nothing at all.  Nothing in this module touches
    ``action_hash``, the registry or the rules.

Two backends, one interface:

``lexical``
    Deterministic, offline, dependency-free weighted-lexicon scorer
    (:mod:`jarvis.ml.signals`).  This is the default and it is the only backend
    that needs no weights on disk.
``onnx``
    The fine-tuned model the human trains in Colab (docs/08 sections 4-5),
    served by :mod:`jarvis.ml.inference`.  Selected when a model directory is
    configured and loadable; otherwise the classifier warns and the agent
    continues on the rules alone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jarvis.ml import serialize, signals
from jarvis.ml.inference import ClassifierUnavailable, OnnxPredictor, load_threshold

__all__ = [
    "BACKEND_AUTO",
    "BACKEND_LEXICAL",
    "BACKEND_ONNX",
    "DEFAULT_MODEL_DIR",
    "DEFAULT_THRESHOLD",
    "Prediction",
    "RiskClassifier",
    "build_classifier",
    "model_dir",
]

BACKEND_LEXICAL = "lexical"
BACKEND_ONNX = "onnx"
BACKEND_AUTO = "auto"

_LOG = logging.getLogger("jarvis.ml")

#: Where ``docs/08`` section 5 says the exported model is packaged.
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models" / "risk_classifier"

DEFAULT_THRESHOLD = 0.5


@dataclass(frozen=True)
class Prediction:
    """One classifier verdict, with enough detail to explain it."""

    label: str
    confidence: float
    min_tier: int
    backend: str
    escalated: bool = False
    signals: tuple[str, ...] = ()

    def reason(self) -> str:
        """Short, log-safe explanation of why this tier was proposed."""
        detail = f" label={self.label} confidence={self.confidence:.2f}"
        if self.escalated:
            detail += " escalated=low-confidence"
        if self.signals:
            detail += f" signals={','.join(self.signals[:4])}"
        return f"risk classifier ({self.backend}){detail}"


def model_dir(configured: str = "") -> Path:
    """Resolve the model directory: the configured one, else the packaged one."""
    return Path(configured).expanduser() if configured else DEFAULT_MODEL_DIR


def _has_model(directory: Path) -> bool:
    """True when a trained model is present to load."""
    return (directory / "model.onnx").is_file()


class RiskClassifier:
    """Escalate-only risk classifier for planned actions.

    ``prior_tiers`` is the registry's declared ``base_tier`` per tool name.  It
    is the classifier's starting point (docs/08 section 2's "rules already know
    this"), and it is the reason an unremarkable command for a Tier 2 tool is not
    read as harmless.  An unknown tool gets no prior, so only the text decides.
    """

    def __init__(
        self,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        prior_tiers: dict[str, int] | None = None,
        predictor: OnnxPredictor | None = None,
        backend: str = BACKEND_LEXICAL,
        model_path: Path | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.threshold = threshold
        self.prior_tiers = dict(prior_tiers or {})
        self._predictor = predictor
        self._model_path = model_path
        self._log = logger or logging.getLogger("jarvis.ml")
        self.backend = backend
        if predictor is not None:
            self.backend = BACKEND_ONNX

    # â”€â”€ construction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @classmethod
    def from_registry(
        cls,
        registry: Any,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        backend: str = BACKEND_AUTO,
        model_path: Path | None = None,
        logger: logging.Logger | None = None,
    ) -> RiskClassifier:
        """Build a classifier that takes its priors from ``registry``."""
        log = logger or logging.getLogger("jarvis.ml")
        priors = {spec.name: int(spec.base_tier) for spec in registry.iter_all()}
        directory = model_path or DEFAULT_MODEL_DIR
        predictor: OnnxPredictor | None = None
        resolved = BACKEND_LEXICAL
        if backend == BACKEND_ONNX or (backend == BACKEND_AUTO and _has_model(directory)):
            try:
                predictor = OnnxPredictor.load(directory)
                threshold = load_threshold(directory) if backend == BACKEND_AUTO else threshold
                resolved = BACKEND_ONNX
            except ClassifierUnavailable as exc:
                if backend == BACKEND_ONNX:
                    # Explicitly requested and unusable: run on the rules alone,
                    # exactly as docs/08 section 6 requires.  Never silently
                    # pretend the model is in play.
                    log.warning("risk classifier disabled: %s", exc)
                    resolved = "disabled"
                else:
                    log.info("risk classifier falling back to the lexical backend: %s", exc)
        return cls(
            threshold=threshold,
            prior_tiers=priors,
            predictor=predictor,
            backend=resolved,
            model_path=directory,
            logger=log,
        )

    # â”€â”€ the interface the policy engine uses â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def min_tier(self, step: Any, user_input: str = "", tainted: bool | None = None) -> int:
        """Return the minimum tier this action implies, 0..2.

        Contract: never raises, never returns a negative or Tier 3 value, and
        the caller must combine it with ``max()`` (docs/03 section 4, step 5).
        """
        try:
            return self.predict(step, user_input, tainted).min_tier
        except Exception:  # noqa: BLE001 - a classifier must never break a task
            self._safe_warn("risk classifier failed; continuing on the rules only")
            return 0

    def predict(self, step: Any, user_input: str = "", tainted: bool | None = None) -> Prediction:
        """Classify one planned step and explain the verdict.

        ``tainted=None`` falls back to the step's own ``depends_on_untrusted``
        flag; the engine passes its computed value explicitly, because that
        value also accounts for tainted text the LLM never mentioned
        (docs/03 section 8).
        """
        flag = (
            bool(getattr(step, "depends_on_untrusted", False)) if tainted is None else bool(tainted)
        )
        text = serialize.serialize_step(step, user_input, flag)
        tool = str(getattr(step, "tool", ""))
        if self._predictor is not None:
            return self._predict_onnx(text, self._predictor)
        return self._predict_lexical(text, tool, flag)

    def explain(
        self, step: Any, user_input: str = "", tainted: bool | None = None
    ) -> dict[str, Any]:
        """Diagnostic view of a verdict (never contains the raw input)."""
        prediction = self.predict(step, user_input, tainted)
        return {
            "tool": str(getattr(step, "tool", "")),
            "label": prediction.label,
            "confidence": prediction.confidence,
            "min_tier": prediction.min_tier,
            "backend": prediction.backend,
            "escalated": prediction.escalated,
            "tainted": bool(
                getattr(step, "depends_on_untrusted", False) if tainted is None else tainted
            ),
            "signals": list(prediction.signals),
        }

    @property
    def active(self) -> bool:
        """False when no backend could be loaded (the rules run alone)."""
        return self.backend != "disabled"

    # â”€â”€ backends â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _predict_lexical(self, text: str, tool: str, tainted: bool) -> Prediction:
        prior: signals.Label | None = None
        if tool in self.prior_tiers:
            prior = signals.label_for_tier(self.prior_tiers[tool])
        result = signals.assess(text, tainted=tainted, prior_label=prior, threshold=self.threshold)
        return Prediction(
            label=result.label,
            confidence=result.confidence,
            min_tier=result.min_tier,
            backend=BACKEND_LEXICAL,
            escalated=result.escalated,
            signals=result.signals,
        )

    def _predict_onnx(self, text: str, predictor: OnnxPredictor) -> Prediction:
        label, confidence = predictor.predict(text)
        known = label if label in signals.LABELS else None
        if known is None:
            # An unrecognised label means the model and this code disagree, so
            # it gets no vote: the rules decide, which is the fail-closed choice
            # for a layer that is only allowed to add a tier.
            self._safe_warn(f"risk classifier returned an unknown label {label!r}; ignoring it")
            return Prediction(
                label=signals.SAFE,
                confidence=confidence,
                min_tier=0,
                backend=BACKEND_ONNX,
                escalated=False,
            )
        tier = signals.LABEL_MIN_TIER[known]
        escalated = confidence < self.threshold
        if escalated:
            escalated_label = signals.more_severe(known)
            tier = max(tier, signals.LABEL_MIN_TIER[escalated_label])
            known = escalated_label
        return Prediction(
            label=known,
            confidence=confidence,
            min_tier=tier,
            backend=BACKEND_ONNX,
            escalated=escalated,
        )

    def _safe_warn(self, message: str) -> None:
        """Log without ever raising (a broken logger must not break a task)."""
        try:
            self._log.warning(message)
        except Exception:  # noqa: BLE001 - a failing logger is the failure we are handling
            _LOG.warning("risk classifier logger failed; warning suppressed: %s", message)


def build_classifier(
    settings: Any,
    registry: Any,
    logger: logging.Logger | None = None,
) -> RiskClassifier | None:
    """Build the classifier described by ``[risk]`` in config, or ``None``.

    ``None`` means "off": the engine then applies rules only, which is the
    behaviour of every JARVIS build before Phase 8.  No exception escapes - a
    broken classifier configuration disables the layer instead of the agent.
    """
    log = logger or logging.getLogger("jarvis.ml")
    try:
        risk = getattr(settings, "risk", None)
        if risk is None or not getattr(risk, "enabled", False):
            return None
        classifier = RiskClassifier.from_registry(
            registry,
            threshold=float(getattr(risk, "threshold", DEFAULT_THRESHOLD)),
            backend=str(getattr(risk, "backend", BACKEND_AUTO)),
            model_path=model_dir(str(getattr(risk, "model_dir", "") or "")),
            logger=log,
        )
        if not classifier.active:
            return None
        return classifier
    except Exception:
        log.warning("risk classifier could not be built; continuing without it", exc_info=True)
        return None
