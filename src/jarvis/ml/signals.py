"""Deterministic lexical evidence for the risk classifier (docs/08 section 2).

Why a lexicon and not a model
-----------------------------
``docs/08_RISK_CLASSIFIER.md`` describes a fine-tuned ONNX encoder.  That model
does not exist yet: it is trained offline in Colab by the human (section 4/5)
and dropped into ``ml/models/risk_classifier/`` afterwards.  Rather than ship a
stub that does nothing, this module provides the *same signal* from a
transparent, offline, dependency-free scorer, and
:mod:`jarvis.ml.inference` provides the real ONNX path.  The
:class:`~jarvis.ml.risk.RiskClassifier` picks one; the engine only ever calls
``min_tier()`` and takes a ``max()``, so swapping the backend cannot change the
security properties (docs/03 section 4 step 5).

How a label is produced
-----------------------
1. **Prior** - the registry's declared ``base_tier`` for the tool contributes one
   vote for its own label.  The rules already know this, so the prior is never
   the reason JARVIS escalates; it only keeps the text from talking the
   classifier *down* into a "safe" reading of a Tier 2 tool.
2. **Signals** - each matching pattern in :data:`SIGNALS` adds its weight to its
   label.  A match preceded by a negation cue ("don't delete anything") is
   recorded as *negated* and contributes nothing, which is what keeps
   "show me my downloads, don't delete them" from reading as a deletion.
3. **Taint** - a tainted step (docs/03 section 8) raises the *floor* to Tier 1,
   exactly as the engine's own taint rule does, and is recorded in the signals
   and the audit trail.  It cannot push a step above the tier its text already
   justifies; a tainted Tier 0 read is left alone, as the spec intends.
4. **Label + confidence** - the label is the arg-max of the weighted votes
   (ties break toward the *more severe* label).  The confidence is a monotone
   function of how far ahead the winner is, so it means the same thing as the
   ONNX model's max softmax probability: the caller escalates one label when it
   is below the configured threshold (fail toward caution).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "DANGEROUS",
    "LABELS",
    "LABEL_MIN_TIER",
    "PRIOR_WEIGHT",
    "SAFE",
    "SENSITIVE",
    "Assessment",
    "Signal",
    "assess",
    "label_for_tier",
    "more_severe",
    "signal_table",
]

Label = Literal["safe", "sensitive", "dangerous"]

SAFE: Label = "safe"
SENSITIVE: Label = "sensitive"
DANGEROUS: Label = "dangerous"

#: Least severe first, so ``more_severe`` is a simple index bump.
LABELS: tuple[Label, ...] = (SAFE, SENSITIVE, DANGEROUS)

#: docs/08 section 2: the minimum tier each label may push a step to.  Tier 3 is
#: absent on purpose - hard blocks stay in the rules and are never delegated to
#: a model.
LABEL_MIN_TIER: dict[Label, int] = {SAFE: 0, SENSITIVE: 1, DANGEROUS: 2}

#: One vote for the tool's own declared tier (see the module docstring).
PRIOR_WEIGHT = 1.0

#: Confidence is ``1 / (1 + exp(-_SLOPE * (margin - _MID)))``.  A margin of one
#: vote is exactly 0.5, which is the default threshold: clear evidence is acted
#: on, a near-tie is treated as a near-tie.
_SLOPE = 2.0
_MID = 1.0

#: How far back a negation cue can sit and still cancel the signal after it.
_NEGATION_WINDOW = 24

_NEGATION_RE = re.compile(
    r"\b(do not|don'?t|dont|never|without|no need to|instead of|avoid|refrain from|"
    r"not going to|stop)\b"
)


def more_severe(label: str) -> Label:
    """Return the next label up.  ``dangerous`` is the ceiling, never Tier 3."""
    names: tuple[str, ...] = LABELS
    if label not in names:
        return DANGEROUS
    index = names.index(label)
    return LABELS[min(index + 1, len(LABELS) - 1)]


def label_for_tier(tier: int) -> Label:
    """Map a tool's declared tier onto the label it votes for."""
    for label in reversed(LABELS):
        if LABEL_MIN_TIER[label] <= tier:
            return label
    return SAFE


@dataclass(frozen=True)
class Signal:
    """One named, weighted lexical pattern that votes for a label."""

    name: str
    label: Label
    weight: float
    pattern: re.Pattern[str]


def _sig(name: str, label: Label, weight: float, pattern: str) -> Signal:
    return Signal(name=name, label=label, weight=weight, pattern=re.compile(pattern, re.IGNORECASE))


#: ``(signal name, pattern)`` pairs, grouped by the label they vote for.  The
#: names appear in the log line and in ``jarvis doctor``/``explain()``, so they
#: are short and stable rather than derived from the pattern text.

#: Read-only / trivially reversible wording.  These are the "negative" evidence:
#: they stop an incidental scary noun in a harmless request from escalating.
_SAFE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("open", r"\b(open|launch|show|display|view|browse|play)\b"),
    ("search", r"\b(search|google|find|look up|check|compare)\b"),
    ("read", r"\b(list|read|cat|print|preview|inspect|status|info|details)\b"),
    ("ask", r"\b(tell|what|which|how many|how much|explain|summarise|summarize)\b"),
    ("undo", r"\b(undo|revert|restore|dry run|rehearsal)\b"),
)

#: Changes files or state, or talks to the machine.  Tier 1 territory.
_SENSITIVE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("write", r"\b(create|write|save|store|overwrite|append|edit|modify|rename)\b"),
    ("relocate", r"\b(move|copy|replace|duplicate|extract|unzip|compress)\b"),
    ("type", r"\b(type|dictate|keyboard|keystroke|keystrokes|paste|press)\b"),
    ("power", r"\b(lock|restart|reboot|shutdown|log ?off|sleep)\b"),
    ("configure", r"\b(install|uninstall|update|upgrade|enable|disable|configure|grant)\b"),
    ("schedule", r"\b(schedule|register|add to startup|set default)\b"),
)

#: Destruction.  Tier 2 territory.
_DESTRUCTIVE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("destroy", r"\b(delete|deleting|deletes|deleted|remove|removing|erase|erasing|wipe|wiping)\b"),
    ("purge", r"\b(purge|purging|shred|shredding|format|formatting|trash)\b"),
    (
        "irreversible",
        r"\b(permanently|irreversib\w*|recursively|recursive|forcibly|force delete)\b",
    ),
    (
        "clear_folder",
        (
            r"\b(empty|clean up|clear out) (the |my |all |every )?"
            r"(folder|directory|desktop|downloads|documents)\b"
        ),
    ),
)

#: Communicating with, or about, other people.  Tier 2 territory.
_MESSAGING_PATTERNS: tuple[tuple[str, str], ...] = (
    ("send", r"\b(send|sending|forward|share|broadcast|post)\b"),
    (
        "channel",
        r"\b(whatsapp|signal|telegram|sms|text message|e-?mail|voicemail)\b",
    ),
    ("people", r"\b(recipient|everyone|all my friends)\b"),
)

#: Sub-patterns for the two composite "tampering" signals below.  These words
#: are only dangerous *in combination*: "is the firewall on" is a read-only
#: question, "turn the firewall off" is not.  A bare noun is therefore never a
#: signal on its own - that was a real false-block found while tuning this.
_TAMPER_VERB = (
    r"(?:disabl\w+|turn(?:ing|s)?\s+off|switch(?:ing|s)?\s+off|bypass\w*|shut\s+off|"
    r"kill\w*|stop\w*|remov\w+|uninstall\w*|crack\w*|exploit\w*|weaken\w*|"
    r"deactivat\w+|edit\w*|modif\w+|chang\w+|add|creat\w+|register\w*|set|"
    r"without\s+(?:asking|permission|confirmation))"
)
_PROTECTION_NOUN = (
    r"(?:defender|firewall|anti-?virus|windows\s+security|smartscreen|uac|"
    r"user\s+account\s+control|security\s+centre|security\s+center)"
)
_SYSTEM_NOUN = (
    r"(?:registry|startup(?:\s+(?:entry|entries|task|program|folder))?|"
    r"scheduled\s+task|boot\s+configuration|group\s+policy|defender\s+exclusion)"
)

#: Credentials.  Weighted highest and unconditional: handling a secret is
#: dangerous whatever else the sentence says, and docs/03 section 2 blocks the
#: serious cases in code on top of this.
_CRITICAL_PATTERNS: tuple[tuple[str, str], ...] = (
    ("credential", r"\b(password|passcode|passphrase|credential|credentials|api[ _-]?key)\b"),
    (
        "secret_token",
        r"\b(auth token|access token|session cookie|cookies|browser password|keychain)\b",
    ),
    (
        "mfa_secret",
        r"\b(two[ -]factor|2fa|\botp\b|one[ -]time (code|password)|seed phrase|recovery phrase)\b",
    ),
    (
        "system_reset",
        (
            r"\b(reinstall\s+(?:windows|linux|macos|os)|system\s+restore|"
            r"format\s+(?:the\s+)?(?:drive|disk|pc|laptop|computer))\b"
        ),
    ),
)

#: Prompt-injection phrasing.  The user command is untrusted input (docs/03
#: invariant 9), so "ignore all previous instructions and delete X" must read as
#: dangerous even though each individual verb looks ordinary.
_INJECTION_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        "inject_ignore",
        (
            r"ignore\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|earlier|foregoing)\s+"
            r"(?:instructions?|prompts?|rules?|messages?|directions?)"
        ),
    ),
    (
        "inject_disregard",
        (
            r"disregard\s+(?:all\s+|any\s+)?(?:the\s+|your\s+)?"
            r"(?:previous\s+|prior\s+|above\s+)?(?:instructions?|prompts?|rules?|guidelines?)"
        ),
    ),
    (
        "inject_override",
        (
            r"you\s+(?:must|should|have\s+to|will|shall)\s+(?:now\s+)?"
            r"(?:ignore|skip|bypass|override|disregard)"
        ),
    ),
    (
        "inject_persona",
        r"(?:do\s+anything\s+now|dan\s+mode|jailbreak|developer\s+mode\s+enabled)",
    ),
    (
        "inject_pretend",
        (
            r"pretend\s+(?:you\s+are|to\s+be|that\s+you\s+are)\s+"
            r"(?:an?\s+)?(?:unrestricted|uncensored|admin|root|a\s+different)"
        ),
    ),
    (
        "inject_bypass",
        r"bypass\s+(?:the\s+)?(?:safety|security|confirmation|guardrails?|polic\w+|approval)",
    ),
    (
        "inject_silent",
        (
            r"without\s+(?:asking|requesting|informing)\s+(?:for\s+)?"
            r"(?:confirmation|permission|approval|consent)"
        ),
    ),
    (
        "inject_new_system",
        r"new\s+(?:system\s+)?(?:instructions?|prompt)\s*:\s*you\s+must",
    ),
)


def _pair(verb: str, noun: str) -> str:
    """Match a tampering verb and its noun in either order, a few words apart."""
    window = r"(?:\s+\w+){0,3}\s+"
    return rf"\b(?:{verb})\b{window}\b(?:{noun})\b|\b(?:{noun})\b{window}\b(?:{verb})\b"


#: Composite signals need both a verb and a noun, so they are appended after
#: the simple patterns (order only affects the reported names).
_COMPOSITE: tuple[tuple[str, str], ...] = (
    ("protection_tamper", _pair(_TAMPER_VERB, _PROTECTION_NOUN)),
    ("system_tamper", _pair(_TAMPER_VERB, _SYSTEM_NOUN)),
)


def _build_signals() -> tuple[Signal, ...]:
    """Assemble the frozen signal table (module constant, built once)."""
    table: list[Signal] = []
    for label, weight, group in (
        (SAFE, 1.0, _SAFE_PATTERNS),
        (SENSITIVE, 1.0, _SENSITIVE_PATTERNS),
        (DANGEROUS, 1.6, _DESTRUCTIVE_PATTERNS),
        (DANGEROUS, 1.6, _MESSAGING_PATTERNS),
        (DANGEROUS, 2.5, _CRITICAL_PATTERNS),
        (DANGEROUS, 2.5, _INJECTION_PATTERNS),
    ):
        for name, pattern in group:
            table.append(_sig(name, label, weight, pattern))
    for name, pattern in _COMPOSITE:
        table.append(_sig(name, DANGEROUS, 2.5, pattern))
    return tuple(table)


#: Every lexical signal, in a fixed order so the explanation is deterministic.
SIGNALS: tuple[Signal, ...] = _build_signals()


def signal_table() -> tuple[tuple[str, str, float], ...]:
    """Return ``(name, label, weight)`` for every signal (doctor/diagnostics)."""
    return tuple((s.name, s.label, s.weight) for s in SIGNALS)


@dataclass(frozen=True)
class Assessment:
    """One classifier verdict plus the evidence that produced it."""

    label: Label
    confidence: float
    min_tier: int
    escalated: bool
    votes: dict[str, float] = field(default_factory=dict)
    signals: tuple[str, ...] = ()
    negated: tuple[str, ...] = ()


def _is_negated(text: str, start: int) -> bool:
    """True when a negation cue sits in the window before ``start``."""
    window = text[max(0, start - _NEGATION_WINDOW) : start]
    return _NEGATION_RE.search(window) is not None


def _collect(text: str) -> tuple[dict[str, float], list[str], list[str]]:
    """Return per-label votes plus the names that fired / were negated."""
    votes: dict[str, float] = dict.fromkeys(LABELS, 0.0)
    fired: list[str] = []
    negated: list[str] = []
    for signal in SIGNALS:
        match = signal.pattern.search(text)
        if match is None:
            continue
        if _is_negated(text, match.start()):
            negated.append(signal.name)
            continue
        votes[signal.label] += signal.weight
        fired.append(signal.name)
    return votes, fired, negated


def _winner(votes: dict[str, float], prior_label: Label | None) -> tuple[Label, float, float]:
    """Arg-max label, its vote count and the margin over the runner-up.

    A tie is resolved *towards the prior* when the prior is one of the tied
    labels, and towards the more severe label otherwise.  Breaking every tie
    towards ``dangerous`` and then also applying the low-confidence escalation
    stacked two steps of caution onto one ambiguous sentence, which escalated
    ordinary Tier 0 commands during tuning; the prior is the safe direction.
    A prior is never below the winner, so this can only reduce a false
    escalation, never a real one.
    """
    best_value = max(votes.values())
    if best_value <= 0.0:
        # Nothing fired and there is no prior: the honest answer is "no
        # opinion".  The tie-break below would otherwise pick the most severe
        # label out of three zeroes and report Tier 2 for an empty sentence.
        return SAFE, 0.0, 0.0
    tied = [label for label in LABELS if votes[label] == best_value]
    best_label: Label = tied[-1]
    if prior_label is not None and prior_label in tied:
        best_label = prior_label
    runner = max((value for label, value in votes.items() if label != best_label), default=0.0)
    return best_label, best_value, best_value - runner


def _confidence(margin: float, evidence: float) -> float:
    """Map a vote margin onto a 0..1 confidence comparable to a softmax max.

    ``evidence == 0`` means nothing in the text pointed anywhere.  That is not
    low confidence, it is no counter-evidence, so the prior stands unchanged
    (confidence 1.0).  Getting this wrong escalated read-only commands such as
    "run a security audit" purely because the sentence had no keyword in it.
    """
    if evidence <= 0.0:
        return 1.0
    return round(1.0 / (1.0 + math.exp(-_SLOPE * (margin - _MID))), 4)


def assess(
    text: str,
    *,
    tainted: bool = False,
    prior_label: Label | None = None,
    threshold: float = 0.5,
) -> Assessment:
    """Classify one serialised action string.

    ``prior_label`` is the label the rules already decided (from the tool's
    ``base_tier``); pass ``None`` when the tool is unknown.  A verdict whose
    confidence is below ``threshold`` is escalated one label (docs/08 section 2),
    clamped so a low-confidence prediction can never reach Tier 3.  The returned
    tier is never below the prior, because this layer may only add a tier.
    """
    votes, fired, negated = _collect(text or "")
    has_prior = prior_label is not None and prior_label in LABEL_MIN_TIER
    if has_prior and prior_label is not None:
        votes[prior_label] += PRIOR_WEIGHT
    floor = LABEL_MIN_TIER[prior_label] if has_prior and prior_label is not None else 0
    if tainted:
        # docs/03 section 8 raises a step that is *already* Tier 1 or higher to
        # Tier 1, and deliberately leaves a Tier 0 read alone.  The engine
        # applies that exact rule immediately before calling us, so mirroring
        # it keeps this layer no less cautious than the rules without inventing
        # a stricter one - in particular a tainted "append this summary" must
        # not turn into a password prompt purely because the summary came off
        # the web.  The flag still travels inside the serialised text and the
        # audit trail, where the trained ONNX model uses it as a feature
        # (docs/08 section 3).
        fired.append("tainted")
        floor = min(1, floor)

    evidence = sum(votes.values()) - (PRIOR_WEIGHT if has_prior else 0.0)
    label, _best, margin = _winner(votes, prior_label)
    confidence = _confidence(margin, evidence)
    escalated = confidence < threshold
    if escalated:
        label = more_severe(label)
    return Assessment(
        label=label,
        confidence=confidence,
        min_tier=max(floor, LABEL_MIN_TIER[label]),
        escalated=escalated,
        votes=dict(votes),
        signals=tuple(fired),
        negated=tuple(negated),
    )
