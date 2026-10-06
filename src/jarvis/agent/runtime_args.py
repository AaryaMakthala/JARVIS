"""Per-step markers for arguments that depend on runtime output.

Pure, side-effect free, no I/O, no Tier / hash / decision changes.  The only
thing this module does is label a Step so a later stage can choose between
batch-approving it (args fully known at plan time) and re-gating it individually
with a short readback (args derive from earlier runtime output).

Invariant this module must never violate: it does **not** change args, tiers,
hashes, decisions, or any other field of a Step or Plan.  It only sets the new
``resolved_from_runtime`` flag on copies of steps.
"""

from __future__ import annotations

from typing import Any

#: Substring patterns that look like a reference to earlier runtime output.
#: None of these are the repo's established cross-step syntax today (the brain
#: assigns step ids ``s1...`` but the planner does not reference earlier step
#: ids inside args anywhere I could find); they are used conservatively as a
#: fail-closed signal for 3c-2.
_RUNTIME_REF_SUBSTRINGS = (
    "$",
    "{{",
    "}}",
    "<step",
    "output of",
)


def mark_runtime_args(plan: PlanT) -> None:
    """Set ``resolved_from_runtime`` on each step of *plan* when appropriate.

    Runs for the initial plan and for replans.  Mutates the steps in place
    (the Step objects already live inside the Plan that validate owns); does not
    touch args, tiers, hashes, decisions, or any other field.
    """
    for step in plan.steps:
        if _step_is_runtime_dependent(step):
            step.resolved_from_runtime = True


def _step_is_runtime_dependent(step: Any) -> bool:
    """Return True when the step's arguments depend on runtime output.

    Fail closed: return True whenever unsure.
    """
    # (a) The LLM already asserted the args derive from untrusted text.
    if bool(getattr(step, "depends_on_untrusted", False)):
        return True

    # (b) Any string value (including inside nested lists/dicts) looks like a
    #     reference to an earlier step's output.
    return _any_value_refs_runtime(step.args)


def _any_value_refs_runtime(value: Any) -> bool:
    """True when *value* (possibly nested) contains a runtime-reference pattern."""
    if isinstance(value, str):
        lower = value.lower()
        for pattern in _RUNTIME_REF_SUBSTRINGS:
            if pattern in lower:
                return True
        return False
    if isinstance(value, dict):
        return any(_any_value_refs_runtime(v) for v in value.values())
    if isinstance(value, list):
        return any(_any_value_refs_runtime(item) for item in value)
    return False


# ---------------------------------------------------------------------------
# Types shim so this module compiles without importing the real Step at module
# level (keeps it importable even if state.py has a transient issue).  The real
# Step is used at runtime via duck typing: we only read ``depends_on_untrusted``
# and write ``resolved_from_runtime``.
# ---------------------------------------------------------------------------

StepT = Any
PlanT = Any
