"""Permission tiers for the JARVIS policy engine.

The four tiers follow docs/03_SECURITY_AND_POLICY.md section 3:

* 0 Safe - no lasting change or harm; runs automatically (logged).
* 1 Confirm - reversible change; the user must approve the exact action.
* 2 Confirm + unlocked - hard to reverse or affects others; approve **and**
  have an unlocked session (password in the daemon layer, Phase 2).
* 3 Blocked - attempted in code with a hard refusal; no tool is ever
  registered at this tier and rules also block if a plan tries.

Invariant: the final tier for a step is ``max(base, rules, classifier,
taint)`` - nothing may lower it.  The planner LLM never sees tiers and its
output is never consulted when computing them (see ``engine.decide``).
"""

from __future__ import annotations

#: Tier values used throughout the codebase.
TIER_SAFE = 0
TIER_CONFIRM = 1
TIER_CONFIRM_UNLOCK = 2
TIER_BLOCKED = 3

#: Tiers a tool may be *registered* at.  Tier 3 is never a registered tool.
REGISTRABLE_TIERS = (TIER_SAFE, TIER_CONFIRM, TIER_CONFIRM_UNLOCK)

#: Human-readable labels (for confirmations and logs).
TIER_LABELS: dict[int, str] = {
    TIER_SAFE: "Tier 0 - Safe",
    TIER_CONFIRM: "Tier 1 - Confirm",
    TIER_CONFIRM_UNLOCK: "Tier 2 - Confirm + unlocked",
    TIER_BLOCKED: "Tier 3 - Blocked",
}


def tier_label(tier: int) -> str:
    """Return the human-readable label for a tier (unknown tiers are flagged)."""
    if tier not in TIER_LABELS:
        raise ValueError(f"tier must be 0..3, got {tier!r}")
    return TIER_LABELS[tier]


def valid_tier(tier: int) -> bool:
    """Return ``True`` when ``tier`` is a recognised tier value."""
    return tier in TIER_LABELS
