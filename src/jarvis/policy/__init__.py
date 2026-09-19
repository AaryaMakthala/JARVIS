"""Deterministic safety policy engine.

The LLM only proposes; this package decides (docs/03_SECURITY_AND_POLICY.md).
Invariants that must never be weakened live here and in ``engine.py``: tiers
come from base_tier + rules (never LLM output), Tier 3 is blocked in code, and
path arguments are resolved + contained before any decision.
"""

__all__ = ["engine", "paths", "rules", "tiers", "unlock"]
