"""Ablation profiles from docs/07_TESTING_AND_BENCHMARK.md section 3.4.

Each profile is a *named* combination of the five capabilities the report makes
claims about (C1 self-policing, C2 verify/replan, C3 memory, C4 classifier).
``jarvis benchmark run --config <name>`` resolves one of these, applies it to a
copy of the real :class:`~jarvis.config.Settings`, and the runner stamps the name
into every CSV row so a result can never be filed under the wrong config.

The safety-relevant line in that table is the policy engine, and this module
treats it as a hard constraint rather than a knob: ``llm-self-police`` is the
only profile that turns the engine off, it is marked ``dry_run_only``, and
:func:`apply_profile` refuses to apply it to a real-tools run.  Every other
profile keeps the engine on (docs/07 section 3.4: "safety kept on so runs are
safe"), which is also what makes an ablated run safe to execute at all.
"""

from __future__ import annotations

from dataclasses import dataclass

from jarvis.config import Settings

__all__ = ["PROFILES", "AblationProfile", "ProfileError", "apply_profile", "get_profile", "profile_names"]


class ProfileError(ValueError):
    """An unknown profile, or a profile that cannot be used safely."""


@dataclass(frozen=True)
class AblationProfile:
    """One row of the docs/07 section 3.4 ablation table."""

    name: str
    verify: bool
    replan: bool
    memory: bool
    policy: bool
    classifier: bool
    dry_run_only: bool = False
    description: str = ""


PROFILES: dict[str, AblationProfile] = {
    p.name: p
    for p in (
        AblationProfile(
            name="baseline",
            verify=False,
            replan=False,
            memory=False,
            policy=True,
            classifier=False,
            description="Plan-once execution. Policy engine stays on so the run is safe.",
        ),
        AblationProfile(
            name="+verify",
            verify=True,
            replan=False,
            memory=False,
            policy=True,
            classifier=False,
            description="Adds deterministic verification and retry. Claim C2.",
        ),
        AblationProfile(
            name="+replan",
            verify=True,
            replan=True,
            memory=False,
            policy=True,
            classifier=False,
            description="Adds LLM replanning on failure. Claim C2.",
        ),
        AblationProfile(
            name="+memory",
            verify=True,
            replan=True,
            memory=True,
            policy=True,
            classifier=False,
            description="Adds skill memory; run 3 rounds with one memory DB. Claim C3.",
        ),
        AblationProfile(
            name="full",
            verify=True,
            replan=True,
            memory=True,
            policy=True,
            classifier=True,
            description="The final system.",
        ),
        AblationProfile(
            name="llm-self-police",
            verify=False,
            replan=False,
            memory=False,
            policy=False,
            classifier=False,
            dry_run_only=True,
            description="Baseline for C1: the LLM is asked to judge safety itself. "
            "Dry-run tools only; never run against real tools.",
        ),
    )
}


def profile_names() -> list[str]:
    """The profile names in table order."""
    return list(PROFILES)


def get_profile(name: str) -> AblationProfile:
    """Look up a profile by name, with the valid names in the error."""
    try:
        return PROFILES[name]
    except KeyError:
        raise ProfileError(f"unknown benchmark config {name!r}; choose one of: {', '.join(PROFILES)}")


def apply_profile(
    settings: Settings, profile: AblationProfile, *, dry_run: bool
) -> Settings:
    """Return a copy of ``settings`` with the profile's switches applied.

    The input is never mutated, so a profile cannot leak into a later task in the
    same process.  ``dry_run`` must be True for a ``dry_run_only`` profile: the
    whole point of ``llm-self-police`` is that it removes the deterministic
    engine, so letting it reach a real tool would be a Tier 3 hazard rather than
    an experiment.
    """
    if profile.dry_run_only and not dry_run:
        raise ProfileError(
            f"config {profile.name!r} disables the policy engine and may only run with --dry-run"
        )
    tuned = settings.model_copy(deep=True)
    tuned.agent.verify_enabled = profile.verify
    tuned.agent.replan_enabled = profile.replan
    tuned.memory.enabled = profile.memory
    tuned.risk.enabled = profile.classifier
    return tuned
