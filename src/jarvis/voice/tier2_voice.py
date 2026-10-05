"""The opt-in "Tier 2 may be approved by voice" gate (Stage 3, pure).

**This is a documented relaxation.**  Until now voice was Tier 1 only
(docs/03 §7.8): every Tier 2 confirmation needed the terminal password.  This
module is the *only* place that relaxation is expressed, and it is inert until
the owner sets ``voice.allow_tier2_by_voice = true`` in ``config.toml``.  While
the flag is false (the default) :func:`allowed` returns ``ok=False`` for every
Tier 2 action, so the behaviour is bit-for-bit what it was before Stage 3.

What the flag does **not** do:
* it never lowers a tier - the tier still comes from the policy engine, and a
  step whose ``Decision`` says Tier 3 is refused here unconditionally;
* it never unlocks the session - it never touches ``ctx.unlock`` and never
  calls :meth:`~jarvis.policy.unlock.UnlockManager.verify`;
* it never covers Tier 3, in any mode (NORMAL, AUTO, batch, voice, flag on);
* it never widens *which* Tier 2 actions qualify: only
  :data:`VOICE_TIER2_TOOLS`, and only when the exact readback succeeds;
* it never changes AUTO - :mod:`jarvis.agent.plan_summary` and the batch
  machinery are Tier 1 only, and no code path here reads the mode.

Fail closed at every step: a missing/!bool flag, an unknown tool, a tier that
is not exactly 2, a blocked decision, a typed folder-name requirement, or a
readback that cannot state the action in full are all refusals.

The marker :data:`KEY_VOICE_TIER2` ("voice_tier2") is set by the *daemon* on the
resume answer.  It is never trusted on its own: the gate that consumes it
re-verifies every condition here before accepting a Tier 2 step while the
session is locked.
"""

from __future__ import annotations

from typing import Any

from jarvis.voice.readback import ReadbackResult, tier2_readback

__all__ = [
    "KEY_VOICE_TIER2",
    "TIER2_PROCEED_WORDS",
    "VOICE_TIER2_TOOLS",
    "allowed",
    "flag_on",
    "readback_for",
    "voice_marker_present",
]

#: The only two tools whose Tier 2 step a spoken "proceed" may approve.
VOICE_TIER2_TOOLS = frozenset({"delete_path", "whatsapp_send"})

#: Key the daemon sets on the resume answer to say "this approval came from the
#: voice path, which already re-verified the relaxation".  Advisory only: the
#: gate re-derives the verdict itself (see :func:`allowed`).
KEY_VOICE_TIER2 = "voice_tier2"

#: Tier 2 is approved by exactly one word.  "yes"/"confirm"/"ok"/"go" are
#: deliberately NOT in this set: a destructive action must be confirmed with an
#: unambiguous word, and "go" after a readback is far too easy to say by
#: accident while the room is noisy.
TIER2_PROCEED_WORDS = frozenset({"proceed"})


def flag_on(settings: Any) -> bool:
    """Whether the owner enabled the Tier 2 voice relaxation.

    ``settings`` may be the whole :class:`~jarvis.config.Settings` (has
    ``.voice``), a bare :class:`~jarvis.config.VoiceSettings`, or ``None``
    (what a directly-constructed :class:`~jarvis.voice.loop.VoiceLoop` gets in
    tests).  Anything that is not an explicit ``True`` is off - fail closed.
    """
    voice = getattr(settings, "voice", None)
    source = voice if voice is not None else settings
    return getattr(source, "allow_tier2_by_voice", None) is True


def allowed(
    tool_name: Any,
    decision: Any,
    settings: Any,
    *,
    args: Any = None,
) -> tuple[bool, str]:
    """May this Tier 2 step be approved by voice?  Returns ``(ok, reason)``.

    ``decision`` is the policy :class:`~jarvis.agent.state.Decision` (preferred)
    or the interrupt payload mapping - both expose the same field names.
    ``args`` optionally supplies the validated tool args, which the readback
    needs to state a WhatsApp message in full; when absent the readback falls
    back to whatever ``decision`` exposes.  ``reason`` is always populated (it
    is the refusal explanation), so a caller can log or speak it verbatim.
    """
    result, reason = _verify(tool_name, decision, args, flag_ok=flag_on(settings))
    return result.ok, reason


def readback_for(tool_name: Any, decision: Any, *, args: Any = None) -> ReadbackResult:
    """The exact readback to speak before a Tier 2 voice window opens.

    Applies the same tool/tier/typed-name gating as :func:`allowed` but not the
    owner flag, so the loop can build the text it must speak.  ``ok`` is still
    the only thing a caller may act on: a non-ok result carries no text.
    """
    result, _ = _verify(tool_name, decision, args, flag_ok=True)
    return result


def voice_marker_present(answer: Any) -> bool:
    """Whether *answer* carries the daemon's advisory voice marker.

    This says nothing about whether the answer is *acceptable* - it exists only
    so the gate can tell "the daemon already ran the voice exemption" from "the
    client claimed a voice origin".  The gate still re-verifies with
    :func:`allowed` before it acts.
    """
    return isinstance(answer, dict) and answer.get(KEY_VOICE_TIER2) is True


# ── internals ─────────────────────────────────────────────────────────────


def _verify(
    tool_name: Any,
    decision: Any,
    args: Any,
    *,
    flag_ok: bool,
) -> tuple[ReadbackResult, str]:
    """Every condition, in fail-closed order.  Returns (readback, reason)."""
    name = str(tool_name or "").strip().lower()

    # Tier 3 first and unconditionally: no flag, no mode and no tool can reach
    # it.  A tier that is not exactly 2 is refused just the same.
    tier = _int(_field(decision, "tier"))
    if tier >= 3:
        return _refuse("Tier 3 is never approved by voice")
    if tier != 2:
        return _refuse(f"only Tier 2 actions can use the Tier 2 voice path (tier={tier})")

    if not flag_ok:
        return _refuse("Tier 2 by voice is disabled (voice.allow_tier2_by_voice is off)")

    if name not in VOICE_TIER2_TOOLS:
        return _refuse(f"{name or 'that tool'!r} cannot be approved by voice at any tier")

    if _field(decision, "allowed") is not True:
        return _refuse("the policy engine blocked this step")

    # A folder delete needs a *typed folder name*: there is nothing safe for the
    # voice path to compare, so it keeps the terminal password.  The readback
    # enforces this too; both are checked so neither can drift.
    if _field(decision, "needs_typed_confirmation"):
        return _refuse("a typed folder-name confirmation needs the terminal")

    result = tier2_readback(name, args if args is not None else decision)
    if not result.ok:
        reason = result.reason or "the action cannot be read back in full"
        return ReadbackResult(False, "", reason), reason
    return result, "tier 2 voice approval is allowed"


def _refuse(reason: str) -> tuple[ReadbackResult, str]:
    """A refusal: no text, so nothing can be spoken by accident."""
    return ReadbackResult(False, "", reason), reason


def _field(source: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from a mapping, a pydantic model or any object."""
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def _int(value: Any) -> int:
    """Coerce to int; anything unusable is 0 (i.e. not Tier 2, i.e. refused)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
