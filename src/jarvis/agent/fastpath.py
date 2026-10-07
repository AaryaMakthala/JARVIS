"""Local fast path: a closed vocabulary answered without the LLM call.

Safety contract (stage 4.8; docs/03):

* **Closed vocabulary, exact matches only.** Whole-utterance matches against
  fixed phrase sets plus ``open|launch|start <name>`` where ``<name>`` is an
  exact normalised key of the ``[apps]`` allowlist. No fuzzy matching, no
  substring matching, no arithmetic dispatch (the ``test_math_phrasings``
  structural guard must stay green).
* **Only the LLM call is skipped.** The returned :class:`Plan` has the same
  shape the brain node projects (kind/tool/steps) and still runs the unchanged
  pipeline ``validate -> policy_gate -> act -> verify``. The deterministic
  policy engine is the only authority that assigns tiers or approves anything;
  this module never reads or writes a tier.
* **Fail-closed guard first:** after stripping ONE trailing ``. ? !`` from the
  raw utterance, a raw utterance over 60 characters, or containing
  ``/ \\ . : & ; | % $ "`` or a newline, is never fast-pathed — it falls through
  to the LLM. Everything that does not hit a phrase falls
  through byte-for-byte unchanged, including the provider-not-configured halt.
"""

from __future__ import annotations

import re
import string
from typing import Any

from jarvis.agent.context import AppContext
from jarvis.agent.state import Plan, Step
from jarvis.config import Settings

#: Whole-utterance phrases (normalised) that map directly to ``get_time``.
_TIME_PHRASES = frozenset(
    {
        "what time is it",
        "what is the time",
        "whats the time",
        "tell me the time",
    }
)

#: Whole-utterance phrases (normalised) that map to ``list_windows``.
_WINDOWS_PHRASES = frozenset(
    {
        "what windows are open",
        "list windows",
        "list the windows",
    }
)

#: Leading verbs for the allowlisted-app pattern (checked AFTER the phrases).
_OPEN_VERBS = frozenset({"open", "launch", "start"})

#: Raw-utterance reject set (decision 4), scanned BEFORE normalisation.
#: Apostrophes are deliberately absent: "what's the time" must normalise.
_REJECT_CHARS = frozenset('/\\.:&;%|$"\n\r')

#: Raw utterances longer than this never fast-path.
_MAX_RAW_CHARS = 60

_WS_RE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace (whole utterance)."""
    lowered = text.strip().lower()
    dropped = "".join(ch for ch in lowered if ch not in string.punctuation)
    return _WS_RE.sub(" ", dropped).strip()


def _rejected(raw: str) -> bool:
    """Whether the raw utterance must go to the LLM (fail-closed guard)."""
    if len(raw) > _MAX_RAW_CHARS:
        return True
    return any(ch in _REJECT_CHARS for ch in raw)


def match(raw: str, settings: Settings) -> tuple[str, dict[str, Any], str] | None:
    """Return ``(tool, args, phrase)`` for a closed-vocabulary hit, else None.

    The phrase table is checked first and wins over app keys; ``<name>`` must
    equal a normalised ``[apps]`` key exactly (multi-word keys allowed).
    """
    if not raw:
        return None
    guarded = raw[:-1] if raw[-1] in ".?!" else raw
    if _rejected(guarded):
        return None
    text = normalise(guarded)
    if not text:
        return None
    if text in _TIME_PHRASES:
        return "get_time", {}, text
    if text in _WINDOWS_PHRASES:
        return "list_windows", {}, text
    tokens = text.split()
    if len(tokens) >= 2 and tokens[0] in _OPEN_VERBS:
        name = " ".join(tokens[1:])
        for key in settings.apps.model_dump():
            if normalise(str(key)) == name:
                return "open_app", {"name": str(key)}, text
    return None


def plan_for(raw: str, settings: Settings) -> Plan | None:
    """Build the one-step Plan a fast-path hit returns (brain conventions)."""
    hit = match(raw, settings)
    if hit is None:
        return None
    tool, args, phrase = hit
    return Plan(
        kind="tool",
        goal=f"fastpath: {phrase}",
        steps=[
            Step(
                id="s1",
                tool=tool,
                args=args,
                rationale=f"fastpath: {phrase}",
            )
        ],
    )


def brain_update(state: dict[str, Any], ctx: AppContext) -> dict[str, Any] | None:
    """State update mirroring brain's action path, or None to fall through.

    Mirrors ``jarvis.agent.nodes.brain`` exactly: same keys, ``api_calls`` and
    ``tokens`` unchanged (+0), clarification fields cleared, ``request_kind``
    set to ``"tool"``. The only difference is that no LLM call happens.
    """
    plan = plan_for(state.get("user_input") or "", ctx.settings)
    if plan is None:
        return None
    ctx.logger.info(
        "agent event=fastpath task_id=%s tool=%s",
        state.get("task_id"),
        plan.steps[0].tool,
    )
    return {
        "api_calls": int(state.get("api_calls") or 0),
        "tokens": int(state.get("tokens") or 0),
        "clarification_question": None,
        "clarification_answer": None,
        "plan": plan,
        "request_kind": "tool",
        "error": None,
    }
