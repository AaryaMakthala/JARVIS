"""Spoken text for confirmations, announcements and readbacks (Stage 3, pure).

Every string JARVIS *says* at a confirmation point is assembled here so that:

* **B9 - one redaction choke point.**  :func:`speakable` is the only thing that
  hands text to TTS, so registered secrets, key-shaped strings, sensitive
  fields and phone numbers are masked in exactly one place instead of being
  sprinkled over the call sites.  It never truncates: a readback that cannot be
  stated in full is refused upstream (:func:`jarvis.voice.readback.tier2_readback`),
  not shortened here.
* **System-written wording.**  Confirmation prompts, refusals and the
  "I heard" line are constants.  Only the *plan summary* and the *readback*
  carry content, and both are deterministic functions of the plan/decision -
  never LLM prose.  The final-answer generator is untouched by this module.
* **Privacy of the log.**  :func:`echo_line` speaks the user's transcript and
  logs only a length, so the INFO log never contains what was said in the room
  (docs/03 §7.10, AGENTS.md "redact at INFO").

This module is deterministic and side-effect free apart from one INFO log line
in :func:`echo_line`.
"""

from __future__ import annotations

import logging
from typing import Any

from jarvis.logging_setup import redact
from jarvis.voice.readback import PLAN_APPROVAL_PROMPT, READBACK_UNSAFE_TEXT
from jarvis.voice.tier2_voice import readback_for as tier2_readback_for

__all__ = [
    "CONFIRMATION_PROMPT",
    "ECHO_HEARD_PREFIX",
    "confirmation_prompt_text",
    "echo_enabled",
    "echo_line",
    "plan_announcement",
    "speakable",
    "spoken_before_confirm",
]

logger = logging.getLogger(__name__)

#: Spoken prompt for a **single Tier 1** confirmation.  It names only what the
#: user has to say, never the configured wake token: the answer is given inside
#: a wake-free window, so asking for the wake word there would ask for
#: something the window does not listen for.  Canonical here; ``voice.loop``
#: re-exports it so existing importers keep working.
CONFIRMATION_PROMPT = "Say yes to continue, or no to cancel."

#: Spoken before the prompt so a mis-heard command is caught before it is
#: answered.  A constant, like every other Stage-3 spoken string.
ECHO_HEARD_PREFIX = "I heard: "


def speakable(text: Any) -> str:
    """The single TTS choke point: coerce to text and redact it.

    Never truncates and never adds words - only masking.  A non-string value
    (a ``None`` summary, an int tier) becomes a short safe string rather than
    the word "None" being spoken or an exception escaping into the audio path.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return redact(text).strip()


def echo_enabled(settings: Any) -> bool:
    """Whether ``voice.echo_heard`` is on.  Only an explicit True enables it."""
    voice = getattr(settings, "voice", None)
    source = voice if voice is not None else settings
    return getattr(source, "echo_heard", None) is True


def echo_line(transcript: Any) -> str:
    """"I heard: <transcript>", redacted.  Logs a length, never the words."""
    heard = speakable(transcript)
    if not heard:
        # Nothing usable heard: speak nothing rather than a bare "I heard:".
        logger.info("confirmation echo skipped: empty transcript")
        return ""
    # Deliberately logs the character count only.  The transcript itself is the
    # user's speech: it must never reach the INFO log (B9 / docs/03 §7.10).
    logger.info("confirmation echo spoken (chars=%d)", len(heard))
    return f"{ECHO_HEARD_PREFIX}{heard}"


def plan_announcement(payload: Any) -> str:
    """The deterministic plan-level announcement for a ``plan_approval`` payload.

    The summary is produced by :func:`jarvis.agent.plan_summary.summarise` inside
    the payload, so the spoken text is identical in NORMAL and AUTO - the
    wake-free window is the only mechanism involved (docs/03 §7.11, D14).
    """
    data = dict(payload or {})
    summary = speakable(data.get("summary"))
    return f"{summary}. {PLAN_APPROVAL_PROMPT}" if summary else PLAN_APPROVAL_PROMPT


def confirmation_prompt_text(payload: Any, *, settings: Any = None) -> str:
    """The full spoken prompt for one confirmation payload.

    * ``plan_approval`` -> deterministic summary + "Say proceed to approve, or
      cancel."  (Stage 3a; identical in NORMAL and AUTO.)
    * a single Tier 2 step -> the **exact readback** first, then the same
      proceed/cancel prompt (Stage 3e).  When the readback is impossible the
      answer is :data:`~jarvis.voice.readback.READBACK_UNSAFE_TEXT` and nothing
      else: the action is not offered to voice at all.
    * anything else (single Tier 1 step) -> the long-standing
      :data:`CONFIRMATION_PROMPT`, unchanged.

    ``settings`` is accepted for symmetry with :func:`spoken_before_confirm`;
    the prompt wording itself does not depend on a flag (a Tier 2 payload can
    only reach here once :func:`jarvis.voice.tier2_voice.allowed` allowed it).
    """
    del settings  # wording is flag-independent by design
    data = dict(payload or {})
    tier = _tier(data.get("tier"))
    if data.get("type") == "plan_approval":
        return plan_announcement(data)
    if tier == 2:
        readback = tier2_readback_for(
            data.get("tool"), data, args=data.get("readback_args")
        )
        if not readback.ok:
            return READBACK_UNSAFE_TEXT
        return f"{readback.text}. {PLAN_APPROVAL_PROMPT}"
    if tier >= 3:
        # A Tier 3 payload is refused by policy before it can be spoken; if one
        # ever reaches here, say the refusal and nothing about the action.
        return READBACK_UNSAFE_TEXT
    summary = speakable(data.get("summary"))
    prompt = CONFIRMATION_PROMPT
    return f"{summary}. {prompt}" if summary else prompt


def spoken_before_confirm(
    payload: Any,
    *,
    settings: Any = None,
    transcript: Any = "",
) -> str:
    """Everything to speak before a confirmation window opens, in order.

    ``"I heard: ..."`` (when ``voice.echo_heard`` is on) then the prompt, joined
    so the user hears one continuous utterance.  Both halves are redacted by
    :func:`speakable`.  With the flag off this is byte-for-byte the Stage-2
    behaviour.
    """
    parts: list[str] = []
    if echo_enabled(settings):
        echo = echo_line(transcript)
        if echo:
            parts.append(echo)
    parts.append(confirmation_prompt_text(payload, settings=settings))
    return speakable(". ".join(p for p in parts if p))


def _tier(value: Any) -> int:
    """Coerce a payload tier to int; unusable values are 0 (never Tier 2)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0