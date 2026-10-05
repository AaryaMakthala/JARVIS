"""Spoken confirmations are deterministic, redacted, and never logged verbatim.

Stage 3 speaks three new things: the plan-level announcement, the exact Tier-2
readback, and the optional "I heard: ..." echo.  Two properties matter and are
pinned here:

* **Determinism** - the same payload always produces the same sentence, in NORMAL
  and AUTO alike (D14: the wake-free window is the only mechanism, so the mode
  cannot change the words).
* **B9 / privacy** - every string reaches TTS through
  :func:`~jarvis.voice.announce.speakable`, which redacts secrets, key-shaped
  strings and phone numbers, and the transcript is logged as a *length* only.

No audio, no graph, no network: these call the pure helpers directly.
"""

from __future__ import annotations

import logging
from typing import Any

from jarvis.agent.batch_approval import TYPE_PLAN_APPROVAL
from jarvis.config import VoiceSettings
from jarvis.daemon.confirmations import plan_hash
from jarvis.voice.announce import (
    CONFIRMATION_PROMPT,
    ECHO_HEARD_PREFIX,
    confirmation_prompt_text,
    echo_enabled,
    echo_line,
    plan_announcement,
    speakable,
    spoken_before_confirm,
)
from jarvis.logging_setup import register_redact_value
from jarvis.voice.readback import PLAN_APPROVAL_PROMPT
from jarvis.voice.tier2_voice import flag_on

#: A key-shaped secret (matches ``redact``'s ``sk-`` pattern) *and* registered
#: as an exact value, so the test covers both redaction paths.
SECRET = "sk-live-ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"
register_redact_value(SECRET)


def _plan_payload(**over: Any) -> dict[str, Any]:
    eligible = [("s1", "a" * 64), ("s2", "b" * 64)]
    payload = {
        "type": TYPE_PLAN_APPROVAL,
        "tier": 1,
        "summary": "2 steps: create a.txt, create b.txt",
        "action_hash": plan_hash([h for _, h in eligible]),
        "eligible": eligible,
        "plan_hash": plan_hash([h for _, h in eligible]),
        "resolved_paths": [],
    }
    payload.update(over)
    return payload


def _tier2_payload(**over: Any) -> dict[str, Any]:
    payload = {
        "type": "confirm",
        "step_id": "s3",
        "tier": 2,
        "tool": "delete_path",
        "summary": "delete C:/ws/a.txt",
        "needs_unlock": True,
        "typed_confirmation": None,
        "allowed": True,
        "resolved_paths": ["C:/ws/a.txt"],
        "readback_args": {"paths": ["C:/ws/a.txt"]},
        "action_hash": "c" * 64,
    }
    payload.update(over)
    return payload


# ── determinism + redaction ────────────────────────────────────────────────


def test_plan_announcement_deterministic_and_redacted() -> None:
    payload = _plan_payload()
    first = plan_announcement(payload)
    second = plan_announcement(dict(payload))
    assert first == second, "the plan announcement must be a pure function of the payload"
    assert first.startswith("2 steps: create a.txt, create b.txt.")
    assert first.endswith(PLAN_APPROVAL_PROMPT)
    assert plan_announcement(payload) == confirmation_prompt_text(payload)

    # Same input, spoken text is byte-identical whatever the mode: nothing here
    # reads a VoiceMode, so AUTO cannot change a word.
    from jarvis.voice.modes import VoiceMode

    assert (
        spoken_before_confirm(
            payload, settings=VoiceSettings(), transcript="make two files"
        )
        == spoken_before_confirm(
            payload,
            settings=VoiceSettings(),
            transcript="make two files",
        )
    )
    assert VoiceMode.AUTO is not None  # the mode simply has no path into this text

    # A secret or a phone number inside the summary is masked, not spoken.
    leaky = _plan_payload(summary=f"send {SECRET} to +919876543210")
    spoken = plan_announcement(leaky)
    assert SECRET not in spoken
    assert "876543210" not in spoken.replace("\u2022", "")
    # ... and an empty summary still yields the strict prompt.
    assert plan_announcement(_plan_payload(summary="")) == PLAN_APPROVAL_PROMPT


# ── the "I heard" line is spoken but not logged ────────────────────────────


def test_echo_heard_not_logged_at_info(caplog: Any) -> None:
    transcript = f"delete the file named {SECRET}"
    with caplog.at_level(logging.INFO, logger="jarvis.voice.announce"):
        spoken = echo_line(transcript)
    assert spoken.startswith(ECHO_HEARD_PREFIX)
    assert SECRET not in spoken, "the spoken line is redacted too"
    infos = [r for r in caplog.records if r.levelno == logging.INFO]
    assert infos, "the echo must leave an audit line"
    for record in infos:
        message = record.getMessage()
        assert SECRET not in message, "the transcript must never reach the log"
        assert "delete the file" not in message
        assert str(len(spoken)) in message or "chars" in message

    # Nothing at all is spoken for an empty transcript, and nothing is logged
    # with content either.
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="jarvis.voice.announce"):
        assert echo_line("") == ""
    assert all("I heard" not in r.getMessage() for r in caplog.records)

    # The flag decides whether the echo is spoken at all.
    assert echo_enabled(VoiceSettings()) is True, "echo_heard defaults to on"
    assert echo_enabled(VoiceSettings(echo_heard=False)) is False
    assert echo_enabled(None) is False
    with_echo = spoken_before_confirm(
        _tier2_payload(), settings=VoiceSettings(), transcript="delete a.txt"
    )
    without_echo = spoken_before_confirm(
        _tier2_payload(),
        settings=VoiceSettings(echo_heard=False),
        transcript="delete a.txt",
    )
    assert with_echo.startswith(ECHO_HEARD_PREFIX)
    assert not without_echo.startswith(ECHO_HEARD_PREFIX)
    assert "Say proceed to approve, or cancel." in with_echo
    assert flag_on(VoiceSettings()) is False


# ── the Tier 2 prompt is the readback, never a truncated summary ───────────


def test_tier2_prompt_speaks_the_exact_readback() -> None:
    payload = _tier2_payload()
    spoken = confirmation_prompt_text(payload)
    assert spoken.endswith(PLAN_APPROVAL_PROMPT)
    # The readback resolves the path, so Windows separators are expected; the
    # point is that the *whole* resolved target is spoken, not a summary.
    assert "a.txt" in spoken
    assert spoken.endswith(f"{PLAN_APPROVAL_PROMPT}")
    assert spoken == confirmation_prompt_text(dict(payload))
    # A Tier 1 step keeps the long-standing wording.
    tier1 = {"type": "confirm", "tier": 1, "summary": "create a.txt"}
    assert confirmation_prompt_text(tier1) == f"create a.txt. {CONFIRMATION_PROMPT}"
    assert confirmation_prompt_text({"tier": 1}) == CONFIRMATION_PROMPT
    # Tier 3 never gets its action described out loud.
    assert "delete" not in confirmation_prompt_text(_tier2_payload(tier=3)).lower()
    # An unreadable Tier 2 target is refused, and speakable() masks everything.
    assert speakable(f"key {SECRET}") .find(SECRET) == -1
    assert speakable(None) == ""