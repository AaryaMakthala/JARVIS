"""Exact readback for the two Tier-2 actions voice may approve (Stage 3).

Owner decision D13/B1: batch/plan approval never covers Tier 2.  Each Tier-2
step gets its **own** exact readback, and only the word "proceed" (or "cancel")
inside the window may approve it.  This module builds that readback text and
decides whether the action is even eligible.  It is deterministic and
side-effect free apart from read-only path resolution; it speaks nothing and
never touches the graph.

Fail-closed rule: :func:`tier2_readback` returns ``ok=False`` unless it can
state **exactly** what will happen, in full and unbounded-visible form.  A
Tier-2 readback is never truncated -- if the target cannot be read back inside
the limits it is refused, not abbreviated.

Eligible tools (only these):
* ``delete_path`` -- a single **file** target with no typed folder-name
  requirement, whole resolved path <= 150 chars.
* ``whatsapp_send`` -- message <= 200 chars; text names the recipient, the
  masked number, and the **full** message.
* ``whatsapp_send_file`` -- exactly one resolved file with a readable name,
  whole path <= 150 chars; text names the recipient and the file, and there
  is no caption/message argument to smuggle text into.

The prompt/refusal constants live here too so every Stage-3 spoken string is
system-written, never LLM prose.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jarvis.logging_setup import redact

__all__ = [
    "PLAN_APPROVAL_PROMPT",
    "PLAN_APPROVED_TEXT",
    "PLAN_CANCELLED_TEXT",
    "PLAN_UNHEARD_TEXT",
    "READBACK_UNSAFE_TEXT",
    "ReadbackResult",
    "tier2_readback",
]

#: Maximum length of a Tier-2 path readback; longer is refused, never cut.
_MAX_PATH_CHARS = 150

#: Maximum length of a WhatsApp message that may be read back in full.
_MAX_MESSAGE_CHARS = 200

#: Maximum length of a file name in a WhatsApp attachment readback.
_MAX_FILE_NAME_CHARS = 80

# ── system-written spoken constants (no LLM) ────────────────────────────────

#: Prompt spoken before the plan/Tier-1 approval window opens.
PLAN_APPROVAL_PROMPT = "Say proceed to approve, or cancel."

#: Spoken acknowledgement after an approval.
PLAN_APPROVED_TEXT = "Proceeding."

#: Spoken after an explicit refusal (the user said "cancel"/"no").
PLAN_CANCELLED_TEXT = "Cancelled. I did not run the plan."

#: Spoken when nothing usable was heard inside the window (silence/timeout).
PLAN_UNHEARD_TEXT = "I didn't hear an approval, so I did not run the plan."

#: Spoken when a Tier-2 action cannot be read back safely by voice.
READBACK_UNSAFE_TEXT = "I can't read that back safely, so I won't do it."


@dataclass(frozen=True)
class ReadbackResult:
    """Outcome of a Tier-2 readback attempt.

    ``ok`` is the ONLY gate: a caller must not speak or approve anything unless
    ``ok`` is True.  ``text`` is the exact readback (already redacted); it is
    empty when ``ok`` is False and ``reason`` explains the refusal.
    """

    ok: bool
    text: str = ""
    reason: str = ""


def tier2_readback(tool_name: str, args_or_decision: Any) -> ReadbackResult:
    """Build the exact readback for an eligible Tier-2 tool, or refuse.

    ``args_or_decision`` may be a policy ``Decision`` (preferred: it carries the
    engine's resolved paths and typed-name requirement) or a tool-args model /
    mapping.  Everything unrecognised fails closed.
    """
    name = str(tool_name or "").strip().lower()
    if name == "delete_path":
        return _delete_readback(args_or_decision)
    if name == "whatsapp_send":
        return _whatsapp_readback(args_or_decision)
    if name == "whatsapp_send_file":
        return _whatsapp_file_readback(args_or_decision)
    return ReadbackResult(False, "", f"{name or 'that tool'!r} cannot be approved by voice")


def _field(source: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from a mapping, a pydantic model or an arbitrary object."""
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def _delete_readback(source: Any) -> ReadbackResult:
    """Exactly one file path, fully readable; anything else is refused."""
    resolved = _field(source, "resolved_paths")
    if not resolved:
        raw = _field(source, "paths")
        resolved = _resolve_paths(raw) if raw else None
    if not resolved:
        return ReadbackResult(False, "", "no resolved path to read back")

    if _field(source, "needs_typed_confirmation"):
        return ReadbackResult(False, "", "a folder delete needs a typed folder name")

    paths_list = [str(p) for p in resolved]
    if len(paths_list) != 1:
        return ReadbackResult(
            False, "", f"exactly one file is required, but {len(paths_list)} were given"
        )

    path = paths_list[0]
    if not path:
        return ReadbackResult(False, "", "the path could not be read back")
    if len(path) > _MAX_PATH_CHARS:
        return ReadbackResult(
            False, "", f"the full path is too long to read back ({len(path)} chars)"
        )

    text = redact(path)
    if not text:
        return ReadbackResult(False, "", "the path could not be read back")
    return ReadbackResult(True, text)


def _whatsapp_readback(source: Any) -> ReadbackResult:
    """Recipient + masked number + the full message; long messages are refused."""
    message = _field(source, "message")
    if not isinstance(message, str) or not message:
        return ReadbackResult(False, "", "the message is missing or unreadable")
    if len(message) > _MAX_MESSAGE_CHARS:
        return ReadbackResult(
            False, "", f"the full message is too long to read back ({len(message)} chars)"
        )

    recipient = _field(source, "contact") or _field(source, "recipient") or "unknown contact"
    masked = (
        _field(source, "masked_number")
        or _field(source, "masked")
        or _mask_number(_field(source, "number") or _field(source, "phone"))
    )
    summary = _field(source, "summary")
    if isinstance(summary, str) and summary.strip():
        text = summary.strip()
    else:
        text = f"send WhatsApp message to {recipient!r} ({masked}): {message!r}"
    return ReadbackResult(True, redact(text))


def _whatsapp_file_readback(source: Any) -> ReadbackResult:
    """Recipient + exactly one file name; there is no caption to read back."""
    resolved = _field(source, "resolved_paths")
    if not resolved:
        raw = _field(source, "path")
        resolved = _resolve_paths(raw) if raw else None
    if not resolved:
        return ReadbackResult(False, "", "no resolved file path to read back")

    paths_list = [str(item) for item in resolved]
    if len(paths_list) != 1:
        return ReadbackResult(
            False, "", f"exactly one file is required, but {len(paths_list)} were given"
        )
    path = paths_list[0]
    if len(path) > _MAX_PATH_CHARS:
        return ReadbackResult(
            False, "", f"the full path is too long to read back ({len(path)} chars)"
        )

    name = Path(path).name
    if not name:
        return ReadbackResult(False, "", "the file name could not be read back")
    if len(name) > _MAX_FILE_NAME_CHARS:
        return ReadbackResult(
            False, "", f"the file name is too long to read back ({len(name)} chars)"
        )

    contact = _field(source, "contact")
    if not isinstance(contact, str) or not contact.strip():
        return ReadbackResult(False, "", "the recipient is missing or unreadable")

    number = _field(source, "masked_number") or _field(source, "masked")
    if not number:
        digits = _field(source, "number") or _field(source, "phone")
        number = _mask_number(digits) if digits else ""
    suffix = f" ({number})" if number else ""
    text = f"send {name!r} on WhatsApp to {contact.strip()!r}{suffix}"
    return ReadbackResult(True, redact(text))


def _resolve_paths(raw: Any) -> list[str] | None:
    """Resolve raw path strings read-only; ``None`` on any failure (fail closed)."""
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return None
    try:
        from jarvis.policy import paths
    except Exception:  # noqa: BLE001 - a missing resolver must not approve anything
        return None
    resolved: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            return None
        try:
            resolved.append(str(paths.resolve_safe(item)))
        except Exception:  # noqa: BLE001 - PathError and anything else => refuse
            return None
    return resolved


def _mask_number(number: Any) -> str:
    """Mask a phone number to its last 2 digits (same shape as the tool's)."""
    digits = "".join(ch for ch in str(number or "") if ch.isdigit())
    if not digits:
        return "(number unknown)"
    if len(digits) <= 2:
        return "+" + "\u2022" * len(digits)
    return "+" + "\u2022" * (len(digits) - 2) + digits[-2:]
