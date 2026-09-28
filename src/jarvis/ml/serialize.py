"""Deterministic action serialisation for the risk classifier (docs/08 section 3).

One planned action is turned into exactly one string, in a fixed format::

    tool=delete_path | args={"paths":["<PATH:workspace/report.docx>"]} |
    user="delete the old report" | tainted=false

Training and inference **must** see the same string, so both call
:func:`serialize_action`; nothing else formats a classifier input.

Two properties matter beyond reproducibility:

* **Placeholders.** Real paths become ``<PATH:...>`` and phone numbers become
  ``<PHONE>`` before anything is written down.  The classifier should learn the
  *shape* of an action ("a file inside my workspace"), not one machine's folder
  names, and no private path ever leaves the machine.
* **Bounded.** ``user`` text is clipped to :data:`MAX_USER_CHARS` and args are
  depth/size limited, so a huge file body or a 4 KB injection cannot dominate the
  input or the model's context.
"""

from __future__ import annotations

import json
import re
from typing import Any

__all__ = [
    "MAX_USER_CHARS",
    "placeholder_path",
    "placeholder_phone",
    "redact_text",
    "redact_value",
    "serialize_action",
    "serialize_step",
]

#: The user's own words are a strong signal but must not dominate the input.
MAX_USER_CHARS = 200

#: Absolute (``C:\...``), UNC (``\\server\share``) and home-relative (``~/x``)
#: paths.  The character class deliberately excludes the JSON/quote delimiters
#: so a path inside an args blob is captured cleanly.
_PATH_RE = re.compile(r"(?:[A-Za-z]:[\\/]|\\\\|~[\\/])[^\s\"'<>|]*")

#: Phone numbers: an optional ``+`` then 7+ characters of digits and the usual
#: separators, with word boundaries so "report 2026" and version-like numbers are
#: not mangled.  WhatsApp args are exactly this shape.
_PHONE_RE = re.compile(r"(?<![\w])\+?\d[\d\s().\-]{5,}\d(?![\w])")

_SEPARATORS = re.compile(r"[\\/]+")
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_COMPONENT = 32
_MAX_DEPTH = 6
_MAX_LIST = 20
_MAX_STRING = 120


def _slug(text: str) -> str:
    """Reduce one path component to a safe, short, lowercase hint."""
    return _UNSAFE_CHARS.sub("", text)[:_MAX_COMPONENT].lower()


def placeholder_path(raw: str) -> str:
    """Return ``<PATH:a/b>`` keeping at most the last two components.

    ``C:\\Users\\me\\Documents\\JarvisWorkspace\\report.docx`` becomes
    ``<PATH:workspace/report.docx>``: enough to tell a workspace file from a
    system file, not enough to identify the machine or the user.
    """
    parts = [part for part in _SEPARATORS.split(raw or "") if part]
    if not parts:
        return "<PATH>"
    tail = parts[-2:] if len(parts) >= 2 else parts
    label = "/".join(slug for slug in (_slug(part) for part in tail) if slug)
    return f"<PATH:{label}>" if label else "<PATH>"


def placeholder_phone(raw: str) -> str:
    """Return ``<PHONE>`` for anything that looks like a phone number."""
    del raw
    return "<PHONE>"


def redact_text(text: str) -> str:
    """Replace paths and phone numbers inside a free-text field."""
    out = _PATH_RE.sub(lambda m: placeholder_path(m.group(0)), text or "")
    return _PHONE_RE.sub(lambda _m: placeholder_phone(_m.group(0)), out)


def redact_value(value: Any, depth: int = 0) -> Any:
    """Recursively redact strings inside a JSON-compatible args value.

    Depth, list length, string length and total arg count are all bounded so a
    hostile or enormous argument object cannot be used to blow up the classifier
    input (or the memory of the process building it).
    """
    if depth >= _MAX_DEPTH:
        return "<...>"
    if isinstance(value, str):
        return redact_text(value)[:_MAX_STRING]
    if isinstance(value, bool) or value is None or isinstance(value, int | float):
        return value
    if isinstance(value, dict):
        return {
            str(key)[:40]: redact_value(item, depth + 1)
            for key, item in list(value.items())[:_MAX_LIST]
        }
    if isinstance(value, list | tuple):
        return [redact_value(item, depth + 1) for item in list(value)[:_MAX_LIST]]
    return redact_text(str(value))[:_MAX_STRING]


def _json(value: Any) -> str:
    """Compact, key-sorted, non-ASCII-preserving JSON (stable across runs)."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def serialize_action(
    tool: str,
    args: dict[str, Any] | None = None,
    user_input: str = "",
    tainted: bool = False,
) -> str:
    """Serialise one planned action into the docs/08 section 3 format."""
    safe_args = redact_value(args or {})
    user = redact_text(user_input or "").replace("\n", " ").strip()[:MAX_USER_CHARS]
    return (
        f"tool={tool} | args={_json(safe_args)} | user={_json(user)}"
        f" | tainted={'true' if tainted else 'false'}"
    )


def serialize_step(step: Any, user_input: str = "", tainted: bool | None = None) -> str:
    """Serialise a :class:`~jarvis.agent.state.Step`.

    ``tainted`` defaults to the step's own ``depends_on_untrusted`` flag, so a
    caller that only has the step still produces the documented string.  Typed
    as ``Any`` to keep this module importable without the agent package (the
    training notebook only needs the serializer).
    """
    flag = bool(getattr(step, "depends_on_untrusted", False)) if tainted is None else tainted
    return serialize_action(
        str(getattr(step, "tool", "")),
        dict(getattr(step, "args", {}) or {}),
        user_input,
        flag,
    )
