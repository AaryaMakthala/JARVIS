"""Small, deterministic canonical topics for user preferences and facts.

Identity is based on an explicit closed alias table, never on loose keyword
overlap.  A phrase is only an alias when all of its normalized words appear
contiguously, so "response language" cannot be mistaken for "programming
language" merely because both contain "language".
"""

from __future__ import annotations

import re

__all__ = [
    "CANONICAL_TOPICS",
    "canonical_value",
    "fact_key_for_topic",
    "topic_from_text",
    "topic_from_value",
]

_WORD_RE = re.compile(r"[a-z0-9]+")

_TOPIC_PHRASES: dict[str, tuple[str, ...]] = {
    "programming_language": (
        "preferred programming language",
        "programming language i prefer",
        "favorite programming language",
        "favourite programming language",
        "preferred coding language",
        "coding language i prefer",
        "programming language",
        "coding language",
    ),
    "code_editor": (
        "preferred code editor",
        "favorite code editor",
        "favourite code editor",
        "code editor i use",
        "preferred editor",
        "favorite editor",
        "favourite editor",
        "development environment",
        "code editor",
        "editor i use",
    ),
    "operating_system": (
        "current operating system",
        "operating system i use",
        "operating system",
        "os i use",
    ),
    "response_language": (
        "language i want responses in",
        "preferred response language",
        "language for responses",
        "language i want replies in",
        "preferred reply language",
        "response language",
        "reply language",
        "responses in",
    ),
}

_TOPIC_LABELS: dict[str, str] = {
    "programming_language": "my preferred programming language is {value}",
    "code_editor": "my preferred code editor is {value}",
    "operating_system": "my current operating system is {value}",
    "response_language": "my preferred response language is {value}",
}

_VALUE_PHRASES: dict[str, tuple[str, ...]] = {
    "programming_language": (
        "python",
        "java",
        "javascript",
        "typescript",
        "rust",
        "golang",
        "ruby",
        "php",
        "kotlin",
        "swift",
        "scala",
        "perl",
        "matlab",
        "julia",
        "sql",
        "bash",
        "powershell",
    ),
    "code_editor": (
        "vs code",
        "visual studio code",
        "pycharm",
        "neovim",
        "sublime text",
        "intellij",
        "vim",
        "emacs",
        "atom",
    ),
    "operating_system": (
        "windows",
        "linux",
        "macos",
        "mac os",
        "ubuntu",
        "debian",
        "fedora",
        "android",
    ),
}

CANONICAL_TOPICS: tuple[str, ...] = tuple(_TOPIC_PHRASES)

_ALL_PHRASES: tuple[tuple[str, str], ...] = tuple(
    sorted(
        ((topic, phrase) for topic, phrases in _TOPIC_PHRASES.items() for phrase in phrases),
        key=lambda pair: (len(_WORD_RE.findall(pair[1])), len(pair[1])),
        reverse=True,
    )
)

_VALUE_LOOKUP: tuple[tuple[str, str], ...] = tuple(
    sorted(
        ((topic, phrase) for topic, phrases in _VALUE_PHRASES.items() for phrase in phrases),
        key=lambda pair: (len(_WORD_RE.findall(pair[1])), len(pair[1])),
        reverse=True,
    )
)


def _normalise(text: str) -> str:
    return " ".join(_WORD_RE.findall((text or "").casefold()))


def _matches(normalised: str, phrase: str) -> bool:
    words = _normalise(phrase)
    return bool(words) and f" {words} " in f" {normalised} "


def topic_from_text(text: str) -> str | None:
    """Return the canonical topic when ``text`` states an exact alias phrase."""
    normalised = _normalise(text)
    for topic, phrase in _ALL_PHRASES:
        if _matches(normalised, phrase):
            return topic
    return None


def topic_from_value(value: str) -> str | None:
    """Infer a topic only from an unambiguous, well-known value."""
    normalised = _normalise(value)
    found: set[str] = set()
    for topic, phrase in _VALUE_LOOKUP:
        if _matches(normalised, phrase):
            found.add(topic)
    return found.pop() if len(found) == 1 else None


def canonical_value(topic: str, value: str) -> str:
    """Render a replacement as a self-describing durable fact."""
    if topic not in CANONICAL_TOPICS:
        raise ValueError("unknown canonical topic")
    return _TOPIC_LABELS[topic].format(value=value)


def fact_key_for_topic(topic: str) -> str:
    """Return the stable preference key for a canonical topic.

    ``fact.preference.<topic>`` keeps the existing ``fact_details()`` category
    and needs no schema change; the canonical topic itself becomes the identity
    instead of a hash of one particular phrasing.
    """
    if topic not in CANONICAL_TOPICS:
        raise ValueError("unknown canonical topic")
    return f"fact.preference.{topic}"
