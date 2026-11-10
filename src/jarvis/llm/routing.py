"""Deterministic task-class routing for the LLM model roles.

The brain uses one structured call for every request, but it does not need the
strongest (planner) model for a greeting.  This module classifies the request
with a closed, deterministic vocabulary only - never an LLM - and maps the
class to one of the existing model roles.  The pricing/eligibility decision
still belongs to :mod:`jarvis.llm.provider` and :mod:`jarvis.llm.models`; this
module only chooses a role, and the factory resolves that role per provider.

The class and reason vocabularies live in :mod:`jarvis.taskstore.vocabulary`
so the persisted ``route`` event can allow-list the exact same closed values.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from jarvis.taskstore.vocabulary import TaskRequestClass, TaskRouteReason

__all__ = [
    "RouteDecision",
    "TaskClass",
    "classify_request",
    "model_role_for",
    "route_request",
]

#: The routing classes are the task-store vocabulary, kept under the historic
#: ``TaskClass`` name so callers do not need to know where it is defined.
TaskClass = TaskRequestClass


#: Class -> model role.  Conversation goes to the cheaper fast model.  Every
#: other class stays on the planner: action/reasoning/coding/summarisation
#: need planning, tool-argument generation is planning, and local OCR is a
#: local tool path (it must never be sent to a cloud vision model).
_MODEL_ROLE: dict[TaskClass, str] = {
    TaskClass.CONVERSATION: "fast",
    TaskClass.ACTION: "planner",
    TaskClass.REASONING: "planner",
    TaskClass.CODING: "planner",
    TaskClass.SUMMARIZATION: "planner",
    TaskClass.TOOL_ARGS: "planner",
    TaskClass.LOCAL_OCR: "planner",
}

_ACTION_WORDS = frozenset(
    {
        "append",
        "browse",
        "click",
        "close",
        "copy",
        "create",
        "delete",
        "download",
        "email",
        "find",
        "install",
        "launch",
        "list",
        "lock",
        "message",
        "move",
        "mute",
        "open",
        "pause",
        "play",
        "press",
        "quit",
        "read",
        "remove",
        "rename",
        "restart",
        "screenshot",
        "search",
        "send",
        "set",
        "shutdown",
        "show",
        "start",
        "take",
        "turn",
        "type",
        "unlock",
        "upload",
        "volume",
        "write",
    }
)

_REASONING_WORDS = frozenset(
    {
        "analyse",
        "analyze",
        "compare",
        "design",
        "difference",
        "explain",
        "plan",
        "pros",
        "reason",
        "solve",
        "why",
    }
)

_CODING_WORDS = frozenset(
    {
        "api",
        "bug",
        "class",
        "code",
        "coding",
        "compiler",
        "debug",
        "exception",
        "function",
        "java",
        "javascript",
        "program",
        "python",
        "refactor",
        "regex",
        "script",
        "sql",
        "traceback",
        "typescript",
        "unittest",
    }
)

_SUMMARIZATION_WORDS = frozenset(
    {
        "abstract",
        "condense",
        "key_points",
        "recap",
        "shorten",
        "summarise",
        "summarize",
        "summary",
        "tldr",
    }
)

_TOOL_ARGS_WORDS = frozenset(
    {
        "arguments",
        "args",
        "parameter",
        "parameters",
        "payload",
        "schema",
    }
)

_TOOL_ARGUMENT_HINTS = frozenset(
    {
        "call",
        "create",
        "format",
        "generate",
        "produce",
        "tool",
        "write",
    }
)

_OCR_READ_WORDS = frozenset(
    {
        "analyse",
        "analyze",
        "describe",
        "explain",
        "extract",
        "identify",
        "ocr",
        "read",
        "recognise",
        "recognize",
        "say",
        "see",
        "show",
        "tell",
        "text",
        "visible",
    }
)

_OCR_TARGET_WORDS = frozenset(
    {
        "image",
        "photo",
        "picture",
        "screen",
        "screenshot",
        "window",
    }
)

_OCR_PHRASES = frozenset(
    {
        "on my screen",
        "on the screen",
        "on screen",
        "screen says",
        "screen shows",
        "does my screen say",
        "does the screen say",
    }
)

_CONVERSATION_STARTS = frozenset(
    {
        "can you",
        "could you",
        "good afternoon",
        "good evening",
        "good morning",
        "good night",
        "hello",
        "hey",
        "hi",
        "how are you",
        "how do you",
        "tell me",
        "thank you",
        "thanks",
        "what can you do",
        "what is",
        "whats",
        "when is",
        "where is",
        "who are you",
        "who is",
        "why do you",
    }
)

_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """One deterministic route: class, model role, and the reason for both."""

    task_class: TaskClass
    model_role: str
    reason_code: TaskRouteReason


def _words(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def route_request(text: str, *, has_image: bool = False) -> RouteDecision:
    """Classify one user request and explain the deterministic decision.

    The rules are intentionally conservative: anything that mentions a tool
    action stays on the planner, long/analytical requests stay on the planner,
    and only short conversational turns use the fast model.  Unknown requests
    fall back to ``reasoning`` (planner), never to a cheaper role.
    """
    if has_image:
        return _decision(TaskClass.LOCAL_OCR, TaskRouteReason.HAS_IMAGE)
    stripped = " ".join((text or "").split())
    if not stripped:
        return _decision(TaskClass.CONVERSATION, TaskRouteReason.EMPTY)
    lowered = stripped.lower()
    words = _words(stripped)
    if _looks_like_screen_text(words, lowered):
        return _decision(TaskClass.LOCAL_OCR, TaskRouteReason.SCREEN_TEXT)
    if _looks_like_tool_args(words):
        return _decision(TaskClass.TOOL_ARGS, TaskRouteReason.TOOL_ARGUMENTS)
    if words & _SUMMARIZATION_WORDS:
        return _decision(TaskClass.SUMMARIZATION, TaskRouteReason.SUMMARIZATION)
    if words & _CODING_WORDS:
        return _decision(TaskClass.CODING, TaskRouteReason.CODING)
    if len(stripped) > 200:
        return _decision(TaskClass.REASONING, TaskRouteReason.LONG_REQUEST)
    if words & _REASONING_WORDS:
        return _decision(TaskClass.REASONING, TaskRouteReason.REASONING)
    if words & _ACTION_WORDS:
        return _decision(TaskClass.ACTION, TaskRouteReason.ACTION)
    if any(lowered.startswith(prefix) for prefix in _CONVERSATION_STARTS):
        return _decision(TaskClass.CONVERSATION, TaskRouteReason.CONVERSATION)
    return _decision(TaskClass.REASONING, TaskRouteReason.FALLBACK)


def _decision(task_class: TaskClass, reason_code: TaskRouteReason) -> RouteDecision:
    return RouteDecision(
        task_class=task_class,
        model_role=_MODEL_ROLE[task_class],
        reason_code=reason_code,
    )


def _looks_like_screen_text(words: set[str], lowered: str) -> bool:
    """Whether the request asks to read text from the visible screen/image."""
    if not words & _OCR_TARGET_WORDS:
        return False
    if words & _OCR_READ_WORDS:
        return True
    return any(phrase in lowered for phrase in _OCR_PHRASES)


def _looks_like_tool_args(words: set[str]) -> bool:
    """Whether the request asks how to call a tool (arguments/JSON schema)."""
    return bool(words & _TOOL_ARGS_WORDS) and bool(words & _TOOL_ARGUMENT_HINTS)


def classify_request(text: str, *, has_image: bool = False) -> TaskClass:
    """Classify one user request into a closed :class:`TaskClass`."""
    return route_request(text, has_image=has_image).task_class


def model_role_for(task_class: TaskClass) -> str:
    """Return the model role that serves ``task_class``."""
    return _MODEL_ROLE[TaskClass(task_class)]
