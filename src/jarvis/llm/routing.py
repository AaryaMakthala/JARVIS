"""Deterministic task-class routing for the LLM model roles.

The brain uses one structured call for every request, but it does not need the
strongest (planner) model for a greeting.  This module classifies the request
with a closed, deterministic vocabulary only - never an LLM - and maps the
class to one of the existing model roles.  The pricing/eligibility decision
still belongs to :mod:`jarvis.llm.provider` and :mod:`jarvis.llm.models`; this
module only chooses a role, and the factory resolves that role per provider.
"""

from __future__ import annotations

import re
from enum import Enum

__all__ = ["TaskClass", "classify_request", "model_role_for"]


class TaskClass(str, Enum):
    """Request classes the router distinguishes."""

    CONVERSATION = "conversation"
    ACTION = "action"
    REASONING = "reasoning"
    VISION = "vision"


#: Class -> model role.  Conversation goes to the cheaper fast model; planning
#: and multi-step reasoning stay on the planner; images need a vision model.
_MODEL_ROLE: dict[TaskClass, str] = {
    TaskClass.CONVERSATION: "fast",
    TaskClass.ACTION: "planner",
    TaskClass.REASONING: "planner",
    TaskClass.VISION: "vision",
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
        "calculate",
        "compare",
        "debug",
        "design",
        "difference",
        "explain",
        "plan",
        "pros",
        "reason",
        "solve",
        "summarise",
        "summarize",
        "why",
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

_WORD_RE = re.compile(r"[a-z]+")


def _words(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def classify_request(text: str, *, has_image: bool = False) -> TaskClass:
    """Classify one user request into a closed :class:`TaskClass`.

    The rules are intentionally conservative: anything that mentions a tool
    action stays on the planner, long/analytical requests stay on the planner,
    and only short conversational turns use the fast model.  Unknown requests
    fall back to ``reasoning`` (planner), never to a cheaper role.
    """
    if has_image:
        return TaskClass.VISION
    stripped = " ".join((text or "").split())
    if not stripped:
        return TaskClass.CONVERSATION
    lowered = stripped.lower()
    words = _words(stripped)
    if words & _ACTION_WORDS:
        return TaskClass.ACTION
    if len(stripped) > 200 or words & _REASONING_WORDS:
        return TaskClass.REASONING
    if any(lowered.startswith(prefix) for prefix in _CONVERSATION_STARTS):
        return TaskClass.CONVERSATION
    return TaskClass.REASONING


def model_role_for(task_class: TaskClass) -> str:
    """Return the model role that serves ``task_class``."""
    return _MODEL_ROLE[TaskClass(task_class)]
