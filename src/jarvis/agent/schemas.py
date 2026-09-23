"""Typed boundary schemas for the agent layer (docs/02_ARCHITECTURE.md).

These models describe the structured data that crosses the *edges* of the
lane-graph agent: what the human asks (:class:`UserRequest`), what the
replanner proposes (:class:`ReplanDecision`), the payload shown at a
confirmation interrupt (:class:`ConfirmationRequest`), and the final answer
(:class:`AgentResponse`).  :class:`PlanStep`, :class:`PolicyDecision` and
:class:`ToolCall` are the narrower documents used inside the nodes / prompts.

Invariant: these models never carry passwords or API keys, and any model that
is stored in the checkpointed state (currently only :class:`ReplanDecision`)
must be on the serializer allowlist in :mod:`jarvis.agent.graph`.
"""

from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from jarvis.agent.state import Decision, Plan, Step

__all__ = [
    "AgentResponse",
    "ConfirmationRequest",
    "PlanStep",
    "PolicyDecision",
    "ReplanDecision",
    "ToolCall",
    "UserRequest",
]

PlanStep: TypeAlias = Step
"""One planned action; alias kept so prompts/tests read ``PlanStep``."""

PolicyDecision: TypeAlias = Decision
"""The deterministic policy outcome; alias kept for the boundary docs."""

RequestKind: TypeAlias = Literal["tool", "conversation"]
"""The two request classes the planner may produce."""


class UserRequest(BaseModel):
    """The normalised request entering the agent.

    Built by the ``intake`` node from the raw transcript / CLI line so the
    rest of the graph can rely on a trimmed ``text`` and a validated
    ``source``.  No secrets belong here.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, description="trimmed user input")
    source: Literal["terminal", "voice", "benchmark"] = "terminal"

    @field_validator("text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("user text must not be blank")
        return value


class ToolCall(BaseModel):
    """One attempted tool call, used as LLM context when replanning."""

    model_config = ConfigDict(extra="forbid")

    tool: str
    args: dict = Field(default_factory=dict)
    status: Literal["succeeded", "failed", "unverified"] = "failed"
    output: str = ""
    error: str | None = None
    tainted: bool = False


class ReplanDecision(BaseModel):
    """Structured output of the ``replan`` LLM call.

    ``plan`` is required when ``action == "continue"`` (it replaces the
    remaining steps of the current plan); when the replanner stops it should
    give a short ``message`` so the final answer stays honest.
    """

    model_config = ConfigDict(extra="forbid")

    action: Literal["continue", "stop"] = "stop"
    message: str = ""
    plan: Plan | None = None

    @model_validator(mode="after")
    def _continue_requires_plan(self) -> ReplanDecision:
        if self.action == "continue" and self.plan is None:
            raise ValueError("action='continue' requires a replacement plan")
        return self


class ConfirmationRequest(BaseModel):
    """The exact payload surfaced at a policy confirmation interrupt.

    Matches the dict historically used by the daemon / CLI (``type``, ``tier``,
    ``summary``, ``action_hash``, ...) so existing consumers keep working; the
    model documents the contract and is validated before it reaches the user.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["confirm"] = "confirm"
    step_id: str
    tier: int
    summary: str
    needs_unlock: bool = False
    typed_confirmation: str | None = None
    resolved_paths: list[str] = Field(default_factory=list)
    action_hash: str
    untrusted: bool = False


class AgentResponse(BaseModel):
    """The final structured answer a task returns to its caller.

    ``kind`` mirrors the planner's ``Plan.kind`` (``tool`` vs ``conversation``)
    so the voice layer / callers can decide whether to speak a summary or a
    dialog answer.  ``text`` is always the human-facing string.
    """

    model_config = ConfigDict(extra="forbid")

    task_id: str
    kind: RequestKind = "tool"
    interrupted: bool = False
    text: str
