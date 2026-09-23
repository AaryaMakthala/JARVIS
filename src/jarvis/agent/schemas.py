"""Typed boundary schemas for the agent layer (docs/02_ARCHITECTURE.md).

These models describe the structured data that crosses the *edges* of the
lane-graph agent: what the human asks (:class:`UserRequest`), what the brain
proposes (:class:`BrainDecision`), what the replanner proposes
(:class:`ReplanDecision`), the payloads shown at confirmation /
clarification interrupts (:class:`ConfirmationRequest`,
:class:`ClarificationRequest`), and the final answer (:class:`AgentResponse`).
:class:`PlanStep`, :class:`PolicyDecision` and :class:`ToolCall` are the
narrower documents used inside the nodes / prompts.

Invariant: these models never carry passwords or API keys.  Models that are
stored in the checkpointed state (currently only :class:`ReplanDecision`) must
be on the serializer allowlist in :mod:`jarvis.agent.graph`.
:class:`BrainDecision` and :class:`ActionIntent` are **transient** - the brain
node projects them into the checkpointed :class:`Plan` / :class:`Step` shape
before anything touches AgentState, so they are deliberately *not* allowlisted.
"""

from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from jarvis.agent.state import Decision, Plan, Step

__all__ = [
    "ActionIntent",
    "AgentResponse",
    "BrainDecision",
    "ClarificationRequest",
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


class ActionIntent(BaseModel):
    """One action the brain proposes, before the policy engine decides.

    Mirrors :class:`~jarvis.agent.state.Step` minus the ``id`` (the brain node
    assigns ``s1...`` and projects each intent into a :class:`Step`).  This is
    the single source of tool names, arguments and rationale passed to the
    deterministic validate / policy_gate / act pipeline.

    **Transient**: produced only inside the ``brain`` node's LLM result and
    never written to AgentState or the checkpoint store (not allowlisted).
    """

    model_config = ConfigDict(extra="forbid")

    tool: str
    args: dict = Field(default_factory=dict)
    rationale: str = ""
    expect: str = ""
    depends_on_untrusted: bool = False


class BrainDecision(BaseModel):
    """The single structured output of the ``brain`` LLM call (transient).

    Replaces the Phase-7 three-call split (``plan`` + ``understand_request`` +
    ``converse``) with one call.  ``request_type`` routes the graph
    deterministically after ``brain``:

    * ``conversation`` - the answer is ``response_text``; the brain node
      projects a ``Plan(kind="conversation", dialog_answer=...)`` and sets
      ``final_answer`` so ``respond`` works unchanged;
    * ``action`` - ``actions`` are projected into a multi-step
      ``Plan(kind="tool", ...)`` and run through validate / policy_gate / act;
    * ``clarification`` - the graph must ask ``clarification_question`` via the
      ``clarify`` node before doing anything else;
    * ``unsupported`` - anything out of scope or that needs a human.

    Like :class:`ActionIntent`, this object is request-scoped and **never
    checkpointed**: not on the serializer allowlist, never stored in AgentState.
    """

    model_config = ConfigDict(extra="forbid")

    request_type: Literal["conversation", "action", "clarification", "unsupported"]
    goal: str = ""
    actions: list[ActionIntent] = Field(default_factory=list)
    response_text: str = ""
    clarification_question: str | None = None
    response_hint: str = ""
    conversation_context: str = ""

    @model_validator(mode="after")
    def _direction_requires_field(self) -> BrainDecision:
        if self.request_type == "clarification" and not (self.clarification_question or "").strip():
            raise ValueError(
                "request_type='clarification' requires a non-blank clarification_question"
            )
        if self.request_type == "action" and not self.actions:
            raise ValueError("request_type='action' requires at least one action")
        return self


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


class ClarificationRequest(BaseModel):
    """Payload surfaced at a clarification interrupt.

    Mirrors :class:`ConfirmationRequest` for the ``clarify`` node: the daemon,
    CLI and voice layers read ``question`` and reply with a free-text answer
    that resumes the graph.  The answer itself travels as a plain string on the
    interrupt resume - nothing secret ever rides in this payload.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["clarification"] = "clarification"
    question: str = Field(min_length=1, description="question to ask the human")

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("clarification question must not be blank")
        return value


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
