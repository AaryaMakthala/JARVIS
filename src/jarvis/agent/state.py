"""LangGraph state schema.

Defines the shared :class:`AgentState` TypedDict and the Pydantic models used
inside it (docs/02_ARCHITECTURE.md section 4).  The policy engine reads
:class:`Step` and produces :class:`Decision` values.

Invariant: **never** add passwords, API keys, or raw audio to this state - it
is checkpointed to SQLite in plaintext.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

Source = Literal["terminal", "voice", "benchmark"]


class Step(BaseModel):
    """One planned action produced by the planner LLM."""

    id: str  # "s1", "s2", ...
    tool: str  # must exist in the registry
    args: dict[str, Any]  # validated against the tool's args model in `validate`
    rationale: str  # short, user-visible
    expect: str = ""  # human-readable success condition (used by verify + replan)
    depends_on_untrusted: bool = False  # set when args derive from web/file text


class Plan(BaseModel):
    """The planner's proposed sequence of steps.

    ``kind`` records *what* the request is (docs/02_ARCHITECTURE.md request
    classification): ``"tool"`` requests run desktop tools, ``"conversation"``
    requests only need an answer.  ``dialog_answer`` is optional and ignored by
    the dedicated ``converse`` node; the planner prompt keeps the catalogue
    focused on tools.  Defaults keep old checkpoints and test fixtures valid.
    """

    kind: Literal["tool", "conversation"] = "tool"
    goal: str
    steps: list[Step] = Field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: str | None = None
    dialog_answer: str | None = None


class Decision(BaseModel):
    """Policy outcome for one step, computed deterministically."""

    step_id: str
    tier: int  # 0..3
    allowed: bool  # False when Tier 3 or a hard rule hit
    needs_confirm: bool
    needs_unlock: bool
    needs_typed_confirmation: str | None = None  # e.g. folder name to type
    resolved_paths: list[str] = Field(default_factory=list)  # real paths the gate saw
    reasons: list[str] = Field(default_factory=list)
    summary: str  # exact text shown to the user
    action_hash: str  # sha256(tool + canonical(args)); binds approval to action
    warn_untrusted: bool = False  # True when args overlap with tainted text (deterministic)


class StepResult(BaseModel):
    """Recorded outcome of one tool execution."""

    step_id: str
    ok: bool
    output: str = ""  # short text for the LLM/user (truncated)
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    verified: bool | None = None
    tainted: bool = False  # output contains untrusted external text
    duration_ms: int = 0


class AgentState(TypedDict, total=False):
    """The shared graph state (all keys optional).

    Operational keys added in Phase 1: ``validate_attempts`` (counts plan
    repair retries), ``halted_reason`` (immediate stop signal for respond),
    ``retry_now`` (verify -> retry the same step) and - Phase 7 - ``replan_now``
    (verify -> try a fresh plan), ``pending_replan`` (validate ran for a
    replanned plan), ``request_kind`` (the classified request, set by the
    understand_request node) and ``replan_decision`` (the last ReplanDecision,
    stored so the final answer can stay honest).
    """

    task_id: str
    source: Source
    user_input: str
    memory_context: list[dict]  # retrieved skills/failures/preferences (examples only)
    plan: Plan | None
    step_index: int
    decisions: dict[str, Decision]  # by step_id
    approved_hashes: list[str]  # action hashes the user approved (this task only)
    results: list[StepResult]
    retry_count: int  # for current step
    replan_count: int
    api_calls: int
    tokens: int
    cancelled: bool
    final_answer: str | None
    error: str | None
    # operational (Phase 1)
    validate_attempts: int
    gated_resolved_paths: dict[str, list[str]]  # step_id -> pre-interrupt resolved paths (TOCTOU)
    halted_reason: str | None
    retry_now: bool | None
    # operational (Phase 7: classification, replanning)
    request_kind: str
    pending_replan: bool
    replan_now: bool | None
    replan_decision: Any | None  # jarvis.agent.schemas.ReplanDecision (allowlisted)


class Profile(BaseModel):
    """User profile stored about the human (email, name, reminder phrases)."""

    name: str = ""
    email: str = ""
    reminders: list[str] = Field(default_factory=list)
