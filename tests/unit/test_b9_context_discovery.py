"""B9 acceptance tests: compact discovery, bounded context and routing.

Every test uses a mocked model (scripted structured output) and fake/real
registries only.  No live provider, browser, WhatsApp session or GUI is used.
The tests prove *prompt composition and routing contracts*; they do not claim
that a real model chose correctly.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, Field, create_model

from jarvis.agent.capability_index import (
    build_capability_index,
    discover_candidates,
    render_capability_directory,
)
from jarvis.agent.context import make_app_context
from jarvis.agent.context_composer import (
    CONTEXT_BUDGET_EXCEEDED,
    compose_brain_context,
    compose_replan_context,
    estimate_tokens,
)
from jarvis.agent.conversation import ConversationManager
from jarvis.agent.graph import open_sqlite_checkpointer
from jarvis.agent.nodes.brain import LLM_INVALID_OUTPUT, brain
from jarvis.agent.nodes.memory_retrieve import memory_retrieve
from jarvis.agent.nodes.validate import validate
from jarvis.agent.runner import run_task
from jarvis.agent.schemas import BrainDecision
from jarvis.agent.state import Plan, Step
from jarvis.config import AgentSettings, Settings
from jarvis.llm.client import LLMContextLengthError, Usage
from jarvis.llm.models import model_spec
from jarvis.llm.telemetry import LLMAttemptRecord, LLMContextStats
from jarvis.memory import SqliteMemory, open_db
from jarvis.memory.intent import detect_memory_intent
from jarvis.taskstore import TaskEventType
from jarvis.taskstore.privacy import sanitize_detail
from jarvis.taskstore.vocabulary import TaskAttemptOutcome, TaskStoreVocabulary
from jarvis.tools.base import ToolSpec
from jarvis.tools.registry import build_default_registry
from support import (
    brain_action,
    brain_conversation,
    brain_steps,
    make_spec,
    registry_with,
)


class ScriptedLLM:
    """Minimal structured-output double that records every prompt exactly."""

    def __init__(self, responses: list[BaseModel | Exception]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []
        self.prompts: list[tuple[str, str]] = []

    def structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[Any],
        model_role: str = "planner",
        temperature: float = 0.0,
        max_retries: int | None = None,
        timeout_s: float | None = None,
    ) -> tuple[Any, Usage]:
        self.prompts.append((system, user))
        if not self._responses:
            raise AssertionError("scripted LLM exhausted")
        item = self._responses.pop(0)
        self.calls.append(model_role)
        if isinstance(item, Exception):
            raise item
        if not isinstance(item, schema):
            raise TypeError(f"expected {schema.__name__}")
        return item, Usage(calls=1, prompt_tokens=11, completion_tokens=7)


def _ctx(
    llm: Any,
    *,
    settings: Settings | None = None,
    registry: Any = None,
    memory: Any = None,
) -> Any:
    settings = settings or Settings()
    return make_app_context(
        settings,
        llm=llm,
        registry=registry or build_default_registry(settings),
        memory=memory,
    )


def _state(text: str, **extra: Any) -> dict[str, Any]:
    return {"user_input": text, "source": "terminal", "task_id": "t-b9", **extra}


# ── A. ordinary conversation ────────────────────────────────────────────────


def test_a_conversation_gets_no_tool_schema_and_exactly_one_call() -> None:
    llm = ScriptedLLM([brain_conversation("Decorators wrap a function.")])
    ctx = _ctx(llm)
    out = brain(_state("Explain Python decorators in simple language."), ctx)

    assert out["request_kind"] == "conversation"
    assert out["final_answer"] == "Decorators wrap a function."
    assert len(llm.calls) == 1
    system, user = llm.prompts[0]
    assert "args_schema" not in system
    assert "args_schema" not in user
    assert "Remembered facts" not in user


def test_a_conversation_composer_has_no_candidate_schemas() -> None:
    ctx = _ctx(ScriptedLLM([]))
    composed = compose_brain_context(_state("Explain Python decorators."), ctx)

    assert composed.schema_names == ()
    assert composed.stats.candidate_tools == 0
    assert "args_schema" not in composed.system
    assert "You are the brain" in composed.system
    assert "Explain Python decorators." in composed.user


# ── B. Notepad ──────────────────────────────────────────────────────────────


def test_b_notepad_schema_is_present_and_browser_whatsapp_absent() -> None:
    ctx = _ctx(ScriptedLLM([]))
    composed = compose_brain_context(_state("Open Notepad."), ctx)

    assert "open_app" in composed.schema_names
    assert not any(name.startswith("browser_") for name in composed.schema_names)
    assert not any(name.startswith("whatsapp_") for name in composed.schema_names)
    assert '"args_schema"' in composed.system
    assert "open_app" in composed.system


def test_b_brain_selects_open_app_and_policy_engine_still_decides() -> None:
    llm = ScriptedLLM([brain_action("open_app", {"name": "notepad"})])
    ctx = _ctx(llm)
    out = brain(_state("Open Notepad."), ctx)

    assert out["plan"].steps[0].tool == "open_app"
    validated = validate({"plan": out["plan"]}, ctx)
    step = validated["plan"].steps[0]
    decision = ctx.engine.decide(step, ctx.policy_ctx)
    # The decision is computed by the engine, not carried in the brain output.
    assert len(decision.action_hash) == 64
    assert decision.step_id == step.id


# ── C. browser ──────────────────────────────────────────────────────────────


def test_c_browser_request_gets_web_tools_and_no_unrelated_app_schemas() -> None:
    ctx = _ctx(ScriptedLLM([]))
    composed = compose_brain_context(_state("Search the web for recent AI research."), ctx)

    assert composed.schema_names
    assert all(
        name.startswith("browser_") or name in {"web_answer", "google_search", "open_url"}
        for name in composed.schema_names
    )
    assert not any(name.startswith("whatsapp_") for name in composed.schema_names)
    assert "open_app" not in composed.schema_names
    assert "Capability directory" in composed.system


def test_c_browser_action_remains_engine_authorized() -> None:
    ctx = _ctx(ScriptedLLM([]))
    step = Step(
        id="s1",
        tool="browser_search",
        args={"query": "recent AI research", "site": "Google"},
        rationale="search",
    )
    decision = ctx.engine.decide(step, ctx.policy_ctx)
    # Whatever the platform allows, the tier/hash/confirmation contract is
    # produced by the PolicyEngine and cannot be supplied by the model.
    assert decision.step_id == "s1"
    assert isinstance(decision.tier, int)
    assert len(decision.action_hash) == 64


# ── D. personal memory ──────────────────────────────────────────────────────


@pytest.fixture
def memory_store() -> SqliteMemory:
    store = SqliteMemory(open_db(":memory:"), settings=Settings())
    try:
        yield store
    finally:
        store.close()


def _remember_python(store: SqliteMemory) -> None:
    intent = detect_memory_intent("Remember that my preferred coding language is Python.")
    assert intent is not None
    store.preferences.set(intent.key, intent.content)


def test_d_relevant_preference_is_retrieved_and_unrelated_one_is_excluded(
    memory_store: SqliteMemory,
) -> None:
    _remember_python(memory_store)
    memory_store.preferences.set("favorite_editor", "vim")
    ctx = _ctx(ScriptedLLM([]), memory=memory_store)

    out = memory_retrieve(_state("What is my preferred coding language?", memory_context=[]), ctx)
    texts = [item["text"] for item in out["memory_context"]]
    assert any("Python" in text for text in texts)
    assert not any("vim" in text for text in texts)

    composed = compose_brain_context(
        _state("What is my preferred coding language?", memory_context=out["memory_context"]),
        ctx,
    )
    assert "Python" in composed.user
    assert "vim" not in composed.user
    assert "never guess or infer" in composed.user
    assert composed.schema_names == ()


def test_d_missing_preference_is_not_invented(memory_store: SqliteMemory) -> None:
    ctx = _ctx(ScriptedLLM([]), memory=memory_store)
    out = memory_retrieve(_state("What is my preferred coding language?"), ctx)
    composed = compose_brain_context(
        _state("What is my preferred coding language?", memory_context=out["memory_context"]),
        ctx,
    )
    assert "Python" not in composed.user
    assert "Remembered facts" not in composed.user


def test_d_unrelated_question_never_receives_the_python_fact(
    memory_store: SqliteMemory,
) -> None:
    _remember_python(memory_store)
    ctx = _ctx(ScriptedLLM([]), memory=memory_store)
    out = memory_retrieve(_state("Explain Python decorators in simple language."), ctx)
    composed = compose_brain_context(
        _state("Explain Python decorators.", memory_context=out["memory_context"]), ctx
    )
    assert "my preferred coding language is Python" not in composed.user


# ── E. tool-selection failure ───────────────────────────────────────────────


def test_e_unknown_tool_name_is_rejected_before_execution() -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    registry = registry_with(make_spec("fake_echo", base_tier=0, record=record))
    ctx = _ctx(ScriptedLLM([brain_action("no_such_tool", {})]), registry=registry)

    out = brain(_state("echo hi"), ctx)
    validated = validate({"plan": out["plan"]}, ctx)

    assert record == []
    assert "not available" in (validated.get("error") or "")


def test_e_registered_tool_without_schema_gets_one_bounded_expansion() -> None:
    llm = ScriptedLLM(
        [
            brain_action("open_app", {"name": "notepad"}),
            brain_action("open_app", {"name": "notepad"}),
        ]
    )
    ctx = _ctx(llm)
    initial = compose_brain_context(_state("do the thing"), ctx)
    assert "open_app" not in initial.schema_names  # the premise of the test

    out = brain(_state("do the thing"), ctx)

    assert len(llm.calls) == 2  # exactly one schema-resolution retry
    assert out["plan"].steps[0].tool == "open_app"
    assert "open_app" in llm.prompts[1][0]
    assert '"args_schema"' in llm.prompts[1][0]


def test_e_still_unresolved_request_becomes_clarification() -> None:
    request = BrainDecision(request_type="tool_schema", requested_tool="open_app")
    llm = ScriptedLLM([request, request])
    ctx = _ctx(llm)

    out = brain(_state("do the thing"), ctx)

    assert len(llm.calls) == 2
    assert out["request_kind"] == "clarification"
    assert out["plan"].needs_clarification is True
    assert not out.get("plan").steps


def test_e_unregistered_schema_request_fails_closed() -> None:
    llm = ScriptedLLM([BrainDecision(request_type="tool_schema", requested_tool="no_such_tool")])
    out = brain(_state("do the thing"), _ctx(llm))
    assert out["halted_reason"] == LLM_INVALID_OUTPUT
    assert len(llm.calls) == 1


def test_e_policy_engine_still_blocks_prohibited_actions() -> None:
    ctx = _ctx(ScriptedLLM([]))
    step = Step(id="s1", tool="delete_path", args={"paths": ["/"]}, rationale="delete root")
    decision = ctx.engine.decide(step, ctx.policy_ctx)
    assert decision.allowed is False
    assert decision.tier >= 1


# ── F. single-tool efficiency ───────────────────────────────────────────────


def test_f_clear_single_tool_graph_uses_one_brain_call() -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    specs = [make_spec(f"fake_{i}", base_tier=0, record=record) for i in range(9)]
    registry = registry_with(*specs, make_spec("fake_echo", base_tier=0, record=record))
    llm = ScriptedLLM([brain_action("fake_echo", {"text": "hi"})])
    ctx = _ctx(llm, registry=registry)
    saver = open_sqlite_checkpointer(":memory:")
    try:
        out = run_task(ctx, saver, "echo hi", thread_id="b9-f")
    finally:
        if getattr(saver, "conn", None) is not None:
            saver.conn.close()

    assert len(llm.calls) == 1
    assert record == [("fake_echo", {"text": "hi"})]
    assert out.interrupted is False


# ── G. multi-step planning ──────────────────────────────────────────────────


def test_g_multi_step_planner_sees_compact_capabilities_and_step_schemas() -> None:
    ctx = _ctx(ScriptedLLM([]))
    composed = compose_brain_context(
        _state("Open Notepad and write buy milk, then save it to a file."), ctx
    )
    assert "open_app" in composed.schema_names
    assert "create_file" in composed.schema_names
    assert "whatsapp_send" not in composed.schema_names
    assert "type_text" in composed.system  # capability directory
    # Only the shortlisted tools carry complete schemas.
    assert composed.system.count('"args_schema"') == composed.stats.schema_tools


def test_g_two_step_graph_has_no_executor_llm_call() -> None:
    record: list[tuple[str, dict[str, Any]]] = []
    registry = registry_with(
        make_spec("fake_a", base_tier=0, record=record),
        make_spec("fake_b", base_tier=0, record=record),
    )
    decision = brain_steps(
        [
            brain_action("fake_a", {"text": "A"}).actions[0],
            brain_action("fake_b", {"text": "B"}).actions[0],
        ]
    )
    llm = ScriptedLLM([decision])
    ctx = _ctx(llm, registry=registry)
    saver = open_sqlite_checkpointer(":memory:")
    try:
        out = run_task(ctx, saver, "first a then b", thread_id="b9-g")
    finally:
        if getattr(saver, "conn", None) is not None:
            saver.conn.close()

    assert len(llm.calls) == 1
    assert record == [("fake_a", {"text": "A"}), ("fake_b", {"text": "B"})]
    assert out.final_answer and "Could not complete" not in out.final_answer


# ── H. context budget ───────────────────────────────────────────────────────


def _conversation_with_turns(count: int) -> tuple[ConversationManager, str]:
    manager = ConversationManager()
    conversation = manager.new_conversation()
    for index in range(count):
        manager.record(
            conversation,
            f"task-{index}",
            f"Question {index} " + "q" * 350,
            f"Answer {index} " + "a" * 350,
            "answered",
        )
    return manager, conversation


def test_h_budget_trims_history_then_memory_and_keeps_the_request() -> None:
    settings = Settings(
        agent=AgentSettings(
            context_budget_tokens=3600,
            context_reserved_output_tokens=0,
            context_safety_margin=0.0,
        )
    )
    manager, conversation = _conversation_with_turns(8)
    memory = [{"kind": "preference", "text": "memory " + "m" * 280} for _ in range(10)]
    ctx = _ctx(ScriptedLLM([]), settings=settings)
    ctx.conversations = manager
    composed = compose_brain_context(
        _state(
            "Explain Python decorators in simple language.",
            conversation_id=conversation,
            memory_context=memory,
        ),
        ctx,
    )

    assert composed.stats.fits is True
    assert composed.stats.trimmed is True
    assert composed.stats.history_turns < 6
    assert composed.stats.memory_candidates < 3
    assert "Explain Python decorators in simple language." in composed.user
    assert "You are the brain" in composed.system


def test_h_too_many_candidate_schemas_are_trimmed_without_corruption() -> None:
    settings = Settings(
        agent=AgentSettings(
            context_budget_tokens=3800,
            context_reserved_output_tokens=0,
            context_safety_margin=0.0,
            tool_schema_limit=8,
        )
    )
    ctx = _ctx(ScriptedLLM([]), settings=settings)
    composed = compose_brain_context(
        _state("Open Notepad and write buy milk, then save it to a file."), ctx
    )

    assert composed.stats.fits is True
    assert composed.stats.schema_tools < 8
    assert composed.stats.schema_tools == len(composed.schema_names)
    assert composed.system.count('"args_schema"') == composed.stats.schema_tools
    assert "You are the brain" in composed.system


def test_h_oversized_large_schema_fails_closed_without_truncating_json() -> None:
    big_args = create_model(
        "BigArgs",
        **{f"field_{i}": (str, Field(default="x" * 40)) for i in range(40)},
    )
    big = ToolSpec(
        name="big_tool",
        description="A deliberately large schema.",
        args_model=big_args,
        base_tier=0,
        run=lambda args, ctx: None,  # type: ignore[arg-type,return-value]
    )
    settings = Settings(
        agent=AgentSettings(
            context_budget_tokens=900,
            context_reserved_output_tokens=0,
            context_safety_margin=0.0,
        )
    )
    ctx = _ctx(ScriptedLLM([]), settings=settings, registry=registry_with(big))
    out = brain(_state("use the big tool"), ctx)

    assert out["halted_reason"] == CONTEXT_BUDGET_EXCEEDED
    composed = compose_brain_context(_state("use the big tool"), ctx)
    assert composed.stats.fits is False
    # The canonical schema is present whole; the request is never dropped.
    assert composed.system.count('"args_schema"') == 1
    assert "use the big tool" in composed.user


def test_h_unknown_model_window_stays_unknown_and_budget_is_explicit() -> None:
    assert model_spec("groq", "openai/gpt-oss-120b").context_tokens is None
    ctx = _ctx(ScriptedLLM([]))
    composed = compose_brain_context(_state("hello"), ctx)
    assert composed.stats.budget_tokens == int(24000 * 0.85) - 1024
    assert composed.stats.estimated_input_tokens == estimate_tokens(
        composed.system
    ) + estimate_tokens(composed.user)


def test_h_context_length_provider_error_is_not_retried_forever() -> None:
    llm = ScriptedLLM([LLMContextLengthError("context window exceeded")])
    out = brain(_state("Explain Python decorators."), _ctx(llm))
    assert len(llm.calls) == 1
    assert out["halted_reason"] == CONTEXT_BUDGET_EXCEEDED


# ── I. zero-cost deterministic routing ─────────────────────────────────────


def test_i_deterministic_time_path_makes_no_llm_call() -> None:
    llm = ScriptedLLM([])
    out = brain(_state("what time is it"), _ctx(llm))
    assert out["api_calls"] == 0
    assert out["plan"] is not None
    assert llm.calls == []


# ── J. telemetry and regression contracts ──────────────────────────────────


def test_j_context_telemetry_is_counts_only_and_allow_listed() -> None:
    record = LLMAttemptRecord(
        task_id="t-b9",
        provider="groq",
        model="openai/gpt-oss-120b",
        role="planner",
        outcome=TaskAttemptOutcome.SUCCESS,
        context=LLMContextStats(
            estimated_input_tokens=123,
            candidate_tools=5,
            schema_tools=4,
            memory_candidates=2,
            history_turns=1,
            trimmed=True,
        ),
    )
    detail = record.detail()
    assert detail["estimated_input_tokens"] == 123
    assert detail["context_trimmed"] is True
    assert "prompt" not in detail and "response" not in detail
    vocab = TaskStoreVocabulary(
        providers=frozenset({"groq"}),
        models=frozenset({("groq", "openai/gpt-oss-120b")}),
        tools=frozenset(),
        nodes=frozenset(),
        failure_codes=frozenset(),
        failure_categories=frozenset({"unknown"}),
    )
    clean = sanitize_detail(TaskEventType.LLM_CALL, detail, vocab)
    assert clean["estimated_input_tokens"] == 123
    assert clean["schema_tools"] == 4


def test_j_capability_index_is_a_view_of_the_registry() -> None:
    settings = Settings()
    registry = build_default_registry(settings)
    index = build_capability_index(registry)
    assert index.count() == registry.count()
    assert index.names() == tuple(registry.names())
    directory = render_capability_directory(index)
    assert "open_app" in directory
    assert "args_schema" not in directory  # no full schemas in the directory
    assert discover_candidates(index, "Open Notepad.", limit=5)[0] == "open_app"


def test_j_replan_context_uses_compact_directory_and_bounded_schemas() -> None:
    ctx = _ctx(ScriptedLLM([]))
    state = {
        "user_input": "open notepad and type hello",
        "plan": Plan(goal="open notepad", steps=[]),
        "memory_context": [],
        "error": None,
    }
    composed = compose_replan_context(state, ctx, toolcalls=[], goal="open notepad")
    assert composed.stats.fits is True
    assert "Capability directory" in composed.system
    assert composed.system.count('"args_schema"') == composed.stats.schema_tools
    assert "brain" not in composed.user.lower() or "brain" not in composed.system.lower()
