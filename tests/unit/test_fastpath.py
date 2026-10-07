"""Local fast path: closed vocabulary, LLM skipped, all gates unchanged (4.8)."""

from __future__ import annotations

from typing import Any

from jarvis.agent import fastpath
from jarvis.agent.context import make_app_context
from jarvis.agent.graph import build_graph
from jarvis.agent.nodes.brain import LLM_NOT_CONFIGURED, brain
from jarvis.agent.nodes.validate import validate as validate_fn
from jarvis.config import AppSettings, Settings


class _ExplodingLLM:
    """Any call to the LLM is a test failure."""

    def structured(self, **kwargs: Any) -> Any:
        raise AssertionError("fast path must not call the LLM")

    def text(self, **kwargs: Any) -> Any:
        raise AssertionError("fast path must not call the LLM")


# ── matcher: positives ──────────────────────────────────────────────────────


def test_time_phrases_hit_get_time() -> None:
    for text in (
        "What time is it?",
        "WHAT TIME IS IT?",
        "what is the time",
        "whats the time",
        "what's the time",
        "tell me the time",
    ):
        hit = fastpath.match(text, Settings())
        assert hit is not None, text
        tool, args, _phrase = hit
        assert tool == "get_time"
        assert args == {}


def test_windows_phrases_hit_list_windows() -> None:
    for text in ("what windows are open", "list windows", "list the windows"):
        hit = fastpath.match(text, Settings())
        assert hit is not None, text
        tool, args, _phrase = hit
        assert tool == "list_windows"
        assert args == {}


def test_open_variants_hit_open_app_case_insensitive() -> None:
    cases = {
        "open notepad": "notepad",
        "Open Notepad": "notepad",
        "OPEN NOTEPAD": "notepad",
        "launch chrome": "chrome",
        "start calculator": "calculator",
    }
    for text, expected_name in cases.items():
        hit = fastpath.match(text, Settings())
        assert hit is not None, text
        tool, args, _phrase = hit
        assert tool == "open_app"
        assert args == {"name": expected_name}


def test_multiword_app_key_matches_exactly() -> None:
    apps = AppSettings.model_validate({"notepad": "notepad.exe", "visual studio code": "code.exe"})

    hit = fastpath.match("open visual studio code", Settings(apps=apps))

    assert hit is not None
    assert hit[0] == "open_app"
    assert hit[1] == {"name": "visual studio code"}


# ── matcher: negatives (all must fall through to the LLM) ───────────────────


def test_negatives_never_fast_path() -> None:
    for text in (
        "time",
        "open chrome and search cats",
        "what time is it in tokyo",
        "open ../../x",
        "open notepad.exe",
        "open C:\\x",
        "open the calculator",
        "start chrom",
        "open notes",
        "a" * 61,
        "",
        "open",
        "Open Notepad.",  # decision 4: raw "." -> LLM (fail-closed)
    ):
        assert fastpath.match(text, Settings()) is None, text


def test_phrase_wins_over_app_key() -> None:
    apps = AppSettings.model_validate({"notepad": "notepad.exe", "windows": "notepad.exe"})

    hit = fastpath.match("list windows", Settings(apps=apps))

    assert hit is not None
    assert hit[0] == "list_windows"


# ── plan validity through the real validate node ────────────────────────────


def test_plans_validate_with_expected_shape() -> None:
    ctx = make_app_context(Settings())
    for text in ("what time is it", "open notepad"):
        plan = fastpath.plan_for(text, Settings())
        assert plan is not None
        assert plan.kind == "tool"
        assert plan.goal.startswith("fastpath: ")
        step = plan.steps[0]
        assert step.id == "s1"
        ctx.registry.get(step.tool).args_model.model_validate(step.args)
        out = validate_fn({"plan": plan}, ctx)
        assert out["error"] is None
        assert out["halted_reason"] is None


# ── brain: LLM is never called on a hit; non-hits are unchanged ─────────────


def test_brain_hit_skips_even_an_exploding_llm() -> None:
    ctx = make_app_context(Settings(), llm=_ExplodingLLM())

    update = brain({"user_input": "what time is it", "task_id": "t1"}, ctx)

    assert update["plan"].steps[0].tool == "get_time"
    assert update["request_kind"] == "tool"
    assert update["api_calls"] == 0
    assert update["tokens"] == 0


def test_brain_hit_works_with_no_provider_configured() -> None:
    ctx = make_app_context(Settings(), llm=None)

    update = brain({"user_input": "open notepad", "task_id": "t1"}, ctx)

    assert update["plan"].steps[0].tool == "open_app"
    assert update["plan"].steps[0].args == {"name": "notepad"}


def test_brain_non_hit_without_provider_halts_exactly_as_before() -> None:
    ctx = make_app_context(Settings(), llm=None)

    update = brain({"user_input": "open chrome and search cats", "task_id": "t1"}, ctx)

    assert update == {"halted_reason": LLM_NOT_CONFIGURED}


# ── graph-level: full pipeline, zero API calls ──────────────────────────────


def test_graph_fastpath_end_to_end_with_exploding_llm() -> None:
    ctx = make_app_context(Settings(), llm=_ExplodingLLM())
    graph = build_graph(ctx, None, None)

    values = graph.invoke(
        {"user_input": "what time is it", "source": "terminal", "task_id": "fp1"},
        {"configurable": {"thread_id": "fp1"}},
    )

    assert values["final_answer"]
    assert values["api_calls"] == 0
    assert values["tokens"] == 0
    assert values["plan"].goal == "fastpath: what time is it"
