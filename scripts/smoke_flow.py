#!/usr/bin/env python
"""End-to-end smoke test of the agent flow (no pytest, no network).

Runs the real LangGraph with fakes only: the scripted ``FakeLLM`` from
``jarvis.llm.client``, the fake tools in ``tests/support.py``, and a temporary
SQLite checkpointer.  Nothing here reaches a model, a browser or the network.

    .venv\\Scripts\\python.exe scripts/smoke_flow.py

Scenarios: S1 a conversation answer, S2 one safe action that runs, S3 every LLM
provider failing, S4 a clarification then the action, S5 an invalid plan that is
repaired, S6 a confirmation that is approved, S7 a confirmation that is refused,
S8 a failing tool that is replanned, S9 a cancel before the approved step.  Each
prints the node path (the A1 trace), the final answer, the halted reason and
PASS/FAIL; exit code 1 if any scenario fails.
"""

from __future__ import annotations

import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT, _ROOT / "src", _ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from jarvis.agent.context import make_app_context
from jarvis.agent.graph import open_sqlite_checkpointer
from jarvis.agent.nodes import telemetry
from jarvis.agent.runner import TaskOutcome, resume_task, run_task
from jarvis.agent.state import Plan, Step
from jarvis.config import Settings
from jarvis.llm.client import FakeLLM
from jarvis.tools.base import Cancelled, CancelToken
from support import (
    approve,
    brain_action,
    brain_clarify,
    brain_conversation,
    deny,
    make_spec,
    registry_with,
    replan_continue,
)

CONVERSATION_ANSWER = "Here is your answer."
CLARIFY_QUESTION = "Which file did you mean?"
"""Scenario result: (outcome or ``None`` when the run raised, passed, note)."""
Scenario = tuple[TaskOutcome | None, bool, str]


def _keep_trace_after_respond(task_id: str) -> None:
    """Keep the A1 trace instead of freeing it, so this script can print it.

    ``respond`` discards the trace once the task row has been written.  This
    observer only replaces that one call for the duration of the script; the
    graph itself is untouched, and the recorded path stays readable through
    ``telemetry.get_trace`` after the run.
    """
    del task_id


@contextmanager
def _session(llm: FakeLLM, *specs: Any) -> Iterator[tuple[Any, Any]]:
    """Yield one (context, checkpointer) pair on a temporary database."""
    ctx = make_app_context(Settings(), llm=llm, registry=registry_with(*specs))
    with tempfile.TemporaryDirectory() as tmp:
        saver = open_sqlite_checkpointer(str(Path(tmp) / "checkpoints.db"))
        try:
            yield ctx, saver
        finally:
            saver.conn.close()


def _run(llm: FakeLLM, record: list[tuple[str, dict[str, Any]]]) -> TaskOutcome:
    """Run one task through the real graph against one fake tool."""
    with _session(llm, make_spec("fake_echo", base_tier=0, record=record)) as (ctx, saver):
        return run_task(ctx, saver, "smoke test task")


def _node_path(out: TaskOutcome) -> str:
    """The run's node path from the A1 trace (short codes), or a marker."""
    task_id = str(out.state.get("task_id") or out.task_id)
    return telemetry.format_trace(telemetry.get_trace(task_id)) or "(none recorded)"


def _scenario_conversation() -> Scenario:
    """S1: a plain conversation answer runs no tool."""
    record: list[tuple[str, dict[str, Any]]] = []
    out = _run(FakeLLM([brain_conversation(CONVERSATION_ANSWER)]), record)
    passed = (
        out.final_answer == CONVERSATION_ANSWER
        and not out.results
        and not record
        and out.halted_reason is None
    )
    return out, passed, ""


def _scenario_safe_action() -> Scenario:
    """S2: one Tier 0 action is planned, executed once and recorded."""
    record: list[tuple[str, dict[str, Any]]] = []
    out = _run(FakeLLM([brain_action("fake_echo", {"text": "hi"})]), record)
    passed = (
        len(out.results) == 1
        and out.results[0].ok is True
        and len(record) == 1
        and out.halted_reason is None
    )
    return out, passed, ""


def _scenario_all_llm_fail() -> Scenario:
    """S3: with every provider failing the run ends with a clear message."""
    record: list[tuple[str, dict[str, Any]]] = []
    out = _run(FakeLLM([]), record)  # an exhausted script raises LLMError on every call
    passed = bool(out.final_answer) and bool(out.halted_reason) and not record
    return out, passed, ""


def _scenario_clarify_then_act() -> Scenario:
    """S4: a clarification question, then the answer, then the action."""
    record: list[tuple[str, dict[str, Any]]] = []
    llm = FakeLLM([brain_clarify(CLARIFY_QUESTION), brain_action("fake_echo", {"text": "hi"})])
    with _session(llm, make_spec("fake_echo", base_tier=0, record=record)) as (ctx, saver):
        first = run_task(ctx, saver, "open it")
        asked = first.interrupted and (first.confirmation or {}).get("question") == CLARIFY_QUESTION
        out = resume_task(ctx, saver, first.task_id, "the report")
    passed = (
        asked
        and out.interrupted is False
        and len(out.results) == 1
        and out.results[0].ok is True
        and len(record) == 1
        and out.halted_reason is None
    )
    return out, passed, f"asked first: {asked}"


def _scenario_repair_plan() -> Scenario:
    """S5: a plan naming an unknown tool is rejected and repaired once."""
    record: list[tuple[str, dict[str, Any]]] = []
    llm = FakeLLM([brain_action("no_such_tool", {}), brain_action("fake_echo", {"text": "hi"})])
    with _session(llm, make_spec("fake_echo", base_tier=0, record=record)) as (ctx, saver):
        out = run_task(ctx, saver, "do the thing")
    passed = (
        out.halted_reason is None
        and len(out.results) == 1
        and out.results[0].ok is True
        and len(record) == 1
    )
    return out, passed, f"validate attempts: {out.state.get('validate_attempts')}"


def _scenario_confirm_yes() -> Scenario:
    """S6: a Tier 1 step waits for approval and then runs once."""
    record: list[tuple[str, dict[str, Any]]] = []
    llm = FakeLLM([brain_action("fake_echo", {"text": "hi"})])
    with _session(llm, make_spec("fake_echo", base_tier=1, record=record)) as (ctx, saver):
        first = run_task(ctx, saver, "echo hi")
        pending = first.interrupted and first.confirmation is not None and not record
        out = resume_task(ctx, saver, first.task_id, approve(first.confirmation or {}))
    passed = (
        pending
        and out.interrupted is False
        and len(out.results) == 1
        and out.results[0].ok is True
        and len(record) == 1
    )
    return out, passed, f"asked first: {pending}"


def _scenario_confirm_no() -> Scenario:
    """S7: refusing the confirmation runs nothing and says so."""
    record: list[tuple[str, dict[str, Any]]] = []
    llm = FakeLLM([brain_action("fake_echo", {"text": "hi"})])
    with _session(llm, make_spec("fake_echo", base_tier=1, record=record)) as (ctx, saver):
        first = run_task(ctx, saver, "echo hi")
        out = resume_task(ctx, saver, first.task_id, deny(first.confirmation or {}))
    passed = (
        not record
        and not out.results
        and "Refused by you" in (out.final_answer or "")
        and out.interrupted is False
    )
    return out, passed, ""


def _scenario_tool_fails() -> Scenario:
    """S8: a failing tool is replanned into a working step."""
    record: list[tuple[str, dict[str, Any]]] = []
    replacement = Plan(
        goal="repair",
        steps=[Step(id="s2", tool="fake_fix", args={"text": "fixed"}, rationale="repair path")],
    )
    llm = FakeLLM(
        [brain_action("fake_echo", {"text": "hi"}), replan_continue(replacement)],
    )
    specs = (
        make_spec("fake_echo", base_tier=0, record=record, run_ok=False),
        make_spec("fake_fix", base_tier=0, record=record),
    )
    with _session(llm, *specs) as (ctx, saver):
        out = run_task(ctx, saver, "do the thing")
    calls = [name for name, _args in record]
    # The failing step is retried up to ``agent.max_retries_per_step`` first,
    # then the replanner's replacement plan runs; only its result survives.
    passed = (
        out.halted_reason is None
        and calls[0] == "fake_echo"
        and calls[-1] == "fake_fix"
        and calls.count("fake_echo") >= 2
        and len(out.results) == 1
        and out.results[-1].ok is True
    )
    return out, passed, f"tools called: {calls}"


def _scenario_cancelled() -> Scenario:
    """S9: cancelling after the prompt runs nothing, even if approval arrives."""
    record: list[tuple[str, dict[str, Any]]] = []
    llm = FakeLLM([brain_action("fake_echo", {"text": "hi"})])
    token = CancelToken()
    note = ""
    try:
        with _session(llm, make_spec("fake_echo", base_tier=1, record=record)) as (ctx, saver):
            first = run_task(ctx, saver, "echo hi", cancel=token)
            token.cancel()
            resume_task(ctx, saver, first.task_id, approve(first.confirmation or {}), cancel=token)
    except Cancelled:
        note = "the run raised Cancelled at the next node"
    passed = not record and bool(note)
    return None, passed, note or "no cancel was observed"


SCENARIOS = (
    ("S1 conversation answer", _scenario_conversation),
    ("S2 one safe action", _scenario_safe_action),
    ("S3 all LLM providers fail", _scenario_all_llm_fail),
    ("S4 clarify, then answer, then act", _scenario_clarify_then_act),
    ("S5 invalid plan, then repair", _scenario_repair_plan),
    ("S6 confirmation yes", _scenario_confirm_yes),
    ("S7 confirmation no", _scenario_confirm_no),
    ("S8 tool fails, then replans", _scenario_tool_fails),
    ("S9 cancelled before acting", _scenario_cancelled),
)


def main() -> int:
    """Run every scenario; return 0 when all pass, 1 otherwise."""
    telemetry.discard_trace = _keep_trace_after_respond
    failed = 0
    for name, scenario in SCENARIOS:
        print(f"[{name}]")
        try:
            out, passed, note = scenario()
        except Exception as exc:  # noqa: BLE001 - a crash is a failed scenario here
            print(f"   error: {type(exc).__name__}: {exc}")
            print("   FAIL")
            failed += 1
            continue
        if note:
            print(f"   note: {note}")
        if out is None:
            print("   outcome: none (the run ended by raising)")
            print(f"   results: {[]}")
        else:
            print(f"   node path: {_node_path(out)}")
            print(f"   final answer: {out.final_answer!r}")
            print(f"   halted_reason: {out.halted_reason!r}")
            print(f"   results: {[(r.step_id, r.ok) for r in out.results]}")
        print("   PASS" if passed else "   FAIL")
        if not passed:
            failed += 1
    print(f"smoke: {len(SCENARIOS) - failed}/{len(SCENARIOS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
