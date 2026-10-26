#!/usr/bin/env python
"""End-to-end smoke test of the agent flow (no pytest, no network).

Runs the real LangGraph with fakes only: the scripted ``FakeLLM`` from
``jarvis.llm.client``, the fake tools in ``tests/support.py``, and a temporary
SQLite checkpointer.  Nothing here reaches a model, a browser or the network.

    .venv\\Scripts\\python.exe scripts/smoke_flow.py

Scenarios: S1 a conversation answer, S2 one safe action that runs, S3 every LLM
provider failing.  Each prints the node path (the A1 trace), the final answer,
the halted reason and PASS/FAIL; exit code 1 if any scenario fails.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT, _ROOT / "src", _ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from jarvis.agent.context import make_app_context
from jarvis.agent.graph import open_sqlite_checkpointer
from jarvis.agent.nodes import telemetry
from jarvis.agent.runner import TaskOutcome, run_task
from jarvis.config import Settings
from jarvis.llm.client import FakeLLM
from support import brain_action, brain_conversation, make_spec, registry_with

CONVERSATION_ANSWER = "Here is your answer."


def _keep_trace_after_respond(task_id: str) -> None:
    """Keep the A1 trace instead of freeing it, so this script can print it.

    ``respond`` discards the trace once the task row has been written.  This
    observer only replaces that one call for the duration of the script; the
    graph itself is untouched, and the recorded path stays readable through
    ``telemetry.get_trace`` after the run.
    """
    del task_id


def _run(llm: FakeLLM, record: list[tuple[str, dict[str, Any]]]) -> TaskOutcome:
    """Run one task through the real graph against one fake tool."""
    ctx = make_app_context(
        Settings(),
        llm=llm,
        registry=registry_with(make_spec("fake_echo", base_tier=0, record=record)),
    )
    with tempfile.TemporaryDirectory() as tmp:
        saver = open_sqlite_checkpointer(str(Path(tmp) / "checkpoints.db"))
        try:
            return run_task(ctx, saver, "smoke test task")
        finally:
            saver.conn.close()


def _node_path(out: TaskOutcome) -> str:
    """The run's node path from the A1 trace (short codes), or a marker."""
    task_id = str(out.state.get("task_id") or out.task_id)
    return telemetry.format_trace(telemetry.get_trace(task_id)) or "(none recorded)"


def _scenario_conversation() -> tuple[TaskOutcome, bool]:
    """S1: a plain conversation answer runs no tool."""
    record: list[tuple[str, dict[str, Any]]] = []
    out = _run(FakeLLM([brain_conversation(CONVERSATION_ANSWER)]), record)
    passed = (
        out.final_answer == CONVERSATION_ANSWER
        and not out.results
        and not record
        and out.halted_reason is None
    )
    return out, passed


def _scenario_safe_action() -> tuple[TaskOutcome, bool]:
    """S2: one Tier 0 action is planned, executed once and recorded."""
    record: list[tuple[str, dict[str, Any]]] = []
    out = _run(FakeLLM([brain_action("fake_echo", {"text": "hi"})]), record)
    passed = (
        len(out.results) == 1
        and out.results[0].ok is True
        and len(record) == 1
        and out.halted_reason is None
    )
    return out, passed


def _scenario_all_llm_fail() -> tuple[TaskOutcome, bool]:
    """S3: with every provider failing the run ends with a clear message."""
    record: list[tuple[str, dict[str, Any]]] = []
    out = _run(FakeLLM([]), record)  # an exhausted script raises LLMError on every call
    passed = bool(out.final_answer) and bool(out.halted_reason) and not record
    return out, passed


SCENARIOS = (
    ("S1 conversation answer", _scenario_conversation),
    ("S2 one safe action", _scenario_safe_action),
    ("S3 all LLM providers fail", _scenario_all_llm_fail),
)


def main() -> int:
    """Run every scenario; return 0 when all pass, 1 otherwise."""
    telemetry.discard_trace = _keep_trace_after_respond
    failed = 0
    for name, scenario in SCENARIOS:
        print(f"[{name}]")
        try:
            out, passed = scenario()
        except Exception as exc:  # noqa: BLE001 - a crash is a failed scenario here
            print(f"   error: {type(exc).__name__}: {exc}")
            print("   FAIL")
            failed += 1
            continue
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
