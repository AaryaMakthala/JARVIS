"""Tests for :mod:`jarvis.agent.runtime_args`."""

from __future__ import annotations

from typing import Any

from jarvis.agent.runtime_args import mark_runtime_args
from jarvis.agent.state import Plan, Step


def _plan(**step_kwargs: Any) -> Plan:
    step_kwargs.setdefault("args", {"x": "hi"})
    return Plan(
        goal="t",
        steps=[
            Step(id="s1", tool="fake", rationale="r", **step_kwargs),
        ],
    )


# ---- plain literal args ----


def test_plain_literal_args_not_runtime() -> None:
    p = _plan()
    mark_runtime_args(p)
    assert p.steps[0].resolved_from_runtime is False


# ---- depends_on_untrusted ----


def test_depends_on_untrusted_marks_runtime() -> None:
    p = _plan(depends_on_untrusted=True)
    mark_runtime_args(p)
    assert p.steps[0].resolved_from_runtime is True


# ---- placeholder / reference patterns ----


def test_placeholder_arg_marks_runtime() -> None:
    p = _plan(args={"x": "use the $ earlier value"})
    mark_runtime_args(p)
    assert p.steps[0].resolved_from_runtime is True


def test_nested_placeholder_marks_runtime() -> None:
    nested: dict[str, Any] = {
        "x": ["safe", "output of s1"],
        "y": {"z": "{{prev}}"},
    }
    p = _plan(args=nested)
    mark_runtime_args(p)
    assert p.steps[0].resolved_from_runtime is True


# ---- replan ----


def test_flag_set_on_replan_too() -> None:
    p = _plan(depends_on_untrusted=True)
    # Run twice to model the "replan also resolved" expectation.
    mark_runtime_args(p)
    mark_runtime_args(p)
    assert p.steps[0].resolved_from_runtime is True


# ---- no mutation of args / hash ----


def test_args_and_hash_unchanged_by_marking() -> None:
    p = _plan(args={"path": "C:\\Users\\aarya\\Desktop\\notes.txt"})
    before_args = p.steps[0].args
    before_id = p.steps[0].id
    before_tool = p.steps[0].tool
    mark_runtime_args(p)
    assert p.steps[0].args is before_args  # same object; definitely not rewritten
    assert p.steps[0].id == before_id
    assert p.steps[0].tool == before_tool
    # resolved_from_runtime is the only field that changed.
    assert p.steps[0].resolved_from_runtime is False  # plain literal -> False


# ---- old checkpoints / payloads without the field ----


def test_old_step_without_field_loads() -> None:
    # Build a Step exactly as an older checkpoint / payload would: without ever
    # passing resolved_from_runtime.  Pydantic v2 should default it to False.
    old = Step(id="s1", tool="fake", args={"x": "hi"}, rationale="r")
    assert getattr(old, "resolved_from_runtime", None) is False
    # Marking must not blow up on a step that was never explicitly set.
    p = _plan()
    mark_runtime_args(p)
    assert p.steps[0].resolved_from_runtime is False
