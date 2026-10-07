"""4.16: a failure must say WHY in the journal.

Covers the two log lines added for that: the policy gate logs
``Decision.reasons`` for every decision, and the tool-end line carries the
truncated ``result.error`` so a failed tool (e.g. ``type_text`` failing three
times) shows its reason instead of a bare ``ok=False``.
"""

from __future__ import annotations

import logging

import pytest
from pydantic import BaseModel

from jarvis.agent.context import make_app_context
from jarvis.agent.nodes.policy_gate import policy_gate
from jarvis.agent.state import Plan, Step
from jarvis.config import Settings
from jarvis.tools.base import ToolContext, ToolSpec
from jarvis.tools.registry import ToolRegistry


class _NoArgs(BaseModel):
    """Empty args for the synthetic failing tool."""


def _boom(args: BaseModel, ctx: ToolContext) -> None:
    """Fail with a long message so the 200-char truncation is exercised."""
    raise ValueError("x" * 300)


def test_failure_logs_carry_the_reason(caplog: pytest.LogCaptureFixture) -> None:
    settings = Settings()
    registry = ToolRegistry()

    # (a) a tool that raises: the "tool end" line must say why, truncated.
    spec = ToolSpec(
        name="boom_416",
        description="synthetic failing tool",
        args_model=_NoArgs,
        base_tier=0,
        run=_boom,
    )
    with caplog.at_level(logging.INFO, logger="jarvis.tools.boom_416"):
        result = spec.execute(_NoArgs(), ToolContext(settings=settings))
    assert result.ok is False
    end = next(r.getMessage() for r in caplog.records if "tool end" in r.getMessage())
    assert "ok=False" in end
    assert "error=ValueError: " + "x" * 188 in end  # [:200] keeps 188 of the 300 x's
    assert "x" * 189 not in end

    # (b) a refused decision: the gate must log Decision.reasons.
    ctx = make_app_context(settings=settings, registry=registry)
    plan = Plan(
        goal="x",
        steps=[
            Step(
                id="s1",
                tool="no_such_tool",
                args={},
                rationale="r",
                depends_on_untrusted=False,
                resolved_from_runtime=False,
            )
        ],
    )
    with caplog.at_level(logging.INFO, logger="jarvis.agent.nodes.policy_gate"):
        out = policy_gate({"plan": plan}, ctx)
    assert "halted_reason" in out
    gate = next(r.getMessage() for r in caplog.records if "policy gate" in r.getMessage())
    assert "reasons=" in gate
    assert "unknown tool not in the registry" in gate
