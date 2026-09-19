"""Deterministic fake tools for Phase 1 agent tests (no Windows, no network).

The fake tools only append to a shared ``record`` list; asserting the list is
empty (or has the exact expected entries) is how tests prove a side effect did
or did not happen.  ``verify_ok`` lets a test simulate a post-condition check
that fails (verification failure) while the action itself "succeeded".
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from jarvis.agent.state import Plan, Step
from jarvis.tools.base import ToolContext, ToolResult, ToolSpec
from jarvis.tools.registry import ToolRegistry


class TextArgs(BaseModel):
    text: str = Field(min_length=1, max_length=64)


def make_spec(
    name: str,
    *,
    base_tier: int,
    record: list[tuple[str, dict[str, Any]]],
    verify_ok: bool = True,
) -> ToolSpec:
    """Build a fake tool that records every call into ``record``."""

    def run(args: TextArgs, ctx: ToolContext) -> ToolResult:
        record.append((name, args.model_dump()))
        return ToolResult(ok=True, output=f"{name} ran {args.text}", data={"text": args.text})

    def verify_always(args: TextArgs, result: ToolResult, ctx: ToolContext) -> ToolResult:
        return result.model_copy(update={"verified": verify_ok})

    return ToolSpec(
        name=name,
        description=f"fake tool {name} for tests",
        args_model=TextArgs,
        base_tier=base_tier,
        run=run,
        verify=verify_always,
        describe=lambda args: f"{name} text={args.text!r}",
    )


def registry_with(*specs: ToolSpec) -> ToolRegistry:
    """A fresh registry containing exactly the given specs."""
    registry = ToolRegistry()
    for spec in specs:
        registry.register(spec)
    return registry


def echo_plan(text: str = "hi") -> Plan:
    """A single-step plan that calls the fake Tier-1 tool ``fake_echo``."""
    return Plan(
        goal=f"echo {text}",
        steps=[Step(id="s1", tool="fake_echo", args={"text": text}, rationale="echo it")],
    )


def approve(confirmation: dict[str, Any]) -> dict[str, Any]:
    """A user answer that approves the exact confirmation payload."""
    return {"approved": True, "action_hash": confirmation["action_hash"]}


def deny(confirmation: dict[str, Any]) -> dict[str, Any]:
    """A user answer that says no (keeps the hash, but disapproved)."""
    return {"approved": False, "action_hash": confirmation["action_hash"]}


def tamper(confirmation: dict[str, Any]) -> dict[str, Any]:
    """A user answer whose action_hash does not match the payload."""
    return {"approved": True, "action_hash": "0" * 64}
