"""Checkpoint serializer security tests.

The SQLite checkpointer must only ever revive the state models that
:func:`jarvis.agent.graph.build_secure_serde` explicitly allowlists (the four
state models plus the ``ReplanDecision`` boundary model stored by the replan
node), must never fall back to pickle, and must never enable the permissive
"warn but allow" ``allowed_msgpack_modules=True`` mode.  ``tests/conftest.py``
additionally turns on ``LANGGRAPH_STRICT_MSGPACK`` so any unmanaged
(strict-mode) code path becomes loud instead of silently importing untrusted
types.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.checkpoint.serde._msgpack import STRICT_MSGPACK_ENABLED
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import BaseModel

from jarvis.agent.graph import build_secure_serde, open_sqlite_checkpointer
from jarvis.agent.state import Decision, Plan, Step, StepResult

_LOGGER = "langgraph.checkpoint.serde.jsonplus"

_ALLOWED = {
    ("jarvis.agent.state", "Plan"),
    ("jarvis.agent.state", "Step"),
    ("jarvis.agent.state", "Decision"),
    ("jarvis.agent.state", "StepResult"),
    ("jarvis.agent.schemas", "ReplanDecision"),
}


def test_strict_msgpack_is_on_in_the_test_env() -> None:
    assert STRICT_MSGPACK_ENABLED is True


def test_build_secure_serde_is_explicit_not_permissive() -> None:
    serde = build_secure_serde()
    assert isinstance(serde, JsonPlusSerializer)
    assert serde.pickle_fallback is False
    assert serde._allowed_json_modules is None
    assert set(serde._allowed_msgpack_modules) == _ALLOWED


def test_checkpointer_uses_the_secure_serde(tmp_path: Any) -> None:
    saver = open_sqlite_checkpointer(str(tmp_path / "checkpoints.db"))
    assert isinstance(saver.serde, JsonPlusSerializer)
    assert saver.serde.pickle_fallback is False
    assert set(saver.serde._allowed_msgpack_modules) == _ALLOWED
    saver.conn.close()


def _fixture_objects() -> list[Plan | Step | Decision | StepResult]:
    plan = Plan(
        goal="open a page",
        steps=[Step(id="s1", tool="open_url", args={"url": "https://example.com"}, rationale="r")],
    )
    return [
        plan,
        plan.steps[0],
        Decision(
            step_id="s1",
            tier=1,
            allowed=True,
            needs_confirm=True,
            needs_unlock=False,
            summary="open_url …",
            action_hash="a" * 64,
        ),
        StepResult(step_id="s1", ok=True, output="opened", verified=True),
    ]


def test_serde_roundtrips_allowlisted_state_models() -> None:
    serde = build_secure_serde()
    for obj in _fixture_objects():
        out = serde.loads_typed(serde.dumps_typed(obj))
        assert type(out) is type(obj)
        assert out == obj


def test_serde_refuses_unregistered_types_as_plain_data() -> None:
    class Sneaky(BaseModel):
        payload: str = "boom"

    serde = build_secure_serde()
    out = serde.loads_typed(serde.dumps_typed(Sneaky()))
    assert type(out) is dict
    assert out == {"payload": "boom"}


def test_allowlisted_flows_emit_no_serde_warnings(caplog: Any) -> None:
    serde = build_secure_serde()
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        for obj in _fixture_objects():
            serde.loads_typed(serde.dumps_typed(obj))
    assert "Blocked deserialization" not in caplog.text
    assert "Deserializing unregistered" not in caplog.text


def test_unregistered_type_is_warned_once(caplog: Any) -> None:
    class ZzzUnallowlistedBlocked(BaseModel):
        payload: str = "zoom"

    serde = build_secure_serde()
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        serde.loads_typed(serde.dumps_typed(ZzzUnallowlistedBlocked()))
    assert "Blocked deserialization" in caplog.text
