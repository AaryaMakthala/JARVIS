"""Recipient prechecks: refuse before confirmation, unlock or dispatch.

The messaging tools declare a read-only ``ToolSpec.precheck``.  It runs in the
policy gate *before* ``interrupt()`` and again in ``act`` immediately before
dispatch, so an unknown or ambiguous contact can never reach a Tier-2 approval
prompt, an unlock, a browser session or a send.

These tests use the real graph and the real tool specs, with an injected
contact store and a scripted brain (no browser, no network, no Windows).
"""

from __future__ import annotations

from typing import Any

import pytest

from jarvis.agent.batch_approval import decide_and_build_payload
from jarvis.agent.context import make_app_context
from jarvis.agent.graph import open_sqlite_checkpointer
from jarvis.agent.nodes.act import act
from jarvis.agent.runner import run_task
from jarvis.agent.state import Decision, Plan, Step
from jarvis.config import Settings
from jarvis.llm.client import FakeLLM
from jarvis.messaging.contacts import ContactStore, make_recipient_precheck, recipient_error
from jarvis.policy import rules
from jarvis.tools.registry import build_default_registry
from jarvis.tools.whatsapp import WhatsAppSendArgs, make_whatsapp_send_spec
from jarvis.tools.whatsapp_web import make_whatsapp_open_chat_spec
from support import brain_action, registry_with

_PHONE = "919876543210"
_TASK = "t1"


@pytest.fixture(autouse=True)
def _windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test runs as if on Windows unless it overrides this."""
    monkeypatch.setattr("jarvis.platform_guard.is_windows", lambda: True)


def _store(tmp_path: Any, *contacts: tuple[str, str]) -> ContactStore:
    store = ContactStore(tmp_path / "contacts.json")
    for name, number in contacts or (("Rahul", _PHONE),):
        store.add(name, number)
    return store


def _saver(tmp_path: Any) -> Any:
    return open_sqlite_checkpointer(str(tmp_path / "checkpoints.db"))


def _run(tmp_path: Any, store: Any, tool: str, args: dict[str, Any]) -> Any:
    ctx = make_app_context(
        Settings(),
        llm=FakeLLM([brain_action(tool, args)]),
        registry=registry_with(
            make_whatsapp_send_spec(store=store)
            if tool == "whatsapp_send"
            else make_whatsapp_open_chat_spec(store=store)
        ),
    )
    return run_task(ctx, _saver(tmp_path), "do the messaging action")


# ── the shared helper ─────────────────────────────────────────────────────


def test_recipient_error_accepts_exactly_one_contact(tmp_path: Any) -> None:
    assert recipient_error(_store(tmp_path), "Rahul") is None


def test_recipient_error_reports_unknown_contacts(tmp_path: Any) -> None:
    error = recipient_error(_store(tmp_path), "Pappa")
    assert error is not None
    assert "no contact matches" in error
    assert "Rahul" in error


def test_recipient_error_reports_ambiguous_contacts(tmp_path: Any) -> None:
    store = _store(tmp_path, ("Ali Khan", "9112345678"), ("Ali Ahmed", _PHONE))
    error = recipient_error(store, "Ali")
    assert error is not None
    assert "multiple contacts match" in error
    assert "Ali Khan" in error and "Ali Ahmed" in error


def test_the_precheck_returns_the_error_without_raising(tmp_path: Any) -> None:
    check = make_recipient_precheck(_store(tmp_path))
    assert check(WhatsAppSendArgs(contact="Pappa", message="hi"), None) is not None  # type: ignore[arg-type]
    assert check(WhatsAppSendArgs(contact="Rahul", message="hi"), None) is None  # type: ignore[arg-type]


# ── the tools declare it ──────────────────────────────────────────────────


def test_the_default_messaging_specs_declare_a_precheck() -> None:
    registry = build_default_registry()
    for name in ("whatsapp_send", "whatsapp_send_file", "whatsapp_open_chat"):
        spec = registry.get_optional(name)
        assert spec is not None
        assert spec.precheck is not None, name


# ── the gate refuses before any interrupt ─────────────────────────────────


def test_an_unknown_recipient_is_refused_before_confirmation(tmp_path: Any) -> None:
    outcome = _run(
        tmp_path,
        _store(tmp_path),
        "whatsapp_send",
        {"contact": "Pappa", "message": "hi"},
    )

    assert outcome.interrupted is False
    assert outcome.confirmation is None
    assert "no contact matches" in (outcome.halted_reason or "")
    assert outcome.results == []
    assert not any(getattr(r, "ok", False) for r in outcome.results)


def test_an_ambiguous_recipient_is_refused_before_confirmation(tmp_path: Any) -> None:
    store = _store(tmp_path, ("Ali Khan", "9112345678"), ("Ali Ahmed", _PHONE))
    outcome = _run(
        tmp_path,
        store,
        "whatsapp_send",
        {"contact": "Ali", "message": "hi"},
    )

    assert outcome.interrupted is False
    assert "multiple contacts match" in (outcome.halted_reason or "")


def test_open_chat_is_prechecked_too(tmp_path: Any) -> None:
    outcome = _run(tmp_path, _store(tmp_path), "whatsapp_open_chat", {"contact": "Nobody"})

    assert outcome.interrupted is False
    assert "no contact matches" in (outcome.halted_reason or "")


def test_a_valid_recipient_still_reaches_the_tier2_confirmation(tmp_path: Any) -> None:
    outcome = _run(
        tmp_path,
        _store(tmp_path),
        "whatsapp_send",
        {"contact": "Rahul", "message": "hi"},
    )

    assert outcome.interrupted is True
    assert outcome.interrupt_kind == "confirm"
    assert outcome.confirmation is not None
    assert outcome.confirmation["tier"] == 2
    assert outcome.confirmation["needs_unlock"] is True
    assert outcome.results == []


# ── act closes the window between the gate and dispatch ───────────────────


def _act_state(step: Step, approved: list[str]) -> dict[str, Any]:
    return {
        "plan": Plan(goal="send", steps=[step]),
        "step_index": 0,
        "task_id": _TASK,
        "approved_hashes": list(approved),
        "gated_resolved_paths": {},
        "results": [],
        "user_input": "send hi",
    }


def test_act_refuses_when_the_recipient_disappeared_after_approval(tmp_path: Any) -> None:
    step = Step(
        id="s1",
        tool="whatsapp_send",
        args={"contact": "Rahul", "message": "hi"},
        rationale="send it",
    )
    empty = ContactStore(tmp_path / "empty-contacts.json")  # no contacts at dispatch time
    ctx = make_app_context(
        Settings(),
        registry=registry_with(make_whatsapp_send_spec(store=empty)),
    )
    spec = ctx.registry.get(step.tool)
    args = spec.args_model.model_validate(step.args)
    approved = [str(rules.action_hash(spec, args))]

    update = act(_act_state(step, approved), ctx)

    result = update["results"][-1]
    assert result.ok is False
    assert "no contact matches" in (result.error or "")
    assert update.get("halted_reason")


# ── a batch never includes a precheck-failing step ────────────────────────


def _decision(step_id: str, action_hash: str) -> Decision:
    return Decision(
        step_id=step_id,
        tier=1,
        allowed=True,
        needs_confirm=True,
        needs_unlock=False,
        summary=f"do {step_id}",
        action_hash=action_hash,
    )


def _two_step_plan() -> Plan:
    return Plan(
        goal="two steps",
        steps=[
            Step(id="s1", tool="t1", args={}, rationale="one"),
            Step(id="s2", tool="t2", args={}, rationale="two"),
        ],
    )


def test_a_blocked_step_ends_the_contiguous_batch() -> None:
    plan = _two_step_plan()
    decisions = {"s1": _decision("s1", "a"), "s2": _decision("s2", "b")}

    assert decide_and_build_payload(plan, decisions, step_index=0)["type"] == "plan_approval"

    payload = decide_and_build_payload(plan, decisions, step_index=0, blocked_steps={"s2"})
    assert payload["type"] == "confirm"
    assert payload["step_id"] == "s1"
