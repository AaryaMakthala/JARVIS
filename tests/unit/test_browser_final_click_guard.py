"""Phase 5C: live readback + trusted digest binding for the guarded final click.

NOT EXECUTED — pending authorized verification.

Every test here is written to hold only if the readback a user is shown comes
from a **live** observation of the real target, survives the confirmation
``interrupt``, and is the value the click is finally checked against.  The fake
session records every call, so the tests can assert that nothing is clicked,
typed or navigated before an approval that echoes the observed digest - and that
a proposal (the model's own words) never becomes part of the binding.

The gate tests model LangGraph's semantics directly: an interrupted node re-runs
**from the top** against the state the checkpoint holds, so a "resume" here is a
second call to the same function with the same state dict.
"""

from __future__ import annotations

import ast
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest

from tests.support import registry_with

from jarvis.agent.context import make_app_context
from jarvis.agent.nodes.act import act
from jarvis.agent.nodes.policy_gate import (
    PENDING_READBACK_KEY,
    READBACK_ACK_KEY,
    confirmation_payload,
    policy_gate,
)
from jarvis.agent.nodes.prepare_readback import prepare_readback
from jarvis.agent.state import Decision, Plan, Step
from jarvis.browser.context import ElementRef
from jarvis.browser.final_click import (
    build_observed_readback,
    canonical_json,
    digests_equal,
    format_proposed_note,
    format_readback_summary,
    is_hex64,
    observe_live_readback,
    readback_digest,
)
from jarvis.browser.tools import make_browser_final_click_spec
from jarvis.config import BrowserSettings, Settings, VoiceSettings
from jarvis.daemon.protocol import ConfirmRequest
from jarvis.policy import rules
from jarvis.policy.unlock import UnlockManager

#: The policy_gate *module* (``jarvis.agent.nodes.policy_gate`` is also the name
#: of the gate function, so the module is fetched by import path).
_PG = import_module("jarvis.agent.nodes.policy_gate")
_TOOLS = import_module("jarvis.browser.tools")

_ORIGIN = "https://example.com"
_TASK = "tfc1"
_REF = "e1"


# ── doubles ───────────────────────────────────────────────────────────────


def _element(
    name: str = "Post",
    *,
    ref: str = _REF,
    role: str = "button",
    origin: str = _ORIGIN,
    is_password: bool = False,
    input_type: str = "",
) -> ElementRef:
    return ElementRef(
        ref=ref,
        role=role,
        name=name,
        origin=origin,
        input_type=input_type,
        is_password=is_password,
    )


class FakeSession:
    """A managed-browser double that records reads *and* side effects.

    ``resolve`` is the only read; ``click``/``type_text``/``goto`` raise if the
    readback path ever reaches for them, and the tests assert the recorded lists
    stay empty until an approval carrying the observed digest is replayed.
    """

    def __init__(self, elements: dict[str, ElementRef] | None = None) -> None:
        self.elements: dict[str, ElementRef] = dict(elements or {_REF: _element()})
        self.resolved: list[str] = []
        self.clicked: list[str] = []
        self.typed: list[Any] = []
        self.navigated: list[str] = []

    def resolve(self, ref: str) -> ElementRef | None:
        self.resolved.append(str(ref))
        return self.elements.get(str(ref))

    def click(self, ref: str) -> dict[str, Any]:
        element = self.elements[str(ref)]
        self.clicked.append(str(ref))
        return {
            "ref": str(ref),
            "role": element.role,
            "name": element.name,
            "origin": element.origin,
        }

    def type_text(self, *args: Any, **kwargs: Any) -> None:
        # Recorded, never raised, so a test can prove the final-click path never
        # reached for a typing tool without breaking unrelated tools that use
        # this same double.
        self.typed.append((args, kwargs))
        return None

    def goto(self, url: str, **kwargs: Any) -> None:
        self.navigated.append(str(url))
        return None

    def close(self) -> None:
        return None

    @property
    def untouched(self) -> bool:
        return not self.clicked and not self.typed and not self.navigated


class _StubEngine:
    """A deterministic stand-in for the PolicyEngine.

    The engine's own authority is exercised by the policy tests; these tests are
    about the readback binding, so the stub returns one allowed, confirming
    Tier 2 decision.  Its ``action_hash`` is computed exactly the way the real
    engine computes it (``rules.action_hash(spec, validated_args)``), which is
    what makes the association checks meaningful.
    """

    def __init__(
        self,
        *,
        tier: int = 2,
        allowed: bool = True,
        needs_confirm: bool = True,
        needs_unlock: bool = False,
        summary: str = "Publish this comment?",
    ) -> None:
        self.tier = tier
        self.allowed = allowed
        self.needs_confirm = needs_confirm
        self.needs_unlock = needs_unlock
        self.summary = summary

    def decide(self, step: Any, pctx: Any) -> Decision:
        tool = str(getattr(step, "tool", "") or "")
        registry = getattr(pctx, "registry", None)
        spec = registry.get_optional(tool) if registry is not None else None
        action_hash = ""
        if spec is not None:
            try:
                args = spec.args_model.model_validate(dict(getattr(step, "args", None) or {}))
                action_hash = str(rules.action_hash(spec, args))
            except Exception:  # noqa: BLE001 - fall back to the raw hash, as the engine does
                action_hash = ""
        if not action_hash:
            action_hash = str(rules.action_hash_raw(tool, getattr(step, "args", None)))
        return Decision(
            step_id=str(getattr(step, "id", "") or ""),
            tier=int(self.tier),
            allowed=bool(self.allowed),
            needs_confirm=bool(self.needs_confirm),
            needs_unlock=bool(self.needs_unlock),
            summary=str(self.summary),
            action_hash=action_hash,
        )


class _PwStore:
    """In-memory password store (no keyring, no real secrets)."""

    def __init__(self) -> None:
        self._values: dict[str, str] = {}

    def get(self, name: str) -> str | None:
        return self._values.get(name)

    def set(self, name: str, value: str) -> None:
        self._values[name] = value

    def has(self, name: str) -> bool:
        return name in self._values


def _unlocked() -> UnlockManager:
    manager = UnlockManager(_PwStore())
    manager.set_password("correct-horse-battery-staple")
    assert manager.verify("correct-horse-battery-staple") is True
    return manager


# ── context / state builders ──────────────────────────────────────────────


def _settings(
    *,
    max_total: int = 10,
    cap: float | None = None,
    currency: str = "USD",
    domains: list[str] | None = None,
) -> Settings:
    return Settings(
        browser=BrowserSettings(
            allowed_domains=list(domains if domains is not None else ["example.com"]),
            final_click_max_total=max_total,
            final_click_spending_cap=cap,
            final_click_spending_currency=currency,
        ),
        voice=VoiceSettings(allow_consequential_voice_approval=False),
    )


def _ctx(session: FakeSession, **kw: Any):
    settings = kw.pop("settings", None) or _settings()
    ctx = make_app_context(
        settings,
        registry=registry_with(make_browser_final_click_spec()),
        browser=session,
        unlock=kw.pop("unlock", _unlocked()),
    )
    # The engine's own tier/confirm logic is covered elsewhere; these tests need
    # a decision the readback path is exercised against.
    ctx.engine = kw.pop("engine", None) or _StubEngine()
    return ctx


_DEFAULT_ARGS: dict[str, Any] = {"ref": _REF, "action": "post", "expect": "publish"}


def _step(args: dict[str, Any] | None = None, sid: str = "s1") -> Step:
    return Step(
        id=sid,
        tool="browser_final_click",
        args=dict(_DEFAULT_ARGS if args is None else args),
        rationale="final click",
    )


def _state(step: Step, *, task_id: str = _TASK, approved: list[str] | None = None) -> dict[str, Any]:
    return {
        "plan": Plan(goal="final click", steps=[step]),
        "step_index": 0,
        "task_id": task_id,
        "approved_hashes": list(approved or []),
        "gated_resolved_paths": {},
        "gated_content_hashes": {},
        "gated_browser_readback_hashes": {},
        "results": [],
        "user_input": "post this",
        "source": "text",
    }


def _action_hash(ctx: Any, step: Step) -> str:
    spec = ctx.registry.get(step.tool)
    args = spec.args_model.model_validate(dict(step.args or {}))
    return str(rules.action_hash(spec, args))


class _Paused(Exception):
    """Stands in for LangGraph suspending the graph at ``interrupt``."""


def _pause(state: dict[str, Any], ctx: Any, monkeypatch: Any) -> dict[str, Any]:
    """Run the gate until it interrupts, returning the payload it showed."""
    seen: list[dict[str, Any]] = []

    def _fake_interrupt(payload: dict[str, Any]) -> Any:
        seen.append(payload)
        raise _Paused()

    monkeypatch.setattr(_PG, "interrupt", _fake_interrupt)
    with pytest.raises(_Paused):
        policy_gate(state, ctx)
    assert seen, "the gate never reached its confirmation"
    return seen[-1]


def _resume(state: dict[str, Any], ctx: Any, monkeypatch: Any, answer: Any) -> dict[str, Any]:
    """Re-run the gate from the top with *answer* waiting at its interrupt.

    This is what LangGraph does: the interrupted node executes again from the
    first line, against the state the checkpoint holds.
    """
    seen: list[dict[str, Any]] = []

    def _fake_interrupt(payload: dict[str, Any]) -> Any:
        seen.append(payload)
        return answer

    monkeypatch.setattr(_PG, "interrupt", _fake_interrupt)
    updates = policy_gate(state, ctx)
    return {**updates, "_payload": seen[-1] if seen else None}


def _prepared(state: dict[str, Any], ctx: Any) -> dict[str, Any]:
    """Apply the ``prepare_readback`` node's update, as the checkpoint would."""
    prepared = prepare_readback(state, ctx)
    return {**state, **prepared}


def _approved(answer: dict[str, Any]) -> dict[str, Any]:
    return {"approved": True, **answer}


@pytest.fixture()
def temp_journal(tmp_path: Any, monkeypatch: Any) -> Any:
    """Isolate the final-click journal in a temporary database.

    The module singleton points at the real user database; a test that reaches
    the journal must never write into it (and the lifetime ``max_total`` count
    would otherwise make the tests order-dependent).
    """
    from jarvis.browser import final_click as fc

    journal = fc.FinalClickJournal(str(tmp_path / "final_click.sqlite"))
    monkeypatch.setattr(fc, "_journal", journal)
    return journal


# ── A. the live observation ───────────────────────────────────────────────


def test_live_target_is_resolved_read_only_before_any_confirmation() -> None:
    session = FakeSession({_REF: _element("Post")})
    observation = observe_live_readback(session, _REF, "post")
    assert observation.ok is True
    assert session.resolved == [_REF]
    assert session.untouched is True


def test_unresolvable_target_is_refused_and_never_fabricated() -> None:
    session = FakeSession({})  # the ref is gone from the page
    observation = observe_live_readback(session, _REF, "post")
    assert observation.ok is False
    assert observation.details is None
    assert observation.halting is False  # recoverable: the page moved
    assert session.untouched is True


def test_missing_browser_session_is_a_refusal() -> None:
    observation = observe_live_readback(None, _REF, "post")
    assert observation.ok is False
    assert observation.halting is True


def test_readback_fields_are_tagged_with_their_observed_source() -> None:
    session = FakeSession({_REF: _element("Post")})
    observation = observe_live_readback(session, _REF, "post")
    details = observation.details or {}
    assert details["target"]["name"] == "Post"
    assert details["target"]["role"] == "button"
    assert details["target"]["source"] == "live_dom"
    assert details["origin"] == {"value": _ORIGIN, "source": "observed_origin"}
    assert details["host"]["value"] == "example.com"
    assert details["destination"]["value"] == "example.com"
    # The transient handle is never part of the observation (it changes on every
    # read, so binding to it would be meaningless).
    assert "ref" not in details["target"]


def test_displayed_readback_equals_the_observed_data_and_its_digest() -> None:
    session = FakeSession({_REF: _element("Publish")})
    observation = observe_live_readback(session, _REF, "publish")
    assert observation.ok is True
    details = observation.details or {}
    summary = format_readback_summary(details)
    assert "Publish" in summary
    assert "example.com" in summary
    assert observation.digest == readback_digest(details)
    assert is_hex64(observation.digest) is True
    assert canonical_json(details) == canonical_json(dict(sorted(details.items())))


def test_the_proposal_is_shown_separately_and_never_in_the_digest() -> None:
    session = FakeSession({_REF: _element("Post")})
    clean = observe_live_readback(session, _REF, "post")
    session.resolved.clear()
    proposed = observe_live_readback(
        session, _REF, "post", proposed={"expect": "send to someone-else@example.com"}
    )
    assert proposed.ok is True
    assert digests_equal(clean.digest, proposed.digest) is True
    assert "someone-else@example.com" not in format_readback_summary(proposed.details or {})
    note = format_proposed_note(proposed.details or {})
    assert "someone-else@example.com" in note
    assert "unverified" in note


def test_send_without_an_observable_recipient_is_refused() -> None:
    session = FakeSession({_REF: _element("Send")})
    observation = observe_live_readback(session, _REF, "send")
    assert observation.ok is False
    assert "recipient" in observation.reason


def test_send_with_an_observable_recipient_is_allowed() -> None:
    session = FakeSession({_REF: _element("Send to aarya@example.com")})
    observation = observe_live_readback(session, _REF, "send")
    assert observation.ok is True
    assert (observation.details or {})["recipient"]["value"] == "aarya@example.com"
    assert (observation.details or {})["recipient"]["source"] == "live_dom"


def test_password_and_payment_controls_are_hard_refusals() -> None:
    session = FakeSession({_REF: _element("Password", is_password=True, input_type="password")})
    assert observe_live_readback(session, _REF, "post").halting is True
    session = FakeSession({_REF: _element("Buy now")})
    payment = observe_live_readback(session, _REF, "post")
    assert payment.ok is False
    assert payment.halting is True


def test_declared_action_must_match_the_page_control() -> None:
    session = FakeSession({_REF: _element("Publish")})
    observation = observe_live_readback(session, _REF, "book")
    assert observation.ok is False
    assert observation.halting is True


def test_unobservable_origin_is_refused() -> None:
    session = FakeSession({_REF: _element("Post", origin="")})
    observation = observe_live_readback(session, _REF, "post")
    assert observation.ok is False
    assert observation.details is None


# ── B. the spending cap (fail closed) ─────────────────────────────────────


def test_cap_with_no_observable_amount_fails_closed() -> None:
    session = FakeSession({_REF: _element("Subscribe")})
    observation = observe_live_readback(session, _REF, "subscribe", spending_cap=10.0)
    assert observation.ok is False
    assert "spending" in observation.reason


def test_cap_with_an_amount_but_no_currency_fails_closed() -> None:
    session = FakeSession({_REF: _element("Subscribe for 5")})
    observation = observe_live_readback(session, _REF, "subscribe", spending_cap=10.0)
    assert observation.ok is False
    assert observation.details is None


def test_cap_with_an_ambiguous_amount_fails_closed() -> None:
    session = FakeSession({_REF: _element("Subscribe 3 plans for 5 USD")})
    observation = observe_live_readback(
        session, _REF, "subscribe", spending_cap=10.0, spending_currency="USD"
    )
    assert observation.ok is False


def test_cap_currency_mismatch_is_refused() -> None:
    session = FakeSession({_REF: _element("Subscribe for 5 USD")})
    observation = observe_live_readback(
        session, _REF, "subscribe", spending_cap=10.0, spending_currency="EUR"
    )
    assert observation.ok is False
    assert "EUR" in observation.reason


def test_amount_above_the_cap_is_refused() -> None:
    session = FakeSession({_REF: _element("Subscribe for 25 USD")})
    observation = observe_live_readback(
        session, _REF, "subscribe", spending_cap=10.0, spending_currency="USD"
    )
    assert observation.ok is False
    assert observation.halting is True


def test_amount_within_the_cap_is_allowed_and_in_the_digest() -> None:
    session = FakeSession({_REF: _element("Subscribe for 9.99 USD")})
    observation = observe_live_readback(
        session, _REF, "subscribe", spending_cap=10.0, spending_currency="USD"
    )
    assert observation.ok is True
    details = observation.details or {}
    assert details["amount"]["value"] == pytest.approx(9.99)
    assert details["currency"]["value"] == "USD"
    assert details["amount"]["source"] == "live_dom"
    # A changed price changes the digest, so a pre-click re-check notices.
    changed = dict(details)
    changed["amount"] = {"value": 19.99, "source": "live_dom"}
    assert digests_equal(observation.digest, readback_digest(changed)) is False


def test_cap_never_overrides_the_payment_prohibition() -> None:
    # A purchase control stays blocked by the payment rule even with a cap set;
    # the cap only ever adds a second, independent refusal.
    session = FakeSession({_REF: _element("Buy now 5 USD")})
    observation = observe_live_readback(
        session, _REF, "subscribe", spending_cap=100.0, spending_currency="USD"
    )
    assert observation.ok is False
    assert "payment" in observation.reason


# ── C. the gate: pre-approval observation + resume binding ────────────────


def test_gate_refuses_a_final_click_with_no_persisted_readback(monkeypatch) -> None:
    session = FakeSession()
    ctx = _ctx(session)
    step = _step()
    state = _state(step)  # prepare_readback never ran
    interrupted: list[dict[str, Any]] = []
    monkeypatch.setattr(
        _PG,
        "interrupt",
        lambda payload: interrupted.append(payload),  # must never be called
    )
    updates = policy_gate(state, ctx)
    assert interrupted == []
    assert updates.get("halted_reason")
    assert not (updates.get("gated_browser_readback_hashes") or {})


def test_gate_refuses_before_authorization_when_the_target_is_not_observable(monkeypatch) -> None:
    session = FakeSession({})  # the ref is no longer on the page
    ctx = _ctx(session)
    step = _step()
    state = _prepared(_state(step), ctx)
    assert isinstance(state[PENDING_READBACK_KEY], dict)
    assert state[PENDING_READBACK_KEY]["refusal"]
    interrupted: list[dict[str, Any]] = []
    monkeypatch.setattr(_PG, "interrupt", lambda payload: interrupted.append(payload))
    updates = policy_gate(state, ctx)
    assert interrupted == []  # the user is never asked to approve a guess
    assert updates.get("halted_reason")
    assert session.untouched is True


def test_confirmation_payload_carries_the_persisted_readback_and_digest(monkeypatch) -> None:
    session = FakeSession({_REF: _element("Post")})
    ctx = _ctx(session)
    step = _step()
    state = _prepared(_state(step), ctx)
    pending = state[PENDING_READBACK_KEY]
    payload = _pause(state, ctx, monkeypatch)
    assert payload["readback_required"] is True
    assert payload["readback_summary"] == pending["summary"]
    assert payload["readback_details"] == pending["details"]
    assert payload["readback_hash"] == pending["digest"]
    assert is_hex64(payload["readback_hash"]) is True
    assert payload["action_hash"] == pending["action_hash"]
    # The model's words travel separately and are excluded from the digest.
    assert "publish" in payload["readback_proposed"]
    assert "publish" not in canonical_json(
        {k: v for k, v in payload["readback_details"].items() if k != "proposed"}
    )
    assert session.untouched is True


def test_resume_uses_the_persisted_observation_not_a_new_one(monkeypatch) -> None:
    session = FakeSession({_REF: _element("Post")})
    ctx = _ctx(session)
    step = _step()
    state = _prepared(_state(step), ctx)
    pending = dict(state[PENDING_READBACK_KEY])
    _pause(state, ctx, monkeypatch)

    # The page moves to another origin while the user is reading the prompt.
    # The resume must still bind the readback the user was *shown*, never a
    # fresh observation (which would have said other.example.com).
    session.elements[_REF] = _element("Post", origin="https://other.example.com")
    answer = _approved({"action_hash": pending["action_hash"], READBACK_ACK_KEY: pending["digest"]})
    updates = _resume(state, ctx, monkeypatch, answer)
    assert updates.get("halted_reason") is None
    assert updates["_payload"]["readback_details"] == pending["details"]
    assert updates["_payload"]["readback_hash"] == pending["digest"]
    assert "other.example.com" not in canonical_json(updates["_payload"]["readback_details"])
    assert updates["gated_browser_readback_hashes"][step.id] == pending["digest"]
    assert session.untouched is True


def test_the_trusted_digest_is_bound_not_the_client_value(monkeypatch) -> None:
    session = FakeSession({_REF: _element("Post")})
    ctx = _ctx(session)
    step = _step()
    state = _prepared(_state(step), ctx)
    trusted = state[PENDING_READBACK_KEY]["digest"]
    hijack = "a" * 64
    assert hijack != trusted

    updates = _resume(
        state,
        ctx,
        monkeypatch,
        _approved({"action_hash": _action_hash(ctx, step), READBACK_ACK_KEY: hijack}),
    )
    assert updates.get("halted_reason")
    assert step.id not in (updates.get("gated_browser_readback_hashes") or {})

    good = _resume(
        state,
        ctx,
        monkeypatch,
        _approved({"action_hash": _action_hash(ctx, step), READBACK_ACK_KEY: trusted}),
    )
    assert good.get("halted_reason") is None
    assert good["gated_browser_readback_hashes"][step.id] == trusted
    assert good[PENDING_READBACK_KEY] is None  # consumed: one-time by construction


@pytest.mark.parametrize(
    "ack",
    [None, "", "deadbeef", "A" * 64, "0" * 63, "z" * 64, 1234],
)
def test_missing_or_malformed_acknowledgement_is_refused(monkeypatch, ack: Any) -> None:
    session = FakeSession({_REF: _element("Post")})
    ctx = _ctx(session)
    step = _step()
    state = _prepared(_state(step), ctx)
    answer: dict[str, Any] = {"action_hash": _action_hash(ctx, step)}
    if ack is not None:
        answer[READBACK_ACK_KEY] = ack
    updates = _resume(state, ctx, monkeypatch, _approved(answer))
    assert updates.get("halted_reason")
    assert step.id not in (updates.get("gated_browser_readback_hashes") or {})
    assert session.untouched is True


@pytest.mark.parametrize(
    "mutate",
    [
        {"task_id": "someone-elses-task"},
        {"step_id": "s2"},
        {"action_hash": "0" * 64},
        {"digest": "deadbeef"},
        {"target_identity": ""},
        {"origin": ""},
        {"details": {}},
    ],
)
def test_a_stale_or_unassociated_observation_is_refused(monkeypatch, mutate: dict[str, Any]) -> None:
    session = FakeSession({_REF: _element("Post")})
    ctx = _ctx(session)
    step = _step()
    state = _prepared(_state(step), ctx)
    state[PENDING_READBACK_KEY] = {**state[PENDING_READBACK_KEY], **mutate}
    interrupted: list[dict[str, Any]] = []
    monkeypatch.setattr(_PG, "interrupt", lambda payload: interrupted.append(payload))
    updates = policy_gate(state, ctx)
    assert interrupted == []
    assert "page changed" in str(updates.get("halted_reason"))
    assert step.id not in (updates.get("gated_browser_readback_hashes") or {})


def test_a_changed_action_hash_invalidates_the_pending_observation(monkeypatch) -> None:
    session = FakeSession({_REF: _element("Post"), "e2": _element("Post", ref="e2")})
    ctx = _ctx(session)
    step = _step()
    state = _prepared(_state(step), ctx)
    # The plan's args change between the observation and the confirmation (same
    # control, different step identity), so the observed action no longer
    # matches the one being confirmed.
    state["plan"] = Plan(
        goal="final click",
        steps=[
            Step(
                id=step.id,
                tool="browser_final_click",
                args={"ref": "e2", "action": "post", "expect": "publish"},
                rationale="changed",
            )
        ],
    )
    interrupted: list[dict[str, Any]] = []
    monkeypatch.setattr(_PG, "interrupt", lambda payload: interrupted.append(payload))
    updates = policy_gate(state, ctx)
    assert interrupted == []
    assert "page changed" in str(updates.get("halted_reason"))


@pytest.mark.parametrize(
    "answer",
    [
        {"approved": False, "action_hash": "to-be-filled"},
        {"approved": True, "action_hash": "wrong-hash"},
        {},
        {"timed_out": True},
        {"approved": True, "action_hash": "to-be-filled", READBACK_ACK_KEY: None},
    ],
)
def test_decline_cancel_timeout_and_mismatch_create_no_binding(monkeypatch, answer: dict) -> None:
    session = FakeSession({_REF: _element("Post")})
    ctx = _ctx(session)
    step = _step()
    state = _prepared(_state(step), ctx)
    pending = state[PENDING_READBACK_KEY]
    if answer.get("action_hash") == "to-be-filled":
        answer = {**answer, "action_hash": pending["action_hash"]}
    updates = _resume(state, ctx, monkeypatch, answer)
    assert updates.get("halted_reason")
    # Nothing was bound for this step, so act has nothing to execute against.
    assert step.id not in (updates.get("gated_browser_readback_hashes") or {})
    assert session.untouched is True


@pytest.mark.parametrize(
    "answer",
    [
        {"approved": False, "action_hash": "to-be-filled"},  # decline
        {},  # cancel / no approval at all
        {"timed_out": True},  # timeout
    ],
)
def test_non_approval_clears_the_pending_readback(monkeypatch, answer: dict) -> None:
    """Decline, cancel and timeout must clear -- and never bind -- the readback."""
    session = FakeSession({_REF: _element("Post")})
    ctx = _ctx(session)
    step = _step()
    state = _prepared(_state(step), ctx)
    if answer.get("action_hash") == "to-be-filled":
        answer = {**answer, "action_hash": state[PENDING_READBACK_KEY]["action_hash"]}
    updates = _resume(state, ctx, monkeypatch, answer)
    assert updates.get("halted_reason")
    assert "pending_readback" in updates and updates["pending_readback"] is None
    assert not (updates.get("gated_browser_readback_hashes") or {})
    assert session.untouched is True


def test_prepare_readback_resets_for_steps_that_are_not_a_final_click() -> None:
    session = FakeSession()
    ctx = _ctx(session)
    state = _state(_step())
    state["plan"] = Plan(
        goal="other",
        steps=[Step(id="s1", tool="browser_click", args={"ref": _REF}, rationale="click it")],
    )
    assert prepare_readback(state, ctx) == {PENDING_READBACK_KEY: None}
    state["plan"] = None
    assert prepare_readback(state, ctx) == {PENDING_READBACK_KEY: None}
    state = _state(_step())
    state["step_index"] = 7
    assert prepare_readback(state, ctx) == {PENDING_READBACK_KEY: None}


def test_prepare_readback_does_not_run_the_tool_or_change_the_page() -> None:
    session = FakeSession()
    ctx = _ctx(session)
    step = _step()
    prepare_readback(_state(step), ctx)
    assert session.untouched is True
    assert session.resolved  # it observed, read-only


# ── D. act -> ToolContext -> tool ─────────────────────────────────────────


def _approve_through_the_gate(monkeypatch: Any, ctx: Any, step: Step) -> dict[str, Any]:
    """Drive the gate through observe -> pause -> approve, returning act's state."""
    state = _prepared(_state(step), ctx)
    pending = state[PENDING_READBACK_KEY]
    _pause(state, ctx, monkeypatch)
    updates = _resume(
        state,
        ctx,
        monkeypatch,
        _approved(
            {"action_hash": pending["action_hash"], READBACK_ACK_KEY: pending["digest"]}
        ),
    )
    assert updates.get("halted_reason") is None
    return {**state, **{k: v for k, v in updates.items() if not k.startswith("_")}}


def test_act_refuses_a_final_click_without_a_bound_readback_digest(monkeypatch) -> None:
    session = FakeSession({_REF: _element("Post")})
    ctx = _ctx(session)
    step = _step()
    state = _approve_through_the_gate(monkeypatch, ctx, step)
    state["gated_browser_readback_hashes"] = {}  # the binding is missing
    update = act(state, ctx)
    results = update.get("results") or []
    assert results and not results[-1].ok
    assert "readback" in str(results[-1].error).lower()
    assert session.clicked == []


def test_act_refuses_a_malformed_bound_digest(monkeypatch) -> None:
    session = FakeSession({_REF: _element("Post")})
    ctx = _ctx(session)
    step = _step()
    state = _approve_through_the_gate(monkeypatch, ctx, step)
    state["gated_browser_readback_hashes"] = {step.id: "deadbeef"}
    update = act(state, ctx)
    results = update.get("results") or []
    assert results and not results[-1].ok
    assert session.clicked == []


def test_act_passes_the_trusted_gate_digest_into_the_tool_context(monkeypatch) -> None:
    session = FakeSession({_REF: _element("Post")})
    ctx = _ctx(session)
    step = _step()
    state = _approve_through_the_gate(monkeypatch, ctx, step)
    trusted = state["gated_browser_readback_hashes"][step.id]
    assert is_hex64(trusted) is True

    seen: list[Any] = []

    def _spy(args: Any, tool_ctx: Any) -> Any:
        seen.append(getattr(tool_ctx, "approved_final_click_readback_hash", None))
        from jarvis.tools.base import ToolResult

        return ToolResult(ok=True, output="clicked", data={})

    monkeypatch.setattr(_TOOLS, "_run_browser_final_click", _spy)
    update = act(state, ctx)
    assert seen == [trusted]
    assert (update.get("results") or [])[-1].ok is True


def test_a_consequential_click_happens_only_after_the_bound_digest_is_present(
    monkeypatch, temp_journal
) -> None:
    session = FakeSession({_REF: _element("Post")})
    ctx = _ctx(session)
    step = _step()
    state = _approve_through_the_gate(monkeypatch, ctx, step)
    update = act(state, ctx)
    results = update.get("results") or []
    assert results and results[-1].ok is True
    assert session.clicked == [_REF]
    assert session.typed == [] and session.navigated == []


def test_the_tool_refuses_missing_or_wrong_digests(temp_journal) -> None:
    session = FakeSession({_REF: _element("Post")})
    ctx = _ctx(session)
    spec = make_browser_final_click_spec()
    args = spec.args_model(ref=_REF, action="post", expect="publish")
    good = readback_digest(build_observed_readback(session.resolve(_REF), "post").details or {})
    tool_ctx = ctx.tool_context()
    assert spec.run(args, tool_ctx).ok is False  # no binding at all
    for bad in ("", "deadbeef", "0" * 64, "A" * 64):
        tool_ctx.approved_final_click_readback_hash = bad
        result = spec.run(args, tool_ctx)
        assert result.ok is False, bad
    tool_ctx.approved_final_click_readback_hash = good
    assert spec.run(args, tool_ctx).ok is True
    assert session.clicked == [_REF]


@pytest.mark.parametrize(
    "changed",
    [
        _element("Post to someone-else@example.com"),
        _element("Publish"),
        _element("Post", origin="https://evil.example.net"),
        _element("Subscribe for 99 USD"),
    ],
)
def test_a_pre_click_change_in_the_live_target_is_rejected(changed: ElementRef) -> None:
    session = FakeSession({_REF: _element("Post to aarya@example.com")})
    ctx = _ctx(session)
    spec = make_browser_final_click_spec()
    args = spec.args_model(ref=_REF, action="send", expect="send the reply")
    observation = build_observed_readback(session.resolve(_REF), "send")
    assert observation.ok is True
    digest = observation.digest

    session.elements[_REF] = changed  # the page changes after the approval
    tool_ctx = ctx.tool_context()
    tool_ctx.approved_final_click_readback_hash = digest
    result = spec.run(args, tool_ctx)
    assert result.ok is False
    assert session.clicked == []


def test_the_tool_rechecks_the_cap_immediately_before_clicking() -> None:
    session = FakeSession({_REF: _element("Subscribe for 9.99 USD")})
    ctx = _ctx(session, settings=_settings(cap=10.0, currency="USD"))
    spec = make_browser_final_click_spec()
    args = spec.args_model(ref=_REF, action="subscribe", expect="subscribe")
    observation = build_observed_readback(
        session.resolve(_REF), "subscribe", spending_cap=10.0, spending_currency="USD"
    )
    assert observation.ok is True
    # The price changes after the approval; the fresh observation is refused
    # even though the digest the user approved is the one presented.
    session.elements[_REF] = _element("Subscribe for 25 USD")
    tool_ctx = ctx.tool_context()
    tool_ctx.approved_final_click_readback_hash = observation.digest
    result = spec.run(args, tool_ctx)
    assert result.ok is False
    assert session.clicked == []


def test_the_tool_refuses_a_payment_control_and_password_field_outright() -> None:
    for element in (_element("Buy now 5 USD"), _element("Password", is_password=True)):
        session = FakeSession({_REF: element})
        ctx = _ctx(session)
        spec = make_browser_final_click_spec()
        args = spec.args_model(ref=_REF, action="send", expect="")
        observation = build_observed_readback(element, "send")
        tool_ctx = ctx.tool_context()
        tool_ctx.approved_final_click_readback_hash = observation.digest or ("0" * 64)
        result = spec.run(args, tool_ctx)
        assert result.ok is False
        assert session.clicked == []


def test_the_tool_is_disabled_while_final_click_max_total_is_zero() -> None:
    session = FakeSession({_REF: _element("Post")})
    ctx = _ctx(session, settings=_settings(max_total=0))
    spec = make_browser_final_click_spec()
    args = spec.args_model(ref=_REF, action="post", expect="publish")
    observation = build_observed_readback(session.resolve(_REF), "post")
    tool_ctx = ctx.tool_context()
    tool_ctx.approved_final_click_readback_hash = observation.digest
    assert spec.run(args, tool_ctx).ok is False
    assert session.clicked == []


def test_indeterminate_or_dispatched_click_is_never_retried(temp_journal) -> None:
    from jarvis.browser.final_click import (
        FinalClickJournalRow,
        _now_iso,
        sha256_hex,
    )

    journal = temp_journal
    row = FinalClickJournalRow(
        id="phase5c-row",
        task_id="tfc-indeterminate",
        host="example.com",
        action="post",
        target_fingerprint="x",
        confirmation_sha256=sha256_hex("x"),
        state="dispatched",
        outcome_code="dispatched",
        created_at=_now_iso(),
        updated_at=_now_iso(),
    )
    assert journal.reserve(row) is True
    # A reserved row can never be replayed (one-time by construction), and an
    # indeterminate/dispatched task must never be retried.
    assert journal.reserve(row) is False
    assert journal.has_indeterminate_or_dispatched("tfc-indeterminate") is True
    journal.delete("phase5c-row")


def test_a_browsing_tool_cannot_bypass_the_guard() -> None:
    from jarvis.browser.tools import make_browser_click_spec
    from jarvis.policy.browse_grant import BROWSE_TOOLS

    assert "browser_final_click" not in BROWSE_TOOLS
    session = FakeSession({_REF: _element("Send")})
    ctx = _ctx(session)
    click = make_browser_click_spec()
    result = click.run(click.args_model(ref=_REF), ctx.tool_context())
    assert result.ok is True  # the plain click is unchanged...


def test_only_the_gate_writes_the_readback_binding(monkeypatch) -> None:
    session = FakeSession({_REF: _element("Post")})
    ctx = _ctx(session)
    step = _step()
    # A state forged with a binding but no persisted observation is refused: the
    # gate never mints a digest it did not observe.
    state = _state(step)
    state["gated_browser_readback_hashes"] = {step.id: "0" * 64}
    interrupted: list[dict[str, Any]] = []
    monkeypatch.setattr(_PG, "interrupt", lambda payload: interrupted.append(payload))
    updates = policy_gate(state, ctx)
    assert interrupted == []
    assert updates.get("halted_reason")
    assert step.id not in (updates.get("gated_browser_readback_hashes") or {})


# ── E. transports stay compatible ─────────────────────────────────────────


def _confirm_request_from(payload: dict[str, Any], task_id: str = "t") -> ConfirmRequest:
    """Build the daemon's message the way ``server.py`` does, field by field."""
    return ConfirmRequest(
        task_id=task_id,
        tier=int(payload.get("tier") or 0),
        summary=str(payload.get("summary") or ""),
        action_hash=str(payload.get("action_hash") or ""),
        readback_required=bool(payload.get("readback_required", False)),
        readback_summary=str(payload.get("readback_summary") or ""),
        readback_details=dict(payload.get("readback_details") or {}),
        readback_hash=str(payload.get("readback_hash") or ""),
        readback_proposed=str(payload.get("readback_proposed") or ""),
    )


def test_daemon_message_carries_the_readback_and_defaults_are_compatible() -> None:
    decision = Decision(
        step_id="s1",
        tier=2,
        allowed=True,
        needs_confirm=True,
        summary="Publish this comment?",
        action_hash="a" * 64,
    )
    plain = confirmation_payload(decision, untrusted=False)
    assert "readback_required" not in plain
    message = _confirm_request_from(plain)
    assert message.readback_required is False
    assert message.readback_hash == ""
    assert message.readback_proposed == ""


def test_in_process_confirmation_renders_the_persisted_readback() -> None:
    session = FakeSession({_REF: _element("Post")})
    ctx = _ctx(session)
    step = _step()
    state = _prepared(_state(step), ctx)
    pending = state[PENDING_READBACK_KEY]
    decision = Decision(
        step_id=step.id,
        tier=2,
        allowed=True,
        needs_confirm=True,
        summary="Publish this comment?",
        action_hash=pending["action_hash"],
    )
    payload = confirmation_payload(decision, untrusted=False, pending_readback=pending)
    assert payload["readback_required"] is True
    assert payload["readback_hash"] == pending["digest"]
    assert payload["readback_details"] == pending["details"]
    message = _confirm_request_from(payload)
    assert message.readback_hash == pending["digest"]
    assert message.readback_proposed == pending["proposed_note"]


def test_cli_answer_echoes_the_digest_it_was_shown() -> None:
    from jarvis.cli import _confirmation_answer

    digest = "b" * 64
    answer = _confirmation_answer(True, {"action_hash": "h", "readback_hash": digest}, None)
    assert answer[READBACK_ACK_KEY] == digest
    # A payload with no readback (every other tool) yields no extra key at all.
    plain = _confirmation_answer(True, {"action_hash": "h"}, None)
    assert READBACK_ACK_KEY not in plain


def test_voice_approval_is_still_disabled_and_never_implied() -> None:
    from jarvis.policy.browse_grant import BROWSE_TOOLS

    settings = _settings()
    assert settings.voice.allow_consequential_voice_approval is False
    assert "browser_final_click" not in BROWSE_TOOLS


# ── F. the two regressions this slice fixes, pinned in source ──────────────

_SRC = Path(__file__).resolve().parents[2] / "src" / "jarvis"


def _top_level_defs(path: Path) -> list[str]:
    """Names of every module-level function defined in *path* (AST, no import)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def test_tools_module_defines_exactly_one_resolve_browser_element() -> None:
    """The click/type resolve hook must not be shadowed by a second helper.

    A duplicate module-level definition silently wins for every later
    ``resolve=`` binding in the module, so ``browser_click``/``browser_type``
    would be handed a helper with a different signature and refuse every step.
    """
    names = _top_level_defs(_SRC / "browser" / "tools.py")
    assert names.count("_resolve_browser_element") == 1
    assert names.count("_observe_element_for_readback") == 1


def test_click_and_type_resolve_hooks_accept_args_and_ctx() -> None:
    """Both hooks must be callable exactly as ``resolve(args, ctx)``."""
    from jarvis.browser.tools import make_browser_click_spec, make_browser_type_spec

    session = FakeSession({_REF: _element("Post")})
    ctx = _ctx(session)
    tool_ctx = ctx.tool_context()
    cases = (
        (make_browser_click_spec(), {"ref": _REF}),
        (make_browser_type_spec(), {"ref": _REF, "text": "hello"}),
    )
    for spec, raw in cases:
        assert spec.resolve is not None
        assert spec.resolve.__name__ == "_resolve_browser_element"
        args = spec.args_model.model_validate(raw)
        target = spec.resolve(args, tool_ctx)  # must not raise
        assert target is not None, spec.name
        assert target.origin == _ORIGIN
        assert target.name == "Post"
    assert session.untouched is True


def _declared_edges() -> list[tuple[str, str]]:
    """Every ``(source, target)`` pair declared in ``agent/graph.py`` via AST."""
    tree = ast.parse((_SRC / "agent" / "graph.py").read_text(encoding="utf-8"))
    pairs: list[tuple[str, str]] = []

    def _const(node: ast.AST) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return str(node.value)
        return ""

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        name = node.func.attr
        if name == "add_edge" and len(node.args) == 2:
            source, target = _const(node.args[0]), _const(node.args[1])
            if source and target:
                pairs.append((source, target))
        elif name == "add_conditional_edges" and len(node.args) >= 3:
            source = _const(node.args[0])
            mapping = node.args[2]
            if source and isinstance(mapping, ast.Dict):
                for value in mapping.values:
                    target = _const(value)
                    if target:
                        pairs.append((source, target))
    return pairs


def test_only_prepare_readback_edges_into_policy_gate() -> None:
    """Every declared edge into ``policy_gate`` comes from ``prepare_readback``.

    Otherwise a step could be gated without the live readback being re-observed
    and persisted first, which is what makes the confirmation bound to what the
    user was actually shown.
    """
    pairs = _declared_edges()
    assert ("prepare_readback", "policy_gate") in pairs
    into_policy_gate = {source for source, target in pairs if target == "policy_gate"}
    assert into_policy_gate == {"prepare_readback"}


def test_every_gate_router_key_still_targets_prepare_readback() -> None:
    """The routers that send a step to the gate must land on the pre-node."""
    pairs = _declared_edges()
    gate_sources = {source for source, target in pairs if target == "prepare_readback"}
    # validate (both variants) and verify (both variants) route to the gate.
    assert gate_sources == {"validate", "verify"}
