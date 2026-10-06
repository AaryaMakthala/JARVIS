"""Tests for :mod:`jarvis.daemon.confirmations` and the new payload fields (I3)."""

from __future__ import annotations

from jarvis.agent.schemas import ConfirmationRequest
from jarvis.daemon.confirmations import ConfirmationRegistry, plan_hash
from jarvis.daemon.protocol import ConfirmRequest, ConfirmResponse
from jarvis.daemon.task_runtime import CONFIRMATION_TIMEOUT_S

_HASH_A = "a" * 64
_HASH_B = "b" * 64


def test_late_proceed_for_a_leaves_b_unapproved() -> None:
    reg = ConfirmationRegistry()
    a = reg.issue("t1", "s1", 1, _HASH_A, plan_hash([_HASH_A]))
    b = reg.issue("t2", "s1", 1, _HASH_B, plan_hash([_HASH_B]))

    # A "proceed" carrying A's id cannot satisfy B.
    assert reg.check(a.confirmation_id, b.plan_hash, b.action_hash).ok is False
    assert reg.check(a.confirmation_id, b.plan_hash, b.action_hash).reason == "plan_hash_mismatch"

    # A is still valid; resolving A does not consume B.
    assert reg.check(a.confirmation_id, a.plan_hash, a.action_hash).ok is True
    assert reg.resolve(a.confirmation_id) is True
    assert reg.check(b.confirmation_id, b.plan_hash, b.action_hash).ok is True


def test_stale_id_refused() -> None:
    reg = ConfirmationRegistry()
    first = reg.issue("t1", "s1", 1, _HASH_A, plan_hash([_HASH_A]))
    second = reg.issue("t1", "s2", 1, _HASH_B, plan_hash([_HASH_B]))

    assert (
        reg.check(first.confirmation_id, first.plan_hash, first.action_hash).reason == "superseded"
    )
    assert reg.check(second.confirmation_id, second.plan_hash, second.action_hash).ok is True


def test_unknown_or_empty_id_refused() -> None:
    reg = ConfirmationRegistry()

    assert reg.check("", _HASH_A, _HASH_A).reason == "unknown_id"
    assert reg.check("deadbeef", _HASH_A, _HASH_A).reason == "unknown_id"


def test_expired_refused() -> None:
    now = [1000.0]
    reg = ConfirmationRegistry(clock=lambda: now[0])
    pc = reg.issue("t1", "s1", 1, _HASH_A, plan_hash([_HASH_A]))

    now[0] += CONFIRMATION_TIMEOUT_S + 0.001

    assert reg.check(pc.confirmation_id, pc.plan_hash, pc.action_hash).reason == "expired"


def test_expiry_reuses_single_constant() -> None:
    now = [0.0]
    reg = ConfirmationRegistry(clock=lambda: now[0])

    pc = reg.issue("t1", "s1", 1, _HASH_A, plan_hash([_HASH_A]))

    assert pc.expires_at == now[0] + CONFIRMATION_TIMEOUT_S


def test_plan_hash_mismatch_refused() -> None:
    reg = ConfirmationRegistry()
    pc = reg.issue("t1", "s1", 1, _HASH_A, plan_hash([_HASH_A]))

    assert (
        reg.check(pc.confirmation_id, plan_hash(["other"]), _HASH_A).reason == "plan_hash_mismatch"
    )


def test_action_hash_mismatch_refused() -> None:
    reg = ConfirmationRegistry()
    pc = reg.issue("t1", "s1", 1, _HASH_A, plan_hash([_HASH_A]))

    assert reg.check(pc.confirmation_id, pc.plan_hash, _HASH_B).reason == "action_hash_mismatch"


def test_id_is_one_shot() -> None:
    reg = ConfirmationRegistry()
    pc = reg.issue("t1", "s1", 1, _HASH_A, plan_hash([_HASH_A]))

    assert reg.check(pc.confirmation_id, pc.plan_hash, pc.action_hash).ok is True
    assert reg.resolve(pc.confirmation_id) is True
    assert reg.check(pc.confirmation_id, pc.plan_hash, pc.action_hash).reason == "already_used"


def test_plan_hash_deterministic_and_order_sensitive() -> None:
    assert plan_hash(["a", "b"]) == plan_hash(["a", "b"])
    assert plan_hash(["a", "b"]) != plan_hash(["b", "a"])


def test_old_payload_without_new_fields_still_loads() -> None:
    req = ConfirmationRequest.model_validate(
        {"type": "confirm", "step_id": "s1", "tier": 1, "summary": "x", "action_hash": "h"}
    )
    assert req.confirmation_id == ""
    assert req.plan_hash == ""
    assert req.expires_at == 0.0

    confirm = ConfirmRequest.model_validate(
        {"type": "confirm_request", "task_id": "t", "action_hash": "h"}
    )
    assert confirm.confirmation_id == ""
    assert confirm.plan_hash == ""

    response = ConfirmResponse.model_validate(
        {"type": "confirm_response", "task_id": "t", "approved": True, "action_hash": "h"}
    )
    assert response.confirmation_id == ""
    assert response.plan_hash == ""
