from __future__ import annotations

import hashlib

from jarvis.agent.target_binding import (
    TargetBindingStore,
    authorization_hash,
)


class FakeClock:
    def __init__(self, start: float = 500.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


ACTION = "a" * 64
FP = "f" * 64
ORIGIN = "https://web.whatsapp.com"
CID = "conf-1"


def _store(clock=None) -> TargetBindingStore:
    return TargetBindingStore(clock=clock or FakeClock())


def _put(store: TargetBindingStore, **over: object):
    args: dict[str, object] = {
        "action_hash": ACTION,
        "fingerprint": FP,
        "origin": ORIGIN,
        "scope_confirmation_id": CID,
    }
    args.update(over)
    return store.put("t1", "s1", **args)  # type: ignore[arg-type]


def test_authorization_hash_matches_reviewed_formula() -> None:
    payload = ACTION + "\n" + FP + "\n" + ORIGIN + "\n" + CID
    expected = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    assert authorization_hash(ACTION, FP, ORIGIN, CID) == expected
    assert authorization_hash(ACTION, FP, ORIGIN, CID) != authorization_hash(ACTION, FP, "", CID)


def test_put_then_consume_is_one_shot() -> None:
    store = _store()
    binding = _put(store)
    assert binding.authorization_hash == authorization_hash(ACTION, FP, ORIGIN, CID)

    first = store.consume(
        "t1", "s1", action_hash=ACTION, fingerprint=FP, origin=ORIGIN, scope_confirmation_id=CID
    )
    assert first.ok and first.binding is binding

    replay = store.consume(
        "t1", "s1", action_hash=ACTION, fingerprint=FP, origin=ORIGIN, scope_confirmation_id=CID
    )
    assert not replay.ok and replay.reason == "no_binding"


def test_recomputed_field_mismatch_refused_and_invalidated() -> None:
    cases = {
        "fingerprint": {
            "action_hash": ACTION,
            "fingerprint": "x" * 64,
            "origin": ORIGIN,
            "scope_confirmation_id": CID,
        },
        "action_hash": {
            "action_hash": "b" * 64,
            "fingerprint": FP,
            "origin": ORIGIN,
            "scope_confirmation_id": CID,
        },
        "origin": {
            "action_hash": ACTION,
            "fingerprint": FP,
            "origin": "https://evil.example",
            "scope_confirmation_id": CID,
        },
        "confirmation": {
            "action_hash": ACTION,
            "fingerprint": FP,
            "origin": ORIGIN,
            "scope_confirmation_id": "other",
        },
    }
    for field, args in cases.items():
        store = _store()
        _put(store)
        result = store.consume("t1", "s1", **args)  # type: ignore[arg-type]
        assert not result.ok and result.reason == "mismatch", field
        # A mismatch invalidates the binding: it is gone, not reusable.
        assert store.get("t1", "s1") is None, field
        reuse = store.consume(
            "t1", "s1", action_hash=ACTION, fingerprint=FP, origin=ORIGIN, scope_confirmation_id=CID
        )
        assert not reuse.ok and reuse.reason == "no_binding", field


def test_missing_binding_refused() -> None:
    store = _store()
    result = store.consume(
        "t1", "s1", action_hash=ACTION, fingerprint=FP, origin=ORIGIN, scope_confirmation_id=CID
    )
    assert not result.ok and result.reason == "no_binding"


def test_put_supersedes_previous_binding_for_key() -> None:
    store = _store()
    _put(store, fingerprint="1" * 64)
    latest = _put(store, fingerprint="2" * 64)
    assert store.get("t1", "s1") == latest
    stale = store.consume(
        "t1",
        "s1",
        action_hash=ACTION,
        fingerprint="1" * 64,
        origin=ORIGIN,
        scope_confirmation_id=CID,
    )
    assert not stale.ok and stale.reason == "mismatch"
    # Any mismatch invalidates the slot: fresh resolution is required next.
    assert store.get("t1", "s1") is None


def test_ttl_expiry_refused_and_invalidated() -> None:
    clock = FakeClock()
    store = _store(clock)
    _put(store, ttl_s=5.0)
    clock.advance(5.0)
    result = store.consume(
        "t1", "s1", action_hash=ACTION, fingerprint=FP, origin=ORIGIN, scope_confirmation_id=CID
    )
    assert not result.ok and result.reason == "expired"
    assert store.get("t1", "s1") is None
    reuse = store.consume(
        "t1", "s1", action_hash=ACTION, fingerprint=FP, origin=ORIGIN, scope_confirmation_id=CID
    )
    assert not reuse.ok and reuse.reason == "no_binding"


def test_discard_and_clear_task() -> None:
    store = _store()
    _put(store)
    store.put(
        "t1",
        "s2",
        action_hash=ACTION,
        fingerprint=FP,
        origin=ORIGIN,
        scope_confirmation_id=CID,
    )
    store.put(
        "t2",
        "s1",
        action_hash=ACTION,
        fingerprint=FP,
        origin=ORIGIN,
        scope_confirmation_id=CID,
    )
    store.discard("t1", "s1")
    assert store.get("t1", "s1") is None
    assert store.get("t1", "s2") is not None
    store.clear_task("t1")
    assert store.get("t1", "s2") is None
    assert store.get("t2", "s1") is not None  # other task untouched


def test_same_task_task_id_whitespace_normalised() -> None:
    store = _store()
    store.put(
        "  t1 ",
        " s1 ",
        action_hash=ACTION,
        fingerprint=FP,
        origin=ORIGIN,
        scope_confirmation_id=CID,
    )
    assert store.get("t1", "s1") is not None
