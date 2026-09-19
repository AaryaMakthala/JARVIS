"""UnlockManager tests: Argon2id password lifecycle, TTL and lockout backoff.

docs/03_SECURITY_AND_POLICY.md invariants 8 and 14 - the password only ever
exists as an Argon2id hash and verification never touches logs.  All timing is
injected (a fake clock) so expiry and lockouts are deterministic and fast.
"""

from __future__ import annotations

import logging

import pytest

from jarvis.config import PolicySettings, Settings
from jarvis.policy.unlock import HASH_NAME, MAX_LOCKOUT_SECONDS, MIN_PASSWORD_LENGTH, UnlockManager

PW = "correct-horse-battery"
SHORT = "short"


class FakeClock:
    def __init__(self, start: float = 1_000.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


class FakePasswordStore:
    """In-memory PasswordStore that can be made to fail like a bad keyring."""

    def __init__(self) -> None:
        self._values: dict[str, str] = {}
        self._fail = False

    def get(self, name: str) -> str | None:
        if self._fail:
            raise RuntimeError("keyring unreachable")
        return self._values.get(name)

    def set(self, name: str, value: str) -> None:
        if self._fail:
            raise RuntimeError("keyring unreachable")
        self._values[name] = value

    def has(self, name: str) -> bool:
        if self._fail:
            raise RuntimeError("keyring unreachable")
        return name in self._values

    def fail(self) -> None:
        self._fail = True


def _manager(
    clock: FakeClock | None = None, store: FakePasswordStore | None = None
) -> tuple[UnlockManager, FakePasswordStore, FakeClock]:
    store = store or FakePasswordStore()
    clock = clock or FakeClock()
    manager = UnlockManager(store, settings=Settings(), clock=clock)
    return manager, store, clock


# ---------------------------------------------------------------------------
# password lifecycle
# ---------------------------------------------------------------------------


def test_set_password_stores_only_an_argon2id_hash() -> None:
    manager, store, _ = _manager()
    manager.set_password(PW)
    stored = store.get(HASH_NAME)
    assert stored is not None
    assert stored.startswith("$argon2id$")
    assert PW not in stored


def test_set_password_rejects_too_short() -> None:
    manager, store, _ = _manager()
    with pytest.raises(ValueError):
        manager.set_password("x" * (MIN_PASSWORD_LENGTH - 1))
    assert store.has(HASH_NAME) is False


def test_verify_correct_opens_a_session_with_ttl() -> None:
    manager, _store, clock = _manager()
    manager.set_password(PW)
    assert manager.is_unlocked() is False
    assert manager.verify(PW) is True
    assert manager.is_unlocked() is True
    assert manager.remaining_seconds() > 0
    clock.advance(Settings().policy.unlock_ttl_seconds)
    assert manager.is_unlocked() is False  # TTL expired
    assert manager.remaining_seconds() == 0


def test_verify_wrong_password_never_unlocks() -> None:
    manager, _store, _ = _manager()
    manager.set_password(PW)
    assert manager.verify(PW + "nope") is False
    assert manager.is_unlocked() is False


def test_verify_with_no_password_stored_is_false() -> None:
    manager, _store, _ = _manager()
    assert manager.verify(PW) is False
    assert manager.has_password() is False


def test_lock_invalidates_an_unlocked_session() -> None:
    manager, _store, _clock = _manager()
    manager.set_password(PW)
    assert manager.verify(PW) is True
    manager.lock()
    assert manager.is_unlocked() is False
    assert manager.remaining_seconds() == 0


def test_clock_going_backwards_does_not_extend_session() -> None:
    manager, _store, clock = _manager()
    manager.set_password(PW)
    assert manager.verify(PW) is True
    clock.advance(-60)  # clock jump backwards must not re-open the session
    assert manager.is_unlocked() is True  # expiry is monotonic: still open
    clock.advance(Settings().policy.unlock_ttl_seconds + 61)
    assert manager.is_unlocked() is False


# ---------------------------------------------------------------------------
# lockout
# ---------------------------------------------------------------------------


def _lockout_settings() -> Settings:
    return Settings(policy=PolicySettings(password_max_failures=2, lockout_seconds=60))


def test_lockout_doubles_and_caps_in_seconds() -> None:
    store = FakePasswordStore()
    clock = FakeClock()
    manager = UnlockManager(store, settings=_lockout_settings(), clock=clock)
    manager.set_password(PW)

    expected = 60
    level = 0
    while expected < MAX_LOCKOUT_SECONDS:
        manager.verify("bad-one")
        manager.verify("bad-two")
        assert manager.lockout_remaining() > 0
        assert manager.verify(PW) is False  # correct password refused while locked
        clock.advance(manager.lockout_remaining())
        level += 1
        expected = min(60 * (2**level), MAX_LOCKOUT_SECONDS)
    manager.verify("bad-one")
    manager.verify("bad-two")
    assert manager.lockout_remaining() == MAX_LOCKOUT_SECONDS
    assert manager.lockout_remaining() <= MAX_LOCKOUT_SECONDS


def test_lockout_clears_after_backoff_window() -> None:
    store = FakePasswordStore()
    clock = FakeClock()
    manager = UnlockManager(store, settings=_lockout_settings(), clock=clock)
    manager.set_password(PW)
    manager.verify("bad")
    manager.verify("bad")
    assert manager.lockout_remaining() == 60
    clock.advance(61)
    assert manager.lockout_remaining() == 0
    assert manager.verify(PW) is True


def test_successful_verify_resets_the_backoff_escalation() -> None:
    store = FakePasswordStore()
    clock = FakeClock()
    manager = UnlockManager(store, settings=_lockout_settings(), clock=clock)
    manager.set_password(PW)
    for _ in range(2):
        manager.verify("bad")
    clock.advance(61)
    assert manager.verify(PW) is True
    # a fresh two failures now cost only the base delay again
    manager.verify("bad")
    manager.verify("bad")
    assert manager.lockout_remaining() == 60


# ---------------------------------------------------------------------------
# fail closed
# ---------------------------------------------------------------------------


def test_credential_store_failure_means_not_unlocked() -> None:
    store = FakePasswordStore()
    manager, _store, _ = _manager(store=store)
    manager.set_password(PW)
    store.fail()
    assert manager.has_password() is False
    assert manager.verify(PW) is False
    with pytest.raises(RuntimeError):
        manager.set_password(PW)


def test_stale_hash_is_rehashed_on_successful_verify() -> None:
    from argon2 import PasswordHasher

    store = FakePasswordStore()
    manager, _store, _ = _manager(store=store)
    old = PasswordHasher(time_cost=1, memory_cost=64)
    store.set(HASH_NAME, old.hash(PW))
    before = store.get(HASH_NAME)
    assert manager.verify(PW) is True
    assert store.get(HASH_NAME) != before  # re-hashed with current params
    assert manager.verify(PW) is True


# ---------------------------------------------------------------------------
# the password never reaches logs
# ---------------------------------------------------------------------------


def test_password_must_never_appear_in_logs(caplog: pytest.LogCaptureFixture) -> None:
    manager, _store, _ = _manager()
    manager.set_password(PW)
    with caplog.at_level(logging.DEBUG):
        manager.verify(PW + "wrong")
        manager.verify(PW)
        manager.lock()
    assert PW not in caplog.text
    assert "secret" not in manager.__dict__  # no plaintext kept on the instance
