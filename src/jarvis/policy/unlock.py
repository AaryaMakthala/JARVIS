"""Unlock manager: Argon2id password check + timed unlock sessions.

Who guards this / where (docs/03_SECURITY_AND_POLICY.md - invariants 8 and 14):

* only the **Argon2id hash** of the password is stored, in the OS credential
  store (keyring) via :class:`jarvis.secrets.SecretStore` - never a plaintext
  password, and never in a config file, log, checkpoint or LLM prompt;
* password verification happens **outside the graph state**: the CLI/daemon
  calls :meth:`UnlockManager.verify` before resuming a Tier-2 confirmation, so
  the password itself never enters the resume payload (the graph only ever sees
  ``is_unlocked()``);
* a session is "unlocked" for :attr:`UnlockManager.ttl` seconds on a *monotonic
  clock* (immune to clock jumps); ``lock()`` or TTL expiry invalidate every
  pending Tier-2 approval, because the policy gate re-checks
  ``is_unlocked()`` at resume time;
* brute force is throttled by an exponentially growing lockout after
  ``password_max_failures`` bad attempts, capped at 15 minutes.  The correct
  password is refused while locked out (fail closed).

Every ``set``/``verify`` uses the argon2id hash freshly salted by argon2-cffi's
``PasswordHasher``; ``verify()`` re-hashes with new parameters when the stored
hash is flagged by ``check_needs_rehash``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Protocol

from jarvis.config import Settings

logger = logging.getLogger(__name__)

#: Secret name under the keyring service that holds the Argon2id hash.
HASH_NAME = "password_hash"

MIN_PASSWORD_LENGTH = 8

#: Upper bound for lockout backoff (15 minutes).
MAX_LOCKOUT_SECONDS = 15 * 60


class PasswordStore(Protocol):
    """Everything the manager reads to persist the hash (``SecretStore``)."""

    def get(self, name: str) -> str | None: ...
    def set(self, name: str, value: str) -> None: ...
    def has(self, name: str) -> bool: ...


class UnlockManager:
    """Argon2id password management + timed, lockout-guarded unlock sessions.

    All timing is injected (``clock`` defaults to ``time.monotonic``) so tests
    can walk expiry and lockouts deterministically without sleeping.
    """

    def __init__(
        self,
        store: PasswordStore,
        *,
        settings: Settings | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        from argon2 import PasswordHasher

        self._store = store
        self._settings = settings or Settings()
        self._clock = clock or time.monotonic
        self._hasher = PasswordHasher()
        self._ttl = float(self._settings.policy.unlock_ttl_seconds)
        self._max_failures = self._settings.policy.password_max_failures
        self._lockout_base = float(self._settings.policy.lockout_seconds)
        self._unlocked_until = 0.0
        self._fail_count = 0
        self._lockout_level = 0  # doubling multiplier (0 = base, 1 = 2x, ...)
        self._lockout_until = 0.0

    # -- password lifecycle -------------------------------------------------

    def has_password(self) -> bool:
        """Return ``True`` once an Argon2id hash has been stored."""
        try:
            return self._store.has(HASH_NAME)
        except Exception:  # noqa: BLE001 - credential store failure means "no"
            return False

    def set_password(self, password: str) -> None:
        """Hash ``password`` (argon2id) and store it. Never stores plaintext.

        Raises :class:`ValueError` for a too-short password (quietly - no echo
        of the value).
        """
        if len(password) < MIN_PASSWORD_LENGTH:
            raise ValueError(f"JARVIS password must be at least {MIN_PASSWORD_LENGTH} characters")
        hashed = self._hasher.hash(password)
        self._store.set(HASH_NAME, hashed)
        self._reset_failures()

    def verify(self, password: str) -> bool:
        """Return ``True`` only when ``password`` matches and a 0-length lock
        is absent.  A successful verify starts a fresh unlock session."""
        if self._locked_out():
            logger.warning("password verify refused: session is locked out")
            return False
        stored = self._read_hash()
        if stored is None:
            return False
        try:
            self._hasher.verify(stored, password)
        except Exception:  # noqa: BLE001 - wrong password / bad hash -> fail
            self._record_failure()
            return False
        if self._hasher.check_needs_rehash(stored):
            self._store.set(HASH_NAME, self._hasher.hash(password))
        self._reset_failures()
        self._unlocked_until = self._clock() + self._ttl
        logger.info("JARVIS session unlocked for %.0f seconds", self._ttl)
        return True

    # -- session state ------------------------------------------------------

    def unlock(self, password: str) -> bool:
        """Alias of :meth:`verify` used by the CLI/daemon."""
        return self.verify(password)

    def is_unlocked(self) -> bool:
        """``True`` while the TTL window from the last successful verify holds."""
        if self._unlocked_until == 0.0:
            return False
        if self._clock() >= self._unlocked_until:
            self._unlocked_until = 0.0  # lazy expiry
            return False
        return True

    def lock(self) -> None:
        """Invalidate the session immediately (also invalidates pending
        Tier-2 approvals, since the policy gate re-checks on resume)."""
        self._unlocked_until = 0.0
        logger.info("JARVIS session locked")

    def remaining_seconds(self) -> float:
        """Seconds left in the current unlock window (0 when locked)."""
        if not self.is_unlocked():
            return 0.0
        return max(0.0, self._unlocked_until - self._clock())

    def lockout_remaining(self) -> float:
        """Seconds left in a lockout (0 when not locked out)."""
        if self._clock() < self._lockout_until:
            return self._lockout_until - self._clock()
        return 0.0

    # -- internals ----------------------------------------------------------

    def _now(self) -> float:
        return self._clock()

    def _read_hash(self) -> str | None:
        try:
            return self._store.get(HASH_NAME)
        except Exception:  # noqa: BLE001 - credential store failure -> fail closed
            logger.warning("could not read the password hash from the credential store")
            return None

    def _locked_out(self) -> bool:
        return self.lockout_remaining() > 0

    def _record_failure(self) -> None:
        self._fail_count += 1
        if self._fail_count < self._max_failures:
            return
        delay = min(self._lockout_base * (2**self._lockout_level), MAX_LOCKOUT_SECONDS)
        self._lockout_until = self._clock() + delay
        self._lockout_level += 1
        self._fail_count = 0
        logger.warning("too many wrong passwords: JARVIS locked for %.0f seconds", delay)

    def _reset_failures(self) -> None:
        self._fail_count = 0
        self._lockout_level = 0
