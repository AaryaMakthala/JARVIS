"""Lazy, profile-scoped managed-browser holder (Stage 8, Slice 4a).

The daemon owns one :class:`~jarvis.browser.context.PlaywrightBrowser` per
JARVIS profile for its whole lifetime.  This holder encapsulates the
*lifecycle* of that session so a second profile (``whatsapp``) can be wired
alongside the general one without duplicating the lazy-start/fail-closed logic
in the daemon (ROADMAP_V2.1 amendment B6: the two profiles never share an
authenticated context).

Construction never touches Chrome or Playwright - it only records the profile
and an optional factory.  The persistent context starts on the first
:meth:`acquire` call and is reused thereafter.  A failed start returns ``None``
(fail closed) and never raises, so a machine without Playwright/Chrome cannot
take down unrelated tasks or the daemon; the caller decides what to do when no
session is available.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

from jarvis.browser.context import PlaywrightBrowser
from jarvis.config import BROWSER_PROFILE_NAMES, Settings

__all__ = ["LazyProfileBrowser"]


class LazyProfileBrowser:
    """Own a managed browser for one JARVIS profile, starting it on first use."""

    def __init__(
        self,
        profile: str,
        settings: Settings | None = None,
        *,
        factory: Callable[[], Any] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Record the profile; never start a browser here.

        ``factory`` is the seam tests use to inject a fake session; it defaults
        to a real :class:`PlaywrightBrowser` bound to this profile.
        """
        if profile not in BROWSER_PROFILE_NAMES:
            raise ValueError(
                f"unknown browser profile {profile!r}; expected one of {BROWSER_PROFILE_NAMES}"
            )
        self._profile = profile
        self._settings = settings or Settings()
        self._factory = factory if factory is not None else self._make_session
        self._logger = logger or logging.getLogger("jarvis.browser.lazy")
        self._session: Any | None = None
        self._lock = threading.Lock()

    def _make_session(self) -> Any:
        """Build (but do not start) the real session for this profile."""
        return PlaywrightBrowser(settings=self._settings, profile=self._profile)

    @property
    def profile(self) -> str:
        """The JARVIS browser profile this holder owns."""
        return self._profile

    @property
    def started(self) -> bool:
        """Whether a session is live (i.e. :meth:`acquire` has started one)."""
        session = self._session
        return session is not None and bool(getattr(session, "started", False))

    def acquire(self) -> Any | None:
        """Return the started session, starting it on first use.

        Idempotent and single-threaded: at most one ``PlaywrightBrowser`` is
        created for this profile, and a failed start is dropped (not cached) so
        a retry fails the same clean way.  ``None`` means "no session available"
        and is the caller's signal to refuse (fail closed).
        """
        with self._lock:
            if self._session is not None:
                return self._session
            try:
                session = self._factory()
                session.start()
            except Exception:  # a failed start fails closed, never raises
                self._logger.warning(
                    "the %s browser profile could not start", self._profile, exc_info=True
                )
                self._session = None
                return None
            self._session = session
            return session

    def close(self) -> None:
        """Close the session if one was started (idempotent, never raises).

        The reference is cleared first so a concurrent :meth:`acquire` can never
        resurrect a closing session.
        """
        with self._lock:
            session, self._session = self._session, None
        if session is None:
            return
        try:
            session.close()
        except Exception:  # shutdown must never raise
            self._logger.debug(
                "closing the %s browser profile failed", self._profile, exc_info=True
            )
