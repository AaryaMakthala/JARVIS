"""JARVIS-owned managed browser session (Stage 7, Slice 6).

A single headed persistent Chrome context owned by JARVIS, launched with
``channel="chrome"`` and driven through Playwright.  It always runs against
the JARVIS-owned ``browser_profile/general`` directory - never the owner's
normal profile and never the WhatsApp profile (docs/03 B6).

Design rules
------------
* Playwright is imported *lazily* inside :meth:`PlaywrightBrowser.start`;
  merely importing ``jarvis.browser`` must work where Playwright is missing
  or the platform cannot launch Chrome.
* Nothing here starts Chrome at import, registration or construction time.
* Concurrent tasks that share a profile are serialized by a per-profile lock
  held for the whole session (acquired in ``start()``, released in
  ``close()``).
* References produced by :meth:`PlaywrightBrowser.read` are transient,
  page-scoped identity handles (``e1..eN``).  They are re-located and
  re-validated - accessible role, accessible name, origin and match count -
  immediately before every click/type, and are never stored in agent state.
* Redirects to an origin that is neither the requested one nor in the
  approved domain allow-list lock further interaction until a successful,
  authorized ``goto`` clears the block.
* Sensitive data (cookies, credentials, password values, typed text) is
  never logged, returned, fingerprinted or stored.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Protocol

from jarvis.browser.find import match_identity, match_indexes
from jarvis.config import BROWSER_PROFILE_NAMES, Settings, browser_profile_dir
from jarvis.platform_guard import is_windows
from jarvis.policy.rules import domain_allowed, url_origin
from jarvis.tools.web import is_allowed_http_url

__all__ = [
    "MANAGED_BROWSER_CHANNEL",
    "BrowserError",
    "BrowserSession",
    "ElementRef",
    "Observation",
    "PlaywrightBrowser",
    "profile_lock",
]

#: Chromium channel the managed browser is launched with.  Also the app token
#: the policy engine compares against ``type_text``'s ``target_app``: generic
#: typing into the managed browser must go through ``browser_type`` (Slice 9).
MANAGED_BROWSER_CHANNEL = "chrome"

#: Hard cap on observed interactive elements reported by one ``read``.
_MAX_OBSERVED_ELEMENTS = 80
#: Hard cap on the elements scanned while *finding* a match (a query may
#: target an element that lies past the ``read`` limit).
_MAX_FIND_ELEMENTS = 300
#: Hard cap on a single accessible name returned to the caller.
_MAX_ELEMENT_NAME_CHARS = 120
_NAV_TIMEOUT_MS = 30_000
_CLICK_TIMEOUT_MS = 10_000
_TYPE_TIMEOUT_MS = 10_000

#: Compact locator covering common interactive controls plus explicit roles.
_INTERACTIVE_SELECTOR = "button, a[href], input, textarea, select, [role], [tabindex]"

#: tag -> default accessibility role when no explicit ``role`` attribute.
_TAG_ROLES = {
    "button": "button",
    "a": "link",
    "input": "textbox",
    "textarea": "textbox",
    "select": "combobox",
}

_PASSWORD_ROLES = frozenset({"password", "passwordfield", "passwordtextbox"})


class BrowserError(RuntimeError):
    """Managed-browser failure raised by the driver.

    Tools wrap every ``BrowserError`` into :class:`ToolResult(ok=False, ...)`
    per docs/04 §1; it never propagates to the graph.
    """


@dataclass(frozen=True)
class ElementRef:
    """Public metadata for one observed interactive element.

    Only identity metadata is exposed: the transient handle (``ref``), the
    accessible role/name and the page origin.  Never a value, never typed
    text, never credentials.  ``input_type``/``is_password`` are included so
    policy can refuse credential fields before any interaction.
    """

    ref: str
    role: str
    name: str
    origin: str
    input_type: str = ""
    is_password: bool = False

    def as_dict(self) -> dict[str, Any]:
        """A JSON-safe dict for the caller (no values, no text)."""
        return {
            "ref": self.ref,
            "role": self.role,
            "name": self.name,
            "origin": self.origin,
            "input_type": self.input_type,
            "is_password": self.is_password,
        }


@dataclass(frozen=True)
class Observation:
    """One bounded, structured accessibility observation of the page."""

    url: str
    origin: str
    title: str
    elements: tuple[ElementRef, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "origin": self.origin,
            "title": self.title,
            "elements": [e.as_dict() for e in self.elements],
        }


class BrowserSession(Protocol):
    """The minimal session contract the browser tools depend on.

    Implemented by :class:`PlaywrightBrowser` (live) and by test fakes.
    """

    def goto(self, url: str) -> dict[str, Any]:
        """Navigate to an absolute http(s) URL; return ``{url, origin, title}``."""

    def read(self, max_elements: int = _MAX_OBSERVED_ELEMENTS) -> Observation:
        """Observe the current page (bounded); refs are page-scoped handles."""

    def find(
        self, query: str, *, role: str = "", limit: int = _MAX_FIND_ELEMENTS
    ) -> list[ElementRef]:
        """Return the observed elements whose accessible name matches ``query``."""

    def open_match(
        self,
        query: str,
        *,
        role: str = "",
        index: int | None = None,
        expect_identity: str | None = None,
    ) -> dict[str, Any]:
        """Find the matching element and click it; refuse when ambiguous."""

    def resolve(self, ref: str) -> ElementRef | None:
        """Return the observed identity for ``ref`` (transient), else ``None``."""

    def click(self, ref: str) -> dict[str, Any]:
        """Re-validate then click a ref; return ``{ref, role, name, origin}``."""

    def type_text(self, ref: str, text: str) -> dict[str, Any]:
        """Re-validate then type confined to the managed page; never records text."""

    def close(self) -> None:
        """Release the profile lock, close context and Playwright."""


@dataclass
class _ObservedElement:
    """Internal snapshot used for later identity re-validation."""

    ref: str
    role: str
    name: str
    origin: str
    input_type: str
    is_password: bool
    nth: int
    total: int


@dataclass
class _Candidate:
    """One observed interactive element, before it is given a transient ref."""

    loc: Any
    role: str
    name: str
    input_type: str
    is_password: bool


_PROFILE_LOCKS: dict[str, threading.Lock] = {}
_PROFILE_LOCKS_GUARD = threading.Lock()


def profile_lock(profile: str) -> threading.Lock:
    """Return the process-wide lock for ``profile`` (created on demand).

    Held for the whole session so concurrent tasks sharing a profile are
    serialized; two different profiles run in parallel.
    """
    with _PROFILE_LOCKS_GUARD:
        lock = _PROFILE_LOCKS.get(profile)
        if lock is None:
            lock = threading.Lock()
            _PROFILE_LOCKS[profile] = lock
        return lock


def _norm_text(value: Any, limit: int = _MAX_ELEMENT_NAME_CHARS) -> str:
    """Collapse whitespace and bound the length of a page-derived name."""
    return " ".join(str(value or "").split())[:limit]


def _import_playwright() -> Any:
    """Import ``playwright.sync_api`` lazily; raise a clean :class:`BrowserError`."""
    try:
        from playwright.sync_api import sync_playwright  # type: ignore[import-not-found]
    except ImportError as exc:
        raise BrowserError(
            "Playwright is not installed; run `pip install jarvis[browser]` "
            "so the managed browser can start"
        ) from exc
    return sync_playwright


class PlaywrightBrowser:
    """Live managed browser backed by Playwright (started on demand only)."""

    def __init__(self, settings: Settings | None = None, profile: str = "general") -> None:
        if profile not in BROWSER_PROFILE_NAMES:
            raise ValueError(
                f"unknown browser profile {profile!r}; expected one of {BROWSER_PROFILE_NAMES}"
            )
        self._settings = settings or Settings()
        self._profile = profile
        self._pw: Any | None = None
        self._context: Any | None = None
        self._page: Any | None = None
        self._snapshots: dict[str, _ObservedElement] = {}
        self._redirect_blocked = False
        self._next_ref = 1
        self._profile_lease: threading.Lock | None = None
        self._started = False

    # ── lifecycle ──────────────────────────────────────────────────────────

    @property
    def started(self) -> bool:
        """Whether the persistent context is currently attached (``start()`` ok)."""
        return self._started

    def start(self) -> None:
        """Launch the persistent Chrome context (blocking, never at import time)."""
        if self._started:
            return
        if not is_windows():
            raise BrowserError("the managed Chrome browser is only available on Windows")
        lease = profile_lock(self._profile)
        lease.acquire()
        self._profile_lease = lease
        try:
            sync_playwright = _import_playwright()
            playwright = sync_playwright().start()
            user_data_dir = browser_profile_dir(self._profile)
            context = playwright.chromium.launch_persistent_context(
                user_data_dir=str(user_data_dir),
                channel=MANAGED_BROWSER_CHANNEL,
                headless=False,
            )
            self._pw = playwright
            self._context = context
            self._page = context.pages[0] if context.pages else context.new_page()
            self._snapshots.clear()
            self._redirect_blocked = False
            self._started = True
        except BrowserError:
            self._release_lease()
            raise
        except Exception as exc:
            self._release_lease()
            raise BrowserError(
                f"failed to start the managed browser ({self._profile} profile): {exc}"
            ) from exc

    def close(self) -> None:
        """Close the context and Playwright, release the profile lock."""
        started = self._started
        self._started = False
        self._snapshots.clear()
        context, self._context = self._context, None
        pw, self._pw = self._pw, None
        self._page = None
        self._redirect_blocked = False
        if context is not None:
            try:
                context.close()
            except Exception:  # noqa: BLE001, S110 - shutdown must never raise
                pass
        if pw is not None:
            try:
                pw.stop()
            except Exception:  # noqa: BLE001, S110 - shutdown must never raise
                pass
        if started:
            self._release_lease()

    def _release_lease(self) -> None:
        lease, self._profile_lease = self._profile_lease, None
        if lease is not None:
            try:
                lease.release()
            except RuntimeError:
                pass

    def _require_page(self) -> Any:
        if not self._started or self._page is None:
            raise BrowserError("the managed browser session is not started")
        return self._page

    # ── tools ──────────────────────────────────────────────────────────────

    def goto(self, url: str) -> dict[str, Any]:
        """Navigate and fail closed on unauthorized redirects."""
        page = self._require_page()
        if not is_allowed_http_url(url) or url_origin(url) is None:
            raise BrowserError("refusing a URL that is not absolute http(s) without userinfo")
        intents = url_origin(url)
        assert intents is not None  # guaranteed by the check above
        page.goto(url, wait_until="domcontentloaded", timeout=_NAV_TIMEOUT_MS)
        self._snapshots.clear()
        final_url = str(page.url)
        final_origin = url_origin(final_url)
        allowed = self._settings.browser.allowed_domains
        # An empty allow-list pre-approves nothing (fail closed): any redirect
        # off the requested origin must land on an approved host or the task is
        # stopped before any subsequent interaction.
        if (
            final_origin is not None
            and final_origin != intents
            and not domain_allowed(final_url, allowed)
        ):
            self._redirect_blocked = True
            raise BrowserError(f"navigation redirected to unauthorized origin {final_origin!r}")
        self._redirect_blocked = False
        return {
            "url": final_url,
            "origin": final_origin or "",
            "title": _norm_text(self._safe_title(page), 200),
        }

    def read(self, max_elements: int = _MAX_OBSERVED_ELEMENTS) -> Observation:
        """Take a bounded accessibility observation of the current page."""
        page = self._require_page()
        if self._redirect_blocked:
            raise BrowserError(
                "interactions are blocked after an unauthorized redirect; "
                "authorize a valid navigation first"
            )
        url = str(page.url)
        origin = url_origin(url) or ""

        max_elems = max(1, min(int(max_elements), _MAX_OBSERVED_ELEMENTS))
        candidates, groups = self._collect_candidates(page, max_elems)
        elements = self._register(candidates, groups, origin)
        return Observation(
            url=url,
            origin=origin,
            title=_norm_text(self._safe_title(page), 200),
            elements=tuple(elements),
        )

    # ── query-driven finding (Stage 8.6) ──────────────────────────────────

    def find(
        self, query: str, *, role: str = "", limit: int = _MAX_FIND_ELEMENTS
    ) -> list[ElementRef]:
        """Return the observed elements whose accessible name matches ``query``.

        The match order and the rules are :func:`jarvis.browser.find.match_indexes`
        (exact, then prefix, then contains).  The returned refs are transient
        page-scoped handles, exactly like :meth:`read`; a fresh observation
        invalidates the refs of the previous one.
        """
        page = self._require_page()
        if self._redirect_blocked:
            raise BrowserError(
                "interactions are blocked after an unauthorized redirect; "
                "authorize a valid navigation first"
            )
        origin = url_origin(str(page.url)) or ""
        candidates, groups = self._collect_candidates(page, _MAX_FIND_ELEMENTS)
        pairs = [(item.role, item.name) for item in candidates]
        indexes = match_indexes(pairs, query, role=role)
        elements = self._register(candidates, groups, origin)
        matched = [elements[index] for index in indexes[: max(1, int(limit))]]
        return matched

    def open_match(
        self,
        query: str,
        *,
        role: str = "",
        index: int | None = None,
        expect_identity: str | None = None,
    ) -> dict[str, Any]:
        """Find the matching element and activate it (click it).

        The element is located by the same deterministic rules as :meth:`find`
        and is then clicked through the ordinary re-validation path (role,
        accessible name, origin and match count are re-checked immediately
        before the click; there is no coordinate fallback).

        Refuses when nothing matches and when several elements match and the
        caller did not say which one it meant - an ambiguous "the result" is
        never guessed (fail closed).

        ``expect_identity`` is the stable identity of the element the user
        confirmed.  When it is given and the freshly observed match no longer
        has that identity (the page changed after the confirmation), the click
        is **not** performed and an error is raised.
        """
        page = self._require_page()
        if self._redirect_blocked:
            raise BrowserError("interactions are blocked after an unauthorized redirect")
        origin = url_origin(str(page.url)) or ""
        candidates, groups = self._collect_candidates(page, _MAX_FIND_ELEMENTS)
        pairs = [(item.role, item.name) for item in candidates]
        indexes = match_indexes(pairs, query, role=role)
        if not indexes:
            raise BrowserError(f"no element on the page matches {query!r}")
        if index is None and len(indexes) > 1:
            names = ", ".join(repr(pairs[i][1]) for i in indexes[:5])
            raise BrowserError(
                f"{len(indexes)} elements match {query!r} ({names}); "
                "say which one (index) instead of guessing"
            )
        position = 1 if index is None else int(index)
        if position < 1 or position > len(indexes):
            raise BrowserError(
                f"index {position} is out of range: {len(indexes)} element(s) match {query!r}"
            )
        chosen_index = indexes[position - 1]
        if expect_identity:
            identity = match_identity(pairs[chosen_index][0], pairs[chosen_index][1], position)
            if identity != expect_identity:
                raise BrowserError(
                    "the page element changed after you confirmed it; nothing was opened"
                )
        elements = self._register(candidates, groups, origin)
        chosen = elements[chosen_index]
        data = self.click(chosen.ref)
        data["position"] = position
        data["matches"] = len(indexes)
        return data

    def _collect_candidates(
        self, page: Any, max_elements: int
    ) -> tuple[list[_Candidate], dict[tuple[str, str], int]]:
        """Collect the visible interactive elements of ``page`` (bounded)."""
        limit = max(1, min(int(max_elements), _MAX_FIND_ELEMENTS))
        candidates: list[_Candidate] = []
        groups: dict[tuple[str, str], int] = {}
        for loc in page.locator(_INTERACTIVE_SELECTOR).all():
            if not loc.is_visible():
                continue
            role = self._live_role(loc)
            name = _norm_text(self._live_name(loc))
            if not name:
                continue  # unnamed controls cannot be safely re-validated
            input_type = self._live_input_type(loc)
            is_password = input_type == "password" or role in _PASSWORD_ROLES
            group = (role, name)
            groups[group] = groups.get(group, 0) + 1
            candidates.append(
                _Candidate(
                    loc=loc,
                    role=role,
                    name=name,
                    input_type=input_type,
                    is_password=is_password,
                )
            )
            if len(candidates) >= limit:
                break
        return candidates, groups

    def _register(
        self,
        candidates: list[_Candidate],
        groups: dict[tuple[str, str], int],
        origin: str,
    ) -> list[ElementRef]:
        """Assign fresh transient refs to the candidates and record them.

        Every observation replaces the previous one: the refs are page-scoped
        handles, so a stale ref must not resolve to a new element.
        """
        self._snapshots.clear()
        seen: dict[tuple[str, str], int] = {}
        elements: list[ElementRef] = []
        for item in candidates:
            group = (item.role, item.name)
            nth = seen.get(group, 0)
            seen[group] = nth + 1
            ref = f"e{self._next_ref}"
            self._next_ref += 1
            self._snapshots[ref] = _ObservedElement(
                ref=ref,
                role=item.role,
                name=item.name,
                origin=origin,
                input_type=item.input_type,
                is_password=item.is_password,
                nth=nth,
                total=groups[group],
            )
            elements.append(
                ElementRef(
                    ref=ref,
                    role=item.role,
                    name=item.name,
                    origin=origin,
                    input_type=item.input_type,
                    is_password=item.is_password,
                )
            )
        return elements

    def resolve(self, ref: str) -> ElementRef | None:
        """Return the transient observed identity for ``ref``, else ``None``."""
        snap = self._snapshots.get(ref)
        if snap is None:
            return None
        return ElementRef(
            ref=snap.ref,
            role=snap.role,
            name=snap.name,
            origin=snap.origin,
            input_type=snap.input_type,
            is_password=snap.is_password,
        )

    def click(self, ref: str) -> dict[str, Any]:
        """Re-validate then click ``ref``; never falls back to coordinates."""
        page = self._require_page()
        if self._redirect_blocked:
            raise BrowserError("interactions are blocked after an unauthorized redirect")
        snap, loc, live = self._relocate(page, ref)
        loc.click(timeout=_CLICK_TIMEOUT_MS)
        return {
            "ref": snap.ref,
            "role": live.role,
            "name": live.name,
            "origin": snap.origin,
        }

    def type_text(self, ref: str, text: str) -> dict[str, Any]:
        """Re-validate then type into ``ref``, confined to the managed page."""
        page = self._require_page()
        if self._redirect_blocked:
            raise BrowserError("interactions are blocked after an unauthorized redirect")
        if not text or len(text) > 4096:
            raise BrowserError("refusing to type an empty or oversized value")
        snap, loc, live = self._relocate(page, ref)
        if live.is_password or snap.input_type == "password":
            raise BrowserError("refusing to type into a credential field")
        tag = self._live_tag(loc)
        if tag not in ("input", "textarea"):
            raise BrowserError("target is not a text input; trying to type into it is refused")
        loc.fill(text, timeout=_TYPE_TIMEOUT_MS)
        # Deliberately no text anywhere in the return payload or any log line.
        return {
            "ref": snap.ref,
            "role": live.role,
            "name": live.name,
            "origin": snap.origin,
            "characters": len(text),
        }

    # ── WhatsApp (Stage 8, Slice 8.4b) ─────────────────────────────────────

    def whatsapp_page(self) -> Any:
        """Return a narrow WhatsApp page handle for the ``whatsapp`` profile.

        The returned object is a
        :class:`jarvis.browser.whatsapp_page.PlaywrightWhatsAppPage`: it
        exposes only the read/act methods the WhatsApp driver needs and no
        generic DOM handle.

        This accessor is deliberately **not** part of :class:`BrowserSession`,
        so the ``browser_*`` tools and the planner can never reach WhatsApp
        through it, and it refuses to hand out a page for any other profile
        (the two profiles never share an authenticated context, B6).
        """
        if self._profile != "whatsapp":
            raise BrowserError(
                "the WhatsApp page accessor is only available on the whatsapp profile"
            )
        page = self._require_page()
        from jarvis.browser.whatsapp_page import PlaywrightWhatsAppPage

        return PlaywrightWhatsAppPage(page)

    # ── identity re-validation ─────────────────────────────────────────────

    def _relocate(self, page: Any, ref: str) -> tuple[_ObservedElement, Any, ElementRef]:
        """Locate the live element and re-validate its identity (fail closed)."""
        snap = self._snapshots.get(ref)
        if snap is None:
            raise BrowserError(f"unknown or stale element reference {ref!r}")
        origin = url_origin(str(page.url))
        if snap.origin and origin != snap.origin:
            raise BrowserError("the page navigated since the element was observed")
        base = page.get_by_role(role=snap.role, name=snap.name, exact=True)
        count = base.count()
        if count == 0:
            raise BrowserError(f"element {ref!r} is no longer present (stale)")
        if count != snap.total:
            raise BrowserError(f"element {ref!r} is ambiguous or changed since it was observed")
        loc = base.nth(snap.nth)
        if not loc.is_visible():
            raise BrowserError(f"element {ref!r} is no longer visible")
        live_role = self._live_role(loc)
        live_name = _norm_text(self._live_name(loc))
        live_input_type = self._live_input_type(loc)
        if live_role != snap.role or live_name != snap.name:
            raise BrowserError(f"element {ref!r} identity changed since it was observed")
        live = ElementRef(
            ref=snap.ref,
            role=live_role,
            name=live_name,
            origin=origin or snap.origin,
            input_type=live_input_type,
            is_password=live_input_type == "password" or live_role in _PASSWORD_ROLES,
        )
        return snap, loc, live

    # ── live DOM readers (best-effort, Playwright-only) ────────────────────

    @staticmethod
    def _live_tag(loc: Any) -> str:
        try:
            return str(
                loc.element_handle().evaluate("el => el.tagName.toLowerCase()") or ""
            ).lower()
        except Exception:  # noqa: BLE001 - best-effort observation
            return ""

    @classmethod
    def _live_role(cls, loc: Any) -> str:
        try:
            explicit = (loc.get_attribute("role") or "").strip().lower()
            if explicit:
                return explicit
            return _TAG_ROLES.get(cls._live_tag(loc), "unknown")
        except Exception:  # noqa: BLE001 - best-effort observation
            return "unknown"

    @staticmethod
    def _live_input_type(loc: Any) -> str:
        attrib = loc.get_attribute("type")
        return (attrib or "").strip().lower() if attrib else ""

    @staticmethod
    def _live_name(loc: Any) -> str:
        try:
            aria = loc.get_attribute("aria-label")
            if aria:
                return aria
            return loc.inner_text()
        except Exception:  # noqa: BLE001 - best-effort observation
            return ""

    @staticmethod
    def _safe_title(page: Any) -> str:
        try:
            return str(page.title() or "")
        except Exception:  # noqa: BLE001 - best-effort observation
            return ""
