"""Bounded, in-memory result sets for page result selection (Stage 8, P2).

``browser_find`` lists the elements of the current page that match a query.
This module remembers that list as **one coherent result set** so a follow-up
selection - in the same task or a later turn - can say *which* entry it means:

* ``"open the first link"`` / ``"open top one"`` -> position 1 of the set;
* ``"open the official Python documentation result"`` -> a unique title match;
* ``"open Java Basics"`` -> the entry whose title matches best.

Why a separate store
--------------------
The page itself is the only source of the *content*, but the page is mutable:
it reloads, re-orders and re-renders.  A position is therefore only meaningful
inside the exact list it came from, observed on the exact page it came from.
The store keeps at most one set per browser session, binds it to the page URL
it was observed on, ages it out after a bounded TTL and drops it as soon as
JARVIS navigates.  A new observation *supersedes* the previous set, so an old
index can never be silently applied to a new list.

Design guarantees
-----------------
* **In-memory only.**  The store lives on ``AppContext`` and is never written
  to ``AgentState`` or the checkpoint store (the serde allowlist is unchanged).
* **Bounded.**  At most :data:`MAX_RESULT_ITEMS` entries per set and at most
  :data:`MAX_RESULT_SETS` sessions; the oldest session is dropped first.
* **Metadata only.**  Each entry carries the position in the observed order,
  its accessible name (title), its role and its validated http(s) destination.
  No DOM, no page text, no typed text, no credentials.
* **Tainted.**  Titles and URLs are page-derived and therefore untrusted: they
  are only ever used as *search keys* and as quoted data in a message, never as
  instructions or as a new objective (docs/03 §8).
* **Never a fallback.**  A missing, expired, superseded or navigated-away set
  is a refusal - never "the most recent URL", never "the newest item".
* **Not an authorization.**  Selecting an entry says *which element* the user
  meant; the ordinary policy/gate/binding path still authorizes the action.
"""

from __future__ import annotations

import re
import threading
import time
import urllib.parse
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Literal

from jarvis.tools.web import is_allowed_http_url

__all__ = [
    "DEFAULT_RESULT_TTL_SECONDS",
    "MAX_RESULT_ITEMS",
    "MAX_RESULT_SETS",
    "BrowserResultSet",
    "BrowserResultStore",
    "ResultItem",
    "Selection",
    "parse_ordinal",
    "rank_title_matches",
]

#: Hard cap on the entries one result set may hold (the same bound as one
#: ``browser_find`` listing, so the user can select anything they were shown).
MAX_RESULT_ITEMS = 20

#: Hard cap on how many browser sessions are remembered at once.
MAX_RESULT_SETS = 4

#: How long a result set stays selectable.  Matches the session artifact TTL
#: (``[session] artifact_ttl_seconds``) - short on purpose.
DEFAULT_RESULT_TTL_SECONDS = 600.0

#: Bound on a stored title (a page-derived accessible name).
MAX_TITLE_CHARS = 200

#: Bound on a stored destination URL.
MAX_URL_CHARS = 2048

#: Words that carry no discriminating information in a title selection.  They
#: are dropped from the query before scoring, which can only *widen* the
#: candidate pool; the result is therefore still checked for ambiguity.
_TITLE_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "at",
        "by",
        "for",
        "in",
        "is",
        "it",
        "link",
        "links",
        "of",
        "official",
        "on",
        "one",
        "or",
        "page",
        "result",
        "results",
        "site",
        "the",
        "to",
        "website",
        "with",
    }
)

#: Minimum share of the user's title words that must appear in a title before
#: it counts as a candidate.  A single-word query must match that word.
_MIN_TITLE_COVERAGE = 0.6

_WORD_ORDINALS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}

_ORDINAL_PREFIXES = frozenset({"#", "the", "top", "number", "no", "no."})
_ORDINAL_SUFFIXES = frozenset(
    {
        "choice",
        "entry",
        "item",
        "items",
        "link",
        "links",
        "match",
        "one",
        "option",
        "page",
        "result",
        "results",
        "site",
    }
)
_DIGIT_ORDINAL_RE = re.compile(r"^(\d{1,2})(?:st|nd|rd|th)?$")

SelectionKind = Literal[
    "resolved",
    "no_result_set",
    "not_found",
    "invalid_position",
    "expired",
    "stale",
    "superseded",
    "ambiguous",
    "no_selection",
]


def _normalise(value: str) -> str:
    """Case-fold a title or query and collapse its whitespace."""
    return " ".join(str(value or "").split()).casefold()


def _tokens(value: str) -> tuple[str, ...]:
    """Split a title/query into comparable words (letters and digits only)."""
    cleaned = re.sub(r"[^\w\s]+", " ", str(value or "").casefold())
    return tuple(word for word in cleaned.split() if word)


def parse_ordinal(text: str) -> int | None:
    """Return the 1-based position an ordinal phrase means, else ``None``.

    Understands ``"1"``, ``"#1"``, ``"1st"``, ``"first"``, ``"the first
    result"``, ``"top one"`` and ``"number two"``.  Anything that is not
    *only* an ordinal (a real title such as ``"Python 1"``) returns ``None``,
    so a genuine title is never silently read as a position.
    """
    words = str(text or "").casefold().replace("#", " # ").split()
    if not words:
        return None
    while words and words[0] in _ORDINAL_PREFIXES:
        words.pop(0)
    if not words:
        return None
    while len(words) > 1 and words[-1] in _ORDINAL_SUFFIXES:
        words.pop()
    if len(words) != 1:
        return None
    token = words[0].strip(".,;:!?")
    if not token:
        return None
    if token in _WORD_ORDINALS:
        return _WORD_ORDINALS[token]
    match = _DIGIT_ORDINAL_RE.match(token)
    if match is None:
        return None
    return int(match.group(1))


def page_key(url: str) -> str:
    """A comparable key for a page URL (fragment dropped, host case-folded)."""
    try:
        parts = urllib.parse.urlsplit(str(url or "").strip())
    except ValueError:
        return ""
    if not parts.scheme or not parts.netloc:
        return ""
    return urllib.parse.urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), parts.path, parts.query, "")
    )


@dataclass(frozen=True)
class ResultItem:
    """One selectable entry of a result set (metadata only, page-derived)."""

    position: int
    title: str
    url: str = ""
    role: str = ""

    @property
    def is_link(self) -> bool:
        """Whether this entry is a link (and therefore needs a destination)."""
        return (self.role or "").strip().lower() == "link"


@dataclass(frozen=True)
class BrowserResultSet:
    """One immutable observation of a page's result list."""

    id: str
    session_id: str
    page_url: str
    created_at: float
    expires_at: float
    items: tuple[ResultItem, ...]

    def age_seconds(self, now: float) -> float:
        """Seconds since the set was recorded (never negative)."""
        return max(0.0, float(now) - self.created_at)


@dataclass(frozen=True)
class Selection:
    """Outcome of resolving a user's selection against the current result set."""

    kind: SelectionKind
    item: ResultItem | None = None
    candidates: tuple[ResultItem, ...] = ()
    result_set: BrowserResultSet | None = None
    message: str = ""

    @property
    def ok(self) -> bool:
        """Whether exactly one entry was selected."""
        return self.kind == "resolved" and self.item is not None


def _first_line(message: str) -> str:
    return " ".join(str(message or "").split())


def _quoted(items: Sequence[ResultItem], limit: int = 5) -> str:
    """Render up to ``limit`` titles as quoted, bounded data for a message."""
    shown = ", ".join(repr(item.title) for item in items[:limit])
    if len(items) > limit:
        shown += f", and {len(items) - limit} more"
    return shown


def rank_title_matches(items: Iterable[ResultItem], query: str) -> list[ResultItem]:
    """Return the entries whose title best matches ``query``, in set order.

    Exact titles win outright.  Otherwise every entry that contains at least
    :data:`_MIN_TITLE_COVERAGE` of the user's (stopword-free) words is scored by
    that share, and only the best score is kept.  The caller must treat more
    than one result as ambiguous - this function never picks a winner.
    """
    wanted = _normalise(query)
    if not wanted:
        return []
    ordered = list(items)
    exact = [item for item in ordered if _normalise(item.title) == wanted]
    if exact:
        return exact
    words = tuple(token for token in _tokens(query) if token not in _TITLE_STOPWORDS)
    if not words:
        words = _tokens(query)
    if not words:
        return []
    scored: list[tuple[float, ResultItem]] = []
    for item in ordered:
        title_words = set(_tokens(item.title))
        matched = sum(1 for word in words if word in title_words)
        coverage = matched / len(words)
        if matched and coverage >= _MIN_TITLE_COVERAGE:
            scored.append((coverage, item))
    if not scored:
        return []
    best = max(score for score, _ in scored)
    return [item for score, item in scored if score == best]


class BrowserResultStore:
    """Bounded, thread-safe, in-memory result sets, one per browser session."""

    def __init__(
        self,
        *,
        max_items: int = MAX_RESULT_ITEMS,
        ttl_seconds: float = DEFAULT_RESULT_TTL_SECONDS,
        max_sets: int = MAX_RESULT_SETS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._lock = threading.Lock()
        self._max_items = max(1, int(max_items))
        self._ttl = max(1.0, float(ttl_seconds))
        self._max_sets = max(1, int(max_sets))
        self._clock = clock
        self._sets: dict[str, BrowserResultSet] = {}

    @property
    def ttl_seconds(self) -> float:
        """How long a result set stays selectable, in seconds."""
        return self._ttl

    @property
    def max_items(self) -> int:
        """Hard cap on the entries one set may hold."""
        return self._max_items

    def record(
        self,
        session_id: str,
        items: Iterable[ResultItem],
        *,
        page_url: str = "",
        now: float | None = None,
    ) -> BrowserResultSet | None:
        """Store ``items`` as the session's one current result set.

        Every recorded entry is re-numbered from 1 in the order given, titles
        and URLs are bounded, and a destination URL is kept only when it is a
        safe absolute http(s) URL (otherwise the entry keeps no destination and
        the positional open will refuse it).  An empty list drops the previous
        set - a new observation with no matches supersedes the old list rather
        than leaving stale indices behind.  ``None`` means "no set stored".
        """
        moment = self._clock() if now is None else float(now)
        kept: list[ResultItem] = []
        for raw in list(items)[: self._max_items]:
            title = _first_line(raw.title)[:MAX_TITLE_CHARS]
            if not title:
                continue
            url = str(raw.url or "").strip()[:MAX_URL_CHARS]
            if url and not is_allowed_http_url(url):
                url = ""
            kept.append(
                replace(
                    raw,
                    position=len(kept) + 1,
                    title=title,
                    url=url,
                    role=_first_line(raw.role)[:32].lower(),
                )
            )
        scope = str(session_id or "").strip()
        with self._lock:
            if not kept:
                self._sets.pop(scope, None)
                return None
            entry = BrowserResultSet(
                id=uuid.uuid4().hex,
                session_id=scope,
                page_url=str(page_url or "").strip()[:MAX_URL_CHARS],
                created_at=moment,
                expires_at=moment + self._ttl,
                items=tuple(kept),
            )
            self._sets[entry.session_id] = entry
            self._prune_sessions()
            return entry

    def current(
        self, session_id: str, *, page_url: str = "", now: float | None = None
    ) -> BrowserResultSet | None:
        """Return the session's valid current result set, or ``None``.

        Applies the TTL and the page binding: an expired set, or a set observed
        on a different page than the one the browser is on now, is dropped and
        reported as absent (fail closed).
        """
        selection = self.select(session_id, position=None, page_url=page_url, now=now)
        if selection.kind in {"no_result_set", "expired", "stale"}:
            return None
        return selection.result_set

    def invalidate(self, session_id: str) -> None:
        """Drop the session's result set (e.g. after a navigation)."""
        with self._lock:
            self._sets.pop(str(session_id or "").strip(), None)

    def select(
        self,
        session_id: str,
        *,
        position: int | None = None,
        query: str = "",
        role: str = "",
        set_id: str = "",
        page_url: str = "",
        now: float | None = None,
    ) -> Selection:
        """Resolve one entry of the session's current result set.

        ``position`` wins.  Otherwise an ordinal ``query`` (``"first"``,
        ``"2"``, ``"top one"``) is read as a position, and any other query is a
        title selection.  More than one title candidate (or an exact title
        shared by several entries) is reported as ``ambiguous`` - never guessed.
        """
        moment = self._clock() if now is None else float(now)
        scope = str(session_id or "").strip()
        with self._lock:
            entry = self._sets.get(scope)
            if entry is None:
                return Selection(
                    kind="no_result_set",
                    message=(
                        "there is no current result list for this browser; "
                        "find or search for something first"
                    ),
                )
            if set_id and str(set_id) != entry.id:
                return Selection(
                    kind="superseded",
                    result_set=entry,
                    message="that result list has been replaced by a newer one; ask again",
                )
            if moment >= entry.expires_at:
                del self._sets[scope]
                return Selection(
                    kind="expired",
                    message="the result list has expired; search again",
                )
            if page_url and entry.page_url and page_key(page_url) != page_key(entry.page_url):
                del self._sets[scope]
                return Selection(
                    kind="stale",
                    message="the page changed since the result list was listed; search again",
                )
            items = entry.items

        # Filtering by role is only meaningful for a title query: a position
        # always means "this entry of the list the user was shown".
        wanted_position = position
        if wanted_position is None:
            wanted_position = parse_ordinal(query)
        if wanted_position is not None:
            if 1 <= int(wanted_position) <= len(items):
                return Selection(
                    kind="resolved", item=items[int(wanted_position) - 1], result_set=entry
                )
            return Selection(
                kind="invalid_position",
                result_set=entry,
                message=(
                    f"there is no result {int(wanted_position)}; the list has "
                    f"{len(items)} entr{'y' if len(items) == 1 else 'ies'}"
                ),
            )

        text = str(query or "").strip()
        if not text:
            return Selection(
                kind="no_selection",
                result_set=entry,
                message="say which result to open - its position or its title",
            )
        pool = [item for item in items if not role or item.role == role.strip().lower()]
        matches = rank_title_matches(pool, text)
        if len(matches) == 1:
            return Selection(kind="resolved", item=matches[0], result_set=entry)
        if len(matches) > 1:
            return Selection(
                kind="ambiguous",
                result_set=entry,
                candidates=tuple(matches),
                message=(
                    f"{len(matches)} results match {text!r} ({_quoted(matches)}); "
                    "say which one to open"
                ),
            )
        return Selection(
            kind="not_found",
            result_set=entry,
            message=(
                f"no result matches {text!r}; the current results are {_quoted(items) or '(none)'}"
            ),
        )

    def _prune_sessions(self) -> None:
        """Keep at most ``max_sets`` sessions, dropping the oldest first."""
        if len(self._sets) <= self._max_sets:
            return
        oldest = sorted(self._sets.values(), key=lambda entry: entry.created_at)
        for entry in oldest[: len(self._sets) - self._max_sets]:
            self._sets.pop(entry.session_id, None)
