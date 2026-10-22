"""Match observed page elements against a user's query (Stage 8.6, slice 8.6.2).

Pure functions over ``(role, name)`` pairs - no Playwright, no DOM - so the
matching rules are deterministic and testable on their own.

Why matching at all
-------------------
``browser_read`` returns a numbered list of up to 80 elements and the brain
picks one by number.  That works for a small page, but it means the *model*
has to read the list and copy a handle, and a long results page can push the
interesting element past the observation limit.  A query-based matcher lets the
model say *what* it wants ("the third result about Python") and lets the code
decide *which* element that is - deterministically, and without inventing
element ids.

The rules are deliberately conservative:

* exact match first, then a name that starts with the query, then a name that
  merely contains it - so the most specific matches always win;
* document order is preserved inside a tier;
* an optional ``role`` filter narrows the pool before matching;
* the caller decides what to do with several matches (the tool refuses unless
  the caller picked one explicitly).
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["match_identity", "match_indexes", "normalise"]

#: Hard cap on the query length (bounded input, bounded work).
MAX_QUERY_CHARS = 200


def normalise(value: str) -> str:
    """Case-fold and collapse whitespace, so "Sign  In" == "sign in"."""
    return " ".join(str(value or "").split()).casefold()


def match_indexes(
    candidates: Sequence[tuple[str, str]],
    query: str,
    *,
    role: str = "",
) -> list[int]:
    """Return the indices of the candidates whose name matches ``query``.

    ``candidates`` is the page's elements in document order as
    ``(role, accessible_name)`` pairs.  The result is the matching indices in
    the same order, from the most specific tier that matched:

    1. the normalised name equals the normalised query;
    2. the normalised name starts with the normalised query;
    3. the normalised name contains the normalised query.

    An empty query matches nothing (never "everything").
    """
    text = normalise(query)
    if not text:
        return []
    wanted_role = normalise(role)
    pool = [
        index
        for index, (element_role, _) in enumerate(candidates)
        if not wanted_role or normalise(element_role) == wanted_role
    ]
    names = {index: normalise(candidates[index][1]) for index in pool}
    for test in (
        lambda name: name == text,
        lambda name: name.startswith(text),
        lambda name: text in name,
    ):
        matched = [index for index in pool if test(names[index])]
        if matched:
            return matched
    return []


def match_identity(role: str, name: str, position: int) -> str:
    """A stable identity for the ``position``-th match (1-based).

    This is what the composite browser tools bind an approval to.  It must be
    the same for the policy gate's observation and for the act node's
    re-observation, so it may **not** contain the transient ``e1..eN`` handle
    (which changes with every observation).  Role, name and the position among
    the matches do not change unless the page itself changed - and a changed
    page is exactly what the approval binding must detect.
    """
    return f"{position}|{normalise(role)}|{normalise(name)}"
