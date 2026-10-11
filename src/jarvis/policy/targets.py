"""Deterministic identity for an interaction target (browser DOM / UIA).

A :class:`ResolvedTarget` carries *only* the minimum metadata needed to
authorize, re-validate and safely execute one click/type action.  Its
:func:`fingerprint` is a stable digest of that identity so the policy gate and
the act node can prove they are talking about the *same* element across an
``interrupt()`` boundary (owner decision 2, docs/03 I7).

Hard rules:

* fingerprints are **deterministic** and independent of mapping insertion order;
* ambiguous or incomplete identity (unknown kind, blank stable ref / accessible
  name / role, browser target without a canonical origin) yields ``None`` -
  we refuse rather than imply a reliable target;
* only identity metadata is included - never typed text, password values, page
  contents or other sensitive data.  The dataclass deliberately exposes no such
  field, so they cannot leak in by accident.

The target is transient: it is never placed in ``AgentState`` nor added to
``_ALLOWED_MSGPACK_TYPES``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from jarvis.policy.rules import url_origin

__all__ = ["TARGET_KINDS", "ResolvedTarget", "fingerprint"]

#: Recognised target kinds.  Browser targets belong to a web origin; UIA
#: targets belong to a Windows window/control and carry no origin.
TARGET_KINDS = frozenset({"browser", "uia"})


@dataclass(frozen=True)
class ResolvedTarget:
    """Minimum identity metadata for one interaction target.

    ``ref`` must be a *stable* reference the driving backend can re-resolve
    (e.g. a backend-issued node reference for the browser, or an automation id /
    runtime id for UIA).  ``name``/``role`` are the accessible name and
    role/control type.  ``origin`` is only meaningful for browser targets; it
    is canonicalised through ``policy.rules.url_origin`` (scheme/host case,
    default ports, trailing dots, IDNs) and an invalid origin is rejected.
    ``input_type`` and ``is_password`` are password-related metadata used to
    refuse typing into credential fields (docs/03 I7).
    """

    kind: str
    ref: str
    name: str
    role: str
    origin: str = ""
    input_type: str = ""
    is_password: bool = False

    def fingerprint(self) -> str | None:
        """Stable hex digest of this target's identity, or ``None`` if ambiguous."""
        return fingerprint(self)


def _normalise_text(value: str) -> str:
    """Collapse internal whitespace and strip; preserve case for the name."""
    return " ".join(str(value or "").split())


def fingerprint(target: ResolvedTarget) -> str | None:
    """Return the deterministic identity digest for ``target``.

    Returns ``None`` when the identity is missing or ambiguous, so the caller
    fails closed instead of authorizing a target it cannot reliably re-resolve.
    """
    if not isinstance(target, ResolvedTarget):
        return None
    kind = str(target.kind or "").strip().lower()
    if kind not in TARGET_KINDS:
        return None
    ref = str(target.ref or "").strip()
    name = _normalise_text(target.name)
    role = str(target.role or "").strip().lower()
    if not ref or not name or not role:
        return None
    origin = str(target.origin or "").strip()
    input_type = str(target.input_type or "").strip().lower()
    if kind == "browser":
        # Canonicalise via the shared rule so equivalent origins (scheme/host
        # case, default ports, trailing dots, IDNs) collapse to one identity.
        # An origin that is not a valid http(s) origin is rejected outright.
        canonical = url_origin(origin)
        if canonical is None:
            return None
        origin_value = canonical
    else:
        # UIA targets carry no origin.
        origin_value = ""
    payload = json.dumps(
        {
            "kind": kind,
            "ref": ref,
            "name": name,
            "role": role,
            "origin": origin_value,
            "input_type": input_type,
            "is_password": bool(target.is_password),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
