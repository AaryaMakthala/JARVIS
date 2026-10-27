"""Guarded final click (browser_final_click) - implementation."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from jarvis.config import Settings
from jarvis.memory.db import connect, init_schema


ALLOWED_ACTIONS = {"send", "submit", "post", "publish", "subscribe", "book"}

#: The tool that performs a consequential browser click.  Defined here (not in
#: the tool module) so the gate, the pre-confirmation observation and the tool
#: itself all compare against one literal.
FINAL_CLICK_TOOL = "browser_final_click"
PAYMENT_WORDS = {
    "buy",
    "purchase",
    "pay",
    "payment",
    "payments",
    "checkout",
    "order",
    "place order",
    "donate",
}


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", str(s or "")).lower().split())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_hex(s: str) -> str:
    return hashlib.sha256(str(s or "").encode("utf-8")).hexdigest()


# ── the observed readback (Phase 5C) ─────────────────────────────────────
#
# A readback is the exact list of *observed* facts the user is asked to
# approve.  Two rules make it trustworthy:
#
# * every field carries the **source** it came from - ``live_dom``/
#   ``observed_origin`` (read from the live page), ``validated_typed`` (data
#   JARVIS itself typed after validation) or ``proposed_unverified`` (the
#   model's unverified guess);
# * the digest is computed over **verified fields only**, so a proposal value
#   can be displayed under an explicit "proposed, unverified" label but can
#   never become part of the binding as if it had been observed.

#: Schema version of the readback structure.  It is part of the digest, so a
#: readback prepared by another version never authorizes an action here.
READBACK_SCHEMA = 1

SOURCE_LIVE_DOM = "live_dom"
SOURCE_OBSERVED_ORIGIN = "observed_origin"
SOURCE_VALIDATED_TYPED = "validated_typed"
SOURCE_PROPOSED = "proposed_unverified"

#: A sha256 hex digest, lowercase, exactly 64 characters.
_HEX64_RE = re.compile(r"\A[0-9a-f]{64}\Z")
#: A recipient JARVIS can read on the control itself (email, then phone).
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE_RE = re.compile(r"\+?[0-9][0-9 ()-]{6,}[0-9]")
#: A number immediately followed by a three-letter currency code.
_AMOUNT_RE = re.compile(r"(?<![\w.])([0-9]{1,9}(?:\.[0-9]{1,2})?)\s*([A-Za-z]{3})(?![\w])")
#: A standalone number, used only to detect an amount shown *without* a
#: currency (which the cap must refuse rather than default to ``0``).
_BARE_NUMBER_RE = re.compile(r"(?<![\w.])[0-9]{1,9}(?:\.[0-9]{1,2})?(?![\w.])")

#: The amount fields a typed-data dict may contribute.  Anything else is
#: ignored rather than trusted.
_AMOUNT_KEYS = ("amount", "price", "total")


def is_hex64(value: Any) -> bool:
    """Whether *value* is a well-formed lowercase sha256 hex digest."""
    return isinstance(value, str) and bool(_HEX64_RE.match(value))


def digests_equal(left: Any, right: Any) -> bool:
    """Constant-time comparison of two digests (both must be well-formed)."""
    if not is_hex64(left) or not is_hex64(right):
        return False
    return hmac.compare_digest(str(left), str(right))


def canonical_json(obj: Any) -> str:
    """The one canonical serialisation every readback digest is computed over.

    Sorted keys and no insignificant whitespace, so two structurally equal
    readbacks always produce the same bytes; UTF-8 is applied by the caller.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def readback_material(details: dict[str, Any]) -> dict[str, Any]:
    """The digest material: every verified field, never the proposal.

    ``details["proposed"]`` is dropped whole.  It is display-only, so a model
    guess that leaks into the payload still cannot change the binding - and a
    proposal that *is* the danger (a changed recipient) cannot be laundered
    into a digest the user never saw.
    """
    if not isinstance(details, dict):
        return {}
    return {k: v for k, v in details.items() if k != "proposed"}


def readback_digest(details: dict[str, Any]) -> str:
    """sha256 hex of the canonical JSON of the verified readback material."""
    blob = canonical_json(readback_material(details))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def stable_identity(role: str, name: str) -> str:
    """The stable ``position|role|name`` identity of a control.

    Never the transient ``e1..eN`` handle: the handle changes on every
    observation, so a digest bound to it would refuse every honest
    re-observation while hiding the changes that matter.
    """
    try:
        from jarvis.browser.find import match_identity

        return str(match_identity(str(role or ""), str(name or ""), 1))
    except Exception:  # noqa: BLE001 - a defensive fallback, never a weaker check
        return f"1|{str(role or '').strip().lower()}|{str(name or '').strip().lower()}"


def _host_of(origin: str) -> str:
    try:
        from urllib.parse import urlsplit

        return str(urlsplit(str(origin or "")).hostname or "").lower().rstrip(".")
    except Exception:  # noqa: BLE001 - an unparsable origin is simply not observed
        return ""


def _recipient_from(text: str) -> str:
    """A recipient JARVIS can actually *see* on the control, else ``""``."""
    raw = str(text or "")
    match = _EMAIL_RE.search(raw) or _PHONE_RE.search(raw)
    return match.group(0) if match else ""


def _amount_from(text: str) -> tuple[float | None, str, str]:
    """Return ``(amount, currency, reason)`` observed in *text*.

    ``reason`` is non-empty only when the text carries money-like content that
    cannot be read unambiguously (a bare number with no currency, or several
    different numbers).  A text with no money-like content at all returns
    ``(None, "", "")`` - "no amount here", not "an amount I guessed".
    """
    raw = str(text or "")
    paired = _AMOUNT_RE.findall(raw)
    amounts = sorted({(float(value), currency.upper()) for value, currency in paired})
    if len(amounts) > 1:
        return None, "", "more than one amount is shown on this control"
    bare = _BARE_NUMBER_RE.findall(raw)
    if not amounts:
        if bare:
            # A number with no currency next to it is exactly the ambiguity the
            # cap must not paper over (JARVIS will never default it to 0).
            return None, "", "an amount is shown without a currency"
        return None, "", ""
    amount, currency = amounts[0]
    if bare and len(set(bare)) > 1:
        return None, "", "the amount shown on this control is ambiguous"
    return amount, currency, ""


def _observe_element_fields(element: Any) -> dict[str, Any]:
    """The fields read straight off the live control, tagged ``live_dom``."""
    name = str(getattr(element, "name", "") or "")
    role = str(getattr(element, "role", "") or "")
    origin = str(getattr(element, "origin", "") or "")
    host = _host_of(origin)
    return {
        "schema": READBACK_SCHEMA,
        "target": {
            "name": name,
            "role": role,
            "identity": stable_identity(role, name),
            "input_type": str(getattr(element, "input_type", "") or ""),
            "source": SOURCE_LIVE_DOM,
        },
        "origin": {"value": origin, "source": SOURCE_OBSERVED_ORIGIN},
        "host": {"value": host, "source": SOURCE_OBSERVED_ORIGIN},
        "destination": {"value": host, "source": SOURCE_OBSERVED_ORIGIN},
    }


def _typed_fields(typed: Any) -> dict[str, Any]:
    """The ``validated_typed`` fields, restricted to known, primitive keys.

    Only data JARVIS validated and itself typed qualifies; unknown keys are
    dropped rather than trusted, and the free-text fields are length-capped so
    a readback can never become a channel for a whole page of text.
    """
    out: dict[str, Any] = {}
    if not isinstance(typed, dict):
        return out
    for key in ("recipient", "destination", "message", "item", "service"):
        value = typed.get(key)
        if isinstance(value, str) and value.strip():
            out[key] = {"value": value.strip()[:200], "source": SOURCE_VALIDATED_TYPED}
    for key in _AMOUNT_KEYS:
        value = typed.get(key)
        if value is None or isinstance(value, bool):
            continue
        try:
            out["amount"] = {"value": float(value), "source": SOURCE_VALIDATED_TYPED}
        except (TypeError, ValueError):
            continue
        break
    currency = typed.get("currency")
    if isinstance(currency, str) and currency.strip():
        out["currency"] = {"value": currency.strip().upper(), "source": SOURCE_VALIDATED_TYPED}
    return out


def _proposed_fields(proposed: Any) -> dict[str, Any]:
    """The untrusted proposal, shown only under an unverified label."""
    out: dict[str, Any] = {}
    if not isinstance(proposed, dict):
        return out
    for key, value in sorted(proposed.items()):
        if not isinstance(key, str) or not key.isidentifier():
            continue
        text = str(value or "").strip()
        if text:
            out[key] = {"value": text[:160], "source": SOURCE_PROPOSED}
    return out


def final_click_limits(settings: Any) -> tuple[int, float | None, str, float, bool]:
    """``(max_total, spending_cap, currency, binding_ttl_s, voice_opt_in)``.

    One reader for the ``browser.final_click_*`` settings, so the gate's
    pre-approval observation and the tool's pre-click re-check can never
    disagree about whether a cap is configured.  A missing settings object
    yields the fail-closed defaults (``max_total == 0`` disables the tool).
    """
    browser_cfg = getattr(settings, "browser", None)
    if browser_cfg is None:
        return 0, None, "USD", 180.0, False
    return (
        int(getattr(browser_cfg, "final_click_max_total", 0) or 0),
        getattr(browser_cfg, "final_click_spending_cap", None),
        str(getattr(browser_cfg, "final_click_spending_currency", "USD") or "USD"),
        float(getattr(browser_cfg, "final_click_binding_ttl_s", 180) or 180),
        bool(getattr(browser_cfg, "final_click_voice_approval", False)),
    )


@dataclass(frozen=True)
class ReadbackObservation:
    """One live observation: the validated readback, or the reason to refuse."""

    details: dict[str, Any] | None
    reason: str = ""
    #: ``True`` when the refusal is a hard policy block (re-observing will not
    #: help - a payment control, a password field, a cap breach); ``False`` when
    #: it is recoverable (the page moved, a detail is not on screen yet).
    halting: bool = False

    @property
    def ok(self) -> bool:
        return self.details is not None

    @property
    def digest(self) -> str:
        return readback_digest(self.details) if self.details else ""


def build_observed_readback(
    element: Any,
    action: str,
    *,
    typed: Any = None,
    proposed: Any = None,
    spending_cap: float | None = None,
    spending_currency: str = "USD",
) -> ReadbackObservation:
    """Build the readback for *element* from observed/validated data only.

    Runs every pre-approval check that does not need the engine: the action must
    be one of :data:`ALLOWED_ACTIONS`, the target must resolve to exactly one
    named/roled control, its origin must be observable, it must not be a
    password or payment/purchase control, and the control must match the action
    the plan declared.  When a spending cap is configured the amount **and** its
    currency must both be observable on the control - a missing, ambiguous,
    unparsable, negative or mismatched amount is a refusal, never a ``0``.

    Returns a :class:`ReadbackObservation`; a refusal carries a human sentence
    and whether re-observing could ever help.
    """
    if element is None:
        return ReadbackObservation(None, "the target could not be resolved on the live page")
    declared = str(action or "")
    if declared not in ALLOWED_ACTIONS:
        return ReadbackObservation(None, "the declared action is not an allowed one", True)
    name = str(getattr(element, "name", "") or "")
    role = str(getattr(element, "role", "") or "")
    if getattr(element, "is_password", False):
        return ReadbackObservation(None, "refusing to interact with a password field", True)
    if classify_label_for_payment(name) == "payment":
        return ReadbackObservation(None, "payment/purchase control is blocked", True)
    fields = _observe_element_fields(element)
    if not fields["origin"]["value"] or not fields["host"]["value"]:
        return ReadbackObservation(None, "the target's origin could not be observed")
    if not name and not role:
        return ReadbackObservation(None, "the target has no observable name or role")
    if not label_matches_allowed_action(name, declared):
        return ReadbackObservation(
            None, "the declared action does not match the page control", True
        )
    details = dict(fields)
    details["action"] = declared

    # Observed on the control itself, then overridden by nothing: validated
    # typed data is a *second* source, never a way to overwrite an observation.
    seen_recipient = _recipient_from(name)
    if seen_recipient:
        details["recipient"] = {"value": seen_recipient, "source": SOURCE_LIVE_DOM}
    amount, currency, amount_problem = _amount_from(name)
    if amount_problem:
        if spending_cap is not None:
            return ReadbackObservation(None, amount_problem, True)
    elif amount is not None:
        details["amount"] = {"value": amount, "source": SOURCE_LIVE_DOM}
        details["currency"] = {"value": currency, "source": SOURCE_LIVE_DOM}

    for key, entry in _typed_fields(typed).items():
        details.setdefault(key, entry)

    if declared == "send" and "recipient" not in details:
        return ReadbackObservation(
            None, "the recipient could not be observed on the page, so nothing was sent"
        )

    if spending_cap is not None:
        refusal = _cap_refusal(details, spending_cap, spending_currency)
        if refusal is not None:
            return refusal

    proposal = _proposed_fields(proposed)
    if proposal:
        details["proposed"] = proposal
    return ReadbackObservation(details)


def _cap_refusal(
    details: dict[str, Any], spending_cap: float, spending_currency: str
) -> ReadbackObservation | None:
    """Fail closed: an unverifiable or over-cap amount never proceeds."""
    amount_entry = details.get("amount")
    currency_entry = details.get("currency")
    if not isinstance(amount_entry, dict) or not isinstance(currency_entry, dict):
        return ReadbackObservation(
            None,
            "a spending limit is configured but the amount and currency could not both be "
            "observed, so nothing was done",
        )
    try:
        amount = float(amount_entry.get("value"))
    except (TypeError, ValueError):
        return ReadbackObservation(None, "the amount shown could not be read as a number", True)
    if amount < 0:
        return ReadbackObservation(None, "the amount shown is negative", True)
    currency = str(currency_entry.get("value") or "").upper()
    expected = str(spending_currency or "USD").upper()
    if not currency or currency != expected:
        return ReadbackObservation(
            None,
            f"the amount is in {currency or 'an unknown currency'}, not {expected}",
            True,
        )
    try:
        cap = float(spending_cap)
    except (TypeError, ValueError):
        return ReadbackObservation(None, "the configured spending limit is not a number", True)
    if cap < 0 or amount > cap:
        return ReadbackObservation(
            None, f"the amount {amount:.2f} {currency} is above the configured spending limit", True
        )
    return None


def observe_live_readback(
    session: Any,
    ref: str,
    action: str,
    *,
    typed: Any = None,
    proposed: Any = None,
    spending_cap: float | None = None,
    spending_currency: str = "USD",
) -> ReadbackObservation:
    """Observe the real target read-only and build its readback.

    The only session call is ``resolve`` - no click, no type, no navigation, no
    page change of any kind - and an element that cannot be resolved is never
    fabricated from the plan's own words (fail closed).
    """
    if session is None:
        return ReadbackObservation(
            None, "the managed browser is not available, so the target cannot be observed", True
        )
    try:
        element = session.resolve(str(ref))
    except Exception:  # noqa: BLE001 - a failed read is a refusal, never a guess
        element = None
    return build_observed_readback(
        element,
        action,
        typed=typed,
        proposed=proposed,
        spending_cap=spending_cap,
        spending_currency=spending_currency,
    )


def init_final_click_schema(conn: Any) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS final_click_journal (
            id TEXT PRIMARY KEY,
            task_id TEXT,
            host TEXT,
            action TEXT,
            target_fingerprint TEXT,
            confirmation_sha256 TEXT,
            state TEXT NOT NULL,
            outcome_code TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_final_click_task ON final_click_journal(task_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_final_click_state ON final_click_journal(state)")
    conn.commit()


@dataclass(frozen=True)
class FinalClickJournalRow:
    id: str
    task_id: str
    host: str
    action: str
    target_fingerprint: str
    confirmation_sha256: str
    state: str
    outcome_code: str
    created_at: str
    updated_at: str


class FinalClickJournal:
    def __init__(self, path: str | None = None):
        self.path = path
        self._conn = None

    def _get(self):
        if self._conn is not None:
            return self._conn
        conn = connect(self.path)
        init_schema(conn)
        init_final_click_schema(conn)
        self._conn = conn
        return conn

    def prune_old(self, max_rows: int = 50) -> None:
        conn = self._get()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM final_click_journal")
        row = cur.fetchone()
        count = int(row[0]) if row else 0
        if count <= max_rows:
            return
        cur.execute(
            "SELECT id FROM final_click_journal ORDER BY created_at ASC LIMIT ?",
            (count - max_rows,),
        )
        ids = [r[0] for r in cur.fetchall()]
        if not ids:
            return
        cur.execute(
            f"DELETE FROM final_click_journal WHERE id IN ({','.join(['?'] * len(ids))})",
            ids,
        )
        conn.commit()

    def reserve(self, row: FinalClickJournalRow) -> bool:
        conn = self._get()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO final_click_journal (id, task_id, host, action, target_fingerprint, confirmation_sha256, state, outcome_code, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    row.id,
                    row.task_id,
                    row.host,
                    row.action,
                    row.target_fingerprint,
                    row.confirmation_sha256,
                    row.state,
                    row.outcome_code,
                    row.created_at,
                    row.updated_at,
                ),
            )
            conn.commit()
            self.prune_old(50)
            return True
        except Exception:
            return False

    def get_by_id(self, row_id: str) -> dict[str, Any] | None:
        conn = self._get()
        cur = conn.cursor()
        cur.execute("SELECT * FROM final_click_journal WHERE id=?", (row_id,))
        row = cur.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))

    def has_indeterminate_or_dispatched(self, task_id: str) -> bool:
        conn = self._get()
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM final_click_journal WHERE task_id=? AND state IN ('dispatched','indeterminate')",
            (task_id,),
        )
        row = cur.fetchone()
        return int(row[0]) if row else 0 > 0

    def update_state(self, row_id: str, state: str, outcome_code: str = "") -> bool:
        conn = self._get()
        cur = conn.cursor()
        cur.execute(
            "UPDATE final_click_journal SET state=?, outcome_code=?, updated_at=? WHERE id=?",
            (state, outcome_code, _now_iso(), row_id),
        )
        conn.commit()
        return cur.rowcount > 0

    def delete(self, row_id: str) -> None:
        conn = self._get()
        cur = conn.cursor()
        cur.execute("DELETE FROM final_click_journal WHERE id=?", (row_id,))
        conn.commit()

    def total_count_all(self) -> int:
        conn = self._get()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM final_click_journal")
        row = cur.fetchone()
        return int(row[0]) if row else 0


_journal = FinalClickJournal()


def get_journal() -> FinalClickJournal:
    return _journal


def classify_label_for_payment(name: str) -> str | None:
    n = _norm(name)
    if not n:
        return None
    for w in PAYMENT_WORDS:
        wnorm = _norm(w)
        if wnorm and wnorm in n:
            return "payment"
    return None


def label_matches_allowed_action(name: str, action: str) -> bool:
    n = _norm(name)
    a = _norm(action)
    if not n or action not in ALLOWED_ACTIONS:
        return False
    synonyms = {
        "send": ["send"],
        "submit": ["submit"],
        "post": ["post"],
        "publish": ["publish"],
        "subscribe": ["subscribe", "follow"],
        "book": ["book", "reserve", "rsvp"],
    }
    for s in synonyms.get(action, [action]):
        if _norm(s) in n:
            return True
    return False


def _field_value(details: dict[str, Any], key: str) -> str:
    """The display value of one readback field (``""`` when not observed)."""
    entry = details.get(key)
    if isinstance(entry, dict):
        value = entry.get("value")
        return "" if value is None else str(value)
    return "" if entry is None else str(entry)


def format_readback_summary(details: dict[str, Any]) -> str:
    """The one line shown to the user, built only from observed fields."""
    if not isinstance(details, dict):
        return ""
    target = details.get("target") if isinstance(details.get("target"), dict) else {}
    action = str(details.get("action") or "").strip()
    name = str(target.get("name") or "").strip()
    role = str(target.get("role") or "").strip()
    parts = [f"Final click action: {action}"]
    parts.append(f"Target: {name!r} ({role})" if name else f"Target: ({role})")
    for key, label in (
        ("recipient", "Recipient"),
        ("message", "Message"),
        ("destination", "Destination"),
        ("item", "Item"),
        ("service", "Service"),
    ):
        value = _field_value(details, key)
        if value:
            parts.append(f"{label}: {value}")
    currency = _field_value(details, "currency")
    amount = _field_value(details, "amount")
    if amount:
        parts.append(f"Amount: {amount} {currency}".strip())
    host = _field_value(details, "host") or _field_value(details, "origin")
    if host:
        parts.append(f"Site: {host}")
    return " | ".join(parts)


def format_proposed_note(details: dict[str, Any]) -> str:
    """The separate, explicitly *unverified* line for the model's proposal.

    Kept out of :func:`format_readback_summary` on purpose: what JARVIS
    observed and what the model claimed must never be one line of text.
    """
    proposal = details.get("proposed") if isinstance(details, dict) else None
    if not isinstance(proposal, dict) or not proposal:
        return ""
    bits = []
    for key in sorted(proposal):
        entry = proposal[key]
        value = entry.get("value") if isinstance(entry, dict) else entry
        if str(value or "").strip():
            bits.append(f"{key}={value}")
    if not bits:
        return ""
    return "Proposed, unverified (not checked against the page): " + ", ".join(bits)


def readback_identity(details: dict[str, Any]) -> tuple[str, str]:
    """``(target_identity, origin)`` of an observed readback, for association."""
    if not isinstance(details, dict):
        return "", ""
    target = details.get("target") if isinstance(details.get("target"), dict) else {}
    return str(target.get("identity") or ""), _field_value(details, "origin")
