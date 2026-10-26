"""memory_save node: record what worked (and what did not) — Phase 8.

This is the **only** node that writes to memory, and it runs after ``respond``
so every terminal path (success, refusal, cancel, failure) passes through it
exactly once.  It is deliberately dumb and deterministic: no LLM, no policy
call, no tool execution.  It cannot approve anything — a stored skill is only
ever *text handed to the planner*, which must still produce a new plan that
goes through ``validate`` and ``policy_gate``.

The skill gate (docs/05_BUILD_PLAN.md, Phase 8: "only successful, fully
verified, non-tainted plans are saved") is :func:`_skill_gate` and every one of
its checks has a test in ``tests/unit/test_memory_save_node.py``:

============================  =============================================
Condition                     Why it blocks a save
============================  =============================================
no memory backend             nothing to write to (``NullMemory``)
``--dry-run``                 a rehearsal must not create side effects
``cancelled`` / halted        the human stopped; nothing was proven
not a tool plan               a conversation is not an executable skill
any result not ``ok``         the task did not succeed
any ``verified is False``     the tool's own check said it did not hold
any tainted result            output came from web/file/window text
any ``depends_on_untrusted``  the args were built from untrusted text
============================  =============================================

``verified is None`` ("succeeded, but there is no post-condition to check")
is a *third* state, not a failure: the step executed successfully and only the
independent proof is missing.  Recording it as a failure poisoned the failure
memory — a later "lock my computer" was refused with "a previous attempt
failed" although the lock had succeeded — so it is excluded here and in
:func:`_failure_record`.

Failures are still worth recording, and that is the *other* half of this node:
a failed or verification-failed step is appended to the failure log so a later
replan sees "this approach already failed here".  A tainted step is still
recorded as a failure (the failure text is redacted) but never becomes a skill.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from jarvis.agent.context import AppContext
from jarvis.agent.state import Plan, StepResult
from jarvis.memory.base import MemoryRecord
from jarvis.memory.db import looks_secret, safe_value
from jarvis.memory.failures import failure_category

__all__ = ["REASONS", "detect_name_statement", "memory_save"]

_LOG = logging.getLogger("memory")

#: Closed vocabulary for ``memory_saved_reason``.  A code, not prose: callers
#: compare it, and the human detail (which check failed, which tool) lives in
#: the log line and in the failure row itself.
SAVED = "saved"
SKIPPED_CANCELLED = "skipped_cancelled"
SKIPPED_HALTED = "skipped_halted"
SKIPPED_UNVERIFIED = "skipped_unverified"
SKIPPED_SENSITIVE = "skipped_sensitive"
SKIPPED_CONVERSATION = "skipped_conversation"
NAME_SAVED = "name_saved"
#: Nothing was recorded for a reason outside the task itself: no backend (or a
#: disabled memory), ``--dry-run``, ``save_skills`` off, or a write that raised.
ERROR = "error"

REASONS = (
    SAVED,
    SKIPPED_CANCELLED,
    SKIPPED_HALTED,
    SKIPPED_UNVERIFIED,
    SKIPPED_SENSITIVE,
    SKIPPED_CONVERSATION,
    NAME_SAVED,
    ERROR,
)

#: ``meta`` key that marks a record as a *failure* rather than a skill, so the
#: backend routes it to the failure log (shared with
#: :class:`jarvis.memory.store.SqliteMemory`).
_FAILURE_META = "failure_id"

#: Explicit name statements: the phrasing, whether it may replace a name already
#: on file, and whether every candidate word must carry name-shape evidence.
#: Softer phrasing only fills an empty slot, so a passing "you can call me Bob"
#: cannot silently rename the owner.  Only the unambiguous declarative "my name
#: is" can accept an all-lowercase candidate; "call me" is also a normal request
#: ("call me again"), so the other phrasings need proper-name evidence.
_NAME_STATEMENTS: tuple[tuple[str, bool, bool], ...] = (
    # Longer alternatives precede the "call me" substring inside them.  Match
    # selection is by input position; this order only breaks an exact tie.
    (r"\byou can call me\b", False, True),
    (r"\bi['\u2019]?m called\b", False, True),
    (r"\bmy name is\b", True, False),
    (r"\bcall me\b", True, True),
)
#: Read the whole word run after the phrase.  The run is capped later: taking
#: only two words would silently turn "Mary Jane Watson" into "Mary Jane".
_NAME_RUN = re.compile(
    r"(?P<run>[^\W\d_]+(?:['\u2019-][^\W\d_]+)?"
    r"(?:\s+[^\W\d_]+(?:['\u2019-][^\W\d_]+)?)*)",
    re.UNICODE,
)
_DISCOURSE_WORDS = (
    r"(?:please|thanks|thank\s+you|instead|actually|really|too|also|"
    r"just|though|however|anyway|now|for\s+short)"
)
#: A short, closed list of discourse words that can follow a valid name.
#: Applied only to the end of the local run, never used to shorten a longer
#: contextual phrase ("call me after lunch" must stay invalid).
_TRAILING_DISCOURSE = re.compile(rf"(?:\s+{_DISCOURSE_WORDS})+$", re.IGNORECASE)
#: The same closed list after a comma: "Call me Bob, please."  Requiring the
#: whole tail keeps "Call me Bob, please stop." invalid.
_SEPARATED_DISCOURSE = re.compile(
    rf"^\s*,\s*{_DISCOURSE_WORDS}(?:\s*[.!?]+)?\s*$", re.IGNORECASE
)
#: A comma may continue into a clause only when that clause contains another
#: supported name statement.  This preserves the documented earliest-statement
#: rule for "You can call me Bob, but my name is Alice."
_COORDINATING_CLAUSE = re.compile(
    r"^\s*,\s*(?:but|and|while|whereas)\b", re.IGNORECASE
)
_NAME_TERMINAL = ".!?\u2026"
_NAME_CLOSING_WRAPPERS = "\"'\u201d\u2019`)]}>"
_MAX_NAME_CHARS = 40
_MIN_NAME_CHARS = 2
_MAX_PREFIX_CHARS = 120

#: Text inside these spans is quoted or reported speech, not a declaration.
#: The single-quote patterns require a non-word character on each side so an
#: apostrophe (``I'm``, ``O'Brien``) never opens a span.
_QUOTE_PATTERNS = (
    re.compile(r'"[^"\n]*"'),
    re.compile(r"\u201c[^\u201d\n]*\u201d"),
    re.compile(r"`[^`\n]*`"),
    re.compile(r"(?<![\w'\u2019])'[^'\n]*'(?![\w'\u2019])"),
    re.compile(r"(?<![\w'\u2019])\u2018[^\u2019\n]*\u2019(?![\w'\u2019])"),
)

#: Phrases immediately before a name that make it reported, hypothetical,
#: uncertain or an example rather than a declaration.  Kept deliberately
#: conservative: a missed learning can be retried, a wrong name is persistent.
_NON_DECLARATION = re.compile(
    r"(?:"
    r"\bwhat\s+if\b|\bif\b|\bsuppose\b|\bsupposing\b|\bimagine\b|\bimagining\b|"
    r"\bpretend\b|\bpretending\b|\bhypothetical(?:ly)?\b|\bfor\s+example\b|"
    r"\bmaybe\b|\bperhaps\b|\bprobably\b|\bpossibly\b|\bwhether\b|"
    r"\baccording\s+to\b|"
    r"\b(?:said|says|saying|asked|asks|told|tells|telling|wrote|writes|written|"
    r"texted|reported|claims?|claimed|mentioned|mentions)\b|"
    r"\b(?:read|reads|heard|hears)\b[^.!?]{0,40}\bthat\b"
    r")",
    re.IGNORECASE,
)

#: A negation directly before the phrase: "don't call me Bob".  The short word
#: tail allows "I do not want you to call me Bob" while a comma ("Do not forget,
#: my name is Bob") ends the guard.
_NEGATED_DECLARATION = re.compile(
    r"\b(?:do\s+not|don['\u2019]?t|never|stop|not)\b(?:\s+\w+){0,4}\s*$",
    re.IGNORECASE,
)

#: Punctuation allowed between a statement phrase and the name.
_NAME_LEADING_JUNK = " \t\r\n.,;:!?\"'\u201c\u201d\u2018\u2019`()[]{}<>-\u2013\u2014"

#: Words that are never a name: command verbs, pronouns, fillers and the
#: everyday nouns a mis-parse would grab ("call me a taxi every morning").
#: Kept as one whitespace-separated string so the set stays readable.
_NOT_A_NAME_WORDS = """
    a an the and or but if then than that this these those it its is are am was were be been
    i me my you your we us our he she they them his her theirs
    all any both each either neither every few many more most much other some such
    what whatever which whichever who whom whose why how
    jarvis sir madam boss mate dude man woman
    please thanks thank ok okay yes no not never now later tomorrow today yesterday
    back
    about above across after against along amid among around at before behind below beneath
    beside besides between beyond by despite down during except inside into near off on onto
    out outside over past per since through throughout till toward towards under underneath
    until up upon via within
    tonight soon morning afternoon evening night noon midnight lunch dinner breakfast
    meeting appointment week weekend weekday monday tuesday wednesday thursday friday
    saturday sunday minute minutes hour hours day days first second third fourth fifth
    next last previous coming asap immediately anytime sometime whenever away right
    work home school office cell mobile landline
    open close start stop run send play pause search find look show tell say
    delete remove create make write read save set get go go to wait click type press
    call lock unlock load move copy paste cut undo redo scroll
    whatsapp message email file folder app application program website site page link
    photo picture camera screen window tab computer laptop phone
    time date weather music video song news mail
    hello hi hey goodbye bye goodnight
    one two three four five six seven eight nine ten
    http https www url
    password passwd passphrase pin token secret key apikey login credential credentials
    actually really instead too also just maybe perhaps though however anyway again quickly
    for from as to of with without because when whether while like
    """
_NOT_A_NAME = frozenset(_NOT_A_NAME_WORDS.split())


def memory_save(state: dict[str, Any], ctx: AppContext) -> dict[str, Any]:
    """Persist one bounded, secret-free memory record for this task.

    It also learns the user's name when they state it explicitly (see
    :func:`_learn_name`).  That happens here, after ``respond``, so it can
    never change the answer to the turn it learned from: the name is used from
    the *next* turn on, when ``compose`` reads the ``user_name`` preference.

    Returns three state keys, and **every** return path sets all three:

    * ``memory_saved`` - a **skill** was stored (a failure record does not
      count: learning from a failure is not the same as learning a skill);
    * ``memory_saved_reason`` - one of :data:`REASONS` (``"saved"``,
      ``"skipped_cancelled"``, ``"skipped_halted"``, ``"skipped_unverified"``,
      ``"skipped_sensitive"``, ``"skipped_conversation"``, ``"error"``,
      ``"name_saved"``), never ``None``;
    * ``memory_failure_logged`` - a failure example was written instead.
    """
    outcome = _outcome(state, ctx)
    if outcome.reason != SAVED:
        ctx.logger.debug("memory: %s", outcome.reason)
    name_saved = _learn_name(state, ctx)
    reason = SAVED if outcome.record is not None and not outcome.is_failure else outcome.reason
    if name_saved and reason != SAVED:
        reason = NAME_SAVED
    return {
        "memory_saved": outcome.record is not None and not outcome.is_failure,
        "memory_saved_reason": reason,
        "memory_failure_logged": outcome.is_failure,
    }


# --------------------------------------------------------------------- gate


class _Outcome:
    """What :func:`memory_save` decided, and why."""

    __slots__ = ("is_failure", "reason", "record")

    def __init__(
        self, reason: str = SAVED, record: MemoryRecord | None = None, *, is_failure: bool = False
    ) -> None:
        self.reason = reason
        self.record = record
        self.is_failure = is_failure


def _outcome(state: dict[str, Any], ctx: AppContext) -> _Outcome:
    """Decide the single record (if any) this task contributes to memory."""
    backend = ctx.memory
    if backend is None:
        return _Outcome(ERROR)
    if ctx.dry_run:
        return _Outcome(ERROR)
    if _sensitive(state):
        # A credential (or credential-looking value) is anywhere in this turn:
        # store nothing at all rather than a scrubbed half-example.
        return _Outcome(SKIPPED_SENSITIVE)

    gate = _skill_gate(state, ctx)
    if gate == SAVED:
        return _write(backend, _skill_record(state), SAVED)

    failure = _failure_record(state, ctx)
    if failure is None:
        return _Outcome(gate)
    return _write(backend, failure, gate, is_failure=True)


def _sensitive(state: dict[str, Any]) -> bool:
    """True when the user's text or any planned argument looks like a credential.

    Reuses :func:`jarvis.memory.db.looks_secret` - the same guard the stores
    apply - so nothing is stored for a turn that carries a secret, and the
    reason says exactly that.
    """
    if looks_secret(str(state.get("user_input") or "")):
        return True
    plan: Plan | None = state.get("plan")
    for step in plan.steps if plan is not None else []:
        for value in step.args.values():
            if looks_secret(str(value)):
                return True
    return False


def _skill_gate(state: dict[str, Any], ctx: AppContext) -> str:
    """The reason this task must **not** become a skill (``SAVED`` if it may)."""
    if state.get("cancelled"):
        return SKIPPED_CANCELLED
    if state.get("halted_reason"):
        return SKIPPED_HALTED

    plan: Plan | None = state.get("plan")
    if plan is None or plan.kind != "tool" or not plan.steps:
        return SKIPPED_CONVERSATION

    results: list[StepResult] = list(state.get("results") or [])
    if not results:
        return SKIPPED_UNVERIFIED
    if len(results) != len(plan.steps):
        return SKIPPED_UNVERIFIED
    if any(not result.ok for result in results):
        return SKIPPED_UNVERIFIED
    if any(result.verified is False for result in results):
        return SKIPPED_UNVERIFIED
    if any(result.verified is None for result in results):
        return SKIPPED_UNVERIFIED
    if any(result.tainted for result in results):
        return SKIPPED_UNVERIFIED
    if any(step.depends_on_untrusted for step in plan.steps):
        return SKIPPED_UNVERIFIED

    if not bool(getattr(ctx.settings.memory, "save_skills", True)):
        return ERROR
    return SAVED


# ------------------------------------------------------------------ records


def _skill_record(state: dict[str, Any]) -> MemoryRecord:
    """The verified plan to remember (goal text + tool names, args redacted)."""
    plan: Plan = state["plan"]
    return MemoryRecord(
        kind="episodic",
        text=plan.goal.strip() or "an unnamed task",
        meta={
            "goal_text": plan.goal,
            "tools": [step.tool for step in plan.steps],
            "plan": _safe_plan(plan),
        },
    )


def _failure_record(state: dict[str, Any], ctx: AppContext) -> MemoryRecord | None:
    """The first bad step to remember, if the task did not fully succeed.

    "Bad" means failed, failed-verification **or tainted**: a step whose
    output was untrusted external text is exactly the approach a later run
    should not repeat, even though it technically succeeded.  A step with
    ``verified=None`` is *not* bad — "succeeded, unverifiable" is not
    "failed" — and must never enter the failure log (that poisoned it and
    blocked later successful runs of the same task).  A tainted result is
    recorded with its output text dropped, since that text is the untrusted
    part.

    The step's outcome is stored alongside the row (``step_state``) as
    provenance, so the store can build an honest hint and so a later reader can
    tell a genuine failure from an artefact of an older build.  Rows written
    before this existed are handled by
    :func:`jarvis.memory.db.migrate_failures`.

    The stored text is the **tool name plus a short category** (M3), never the
    tool's own error string, a page title or any file content: a failure example
    is a hint about which approach failed, and the raw text is exactly the
    untrusted, unbounded part.  Only a terminal failure is recorded - the run has
    already ended (``memory_save`` is the post-answer node) and the replan budget
    is spent or the task halted, so a retry still in flight is never archived as
    a dead end.
    """
    results: list[StepResult] = list(state.get("results") or [])
    bad = next(
        (r for r in results if not r.ok or r.verified is False or r.tainted),
        None,
    )
    if bad is None or not _failure_is_terminal(state, ctx):
        return None
    plan: Plan | None = state.get("plan")
    step = next((s for s in (plan.steps if plan else []) if s.id == bad.step_id), None)
    goal = plan.goal if plan else ""
    outcome = {"ok": bad.ok, "verified": bad.verified, "tainted": bad.tainted}
    category = failure_category(outcome)
    tool = step.tool if step else ""
    return MemoryRecord(
        kind="episodic",
        text=f"{tool or 'tool'} failed ({category})",
        meta={
            _FAILURE_META: True,  # routes this record to the failure log
            "goal_text": goal,
            "step": {
                "tool": tool,
                "id": bad.step_id,
                "tainted": bad.tainted,
                "category": category,
            },
            "step_state": outcome,
        },
    )


def _failure_is_terminal(state: dict[str, Any], ctx: AppContext) -> bool:
    """True when this failure ended the task (not a retry still in flight).

    ``memory_save`` runs after ``respond``, so in practice every path here is
    terminal.  The check is explicit anyway: a halted run is terminal, an
    exhausted replan budget is terminal, and so is a result with no retry or
    replan requested - while a run that is still asking to retry is not.
    """
    if state.get("halted_reason"):
        return True
    budget = int(getattr(ctx.settings.agent, "max_replans", 2) or 0)
    if int(state.get("replan_count") or 0) >= budget:
        return True
    return not (state.get("retry_now") or state.get("replan_now"))


def _safe_plan(plan: Plan) -> dict:
    """Plan as a plain dict with credential-looking argument values removed.

    Uses the same :func:`jarvis.memory.db.safe_value` the store uses, so a
    secret nested inside a list argument (``{"paths": [{"password": ...}]}``)
    is stripped here too, not just at the top level of ``args``.
    """
    steps = []
    for step in plan.steps:
        args = {key: safe_value(key, value) for key, value in step.args.items()}
        steps.append({"id": step.id, "tool": step.tool, "args": args, "expect": step.expect})
    return {"goal": plan.goal, "kind": plan.kind, "steps": steps}


def _write(
    backend: Any,
    record: MemoryRecord | None,
    reason: str,
    *,
    is_failure: bool = False,
) -> _Outcome:
    """Hand the record to the backend; a memory error never fails the task."""
    if record is None:
        return _Outcome(reason or ERROR)
    try:
        backend.remember(record)
    except Exception as exc:  # noqa: BLE001 - memory must never break a task
        _LOG.warning("memory write failed: %s", exc)
        return _Outcome(ERROR)
    return _Outcome(reason, record=record, is_failure=is_failure)


# --------------------------------------------------------- name learning


def detect_name_statement(user_input: str) -> tuple[str, bool] | None:
    """The name the user just stated, and whether the phrasing is explicit.

    Deterministic and rule-based: no LLM, no inference from behaviour.  Only the
    four phrasings in :data:`_NAME_STATEMENTS` count, and the name must be one
    or two words of letters, apostrophes and hyphens (2-40 characters).  Digits,
    dots/slashes (so no URLs), stop words and command verbs are refused, and the
    candidate must pass the same secret check the preference store applies.

    Except for the unambiguous declarative "my name is", every candidate word
    must look like a proper name (a capital letter, name punctuation, or a
    caseless script).  That keeps an ordinary request such as "call me again"
    from being read as the name "again", while "my name is arya" can still be
    learned without title case.

    Returns ``(name, strong)`` where  ``strong`` is true for "my name is" / "call
    me" - the phrasings allowed to replace a name already on file - or ``None``
    when the input states no usable name.

    Matches are resolved by their position in the input, not by the order of
    :data:`_NAME_STATEMENTS`: an earlier valid statement wins over a later one,
    while pattern length and strength only break ties at the same position.
    A phrase inside quotes, or preceded by reported/hypothetical/uncertain
    wording, is ignored.
    """
    text = " ".join(str(user_input or "").split())
    if not text:
        return None
    quoted = _quoted_spans(text)
    candidates: list[tuple[int, int, bool, str]] = []
    for phrase, strong, require_name_shape in _NAME_STATEMENTS:
        for match in re.finditer(phrase, text, re.IGNORECASE):
            if _inside_quotes(match.start(), quoted):
                continue
            prefix = text[max(0, match.start() - _MAX_PREFIX_CHARS) : match.start()]
            if _NON_DECLARATION.search(prefix) or _NEGATED_DECLARATION.search(prefix):
                continue
            name = _clean_name(
                text[match.end() :], require_name_shape=require_name_shape
            )
            if name is not None:
                candidates.append(
                    (match.start(), match.end() - match.start(), strong, name)
                )
    if not candidates:
        return None
    _, _, strong, name = min(
        candidates, key=lambda item: (item[0], -item[1], not item[2])
    )
    return name, strong


def _quoted_spans(text: str) -> list[tuple[int, int]]:
    """Character spans covered by paired quotes, in any supported style."""
    return [
        (match.start(), match.end())
        for pattern in _QUOTE_PATTERNS
        for match in pattern.finditer(text)
    ]


def _inside_quotes(position: int, spans: list[tuple[int, int]]) -> bool:
    """True when ``position`` sits inside one of ``spans``."""
    return any(start < position < end for start, end in spans)


def _has_name_shape(word: str) -> bool:
    """True when a candidate word looks like a name, not ordinary command text.

    A capital letter, apostrophe/hyphen, or a caseless script is positive
    evidence.  An all-lowercase Latin word is ambiguous outside "my name is"
    ("call me again"), so it is not accepted there without other evidence.
    """
    if any(char.isupper() for char in word):
        return True
    if any(char in "'\u2019-" for char in word):
        return True
    return not any(char.islower() for char in word)


def _valid_name_boundary(tail: str) -> bool:
    """True when the text after the candidate cannot hide a longer utterance.

    Terminal punctuation and closing wrappers may end the name; terminal
    punctuation followed by whitespace starts a separate sentence.  A comma may
    introduce only the closed discourse list or a coordinating clause that
    itself contains another supported name statement.  Every other separator is
    part of the candidate's context, so the whole candidate is rejected.
    """
    cleaned = tail.strip()
    if not cleaned:
        return True
    end = 0
    saw_terminal = False
    while end < len(cleaned) and (
        cleaned[end] in _NAME_TERMINAL or cleaned[end] in _NAME_CLOSING_WRAPPERS
    ):
        saw_terminal = saw_terminal or cleaned[end] in _NAME_TERMINAL
        end += 1
    if end == len(cleaned):
        return True
    if saw_terminal and cleaned[end].isspace():
        return True
    remaining = cleaned[end:]
    if _SEPARATED_DISCOURSE.fullmatch(remaining):
        return True
    if _COORDINATING_CLAUSE.match(remaining):
        return detect_name_statement(remaining) is not None
    return False


def _clean_name(rest: str, *, require_name_shape: bool = False) -> str | None:
    """Validate the candidate name at the start of ``rest`` (or ``None``).

    Only a short, closed trailing discourse phrase is dropped ("Call me Aarya
    instead" -> "Aarya").  Every remaining word must pass the non-name word
    guard.  When name-shape evidence is required (every phrasing except "my name
    is"), every word must also look like a proper name: this rejects an ordinary
    request such as "call me quickly" or "call me Bob again" as a whole, never
    by trimming it to a shorter prefix.  A third word rejects rather than
    truncates ("Mary Jane Watson" is not saved as "Mary Jane").
    """
    cleaned = rest.lstrip(_NAME_LEADING_JUNK)
    found = _NAME_RUN.match(cleaned)
    if found is None:
        return None
    if not _valid_name_boundary(cleaned[found.end() :]):
        return None
    remaining = _TRAILING_DISCOURSE.sub("", found.group("run")).strip()
    words = remaining.split()
    if not 1 <= len(words) <= 2:
        return None
    if any(word.casefold().strip("'-\u2019") in _NOT_A_NAME for word in words):
        return None
    if require_name_shape and not all(_has_name_shape(word) for word in words):
        return None
    name = " ".join(words)
    if not (_MIN_NAME_CHARS <= len(name) <= _MAX_NAME_CHARS):
        return None
    if any(char.isdigit() for char in name):
        return None
    if looks_secret(name):
        return None
    return name


def _learn_name(state: dict[str, Any], ctx: AppContext) -> bool:
    """Store an explicitly stated name as the ``user_name`` preference.

    Only a plain conversation turn counts: an action turn already ran tools,
    and a halted or cancelled turn is not a statement the user meant to keep.
    A sensitive turn writes nothing, even when it also contains a valid name.
    The write goes through the existing :class:`PreferenceStore` (same
    validation, same secret guard); there is no second profile store and no
    tool, so nothing here can act.  Never raises.
    """
    try:
        if _sensitive(state):
            return False
        if not bool(getattr(ctx.settings.memory, "learn_name", True)):
            return False
        if ctx.dry_run or state.get("cancelled") or state.get("halted_reason"):
            return False
        plan: Plan | None = state.get("plan")
        if plan is None or plan.kind != "conversation" or plan.steps:
            return False
        detected = detect_name_statement(str(state.get("user_input") or ""))
        if detected is None:
            return False
        name, strong = detected
        store = getattr(getattr(ctx.memory, "preferences", None), "set", None)
        getter = getattr(getattr(ctx.memory, "preferences", None), "get", None)
        if not callable(store) or not callable(getter):
            return False
        existing = getter("user_name")
        if isinstance(existing, str) and existing.strip():
            if existing.strip().casefold() == name.casefold():
                return False  # already known: nothing changed
            if not strong:
                return False  # only an explicit phrasing may replace a name
        store("user_name", name)
        ctx.logger.info("memory event=name_saved")
        return True
    except Exception as exc:  # noqa: BLE001 - learning a name must never break a task
        _LOG.warning("name preference not stored: %s", exc)
        return False
