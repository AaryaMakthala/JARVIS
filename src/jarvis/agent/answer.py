"""The boundary between internal routing documents and human-facing answers.

Root cause of the reported symptom
----------------------------------
The brain's single structured LLM call returns a
:class:`~jarvis.agent.schemas.BrainDecision` whose ``response_text`` field is
supposed to hold natural language.  The brain node projected that field
straight into ``AgentState["final_answer"]``, and the terminal/daemon/voice
layers printed or spoke it verbatim.  A model that echoes its own JSON
template into a string field therefore produced exactly the reported output:
a correct sentence *followed by* ``goal``/``actions``/``response_hint``/
``conversation_context``.

The fix is structural, at the boundary — not a string replacement after the
fact.  Two layers:

1. :func:`looks_like_internal_payload` is a **validator** on
   ``BrainDecision.response_text``.  When it fires, the existing one-shot
   repair path in the LLM clients re-asks the model, telling it precisely what
   was wrong.  The bad value never reaches the graph state.
2. :func:`safe_final_answer` is a **defence in depth** applied by the
   ``respond`` node (and therefore by every output channel at once).  If a
   structured payload ever reaches the final answer — from an old checkpoint,
   a future node, or a provider that ignores the schema — the JSON spans are
   removed *by parsing them*, and if nothing natural remains the caller gets a
   fixed honest message instead of machine output.

Detection is a real parse, not a pattern match.  ``text`` is scanned for
balanced ``{...}`` spans (respecting string literals and escapes); a span is
only treated as internal payload when ``json.loads`` succeeds **and** the
resulting object shares keys with :data:`INTERNAL_KEYS`.  Prose that happens
to contain the word "goal" or "actions" is therefore left completely alone.
"""

from __future__ import annotations

import json
from typing import Any

#: Field names that only ever appear in JARVIS's internal routing documents.
#: Used both to detect an echoed schema and to describe it in a repair prompt.
INTERNAL_KEYS: frozenset[str] = frozenset(
    {
        "request_type",
        "clarification_question",
        "response_hint",
        "conversation_context",
        "depends_on_untrusted",
        "action",
        "goal",
        "actions",
        "steps",
        "expect",
        "rationale",
        "plan",
        "schema",
        "response_text",
    }
)

#: How many internal keys a JSON object must share before it is treated as a
#: leaked routing document.  Two is enough to rule out coincidence while
#: tolerating a partial echo of the schema.
MIN_INTERNAL_KEYS = 2

#: The fixed refusal used when a final answer contains nothing but machine
#: output.  Matches the brain node's other honest-failure messages.
NO_ANSWER_MESSAGE = "I couldn't produce an answer."

#: Guard against pathological input: a span longer than this is not something
#: we are willing to parse.
_MAX_SPAN_CHARS = 200_000

#: Cap on how many JSON spans a single answer may contain.
_MAX_SPANS = 8


def _iter_object_spans(text: str) -> list[tuple[int, int]]:
    """Yield ``(start, end)`` index pairs of balanced top-level ``{...}`` spans.

    String literals and their escapes are respected, so a ``{`` or ``}`` inside
    a quoted string cannot unbalance the scan.
    """
    spans: list[tuple[int, int]] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    spans.append((start, index + 1))
                    start = -1
            if len(spans) >= _MAX_SPANS:
                return spans
    return spans


def _is_internal_object(payload: Any) -> bool:
    """Whether a parsed JSON value is a leaked internal routing document."""
    if not isinstance(payload, dict):
        return False
    return len(INTERNAL_KEYS & set(payload)) >= MIN_INTERNAL_KEYS


def strip_code_fence(text: str) -> str:
    """Remove a surrounding markdown fence, if the whole value is fenced."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    body = stripped.split("\n", 1)[-1]
    return body.rsplit("```", 1)[0].strip()


def looks_like_internal_payload(text: str) -> bool:
    """Whether ``text`` is (or embeds) a leaked internal routing document.

    True when the whole value parses as such a document, or when any balanced
    ``{...}`` span inside it parses as one.
    """
    if not text or not text.strip():
        return False
    candidate = strip_code_fence(text)
    if len(candidate) <= _MAX_SPAN_CHARS:
        try:
            if _is_internal_object(json.loads(candidate)):
                return True
        except (json.JSONDecodeError, ValueError, RecursionError):
            pass
    return any(_span_is_internal(text, span) for span in _iter_object_spans(text))


def _span_is_internal(text: str, span: tuple[int, int]) -> bool:
    """Whether the ``(start, end)`` span is a parseable internal document."""
    start, end = span
    if end - start > _MAX_SPAN_CHARS:
        return False
    try:
        return _is_internal_object(json.loads(text[start:end]))
    except (json.JSONDecodeError, ValueError, RecursionError):
        return False


def extract_response_text(text: str) -> str | None:
    """Recover the natural-language answer from an echoed routing document.

    Handles the case where a model put its *whole* JSON object inside
    ``response_text``: if the object carries its own ``response_text`` (or a
    single ``message``/``answer`` string), that inner value is returned.  Only
    one level deep is unwrapped, and the result is re-validated, so a document
    cannot smuggle another document through.

    Returns ``None`` when no recoverable answer exists.
    """
    candidate = strip_code_fence(text)
    payload: Any = None
    parsed = False
    if len(candidate) <= _MAX_SPAN_CHARS:
        try:
            payload = json.loads(candidate)
            parsed = True
        except (json.JSONDecodeError, ValueError, RecursionError):
            parsed = False
    if not parsed:
        for span in _iter_object_spans(text):
            start, end = span
            if end - start > _MAX_SPAN_CHARS:
                continue
            try:
                payload = json.loads(text[start:end])
            except (json.JSONDecodeError, ValueError, RecursionError):
                continue
            if _is_internal_object(payload):
                parsed = True
                break
    if not parsed or not isinstance(payload, dict):
        return None

    for key in ("response_text", "message", "answer", "text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip() and not looks_like_internal_payload(value):
            return value.strip()
    return None


def safe_final_answer(text: str, *, fallback: str = NO_ANSWER_MESSAGE) -> str:
    """Return ``text`` with any leaked internal JSON spans removed.

    Parses each balanced ``{...}`` span and drops the ones that are internal
    routing documents, keeping everything else — so a natural sentence
    surrounding an echoed schema survives intact.  If nothing natural is left,
    ``fallback`` is returned rather than machine output.
    """
    if not text or not text.strip():
        return fallback
    if not looks_like_internal_payload(text):
        return text

    recovered = extract_response_text(text)
    if recovered and not looks_like_internal_payload(recovered):
        return recovered

    candidate = strip_code_fence(text)
    if not _is_internal_object(_try_json(candidate)):
        kept = _remove_internal_spans(candidate)
        if kept.strip():
            return kept
    return fallback


def _try_json(text: str) -> Any:
    """``json.loads`` that returns ``None`` instead of raising."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError, RecursionError):
        return None


def _remove_internal_spans(text: str) -> str:
    """Delete every internal-document span from ``text`` (parse-driven)."""
    spans = [span for span in _iter_object_spans(text) if _span_is_internal(text, span)]
    if not spans:
        return text
    out: list[str] = []
    cursor = 0
    for start, end in spans:
        out.append(text[cursor:start])
        cursor = end
    out.append(text[cursor:])
    cleaned = "".join(out)
    # Tidy the seams left by the removal so the sentence still reads naturally.
    for separator in (",", ";", ":"):
        cleaned = cleaned.replace(f"{separator} {separator}", separator)
    cleaned = cleaned.replace(" ,", ",").replace(" .", ".")
    return cleaned.strip()


#: Longest answer the voice channel will speak.  Reading a multi-source
#: web-search dump aloud produced 79 seconds of dead air; a voice answer is a
#: sentence, not a document.  The full text is still reported on the console
#: (truncated, with the remaining length) and kept in the task result.
SPOKEN_ANSWER_CHARS = 300


def spoken_answer(text: str, *, budget: int = SPOKEN_ANSWER_CHARS) -> str:
    """Return the part of ``text`` that is safe and sensible to speak aloud.

    Guarantees: no internal routing payload, no code fence, and at most
    ``budget`` characters, cut at a sentence boundary where one exists so the
    utterance does not stop mid-word.  An answer that reduces to nothing
    becomes the fixed honest refusal.
    """
    cleaned = safe_final_answer(text)
    if len(cleaned) <= budget:
        return cleaned
    window = cleaned[:budget]
    for terminator in (". ", "! ", "? ", ".\n", "!\n", "?\n"):
        cut = window.rfind(terminator)
        if cut >= budget // 2:
            return window[: cut + 1].strip()
    return window.rstrip() + "…"


def describe_violation(text: str) -> str:
    """Explain what was wrong, for the structured-output repair prompt.

    Names the leaked keys so the model is told exactly what to fix.  Safe: it
    is derived from key *names* only, never from values.
    """
    candidate = strip_code_fence(text)
    keys: set[str] = set()
    for payload in (
        _try_json(candidate),
        *(_try_json(text[s:e]) for s, e in _iter_object_spans(text)),
    ):
        if isinstance(payload, dict):
            keys |= INTERNAL_KEYS & set(payload)
    if keys:
        return (
            "response_text must contain only the natural-language answer for the human. "
            "You put internal routing fields ("
            + ", ".join(sorted(keys))
            + ") into it. Re-send the JSON with response_text set to the plain sentence "
            "you want spoken, and nothing else."
        )
    return (
        "response_text must contain only the natural-language answer for the human, "
        "not JSON, code fences, or internal routing fields."
    )
