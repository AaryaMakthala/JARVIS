"""brain user-prompt tests: the voice/terminal source line (Stage 5a sub-step 7).

The brain prompt is the only place ``state["source"]`` reaches the LLM.  A voice
transcript is lossy, so the prompt tells the brain to prefer ONE clarification
over a guess - but only when a detail is *genuinely* missing/unclear, and typed
input must get no such line.  These tests pin prompt *content* (there is no live
model in unit tests): the line exists for voice, not for terminal/benchmark,
never mentions policy machinery, and its wording cannot turn a fully specified
request into a clarification.
"""

from __future__ import annotations

from typing import Any

from jarvis.agent.context import make_app_context
from jarvis.agent.nodes.brain import brain
from jarvis.agent.prompts import VOICE_SOURCE_LINE, brain_system_prompt, brain_user_prompt
from jarvis.agent.schemas import BrainDecision
from jarvis.config import Settings
from jarvis.llm.client import Usage
from support import brain_conversation, make_spec, registry_with

_NO_LOOSE_WORDS = ("tier", "policy", "permission", "confirm", "unlock")


# ── prompt content: voice gets the line, typed input does not ───────────────


def test_voice_source_appends_the_transcript_line() -> None:
    prompt = brain_user_prompt("open notepad", source="voice")
    assert VOICE_SOURCE_LINE in prompt
    # it lands before the closing instruction and after the user's own words
    assert prompt.index(VOICE_SOURCE_LINE) < prompt.index("Return the BrainDecision JSON only.")
    assert prompt.index("The user said:") < prompt.index(VOICE_SOURCE_LINE)


def test_terminal_and_benchmark_source_never_get_the_line() -> None:
    for source in ("terminal", "benchmark"):
        prompt = brain_user_prompt("open notepad", source=source)
        assert VOICE_SOURCE_LINE not in prompt
        assert "speech transcript" not in prompt
        assert "recognition errors" not in prompt
        assert "never over-ask" not in prompt


def test_default_source_is_terminal_not_a_transcript() -> None:
    assert "speech transcript" not in brain_user_prompt("open notepad")


# ── the line itself is policy-free ──────────────────────────────────────────


def test_voice_line_never_mentions_policy_tiers() -> None:
    low = VOICE_SOURCE_LINE.lower()
    for word in _NO_LOOSE_WORDS:
        assert word not in low, word
    # the whole voice prompt is equally clean, not just the appended line
    prompt = brain_user_prompt("open notepad", source="voice").lower()
    for word in ("tier", "policy", "permission", "confirm", "unlock"):
        assert word not in prompt, word


# ── wording cannot force a clarification for a fully specified request ──────


def test_voice_line_requires_real_ambiguity_before_asking() -> None:
    prompt = brain_user_prompt("open notepad", source="voice")
    # the clarification licence is scoped to genuinely missing/unclear details…
    assert "genuinely missing" in prompt
    assert "too unclear to act on safely" in prompt
    # …and never over-asks a fully specified request
    assert "never over-ask a fully specified request" in prompt
    # nothing invites a question when the request is complete
    assert "always ask" not in prompt
    assert "if in doubt" not in prompt


# ── completeness rules (ROADMAP V2.2 Stage 5a): prompt content ──────────────


def test_system_prompt_has_the_four_completeness_outcomes() -> None:
    prompt = brain_system_prompt([{"name": "open_app", "description": "x", "args_schema": {}}])
    # values come only from the user's words, never from perception (I1)
    assert "Completeness" in prompt
    assert "Never invent a value" in prompt
    assert "screen, page, OCR" in prompt and "UIA text" in prompt
    # session references ("delete it") count as provided (D19)
    assert '"delete it"' in prompt and "count as provided" in prompt
    # outcome 1: required missing -> clarification
    assert "required detail is missing" in prompt and '"clarification"' in prompt
    # outcome 2: optional missing -> proceed + state the assumption
    assert "optional detail is missing" in prompt
    assert '"assumption"' in prompt and "stated to the user" in prompt
    # outcome 3: not executable -> plain spoken explanation, no loop
    assert '"unsupported"' in prompt and "no clarification loop" in prompt
    # outcome 4: policy-disallowed is never decided by the brain
    assert "policy" in prompt.lower() and "never decided here" in prompt
    # the schema template exposes the two new transient fields
    assert '"assumption": str' in prompt
    assert '"missing": [str]' in prompt


def test_fully_specified_user_prompt_never_invites_clarification() -> None:
    for source in ("terminal", "voice"):
        prompt = brain_user_prompt("open notepad", source=source)
        assert "always ask" not in prompt
        assert "if in doubt" not in prompt
        assert "you should ask" not in prompt


def test_clarification_count_round_line_appears_only_after_a_round() -> None:
    assert "Clarification round" not in brain_user_prompt("empty the folder")
    p1 = brain_user_prompt("empty the folder", clarification_count=1)
    assert "Clarification round 1 of up to 2." in p1
    p2 = brain_user_prompt("empty the folder", clarification_count=2)
    assert "This is clarification round 2 (the last one)." in p2
    assert "instead of asking again" in p2
    assert "'missing'" in p2  # the stop path points at the missing list


# ── wiring: state source reaches the LLM call, temperature unchanged ────────


class _RecorderLLM:
    """Records the exact user/system prompt handed to the LLM call."""

    def __init__(self, decision: BrainDecision) -> None:
        self._decision = decision
        self.user: str | None = None
        self.temperature: float | None = None
        self.max_retries: int | None = None
        self.timeout_s: float | None = None

    def structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[Any],
        model_role: str = "planner",
        temperature: float = 0.0,
        max_retries: int | None = None,
        timeout_s: float | None = None,
    ) -> tuple[BrainDecision, Usage]:
        assert issubclass(schema, BrainDecision)
        assert system
        self.user = user
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        return self._decision, Usage(calls=1, prompt_tokens=2, completion_tokens=2)


def _ctx(llm: _RecorderLLM) -> Any:
    record: list[tuple[str, dict[str, Any]]] = []
    return make_app_context(
        Settings(),
        llm=llm,
        registry=registry_with(make_spec("fake_echo", base_tier=0, record=record)),
    )


def test_brain_wires_voice_source_into_the_llm_call() -> None:
    recorder = _RecorderLLM(brain_conversation("hi back"))
    update = brain(
        {"user_input": "say hello", "source": "voice", "task_id": "t1"},
        _ctx(recorder),
    )
    assert recorder.user is not None
    assert VOICE_SOURCE_LINE in recorder.user
    assert recorder.temperature == 0.0  # speech never softens temperature
    assert update["request_kind"] == "conversation"


def test_brain_wires_terminal_source_without_the_line() -> None:
    recorder = _RecorderLLM(brain_conversation("hi back"))
    brain({"user_input": "say hello", "source": "terminal", "task_id": "t2"}, _ctx(recorder))
    assert recorder.user is not None
    assert VOICE_SOURCE_LINE not in recorder.user


def test_brain_wires_clarification_count_into_the_llm_call() -> None:
    recorder = _RecorderLLM(brain_conversation("hi back"))
    update = brain(
        {
            "user_input": "empty the folder",
            "source": "terminal",
            "clarification_count": 2,
            "task_id": "t3",
        },
        _ctx(recorder),
    )
    assert recorder.user is not None
    assert "This is clarification round 2 (the last one)." in recorder.user
    assert update["request_kind"] == "conversation"


def test_brain_voice_forwards_zero_retries_and_a_capped_timeout() -> None:
    """Speech gets no retry budget and at most 10s per attempt (Stage 5a sub-step 9)."""
    recorder = _RecorderLLM(brain_conversation("hi back"))
    brain({"user_input": "say hello", "source": "voice", "task_id": "t4"}, _ctx(recorder))
    assert recorder.max_retries == 0
    assert recorder.timeout_s == min(Settings().llm.timeout_seconds, 10.0)
    assert recorder.timeout_s == 10.0  # the default 30s is already capped


def test_brain_terminal_forwards_no_retry_budget() -> None:
    """Typed input keeps the configured retry budget and timeout (None)."""
    recorder = _RecorderLLM(brain_conversation("hi back"))
    brain({"user_input": "say hello", "source": "terminal", "task_id": "t5"}, _ctx(recorder))
    assert recorder.max_retries is None
    assert recorder.timeout_s is None
