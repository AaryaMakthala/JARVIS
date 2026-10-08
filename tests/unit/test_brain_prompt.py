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
from jarvis.agent.prompts import VOICE_SOURCE_LINE, brain_user_prompt
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


# ── wiring: state source reaches the LLM call, temperature unchanged ────────


class _RecorderLLM:
    """Records the exact user/system prompt handed to the LLM call."""

    def __init__(self, decision: BrainDecision) -> None:
        self._decision = decision
        self.user: str | None = None
        self.temperature: float | None = None

    def structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[Any],
        model_role: str = "planner",
        temperature: float = 0.0,
    ) -> tuple[BrainDecision, Usage]:
        assert issubclass(schema, BrainDecision)
        assert system
        self.user = user
        self.temperature = temperature
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
