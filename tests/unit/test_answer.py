"""Tests for the internal-payload answer boundary.

The reported bug was a correct sentence followed by ``goal``/``actions``/
``response_hint``/``conversation_context``: a model that echoed its own JSON
template into ``response_text``.  These tests pin the two layers that prevent it
(:func:`looks_like_internal_payload` as a validator and :func:`safe_final_answer`
as defence in depth) and prove the detection is a *parse*, not a keyword
pattern — prose mentioning "goal" or "actions" must survive untouched.
"""

from __future__ import annotations

import json

import pytest

from jarvis.agent.answer import (
    INTERNAL_KEYS,
    NO_ANSWER_MESSAGE,
    describe_violation,
    extract_response_text,
    looks_like_internal_payload,
    safe_final_answer,
    spoken_answer,
    strip_code_fence,
)

# The exact reported symptom: a good sentence followed by the echoed schema.
LEAKED = "The capital of France is Paris. " + json.dumps(
    {
        "goal": "state the capital of France",
        "actions": [{"tool": "web.search", "args": {"query": "capital of France"}}],
        "response_hint": "be brief",
    }
)


class TestDetection:
    def test_detects_reported_leak(self) -> None:
        assert looks_like_internal_payload(LEAKED) is True

    def test_detects_whole_value_is_the_document(self) -> None:
        assert looks_like_internal_payload('{"goal": "x", "actions": []}') is True

    def test_detects_fenced_document(self) -> None:
        assert looks_like_internal_payload('```json\n{"goal": "x", "actions": []}\n```') is True

    def test_clean_answer_is_not_flagged(self) -> None:
        assert looks_like_internal_payload("The capital of France is Paris.") is False

    @pytest.mark.parametrize(
        "prose",
        [
            "I will set a goal for you and then take the actions you asked for.",
            'She said "actions" and then left.',
            "My plan is: rest, then act.",
            "Use the --goal flag to change it.",
            "Tell me which steps you want.",
            "",
            "   ",
        ],
    )
    def test_prose_mentioning_internal_words_is_left_alone(self, prose: str) -> None:
        """Detection parses JSON; it never greps for key names in prose."""
        assert looks_like_internal_payload(prose) is False

    def test_single_internal_key_is_not_enough(self) -> None:
        """One coincidental key must not trigger the repair path."""
        assert looks_like_internal_payload('{"goal": "something"}') is False

    def test_unrelated_json_is_not_internal(self) -> None:
        assert looks_like_internal_payload('{"city": "Paris", "population": 2140000}') is False

    def test_malformed_json_is_not_flagged(self) -> None:
        """Unparseable braces are left alone rather than guessed at."""
        assert looks_like_internal_payload('{"goal": "x", "actions": [') is False

    def test_braces_inside_strings_do_not_unbalance(self) -> None:
        text = 'He said "goal and actions" to me, then {"goal": "x", "actions": []}.'
        assert looks_like_internal_payload(text) is True

    def test_response_text_key_counts_as_internal(self) -> None:
        """Regression: the leaked key list must include response_text itself."""
        assert "response_text" in INTERNAL_KEYS
        assert looks_like_internal_payload('{"response_text": "hi", "goal": "g"}') is True


class TestSafeFinalAnswer:
    def test_keeps_the_sentence_and_drops_the_schema(self) -> None:
        result = safe_final_answer(LEAKED)
        assert result == "The capital of France is Paris."

    def test_pure_machine_output_becomes_honest_refusal(self) -> None:
        raw = json.dumps({"goal": "g", "actions": [], "response_hint": "h"})
        assert safe_final_answer(raw) == NO_ANSWER_MESSAGE

    def test_recovers_answer_embedded_in_response_text_key(self) -> None:
        raw = json.dumps({"goal": "g", "actions": [], "response_text": "4 is the answer."})
        assert safe_final_answer(raw) == "4 is the answer."

    def test_clean_answer_passes_through_untouched(self) -> None:
        text = "Opening Notepad now."
        assert safe_final_answer(text) is text

    def test_empty_answer_uses_fallback(self) -> None:
        assert safe_final_answer("") == NO_ANSWER_MESSAGE
        assert safe_final_answer("   ", fallback="nothing") == "nothing"

    def test_custom_fallback_is_honoured(self) -> None:
        raw = json.dumps({"goal": "g", "actions": []})
        assert safe_final_answer(raw, fallback="sorry") == "sorry"

    def test_does_not_smuggle_a_second_document(self) -> None:
        """One level of unwrapping only: a nested document is not recovered."""
        inner = json.dumps({"goal": "g", "actions": []})
        raw = json.dumps({"goal": "outer", "actions": [], "response_text": inner})
        assert safe_final_answer(raw) == NO_ANSWER_MESSAGE


class TestExtractResponseText:
    def test_extracts_from_full_document(self) -> None:
        raw = json.dumps({"goal": "g", "actions": [], "response_text": "Hello."})
        assert extract_response_text(raw) == "Hello."

    def test_returns_none_for_prose(self) -> None:
        assert extract_response_text("Just a sentence.") is None

    def test_returns_none_when_no_answer_key(self) -> None:
        assert extract_response_text(json.dumps({"goal": "g", "actions": []})) is None


class TestSpokenAnswer:
    def test_short_answer_is_spoken_whole(self) -> None:
        assert spoken_answer("4.") == "4."

    def test_never_speaks_internal_payload(self) -> None:
        assert "goal" not in spoken_answer(LEAKED)

    def test_cuts_at_a_sentence_boundary(self) -> None:
        text = "First sentence here. " * 40
        out = spoken_answer(text, budget=60)
        assert len(out) <= 60
        assert out.endswith(".")

    def test_hard_cut_ends_with_ellipsis_when_no_boundary(self) -> None:
        out = spoken_answer("x" * 500, budget=40)
        assert len(out) <= 41
        assert out.endswith("…")

    def test_reduces_to_refusal_when_nothing_natural_survives(self) -> None:
        raw = json.dumps({"goal": "g", "actions": [], "response_text": raw_placeholder()})
        assert spoken_answer(raw) == NO_ANSWER_MESSAGE


def raw_placeholder() -> str:
    """A second nested document, so the outer value has no natural text."""
    return json.dumps({"goal": "inner", "actions": []})


class TestStripCodeFence:
    def test_removes_json_fence(self) -> None:
        assert strip_code_fence('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_leaves_unfenced_text(self) -> None:
        assert strip_code_fence("  hi  ") == "hi"


class TestDescribeViolation:
    def test_names_the_leaked_keys(self) -> None:
        message = describe_violation(LEAKED)
        assert "goal" in message
        assert "actions" in message
        assert "response_hint" in message

    def test_never_echoes_values(self) -> None:
        """Only key names may appear; the values could contain user data."""
        raw = json.dumps({"goal": "secret-user-intent", "actions": [], "plan": "x"})
        assert "secret-user-intent" not in describe_violation(raw)

    def test_falls_back_to_generic_guidance(self) -> None:
        assert "natural-language" in describe_violation("just prose")
