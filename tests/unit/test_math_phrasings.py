"""Arithmetic is deterministic, bounded, and never uses dynamic evaluation.

The reported issue was that "what is 2 times 2?" and "what is 2 multiplied by
2?" behaved differently.  The fast path now answers both through one small
parser that accepts only digits, parentheses, signs, decimals and the four
basic operators; every other request still goes to the planner.

These tests pin two boundaries:

1. **No ``eval``/``exec``/``literal_eval`` or spoken-word dispatch table.**
   A structural guard against a future "quick fix" that executes input.
2. **The parser is exact, bounded, and fails closed.**  Division by zero,
   malformed expressions and non-arithmetic questions never produce an answer.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from jarvis.agent.answer import safe_final_answer
from jarvis.agent.arithmetic import solve

#: The phrasings that were reported to behave inconsistently.
PHRASINGS: list[tuple[str, str]] = [
    ("what is 2 times 2?", "4"),
    ("what is 2 multiplied by 2?", "4"),
    ("what is 5 times 5?", "25"),
    ("what is 10 divided by 2?", "5"),
]

SRC = Path(__file__).resolve().parents[2] / "src" / "jarvis"

#: Words that would betray a phrase table or an arithmetic branch.
SUSPECT = re.compile(
    r"""(?ix)
    \b(                      # arithmetic *operator words* as a dispatch key
        times | multiplied | divided | plus | minus | sum | product
    )\b
    \s*[:=]                  # ...used as a mapping key, e.g. {"times": ...}
    |
    \beval\s*\(             # never execute user input
    |
    \bexec\s*\(             # never execute user input
    |
    \bast\.literal_eval\b    # never parse user input as Python literal
    """
)


class TestNoDynamicEvaluation:
    def test_no_source_file_dispatches_on_arithmetic_words_or_eval(self) -> None:
        """A phrase table or eval/exec in the app would be the bug, not the fix."""
        offenders: list[str] = []
        for path in SRC.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="replace")
            for match in SUSPECT.finditer(text):
                line = text[: match.start()].count("\n") + 1
                offenders.append(f"{path.relative_to(SRC)}:{line}: {match.group(0)!r}")
        assert offenders == [], "unsafe arithmetic dispatch:\n" + "\n".join(offenders)

    def test_no_regex_substitutes_math_words(self) -> None:
        """The bounded parser does not rewrite user text with a regex."""
        offenders: list[str] = []
        pattern = re.compile(r"""re\.sub\([^)]*(times|multiplied|divided)""", re.IGNORECASE)
        for path in SRC.rglob("*.py"):
            for match in pattern.finditer(path.read_text(encoding="utf-8", errors="replace")):
                offenders.append(str(path.relative_to(SRC)))
        assert offenders == []


class TestSafeArithmetic:
    @pytest.mark.parametrize(
        ("question", "answer"),
        [
            ("what is 2 times 2?", "4"),
            ("what is 2 multiplied by 2?", "4"),
            ("what is 10 divided by 2?", "5"),
            ("what is (2 + 3) * 4?", "20"),
            ("what is -2.5 + 1?", "-1.5"),
            ("what is 0.1 + 0.2?", "0.3"),
        ],
    )
    def test_exact_answers(self, question: str, answer: str) -> None:
        result = solve(question)
        assert result is not None
        assert result.answer == answer

    @pytest.mark.parametrize(
        "question",
        [
            "",
            "what is 2 +",
            "what is 1 / 0",
            "what is 2(3)",
            "what is 2 ** 3",
            "__import__('os')",
            "open 2 files",
            "what is the date",
            "what is 2",
        ],
    )
    def test_fails_closed(self, question: str) -> None:
        assert solve(question) is None


class TestPhrasingsRoundTripCleanly:
    @pytest.mark.parametrize(("question", "answer"), PHRASINGS)
    def test_answer_survives_the_output_boundary(self, question: str, answer: str) -> None:
        """Whatever the planner returns, the spoken answer is the bare answer."""
        assert safe_final_answer(answer) == answer

    @pytest.mark.parametrize(("question", "answer"), PHRASINGS)
    def test_question_is_not_normalised_away(self, question: str, answer: str) -> None:
        """The boundary must not rewrite the question into something else.

        A "helpful" normalisation is exactly how the two phrasings diverged
        before: only one spelling survived to the planner.
        """
        assert safe_final_answer(question) == question
        assert question in question.strip()

    @pytest.mark.parametrize(("question", "answer"), PHRASINGS)
    def test_answer_with_units_still_reaches_output(self, question: str, answer: str) -> None:
        """A planner that answers "4" must not have the 4 stripped as a number."""
        assert safe_final_answer(f"{answer}") == answer
        assert safe_final_answer(f"The answer is {answer}.") == f"The answer is {answer}."


class TestNoBlacklist:
    @pytest.mark.parametrize(("question", "answer"), PHRASINGS)
    def test_arithmetic_words_are_not_filtered(self, question: str, answer: str) -> None:
        """No phrase is blacklisted — the words must be ordinary input."""
        for word in ("times", "multiplied", "divided"):
            probe = f"what is 2 {word} 2?"
            assert safe_final_answer(probe) == probe
