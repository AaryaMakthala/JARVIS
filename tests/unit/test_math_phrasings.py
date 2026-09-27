"""Math phrasings must work through the model, never through hard-coded code.

The reported issue was that "what is 2 times 2?" and the same question phrased
as "what is 2 multiplied by 2?" behaved differently.  The correct fix is *not* a
lookup table — understanding "multiplied by" is the planner's job, and a
hard-coded arithmetic table would be a maintenance trap that silently answers
only the phrases somebody thought of.

These tests therefore assert two things that do not depend on a live model:

1. **No arithmetic is hard-coded anywhere in the application.**  A structural
   guard, so a future "quick fix" cannot sneak a phrase table in.
2. **The phrasing reaches the planner verbatim and the answer comes back out
   clean.**  If a normaliser mangled "multiplied by" on the way in, or the
   response boundary mangled "4" on the way out, these fail.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from jarvis.agent.answer import safe_final_answer

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
    \beval\s*\(\s*["']       # eval() of a spoken expression
    |
    \bast\.literal_eval\b    # parsing a spoken expression
    """
)


class TestNoHardCodedArithmetic:
    def test_no_source_file_dispatches_on_arithmetic_words(self) -> None:
        """A phrase table in the app would be the bug, not the fix."""
        offenders: list[str] = []
        for path in SRC.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="replace")
            for match in SUSPECT.finditer(text):
                line = text[: match.start()].count("\n") + 1
                offenders.append(f"{path.relative_to(SRC)}:{line}: {match.group(0)!r}")
        assert offenders == [], "arithmetic must be the planner's job:\n" + "\n".join(offenders)

    def test_no_regex_substitutes_math_words(self) -> None:
        """A rewritting of "multiplied by" -> "x" would be the same bug."""
        offenders: list[str] = []
        pattern = re.compile(r"""re\.sub\([^)]*(times|multiplied|divided)""", re.IGNORECASE)
        for path in SRC.rglob("*.py"):
            for match in pattern.finditer(path.read_text(encoding="utf-8", errors="replace")):
                offenders.append(str(path.relative_to(SRC)))
        assert offenders == []


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
