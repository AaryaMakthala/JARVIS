"""Bounded, deterministic arithmetic for the local fast path.

This is deliberately a tiny calculator, not an expression evaluator: the input
is reduced to digits, parentheses, unary signs, decimals and the four basic
operators (plus a few spoken spellings), then parsed by a hand-written
recursive-descent parser.  There is no ``eval``, ``exec``, shell, or dynamic
import anywhere on this path, and non-arithmetic questions fall through
unchanged to the normal planner.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, DecimalException, localcontext

__all__ = ["ArithmeticAnswer", "solve"]

_PREFIXES: tuple[str, ...] = (
    "what is",
    "what's",
    "whats",
    "how much is",
    "calculate",
    "compute",
    "evaluate",
    "solve",
)

_WORD_OPERATORS: tuple[tuple[str, str], ...] = (
    ("multiplied by", "*"),
    ("divided by", "/"),
    ("times", "*"),
    ("plus", "+"),
    ("minus", "-"),
    ("over", "/"),
)

_ALLOWED = frozenset("0123456789.()+-*/ ")
_MAX_EXPRESSION_CHARS = 60
_MAX_RESULT_CHARS = 100


@dataclass(frozen=True, slots=True)
class ArithmeticAnswer:
    """The original normalised expression and its exact decimal result."""

    expression: str
    answer: str


def solve(raw: str) -> ArithmeticAnswer | None:
    """Return a deterministic answer for a pure arithmetic phrase, else None."""
    expression = _prepare(raw)
    if expression is None:
        return None
    try:
        with localcontext() as context:
            context.prec = 18
            value = _Parser(expression).parse()
    except (ArithmeticError, DecimalException, ValueError):
        return None
    if not value.is_finite():
        return None
    answer = _format(value)
    if len(answer) > _MAX_RESULT_CHARS:
        return None
    return ArithmeticAnswer(expression=expression, answer=answer)


def _prepare(raw: str) -> str | None:
    text = " ".join((raw or "").strip().lower().split())
    if len(text) > _MAX_EXPRESSION_CHARS:
        return None
    text = text.rstrip("?.!").strip()
    for prefix in _PREFIXES:
        if text.startswith(prefix + " "):
            text = text[len(prefix) :].strip()
            break
    for spoken, symbol in _WORD_OPERATORS:
        text = text.replace(" " + spoken + " ", " " + symbol + " ")
    text = text.replace(" x ", " * ")
    text = " ".join(text.split())
    if not text or not any(ch.isdigit() for ch in text):
        return None
    if any(ch not in _ALLOWED for ch in text):
        return None
    if not any(ch in "+-*/" for ch in text):
        return None
    return text


def _format(value: Decimal) -> str:
    if value == 0:
        return "0"
    text = format(value.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


class _Parser:
    """Recursive descent for ``expr := term (+|-) term`` and friends."""

    def __init__(self, expression: str) -> None:
        self._expression = expression
        self._position = 0

    def parse(self) -> Decimal:
        value = self._expression_value()
        self._skip_spaces()
        if self._position != len(self._expression):
            raise ValueError("trailing characters")
        return value

    def _expression_value(self) -> Decimal:
        value = self._term()
        while True:
            operator = self._take("+-")
            if operator is None:
                return value
            right = self._term()
            value = value + right if operator == "+" else value - right

    def _term(self) -> Decimal:
        value = self._factor()
        while True:
            operator = self._take("*/")
            if operator is None:
                return value
            right = self._factor()
            if operator == "*":
                value = value * right
            else:
                if right == 0:
                    raise ZeroDivisionError("division by zero")
                value = value / right

    def _factor(self) -> Decimal:
        sign = self._take("+-")
        if sign is not None:
            value = self._factor()
            return -value if sign == "-" else value
        if self._take("(") is not None:
            value = self._expression_value()
            if self._take(")") is None:
                raise ValueError("missing close parenthesis")
            return value
        return self._number()

    def _number(self) -> Decimal:
        self._skip_spaces()
        start = self._position
        seen_dot = False
        while self._position < len(self._expression):
            char = self._expression[self._position]
            if char.isdigit():
                self._position += 1
            elif char == "." and not seen_dot:
                seen_dot = True
                self._position += 1
            else:
                break
        token = self._expression[start : self._position]
        if not token or token == ".":
            raise ValueError("expected a number")
        return Decimal(token)

    def _take(self, characters: str) -> str | None:
        self._skip_spaces()
        if self._position >= len(self._expression):
            return None
        char = self._expression[self._position]
        if char not in characters:
            return None
        self._position += 1
        return char

    def _skip_spaces(self) -> None:
        while self._position < len(self._expression) and self._expression[self._position] == " ":
            self._position += 1
