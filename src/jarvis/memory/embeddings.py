"""Deterministic local text embeddings for memory retrieval (Phase 8).

Why this is *not* an ONNX/MiniLM model
-------------------------------------
``docs/02_ARCHITECTURE.md`` section 9 and the ``[ml]`` extra assume a local ONNX
encoder (``fastembed`` / ``sentence-transformers``).  Every such encoder
downloads model weights on first use, and JARVIS's own rules forbid a
dependency that downloads or executes code at runtime, forbid network access in
tests, and require audio/text to stay on the machine.  A model that silently
reaches out to the internet the first time a user says "delete that file twice
please" is not acceptable in this project.

So the default encoder is **lexical and offline**: a hashed bag of word tokens
and character trigrams, sublinear term frequency, L2-normalised, packed as
little-endian float32.  Consequences, stated honestly:

* It is strong on *repetition*, which is the acceptance criterion that matters
  ("repeating a command retrieves the earlier skill"): the same sentence always
  produces the same vector, and small edits (punctuation, case, extra filler
  words) barely move it.
* It is weaker on *paraphrase* than MiniLM: "delete the downloads folder" and
  "get rid of my downloads" share only character trigrams.
* No training, no weights, no network, no new dependency, and the "model" is
  ~30 lines anyone can audit before a viva.

:class:`Embedder` is a protocol so a real ONNX encoder can be dropped in later
without touching the stores; ``embed_text`` is the default implementation.
Stored blobs stay compatible because the dimension is recorded next to the
vector in the ``skills`` row (see :func:`from_bytes`).
"""

from __future__ import annotations

import hashlib
import math
import re
import struct
from collections.abc import Iterable, Sequence
from typing import Protocol

__all__ = [
    "DEFAULT_SIMILARITY",
    "EMBED_DIMS",
    "Embedder",
    "HashingEmbedder",
    "cosine",
    "embed_text",
    "from_bytes",
    "similarity",
    "to_bytes",
    "vector_size",
]

#: Dimensionality of the stored vectors.  256 float32 = 1 KiB per skill, which
#: is small enough for the < 10k skills the architecture targets (docs/02 §9)
#: while still discriminating on short command sentences.
EMBED_DIMS = 256

#: Default cosine floor from docs/02_ARCHITECTURE.md section 9 ("retrieval
#: threshold configurable (default 0.75)").  With the lexical encoder this is a
#: deliberately high bar: a wrong example in the planner's prompt costs an API
#: call and can mislead the plan, so we only surface near-duplicates.
DEFAULT_SIMILARITY = 0.75

_WORD_RE = re.compile(r"[a-z0-9]+")

#: Words that carry no retrieval signal but dominate a cosine score.
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "than",
        "that",
        "this",
        "these",
        "those",
        "of",
        "to",
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "doing",
        "have",
        "has",
        "had",
        "having",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "it",
        "its",
        "as",
        "so",
        "not",
        "no",
        "can",
        "could",
        "would",
        "should",
        "will",
        "shall",
        "may",
        "might",
        "must",
        "just",
        "please",
    ]
)


def _features(text: str) -> dict[str, float]:
    """Hash-ready weighted features for ``text`` (words + char trigrams)."""
    low = (text or "").lower()
    words = _WORD_RE.findall(low)
    counts: dict[str, float] = {}

    def add(feature: str, weight: float) -> None:
        counts[feature] = counts.get(feature, 0.0) + weight

    for word in words:
        if word in _STOPWORDS or len(word) < 2:
            continue
        add(f"w:{word}", 1.0)
        # Singular/plural folding helps "delete" match "deleting" a little
        # without a stemmer (no extra dependency, fully deterministic).
        if len(word) > 4 and word.endswith("s"):
            add(f"w:{word[:-1]}", 0.5)
        if len(word) > 5 and word.endswith("ing"):
            add(f"w:{word[:-3]}", 0.5)
        if len(word) > 4 and word.endswith("ed"):
            add(f"w:{word[:-2]}", 0.5)

    # Character trigrams give partial credit for typos and for morphological
    # variants that share no whole token.
    for i in range(len(low) - 2):
        chunk = low[i : i + 3]
        if chunk.strip():
            add(f"c:{chunk}", 0.35)
    return counts


def _bucket(feature: str, dims: int) -> tuple[int, float]:
    """Map a feature to a (bucket, sign) pair with a *stable* hash.

    ``hash()`` on ``str`` is salted per process (PYTHONHASHSEED), which would
    make a vector written by the daemon meaningless to the next CLI invocation,
    so a keyed BLAKE2b digest is used instead.  The sign trick keeps unrelated
    collisions from always adding energy in the same direction.
    """
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    return value % dims, 1.0 if (value >> 63) & 1 else -1.0


def embed_text(text: str, dims: int = EMBED_DIMS) -> list[float]:
    """Embed ``text`` into a unit-length vector of ``dims`` floats.

    Deterministic across processes and machines (no randomness, no clock, no
    dictionary ordering dependence).  Empty/blank text yields a zero vector,
    which is never "similar" to anything (all cosines are 0), so an empty
    example can never be retrieved.
    """
    if dims <= 0:
        raise ValueError("dims must be positive")
    vector = [0.0] * dims
    for feature, count in _features(text).items():
        weight = 1.0 + math.log(count) if count > 1.0 else count
        index, sign = _bucket(feature, dims)
        vector[index] += sign * weight
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0.0:
        return vector
    return [value / norm for value in vector]


def to_bytes(vector: Sequence[float]) -> bytes:
    """Pack a vector as little-endian float32 (the stored BLOB format)."""
    return struct.pack(f"<{len(vector)}f", *vector)


def from_bytes(blob: bytes) -> list[float]:
    """Unpack a stored BLOB; returns ``[]`` for anything unreadable.

    A truncated or corrupt row must not break retrieval of the *other* rows,
    so a bad blob degrades to "no embedding" rather than raising.
    """
    if not blob:
        return []
    count = len(blob) // 4
    if count == 0:
        return []
    try:
        return list(struct.unpack(f"<{count}f", blob[: count * 4]))
    except struct.error:
        return []


def vector_size(blob: bytes) -> int:
    """Dimensionality encoded in ``blob`` (0 when empty/corrupt)."""
    return len(from_bytes(blob))


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity in ``[-1, 1]``; ``0.0`` for empty/mismatched input.

    Length mismatch is tolerated (compare over the common prefix) so a vector
    written by a build with a different ``EMBED_DIMS`` degrades to a score
    rather than an exception.
    """
    size = min(len(left), len(right))
    if size == 0:
        return 0.0
    dot = 0.0
    for i in range(size):
        dot += left[i] * right[i]
    return max(-1.0, min(1.0, dot))


def similarity(
    left: Sequence[float], right: Sequence[float], floor: float = DEFAULT_SIMILARITY
) -> float:
    """Cosine similarity, or ``0.0`` when below ``floor``.

    One place decides what "related enough to show the planner" means, so the
    floor cannot be bypassed by a caller forgetting it.
    """
    score = cosine(left, right)
    return score if score >= floor else 0.0


class Embedder(Protocol):
    """Anything that turns text into a comparable vector."""

    def __call__(self, text: str) -> list[float]:  # pragma: no cover - protocol
        ...


class HashingEmbedder:
    """Callable wrapper around :func:`embed_text` (a stable ``Embedder``)."""

    def __init__(self, dims: int = EMBED_DIMS) -> None:
        self.dims = dims

    def __call__(self, text: str) -> list[float]:
        return embed_text(text, self.dims)

    def pack(self, text: str) -> bytes:
        return to_bytes(self(text))

    def unpack(self, blob: bytes) -> list[float]:
        return from_bytes(blob)


def best_matches(
    query: str,
    candidates: Iterable[tuple[str, Sequence[float]]],
    limit: int = 3,
    floor: float = DEFAULT_SIMILARITY,
) -> list[tuple[str, float]]:
    """Rank ``(key, vector)`` candidates against ``query``.

    Returns up to ``limit`` ``(key, score)`` pairs above ``floor``, best first,
    with ties broken by ``key`` so the result is deterministic.  Used by the
    skill store for dedup as well as by retrieval.
    """
    if limit <= 0:
        return []
    query_vector = embed_text(query)
    scored = [
        (key, score)
        for key, vector in candidates
        if (score := similarity(query_vector, vector, floor)) > 0.0
    ]
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored[:limit]
