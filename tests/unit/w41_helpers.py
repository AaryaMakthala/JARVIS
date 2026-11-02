"""One-line helpers used by the W4.1 tests so the fake-page assertions stay readable."""
from __future__ import annotations


def page_is_verified(page: object, phone: str) -> bool:
    """A tiny helper: the fake page's search query was the trusted phone number."""
    return getattr(page, "searched", None) == [phone]
