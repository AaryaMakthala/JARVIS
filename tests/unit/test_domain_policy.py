"""Stage 7 pure policy primitives: origin, domain, risky controls, passwords.

These helpers are deterministic and side-effect-free; every case here drives
them directly (no engine, no registry, no I/O).  They back D18/B4/I5 (domain
and origin), D12/I8 (risky controls) and the password-field rule.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from jarvis.policy import rules

# ── origin canonicalization ───────────────────────────────────────────────


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://Example.COM./Path?q=1#f", "https://example.com"),
        ("http://example.com:80/", "http://example.com"),
        ("https://example.com:443", "https://example.com"),
        ("https://example.com:8443/x?y#z", "https://example.com:8443"),
        ("http://b\u00fccher.de/", "http://xn--bcher-kva.de"),
        ("http://B\u00dcCHER.de/", "http://xn--bcher-kva.de"),
        ("http://xn--bcher-kva.de/", "http://xn--bcher-kva.de"),
        ("http://127.0.0.1:5000/", "http://127.0.0.1:5000"),
        ("http://[::1]/", "http://[::1]"),
        ("http://[::1]:8080/", "http://[::1]:8080"),
        ("http://localhost:8000/", "http://localhost:8000"),
    ],
)
def test_url_origin_canonical(url: str, expected: str) -> None:
    assert rules.url_origin(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "https://user:pass@example.com/",
        "https://@example.com/",
        "https://example.com:99999/",
        "https://example.com:abc/",
        "https://example.com:/",
        "http://example.com:0/",
        "ftp://example.com/",
        "javascript:alert(1)",
        "data:text/html,<x>",
        "example.com",
        "//example.com",
        "",
        "http://exa mple.com",
        "http://evil.com\\@good.com",
        "https://example..com/",
    ],
)
def test_url_origin_rejects(url: str) -> None:
    assert rules.url_origin(url) is None


def test_url_origin_is_stable() -> None:
    """Different spellings of one origin canonicalize to the same string."""
    variants = [
        "HTTPS://Example.com/",
        "https://example.com:443",
        "https://EXAMPLE.com./path",
    ]
    origins = {rules.url_origin(u) for u in variants}
    assert origins == {"https://example.com"}


# ── domain authorization ──────────────────────────────────────────────────


def test_domain_allowed_exact_host() -> None:
    assert rules.domain_allowed("https://youtube.com/watch?v=x", ["youtube.com"]) is True
    assert rules.domain_allowed("https://YOUTUBE.com./", ["youtube.com"]) is True


def test_domain_allowed_rejects_subdomain_and_sibling() -> None:
    assert rules.domain_allowed("https://www.youtube.com/", ["youtube.com"]) is False
    assert rules.domain_allowed("https://music.youtube.com/", ["youtube.com"]) is False
    assert rules.domain_allowed("https://evil-youtube.com/", ["youtube.com"]) is False
    assert rules.domain_allowed("https://youtube.com.evil.com/", ["youtube.com"]) is False


def test_domain_allowed_idn_matches_alabel() -> None:
    assert rules.domain_allowed("https://xn--bcher-kva.de/", ["b\u00fccher.de"]) is True
    assert rules.domain_allowed("https://b\u00fccher.de/", ["xn--bcher-kva.de"]) is True


def test_domain_allowed_fails_closed() -> None:
    assert rules.domain_allowed("https://youtube.com/", []) is False
    assert rules.domain_allowed("not a url", ["youtube.com"]) is False
    assert rules.domain_allowed("https://user@youtube.com/", ["youtube.com"]) is False
    # A non-host allowlist entry can never match.
    assert rules.domain_allowed("https://youtube.com/", ["https://youtube.com"]) is False


def test_domain_allowed_localhost() -> None:
    assert rules.domain_allowed("http://localhost:8000/", ["localhost"]) is True
    assert rules.domain_allowed("http://127.0.0.1:8000/", ["127.0.0.1"]) is True


# ── risky control classification ──────────────────────────────────────────


@pytest.mark.parametrize(
    ("name", "role", "expected"),
    [
        ("Send", "button", rules.CONTROL_FRESH_CONFIRM),
        ("Send message", "button", rules.CONTROL_FRESH_CONFIRM),
        ("Submit", "", rules.CONTROL_FRESH_CONFIRM),
        ("Delete", "button", rules.CONTROL_FRESH_CONFIRM),
        ("Confirm", "", rules.CONTROL_FRESH_CONFIRM),
        ("Allow", "button", rules.CONTROL_FRESH_CONFIRM),
        ("Buy now", "button", rules.CONTROL_PAYMENT),
        ("Purchase", "link", rules.CONTROL_PAYMENT),
        ("Pay", "", rules.CONTROL_PAYMENT),
        ("Pay now", "button", rules.CONTROL_PAYMENT),
        ("Pay with card", "button", rules.CONTROL_PAYMENT),
        ("Payment", "button", rules.CONTROL_PAYMENT),
        ("Make Payment", "button", rules.CONTROL_PAYMENT),
        ("Payments", "link", rules.CONTROL_PAYMENT),
        ("Checkout", "button", None),
    ],
)
def test_risky_control_classification(name: str, role: str, expected: str | None) -> None:
    assert rules.risky_control(name, role) == expected


@pytest.mark.parametrize(
    ("name", "role"),
    [
        ("Sender", "button"),
        ("Allowance", ""),
        ("Confirmation code", "textbox"),
        ("All", "button"),
        ("Allowlist", ""),
        ("Buyer", ""),
        ("Repayment", ""),
        ("Payload", ""),
        ("Payer", "button"),
        ("Login", "button"),
        ("", ""),
    ],
)
def test_risky_control_false_positives(name: str, role: str) -> None:
    """Token matching must not fire on words that merely contain a fragment."""
    assert rules.risky_control(name, role) is None


# ── password-target detection ─────────────────────────────────────────────


def test_password_target_dom_and_uia() -> None:
    assert rules.is_password_target({"input_type": "password"}) is True
    assert rules.is_password_target({"role": "passwordfield"}) is True
    assert rules.is_password_target({"is_password": True}) is True
    assert rules.is_password_target(SimpleNamespace(control_type="Edit", is_password=True)) is True


def test_password_target_non_password() -> None:
    assert rules.is_password_target({"role": "textbox", "input_type": "text"}) is False
    assert rules.is_password_target({"is_password": False}) is False
    assert rules.is_password_target(SimpleNamespace(control_type="Button")) is False
    assert rules.is_password_target({"role": "button", "name": "Submit"}) is False


def test_password_target_fails_closed() -> None:
    assert rules.is_password_target(None) is True
    assert rules.is_password_target(object()) is True
    assert rules.is_password_target({"name": "Search"}) is True
