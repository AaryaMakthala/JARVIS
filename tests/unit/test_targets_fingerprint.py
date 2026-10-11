from __future__ import annotations

import dataclasses
import re

import pytest

from jarvis.policy.targets import TARGET_KINDS, ResolvedTarget, fingerprint


def _browser(**over: object) -> ResolvedTarget:
    base: dict[str, object] = {
        "kind": "browser",
        "ref": "backend-node-17",
        "name": "Send message",
        "role": "button",
        "origin": "https://web.whatsapp.com",
    }
    base.update(over)
    return ResolvedTarget(**base)  # type: ignore[arg-type]


def _uia(**over: object) -> ResolvedTarget:
    base: dict[str, object] = {
        "kind": "uia",
        "ref": "runtime-42",
        "name": "Save",
        "role": "button",
    }
    base.update(over)
    return ResolvedTarget(**base)  # type: ignore[arg-type]


def test_fingerprint_is_deterministic_hex() -> None:
    fp = fingerprint(_browser())
    assert fp is not None and re.fullmatch(r"[0-9a-f]{64}", fp)
    assert _browser().fingerprint() == fp
    assert fingerprint(_browser()) == fp


def test_fingerprint_independent_of_whitespace_and_case_noise() -> None:
    a = _browser(name="  Send   message ", role="BUTTON", origin="https://Web.WhatsApp.com")
    assert fingerprint(a) == fingerprint(_browser())


def test_identity_changes_change_fingerprint() -> None:
    base = fingerprint(_browser())
    assert fingerprint(_browser(ref="backend-node-18")) != base
    assert fingerprint(_browser(name="Send file")) != base
    assert fingerprint(_browser(role="link")) != base
    assert fingerprint(_browser(origin="https://evil.example")) != base


def test_kind_changes_fingerprint() -> None:
    assert fingerprint(_browser()) != fingerprint(_uia())


def test_uia_ignores_origin() -> None:
    assert fingerprint(_uia()) == fingerprint(_uia(origin="https://ignored.example"))


def test_browser_origin_canonicalised() -> None:
    base = fingerprint(_browser(origin="https://web.whatsapp.com"))
    assert base is not None
    for equivalent in (
        "https://WEB.WhatsApp.com",
        "https://web.whatsapp.com:443",
        "https://web.whatsapp.com.",
        "https://WEB.WhatsApp.com.:443",
    ):
        assert fingerprint(_browser(origin=equivalent)) == base, equivalent


def test_browser_non_default_port_is_identity_significant() -> None:
    assert fingerprint(_browser(origin="https://web.whatsapp.com:8443")) != fingerprint(
        _browser(origin="https://web.whatsapp.com")
    )


def test_browser_idn_origin_canonicalised() -> None:
    unicode_fp = fingerprint(_browser(origin="http://B\u00dcCHER.de"))
    assert unicode_fp is not None
    assert unicode_fp == fingerprint(_browser(origin="http://xn--bcher-kva.de"))


@pytest.mark.parametrize(
    "bad_origin",
    [
        "",
        "   ",
        "ftp://web.whatsapp.com",
        "javascript:alert(1)",
        "web.whatsapp.com",
        "not-a-url",
        "https://user@web.whatsapp.com",
        "http://exa mple.com",
        "https://web.whatsapp.com:99999",
    ],
)
def test_invalid_browser_origin_rejected(bad_origin: str) -> None:
    assert fingerprint(_browser(origin=bad_origin)) is None


def test_password_metadata_is_part_of_identity() -> None:
    plain = _browser(name="Password", role="textbox")
    secret = _browser(name="Password", role="textbox", input_type="password", is_password=True)
    assert fingerprint(secret) != fingerprint(plain)
    assert fingerprint(_uia(role="edit", is_password=True)) != fingerprint(_uia(role="edit"))


@pytest.mark.parametrize(
    "target",
    [
        _browser(kind="desktop"),
        _browser(kind=""),
        _browser(ref=""),
        _browser(ref="   "),
        _browser(name=""),
        _browser(name="  \t "),
        _browser(role=""),
        _browser(origin=""),
    ],
)
def test_missing_or_ambiguous_identity_rejected(target: ResolvedTarget) -> None:
    assert fingerprint(target) is None


def test_unknown_kind_rejected() -> None:
    assert fingerprint(ResolvedTarget(kind="wat", ref="r", name="n", role="button")) is None


def test_non_target_input_rejected() -> None:
    assert fingerprint(object()) is None  # type: ignore[arg-type]


def test_identity_dataclass_excludes_sensitive_fields() -> None:
    names = {f.name for f in dataclasses.fields(ResolvedTarget)}
    assert names == {"kind", "ref", "name", "role", "origin", "input_type", "is_password"}
    for forbidden in ("text", "value", "password", "typed", "content", "body"):
        assert forbidden not in names


def test_target_kinds_declared() -> None:
    assert TARGET_KINDS == frozenset({"browser", "uia"})
