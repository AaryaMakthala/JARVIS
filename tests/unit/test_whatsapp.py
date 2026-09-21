"""WhatsApp Phase 6: contact store, number validation, rate limit, tool flow.

The WhatsApp tool only ever talks to a :class:`DesktopWindowProvider`, so every
behavioural test injects a fake provider (docs/06 Phase 6).  No pywinauto /
Windows desktop is touched.
"""

from __future__ import annotations

import pytest

from jarvis import config
from jarvis.tools.base import ToolContext
from jarvis.tools.registry import build_default_registry
from jarvis.tools.whatsapp import (
    ContactStore,
    RateLimiter,
    RateLimitError,
    WhatsAppSendArgs,
    make_whatsapp_send_spec,
    masked_number,
    normalize_number,
)

# ── fake window / provider ───────────────────────────────────────────────


class FakeWindow:
    """Stands in for the WhatsApp chat window during tests."""

    def __init__(
        self,
        *,
        header: str | None = "Rahul",
        draft: str | None = None,
        confirmed: bool | None = True,
    ) -> None:
        self.header = header
        self.draft = draft
        self.confirmed = confirmed
        self.entered = False

    def chat_header(self) -> str | None:
        return self.header

    def draft_text(self) -> str | None:
        return self.draft

    def press_enter(self) -> None:
        self.entered = True

    def message_visible(self, message: str) -> bool | None:
        return self.confirmed


class FakeProvider:
    """Stands in for the desktop provider during tests."""

    def __init__(self, window: FakeWindow | None = None) -> None:
        self.window = window
        self.opened: list[tuple[str, str]] = []

    def open_chat(self, phone_digits: str, text: str) -> None:
        self.opened.append((phone_digits, text))

    def find_chat_window(self, timeout_s: float) -> FakeWindow | None:
        return self.window


def _ctx() -> ToolContext:
    return ToolContext(settings=config.Settings())


def _patch_desktop(monkeypatch: pytest.MonkeyPatch, window: FakeWindow | None):
    monkeypatch.setattr("jarvis.platform_guard.is_windows", lambda: True)
    monkeypatch.setattr(
        "jarvis.tools.whatsapp._build_provider", lambda settings: FakeProvider(window)
    )


# ── number normalisation and masking ─────────────────────────────────────


class TestNormalizeNumber:
    def test_plain_digits(self) -> None:
        assert normalize_number("919876543210") == "919876543210"

    def test_leading_plus_stripped(self) -> None:
        assert normalize_number("+919876543210") == "919876543210"

    def test_separators_stripped(self) -> None:
        assert normalize_number("+91 98765-43210") == "919876543210"

    def test_minimum_length(self) -> None:
        assert normalize_number("12345678") == "12345678"

    def test_too_short_rejected(self) -> None:
        with pytest.raises(ValueError, match="8-15 digits"):
            normalize_number("1234567")

    def test_too_long_rejected(self) -> None:
        with pytest.raises(ValueError, match="8-15 digits"):
            normalize_number("+1234567890123456")

    def test_junk_rejected(self) -> None:
        with pytest.raises(ValueError, match="8-15 digits"):
            normalize_number("hello world")

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="8-15 digits"):
            normalize_number("   ")


class TestMaskedNumber:
    def test_last_two_digits_only(self) -> None:
        assert masked_number("+919876543210") == "+••••••••••10"

    def test_short_number(self) -> None:
        assert masked_number("12345678").endswith("78")


# ── contact store ────────────────────────────────────────────────────────


def make_store(tmp_path: pytest.TempPathFactory) -> ContactStore:
    return ContactStore(tmp_path / "contacts.json")


class TestContactStore:
    def test_add_and_reload(self, tmp_path) -> None:
        store = make_store(tmp_path)
        store.add("Rahul", "+919876543210")
        contacts = store.load()
        assert len(contacts) == 1
        assert contacts[0].name == "Rahul"
        assert contacts[0].number == "919876543210"

    def test_duplicate_name_rejected(self, tmp_path) -> None:
        store = make_store(tmp_path)
        store.add("Rahul", "+919876543210")
        with pytest.raises(ValueError, match="already exists"):
            store.add("rahul", "+919876543210")

    def test_duplicate_number_rejected(self, tmp_path) -> None:
        store = make_store(tmp_path)
        store.add("Rahul", "+919876543210")
        with pytest.raises(ValueError, match="another contact"):
            store.add("Sam", "+919876543210")

    def test_remove_unknown_raises(self, tmp_path) -> None:
        store = make_store(tmp_path)
        with pytest.raises(ValueError, match="no contact"):
            store.remove("nobody")

    def test_remove_and_reload(self, tmp_path) -> None:
        store = make_store(tmp_path)
        store.add("Rahul", "+919876543210")
        store.remove("rahul")
        assert store.load() == []

    def test_resolve_exact_then_substring(self, tmp_path) -> None:
        store = make_store(tmp_path)
        store.add("Rahul", "+919876543210")
        store.add("Sara", "+9112345678")
        assert [c.name for c in store.resolve("rahul")] == ["Rahul"]
        assert [c.name for c in store.resolve("rah")] == ["Rahul"]
        assert [c.name for c in store.resolve("ra")] == ["Rahul", "Sara"]
        assert [c.name for c in store.resolve("zzz")] == []

    def test_corrupt_file_treated_as_empty(self, tmp_path) -> None:
        store = make_store(tmp_path)
        store.add("Rahul", "+919876543210")
        store.path.write_text("not json", encoding="utf-8")
        assert store.load() == []


# ── rate limiter ─────────────────────────────────────────────────────────


class TestRateLimiter:
    def test_hourly_cap(self, tmp_path) -> None:
        limiter = RateLimiter(tmp_path / "rl.json", max_per_hour=1, min_interval_s=0.0)
        limiter.record()
        with pytest.raises(RateLimitError, match="per hour"):
            limiter.check()

    def test_min_interval(self, tmp_path) -> None:
        limiter = RateLimiter(tmp_path / "rl.json", max_per_hour=10, min_interval_s=5.0)
        limiter.record()
        with pytest.raises(RateLimitError, match="between sends"):
            limiter.check()

    def test_persists_across_instances(self, tmp_path) -> None:
        path = tmp_path / "rl.json"
        RateLimiter(path, max_per_hour=10, min_interval_s=0.0).record()
        limiter = RateLimiter(path, max_per_hour=10, min_interval_s=0.0, clock=lambda: 9999999999.0)
        limiter.check()  # old records age out past the pruned window


# ── the tool ─────────────────────────────────────────────────────────────


def build_spec(tmp_path, *, window: FakeWindow | None = None, **kwargs):
    store = ContactStore(tmp_path / "contacts.json")
    store.add("Rahul", "+919876543210")
    limiter = RateLimiter(tmp_path / "rl.json", max_per_hour=10, min_interval_s=0.0)
    return make_whatsapp_send_spec(store=store, limiter=limiter), store, limiter


class TestWhatsAppSend:
    def test_send_happy_path(self, tmp_path, monkeypatch) -> None:
        _patch_desktop(monkeypatch, FakeWindow(draft="hello rahul", confirmed=True))
        spec, _, _ = build_spec(tmp_path)
        ctx = _ctx()
        result = spec.run_verified(WhatsAppSendArgs(contact="Rahul", message="hello"), ctx)
        assert result.ok, result.error
        assert "Rahul" in result.output
        assert result.data["chat_confirmed"] is True
        assert result.verified is True  # verify() attaches the read-back result

    def test_unknown_contact_refused(self, tmp_path, monkeypatch) -> None:
        _patch_desktop(monkeypatch, FakeWindow())
        spec, _, _ = build_spec(tmp_path)
        result = spec.run(WhatsAppSendArgs(contact="Nobody", message="hi"), ctx=_ctx())
        assert result.ok is False
        assert "no contact matches" in result.error
        assert "Known contacts" in result.error

    def test_ambiguous_contact_lists_options(self, tmp_path, monkeypatch) -> None:
        store = ContactStore(tmp_path / "contacts.json")
        store.add("Ali Khan", "+9112345678")
        store.add("Ali Ahmed", "+919876543210")
        _patch_desktop(monkeypatch, FakeWindow())
        spec = make_whatsapp_send_spec(
            store=store,
            limiter=RateLimiter(tmp_path / "a.json", max_per_hour=10, min_interval_s=0.0),
        )
        result = spec.run(WhatsAppSendArgs(contact="ali", message="hi"), ctx=_ctx())
        assert result.ok is False
        assert "multiple contacts match" in result.error
        assert "Ali Khan" in result.error and "Ali Ahmed" in result.error

    def test_message_over_1000_chars_rejected(self, tmp_path, monkeypatch) -> None:
        _patch_desktop(monkeypatch, FakeWindow())
        with pytest.raises(Exception, match="at most 1000"):
            WhatsAppSendArgs(contact="Rahul", message="x" * 1001)

    def test_rate_limit_refuses_before_open(self, tmp_path, monkeypatch) -> None:
        _patch_desktop(monkeypatch, FakeWindow(confirmed=True))
        store = ContactStore(tmp_path / "contacts.json")
        store.add("Rahul", "+919876543210")
        limiter = RateLimiter(tmp_path / "rl.json", max_per_hour=1, min_interval_s=0.0)
        limiter.record()
        spec = make_whatsapp_send_spec(store=store, limiter=limiter)
        result = spec.run(WhatsAppSendArgs(contact="Rahul", message="hi"), ctx=_ctx())
        assert result.ok is False
        assert "rate limit" in result.error
        assert limiter.traffic_count() == 1  # nothing new recorded

    def test_unverifiable_window_fails_closed(self, tmp_path, monkeypatch) -> None:
        # Invariant 9: an unverifiable WhatsApp window means nothing is sent.
        _patch_desktop(monkeypatch, window=None)
        spec, _, limiter = build_spec(tmp_path)
        result = spec.run(WhatsAppSendArgs(contact="Rahul", message="hello"), ctx=_ctx())
        assert result.ok is False
        assert "Could not verify chat" in result.error
        assert limiter.traffic_count() == 0

    def test_unverifiable_header_fails_closed(self, tmp_path, monkeypatch) -> None:
        # Window found but its UI text cannot be read -> still refuse to send.
        window = FakeWindow(header=None)
        _patch_desktop(monkeypatch, window)
        spec, _, limiter = build_spec(tmp_path)
        result = spec.run(WhatsAppSendArgs(contact="Rahul", message="hello"), ctx=_ctx())
        assert result.ok is False
        assert "Chat identity could not be verified" in result.error
        assert window.entered is False
        assert limiter.traffic_count() == 0

    def test_header_mismatch_never_presses_enter(self, tmp_path, monkeypatch) -> None:
        window = FakeWindow(header="Someone Else")
        _patch_desktop(monkeypatch, window)
        spec, _, limiter = build_spec(tmp_path)
        result = spec.run(WhatsAppSendArgs(contact="Rahul", message="hello"), ctx=_ctx())
        assert result.ok is False
        assert window.entered is False
        assert "Chat identity could not be verified" in result.error
        assert limiter.traffic_count() == 0

    def test_draft_mismatch_never_presses_enter(self, tmp_path, monkeypatch) -> None:
        window = FakeWindow(draft="something different")
        _patch_desktop(monkeypatch, window)
        spec, _, _ = build_spec(tmp_path)
        result = spec.run(WhatsAppSendArgs(contact="Rahul", message="hello"), ctx=_ctx())
        assert result.ok is False
        assert window.entered is False
        assert "Draft text does not match" in result.error

    def test_non_windows_refused(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("jarvis.platform_guard.is_windows", lambda: False)
        spec, _, _ = build_spec(tmp_path)
        result = spec.run(WhatsAppSendArgs(contact="Rahul", message="hi"), ctx=_ctx())
        assert result.ok is False
        assert "only supported on Windows" in result.error

    def test_describe_masks_number_and_shows_message(self, tmp_path) -> None:
        spec, _, _ = build_spec(tmp_path)
        summary = spec.describe(WhatsAppSendArgs(contact="Rahul", message="see you"))
        assert "Rahul" in summary
        assert "••••••••••10" in summary  # last two digits of 919876543210
        assert "'see you'" in summary

    def test_unreadable_draft_allows_send(self, tmp_path, monkeypatch) -> None:
        # Unknown/no edit control -> draft check is skipped, not failed.
        window = FakeWindow(draft=None, confirmed=True)
        _patch_desktop(monkeypatch, window)
        spec, _, _ = build_spec(tmp_path)
        result = spec.run(WhatsAppSendArgs(contact="Rahul", message="hello"), ctx=_ctx())
        assert result.ok, result.error
        assert window.entered is True


# ── registry + config wiring ─────────────────────────────────────────────


class TestWiring:
    def test_registered_in_default_registry(self) -> None:
        registry = build_default_registry()
        spec = registry.get_optional("whatsapp_send")
        assert spec is not None
        assert spec.base_tier == 2
        assert spec.windows_only is True
        assert "whatsapp_send" in registry.names()

    def test_spec_validation(self) -> None:
        spec = make_whatsapp_send_spec()
        assert spec.name == "whatsapp_send"
        assert spec.base_tier == 2
        args = spec.args_model.model_validate({"contact": "X", "message": "hi"})
        assert args.message == "hi"
        schema = spec.args_model.model_json_schema()
        assert schema["properties"]["message"]["maxLength"] == 1000

    def test_config_defaults(self) -> None:
        settings = config.Settings()
        assert settings.whatsapp.max_per_hour == 10
        assert settings.whatsapp.min_interval_s == 5.0
        assert settings.whatsapp.window_timeout_s == 10.0
        override = config.Settings(whatsapp={"max_per_hour": 3})
        assert override.whatsapp.max_per_hour == 3
        assert config.whatsapp_ratelimit_file().name == "whatsapp_ratelimit.json"
