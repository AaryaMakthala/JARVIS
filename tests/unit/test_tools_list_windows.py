"""list_windows tool tests (Tier 0, Windows-only, injectable enumerator).

``jarvis.tools.windows_list.enumerate_windows`` is replaced in every test, so
nothing here touches the real desktop.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from jarvis.agent.state import Step
from jarvis.config import Settings
from jarvis.policy import engine as engine_module
from jarvis.policy import rules
from jarvis.policy.engine import _WINDOWS_ONLY_REASON, PolicyContext, PolicyEngine
from jarvis.tools.base import ToolContext
from jarvis.tools.registry import _FORBIDDEN_RE, build_default_registry
from jarvis.tools.windows_list import (
    MAX_TITLE_CHARS,
    MAX_WINDOWS,
    ListWindowsArgs,
    make_list_windows_spec,
)


def _fake(monkeypatch: pytest.MonkeyPatch, windows: list[tuple[str, str, bool]]) -> None:
    """Replace the enumerator with a fixed list of (title, process, visible)."""
    monkeypatch.setattr("jarvis.tools.windows_list.enumerate_windows", lambda: list(windows))


def _run(*, dry_run: bool = False) -> Any:
    ctx = ToolContext(settings=Settings(), dry_run=dry_run)
    return make_list_windows_spec().run_verified(ListWindowsArgs(), ctx)


class TestOutput:
    def test_lists_titles_and_processes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake(
            monkeypatch,
            [("Untitled - Notepad", "notepad.exe", True), ("Inbox - Chrome", "chrome.exe", True)],
        )
        result = _run()

        assert result.ok is True
        assert result.verified is None
        assert result.data["count"] == 2
        assert result.data["windows"] == [
            {"title": "Untitled - Notepad", "process": "notepad.exe"},
            {"title": "Inbox - Chrome", "process": "chrome.exe"},
        ]
        assert "2 windows are open" in result.output

    def test_empty_desktop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake(monkeypatch, [])
        result = _run()

        assert result.ok is True
        assert result.verified is None
        assert result.data["count"] == 0
        assert result.data["windows"] == []
        assert "any open windows" in result.output

    def test_hidden_and_untitled_windows_are_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake(
            monkeypatch,
            [
                ("Visible", "chrome.exe", True),
                ("Hidden", "chrome.exe", False),
                ("   ", "other.exe", True),
                ("", "another.exe", True),
            ],
        )
        result = _run()

        assert result.data["windows"] == [{"title": "Visible", "process": "chrome.exe"}]

    def test_dedup_identical_title_and_process(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake(
            monkeypatch,
            [("Same", "a.exe", True), ("Same", "a.exe", True), ("same", "A.EXE", True)],
        )
        assert _run().data["count"] == 1

    def test_cap_at_max_windows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert MAX_WINDOWS == 25
        _fake(monkeypatch, [(f"Window {i}", "app.exe", True) for i in range(40)])

        result = _run()

        assert result.data["count"] == MAX_WINDOWS
        assert len(result.data["windows"]) == MAX_WINDOWS

    def test_titles_are_truncated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake(monkeypatch, [("T" * 200, "app.exe", True)])

        title = _run().data["windows"][0]["title"]

        assert len(title) == MAX_TITLE_CHARS
        assert title == "T" * MAX_TITLE_CHARS

    def test_process_is_a_name_never_a_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake(monkeypatch, [("Doc", "code.exe", True)])

        process = _run().data["windows"][0]["process"]

        assert process == "code.exe"
        assert "\\" not in process
        assert "/" not in process

    def test_result_is_tainted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Titles are attacker-influenced (a page title reaches the window)."""
        _fake(monkeypatch, [("Some page", "chrome.exe", True)])
        assert _run().tainted is True

    def test_single_window_sentence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake(monkeypatch, [("Notes", "notepad.exe", True)])
        assert _run().output == "One window is open: Notes."

    def test_spoken_form_is_bounded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake(monkeypatch, [(f"Window {i}", "app.exe", True) for i in range(30)])

        result = _run()

        assert len(result.output) <= 400
        assert "and 21 more" in result.output

    def test_dry_run_reads_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom() -> list[tuple[str, str, bool]]:
            raise AssertionError("dry run must not enumerate windows")

        monkeypatch.setattr("jarvis.tools.windows_list.enumerate_windows", _boom)

        result = _run(dry_run=True)

        assert result.ok is True
        assert "[dry-run]" in result.output


class TestSpec:
    def test_tier_and_platform(self) -> None:
        spec = make_list_windows_spec()
        assert spec.name == "list_windows"
        assert spec.base_tier == 0
        assert spec.windows_only is True

    def test_name_is_not_policy_blocked(self) -> None:
        spec = make_list_windows_spec()
        assert rules.matches_blocked(spec, ListWindowsArgs()) is False

    def test_description_has_no_forbidden_word(self) -> None:
        spec = make_list_windows_spec()
        assert _FORBIDDEN_RE.search(f"{spec.name} {spec.description}") is None

    def test_args_accept_empty_and_reject_extras(self) -> None:
        assert ListWindowsArgs() == ListWindowsArgs()
        with pytest.raises(ValidationError):
            ListWindowsArgs(what="windows")  # type: ignore[call-arg]

    def test_describe(self) -> None:
        spec = make_list_windows_spec()
        assert spec.describe(ListWindowsArgs()) == "list the open windows"


class TestPlatformGate:
    def _decide(self) -> Any:
        registry = build_default_registry(Settings())
        pctx = PolicyContext(registry=registry, settings=Settings())
        step = Step(id="s1", tool="list_windows", args={}, rationale="r")
        return PolicyEngine().decide(step, pctx)

    def test_refused_off_windows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine_module, "is_windows", lambda: False)

        decision = self._decide()

        assert decision.allowed is False
        assert decision.reasons == [_WINDOWS_ONLY_REASON]
        assert decision.needs_confirm is False

    def test_allowed_tier_0_on_windows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine_module, "is_windows", lambda: True)

        decision = self._decide()

        assert decision.allowed is True
        assert decision.tier == 0
        assert decision.needs_confirm is False
        assert decision.reasons == []


class TestRegistration:
    def test_registered_in_default_registry(self) -> None:
        registry = build_default_registry(Settings())
        assert "list_windows" in registry.names()
        assert registry.get("list_windows").base_tier == 0
