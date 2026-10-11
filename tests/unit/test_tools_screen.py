"""Screen tools tests (Stage 6): ``read_screen`` (Tier 0) and ``describe_screen`` (Tier 1).

Every perception dependency is injected: ``jarvis.perception.screen``,
``jarvis.perception.ui_text`` and ``jarvis.perception.ocr`` are replaced, so
nothing here touches a desktop, a screenshot, a model or the network.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from jarvis.agent.state import Step
from jarvis.config import ScreenSettings, Settings
from jarvis.policy import engine as engine_module
from jarvis.policy import rules
from jarvis.policy.engine import _WINDOWS_ONLY_REASON, PolicyContext, PolicyEngine
from jarvis.tools.base import ToolContext
from jarvis.tools.registry import _FORBIDDEN_RE, build_default_registry
from jarvis.tools.screen import (
    DescribeScreenArgs,
    ReadScreenArgs,
    make_describe_screen_spec,
    make_read_screen_spec,
)

_PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 16


class _FakeLLM:
    """Records ``describe_image`` calls and returns a fixed answer."""

    def __init__(self, answer: str = "A news site with headlines.") -> None:
        self.answer = answer
        self.calls: list[dict[str, Any]] = []

    def describe_image(self, **kwargs: Any) -> tuple[str, None]:
        self.calls.append(kwargs)
        return self.answer, None


def _install(
    monkeypatch: pytest.MonkeyPatch,
    *,
    title: str = "Inbox - Chrome",
    process: str = "chrome.exe",
    uia: str = "Breaking news: the market rose today.",
    ocr_available: bool = False,
    ocr: str = "",
    capture_ok: bool = True,
) -> list[str]:
    """Install fakes; return a list recording capture calls."""
    from jarvis.perception.screen import ForegroundWindow
    from jarvis.perception.ui_text import UiaText

    captures: list[str] = []

    monkeypatch.setattr("jarvis.tools.screen.is_windows", lambda: True)
    monkeypatch.setattr(
        "jarvis.perception.screen.foreground_window",
        lambda: ForegroundWindow(title=title, process=process),
    )

    def _capture(**_kwargs: Any) -> bytes:
        if not capture_ok:
            raise RuntimeError("no display")
        captures.append("capture")
        return _PNG

    monkeypatch.setattr("jarvis.perception.screen.capture_foreground", _capture)
    monkeypatch.setattr(
        "jarvis.perception.ui_text.foreground_uia_text",
        lambda **_kw: UiaText(text=uia, controls=3, secure_controls=1),
    )
    monkeypatch.setattr("jarvis.perception.ocr.ocr_available", lambda: ocr_available)
    monkeypatch.setattr("jarvis.perception.ocr.ocr_image", lambda _b: ocr)
    return captures


def _run_read(ctx: ToolContext) -> Any:
    return make_read_screen_spec().run_verified(ReadScreenArgs(), ctx)


def _run_describe(ctx: ToolContext) -> Any:
    return make_describe_screen_spec().run_verified(DescribeScreenArgs(), ctx)


def _ctx(**kwargs: Any) -> ToolContext:
    return ToolContext(**kwargs)


class TestReadScreen:
    def test_returns_uia_text_and_is_tainted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install(monkeypatch, uia="Notepad document contents")

        result = _run_read(_ctx(settings=Settings()))

        assert result.ok is True
        assert result.verified is None
        assert result.tainted is True
        assert "Notepad document contents" in result.output
        assert "Inbox - Chrome" in result.output
        assert result.data["source"] == "uia"
        assert result.data["process"] == "chrome.exe"

    def test_combines_uia_and_local_ocr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install(monkeypatch, uia="Native text", ocr_available=True, ocr="Pixel text")

        result = _run_read(_ctx(settings=Settings()))

        assert result.data["source"] == "uia+ocr"
        assert "Native text" in result.output
        assert "Pixel text" in result.output

    def test_ocr_fallback_when_uia_is_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install(monkeypatch, uia="", ocr_available=True, ocr="Only OCR text")

        result = _run_read(_ctx(settings=Settings()))

        assert result.ok is True
        assert result.data["source"] == "ocr"
        assert "Only OCR text" in result.output

    def test_uia_survives_a_failed_capture(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A capture/OCR failure never discards usable UIA text."""
        _install(monkeypatch, uia="Usable native text", ocr_available=True, capture_ok=False)

        result = _run_read(_ctx(settings=Settings()))

        assert result.ok is True
        assert "Usable native text" in result.output
        assert result.data["capture_failed"] is True

    def test_no_text_reports_ocr_limitation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install(monkeypatch, uia="", ocr_available=False)

        result = _run_read(_ctx(settings=Settings()))

        assert result.ok is True
        assert result.tainted is True
        assert "Local OCR is not installed" in result.output

    def test_denylisted_process_refuses_before_capture(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captures = _install(monkeypatch, title="Vault", process="KeePassXC.exe")

        result = _run_read(_ctx(settings=Settings()))

        assert result.ok is False
        assert result.tainted is False
        assert captures == []  # nothing was captured
        assert "sensitive application" in (result.error or "")

    def test_denylisted_title_refuses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install(monkeypatch, title="Chase Bank - Online Banking", process="chrome.exe")

        result = _run_read(_ctx(settings=Settings()))

        assert result.ok is False
        assert "sensitive" in (result.error or "")

    def test_secret_like_text_is_redacted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install(monkeypatch, uia="Call me on +91 12345 67890 when ready")

        result = _run_read(_ctx(settings=Settings()))

        assert "+91 12345 67890" not in result.output
        assert "+" in result.output  # masked but still recognisable

    def test_window_title_is_redacted_in_output_and_data(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install(monkeypatch, title="Chase +91 99999 00000 reminder", process="chrome.exe")

        result = _run_read(_ctx(settings=Settings()))

        assert "+91 99999 00000" not in result.output
        assert "+91 99999 00000" not in str(result.data["title"])
        assert "Chase +" in result.output
        assert result.data["process"] == "chrome.exe"

    def test_text_is_truncated_to_the_configured_cap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install(monkeypatch, uia="A" * 5000)
        settings = Settings(screen=ScreenSettings(max_text_chars=100))

        result = _run_read(_ctx(settings=settings))

        assert result.data["truncated"] is True
        assert result.output.endswith("[truncated]")

    def test_dry_run_reads_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom() -> Any:
            raise AssertionError("dry run must not touch the desktop")

        monkeypatch.setattr("jarvis.perception.screen.foreground_window", _boom)

        result = _run_read(_ctx(settings=Settings(), dry_run=True))

        assert result.ok is True
        assert "[dry-run]" in result.output


class TestDescribeScreen:
    def test_local_path_does_not_call_the_llm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install(monkeypatch, uia="A settings page with several options")
        llm = _FakeLLM()

        result = _run_describe(_ctx(settings=Settings(), llm=llm))

        assert result.ok is True
        assert result.tainted is True
        assert "A settings page with several options" in result.output
        assert llm.calls == []
        assert "cloud" not in result.data

    def test_cloud_vision_is_off_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install(monkeypatch, uia="")  # thin local text would otherwise allow cloud
        llm = _FakeLLM()

        result = _run_describe(_ctx(settings=Settings(), llm=llm))

        assert result.ok is True
        assert llm.calls == []

    def test_cloud_vision_when_explicitly_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install(monkeypatch, uia="", ocr_available=False)
        llm = _FakeLLM("A photo viewer showing a dog.")
        settings = Settings(screen=ScreenSettings(allow_cloud_vision=True))

        result = _run_describe(_ctx(settings=settings, llm=llm))

        assert result.ok is True
        assert result.tainted is True
        assert result.output.startswith("Looking at your screen:")
        assert len(llm.calls) == 1
        assert llm.calls[0]["image_bytes"] == _PNG
        assert result.data["source"] == "cloud_vision"

    def test_cloud_enabled_but_denylisted_still_refuses(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install(monkeypatch, title="1Password", process="1Password.exe")
        llm = _FakeLLM()
        settings = Settings(screen=ScreenSettings(allow_cloud_vision=True))

        result = _run_describe(_ctx(settings=settings, llm=llm))

        assert result.ok is False
        assert llm.calls == []

    def test_cloud_falls_back_to_local_text_on_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install(monkeypatch, uia="", ocr_available=True, ocr="Some short text")

        class _Broken:
            def describe_image(self, **_kw: Any) -> Any:
                raise RuntimeError("vision unavailable")

        settings = Settings(screen=ScreenSettings(allow_cloud_vision=True))
        # local text ("Some short text") is below MIN_LOCAL_CHARS, so cloud is tried
        result = _run_describe(_ctx(settings=settings, llm=_Broken()))

        assert result.ok is True
        assert "Some short text" in result.output


class TestSpec:
    def test_read_screen_spec(self) -> None:
        spec = make_read_screen_spec()
        assert spec.name == "read_screen"
        assert spec.base_tier == 0
        assert spec.windows_only is True

    def test_describe_screen_spec(self) -> None:
        spec = make_describe_screen_spec()
        assert spec.name == "describe_screen"
        assert spec.base_tier == 1
        assert spec.windows_only is True

    @pytest.mark.parametrize("factory", [make_read_screen_spec, make_describe_screen_spec])
    def test_names_and_descriptions_are_not_blocked(self, factory: Any) -> None:
        spec = factory()
        assert rules.matches_blocked(spec, spec.args_model()) is False
        assert _FORBIDDEN_RE.search(f"{spec.name} {spec.description}") is None

    def test_args_accept_empty_and_reject_extras(self) -> None:
        assert ReadScreenArgs() == ReadScreenArgs()
        with pytest.raises(ValidationError):
            ReadScreenArgs(window="x")  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            DescribeScreenArgs(window="x")  # type: ignore[call-arg]


class TestPlatformGate:
    def _decide(self, tool: str) -> Any:
        registry = build_default_registry(Settings())
        pctx = PolicyContext(registry=registry, settings=Settings())
        step = Step(id="s1", tool=tool, args={}, rationale="r")
        return PolicyEngine().decide(step, pctx)

    @pytest.mark.parametrize("tool", ["read_screen", "describe_screen"])
    def test_refused_off_windows(self, monkeypatch: pytest.MonkeyPatch, tool: str) -> None:
        monkeypatch.setattr(engine_module, "is_windows", lambda: False)

        decision = self._decide(tool)

        assert decision.allowed is False
        assert decision.reasons == [_WINDOWS_ONLY_REASON]

    def test_read_screen_is_tier_0_on_windows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine_module, "is_windows", lambda: True)

        decision = self._decide("read_screen")

        assert decision.allowed is True
        assert decision.tier == 0
        assert decision.needs_confirm is False

    def test_describe_screen_confirms_on_windows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine_module, "is_windows", lambda: True)

        decision = self._decide("describe_screen")

        assert decision.allowed is True
        assert decision.tier == 1
        assert decision.needs_confirm is True


class TestRegistration:
    def test_both_registered_with_declared_tiers(self) -> None:
        registry = build_default_registry(Settings())
        assert registry.get("read_screen").base_tier == 0
        assert registry.get("describe_screen").base_tier == 1


# ---------------------------------------------------------------------------
# M2: UIA secure/visible control handling (no Windows required)
# ---------------------------------------------------------------------------


class _RawElement:
    """Fake ``element_info.element`` (the raw UI Automation proxy)."""

    def __init__(self, prop_value: bool = False, current: bool = False, prop_raises: bool = False):
        self._prop_value = prop_value
        self.CurrentIsPassword = current
        self._prop_raises = prop_raises

    def GetCurrentPropertyValue(self, _prop: int) -> bool:
        if self._prop_raises:
            raise OSError("COM hiccup")
        return self._prop_value


def _fake_element(name: str, *, password: bool = False, visible: bool = True) -> SimpleNamespace:
    raw = _RawElement(prop_value=password, current=password)
    info = SimpleNamespace(element=raw, visible=visible, name=name)
    return SimpleNamespace(
        element_info=info,
        window_text=lambda: name,
        texts=list,
    )


class TestIsPassword:
    def test_no_element_is_not_password(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("jarvis.perception.ui_text._is_password_property_id", lambda: 30019)

        assert ui_text_is_password(SimpleNamespace(element=None)) is False

    def test_property_positive_is_password(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("jarvis.perception.ui_text._is_password_property_id", lambda: 30019)

        assert ui_text_is_password(SimpleNamespace(element=_RawElement(prop_value=True))) is True

    def test_current_is_password_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("jarvis.perception.ui_text._is_password_property_id", lambda: 30019)

        info = SimpleNamespace(element=_RawElement(prop_value=False, current=True))
        assert ui_text_is_password(info) is True

    def test_not_password_when_both_negative(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("jarvis.perception.ui_text._is_password_property_id", lambda: 30019)

        assert ui_text_is_password(SimpleNamespace(element=_RawElement())) is False

    def test_com_error_is_not_password(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("jarvis.perception.ui_text._is_password_property_id", lambda: 30019)

        assert ui_text_is_password(SimpleNamespace(element=_RawElement(prop_raises=True))) is False


def ui_text_is_password(element_info: object) -> bool:
    from jarvis.perception import ui_text

    return ui_text._is_password(element_info)


def ui_text_is_visible(element_info: object) -> bool:
    from jarvis.perception import ui_text

    return ui_text._is_visible(element_info)


class _Ghost:
    """Object whose attribute access raises (failing visibility read)."""

    def __getattr__(self, _name: str) -> Any:
        raise OSError("UIA access failed")


class TestIsVisible:
    def test_visible_passes(self) -> None:
        assert ui_text_is_visible(SimpleNamespace(visible=True)) is True

    def test_hidden_is_not_visible(self) -> None:
        assert ui_text_is_visible(SimpleNamespace(visible=False)) is False

    def test_unknown_visibility_defaults_to_visible(self) -> None:
        assert ui_text_is_visible(SimpleNamespace(name="no visible attr")) is True

    def test_visibility_read_error_defaults_to_visible(self) -> None:
        assert ui_text_is_visible(_Ghost()) is True

    def test_zero_visibility_stays_visible(self) -> None:
        # Pinned quirk: only an explicit ``False`` skips an element.
        assert ui_text_is_visible(SimpleNamespace(visible=0)) is True


class TestUiaTextWalk:
    def _walk(self, monkeypatch: pytest.MonkeyPatch, elements: list[SimpleNamespace]) -> Any:
        from jarvis.perception import ui_text

        monkeypatch.setattr(ui_text, "_uia_elements", lambda: elements)
        return ui_text.foreground_uia_text()

    def test_skips_password_controls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result = self._walk(
            monkeypatch,
            [_fake_element("hello"), _fake_element("secret", password=True)],
        )

        assert "hello" in result.text
        assert "secret" not in result.text
        assert result.secure_controls == 1
        assert result.controls == 1

    def test_skips_invisible_elements(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result = self._walk(
            monkeypatch,
            [_fake_element("hidden", visible=False), _fake_element("shown")],
        )

        assert "shown" in result.text
        assert "hidden" not in result.text

    def test_dedupes_repeated_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result = self._walk(
            monkeypatch,
            [_fake_element("Save"), _fake_element("Save")],
        )

        assert result.text == "Save"
        assert result.controls == 1

    def test_caps_the_element_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from jarvis.perception import ui_text

        elements = [_fake_element(f"item{i}") for i in range(3)]
        monkeypatch.setattr(ui_text, "_uia_elements", lambda: elements)

        result = ui_text.foreground_uia_text(max_elements=1)

        assert result.text == "item0"
        assert result.controls == 1

    def test_empty_tree_yields_empty_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result = self._walk(monkeypatch, [])

        assert result.text == ""
        assert result.controls == 0

    def test_uia_failure_yields_empty_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from jarvis.perception import ui_text

        def _boom() -> list[Any]:
            raise RuntimeError("desktop went away")

        monkeypatch.setattr(ui_text, "_uia_elements", _boom)

        assert ui_text.foreground_uia_text().text == ""


# ---------------------------------------------------------------------------
# M1: foreground-window-only capture (no Windows, no mss required)
# ---------------------------------------------------------------------------


class _FakeWintypes:
    class RECT:
        def __init__(self, *, left: int = 0, top: int = 0, right: int = 0, bottom: int = 0) -> None:
            self.left, self.top, self.right, self.bottom = left, top, right, bottom


class _FakeUser32:
    def __init__(
        self,
        *,
        hwnd: int = 0,
        iconic: bool = False,
        rect: tuple[int, int, int, int] = (0, 0, 100, 100),
        rect_ok: bool = True,
        vs: tuple[int, int, int, int] = (0, 0, 1920, 1080),
    ) -> None:
        self.hwnd = hwnd
        self.iconic = iconic
        self.rect = rect
        self.rect_ok = rect_ok
        self.vs = vs

    def GetForegroundWindow(self) -> int:
        return self.hwnd

    def IsIconic(self, _hwnd: int) -> bool:
        return self.iconic

    def GetWindowRect(self, _hwnd: int, rect: _FakeWintypes.RECT) -> int:
        if not self.rect_ok:
            return 0
        rect.left, rect.top, rect.right, rect.bottom = self.rect
        return 1

    def GetSystemMetrics(self, index: int) -> int:
        return {76: self.vs[0], 77: self.vs[1], 78: self.vs[2], 79: self.vs[3]}[index]


class _FakeCtypes:
    def __init__(self, user32: _FakeUser32) -> None:
        self.windll = SimpleNamespace(user32=user32)
        self.wintypes = _FakeWintypes

    @staticmethod
    def byref(obj: Any) -> Any:
        return obj


def _install_user32(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> Any:
    from jarvis.perception import screen

    monkeypatch.setattr(screen, "ctypes", _FakeCtypes(_FakeUser32(**kwargs)))
    return screen


class TestForegroundBBox:
    def test_normal_window_rectangle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        screen = _install_user32(monkeypatch, hwnd=5, rect=(10, 20, 310, 220))

        assert screen._foreground_bbox() == {"left": 10, "top": 20, "width": 300, "height": 200}

    def test_no_foreground_window_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        screen = _install_user32(monkeypatch, hwnd=0)

        with pytest.raises(RuntimeError, match="no foreground window"):
            screen._foreground_bbox()

    def test_minimized_window_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        screen = _install_user32(monkeypatch, hwnd=5, iconic=True)

        with pytest.raises(RuntimeError, match="minimized"):
            screen._foreground_bbox()

    def test_getwindowrect_failure_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        screen = _install_user32(monkeypatch, hwnd=5, rect_ok=False)

        with pytest.raises(RuntimeError, match="could not read"):
            screen._foreground_bbox()

    def test_empty_rectangle_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        screen = _install_user32(monkeypatch, hwnd=5, rect=(40, 40, 40, 40))

        with pytest.raises(RuntimeError, match="empty or inverted"):
            screen._foreground_bbox()

    def test_inverted_rectangle_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        screen = _install_user32(monkeypatch, hwnd=5, rect=(300, 300, 100, 100))

        with pytest.raises(RuntimeError, match="empty or inverted"):
            screen._foreground_bbox()

    def test_fully_offscreen_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        screen = _install_user32(monkeypatch, hwnd=5, rect=(2000, 2000, 2500, 2100))

        with pytest.raises(RuntimeError, match="not visible"):
            screen._foreground_bbox()

    def test_clips_partially_offscreen_window(self, monkeypatch: pytest.MonkeyPatch) -> None:
        screen = _install_user32(monkeypatch, hwnd=5, rect=(-40, -50, 200, 300))

        assert screen._foreground_bbox() == {"left": 0, "top": 0, "width": 200, "height": 300}

    def test_explicit_hwnd_bypasses_foreground_query(self, monkeypatch: pytest.MonkeyPatch) -> None:
        screen = _install_user32(monkeypatch, hwnd=0, rect=(5, 5, 55, 55))

        assert screen._foreground_bbox(hwnd=9) == {"left": 5, "top": 5, "width": 50, "height": 50}


class TestCaptureForeground:
    def test_captures_the_window_rectangle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from jarvis.perception import screen

        bbox = {"left": 10, "top": 20, "width": 300, "height": 200}
        grabbed: list[dict[str, int]] = []
        monkeypatch.setattr(screen, "_foreground_bbox", lambda _hwnd=0: bbox)
        monkeypatch.setattr(screen, "_grab_bbox", lambda b: grabbed.append(b) or _PNG)

        assert screen.capture_foreground() == _PNG
        assert grabbed == [bbox]  # the window rect, never the whole monitor

    def test_handles_are_threaded_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from jarvis.perception import screen

        seen: dict[str, int] = {}
        monkeypatch.setattr(
            screen,
            "_foreground_bbox",
            lambda hwnd=0: seen.update(hwnd=hwnd) or {"left": 0, "top": 0, "width": 1, "height": 1},
        )
        monkeypatch.setattr(screen, "_grab_bbox", lambda b: b"png")

        screen.capture_foreground(hwnd=7)

        assert seen["hwnd"] == 7

    def test_never_falls_back_to_a_full_monitor_capture(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from jarvis.perception import screen

        grabbed: list[dict[str, int]] = []

        def _boom(_hwnd: int = 0) -> dict[str, int]:
            raise RuntimeError("the foreground window is minimized")

        monkeypatch.setattr(screen, "_foreground_bbox", _boom)
        monkeypatch.setattr(screen, "_grab_bbox", lambda b: grabbed.append(b) or b"x")

        with pytest.raises(RuntimeError, match="minimized"):
            screen.capture_foreground()

        assert grabbed == []  # nothing was captured as a fallback
