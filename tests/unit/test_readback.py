"""Tests for :mod:`jarvis.voice.readback` (Stage 3, pure module)."""

from __future__ import annotations

from jarvis.agent.state import Decision
from jarvis.voice.readback import tier2_readback


def _delete_decision(*paths: str, typed: str | None = None) -> Decision:
    return Decision(
        step_id="s1",
        tier=2,
        allowed=True,
        needs_confirm=True,
        needs_unlock=True,
        summary="Delete item(s)",
        resolved_paths=list(paths),
        needs_typed_confirmation=typed,
        action_hash="a" * 64,
    )


def test_readback_delete_single_short_file_ok() -> None:
    path = r"C:\Users\me\Documents\JarvisWorkspace\notes.txt"

    res = tier2_readback("delete_path", _delete_decision(path))

    assert res.ok is True
    assert res.text == path
    assert res.reason == ""


def test_readback_delete_multi_path_refused() -> None:
    first = r"C:\Users\me\Documents\JarvisWorkspace\a.txt"
    second = r"C:\Users\me\Documents\JarvisWorkspace\b.txt"

    res = tier2_readback("delete_path", _delete_decision(first, second))

    assert res.ok is False
    assert res.text == ""
    assert "exactly one" in res.reason


def test_readback_delete_long_path_refused_not_truncated() -> None:
    long_path = "C:\\" + "a" * 160 + ".txt"  # 166 chars, over the 150 limit
    assert len(long_path) > 150

    res = tier2_readback("delete_path", _delete_decision(long_path))

    assert res.ok is False
    assert res.text == ""  # never a truncated readback
    assert str(len(long_path)) in res.reason


def test_readback_whatsapp_short_ok_full_message() -> None:
    message = "see you at 6, bring the notes please"
    source = {
        "contact": "Mom",
        "message": message,
        "masked_number": "+\u2022\u2022\u2022\u2022\u202210",
    }

    res = tier2_readback("whatsapp_send", source)

    assert res.ok is True
    assert "Mom" in res.text
    assert message in res.text  # full message, not cut
    assert "+\u2022\u2022\u2022\u2022\u202210" in res.text


def test_readback_whatsapp_long_message_refused() -> None:
    source = {"contact": "Mom", "message": "x" * 201, "masked_number": "+\u2022\u202210"}

    res = tier2_readback("whatsapp_send", source)

    assert res.ok is False
    assert res.text == ""
    assert "too long" in res.reason


def test_readback_unknown_tool_refused() -> None:
    res = tier2_readback("open_app", {"name": "calculator"})

    assert res.ok is False
    assert res.text == ""
    assert res.reason
