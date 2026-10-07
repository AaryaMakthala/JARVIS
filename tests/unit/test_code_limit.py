"""Small-code size limit: create_file/append_file refuse oversized code files.

Deterministic, inside the tool run() - no LLM, no tier change, nothing is
written on refusal; non-code suffixes (txt, md, json, ...) are unaffected
(stage 4.6).
"""

from __future__ import annotations

from pathlib import Path

from jarvis.config import PolicySettings, Settings, ToolsSettings
from jarvis.tools.base import ToolContext, ToolResult
from jarvis.tools.files import MAX_CODE_LINES, make_append_file_spec, make_create_file_spec

MSG = "This version supports small code tasks only. (max 4000 characters / 120 lines)"


def _ctx(
    tmp_path: Path, *, tools: ToolsSettings | None = None, dry_run: bool = False
) -> ToolContext:
    settings = Settings(
        policy=PolicySettings(allowed_roots=[str(tmp_path)]),
        tools=tools or ToolsSettings(),
    )
    return ToolContext(settings=settings, dry_run=dry_run)


def _create(tmp_path: Path, name: str, content: str, ctx: ToolContext | None = None) -> ToolResult:
    spec = make_create_file_spec()
    model = spec.args_model(path=str(tmp_path / name), content=content)
    return spec.run_verified(model, ctx or _ctx(tmp_path))


def test_under_limit_python_file_is_created(tmp_path: Path) -> None:
    result = _create(tmp_path, "hello.py", "print('hi')\n")

    assert result.ok is True
    assert (tmp_path / "hello.py").read_text(encoding="utf-8") == "print('hi')\n"


def test_over_char_limit_is_refused_with_exact_message(tmp_path: Path) -> None:
    result = _create(tmp_path, "big.py", "x" * 4001)

    assert result.ok is False
    assert result.error == MSG
    assert not (tmp_path / "big.py").exists()


def test_over_line_limit_is_refused(tmp_path: Path) -> None:
    result = _create(tmp_path, "many.py", "\n".join(["# l"] * 121))

    assert result.ok is False
    assert result.error == MSG
    assert not (tmp_path / "many.py").exists()


def test_exactly_at_both_limits_is_allowed(tmp_path: Path) -> None:
    by_chars = _create(tmp_path, "at_chars.py", "x" * 4000)
    by_lines = _create(tmp_path, "at_lines.py", "\n".join(["# l"] * 120))

    assert by_chars.ok is True
    assert by_lines.ok is True


def test_large_text_file_is_unaffected(tmp_path: Path) -> None:
    result = _create(tmp_path, "notes.txt", "x" * 10000)

    assert result.ok is True
    assert (tmp_path / "notes.txt").exists()


def test_uppercase_suffix_is_refused(tmp_path: Path) -> None:
    result = _create(tmp_path, "BIG.PY", "x" * 4001)

    assert result.ok is False
    assert result.error == MSG


def test_append_crossing_limit_is_refused_and_file_unchanged(tmp_path: Path) -> None:
    target = tmp_path / "app.py"
    original = "print('x')\n"
    target.write_text(original, encoding="utf-8")
    spec = make_append_file_spec()
    model = spec.args_model(path=str(target), content="x" * 4000)

    result = spec.run_verified(model, _ctx(tmp_path))

    assert result.ok is False
    assert result.error == MSG
    assert target.read_text(encoding="utf-8") == original


def test_lowered_char_setting_is_honoured(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, tools=ToolsSettings(max_code_chars=50))

    result = _create(tmp_path, "over.py", "x" * 51, ctx=ctx)

    assert result.ok is False
    assert result.error == (
        "This version supports small code tasks only. (max 50 characters / 120 lines)"
    )


def test_default_limits_are_4000_chars_and_120_lines() -> None:
    assert Settings().tools.max_code_chars == 4000
    assert MAX_CODE_LINES == 120


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    result = _create(tmp_path, "dry.py", "print('hi')\n", ctx=_ctx(tmp_path, dry_run=True))

    assert result.ok is True
    assert "[dry-run]" in (result.output or "")
    assert not (tmp_path / "dry.py").exists()
