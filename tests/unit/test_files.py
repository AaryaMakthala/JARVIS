"""File-tool tests: create/append/list/read + Recycle-Bin delete and undo.

Everything runs inside a ``tmp_path`` allowed root with a :class:`FakeDirTrash`
and an injected undo log - the real Recycle Bin and the real keyring are never
touched here (docs/07_TESTING_AND_BENCHMARK.md).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from jarvis.config import PolicySettings, Settings
from jarvis.policy.paths import resolve_safe
from jarvis.tools.base import ToolContext
from jarvis.tools.files import (
    TrashRecord,
    append_undo_record,
    last_pending_undo_record,
    make_append_file_spec,
    make_create_file_spec,
    make_delete_path_spec,
    make_list_dir_spec,
    make_read_file_spec,
    make_undo_last_delete_spec,
)
from support import FakeDirTrash


def _ctx(ws: Path, tmp_path: Path, *, dry_run: bool = False) -> ToolContext:
    settings = Settings(policy=PolicySettings(allowed_roots=[str(ws.resolve())]))
    return ToolContext(
        settings=settings,
        dry_run=dry_run,
        trash=FakeDirTrash(tmp_path / "trash"),
        undo_log=tmp_path / "undo.jsonl",
    )


def _create(ws: Path, name: str, content: str = "hello") -> Path:
    target = ws / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    return root


# ---------------------------------------------------------------------------
# create_file / append_file (regression + Phase 2 refactor drift checks)
# ---------------------------------------------------------------------------


def test_create_file_writes_and_verifies(ws: Path, tmp_path: Path) -> None:
    ctx = _ctx(ws, tmp_path)
    spec = make_create_file_spec()
    target = ws / "note.txt"
    result = spec.run_verified(spec.args_model(path=str(target), content="hi there"), ctx)
    assert result.ok is True and result.verified is True
    assert target.read_text(encoding="utf-8") == "hi there"


def test_create_file_refuses_outside_allowed_root(ws: Path, tmp_path: Path) -> None:
    ctx = _ctx(ws, tmp_path)
    spec = make_create_file_spec()
    outside = tmp_path / "outside.txt"
    result = spec.run(spec.args_model(path=str(outside), content="x"), ctx)
    assert result.ok is False
    assert "outside the allowed folders" in result.error
    assert outside.exists() is False


def test_append_file_appends_and_verifies(ws: Path, tmp_path: Path) -> None:
    ctx = _ctx(ws, tmp_path)
    target = _create(ws, "a.txt", "first\n")
    spec = make_append_file_spec()
    result = spec.run_verified(spec.args_model(path=str(target), content="second"), ctx)
    assert result.ok is True
    assert result.verified is True
    assert target.read_text(encoding="utf-8") == "first\nsecond"


# ---------------------------------------------------------------------------
# list_dir / read_file (Tier 0, untrusted output)
# ---------------------------------------------------------------------------


def test_list_dir_lists_entries_and_flags_taint(ws: Path, tmp_path: Path) -> None:
    ctx = _ctx(ws, tmp_path)
    _create(ws, "a.txt", "x")
    (ws / "sub").mkdir()
    spec = make_list_dir_spec()
    result = spec.run(spec.args_model(path=str(ws)), ctx)
    assert result.ok is True
    assert result.tainted is True
    names = {entry["name"] for entry in result.data["entries"]}
    assert names == {"a.txt", "sub"}


def test_read_file_reads_and_truncates_at_max_bytes(ws: Path, tmp_path: Path) -> None:
    ctx = _ctx(ws, tmp_path)
    target = _create(ws, "big.txt", "x" * 4000)
    spec = make_read_file_spec()
    result = spec.run(spec.args_model(path=str(target), max_bytes=1000), ctx)
    assert result.ok is True
    assert result.data["truncated"] is True
    assert "[truncated" in result.output
    assert len(target.read_text(encoding="utf-8")) == 4000  # file untouched


def test_read_file_rejects_non_utf8_content(ws: Path, tmp_path: Path) -> None:
    ctx = _ctx(ws, tmp_path)
    target = ws / "bin.dat"
    target.write_bytes(b"\xff\xfe\x00\x01\x02")
    spec = make_read_file_spec()
    result = spec.run(spec.args_model(path=str(target)), ctx)
    assert result.ok is False
    assert "valid UTF-8" in result.error


# ---------------------------------------------------------------------------
# delete_path: Recycle-Bin semantics, undo log, fail-closed checks
# ---------------------------------------------------------------------------


def test_delete_file_moves_to_bin_and_writes_undo_record(ws: Path, tmp_path: Path) -> None:
    ctx = _ctx(ws, tmp_path)
    target = _create(ws, "file.txt", "data-block")
    spec = make_delete_path_spec()
    result = spec.run(spec.args_model(paths=[str(target)]), ctx)
    assert result.ok is True
    assert target.exists() is False
    assert len(list((tmp_path / "trash").iterdir())) == 1  # it went to the bin, not rm

    record = json.loads((tmp_path / "undo.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record["op"] == "delete"
    assert record["original_path"] == str(resolve_safe(str(target)))
    assert record["possible"] is True
    assert record["succeeded"] is True
    assert record["sha256"] == hashlib.sha256(b"data-block").hexdigest()


def test_delete_folder_moves_whole_tree_to_bin(ws: Path, tmp_path: Path) -> None:
    ctx = _ctx(ws, tmp_path)
    folder = ws / "Reports"
    (folder / "nested").mkdir(parents=True)
    (folder / "nested" / "a.txt").write_text("x", encoding="utf-8")
    spec = make_delete_path_spec()
    result = spec.run(spec.args_model(paths=[str(folder)]), ctx)
    assert result.ok is True
    assert folder.exists() is False
    record = last_pending_undo_record(tmp_path / "undo.jsonl")
    assert record is not None
    assert record["kind"] == "dir"
    assert record["item_count"] >= 1


def test_delete_path_fails_closed_on_outside_and_protected_paths(ws: Path, tmp_path: Path) -> None:
    ctx = _ctx(ws, tmp_path)
    outside = _create(tmp_path, "elsewhere.txt")
    spec = make_delete_path_spec()
    result = spec.run(spec.args_model(paths=[str(outside)]), ctx)
    assert result.ok is False
    assert "outside the allowed folders" in result.error
    assert outside.exists() is True

    if os.name == "nt":
        protected = Path(os.environ["SystemRoot"]) / "System32"
        result2 = spec.run(spec.args_model(paths=[str(protected)]), ctx)
        assert result2.ok is False
        assert "protected" in result2.error


def test_delete_path_partial_failure_reports_honestly(ws: Path, tmp_path: Path) -> None:
    ctx = _ctx(ws, tmp_path)
    gone = _create(ws, "gone.txt")
    spec = make_delete_path_spec()
    result = spec.run(spec.args_model(paths=[str(gone), str(ws / "missing.txt")]), ctx)
    assert result.ok is False  # one path failed
    assert "no longer exists" in result.output
    assert gone.exists() is False  # the valid one was still deleted (partial semantics)
    assert len(list((tmp_path / "trash").iterdir())) == 1
    log = (tmp_path / "undo.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(log) == 1


def test_delete_path_dry_run_never_touches_disk(ws: Path, tmp_path: Path) -> None:
    ctx = _ctx(ws, tmp_path, dry_run=True)
    target = _create(ws, "d.txt")
    spec = make_delete_path_spec()
    result = spec.run(spec.args_model(paths=[str(target)]), ctx)
    assert result.ok is True
    assert "[dry-run]" in result.output
    assert target.exists() is True
    assert (tmp_path / "undo.jsonl").exists() is False


def test_delete_never_uses_permanent_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """os.remove/shutil.rmtree must never be the delete path (docs/03 inv. 5)."""
    ws = tmp_path / "ws"
    ws.mkdir()
    ctx = _ctx(ws, tmp_path)
    target = _create(ws, "precious.txt")

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("permanent deletion API used")

    monkeypatch.setattr(os, "remove", _boom)
    monkeypatch.setattr(shutil, "rmtree", _boom)

    spec = make_delete_path_spec()
    result = spec.run(spec.args_model(paths=[str(target)]), ctx)
    assert result.ok is True
    assert target.exists() is False
    assert (tmp_path / "undo.jsonl").is_file()


def test_delete_refuses_when_bin_fails_without_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    settings = Settings(policy=PolicySettings(allowed_roots=[str(ws.resolve())]))
    target = _create(ws, "keepme.txt")

    class FailingTrash:
        def send(self, path: Path) -> TrashRecord:
            return TrashRecord(str(path), "file", 0, 1, None, error="Recycle Bin refused")

        def restore(self, record: TrashRecord) -> bool:
            return False

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("permanent deletion fallback used")

    monkeypatch.setattr(os, "remove", _boom)
    monkeypatch.setattr(shutil, "rmtree", _boom)

    ctx = ToolContext(settings=settings, trash=FailingTrash(), undo_log=tmp_path / "u.jsonl")
    spec = make_delete_path_spec()
    result = spec.run(spec.args_model(paths=[str(target)]), ctx)
    assert result.ok is False
    assert "Recycle Bin refused" in result.output
    assert target.exists() is True  # nothing was destroyed
    assert (tmp_path / "u.jsonl").exists() is False


# ---------------------------------------------------------------------------
# undo_last_delete
# ---------------------------------------------------------------------------


def test_undo_restores_the_latest_delete_and_marks_it(ws: Path, tmp_path: Path) -> None:
    ctx = _ctx(ws, tmp_path)
    first = _create(ws, "first.txt", "one")
    second = _create(ws, "second.txt", "two")
    spec = make_delete_path_spec()
    spec.run(spec.args_model(paths=[str(first)]), ctx)
    spec.run(spec.args_model(paths=[str(second)]), ctx)
    assert first.exists() is False and second.exists() is False

    undo_spec = make_undo_last_delete_spec()
    result = undo_spec.run_verified(undo_spec.args_model(), ctx)
    assert result.ok is True
    assert result.verified is True
    assert second.exists() is True and second.read_text(encoding="utf-8") == "two"

    log = (tmp_path / "undo.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(log) == 3  # two delete records + one restored marker
    assert json.loads(log[-1])["restored"] is True

    again = undo_spec.run(undo_spec.args_model(), ctx)
    assert again.ok is True
    assert again.output.startswith("nothing to undo")


def test_undo_refuses_a_target_that_is_now_outside_allowed_roots(ws: Path, tmp_path: Path) -> None:
    settings = Settings(policy=PolicySettings(allowed_roots=[str(ws.resolve())]))
    outside_file = _create(tmp_path, "outside.txt")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    shutil.copy(str(outside_file), str(bin_dir / "copy"))
    ctx = ToolContext(
        settings=settings,
        trash=FakeDirTrash(tmp_path / "trash"),
        undo_log=tmp_path / "undo.jsonl",
    )
    append_undo_record(
        ctx,
        TrashRecord(
            original_path=str(outside_file.resolve(strict=False)),
            kind="file",
            size=5,
            item_count=1,
            sha256=None,
            recycled_path=str(bin_dir / "copy"),
        ),
    )
    assert outside_file.exists() is True  # still there: restore must refuse, not overwrite

    result = make_undo_last_delete_spec().run(make_undo_last_delete_spec().args_model(), ctx)
    assert result.ok is False
    assert "cannot auto-restore" in result.error
    assert "outside the allowed folders" in result.error
    assert (bin_dir / "copy").exists()  # recycle bin copy untouched -> manual restore


def test_undo_skips_malformed_and_foreign_records(ws: Path, tmp_path: Path) -> None:
    ctx = _ctx(ws, tmp_path)
    log = tmp_path / "undo.jsonl"
    log.write_text(
        "not json at all\n"
        '{"op": "delete", "original_path": "C:\\\\x", "restored": false}\n'
        '{"op": "notadelete", "original_path": "C:\\\\y"}\n'
        "garbage\n",
        encoding="utf-8",
    )
    target = _create(ws, "real.txt")
    ctx.undo_log = log
    append_undo_record(
        ctx, TrashRecord(str(target.resolve(strict=False)), "file", 6, 1, None, recycled_path="N/A")
    )
    parsed = last_pending_undo_record(log)
    assert parsed is not None
    assert parsed["original_path"] == str(target.resolve(strict=False))


def test_undo_reports_when_bin_location_is_unknown(ws: Path, tmp_path: Path) -> None:
    ctx = _ctx(ws, tmp_path)
    target = _create(ws, "orphan.txt")
    append_undo_record(
        ctx,
        TrashRecord(str(target.resolve(strict=False)), "file", 6, 1, None, recycled_path=None),
    )
    result = make_undo_last_delete_spec().run(make_undo_last_delete_spec().args_model(), ctx)
    assert result.ok is False
    assert "Recycle Bin" in result.error and "manually" in result.error


def test_undo_dry_run_only_previews(ws: Path, tmp_path: Path) -> None:
    ctx = _ctx(ws, tmp_path, dry_run=True)
    target = _create(ws, "x.txt")
    append_undo_record(
        ctx, TrashRecord(str(target.resolve(strict=False)), "file", 6, 1, None, recycled_path="N/A")
    )
    result = make_undo_last_delete_spec().run(make_undo_last_delete_spec().args_model(), ctx)
    assert result.ok is True
    assert "[dry-run]" in result.output
    assert target.exists() is True


def test_undo_records_are_list_iterable_from_the_log(ws: Path, tmp_path: Path) -> None:
    ctx = _ctx(ws, tmp_path)
    for name in ("a.txt", "b.txt"):
        append_undo_record(
            ctx,
            TrashRecord(
                str((ws / name).resolve(strict=False)), "file", 1, 1, None, recycled_path="bin"
            ),
        )
    records = [
        json.loads(line)
        for line in (tmp_path / "undo.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [r["original_path"] for r in records] == [
        str((ws / "a.txt").resolve(strict=False)),
        str((ws / "b.txt").resolve(strict=False)),
    ]
