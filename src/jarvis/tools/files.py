"""File tools: create/append/read/list + Recycle-Bin delete and undo.

Tiers (docs/04_TOOLS_SPEC.md): ``list_dir``/``read_file`` Tier 0 (data may be
untrusted -> ``tainted``), ``create_file``/``append_file`` Tier 1, ``delete_path``
Tier 2 (folder deletes also need a typed folder-name confirmation from the
engine), ``undo_last_delete`` Tier 1.

Deletion invariants (docs/03 section 5, invariants 5-6):

* every delete goes to the Windows Recycle Bin via ``send2trash``; ``os.remove`` /
  ``shutil.rmtree`` are never used to delete, and there is **no** permanent
  fallback - if the Recycle Bin fails, the delete fails;
* each delete writes a JSONL undo-log record (what/who/when/possible/succeeded);
* the tool re-resolves every path at execution time and fails closed if a path
  now points outside the allowed roots or into a protected region (the engine
  already did this; this is the second line against path swaps);
* restore is best-effort (Recycle Bin lost the location -> the user is told).
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field
from send2trash import send2trash

from jarvis import config
from jarvis.policy import paths
from jarvis.tools.base import ToolContext, ToolResult, ToolSpec

_MAX_CHARS = 1_048_576  # 1 MiB content/secondary cap
_CAP = f"content exceeds the {_MAX_CHARS} character limit"
_MAX_PATHS = 20
_COUNT_CAP = 50_000  # previews/stat cap for folder scanning
_HASH_BYTES = 64 * 1024 * 1024  # sha256 cap for very large files


# ---------------------------------------------------------------------------
# Args models
# ---------------------------------------------------------------------------


class CreateFileArgs(BaseModel):
    """Create a UTF-8 text file at ``path`` (inside allowed roots)."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=4096, description="Absolute file path.")
    content: str = Field(default="", max_length=_MAX_CHARS, description="Text to write.")
    overwrite: bool = Field(default=False, description="Replace the file if it exists.")


class AppendFileArgs(BaseModel):
    """Append a UTF-8 text block to an existing file."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=4096, description="Absolute file path.")
    content: str = Field(min_length=1, max_length=_MAX_CHARS, description="Text to append.")


class ListDirArgs(BaseModel):
    """List the entries of a directory (names + kind + size)."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=4096, description="Absolute folder path.")


class ReadFileArgs(BaseModel):
    """Read (part of) a UTF-8 text file."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=4096, description="Absolute file path.")
    max_bytes: int = Field(
        default=1_048_576, ge=1, le=1_048_576, description="Max bytes to read (1 MiB cap)."
    )


class DeletePathArgs(BaseModel):
    """Move paths to the Recycle Bin (reversible; undo log written)."""

    model_config = ConfigDict(extra="forbid")

    paths: list[str] = Field(
        min_length=1,
        max_length=_MAX_PATHS,
        description=f"Paths to delete (1..{_MAX_PATHS}).",
    )


class UndoDeleteArgs(BaseModel):
    """Restore the most recent Recycle-Bin delete from the undo log."""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Trash service
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrashRecord:
    """What was moved to the (Recycle) Bin, for the undo log and restore."""

    original_path: str
    kind: str  # "file" | "dir"
    size: int
    item_count: int
    sha256: str | None
    recycled_path: str | None = None  # where it lives in the Bin (or None)
    error: str | None = None


class TrashService(Protocol):
    """Abstract away the real Windows Recycle Bin so tests can fake it."""

    def send(self, path: Path) -> TrashRecord:
        """Move ``path`` to the bin; return its record (error handled inside)."""
        ...

    def restore(self, record: TrashRecord) -> bool:
        """Move the item back to ``record.original_path`` if we know where
        it went; ``False`` means it must be restored manually."""
        ...


class WindowsRecycleBinTrash:
    """send2trash-backed trash service with a recycle-bin location probe.

    ``recycled_path`` is found by snapshotting the per-user ``$Recycle.Bin``
    directory before/after ``send2trash``.  When the location cannot be
    determined (no SID token, unusual layout) the delete still succeeds and the
    undo log simply can't restore it - that is reported honestly.
    """

    def __init__(self) -> None:
        self._bin_dir = _recycle_bin_dir()

    def send(self, path: Path) -> TrashRecord:
        kind, size, count, digest = _describe_target(path)
        if not path.exists():
            return TrashRecord(str(path), kind, size, count, digest, error="path no longer exists")
        before = _bin_snapshot(self._bin_dir)
        try:
            send2trash(str(path))
        except Exception as exc:  # noqa: BLE001 - never raise into a delete
            return TrashRecord(
                str(path),
                kind,
                size,
                count,
                digest,
                error=f"Recycle Bin refused the path: {type(exc).__name__}: {exc}",
            )
        after = _bin_snapshot(self._bin_dir)
        moved = self._find_new_entry(before, after)
        return TrashRecord(str(path), kind, size, count, digest, recycled_path=moved)

    def restore(self, record: TrashRecord) -> bool:
        if not record.recycled_path:
            return False
        return _move_back(Path(record.recycled_path), Path(record.original_path))

    def _find_new_entry(self, before: set[str], after: set[str]) -> str | None:
        if self._bin_dir is None:
            return None
        added = after - before
        if not added:
            return None
        # Windows Recycle Bin stores two entries per deletion: $R<data> (the
        # actual file) and $I<data> (metadata: original path, size, timestamp).
        # We must restore the $R file, not the $I metadata blob.
        data_entries = sorted(name for name in added if name.startswith("$R"))
        if not data_entries:
            return None
        candidate = self._bin_dir / data_entries[0]
        return str(candidate) if candidate.exists() else None


def _recycle_bin_dir() -> Path | None:
    """Best-effort ``C:\\$Recycle.Bin\\<current SID>`` for snapshot/restore."""
    if os.name != "nt":
        return None
    home = os.path.expanduser("~")
    drive = os.path.splitdrive(home)[0] or "C:"
    root = Path(f"{drive}{os.sep}$Recycle.Bin")
    if not root.is_dir():
        return None
    sid = _current_sid()
    if sid is None:
        return None
    directory = root / sid
    return directory if directory.is_dir() else None


def _current_sid() -> str | None:
    try:
        win32api = importlib.import_module("win32api")
        win32con = importlib.import_module("win32con")
        win32security = importlib.import_module("win32security")
        token = win32security.OpenProcessToken(win32api.GetCurrentProcess(), win32con.TOKEN_QUERY)
        try:
            info = win32security.GetTokenInformation(token, win32security.TokenUser)
            return str(win32security.ConvertSidToStringSid(info[0]))
        finally:
            win32api.CloseHandle(token)
    except Exception:  # noqa: BLE001 - fall back to "location unknown"
        return None


def _bin_snapshot(bin_dir: Path | None) -> set[str]:
    if bin_dir is None:
        return set()
    try:
        return {e.name for e in bin_dir.iterdir()}
    except OSError:
        return set()


def _move_back(src: Path, target: Path) -> bool:
    if not src.exists():
        return False
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(target))
    except OSError:
        return False
    return target.exists()


# ---------------------------------------------------------------------------
# Undo log
# ---------------------------------------------------------------------------


def _undo_path(ctx: ToolContext) -> Path:
    return ctx.undo_log or config.undo_log()


def append_undo_record(ctx: ToolContext, record: TrashRecord) -> None:
    """Write one JSONL delete record (what/who/when/possible/succeeded)."""
    ts = datetime.now(UTC).isoformat(timespec="seconds")
    entry = {
        "op": "delete",
        "ts": ts,
        "who": "jarvis",
        "original_path": record.original_path,
        "kind": record.kind,
        "size": record.size,
        "item_count": record.item_count,
        "sha256": record.sha256,
        "recycled_path": record.recycled_path,
        "possible": record.recycled_path is not None,
        "succeeded": record.error is None,
    }
    path = _undo_path(ctx)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")
    except OSError as exc:
        # The delete already happened; surface the log failure to the caller so
        # it can report the delete succeeded but bookkeeping did not.
        raise OSError(f"could not write undo log: {exc}") from exc


def last_pending_undo_record(log_path: Path) -> dict[str, Any] | None:
    """Return the most recent *un-consumed* delete record, or ``None``.

    A delete record is consumed once a later restored marker for the same
    ``original_path`` is appended, so it can never be selected again. This
    tool is strictly one-shot: it returns only the newest delete record and
    never cascades to older ones (each real restore is a single action, and
    anything earlier stays in history untouched). Malformed/foreign lines are
    skipped (fail closed: never guess).
    """
    if not log_path.is_file():
        return None
    last: tuple[dict[str, Any], int] | None = None  # (record, line index)
    marker_index: dict[str, int] = {}  # original_path -> line index of its restore marker
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for idx, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except ValueError:
            continue
        if not isinstance(parsed, dict) or parsed.get("op") != "delete":
            continue
        original = parsed.get("original_path")
        if parsed.get("restored"):
            if isinstance(original, str) and original:
                marker_index[original] = idx
            continue
        last = (parsed, idx)
    if last is None:
        return None
    record, idx = last
    original = record.get("original_path")
    if isinstance(original, str) and marker_index.get(original, -1) > idx:
        return None  # consumed: a later marker already restored this path
    return record


def mark_undo_restored(log_path: Path, original_path: str) -> None:
    """Append an undo marker so the record is not restored twice."""
    entry = {
        "op": "delete",
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "who": "jarvis",
        "original_path": original_path,
        "restored": True,
    }
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")
    except OSError:
        pass  # log is best-effort; the restore itself already succeeded


# ---------------------------------------------------------------------------
# Stat/preview helpers
# ---------------------------------------------------------------------------


def _describe_target(target: Path) -> tuple[str, int, int, str | None]:
    """Return (kind, total_size, item_count, sha256) for a file or folder."""
    if target.is_dir():
        size = 0
        count = 0
        for root, _dirs, files in os.walk(target):
            for name in files:
                try:
                    size += (Path(root) / name).stat().st_size
                except OSError:
                    pass
                count += 1
                if count >= _COUNT_CAP:
                    return ("dir", size, count, None)
        return ("dir", size, count, None)
    try:
        stat = target.stat()
    except OSError:
        return ("file", 0, 1, None)
    digest = _sha256_capped(target) if target.is_file() else None
    return ("file", stat.st_size, 1, digest)


def _sha256_capped(target: Path) -> str | None:
    if not target.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with target.open("rb") as fh:
            remaining = _HASH_BYTES
            while remaining > 0:
                chunk = fh.read(min(65536, remaining))
                if not chunk:
                    break
                digest.update(chunk)
                remaining -= len(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024 or unit == "GB":
            return f"{nbytes:.0f} {unit}" if unit == "B" else f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} GB"


def _preview_line(resolved: Path) -> str:
    kind, size, count, _ = _describe_target(resolved)
    if kind == "dir":
        return f"folder {resolved} ({count} item(s), {_human_size(size)})"
    return f"file {resolved} ({_human_size(size)})"


# ---------------------------------------------------------------------------
# create_file (Phase 1, refactored around shared helpers)
# ---------------------------------------------------------------------------


def _resolve(args: CreateFileArgs) -> Path:
    return paths.resolve_safe(args.path)


def _run_create_file(args: CreateFileArgs, ctx: ToolContext) -> ToolResult:
    try:
        target = _resolve(args)
        if not paths.within_any_root(target, paths.roots_from_settings(ctx.settings)):
            return ToolResult(ok=False, error="refusing: path is outside the allowed folders")
    except paths.PathError as exc:
        return ToolResult(ok=False, error=str(exc))

    if target.exists() and not args.overwrite:
        return ToolResult(ok=False, error=f"file exists; pass overwrite=true to replace: {target}")

    if ctx.dry_run:
        return ToolResult(ok=True, output=f"[dry-run] would create {target}")

    data = args.content.encode("utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=".jarvis-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, target)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return ToolResult(
        ok=True,
        output=f"wrote {len(data)} bytes to {target}",
        data={"path": str(target), "size": len(data)},
    )


def _verify_create_file(args: CreateFileArgs, result: ToolResult, ctx: ToolContext) -> ToolResult:
    if ctx.dry_run:
        return result.model_copy(update={"verified": None})
    if not result.ok:
        return result.model_copy(update={"verified": False})
    try:
        target = _resolve(args)
    except paths.PathError:
        return result.model_copy(update={"verified": False})
    if not target.is_file():
        return result.model_copy(update={"verified": False})
    expected = args.content.encode("utf-8")
    if target.stat().st_size != len(expected):
        return result.model_copy(update={"verified": False})
    if hashlib.sha256(target.read_bytes()).hexdigest() != hashlib.sha256(expected).hexdigest():
        return result.model_copy(update={"verified": False})
    return result.model_copy(update={"verified": True})


def _describe_create_file(args: CreateFileArgs) -> str:
    try:
        target = _resolve(args)
    except paths.PathError:
        target = args.path
    return f"Create file {target} ({len(args.content)} chars)"


def make_create_file_spec() -> ToolSpec:
    """Build the ``create_file`` tool (Tier 1)."""
    return ToolSpec(
        name="create_file",
        description=(
            "Create a UTF-8 text file at a path inside the user's allowed folders. "
            "Refuses to overwrite an existing file unless overwrite is true."
        ),
        args_model=CreateFileArgs,
        base_tier=1,
        timeout_s=30,
        path_args=("path",),
        run=_run_create_file,
        verify=_verify_create_file,
        describe=_describe_create_file,
    )


# ---------------------------------------------------------------------------
# append_file
# ---------------------------------------------------------------------------


def _run_append_file(args: AppendFileArgs, ctx: ToolContext) -> ToolResult:
    try:
        target = paths.resolve_safe(args.path)
        if not paths.within_any_root(target, paths.roots_from_settings(ctx.settings)):
            return ToolResult(ok=False, error="refusing: path is outside the allowed folders")
    except paths.PathError as exc:
        return ToolResult(ok=False, error=str(exc))

    if not target.is_file():
        return ToolResult(ok=False, error=f"file does not exist (use create_file): {target}")

    if ctx.dry_run:
        return ToolResult(
            ok=True, output=f"[dry-run] would append {len(args.content)} chars to {target}"
        )

    data = args.content.encode("utf-8")
    try:
        with target.open("ab") as fh:
            fh.write(data)
    except OSError as exc:
        return ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}")
    return ToolResult(
        ok=True,
        output=f"appended {len(data)} bytes to {target}",
        data={"path": str(target), "size": len(data)},
    )


def _verify_append_file(args: AppendFileArgs, result: ToolResult, ctx: ToolContext) -> ToolResult:
    if ctx.dry_run:
        return result.model_copy(update={"verified": None})
    if not result.ok:
        return result.model_copy(update={"verified": False})
    try:
        target = paths.resolve_safe(args.path)
    except paths.PathError:
        return result.model_copy(update={"verified": False})
    if not target.is_file():
        return result.model_copy(update={"verified": False})
    expected = args.content.encode("utf-8")
    try:
        with target.open("rb") as fh:
            fh.seek(-len(expected), 2)
            tail = fh.read()
    except OSError:
        return result.model_copy(update={"verified": False})
    return result.model_copy(update={"verified": tail == expected})


def make_append_file_spec() -> ToolSpec:
    """Build the ``append_file`` tool (Tier 1)."""
    return ToolSpec(
        name="append_file",
        description=(
            "Append a UTF-8 text block to an existing file inside the allowed "
            "folders. The file must already exist."
        ),
        args_model=AppendFileArgs,
        base_tier=1,
        timeout_s=30,
        path_args=("path",),
        run=_run_append_file,
        verify=_verify_append_file,
        describe=lambda args: f"Append {len(args.content)} chars to {args.path}",
    )


# ---------------------------------------------------------------------------
# list_dir / read_file (Tier 0, untrusted data)
# ---------------------------------------------------------------------------


def _run_list_dir(args: ListDirArgs, ctx: ToolContext) -> ToolResult:
    try:
        target = paths.resolve_safe(args.path)
        if not paths.within_any_root(target, paths.roots_from_settings(ctx.settings)):
            return ToolResult(ok=False, error="refusing: path is outside the allowed folders")
    except paths.PathError as exc:
        return ToolResult(ok=False, error=str(exc))
    if not target.is_dir():
        return ToolResult(ok=False, error=f"not a folder: {target}")
    if ctx.dry_run:
        return ToolResult(ok=True, output=f"[dry-run] would list {target}")
    try:
        entries = sorted(target.iterdir(), key=lambda p: p.name.casefold())
    except OSError as exc:
        return ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}")
    lines: list[str] = []
    listing: list[dict[str, Any]] = []
    for entry in entries:
        try:
            kind = "dir" if entry.is_dir() else "file"
            size = entry.stat().st_size if kind == "file" else 0
        except OSError:
            kind, size = "unknown", 0
        lines.append(f"{entry.name} ({kind}, {_human_size(size)})")
        listing.append({"name": entry.name, "kind": kind, "size": size})
    output = f"{len(entries)} entries in {target}:\n" + "\n".join(lines)
    return ToolResult(ok=True, output=output[:4000], data={"entries": listing}, tainted=True)


def make_list_dir_spec() -> ToolSpec:
    """Build the ``list_dir`` tool (Tier 0; entries are untrusted data)."""
    return ToolSpec(
        name="list_dir",
        description=(
            "List the entries (names, kind, size) of a folder inside the allowed roots. Read-only."
        ),
        args_model=ListDirArgs,
        base_tier=0,
        timeout_s=15,
        path_args=("path",),
        run=_run_list_dir,
        describe=lambda args: f"List folder {args.path}",
    )


def _run_read_file(args: ReadFileArgs, ctx: ToolContext) -> ToolResult:
    try:
        target = paths.resolve_safe(args.path)
        if not paths.within_any_root(target, paths.roots_from_settings(ctx.settings)):
            return ToolResult(ok=False, error="refusing: path is outside the allowed folders")
    except paths.PathError as exc:
        return ToolResult(ok=False, error=str(exc))
    if not target.is_file():
        return ToolResult(ok=False, error=f"not a file: {target}")
    if ctx.dry_run:
        return ToolResult(ok=True, output=f"[dry-run] would read {target}")
    try:
        with target.open("rb") as fh:
            raw = fh.read(args.max_bytes)
    except OSError as exc:
        return ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return ToolResult(ok=False, error="file is not valid UTF-8 text")
    truncated = len(raw) >= args.max_bytes
    note = "\n[truncated at the max_bytes limit]" if truncated else ""
    return ToolResult(
        ok=True,
        output=f"{text}{note}",
        data={"path": str(target), "bytes": len(raw), "truncated": truncated},
        tainted=True,
    )


def make_read_file_spec() -> ToolSpec:
    """Build the ``read_file`` tool (Tier 0; contents are untrusted data)."""
    return ToolSpec(
        name="read_file",
        description=(
            "Read up to max_bytes (UTF-8, capped at 1 MiB) of a file inside "
            "the allowed roots. Read-only."
        ),
        args_model=ReadFileArgs,
        base_tier=0,
        timeout_s=15,
        path_args=("path",),
        run=_run_read_file,
        describe=lambda args: f"Read file {args.path}",
    )


# ---------------------------------------------------------------------------
# delete_path (Tier 2, Recycle Bin only) and undo_last_delete (Tier 1)
# ---------------------------------------------------------------------------


def _check_target(raw: str, ctx: ToolContext) -> tuple[Path | None, str | None]:
    """Re-resolve and re-check one deletion target at execution time.

    Returns (resolved, None) on success, or (None, error) when the path is now
    outside/invalid - the tool refuses that path instead of deleting it.
    """
    try:
        resolved = paths.resolve_safe(raw)
    except paths.PathError as exc:
        return None, str(exc)
    if paths.is_protected(resolved):
        return None, f"path is protected: {resolved}"
    roots = paths.roots_from_settings(ctx.settings)
    if not paths.within_any_root(resolved, roots):
        return None, f"path is outside the allowed folders: {resolved}"
    return resolved, None


def _run_delete_path(args: DeletePathArgs, ctx: ToolContext) -> ToolResult:
    trash: TrashService = ctx.trash if ctx.trash is not None else WindowsRecycleBinTrash()
    deleted: list[str] = []
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    for raw in args.paths:
        resolved, err = _check_target(raw, ctx)
        if err is not None:
            errors.append(f"{raw}: {err}")
            continue
        if not resolved.exists():
            errors.append(f"{raw}: no longer exists")
            continue
        if ctx.dry_run:
            deleted.append(f"[dry-run] {_preview_line(resolved)}")
            continue
        record = trash.send(resolved)
        if record.error is not None:
            errors.append(f"{record.original_path}: {record.error}")
            continue
        try:
            append_undo_record(ctx, record)
        except OSError as exc:
            errors.append(f"deleted {record.original_path} but the undo log failed: {exc}")
            continue
        deleted.append(
            f"{record.original_path} -> Recycle Bin ({_human_size(record.size)}, "
            f"{record.item_count} item(s))"
        )
        records.append(record.__dict__)

    lines = deleted + [f"error: {e}" for e in errors]
    ok = not errors
    output = "\n".join(lines) if lines else "nothing to delete"
    error = "\n".join(errors) if errors else None
    return ToolResult(
        ok=ok, output=output, error=error, data={"deleted": records, "errors": errors}
    )


def _verify_delete_path(args: DeletePathArgs, result: ToolResult, ctx: ToolContext) -> ToolResult:
    if ctx.dry_run:
        return result.model_copy(update={"verified": None})
    if not result.ok:
        return result.model_copy(update={"verified": False})
    for raw in args.paths:
        try:
            resolved = paths.resolve_safe(raw)
        except paths.PathError:
            return result.model_copy(update={"verified": False})
        if resolved.exists():
            return result.model_copy(update={"verified": False})
    return result.model_copy(update={"verified": True})


def _describe_delete(args: DeletePathArgs) -> str:
    lines: list[str] = []
    for raw in args.paths:
        try:
            lines.append(_preview_line(paths.resolve_safe(raw)))
        except paths.PathError:
            lines.append(f"invalid path: {raw}")
    head = f"Delete {len(args.paths)} item(s) to the Recycle Bin (reversible, undo log):"
    return head + "\n" + "\n".join(lines)


def make_delete_path_spec() -> ToolSpec:
    """Build the ``delete_path`` tool (Tier 2, Recycle Bin only, undo log)."""
    return ToolSpec(
        name="delete_path",
        description=(
            "Move files/folders to the Recycle Bin (reversible). Inside the "
            "allowed folders only; never touches protected paths. Writes an "
            "undo-log record so `undo_last_delete` can restore it."
        ),
        args_model=DeletePathArgs,
        base_tier=2,
        timeout_s=60,
        path_args=("paths",),
        run=_run_delete_path,
        verify=_verify_delete_path,
        describe=_describe_delete,
    )


def _record_to_trash(parsed: dict[str, Any]) -> TrashRecord:
    return TrashRecord(
        original_path=str(parsed.get("original_path", "")),
        kind=str(parsed.get("kind", "file")),
        size=int(parsed.get("size") or 0),
        item_count=int(parsed.get("item_count") or 1),
        sha256=str(parsed.get("sha256")) if parsed.get("sha256") else None,
        recycled_path=str(parsed.get("recycled_path")) if parsed.get("recycled_path") else None,
    )


def _run_undo_last_delete(args: UndoDeleteArgs, ctx: ToolContext) -> ToolResult:
    del args
    trash: TrashService = ctx.trash if ctx.trash is not None else WindowsRecycleBinTrash()
    log_path = _undo_path(ctx)
    parsed = last_pending_undo_record(log_path)
    if parsed is None:
        return ToolResult(ok=True, output="nothing to undo (no pending delete records)")
    if ctx.dry_run:
        return ToolResult(ok=True, output=f"[dry-run] would restore {parsed.get('original_path')}")
    # Fail closed: only restore into a place the policy would allow today.
    target, err = _check_target(str(parsed.get("original_path", "")), ctx)
    if err is not None:
        return ToolResult(
            ok=False,
            error=f"cannot auto-restore to {parsed.get('original_path')}: {err}",
        )
    record = _record_to_trash(parsed)
    if trash.restore(record) and target is not None and target.exists():
        mark_undo_restored(log_path, record.original_path)
        return ToolResult(
            ok=True,
            output=f"restored {record.original_path} from the Recycle Bin",
            data={"original_path": record.original_path},
        )
    return ToolResult(
        ok=False,
        error=(
            f"could not auto-restore {record.original_path} from the Recycle Bin; "
            "open the Recycle Bin and restore it manually"
        ),
    )


def _verify_undo_last_delete(
    args: UndoDeleteArgs, result: ToolResult, ctx: ToolContext
) -> ToolResult:
    del args
    if ctx.dry_run:
        return result.model_copy(update={"verified": None})
    if not result.ok:
        return result.model_copy(update={"verified": False})
    target = result.data.get("original_path")
    if not isinstance(target, str):
        return result.model_copy(update={"verified": False})
    try:
        resolved = paths.resolve_safe(target)
    except paths.PathError:
        return result.model_copy(update={"verified": False})
    return result.model_copy(update={"verified": resolved.exists()})


def make_undo_last_delete_spec() -> ToolSpec:
    """Build the ``undo_last_delete`` tool (Tier 1, best effort)."""
    return ToolSpec(
        name="undo_last_delete",
        description=(
            "Restore the most recently deleted files/folders from the Recycle "
            "Bin using the undo log. Best effort; tells you when it must be "
            "done manually."
        ),
        args_model=UndoDeleteArgs,
        base_tier=1,
        timeout_s=60,
        run=_run_undo_last_delete,
        verify=_verify_undo_last_delete,
        describe=lambda args: "Undo the most recent Recycle-Bin delete",
    )
