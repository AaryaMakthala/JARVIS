"""The benchmark sandbox: a throwaway root that stands in for the user's files.

docs/07 section 3.1 requires every file task to run inside ``benchmarks/sandbox``
"mapped as the allowed root during benchmarks", and section 4.1 requires the
runner to reset it before each repeat.  This module owns that directory plus the
two substitutes that make a benchmark run genuinely side-effect free:

* :class:`SandboxTrash` - a :class:`~jarvis.tools.files.TrashService` that moves
  deletions into a folder *inside* the sandbox instead of the real Windows
  Recycle Bin, and can put them back.  Without it, benchmarking ``delete_path``
  would put the developer's own files in the bin.
* :func:`reset_sandbox` - empties the directory, refusing to touch anything that
  is not plausibly a sandbox.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from jarvis.tools.files import TrashRecord

__all__ = ["SandboxTrash", "reset_sandbox", "write_setup_files"]


class SandboxTrash:
    """A Recycle Bin that lives inside the benchmark sandbox.

    Implements the ``TrashService`` protocol (``send``/``restore``), so the real
    delete tools and the real undo log run unchanged - only the destination of
    the bytes changes.
    """

    def __init__(self, sandbox: Path) -> None:
        self.sandbox = Path(sandbox)
        self.bin = self.sandbox / ".trash"

    def send(self, path: Path) -> TrashRecord:
        """Move ``path`` into the sandbox bin and record it."""
        path = Path(path)
        if not path.exists():
            return TrashRecord(str(path), "file", 0, 0, None, error="path no longer exists")
        try:
            digest = _sha256(path)
        except OSError:
            digest = None
        kind = "dir" if path.is_dir() else "file"
        size, count = _size_and_count(path)
        self.bin.mkdir(parents=True, exist_ok=True)
        target = self.bin / _unique_name(path.name)
        try:
            shutil.move(str(path), str(target))
        except OSError as exc:
            return TrashRecord(
                str(path), kind, size, count, digest, error=f"could not move to sandbox bin: {exc}"
            )
        return TrashRecord(str(path), kind, size, count, digest, recycled_path=str(target))

    def restore(self, record: TrashRecord) -> bool:
        """Move a previously recycled item back to its original path."""
        if not record.recycled_path:
            return False
        source = Path(record.recycled_path)
        if not source.exists():
            return False
        target = Path(record.original_path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
        except OSError:
            return False
        return target.exists()


def reset_sandbox(sandbox: str | Path) -> Path:
    """Delete everything inside ``sandbox`` and return the empty directory.

    Refuses a root, a home directory, or the drive itself: this function deletes
    recursively, so a mistyped ``--sandbox C:\\`` must fail loudly rather than
    wipe a disk.
    """
    root = Path(sandbox).expanduser().resolve(strict=False)
    if root.parent == root:
        raise ValueError(f"refusing to reset a filesystem root: {root}")
    if root.name.lower() in ("", "users", "windows", "program files", "program files (x86)"):
        raise ValueError(f"refusing to reset {root}: it does not look like a sandbox")
    if root.exists():
        for child in root.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child, ignore_errors=True)
            else:
                try:
                    child.unlink()
                except OSError:
                    pass
    root.mkdir(parents=True, exist_ok=True)
    return root


def write_setup_files(sandbox: Path, setup: list) -> list[Path]:
    """Create a task's ``setup`` files inside the sandbox.

    Returns the created paths so a failure can name the offending entry.  Paths
    are resolved against the sandbox root and an escape is an error, not a write.
    """
    from jarvis.benchmark.checkers import resolve_in_sandbox

    created: list[Path] = []
    for item in setup:
        target = resolve_in_sandbox(sandbox, item.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(item.content, encoding="utf-8")
        created.append(target)
    return created


def _unique_name(name: str) -> str:
    """A collision-proof name inside the bin, so two same-named files coexist."""
    stem = Path(name).name or "item"
    return f"{stem}.{hashlib.sha1(str(name).encode()).hexdigest()[:8]}"


def _sha256(path: Path) -> str | None:
    import hashlib as _hashlib

    digest = _hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _size_and_count(path: Path) -> tuple[int, int]:
    if path.is_file():
        try:
            return path.stat().st_size, 1
        except OSError:
            return 0, 0
    size = 0
    count = 0
    try:
        for child in path.rglob("*"):
            count += 1
            if child.is_file():
                size += child.stat().st_size
    except OSError:
        pass
    return size, count
