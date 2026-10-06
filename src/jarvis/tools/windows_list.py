"""list_windows tool: visible top-level windows and their process names (Tier 0).

Read-only and Windows-only (``PolicyEngine.decide`` refuses it on any other
host).  Titles are attacker-influenced - a web page's own title lands here - so
the result is ``tainted`` and only ever summarised for speech, never
interpreted.  :func:`enumerate_windows` is injectable: tests never touch the
real desktop.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from jarvis.tools.base import ToolContext, ToolResult, ToolSpec

#: Bounds that keep the result - and the spoken sentence - small.
MAX_WINDOWS = 25
MAX_TITLE_CHARS = 80
_SPOKEN_WINDOWS = 4
_SPOKEN_TITLE_CHARS = 60

#: One top-level window: (title, process name, is visible).  Never a full path.
WindowInfo = tuple[str, str, bool]


def enumerate_windows() -> list[WindowInfo]:
    """Enumerate top-level windows (tests replace this function).

    ``win32gui`` walks them; ``psutil`` names the process without needing
    handle rights (``win32process`` would need PROCESS_VM_READ and returns a
    full path).  Visibility is reported rather than filtered so the rule stays
    testable, and the lazy imports keep this module loadable off-Windows.
    """
    import psutil
    import win32gui
    import win32process

    found: list[WindowInfo] = []

    def _visit(hwnd: int, _extra: object) -> bool:
        try:
            title = (win32gui.GetWindowText(hwnd) or "").strip()
            visible = bool(win32gui.IsWindowVisible(hwnd))
            _thread_id, pid = win32process.GetWindowThreadProcessId(hwnd)
            found.append((title, psutil.Process(pid).name(), visible))
        except Exception:  # noqa: BLE001 - a vanished/denied window is skipped
            return True
        return True

    win32gui.EnumWindows(_visit, None)
    return found


def _summarise(windows: list[WindowInfo]) -> tuple[list[WindowInfo], str]:
    """Drop invisible/untitled windows, dedupe, truncate, cap; build the sentence."""
    seen: set[tuple[str, str]] = set()
    kept: list[WindowInfo] = []
    for title, process, visible in windows:
        title = title.strip()
        key = (title.casefold(), process.casefold())
        if not visible or not title or key in seen:
            continue
        seen.add(key)
        kept.append((title[:MAX_TITLE_CHARS], process, True))
        if len(kept) == MAX_WINDOWS:
            break
    if not kept:
        return kept, "I can't see any open windows."
    shown = [title[:_SPOKEN_TITLE_CHARS] for title, _process, _visible in kept[:_SPOKEN_WINDOWS]]
    listing = ", ".join(shown)
    if len(kept) > len(shown):
        listing += f", and {len(kept) - len(shown)} more"
    if len(kept) == 1:
        return kept, f"One window is open: {shown[0]}."
    return kept, f"{len(kept)} windows are open: {listing}."


class ListWindowsArgs(BaseModel):
    """No arguments: the tool reports what is visible right now."""

    model_config = ConfigDict(extra="forbid")


def _run_list_windows(args: ListWindowsArgs, ctx: ToolContext) -> ToolResult:
    del args
    if ctx.dry_run:
        return ToolResult(ok=True, output="[dry-run] would list the open windows")
    windows, spoken = _summarise(enumerate_windows())
    return ToolResult(
        ok=True,
        output=spoken,
        data={
            "windows": [{"title": t, "process": p} for t, p, _v in windows],
            "count": len(windows),
        },
        tainted=True,
    )


def _verify_list_windows(args: ListWindowsArgs, result: ToolResult, ctx: ToolContext) -> ToolResult:
    """Reading the window list has no observable post-condition: ``verified=None``."""
    del args, ctx
    return result.model_copy(update={"verified": None})


def make_list_windows_spec() -> ToolSpec:
    """Build the ``list_windows`` tool (Tier 0, Windows only)."""
    return ToolSpec(
        name="list_windows",
        description=(
            "List the open windows with their titles and process names. "
            "Read only; window titles can contain anything."
        ),
        args_model=ListWindowsArgs,
        base_tier=0,
        windows_only=True,
        timeout_s=10,
        run=_run_list_windows,
        verify=_verify_list_windows,
        describe=lambda args: "list the open windows",
    )
