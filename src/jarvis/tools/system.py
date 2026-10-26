"""System tools: ``system_info`` (read-only telemetry) plus the two locks.

``system_info`` uses ``psutil`` exclusively - no writes, no subprocesses, no
network.  Verification for read-only information is not meaningful, so
``verify()`` reports ``None`` ("unverifiable") rather than claiming success.

``lock_jarvis`` (Tier 0) locks only this agent's permission session;
``lock_computer`` (Tier 1, docs/04 §2.4) locks the Windows workstation via
``user32.LockWorkStation`` and verifies nothing (the lock screen is not
observable from user mode, so ``verified=None``).

All Windows-specific imports (``ctypes.windll``) live inside the run functions
behind the platform guard, so this module imports on any OS for tests.
"""

from __future__ import annotations

import os
import platform
import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from jarvis.tools.base import ToolContext, ToolResult, ToolSpec

_INFO_KINDS = ("ram", "cpu", "disk", "battery", "uptime", "top_processes")


class _BackendUnavailable(RuntimeError):
    """An optional third-party backend is missing or cannot start."""


try:
    from comtypes import COMError as _ComError
except ImportError:  # pragma: no cover - comtypes is a Windows-only extra
    _ComError = OSError

#: Errors opening the COM audio endpoint through pycaw/comtypes.
_VOLUME_ENDPOINT_ERRORS: tuple[type[BaseException], ...] = (
    _ComError,
    OSError,
    ImportError,
    AttributeError,
)


class SystemInfoArgs(BaseModel):
    """Which read-only system bit to report."""

    model_config = ConfigDict(extra="forbid")

    what: Literal["ram", "cpu", "disk", "battery", "uptime", "top_processes"] = "cpu"


def _disk_target() -> tuple[str, str]:
    """The disk to report on and how to label it.

    The old code measured ``psutil.disk_usage("/")`` (the current drive on
    Windows) but always called it "Disk C:" - wrong on a machine whose system
    drive is not C.  The label now follows the drive that was actually
    measured.
    """
    if platform.system() == "Windows":
        drive = (os.environ.get("SystemDrive") or "C:").strip().rstrip("\\/")
        if not drive:
            drive = "C:"
        if not drive.endswith(":"):
            drive = f"{drive}:"
        return drive, f"{drive}\\"
    return "root", "/"


def _read(args: SystemInfoArgs) -> ToolResult:
    import psutil

    kind = args.what
    if kind == "ram":
        vm = psutil.virtual_memory()
        return ToolResult(
            ok=True,
            output=f"RAM: {vm.used / 1e9:.1f} / {vm.total / 1e9:.1f} GB ({vm.percent}%)",
            data={
                "used_gb": round(vm.used / 1e9, 2),
                "total_gb": round(vm.total / 1e9, 2),
                "percent": vm.percent,
            },
        )
    if kind == "cpu":
        percent = psutil.cpu_percent(interval=0.1)
        count = psutil.cpu_count()
        return ToolResult(
            ok=True,
            output=f"CPU: {percent}% across {count} cores",
            data={"percent": percent, "cores": count},
        )
    if kind == "disk":
        label, path = _disk_target()
        usage = psutil.disk_usage(path)
        return ToolResult(
            ok=True,
            output=f"Disk {label}: {usage.used / 1e9:.1f} / {usage.total / 1e9:.1f} GB ({usage.percent}%)",
            data={
                "used_gb": round(usage.used / 1e9, 2),
                "total_gb": round(usage.total / 1e9, 2),
                "percent": usage.percent,
                "label": label,
                "path": path,
            },
        )
    if kind == "battery":
        battery = psutil.sensors_battery()
        if battery is None:
            return ToolResult(ok=True, output="Battery: not present", data={"present": False})
        percent = battery.percent
        plugged = bool(battery.power_plugged)
        state = "charging" if plugged else "on battery"
        return ToolResult(
            ok=True,
            output=f"Battery: {percent}% ({state})",
            data={"percent": percent, "plugged": plugged},
        )
    if kind == "uptime":
        seconds = int(time.time() - psutil.boot_time())
        hours, remainder = divmod(seconds, 3600)
        minutes = remainder // 60
        return ToolResult(ok=True, output=f"Uptime: {hours}h {minutes}m", data={"seconds": seconds})
    # top_processes
    procs = [
        p.info
        for p in psutil.process_iter(["name", "memory_percent"])
        if p.info.get("name") is not None
    ]
    top = sorted(procs, key=lambda rec: float(rec.get("memory_percent") or 0), reverse=True)[:5]
    names = [f"{rec['name']} ({rec.get('memory_percent') or 0:.1f}%)" for rec in top]
    return ToolResult(ok=True, output="top processes: " + ", ".join(names), data={"top": names})


def _run_system_info(args: SystemInfoArgs, ctx: ToolContext) -> ToolResult:
    if ctx.dry_run:
        return ToolResult(ok=True, output=f"[dry-run] would read system info: {args.what}")
    return _read(args)


def _verify_system_info(args: SystemInfoArgs, result: ToolResult, ctx: ToolContext) -> ToolResult:
    del args, ctx
    return result.model_copy(update={"verified": None})


def _describe_system_info(args: SystemInfoArgs) -> str:
    return f"Read system info: {args.what}"


def make_system_info_spec() -> ToolSpec:
    """Build the ``system_info`` tool (Tier 0)."""
    return ToolSpec(
        name="system_info",
        description=(
            "Read-only machine information (RAM, CPU, disk, battery, uptime, "
            "top processes). Never modifies anything."
        ),
        args_model=SystemInfoArgs,
        base_tier=0,
        timeout_s=10,
        run=_run_system_info,
        verify=_verify_system_info,
        describe=_describe_system_info,
    )


class ProcessListArgs(BaseModel):
    """List running process names and pids (no command lines or users)."""

    model_config = ConfigDict(extra="forbid")


def _load_psutil() -> Any:
    """Import psutil lazily so the module imports on every OS."""
    try:
        import psutil
    except ImportError as exc:
        raise _BackendUnavailable("process listing needs the optional 'psutil' package") from exc
    return psutil


def _run_process_list(args: ProcessListArgs, ctx: ToolContext) -> ToolResult:
    del args, ctx
    try:
        psutil = _load_psutil()
        rows: list[dict[str, Any]] = []
        for proc in psutil.process_iter(["pid", "name"]):
            info = proc.info
            name = info.get("name")
            pid = info.get("pid")
            if name is None or pid is None:
                continue
            rows.append({"name": str(name), "pid": int(pid)})
            if len(rows) >= 40:
                break
    except _BackendUnavailable as exc:
        return ToolResult(ok=False, error=str(exc))
    except Exception as exc:  # noqa: BLE001 - psutil raises platform-specific errors
        return ToolResult(ok=False, error=f"could not list processes: {exc}")
    output = "\n".join(f"{row['name']} ({row['pid']})" for row in rows) or "No processes found."
    return ToolResult(ok=True, output=output, data={"processes": rows, "count": len(rows)})


def _verify_process_list(
    args: ProcessListArgs, result: ToolResult, ctx: ToolContext
) -> ToolResult:
    del args, ctx
    return result.model_copy(update={"verified": None})


def make_process_list_spec() -> ToolSpec:
    """Build ``process_list`` (Tier 1; read-only process names)."""
    return ToolSpec(
        name="process_list",
        description=(
            "List up to 40 running process names with their pids. Read-only: "
            "no command lines, usernames or environment values."
        ),
        args_model=ProcessListArgs,
        base_tier=1,
        timeout_s=15,
        run=_run_process_list,
        verify=_verify_process_list,
        describe=lambda args: "List running processes",
    )


class VolumeControlArgs(BaseModel):
    """Read or change the Windows master volume."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["get", "set", "mute", "unmute"]
    level: int | None = Field(default=None, ge=0, le=100, description="Volume percent for 'set'.")

    @model_validator(mode="after")
    def _level_required_for_set(self) -> VolumeControlArgs:
        if self.action == "set" and self.level is None:
            raise ValueError("level is required when action is 'set'")
        return self


def _load_volume_endpoint() -> Any:
    """Open the Windows master-volume endpoint through optional pycaw."""
    try:
        from pycaw.pycaw import AudioUtilities
    except ImportError as exc:
        raise _BackendUnavailable(
            "volume control needs the optional 'pycaw' package"
        ) from exc
    try:
        speakers = AudioUtilities.GetSpeakers()
        endpoint = getattr(speakers, "EndpointVolume", None)
        if endpoint is not None:
            return endpoint
        from ctypes import POINTER, cast

        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import IAudioEndpointVolume

        interface = speakers.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        return cast(interface, POINTER(IAudioEndpointVolume))
    except _VOLUME_ENDPOINT_ERRORS as exc:
        raise _BackendUnavailable(f"could not open the Windows audio endpoint: {exc}") from exc


def _run_volume_control(args: VolumeControlArgs, ctx: ToolContext) -> ToolResult:
    del ctx
    try:
        endpoint = _load_volume_endpoint()
        if args.action == "set":
            endpoint.SetMasterVolumeLevelScalar(float(args.level or 0) / 100.0, None)
        elif args.action == "mute":
            endpoint.SetMute(1, None)
        elif args.action == "unmute":
            endpoint.SetMute(0, None)
        percent = round(float(endpoint.GetMasterVolumeLevelScalar()) * 100)
        muted = bool(endpoint.GetMute())
    except _BackendUnavailable as exc:
        return ToolResult(ok=False, error=str(exc))
    except Exception as exc:  # noqa: BLE001 - pycaw/COM errors are platform-specific
        return ToolResult(ok=False, error=f"volume control failed: {exc}")
    state = "muted" if muted else "on"
    return ToolResult(
        ok=True,
        output=f"Volume: {percent}% ({state})",
        data={"level": percent, "muted": muted},
    )


def _verify_volume_control(
    args: VolumeControlArgs, result: ToolResult, ctx: ToolContext
) -> ToolResult:
    del args, ctx
    return result.model_copy(update={"verified": None})


def make_volume_control_spec() -> ToolSpec:
    """Build ``volume_control`` (Tier 1; changes the master volume)."""
    return ToolSpec(
        name="volume_control",
        description=(
            "Read or change the Windows master volume: get, set a 0-100 level, "
            "mute or unmute."
        ),
        args_model=VolumeControlArgs,
        base_tier=1,
        timeout_s=15,
        run=_run_volume_control,
        verify=_verify_volume_control,
        describe=lambda args: f"Volume control: {args.action}",
        windows_only=True,
    )


class LockJarvisArgs(BaseModel):
    """Lock the current JARVIS session (no further Tier-2 actions)."""

    model_config = ConfigDict(extra="forbid")


def _run_lock_jarvis(args: LockJarvisArgs, ctx: ToolContext) -> ToolResult:
    del args
    if ctx.unlock is None:
        return ToolResult(ok=False, error="no unlock manager available in this session")
    ctx.unlock.lock()
    return ToolResult(
        ok=True, output="JARVIS session locked (Tier 2 actions now refused)", data={"locked": True}
    )


def _verify_lock_jarvis(args: LockJarvisArgs, result: ToolResult, ctx: ToolContext) -> ToolResult:
    del args
    if ctx.unlock is None:
        return result.model_copy(update={"verified": False})
    locked = not ctx.unlock.is_unlocked()
    return result.model_copy(update={"verified": bool(locked)})


def make_lock_jarvis_spec() -> ToolSpec:
    """Build the ``lock_jarvis`` tool (Tier 0; locking is harmless)."""
    return ToolSpec(
        name="lock_jarvis",
        description=(
            "Lock the JARVIS session: after this, Tier-2 actions (deletes, "
            "anything requiring the password) are refused until the password "
            "is entered again. Safe to call at any time."
        ),
        args_model=LockJarvisArgs,
        base_tier=0,
        timeout_s=10,
        run=_run_lock_jarvis,
        verify=_verify_lock_jarvis,
        describe=lambda args: "Lock the JARVIS session",
    )


class LockComputerArgs(BaseModel):
    """Lock the Windows workstation itself (``LockWorkStation``)."""

    model_config = ConfigDict(extra="forbid")


#: Exit code ``win32api``/``ctypes`` report when ``LockWorkStation`` is called
#: from a non-interactive session (e.g. the daemon runs as a service).  The
#: docs/04 spec promises honest failure, not a retry loop.
_ACCESS_DENIED_EXIT = 5


def _run_lock_computer(args: LockComputerArgs, ctx: ToolContext) -> ToolResult:
    """Call ``user32.LockWorkStation``; best effort, never raises (docs/04)."""
    del args
    if ctx.dry_run:
        return ToolResult(ok=True, output="[dry-run] would lock the Windows workstation")
    try:
        import ctypes

        result_code = ctypes.windll.user32.LockWorkStation()
    except Exception as exc:  # noqa: BLE001 - never raise to the graph
        return ToolResult(ok=False, error=f"LockWorkStation failed: {exc}")
    if result_code == 0:
        error_code = getattr(ctypes, "get_last_error", int)() or 5
        if error_code == _ACCESS_DENIED_EXIT:
            return ToolResult(
                ok=False,
                error="could not lock the workstation (access denied; "
                "the process is not running in your interactive session)",
            )
        return ToolResult(
            ok=False,
            error=f"could not lock the workstation (Win32 error {error_code})",
        )
    return ToolResult(
        ok=True,
        output="Windows workstation locked",
        data={"locked": True},
    )


#: Reported when ``lock_computer`` succeeded: ``LockWorkStation`` is
#: asynchronous (winlogon switches to the secure desktop after it returns) and
#: the lock state is not readable from a user-mode process, so there is no
#: deterministic post-condition to check.  ``verified`` therefore stays ``None``
#: (docs/04 §2.4: "Best effort"), and this note keeps the final answer
#: accurate; the old generic "could not be independently verified" wording is gone (4.14).
_LOCK_VERIFY_NOTE = (
    "best-effort action: the Windows lock screen cannot be observed from this process"
)


def _verify_lock_computer(
    args: LockComputerArgs, result: ToolResult, ctx: ToolContext
) -> ToolResult:
    """Best-effort verification: the lock screen state is not observable (docs/04).

    Deliberately does **not** poll, sleep or retry: ``LockWorkStation`` returns
    before winlogon finishes the switch, so an immediate probe would report a
    false failure for a lock that is in fact engaging.  ``verified=None`` +
    :data:`_LOCK_VERIFY_NOTE` is the honest result.
    """
    del args, ctx
    return result.model_copy(update={"verified": None, "verify_note": _LOCK_VERIFY_NOTE})


def make_lock_computer_spec() -> ToolSpec:
    """Build the ``lock_computer`` tool (Tier 1; ends the interactive session's
    usability until the user returns, so docs/04 §2.4 lists it as Tier 1)."""
    return ToolSpec(
        name="lock_computer",
        description=(
            "Lock the Windows computer (LockWorkStation). The desktop locks "
            "immediately; the user signs back in to resume. Not undoable "
            "remotely."
        ),
        args_model=LockComputerArgs,
        base_tier=1,
        timeout_s=10,
        run=_run_lock_computer,
        verify=_verify_lock_computer,
        describe=lambda args: "Lock the Windows computer",
        windows_only=True,
    )
