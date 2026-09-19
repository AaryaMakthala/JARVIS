"""system_info tool: read-only machine telemetry (Tier 0).

Uses ``psutil`` exclusively - no writes, no subprocesses, no network.
Verification for read-only information is not meaningful, so ``verify()``
reports ``None`` ("unverifiable") rather than claiming success.
"""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, ConfigDict

from jarvis.tools.base import ToolContext, ToolResult, ToolSpec

_INFO_KINDS = ("ram", "cpu", "disk", "battery", "uptime", "top_processes")


class SystemInfoArgs(BaseModel):
    """Which read-only system bit to report."""

    model_config = ConfigDict(extra="forbid")

    what: Literal["ram", "cpu", "disk", "battery", "uptime", "top_processes"] = "cpu"


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
        usage = psutil.disk_usage("/")
        return ToolResult(
            ok=True,
            output=f"Disk C: {usage.used / 1e9:.1f} / {usage.total / 1e9:.1f} GB ({usage.percent}%)",
            data={
                "used_gb": round(usage.used / 1e9, 2),
                "total_gb": round(usage.total / 1e9, 2),
                "percent": usage.percent,
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
