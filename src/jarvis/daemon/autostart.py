"""Daemon autostart via Windows Task Scheduler.

Creates a task named ``JarvisAgent`` that runs ``pythonw -m jarvis daemon``
at log-on of the current user.  Uses ``schtasks /Create /XML`` for full
control over settings (restart on failure, limited user, etc.).

Not a Windows Service — services run in session 0 and cannot access the
user's desktop, audio, or windows.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

from jarvis.platform_guard import is_windows

logger = logging.getLogger(__name__)

TASK_NAME = "JarvisAgent"


def _pythonw_path() -> str:
    """Resolve the ``pythonw.exe`` in the same venv as the running interpreter."""
    python = Path(sys.executable)
    pythonw = python.parent / "pythonw.exe"
    if pythonw.exists():
        return str(pythonw)
    # Fallback: use the same executable (works outside venv)
    return str(python)


def _generate_xml() -> str:
    """Generate the Task Scheduler XML definition."""
    pythonw = _pythonw_path()
    # Escape XML special characters in the path
    pythonw_escaped = pythonw.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    return textwrap.dedent(f"""\
        <?xml version="1.0" encoding="UTF-16"?>
        <Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
          <RegistrationInfo>
            <Description>JARVIS desktop AI agent daemon</Description>
            <Author>{os.environ.get("USERNAME", "user")}</Author>
          </RegistrationInfo>
          <Triggers>
            <LogonTrigger>
              <Enabled>true</Enabled>
              <Delay>PT20S</Delay>
            </LogonTrigger>
          </Triggers>
          <Principals>
            <Principal id="Author">
              <LogonType>InteractiveToken</LogonType>
              <RunLevel>LeastPrivilege</RunLevel>
            </Principal>
          </Principals>
          <Settings>
            <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
            <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
            <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
            <AllowHardTerminate>true</AllowHardTerminate>
            <StartWhenAvailable>true</StartWhenAvailable>
            <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
            <IdleSettings>
              <StopOnIdleEnd>false</StopOnIdleEnd>
              <RestartOnIdle>false</RestartOnIdle>
            </IdleSettings>
            <AllowStartOnDemand>true</AllowStartOnDemand>
            <Enabled>true</Enabled>
            <Hidden>false</Hidden>
            <RunOnlyIfIdle>false</RunOnlyIfIdle>
            <WakeToRun>false</WakeToRun>
            <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
            <Priority>7</Priority>
            <RestartOnFailure>
              <Interval>PT1M</Interval>
              <Count>3</Count>
            </RestartOnFailure>
          </Settings>
          <Actions Context="Author">
            <Exec>
              <Command>{pythonw_escaped}</Command>
              <Arguments>-m jarvis daemon</Arguments>
            </Exec>
          </Actions>
        </Task>
    """)


def enable() -> dict[str, Any]:
    """Create the ``JarvisAgent`` scheduled task.

    Returns a dict with status information.  Requires ``schtasks.exe`` on PATH.
    """
    if not is_windows():
        return {"ok": False, "error": "autostart is only available on Windows"}

    import shutil
    import subprocess

    schtasks = shutil.which("schtasks")
    if not schtasks:
        return {"ok": False, "error": "schtasks.exe not found on PATH"}

    xml_content = _generate_xml()
    xml_path = Path(tempfile.mktemp(suffix=".xml", prefix="jarvis_task_"))
    try:
        xml_path.write_text(xml_content, encoding="utf-16")
        result = subprocess.run(
            [
                schtasks,
                "/Create",
                "/TN",
                TASK_NAME,
                "/XML",
                str(xml_path),
                "/F",  # force overwrite
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode == 0:
            logger.info("autostart task created: %s", TASK_NAME)
            return {"ok": True, "message": f"task '{TASK_NAME}' created"}
        else:
            error_msg = (result.stdout + result.stderr).strip()
            logger.error("schtasks /Create failed: %s", error_msg)
            return {"ok": False, "error": error_msg}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "schtasks timed out"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}
    finally:
        xml_path.unlink(missing_ok=True)


def disable() -> dict[str, Any]:
    """Delete the ``JarvisAgent`` scheduled task."""
    if not is_windows():
        return {"ok": False, "error": "autostart is only available on Windows"}

    import shutil
    import subprocess

    schtasks = shutil.which("schtasks")
    if not schtasks:
        return {"ok": False, "error": "schtasks.exe not found on PATH"}

    result = subprocess.run(
        [schtasks, "/Delete", "/TN", TASK_NAME, "/F"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode == 0:
        logger.info("autostart task deleted: %s", TASK_NAME)
        return {"ok": True, "message": f"task '{TASK_NAME}' deleted"}
    else:
        error_msg = (result.stdout + result.stderr).strip()
        return {"ok": False, "error": error_msg}


def status() -> dict[str, Any]:
    """Query the ``JarvisAgent`` scheduled task status."""
    if not is_windows():
        return {"ok": False, "error": "autostart is only available on Windows"}

    import shutil
    import subprocess

    schtasks = shutil.which("schtasks")
    if not schtasks:
        return {"ok": False, "error": "schtasks.exe not found on PATH"}

    result = subprocess.run(
        [schtasks, "/Query", "/TN", TASK_NAME, "/FO", "LIST", "/V"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        return {"ok": True, "enabled": False, "message": "task not found"}

    # Parse the output for key fields
    output = result.stdout
    info: dict[str, Any] = {"ok": True, "enabled": True, "raw": output}

    for line in output.splitlines():
        if line.startswith("Status:"):
            info["status"] = line.split(":", 1)[1].strip()
        elif line.startswith("Last Run Time:"):
            info["last_run"] = line.split(":", 1)[1].strip()
        elif line.startswith("Next Run Time:"):
            info["next_run"] = line.split(":", 1)[1].strip()
        elif line.startswith("Last Result:"):
            info["last_result"] = line.split(":", 1)[1].strip()

    return info
