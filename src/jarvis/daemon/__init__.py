"""Daemon package: IPC server, client, protocol, and autostart (Phase 3).

Binds to ``127.0.0.1`` and requires a token for authentication.
"""

from jarvis.daemon.autostart import disable as autostart_disable
from jarvis.daemon.autostart import enable as autostart_enable
from jarvis.daemon.autostart import status as autostart_status
from jarvis.daemon.client import DaemonClient, DaemonError
from jarvis.daemon.server import DaemonServer

__all__ = [
    "DaemonClient",
    "DaemonError",
    "DaemonServer",
    "autostart_disable",
    "autostart_enable",
    "autostart_status",
]
