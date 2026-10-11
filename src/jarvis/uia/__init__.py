"""Native Windows UI Automation package (Stage 7, Slice 7).

Two tools - ``ui_list_controls`` and ``ui_click`` - drive identifiable native
application controls through UI Automation.  Importing this package never
imports pywinauto and never touches a desktop; that only happens lazily inside
the real provider's methods.
"""

from __future__ import annotations

from jarvis.uia.tools import (
    ControlIdentity,
    ControlLookup,
    PywinautoUiaProvider,
    UiaClickArgs,
    UiaControl,
    UiaListControlsArgs,
    UiaProvider,
    UiaWindow,
    control_ref,
    make_ui_click_spec,
    make_ui_list_controls_spec,
)

__all__ = [
    "ControlIdentity",
    "ControlLookup",
    "PywinautoUiaProvider",
    "UiaClickArgs",
    "UiaControl",
    "UiaListControlsArgs",
    "UiaProvider",
    "UiaWindow",
    "control_ref",
    "make_ui_click_spec",
    "make_ui_list_controls_spec",
]
