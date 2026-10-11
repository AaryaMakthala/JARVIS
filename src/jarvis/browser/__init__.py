"""JARVIS-owned managed browser package (Stage 7, Slice 6).

A single headed, JARVIS-owned Chrome context driven through Playwright for
``browser_goto`` / ``browser_read`` / ``browser_click`` / ``browser_type``.

Importing this package never starts Chrome and never requires Playwright to be
installed (lifecycle only happens on :meth:`PlaywrightBrowser.start`).
"""

from __future__ import annotations

from jarvis.browser.context import (
    BrowserError,
    BrowserSession,
    ElementRef,
    Observation,
    PlaywrightBrowser,
    profile_lock,
)
from jarvis.browser.tools import (
    BrowserClickArgs,
    BrowserGotoArgs,
    BrowserReadArgs,
    BrowserTypeArgs,
    make_browser_click_spec,
    make_browser_goto_spec,
    make_browser_read_spec,
    make_browser_type_spec,
)

__all__ = [
    "BrowserClickArgs",
    "BrowserError",
    "BrowserGotoArgs",
    "BrowserReadArgs",
    "BrowserSession",
    "BrowserTypeArgs",
    "ElementRef",
    "Observation",
    "PlaywrightBrowser",
    "make_browser_click_spec",
    "make_browser_goto_spec",
    "make_browser_read_spec",
    "make_browser_type_spec",
    "profile_lock",
]
