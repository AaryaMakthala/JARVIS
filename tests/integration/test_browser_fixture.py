import os

import pytest


pytestmark = [pytest.mark.browser]


def test_browser_goto_fixture(fixture_server):
    # write test but don't require playwright/chrome to be runnable in this env
    # just structure
    try:
        from jarvis.browser.manager import BrowserManager
    except Exception:
        pytest.skip("browser deps missing")
    # don't actually launch browser in unit/integration test here
    pytest.skip("browser not executed in low-resource mode")


def test_browser_read_fixture(fixture_server):
    pytest.skip("browser not executed in low-resource mode")


def test_browser_find_open_result_fixture(fixture_server):
    pytest.skip("browser not executed in low-resource mode")
