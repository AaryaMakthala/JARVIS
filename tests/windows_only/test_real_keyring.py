"""Real backend checks - only run on a real Windows machine (or via -m windows_only)."""

from __future__ import annotations

import sys

import pytest

from jarvis.secrets import SecretStore, check_store_access

pytestmark = [
    pytest.mark.windows_only,
    pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific integration test"),
]


def test_real_keyring_access() -> None:
    backend = check_store_access()
    assert backend is not None, "OS credential store must respond to a probe round-trip"


def test_real_keyring_roundtrip() -> None:
    store = SecretStore()
    name = "__jarvis_test_name__"
    assert store.get(name) is None
    try:
        store.set(name, "roundtrip")
        assert store.get(name) == "roundtrip"
    finally:
        store.delete(name)
    assert store.get(name) is None
