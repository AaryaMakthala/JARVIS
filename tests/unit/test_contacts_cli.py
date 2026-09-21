"""Contacts CLI (`jarvis contacts add|list|remove`).

Hermetic: the contact store is pinned to a temp file by monkeypatching
``cli._contacts_store``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from jarvis import cli
from jarvis.tools.whatsapp import ContactStore

runner = CliRunner()


def _patch_store(monkeypatch: pytest.MonkeyPatch, path: Path) -> ContactStore:
    store = ContactStore(path)
    monkeypatch.setattr(cli, "_contacts_store", lambda: store)
    return store


class TestContactsAdd:
    def test_add_and_list_masks_number(self, tmp_path, monkeypatch) -> None:
        _patch_store(monkeypatch, tmp_path / "contacts.json")
        add = runner.invoke(cli.app, ["contacts", "add", "Rahul", "+919876543210"])
        assert add.exit_code == 0, add.output
        listing = runner.invoke(cli.app, ["contacts", "list"])
        assert listing.exit_code == 0
        assert "Rahul" in listing.output
        assert "919876543210" not in listing.output  # raw number never shown
        assert "••••••••••10" in listing.output

    def test_invalid_number_fails(self, tmp_path, monkeypatch) -> None:
        _patch_store(monkeypatch, tmp_path / "contacts.json")
        result = runner.invoke(cli.app, ["contacts", "add", "Rahul", "123"])
        assert result.exit_code == 1
        assert "8-15 digits" in result.output

    def test_duplicate_name_fails(self, tmp_path, monkeypatch) -> None:
        store = _patch_store(monkeypatch, tmp_path / "contacts.json")
        store.add("Rahul", "+919876543210")
        result = runner.invoke(cli.app, ["contacts", "add", "rahul", "+9112345678"])
        assert result.exit_code == 1
        assert "already exists" in result.output


class TestContactsRemove:
    def test_remove(self, tmp_path, monkeypatch) -> None:
        store = _patch_store(monkeypatch, tmp_path / "contacts.json")
        store.add("Rahul", "+919876543210")
        result = runner.invoke(cli.app, ["contacts", "remove", "Rahul"])
        assert result.exit_code == 0
        assert store.load() == []

    def test_remove_unknown_fails(self, tmp_path, monkeypatch) -> None:
        _patch_store(monkeypatch, tmp_path / "contacts.json")
        result = runner.invoke(cli.app, ["contacts", "remove", "Ghost"])
        assert result.exit_code == 1
        assert "no contact named" in result.output


class TestContactsList:
    def test_empty_list(self, tmp_path, monkeypatch) -> None:
        _patch_store(monkeypatch, tmp_path / "contacts.json")
        result = runner.invoke(cli.app, ["contacts", "list"])
        assert result.exit_code == 0
        assert "no contacts yet" in result.output

    def test_sorted_by_name(self, tmp_path, monkeypatch) -> None:
        store = _patch_store(monkeypatch, tmp_path / "contacts.json")
        store.add("Zoe", "+911234567890")
        store.add("Ann", "+919876543210")
        result = runner.invoke(cli.app, ["contacts", "list"])
        assert result.output.index("Ann") < result.output.index("Zoe")
