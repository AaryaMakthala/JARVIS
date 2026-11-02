"""W4.2: one emoji-insensitive comparison key for contact names.

The saved display name is never changed.  These tests pin the shared key, the
unique-contact requirement, and the exact/ambiguous distinction the recipient
precheck and the WhatsApp header verification both rely on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.messaging.contacts import Contact, ContactStore
from jarvis.messaging.resolver import normalize_name, resolve_contact

_PHONE = "919876543210"


@pytest.mark.parametrize(
    ("stored", "query", "key"),
    [
        ("Mom ❤️", "Mom", "mom"),
        ("Mom 🥰", "Mom", "mom"),
        ("❤️ Mom", "Mom", "mom"),
        ("❤️ Mom 🥰", "Mom", "mom"),
        ("Mom 👨‍👩‍👧‍👦", "Mom", "mom"),
        ("Mom 👍🏽", "Mom", "mom"),
        ("My Mother 💗", "My Mother", "my mother"),
        ("Mamá ❤️", "Mamá", "mamá"),
    ],
)
def test_normalize_name_ignores_emoji_components(stored: str, query: str, key: str) -> None:
    assert normalize_name(stored) == key
    assert normalize_name(query) == key


def test_normalize_name_does_not_transliterate_accents() -> None:
    assert normalize_name("Mamá ❤️") == "mamá"
    assert normalize_name("Mama") == "mama"
    assert normalize_name("Mamá ❤️") != normalize_name("Mama")


@pytest.mark.parametrize(
    ("name", "query"),
    [
        ("Mom ❤️", "Mom"),
        ("Mom 🥰", "Mom"),
        ("❤️ Mom", "Mom"),
        ("Mom 👨‍👩‍👧‍👦", "Mom"),
        ("Mom 👍🏽", "Mom"),
        ("My Mother 💗", "My Mother"),
        ("Mamá ❤️", "Mamá"),
    ],
)
def test_a_unique_decorated_contact_is_resolved_by_its_bare_word(
    name: str, query: str
) -> None:
    contact = Contact(name, _PHONE)

    resolution = resolve_contact(query, [contact])

    assert resolution.status == "exact"
    assert resolution.selected == name


def test_the_original_saved_display_name_is_returned_unchanged() -> None:
    contact = Contact("Mom ❤️", _PHONE)

    resolution = resolve_contact("Mom", [contact])

    assert resolution.selected == "Mom ❤️"
    assert resolution.selected != "Mom"


def test_two_emoji_decorated_contacts_are_ambiguous_for_a_bare_word() -> None:
    contacts = [Contact("Mom ❤️", _PHONE), Contact("Mom 💙", "919876543211")]

    resolution = resolve_contact("Mom", contacts)

    assert resolution.status == "ambiguous"
    assert set(resolution.matches) == {"Mom ❤️", "Mom 💙"}
    assert resolution.selected == ""


def test_an_exact_decorated_name_selects_that_contact_only() -> None:
    contacts = [Contact("Mom ❤️", _PHONE), Contact("Mom 💙", "919876543211")]

    resolution = resolve_contact("Mom ❤️", contacts)

    assert resolution.status == "exact"
    assert resolution.selected == "Mom ❤️"


def test_contact_store_lookup_uses_the_same_emoji_insensitive_key(tmp_path: Path) -> None:
    store = ContactStore(tmp_path / "contacts.json")
    store.add("Mom ❤️", _PHONE)

    assert [contact.name for contact in store.resolve("Mom")] == ["Mom ❤️"]
    assert [contact.name for contact in store.resolve("❤️ Mom")] == ["Mom ❤️"]
