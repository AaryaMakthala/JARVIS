"""W4.3: constrained natural-language contact grounding.

The model may interpret a reference, but ContactStore stays the identity
authority.  These tests use scripted, tool-free fakes: no network, no browser,
no live WhatsApp.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from jarvis import config
from jarvis.agent.context import make_app_context
from jarvis.agent.graph import open_sqlite_checkpointer
from jarvis.agent.runner import run_task
from jarvis.agent.state import Step
from jarvis.config import Settings
from jarvis.messaging.contacts import ContactStore
from jarvis.messaging.grounding import ground_contacts
from jarvis.policy import rules
from jarvis.tools.registry import build_default_registry
from jarvis.tools.whatsapp_web import WhatsAppSendMessageArgs
from support import brain_action

_PHONE = "919876543210"
_DAD_PHONE = "919876543211"
_SECOND_MOM_PHONE = "919876543212"


@pytest.fixture(autouse=True)
def _windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test runs as if on Windows unless it overrides this."""
    monkeypatch.setattr("jarvis.platform_guard.is_windows", lambda: True)


class _ScriptedLLM:
    """Tool-free text model that records every prompt and returns one answer."""

    def __init__(self, answer: str = "C1", *, error: Exception | None = None) -> None:
        self.answer = answer
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def text(self, **kwargs: Any) -> tuple[str, Any]:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.answer, _Usage()


class _Usage:
    calls = 1
    prompt_tokens = 7
    completion_tokens = 3


class _NeverCalledLLM:
    def text(self, **kwargs: Any) -> tuple[str, Any]:
        raise AssertionError("the model must not be called for a deterministic match")


def _store(tmp_path: Path, *contacts: tuple[str, str]) -> ContactStore:
    store = ContactStore(tmp_path / "contacts.json")
    for name, number in contacts or (("Mom ❤️", _PHONE),):
        store.add(name, number)
    return store


def _step(contact: str, *, message: str = "hi") -> Step:
    return Step(
        id="s1",
        tool="whatsapp_send_message",
        args={"contact": contact, "message": message},
        rationale="send it",
    )


# ── deterministic first ───────────────────────────────────────────────────


def test_an_exact_normalized_match_needs_no_llm(tmp_path: Path) -> None:
    store = _store(tmp_path, ("Mom ❤️", _PHONE))
    step = _step("Mom")

    outcome = ground_contacts([step], store=store, llm=_NeverCalledLLM())

    assert outcome.clarification is None and outcome.halted_reason is None
    assert outcome.api_calls == 0
    assert step.args["contact"] == "Mom ❤️"


def test_an_exact_decorated_name_needs_no_llm(tmp_path: Path) -> None:
    store = _store(tmp_path, ("Mom ❤️", _PHONE), ("Mom 💙", _SECOND_MOM_PHONE))
    step = _step("Mom ❤️")

    outcome = ground_contacts([step], store=store, llm=_NeverCalledLLM())

    assert outcome.halted_reason is None
    assert step.args["contact"] == "Mom ❤️"


def test_an_accented_emoji_name_is_supported(tmp_path: Path) -> None:
    store = _store(tmp_path, ("Mamá ❤️", _PHONE))
    step = _step("Mamá")

    ground_contacts([step], store=store, llm=_NeverCalledLLM())

    assert step.args["contact"] == "Mamá ❤️"


# ── semantic interpretation ───────────────────────────────────────────────


def test_my_mother_maps_to_the_unique_saved_contact(tmp_path: Path) -> None:
    store = _store(tmp_path, ("Mom ❤️", _PHONE))
    step = _step("my mother")
    llm = _ScriptedLLM("C1")

    outcome = ground_contacts([step], store=store, llm=llm)

    assert outcome.clarification is None and outcome.halted_reason is None
    assert step.args["contact"] == "Mom ❤️"
    assert outcome.api_calls == 1
    assert outcome.tokens == 10
    assert len(llm.calls) == 1


def test_my_dad_maps_to_the_unique_saved_contact(tmp_path: Path) -> None:
    store = _store(tmp_path, ("Mom ❤️", _PHONE), ("Father 💙", _DAD_PHONE))
    step = _step("my dad")
    llm = _ScriptedLLM("C2")

    ground_contacts([step], store=store, llm=llm)

    assert step.args["contact"] == "Father 💙"


def test_the_prompt_contains_display_names_but_no_phone_numbers(tmp_path: Path) -> None:
    store = _store(tmp_path, ("Mom ❤️", _PHONE), ("Father 💙", _DAD_PHONE))
    llm = _ScriptedLLM("C1")

    ground_contacts([_step("my mother")], store=store, llm=llm)

    user = llm.calls[0]["user"]
    assert "Mom ❤️" in user and "Father 💙" in user
    assert _PHONE not in user and _DAD_PHONE not in user
    assert "C1:" in user and "C2:" in user
    assert "never invent" in llm.calls[0]["system"].lower()


def test_no_candidates_refuses_without_calling_the_model(tmp_path: Path) -> None:
    store = ContactStore(tmp_path / "empty.json")
    step = _step("my mother")

    outcome = ground_contacts([step], store=store, llm=_NeverCalledLLM())

    assert outcome.halted_reason
    assert "couldn't find a matching saved contact" in outcome.halted_reason


def test_multiple_plausible_candidates_produce_ambiguity(tmp_path: Path) -> None:
    """"my mom" resolves to no exact key, and any choice shares the key."""
    store = _store(tmp_path, ("Mom ❤️", _PHONE), ("Mom 💙", _SECOND_MOM_PHONE))
    step = _step("my mom")
    llm = _ScriptedLLM("C1")

    outcome = ground_contacts([step], store=store, llm=llm)

    assert outcome.clarification
    assert "multiple possible contacts" in outcome.clarification.lower()
    assert "Mom ❤️" in outcome.clarification and "Mom 💙" in outcome.clarification
    assert step.args["contact"] == "my mom"  # no arbitrary rewrite


def test_a_deterministically_ambiguous_query_asks_without_a_model(tmp_path: Path) -> None:
    store = _store(tmp_path, ("Mom ❤️", _PHONE), ("Mom 💙", _SECOND_MOM_PHONE))
    step = _step("Mom")

    outcome = ground_contacts([step], store=store, llm=_NeverCalledLLM())

    assert outcome.clarification
    assert "Mom ❤️" in outcome.clarification and "Mom 💙" in outcome.clarification


# ── model output validation ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "answer",
    ["UNCERTAIN", "C99", "Mom ❤️", "919876543210", "", "C1 extra"],
)
def test_invalid_or_uncertain_output_is_refused(tmp_path: Path, answer: str) -> None:
    store = _store(tmp_path, ("Mom ❤️", _PHONE))
    step = _step("my mother")

    outcome = ground_contacts([step], store=store, llm=_ScriptedLLM(answer))

    assert outcome.halted_reason
    assert "couldn't reliably identify" in outcome.halted_reason
    assert step.args["contact"] == "my mother"


def test_none_is_a_no_match_refusal_not_a_malformed_answer(tmp_path: Path) -> None:
    """``NONE`` is the model saying "nobody fits" - its own distinct message."""
    store = _store(tmp_path, ("Mom ❤️", _PHONE))
    step = _step("my dentist")

    outcome = ground_contacts([step], store=store, llm=_ScriptedLLM("NONE"))

    assert outcome.halted_reason
    assert "couldn't find a matching saved contact" in outcome.halted_reason
    assert step.args["contact"] == "my dentist"


def test_a_model_failure_is_a_visible_refusal(tmp_path: Path) -> None:
    store = _store(tmp_path, ("Mom ❤️", _PHONE))
    step = _step("my mother")

    outcome = ground_contacts(
        [step], store=store, llm=_ScriptedLLM(error=RuntimeError("provider down"))
    )

    assert outcome.halted_reason
    assert "couldn't reliably identify" in outcome.halted_reason


def test_the_model_cannot_change_the_message_or_create_a_step(tmp_path: Path) -> None:
    store = _store(tmp_path, ("Mom ❤️", _PHONE))
    step = _step("my mother", message="I'll be late")

    ground_contacts([step], store=store, llm=_ScriptedLLM("C1"))

    assert step.args == {"contact": "Mom ❤️", "message": "I'll be late"}


def test_grounding_only_targets_whatsapp_contact_tools(tmp_path: Path) -> None:
    store = _store(tmp_path, ("Mom ❤️", _PHONE))
    other = Step(id="s1", tool="fake_echo", args={"contact": "my mother"}, rationale="x")

    outcome = ground_contacts([other], store=store, llm=_NeverCalledLLM())

    assert outcome.halted_reason is None
    assert other.args["contact"] == "my mother"


# ── confirmation / action binding ─────────────────────────────────────────


def _run_grounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contacts: list[tuple[str, str]],
    *,
    contact: str,
    message: str,
    llm_answer: str | None = None,
) -> Any:
    """Run the real graph with a canonicalizable contact reference."""
    store = _store(tmp_path, *contacts)
    monkeypatch.setattr(config, "contacts_file", lambda: store.path)
    from jarvis.llm.client import FakeLLM

    responses: list[Any] = [
        brain_action(
            "whatsapp_send_message",
            {"contact": contact, "message": message},
        )
    ]
    if llm_answer is not None:
        responses.append(llm_answer)
    ctx = make_app_context(
        Settings(),
        llm=FakeLLM(responses),
        registry=build_default_registry(),
    )
    return run_task(ctx, open_sqlite_checkpointer(str(tmp_path / "cp.db")), "send it")


def test_confirmation_shows_the_canonical_contact_and_binds_the_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcome = _run_grounded(
        tmp_path,
        monkeypatch,
        [("Mom ❤️", _PHONE)],
        contact="my mother",
        message="I'll be late",
        llm_answer="C1",
    )

    assert outcome.interrupted is True
    assert outcome.interrupt_kind == "confirm"
    assert outcome.confirmation is not None
    assert "Mom ❤️" in outcome.confirmation["summary"]
    assert "I'll be late" in outcome.confirmation["summary"]
    plan = outcome.state["plan"]
    assert plan.steps[0].args == {"contact": "Mom ❤️", "message": "I'll be late"}
    spec = build_default_registry().get("whatsapp_send_message")
    expected = rules.action_hash(
        spec, WhatsAppSendMessageArgs(contact="Mom ❤️", message="I'll be late")
    )
    assert outcome.confirmation["action_hash"] == expected
    assert outcome.results == []


def test_an_ambiguous_reference_asks_a_visible_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcome = _run_grounded(
        tmp_path,
        monkeypatch,
        [("Mom ❤️", _PHONE), ("Mom 💙", _SECOND_MOM_PHONE)],
        contact="Mom",
        message="hi",
    )

    assert outcome.interrupted is True
    assert outcome.interrupt_kind == "clarification"
    assert outcome.confirmation is not None
    question = str(outcome.confirmation.get("question") or "")
    assert "Mom ❤️" in question and "Mom 💙" in question


def test_an_unidentifiable_reference_halts_visibly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcome = _run_grounded(
        tmp_path,
        monkeypatch,
        [("Mom ❤️", _PHONE)],
        contact="my mother",
        message="hi",
        llm_answer="NONE",
    )

    assert outcome.interrupted is False
    assert outcome.halted_reason
    assert "couldn't find a matching saved contact" in outcome.halted_reason
    assert outcome.results == []
