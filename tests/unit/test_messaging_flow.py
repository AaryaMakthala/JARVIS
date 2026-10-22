"""Stage 8.6.4: the reusable messaging flow and its channel contract.

The flow is the piece every channel shares, so it is tested on its own with
fake adapters: no WhatsApp, no browser, no desktop, no network.

What these tests pin down:

* an unknown channel is refused instead of silently doing nothing;
* the recipient must match **exactly one** contact - zero or several are a
  clarification, never a guess;
* the rate limit is checked *before* the adapter is called (no side effect on
  a refused send) and is recorded only after a successful send;
* a request carries either text or a file, never both and never neither, and
  the channel must be able to carry it;
* an adapter that fails is turned into a refusal, never an exception that
  escapes to the graph;
* a second adapter plugs into the same flow unchanged - that is what makes the
  flow reusable.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from jarvis.messaging.base import (
    CAPABILITY_FILE,
    CAPABILITY_TEXT,
    MessageError,
    MessageOutcome,
    MessageRequest,
)
from jarvis.messaging.contacts import Contact, ContactStore, RateLimiter
from jarvis.messaging.flow import MessageFlow


class FakeChannel:
    """A channel that records what it was asked to send."""

    def __init__(self, name: str = "fake", *, capabilities: frozenset[str] | None = None) -> None:
        self.name = name
        self._capabilities = capabilities or frozenset({CAPABILITY_TEXT, CAPABILITY_FILE})
        self.sent: list[tuple[MessageRequest, Contact]] = []
        self.error: str | None = None

    def capabilities(self) -> frozenset[str]:
        return self._capabilities

    def send(self, request: MessageRequest, contact: Contact) -> MessageOutcome:
        if self.error is not None:
            raise MessageError(self.error)
        self.sent.append((request, contact))
        return MessageOutcome(
            channel=self.name,
            contact=contact.name,
            sent=True,
            detail=f"sent to {contact.name}",
            data={"recipient": contact.name},
        )


def _store(tmp_path, *names: str) -> ContactStore:
    store = ContactStore(tmp_path / "contacts.json")
    for name in names:
        store.add(name, "+919876543210")
    return store


def _limiter(tmp_path, **kwargs: Any) -> RateLimiter:
    options: dict[str, Any] = {"max_per_hour": 10, "min_interval_s": 0.0}
    options.update(kwargs)
    return RateLimiter(tmp_path / "rl.json", **options)


def _flow(tmp_path, channel: FakeChannel, *, limiter: RateLimiter | None = None) -> MessageFlow:
    return MessageFlow(
        [channel], store=_store(tmp_path, "Rahul"), limiter=limiter or _limiter(tmp_path)
    )


# ── the shared pipeline ────────────────────────────────────────────────────


def test_a_send_goes_through_the_channel(tmp_path) -> None:
    channel = FakeChannel()
    flow = _flow(tmp_path, channel)

    outcome = flow.run(MessageRequest(channel="fake", contact="Rahul", text="hello"))

    assert outcome.sent is True
    assert outcome.contact == "Rahul"
    assert channel.sent[0][0].text == "hello"
    assert channel.sent[0][1].number == "919876543210"


def test_an_unknown_channel_is_refused(tmp_path) -> None:
    channel = FakeChannel()
    flow = _flow(tmp_path, channel)

    with pytest.raises(MessageError, match="unknown messaging channel"):
        flow.run(MessageRequest(channel="carrier-pigeon", contact="Rahul", text="hello"))
    assert channel.sent == []


def test_an_unknown_contact_is_refused_with_the_options(tmp_path) -> None:
    channel = FakeChannel()
    flow = _flow(tmp_path, channel)

    with pytest.raises(MessageError, match="no contact matches"):
        flow.run(MessageRequest(channel="fake", contact="Nobody", text="hello"))
    assert channel.sent == []


def test_an_ambiguous_contact_is_refused_with_the_options(tmp_path) -> None:
    store = ContactStore(tmp_path / "contacts.json")
    store.add("Ali Khan", "+9112345678")
    store.add("Ali Ahmed", "+919876543210")
    channel = FakeChannel()
    flow = MessageFlow([channel], store=store, limiter=_limiter(tmp_path))

    with pytest.raises(MessageError, match="multiple contacts match") as excinfo:
        flow.run(MessageRequest(channel="fake", contact="ali", text="hello"))

    assert "Ali Khan" in str(excinfo.value) and "Ali Ahmed" in str(excinfo.value)
    assert channel.sent == []


def test_the_rate_limit_refuses_before_the_channel_is_called(tmp_path) -> None:
    channel = FakeChannel()
    limiter = _limiter(tmp_path, max_per_hour=1)
    limiter.record()
    flow = _flow(tmp_path, channel, limiter=limiter)

    with pytest.raises(MessageError, match="rate limit"):
        flow.run(MessageRequest(channel="fake", contact="Rahul", text="hello"))
    assert channel.sent == []


def test_a_successful_send_is_recorded_in_the_rate_limit(tmp_path) -> None:
    channel = FakeChannel()
    limiter = _limiter(tmp_path)
    flow = _flow(tmp_path, channel, limiter=limiter)

    flow.run(MessageRequest(channel="fake", contact="Rahul", text="hello"))

    assert limiter.traffic_count() == 1


def test_a_failed_send_is_not_recorded_in_the_rate_limit(tmp_path) -> None:
    channel = FakeChannel()
    channel.error = "the chat could not be verified"
    limiter = _limiter(tmp_path)
    flow = _flow(tmp_path, channel, limiter=limiter)

    with pytest.raises(MessageError, match="could not be verified"):
        flow.run(MessageRequest(channel="fake", contact="Rahul", text="hello"))
    assert limiter.traffic_count() == 0


def test_an_adapter_failure_never_escapes_as_an_exception(tmp_path) -> None:
    class Broken:
        name = "broken"

        def capabilities(self) -> frozenset[str]:
            return frozenset({CAPABILITY_TEXT})

        def send(self, request: MessageRequest, contact: Contact) -> MessageOutcome:
            raise RuntimeError("the driver exploded")

    flow = MessageFlow([Broken()], store=_store(tmp_path, "Rahul"), limiter=_limiter(tmp_path))
    with pytest.raises(MessageError, match="could not be sent"):
        flow.run(MessageRequest(channel="broken", contact="Rahul", text="hello"))


# ── the payload rules ──────────────────────────────────────────────────────


def test_a_message_carries_either_text_or_a_file(tmp_path) -> None:
    channel = FakeChannel()
    flow = _flow(tmp_path, channel)

    with pytest.raises(MessageError, match="either text or a file, never both"):
        flow.run(MessageRequest(channel="fake", contact="Rahul", text="hi", path="/tmp/x.jpg"))
    with pytest.raises(MessageError, match="either text or a file"):
        flow.run(MessageRequest(channel="fake", contact="Rahul"))
    assert channel.sent == []


def test_a_channel_that_cannot_carry_the_payload_is_refused(tmp_path) -> None:
    channel = FakeChannel("text-only", capabilities=frozenset({CAPABILITY_TEXT}))
    flow = _flow(tmp_path, channel)

    with pytest.raises(MessageError, match="cannot carry a file message"):
        flow.run(MessageRequest(channel="text-only", contact="Rahul", path="/tmp/x.jpg"))
    assert channel.sent == []


# ── reusability ───────────────────────────────────────────────────────────


def test_a_second_channel_plugs_into_the_same_flow(tmp_path) -> None:
    """The flow is reusable: a new channel needs no change to the flow."""
    first = FakeChannel("first")
    second = FakeChannel("second")
    store = _store(tmp_path, "Rahul")
    limiter = _limiter(tmp_path)

    one = MessageFlow([first], store=store, limiter=limiter)
    two = MessageFlow([second], store=store, limiter=limiter)
    request = MessageRequest(channel="second", contact="Rahul", text="hello")

    assert one.run(replace(request, channel="first")).sent is True
    assert two.run(request).sent is True
    assert len(first.sent) == 1 and len(second.sent) == 1


def test_the_flow_lists_its_channels(tmp_path) -> None:
    flow = MessageFlow([FakeChannel("b"), FakeChannel("a")], store=_store(tmp_path, "Rahul"))
    assert flow.channel_names() == ("a", "b")


def test_prepare_and_dispatch_can_be_split(tmp_path) -> None:
    """A tool may do policy work between the recipient check and the send."""
    channel = FakeChannel()
    flow = _flow(tmp_path, channel)

    prepared = flow.prepare(MessageRequest(channel="fake", contact="Rahul"))
    assert prepared.contact.name == "Rahul"
    assert channel.sent == []

    outcome = flow.dispatch(prepared, MessageRequest(channel="fake", contact="Rahul", text="hi"))
    assert outcome.sent is True
    assert len(channel.sent) == 1
