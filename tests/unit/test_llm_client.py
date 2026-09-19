"""GroqClient (fake transport), output-mode fallback, retries, and FakeLLM."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, Field

import jarvis.llm.client as llm_module
from jarvis.config import LLMSettings, Settings
from jarvis.llm.client import FakeLLM, GroqClient, LLMError, Usage

PLAN_JSON = '{"action": "open_app", "args": {"app": "notepad"}}'


class Plan(BaseModel):
    action: str
    args: dict[str, str] = Field(default_factory=dict)


class _Obj:
    """Ad-hoc response object mimicking groq's chat completion."""

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


def make_response(content: str, *, prompt_tokens: int = 9, completion_tokens: int = 6) -> _Obj:
    message = _Obj(content=content)
    choice = _Obj(message=message)
    usage = _Obj(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return _Obj(choices=[choice], usage=usage)


def _http_response(status: int) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("POST", "http://localhost/x"))


def bad_request(message: str) -> Exception:
    import groq

    return groq.BadRequestError(message, response=_http_response(400), body=None)


def rate_limit() -> Exception:
    import groq

    return groq.RateLimitError("rate limited", response=_http_response(429), body=None)


def server_error() -> Exception:
    import groq

    return groq.InternalServerError("boom", response=_http_response(500), body=None)


class FakeCompleter:
    """Scripted transport recording every request."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._handlers: list[Callable[[dict[str, Any]], Any]] = []

    def add(self, handler: Callable[[dict[str, Any]], Any]) -> FakeCompleter:
        self._handlers.append(handler)
        return self

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._handlers:
            handler = self._handlers.pop(0)
            return handler(kwargs)
        return make_response(PLAN_JSON)

    def json_schema_calls(self) -> int:
        return sum(
            1 for c in self.calls if (c.get("response_format") or {}).get("type") == "json_schema"
        )

    def json_object_calls(self) -> int:
        return sum(
            1 for c in self.calls if (c.get("response_format") or {}).get("type") == "json_object"
        )


def reject_schema(kwargs: dict[str, Any]) -> Any:
    raise bad_request("does not support response format json_schema")


def ok_json(kwargs: dict[str, Any]) -> Any:
    return make_response(PLAN_JSON)


def settings(**over: Any) -> Settings:
    base = {"planner_model": "p", "fast_model": "f", "vision_model": "v"}
    base.update(over)
    return Settings(llm=LLMSettings(**base))


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_module.time, "sleep", lambda _s: None)


def test_structured_uses_json_schema_mode() -> None:
    completer = FakeCompleter()
    client = GroqClient("gsk-key", settings(), completer=completer)
    plan, usage = client.structured(system="s", user="u", schema=Plan)
    assert plan.action == "open_app"
    assert plan.args == {"app": "notepad"}
    assert completer.json_schema_calls() == 1
    assert completer.json_object_calls() == 0
    assert usage.calls == 1
    assert usage.prompt_tokens == 9
    assert client.usage.calls == 1


def test_structured_falls_back_to_json_mode_when_schema_rejected() -> None:
    completer = FakeCompleter().add(reject_schema).add(ok_json).add(ok_json)
    client = GroqClient("gsk-key", settings(), completer=completer, logger=_dummy_logger())

    plan, _usage = client.structured(system="s", user="u", schema=Plan)
    assert plan.action == "open_app"
    assert client._schema_modes["p"] is False

    plan2, usage2 = client.structured(system="s", user="u", schema=Plan)
    assert plan2.action == "open_app"
    assert usage2.calls == 2
    # cached: second call never tries json_schema again
    assert completer.json_schema_calls() == 1
    assert completer.json_object_calls() == 2


def test_repairs_invalid_json_once() -> None:
    completer = FakeCompleter().add(lambda k: make_response("this is not json")).add(ok_json)
    client = GroqClient("gsk-key", settings(), completer=completer, logger=_dummy_logger())
    plan, usage = client.structured(system="s", user="u", schema=Plan)
    assert plan.action == "open_app"
    assert usage.calls == 2
    repair_messages = completer.calls[1]["messages"]
    assert any("corrected" in str(m.get("content", "")) for m in repair_messages)


def test_repair_failure_raises_llm_error() -> None:
    completer = (
        FakeCompleter().add(lambda k: make_response("bad")).add(lambda k: make_response("also bad"))
    )
    client = GroqClient("gsk-key", settings(), completer=completer)
    with pytest.raises(LLMError):
        client.structured(system="s", user="u", schema=Plan)


def test_json_fence_is_stripped() -> None:
    wrapped = f"```json\n{PLAN_JSON}\n```"
    completer = FakeCompleter().add(lambda k: make_response(wrapped))
    client = GroqClient("gsk-key", settings(), completer=completer)
    plan, _ = client.structured(system="s", user="u", schema=Plan)
    assert plan.action == "open_app"


def test_text_returns_content_and_tracks_usage() -> None:
    completer = FakeCompleter().add(
        lambda k: make_response("hello world", prompt_tokens=3, completion_tokens=2)
    )
    client = GroqClient("gsk-key", settings(), completer=completer)
    text, usage = client.text(system="s", user="u")
    assert text == "hello world"
    assert usage.calls == 1
    assert usage.prompt_tokens == 3
    assert usage.completion_tokens == 2


def test_non_schema_groq_error_wraps_in_llm_error() -> None:
    def _raising(_kwargs: Any) -> Any:
        raise bad_request("some other 400")

    completer = FakeCompleter().add(_raising).add(_raising)
    client = GroqClient("gsk-key", settings(), completer=completer)
    with pytest.raises(LLMError):
        client.text(system="s", user="u")
    with pytest.raises(LLMError):
        client.structured(system="s", user="u", schema=Plan)


def test_rate_limit_retries_then_succeeds() -> None:
    completer = (
        FakeCompleter().add(lambda k: (_ for _ in ()).throw(rate_limit())).add(lambda k: ok_json(k))
    )
    client = GroqClient("gsk-key", settings(), completer=completer)
    plan, usage = client.structured(system="s", user="u", schema=Plan)
    assert plan.action == "open_app"
    assert usage.calls == 1  # only the successful call counts
    assert len(completer.calls) == 2


def test_rate_limit_exhaustion_raises() -> None:
    completer = FakeCompleter().add(lambda k: (_ for _ in ()).throw(rate_limit()))
    client = GroqClient("gsk-key", settings(max_retries=0), completer=completer)
    with pytest.raises(LLMError):
        client.structured(system="s", user="u", schema=Plan)


def test_server_error_retries() -> None:
    completer = (
        FakeCompleter()
        .add(lambda k: (_ for _ in ()).throw(server_error()))
        .add(lambda k: ok_json(k))
    )
    client = GroqClient("gsk-key", settings(), completer=completer)
    plan, _ = client.structured(system="s", user="u", schema=Plan)
    assert plan.action == "open_app"


def test_missing_model_role_raises() -> None:
    client = GroqClient("gsk-key", Settings(), completer=FakeCompleter())
    with pytest.raises(LLMError):
        client.structured(system="s", user="u", schema=Plan)


def test_unknown_role_raises() -> None:
    client = GroqClient("gsk-key", settings(), completer=FakeCompleter())
    with pytest.raises(LLMError):
        client.text(system="s", user="u", model_role="summariser")


class TestFakeLLM:
    def test_structured_returns_scripted_model(self) -> None:
        fake = FakeLLM([Plan(action="a", args={})])
        plan, usage = fake.structured(system="s", user="u", schema=Plan)
        assert plan.action == "a"
        assert usage.calls == 1
        assert fake.calls == [("structured", "planner", "Plan")]

    def test_text_returns_scripted_string(self) -> None:
        fake = FakeLLM(["hi"])
        text, _ = fake.text(system="s", user="u")
        assert text == "hi"
        assert fake.calls == [("text", "fast", "str")]

    def test_default_model_when_script_exhausted(self) -> None:
        fake = FakeLLM(default_model=Plan(action="fallback", args={}))
        plan, _ = fake.structured(system="s", user="u", schema=Plan)
        assert plan.action == "fallback"

    def test_wrong_script_type_raises(self) -> None:
        fake = FakeLLM(["not a model"])
        with pytest.raises(LLMError):
            fake.structured(system="s", user="u", schema=Plan)

    def test_exhausted_without_default_raises(self) -> None:
        fake = FakeLLM()
        with pytest.raises(LLMError):
            fake.structured(system="s", user="u", schema=Plan)

    def test_usage_cumulative(self) -> None:
        fake = FakeLLM(default_text="ok", usage=Usage(prompt_tokens=4, completion_tokens=2))
        _, u1 = fake.text(system="s", user="u")
        _, u2 = fake.text(system="s", user="u")
        assert u1.calls == 1
        assert u2.calls == 2
        assert u2.prompt_tokens == 4


def _dummy_logger() -> Any:
    import logging

    return logging.getLogger("test-dummy")
