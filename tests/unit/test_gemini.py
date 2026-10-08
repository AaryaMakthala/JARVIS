"""Gemini REST client over an injected ``httpx.MockTransport`` (no network).

Covers the Stage 5a/6 vision path: ``describe_image`` sends the image inline
with the right MIME type, image bytes never reach logs or exception text, a
Gemini client without a vision model refuses cleanly, non-vision providers
raise :class:`LLMCapabilityError`, and the composite client skips non-vision
candidates.  ``structured``/``text`` smokes guard against regressions to the
shared ``_generate`` path.
"""

from __future__ import annotations

import base64
import json
import typing

import httpx
import pytest
from pydantic import BaseModel

from jarvis.config import LLMSettings, Settings
from jarvis.llm.client import GroqClient, LLMCapabilityError, LLMError
from jarvis.llm.gemini import GeminiClient
from jarvis.llm.models import model_spec
from jarvis.llm.multi import MultiProviderClient
from jarvis.llm.openai_compat import OpenAICompatClient
from jarvis.llm.provider import ProviderInfo

_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta"
_API_KEY = "AIza-test-key-never-real"
_IMAGE = b"\x89PNG\r\n\x1a\nCANARY-IMAGE-BYTES-SECRET"

Candidate = tuple[ProviderInfo, typing.Any, typing.Any]


class _Note(BaseModel):
    label: str


class _SpyClient:
    """Vision-capable-looking client that must never actually be called."""

    def __init__(self) -> None:
        self.calls = 0

    def describe_image(self, **kwargs: typing.Any) -> tuple[str, typing.Any]:
        self.calls += 1
        raise AssertionError("this client must be skipped by the registry gate")

    def text(self, **kwargs: typing.Any) -> tuple[str, typing.Any]:
        self.calls += 1
        raise AssertionError("this client must not be called")

    def structured(self, **kwargs: typing.Any) -> tuple[object, typing.Any]:
        self.calls += 1
        raise AssertionError("this client must not be called")


def _transport(handler: typing.Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url=_GEMINI_URL,
        timeout=5.0,
    )


def _gemini(
    handler: typing.Callable[[httpx.Request], httpx.Response],
    *,
    vision_model: str = "gemini-3.8-flash",
    retries: int = 1,
) -> GeminiClient:
    return GeminiClient(
        _API_KEY,
        planner_model="gemini-3.8-flash",
        fast_model="gemini-3.7-flash",
        vision_model=vision_model,
        http_client=_transport(handler),
        max_retries=retries,
    )


def _gemini_candidate(client: GeminiClient) -> Candidate:
    return (
        ProviderInfo(name="gemini", model="gemini-3.8-flash", slot=1),
        model_spec("gemini", "gemini-3.8-flash"),
        client,
    )


def _openrouter_candidate(client: typing.Any) -> Candidate:
    return (
        ProviderInfo(name="openrouter", model="openrouter/free", slot=1),
        model_spec("openrouter", "openrouter/free"),
        client,
    )


def _ok_generate(text: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "candidates": [{"content": {"role": "model", "parts": [{"text": text}]}}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
        },
    )


# ── describe_image ─────────────────────────────────────────────────────────


def test_describe_image_sends_inline_data_with_mime() -> None:
    seen: dict[str, typing.Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = request.read()
        return _ok_generate("A chessboard with a black king in the centre.")

    client = _gemini(handler)
    text, usage = client.describe_image(
        system="You are a vision assistant.",
        user="What is in this image?",
        image_bytes=_IMAGE,
        mime="image/png",
    )
    assert text == "A chessboard with a black king in the centre."
    assert usage.calls == 1
    assert "models/gemini-3.8-flash:generateContent" in seen["url"]
    body = json.loads(seen["body"])
    parts = body["contents"][0]["parts"]
    assert parts[0]["inlineData"]["mimeType"] == "image/png"
    assert parts[0]["inlineData"]["data"] == base64.b64encode(_IMAGE).decode("ascii")
    assert parts[1] == {"text": "What is in this image?"}
    assert body["systemInstruction"]["parts"][0]["text"] == "You are a vision assistant."
    assert body["generationConfig"]["temperature"] == 0.0


def test_describe_image_bytes_never_reach_logs_or_exceptions(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    gemini = _gemini(fail, retries=0)
    multi = MultiProviderClient([_gemini_candidate(gemini), _openrouter_candidate(_SpyClient())])
    with pytest.raises(LLMError) as raised:
        multi.describe_image(system="s", user="u", image_bytes=_IMAGE, mime="image/png")
    assert "openrouter: skipped" in str(raised.value)
    assert "CANARY" not in str(raised.value)
    assert "CANARY" not in caplog.text
    assert base64.b64encode(_IMAGE).decode("ascii") not in caplog.text
    assert "gemini vision transient failure" in caplog.text


def test_describe_image_refused_without_vision_model() -> None:
    def unexpected(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"no request expected, got {request.url}")

    client = _gemini(unexpected, vision_model="")
    with pytest.raises(LLMCapabilityError, match="no vision model configured"):
        client.describe_image(system="s", user="u", image_bytes=_IMAGE, mime="image/png")


def test_non_vision_clients_raise_capability_error() -> None:
    def unexpected(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"no request expected, got {request.url}")

    openrouter = OpenAICompatClient(
        "sk-test",
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        planner_model="openrouter/free",
        fast_model="openrouter/free",
        http_client=_transport(unexpected),
    )
    with pytest.raises(LLMCapabilityError, match="vision not supported"):
        openrouter.describe_image(system="s", user="u", image_bytes=_IMAGE, mime="image/png")

    groq = GroqClient("gsk-test", Settings(llm=LLMSettings()))
    with pytest.raises(LLMCapabilityError, match="vision not supported"):
        groq.describe_image(system="s", user="u", image_bytes=_IMAGE, mime="image/png")


# ── MultiProviderClient.describe_image ─────────────────────────────────────


def test_multi_describe_image_skips_non_vision_providers() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _ok_generate("A red car on a highway.")

    spy = _SpyClient()
    gemini = _gemini(handler)
    multi = MultiProviderClient([_openrouter_candidate(spy), _gemini_candidate(gemini)])
    text, usage = multi.describe_image(system="s", user="u", image_bytes=_IMAGE, mime="image/png")
    assert text == "A red car on a highway."
    assert usage.calls == 1
    assert spy.calls == 0


def test_multi_describe_image_raises_when_no_vision_provider() -> None:
    multi = MultiProviderClient([_openrouter_candidate(_SpyClient())])
    with pytest.raises(LLMError, match="openrouter: skipped.*lacks vision support"):
        multi.describe_image(system="s", user="u", image_bytes=_IMAGE, mime="image/png")


def test_multi_describe_image_skips_client_without_vision_method() -> None:
    multi = MultiProviderClient([_gemini_candidate(object())])  # type: ignore[arg-type]
    with pytest.raises(LLMError, match="client has no vision method"):
        multi.describe_image(system="s", user="u", image_bytes=_IMAGE, mime="image/png")


# ── structured / text unchanged ────────────────────────────────────────────


def test_structured_still_works() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.read())
        assert payload["generationConfig"]["responseMimeType"] == "application/json"
        return _ok_generate('{"label": "parsed-and-ok"}')

    client = _gemini(handler)
    parsed, usage = client.structured(system="s", user="u", schema=_Note)
    assert parsed.label == "parsed-and-ok"
    assert usage.calls == 1


def test_text_still_works() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _ok_generate("hello from gemini")

    client = _gemini(handler)
    text, usage = client.text(system="s", user="u")
    assert text == "hello from gemini"
    assert usage.calls == 1
