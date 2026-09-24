"""OpenAI-compatible ``/chat/completions`` client (OpenRouter, NVIDIA).

Both providers expose an OpenAI-style REST endpoint, so one httpx-backed
client serves them.  Structured output uses the ``json_schema``
``response_format`` form; a routed model that rejects it raises
:class:`LLMCapabilityError` so the composite client skips that provider for
the operation and tries the next compatible free provider (never a paid one,
and never an unsafe free-form parse).

Secrets: the API key lives in memory and is sent only as a bearer token; it
is never logged, embedded in messages, wrapped in exceptions, or returned.
Error messages name the provider/model and status, never request bodies.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from jarvis.llm.client import (
    LLMAuthError,
    LLMCapabilityError,
    LLMError,
    LLMModelError,
    LLMTransientError,
    Usage,
)

_MODEL_ROLES = ("planner", "fast")

_CAPABILITY_TOKENS = (
    "response_format",
    "json_schema",
    "structured output",
    "does not support",
    "tool call",
)


class OpenAICompatClient:
    """Minimal chat-completions client for OpenRouter/NVIDIA free endpoints.

    ``http_client`` may be injected in tests (e.g. ``httpx.MockTransport``).
    """

    def __init__(
        self,
        api_key: str,
        *,
        provider: str,
        base_url: str,
        planner_model: str,
        fast_model: str,
        timeout: float = 30.0,
        max_retries: int = 1,
        http_client: httpx.Client | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._provider = provider
        self._planner_model = planner_model
        self._fast_model = fast_model
        self._timeout = timeout
        self._max_retries = max_retries
        self._logger = logger or logging.getLogger("llm.client")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if provider == "openrouter":
            headers["X-Title"] = "JARVIS"
        self._http = http_client or httpx.Client(
            base_url=base_url, timeout=timeout, headers=headers
        )
        self._owns_http = http_client is None
        self.usage = Usage()

    def close(self) -> None:
        """Release the underlying HTTP client (no-op for injected clients)."""
        if self._owns_http:
            self._http.close()

    def _model_for(self, role: str) -> str:
        if role not in _MODEL_ROLES:
            raise LLMError(f"unknown model role {role!r}; expected one of {_MODEL_ROLES}")
        name = self._planner_model if role == "planner" else self._fast_model
        if not name:
            raise LLMError(f"no {role} model configured for {self._provider}")
        return name

    def _status_for(self, model: str, status: int, body_text: str) -> Exception:
        """Map an HTTP status to the right error class (messages stay safe)."""
        if status in (401, 403):
            return LLMAuthError(f"{self._provider} rejected the API key (authentication failed)")
        if status == 404:
            return LLMModelError(f"{self._provider} model {model!r} is not available (not found)")
        if status == 400:
            lowered = body_text.lower()
            if any(token in lowered for token in _CAPABILITY_TOKENS):
                return LLMCapabilityError(
                    f"{self._provider} model {model!r} does not support the required "
                    "structured-output format"
                )
            return LLMError(f"{self._provider} rejected the request (HTTP 400)")
        return LLMError(f"{self._provider} request failed (HTTP {status})")

    def _complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        response_format: dict[str, Any] | None,
        temperature: float,
        max_tokens: int | None = None,
    ) -> tuple[str, int, int]:
        """One completion with bounded transient retries.

        Retries: at most ``max_retries`` for 429/5xx/transport errors, with
        exponential backoff.  Auth and model-availability failures are never
        retried.  Returns ``(content, prompt_tokens, completion_tokens)``.
        """
        backoff = 1.0
        attempts = 0
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        while True:
            try:
                resp = self._http.post("chat/completions", json=payload)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                attempts += 1
                if attempts > self._max_retries:
                    raise LLMTransientError(
                        f"{self._provider} request failed after {attempts} attempts"
                    ) from exc
                self._logger.warning(
                    "%s transport error; retrying in %.1fs", self._provider, backoff
                )
                time.sleep(backoff)
                backoff *= 2
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                attempts += 1
                if attempts > self._max_retries:
                    raise LLMTransientError(
                        f"{self._provider} transient failure (HTTP {resp.status_code}) "
                        f"persisted after {attempts} attempts"
                    )
                self._logger.warning(
                    "%s HTTP %d; retrying in %.1fs", self._provider, resp.status_code, backoff
                )
                time.sleep(backoff)
                backoff *= 2
                continue
            if resp.status_code != 200:
                raise self._status_for(model, resp.status_code, resp.text)

            data = resp.json()
            content = ""
            choices = data.get("choices") or []
            if choices:
                content = (choices[0].get("message") or {}).get("content") or ""
            usage = data.get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens = int(usage.get("completion_tokens", 0) or 0)
            self.usage = Usage(
                calls=self.usage.calls + 1,
                prompt_tokens=self.usage.prompt_tokens + prompt_tokens,
                completion_tokens=self.usage.completion_tokens + completion_tokens,
            )
            return content, prompt_tokens, completion_tokens

    @staticmethod
    def _strip_json_fence(raw: str) -> str:
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            text = text.rsplit("```", 1)[0]
        return text.strip()

    def structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[BaseModel],
        model_role: str = "planner",
        temperature: float = 0.0,
    ) -> tuple[BaseModel, Usage]:
        model = self._model_for(model_role)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": schema.model_json_schema(),
                "strict": True,
            },
        }
        raw, _, _ = self._complete(
            model=model,
            messages=messages,
            response_format=response_format,
            temperature=temperature,
        )
        try:
            parsed = schema.model_validate_json(self._strip_json_fence(raw))
        except ValidationError as first_error:
            self._logger.info("%s structured output parse failed; repairing once", self._provider)
            repair_messages = [
                *messages,
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        "The JSON you just returned could not be parsed into the required schema: "
                        f"{first_error}\nReturn corrected JSON only, matching this schema: "
                        f"{schema.model_json_schema()}."
                    ),
                },
            ]
            raw2, _, _ = self._complete(
                model=model,
                messages=repair_messages,
                response_format=response_format,
                temperature=temperature,
            )
            try:
                parsed = schema.model_validate_json(self._strip_json_fence(raw2))
            except ValidationError as second_error:
                raise LLMError(
                    f"{self._provider} structured output could not be repaired: {second_error}"
                ) from second_error
        return parsed, self._snapshot_usage()

    def text(
        self,
        *,
        system: str,
        user: str,
        model_role: str = "fast",
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> tuple[str, Usage]:
        model = self._model_for(model_role)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        content, _, _ = self._complete(
            model=model,
            messages=messages,
            response_format=None,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return content, self._snapshot_usage()

    def _snapshot_usage(self) -> Usage:
        return Usage(
            calls=self.usage.calls,
            prompt_tokens=self.usage.prompt_tokens,
            completion_tokens=self.usage.completion_tokens,
        )
