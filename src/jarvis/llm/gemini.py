"""Gemini REST client (free-tier flash models).

Uses the ``generateContent`` endpoint with ``responseSchema`` / ``json``
mime type for structured output.  The API key travels as a URL query
parameter (the only way the REST API accepts it); it is never logged,
embedded in messages, wrapped in exceptions, or returned - errors name only
the provider/model/status.

A model that rejects structured output raises :class:`LLMCapabilityError` so
the composite client skips Gemini for that operation instead of degrading to
unsafe free-form parsing.
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
_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

_CAPABILITY_TOKENS = (
    "does not support",
    "response_schema",
    "response_mime_type",
    "responseMimeType",
    "structured",
    "not available for model",
)


class GeminiClient:
    """Minimal httpx-backed Gemini client.

    ``http_client`` may be injected in tests (e.g. ``httpx.MockTransport``).
    """

    def __init__(
        self,
        api_key: str,
        *,
        planner_model: str,
        fast_model: str,
        base_url: str = _GEMINI_BASE_URL,
        timeout: float = 30.0,
        max_retries: int = 1,
        http_client: httpx.Client | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._api_key = api_key
        self._planner_model = planner_model
        self._fast_model = fast_model
        self._timeout = timeout
        self._max_retries = max_retries
        self._logger = logger or logging.getLogger("llm.client")
        self._http = http_client or httpx.Client(base_url=base_url, timeout=timeout)
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
            raise LLMError(f"no {role} model configured for gemini")
        return name

    def _status_for(self, model: str, status: int, body_text: str) -> Exception:
        lowered = body_text.lower()
        if status in (401, 403) or "api key not valid" in lowered:
            return LLMAuthError("Gemini rejected the API key (authentication failed)")
        if status == 404:
            return LLMModelError(f"Gemini model {model!r} is not available (not found)")
        if status == 400:
            if any(token in lowered for token in _CAPABILITY_TOKENS):
                return LLMCapabilityError(
                    f"Gemini model {model!r} does not support the required structured-output format"
                )
            return LLMError("Gemini rejected the request (HTTP 400)")
        return LLMError(f"Gemini request failed (HTTP {status})")

    def _generate(
        self,
        model: str,
        body: dict[str, Any],
    ) -> tuple[str, int, int]:
        """One generateContent call with bounded transient retries.

        Returns ``(content, prompt_tokens, completion_tokens)``; token counts
        are best-effort (Gemini reports usage metadata when available).
        """
        backoff = 1.0
        attempts = 0
        while True:
            try:
                resp = self._http.post(
                    f"models/{model}:generateContent",
                    params={"key": self._api_key},
                    json=body,
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                attempts += 1
                if attempts > self._max_retries:
                    raise LLMTransientError(
                        f"Gemini request failed after {attempts} attempts"
                    ) from exc
                self._logger.warning("Gemini transport error; retrying in %.1fs", backoff)
                time.sleep(backoff)
                backoff *= 2
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                attempts += 1
                if attempts > self._max_retries:
                    raise LLMTransientError(
                        f"Gemini transient failure (HTTP {resp.status_code}) "
                        f"persisted after {attempts} attempts"
                    )
                self._logger.warning("Gemini HTTP %d; retrying in %.1fs", resp.status_code, backoff)
                time.sleep(backoff)
                backoff *= 2
                continue
            if resp.status_code != 200:
                raise self._status_for(model, resp.status_code, resp.text)

            data = resp.json()
            content = ""
            candidates = data.get("candidates") or []
            if candidates:
                parts = ((candidates[0].get("content") or {}).get("parts")) or []
                content = "".join((part.get("text") or "") for part in parts)
            usage = data.get("usageMetadata") or {}
            prompt_tokens = int(usage.get("promptTokenCount", 0) or 0)
            completion_tokens = int(usage.get("candidatesTokenCount", 0) or 0)
            self.usage = Usage(
                calls=self.usage.calls + 1,
                prompt_tokens=self.usage.prompt_tokens + prompt_tokens,
                completion_tokens=self.usage.completion_tokens + completion_tokens,
            )
            return content, prompt_tokens, completion_tokens

    def _body(
        self,
        system: str,
        user: str,
        temperature: float,
        response_schema: dict[str, Any] | None,
    ) -> dict[str, Any]:
        generation_config: dict[str, Any] = {"temperature": temperature}
        if response_schema is not None:
            generation_config["responseMimeType"] = "application/json"
            generation_config["responseSchema"] = response_schema
        return {
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "systemInstruction": {"parts": [{"text": system}]},
            "generationConfig": generation_config,
        }

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
        body = self._body(system, user, temperature, schema.model_json_schema())
        raw, _, _ = self._generate(model, body)
        try:
            parsed = schema.model_validate_json(self._strip_json_fence(raw))
        except ValidationError as first_error:
            self._logger.info("Gemini structured output parse failed; repairing once")
            repair_body = dict(body)
            repair_body["contents"] = [
                *body["contents"],
                {"role": "user", "parts": [{"text": raw}]},
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                "The JSON you just returned could not be parsed into the "
                                f"required schema: {first_error}\nReturn corrected JSON only, "
                                f"matching this schema: {schema.model_json_schema()}."
                            )
                        }
                    ],
                },
            ]
            raw2, _, _ = self._generate(model, repair_body)
            try:
                parsed = schema.model_validate_json(self._strip_json_fence(raw2))
            except ValidationError as second_error:
                raise LLMError(
                    f"Gemini structured output could not be repaired: {second_error}"
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
        body = self._body(system, user, temperature, None)
        if max_tokens is not None:
            body["generationConfig"]["maxOutputTokens"] = max_tokens
        content, _, _ = self._generate(model, body)
        return content, self._snapshot_usage()

    def _snapshot_usage(self) -> Usage:
        return Usage(
            calls=self.usage.calls,
            prompt_tokens=self.usage.prompt_tokens,
            completion_tokens=self.usage.completion_tokens,
        )
