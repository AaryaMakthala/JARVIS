"""LLM client abstraction.

The :class:`LLMClient` protocol gives the rest of JARVIS a single interface for
structured and plain text completions with usage tracking. :class:`GroqClient`
is the real implementation; it tries JSON-schema structured output first and,
for models that reject it (HTTP 400 "does not support response format
json_schema"), falls back to plain JSON mode + Pydantic validation + one repair
retry. :class:`FakeLLM` is the deterministic stand-in used by all graph tests.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ValidationError

from jarvis import config
from jarvis.logging_setup import get_logger

_MODEL_ROLES = ("planner", "fast", "vision")


@dataclass(frozen=True)
class Usage:
    """Accumulated API call and token counters for the benchmark."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMError(RuntimeError):
    """Raised when an LLM call fails after retries or cannot be repaired."""


@runtime_checkable
class LLMClient(Protocol):
    """Interface every LLM provider must implement."""

    def structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[BaseModel],
        model_role: str = "planner",
        temperature: float = 0.0,
    ) -> tuple[BaseModel, Usage]: ...

    def text(
        self,
        *,
        system: str,
        user: str,
        model_role: str = "fast",
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> tuple[str, Usage]: ...


def is_json_schema_unsupported(exc: Exception) -> bool:
    """Return ``True`` when the provider says the model cannot do JSON schema."""
    return "json_schema" in str(exc).lower()


class GroqClient:
    """Real LLM client using the Groq SDK.

    ``api_key`` is required at construction and never logged. A ``completer``
    callable may be injected in tests in place of the SDK transport.
    """

    def __init__(
        self,
        api_key: str,
        settings: config.Settings,
        completer: Callable[..., Any] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        import groq

        self._groq_module = groq
        self._settings = settings
        self._logger = logger or get_logger("llm.client")
        self._groq = groq.Groq(
            api_key=api_key,
            timeout=settings.llm.timeout_seconds,
            max_retries=0,
        )
        self._completer: Callable[..., Any] = completer or self._default_completer
        self.usage = Usage()
        # model name -> json_schema supported? (cached per session)
        self._schema_modes: dict[str, bool] = {}

    def _default_completer(self, **kwargs: Any) -> Any:
        return self._groq.chat.completions.create(**kwargs)

    def _model_for(self, role: str) -> str:
        if role not in _MODEL_ROLES:
            raise LLMError(f"unknown model role {role!r}; expected one of {_MODEL_ROLES}")
        name = getattr(self._settings.llm, f"{role}_model")
        if not name:
            raise LLMError(
                f"no LLM model configured for role {role!r}"
                " - set it in config.toml [llm] after checking the provider docs"
            )
        return str(name)

    def _complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: dict[str, Any] | None,
        temperature: float,
    ) -> tuple[str, int, int]:
        """Run one completion; retry 429/5xx with backoff.

        Returns ``(content, prompt_tokens, completion_tokens)``. HTTP 400
        "json_schema" errors propagate unwrapped so callers can fall back.
        """
        backoff = 1.0
        attempts = 0
        while True:
            try:
                resp = self._completer(
                    model=model,
                    messages=messages,
                    response_format=response_format,
                    temperature=temperature,
                    timeout=self._settings.llm.timeout_seconds,
                )
            except self._groq_module.RateLimitError as exc:
                attempts += 1
                if attempts > self._settings.llm.max_retries:
                    raise LLMError(f"Groq rate limit persisted after {attempts} attempts") from exc
                self._logger.warning("Groq rate limited; retrying in %.1fs", backoff)
                time.sleep(backoff)
                backoff *= 2
                continue
            except self._groq_module.InternalServerError as exc:
                attempts += 1
                if attempts > self._settings.llm.max_retries:
                    raise LLMError(
                        f"Groq server error persisted after {attempts} attempts"
                    ) from exc
                self._logger.warning("Groq server error; retrying in %.1fs", backoff)
                time.sleep(backoff)
                backoff *= 2
                continue
            except self._groq_module.GroqError:
                raise  # let structured()/text() decide how to handle

            content = (resp.choices[0].message.content or "") if resp.choices else ""
            usage = getattr(resp, "usage", None)
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
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

        if self._schema_modes.get(model) is False:
            self._logger.info("structured output mode: json (cached, model %s)", model)
            return self._parse(model, schema, messages, {"type": "json_object"}, temperature)

        try:
            result = self._parse(
                model,
                schema,
                messages,
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "schema": schema.model_json_schema(),
                        "strict": True,
                    },
                },
                temperature,
            )
        except self._groq_module.GroqError as exc:
            if not is_json_schema_unsupported(exc):
                raise LLMError(f"Groq structured output failed: {exc}") from exc
            self._logger.info(
                "model %s does not support json_schema; falling back to JSON mode", model
            )
            self._schema_modes[model] = False
            return self._parse(model, schema, messages, {"type": "json_object"}, temperature)

        self._schema_modes[model] = True
        self._logger.info("structured output mode: json_schema (model %s)", model)
        return result

    def _parse(
        self,
        model: str,
        schema: type[BaseModel],
        messages: list[dict[str, str]],
        response_format: dict[str, Any],
        temperature: float,
    ) -> tuple[BaseModel, Usage]:
        """Ask the model for JSON, validate it, and repair it once if invalid."""
        raw, _, _ = self._complete(
            model=model,
            messages=messages,
            response_format=response_format,
            temperature=temperature,
        )
        try:
            parsed = schema.model_validate_json(self._strip_json_fence(raw))
        except ValidationError as first_error:
            self._logger.info("structured output parse failed; repairing once")
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
                    f"structured output could not be repaired: {second_error}"
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
        try:
            content, _, _ = self._complete(
                model=model,
                messages=messages,
                response_format=None,
                temperature=temperature,
            )
        except self._groq_module.GroqError as exc:
            raise LLMError(f"Groq completion failed: {exc}") from exc
        return content, self._snapshot_usage()

    def _snapshot_usage(self) -> Usage:
        return Usage(
            calls=self.usage.calls,
            prompt_tokens=self.usage.prompt_tokens,
            completion_tokens=self.usage.completion_tokens,
        )


class GeminiClient:
    """Gemini fallback provider (implemented in a later phase).

    Kept behind the same :class:`LLMClient` interface so ``provider_order`` can
    switch providers once implemented.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._init_args = args
        self._init_kwargs = kwargs

    def structured(self, **kwargs: Any) -> tuple[BaseModel, Usage]:
        raise NotImplementedError("GeminiClient is a stub until the fallback provider phase")

    def text(self, **kwargs: Any) -> tuple[str, Usage]:
        raise NotImplementedError("GeminiClient is a stub until the fallback provider phase")


class FakeLLM:
    """Deterministic scripted LLM for tests.

    ``responses`` is consumed in order; each item is either a ``str`` (for
    :meth:`text`) or a Pydantic instance (for :meth:`structured`). Falls back to
    ``default_text`` / ``default_model`` when the script is exhausted.
    """

    def __init__(
        self,
        responses: list[BaseModel | str] | None = None,
        *,
        default_text: str = "ok",
        default_model: BaseModel | None = None,
        usage: Usage | None = None,
    ) -> None:
        self._responses: list[BaseModel | str] = list(responses or [])
        self.default_text = default_text
        self.default_model = default_model
        self.usage = usage or Usage()
        self.calls: list[tuple[str, str, str]] = []  # (method, model_role, value_type)

    def add(self, response: BaseModel | str) -> None:
        """Append a scripted response."""
        self._responses.append(response)

    def structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[BaseModel],
        model_role: str = "planner",
        temperature: float = 0.0,
    ) -> tuple[BaseModel, Usage]:
        if self._responses:
            item = self._responses.pop(0)
        elif self.default_model is not None:
            item = self.default_model
        else:
            raise LLMError(f"FakeLLM script exhausted: expected a {schema.__name__} instance")
        if not isinstance(item, schema):
            raise LLMError(
                f"FakeLLM script: got {type(item).__name__}, expected a {schema.__name__} instance"
            )
        self.calls.append(("structured", model_role, type(item).__name__))
        return item, self._snapshot_usage()

    def text(
        self,
        *,
        system: str,
        user: str,
        model_role: str = "fast",
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> tuple[str, Usage]:
        if self._responses:
            item = self._responses.pop(0)
            out = item if isinstance(item, str) else self.default_text
        else:
            out = self.default_text
        self.calls.append(("text", model_role, "str"))
        return out, self._snapshot_usage()

    def _snapshot_usage(self) -> Usage:
        return Usage(
            calls=len(self.calls),
            prompt_tokens=self.usage.prompt_tokens,
            completion_tokens=self.usage.completion_tokens,
        )
