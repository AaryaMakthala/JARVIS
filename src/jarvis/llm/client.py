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
from jarvis.llm.failures import (
    FailureCategory,
    classify,
    should_retry_same_provider,
)
from jarvis.logging_setup import get_logger

_MODEL_ROLES = ("planner", "fast", "vision")


@dataclass(frozen=True)
class Usage:
    """Accumulated API call and token counters for the benchmark."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMError(RuntimeError):
    """Raised when an LLM call fails after retries or cannot be repaired.

    Messages are safe for the user: they never contain API keys, prompts,
    or raw provider payloads.
    """

    #: Explicit :class:`jarvis.llm.failures.FailureCategory` for this failure.
    #: Read structurally by the composite client so it does not have to re-derive
    #: the reason from the message text.
    category: FailureCategory = FailureCategory.UNEXPECTED

    def __init__(self, message: str, *, category: FailureCategory | None = None) -> None:
        super().__init__(message)
        if category is not None:
            self.category = category


class LLMTransientError(LLMError):
    """Rate limit or 5xx server error.

    The caller may retry (bounded) or fall through to the next free provider.
    """

    category = FailureCategory.SERVER_ERROR


class LLMAuthError(LLMError):
    """The credential was rejected (HTTP 401/403).  Never retried."""

    category = FailureCategory.AUTH


class LLMModelError(LLMError):
    """The model name is unknown/retired on the provider (HTTP 404)."""

    category = FailureCategory.UNSUPPORTED_MODEL


class LLMCapabilityError(LLMError):
    """The selected model cannot satisfy the requested capability

    (e.g. no JSON-schema support for structured output).  The receiving
    provider is skipped for that operation and the next compatible free
    provider is tried instead of silently degrading.
    """

    category = FailureCategory.UNSUPPORTED_CAPABILITY


class LLMInvalidOutputError(LLMError):
    """The model answered but the payload was not the required schema.

    Distinct from a transport failure: the provider is healthy, so the caller
    falls through to the next provider for *this* call and cools this one down
    briefly, rather than retrying the identical request.
    """

    category = FailureCategory.INVALID_STRUCTURED_OUTPUT


class LLMContextLengthError(LLMError):
    """The prompt does not fit the model's context window.

    Retrying the same request can never help, so this fails straight through to
    the next provider and cools the current one for a longer period.
    """

    category = FailureCategory.CONTEXT_LENGTH


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


#: Categories that mean "the provider refused this request shape", which for
#: ``structured()`` is grounds to retry the same call in plain-JSON mode.
_SCHEMA_REJECTIONS = frozenset(
    {FailureCategory.UNSUPPORTED_CAPABILITY, FailureCategory.MALFORMED_REQUEST}
)


def _is_schema_rejection(exc: BaseException) -> bool:
    """Did the provider reject the ``json_schema`` *response format itself*?

    Two signals count, because providers differ in what they say:

    * the error text names ``json_schema``/``response_format``/``does not
      support`` (the original check), or
    * the provider answered HTTP 400/422 with no other explanation.  Groq
      returns a bare ``400 Bad Request`` for an unknown response format, and
      the old text-only check therefore re-paid that failed probe on *every*
      call for the whole session.

    The downgrade is safe: ``{"type": "json_object"}`` demands strictly less of
    the provider, so if the schema shape was the problem it now succeeds, and
    if something else was wrong the retry 400s too and propagates.
    """
    if is_json_schema_unsupported(exc) or classify(exc) in _SCHEMA_REJECTIONS:
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    try:
        return int(status) in (400, 422) if status is not None else False
    except (TypeError, ValueError):
        return False


def _as_llm_error(category: FailureCategory, message: str, exc: BaseException) -> LLMError:
    """Build the most specific :class:`LLMError` subclass for ``category``."""
    mapping: dict[FailureCategory, type[LLMError]] = {
        FailureCategory.AUTH: LLMAuthError,
        FailureCategory.UNSUPPORTED_MODEL: LLMModelError,
        FailureCategory.UNSUPPORTED_CAPABILITY: LLMCapabilityError,
        FailureCategory.CONTEXT_LENGTH: LLMContextLengthError,
        FailureCategory.INVALID_STRUCTURED_OUTPUT: LLMInvalidOutputError,
        FailureCategory.RATE_LIMIT: LLMTransientError,
        FailureCategory.SERVER_ERROR: LLMTransientError,
        FailureCategory.CONNECTION: LLMTransientError,
        FailureCategory.DNS: LLMTransientError,
        FailureCategory.TLS: LLMTransientError,
        FailureCategory.TIMEOUT: LLMTransientError,
    }
    error_cls = mapping.get(category, LLMError)
    return error_cls(message, category=category)


@dataclass
class CallMeta:
    """Safe metadata about the most recent call, for observability.

    No credential, prompt, or raw payload — only the labels, the latency, and
    the failure category.  :class:`~jarvis.voice.status.VoiceStatusReporter`
    reads this to print which provider actually answered.
    """

    provider: str = ""
    model: str = ""
    role: str = ""
    latency_ms: int = 0
    failure_category: str = ""
    ok: bool = True


class _MetaMixin:
    """Shared last-call bookkeeping for the real provider clients."""

    def _reset_meta(self, provider: str) -> None:
        self.last_call = CallMeta(provider=provider)

    def _mark_ok(self, model: str, role: str, latency_ms: int) -> None:
        self.last_call = CallMeta(
            provider=self.last_call.provider,
            model=model,
            role=role,
            latency_ms=latency_ms,
            ok=True,
        )

    def _mark_failed(
        self, model: str, role: str, category: FailureCategory, latency_ms: int
    ) -> None:
        self.last_call = CallMeta(
            provider=self.last_call.provider,
            model=model,
            role=role,
            latency_ms=latency_ms,
            failure_category=category.value,
            ok=False,
        )


class GroqClient(_MetaMixin):
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
        planner_model: str | None = None,
        fast_model: str | None = None,
    ) -> None:
        import groq

        self._groq_module = groq
        self._provider_name = "groq"
        self._settings = settings
        self._planner_model_override = planner_model
        self._fast_model_override = fast_model
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
        self._reset_meta(self._provider_name)

    def _default_completer(self, **kwargs: Any) -> Any:
        return self._groq.chat.completions.create(**kwargs)

    def _model_for(self, role: str) -> str:
        if role not in _MODEL_ROLES:
            raise LLMError(f"unknown model role {role!r}; expected one of {_MODEL_ROLES}")
        override = {"planner": self._planner_model_override, "fast": self._fast_model_override}
        if override.get(role):
            return str(override[role])
        name = self._settings.llm.model_for(self._provider_name, role)
        if not name:
            raise LLMError(
                f"no LLM model configured for role {role!r} on {self._provider_name}"
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
        role: str = "planner",
    ) -> tuple[str, int, int]:
        """Run one completion; bounded retry for genuinely transient failures.

        Returns ``(content, prompt_tokens, completion_tokens)``.  Only
        categories in :data:`~jarvis.llm.failures.RETRYABLE` are retried (429,
        5xx, transport, timeout); a rejected key, an unknown model, a context
        overflow, or a malformed request fails straight through, because
        repeating the identical request cannot help.  When the retries are
        exhausted the error carries its explicit
        :class:`~jarvis.llm.failures.FailureCategory` so the composite client
        knows what to log and how long to cool this provider down.
        """
        backoff = 1.0
        attempts = 0
        started = time.perf_counter()
        while True:
            try:
                resp = self._completer(
                    model=model,
                    messages=messages,
                    response_format=response_format,
                    temperature=temperature,
                    timeout=self._settings.llm.timeout_seconds,
                )
            except Exception as exc:
                category = classify(exc)
                attempts += 1
                if category is FailureCategory.CONTEXT_LENGTH:
                    raise LLMContextLengthError(
                        f"Groq model {model!r} context window exceeded ({category.value})"
                    ) from exc
                if (
                    should_retry_same_provider(category)
                    and attempts <= self._settings.llm.max_retries
                ):
                    self._logger.warning("groq %s; retrying in %.1fs", category.value, backoff)
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                elapsed = int((time.perf_counter() - started) * 1000)
                self._mark_failed(model, role, category, elapsed)
                raise _as_llm_error(
                    category,
                    f"Groq {category.value} after {attempts} attempt(s)",
                    exc,
                ) from exc

            content = (resp.choices[0].message.content or "") if resp.choices else ""
            usage = getattr(resp, "usage", None)
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            self.usage = Usage(
                calls=self.usage.calls + 1,
                prompt_tokens=self.usage.prompt_tokens + prompt_tokens,
                completion_tokens=self.usage.completion_tokens + completion_tokens,
            )
            self._mark_ok(model, role, int((time.perf_counter() - started) * 1000))
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
            return self._parse(
                model, schema, messages, {"type": "json_object"}, temperature, model_role
            )

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
                model_role,
            )
        except Exception as exc:
            if not _is_schema_rejection(exc):
                raise
            # The provider rejected the *json_schema response format itself*.
            # Cache that for the session so the failed probe is paid exactly
            # once: previously the capability cache was only written when the
            # error text contained the literal "json_schema", so a plain HTTP
            # 400 was re-attempted on every single call forever.
            self._logger.info(
                "model %s rejected json_schema (%s); using JSON mode for the rest of the session",
                model,
                exc,
            )
            self._schema_modes[model] = False
            return self._parse(
                model, schema, messages, {"type": "json_object"}, temperature, model_role
            )

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
        role: str = "planner",
    ) -> tuple[BaseModel, Usage]:
        """Ask the model for JSON, validate it, and repair it once if invalid.

        A single repair round is attempted.  If that also fails the provider
        has answered but cannot produce the schema, so the error is
        :class:`LLMInvalidOutputError` (category
        ``invalid_structured_output``): the provider is healthy, so the
        composite client falls through to the next one for this call rather
        than re-sending the identical request.
        """
        raw, _, _ = self._complete(
            model=model,
            messages=messages,
            response_format=response_format,
            temperature=temperature,
            role=role,
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
                role=role,
            )
            try:
                parsed = schema.model_validate_json(self._strip_json_fence(raw2))
            except ValidationError as second_error:
                raise LLMInvalidOutputError(
                    f"structured output could not be repaired: {second_error}",
                    category=FailureCategory.INVALID_STRUCTURED_OUTPUT,
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
                role=model_role,
            )
        except LLMError:
            raise
        except Exception as exc:
            category = classify(exc)
            raise _as_llm_error(category, f"Groq completion failed: {category.value}", exc) from exc
        return content, self._snapshot_usage()

    def _snapshot_usage(self) -> Usage:
        return Usage(
            calls=self.usage.calls,
            prompt_tokens=self.usage.prompt_tokens,
            completion_tokens=self.usage.completion_tokens,
        )


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
