"""Composite :class:`LLMClient` that falls through free providers per call.

Wraps the eligible free providers (configured order).  For every operation it
walks the list and:

* skips providers whose registry entry lacks the capability the operation
  needs (structured output + tool planning for ``structured()``; plain text
  generation for ``text()``);
* on a transient failure (rate limit / 5xx) moves to the next free provider
  (bounded retries already happened inside the individual client);
* on invalid credentials or an unknown/retired model moves immediately to the
  next provider (never retried);
* on an unsupported structured-output format moves immediately to the next
  provider - JARVIS never silently degrades to free-form text and unsafe
  parsing;
* when every provider has failed, raises :class:`LLMError` with a joined set
  of safe reasons (provider/model/status only, never credentials).

The loop is bounded by the number of eligible providers, so fallback can
never spin forever and never reaches a paid model (ineligible providers never
enter this client).
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from jarvis.llm.client import (
    LLMAuthError,
    LLMCapabilityError,
    LLMError,
    LLMModelError,
    LLMTransientError,
    Usage,
)
from jarvis.llm.models import ModelSpec
from jarvis.llm.provider import ProviderInfo


class MultiProviderClient:
    """Fallback LLMClient over an ordered list of eligible free clients."""

    def __init__(
        self,
        candidates: list[tuple[ProviderInfo, ModelSpec, Any]],
        logger: logging.Logger | None = None,
    ) -> None:
        if not candidates:
            raise ValueError("MultiProviderClient needs at least one candidate")
        self._candidates = list(candidates)
        self._logger = logger or logging.getLogger("llm.multi")

    @property
    def providers(self) -> list[str]:
        """Provider names in fallback order (for doctor/status)."""
        return [info.name for info, _spec, _client in self._candidates]

    @property
    def usage(self) -> Usage:
        """Aggregate usage across all wrapped clients (best-effort)."""
        calls = sum(int(getattr(c, "usage", Usage()).calls) for _, _, c in self._candidates)
        prompt = sum(
            int(getattr(c, "usage", Usage()).prompt_tokens) for _, _, c in self._candidates
        )
        completion = sum(
            int(getattr(c, "usage", Usage()).completion_tokens) for _, _, c in self._candidates
        )
        return Usage(calls=calls, prompt_tokens=prompt, completion_tokens=completion)

    def structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[BaseModel],
        model_role: str = "planner",
        temperature: float = 0.0,
    ) -> tuple[BaseModel, Usage]:
        failures: list[str] = []
        for info, spec, client in self._candidates:
            caps = spec.capabilities
            if not (caps.supports_structured_output and caps.supports_tools):
                failures.append(
                    f"{info.name}: skipped (model lacks structured-output/tool-planning support)"
                )
                continue
            try:
                return client.structured(
                    system=system,
                    user=user,
                    schema=schema,
                    model_role=model_role,
                    temperature=temperature,
                )
            except LLMCapabilityError as exc:
                failures.append(f"{info.name}: {exc}")
                self._logger.info("provider %s skipped for structured output: %s", info.name, exc)
            except LLMTransientError as exc:
                failures.append(f"{info.name}: transient failure ({exc})")
                self._logger.warning("provider %s transient failure, falling through", info.name)
            except LLMAuthError as exc:
                failures.append(f"{info.name}: {exc}")
            except LLMModelError as exc:
                failures.append(f"{info.name}: {exc}")
            except LLMError as exc:
                failures.append(f"{info.name}: {exc}")
        raise LLMError(
            "all free LLM providers failed for structured output: " + "; ".join(failures)
        )

    def text(
        self,
        *,
        system: str,
        user: str,
        model_role: str = "fast",
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> tuple[str, Usage]:
        failures: list[str] = []
        for info, _spec, client in self._candidates:
            try:
                return client.text(
                    system=system,
                    user=user,
                    model_role=model_role,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except LLMCapabilityError as exc:
                failures.append(f"{info.name}: {exc}")
            except LLMTransientError as exc:
                failures.append(f"{info.name}: transient failure ({exc})")
                self._logger.warning("provider %s transient failure, falling through", info.name)
            except LLMAuthError as exc:
                failures.append(f"{info.name}: {exc}")
            except LLMModelError as exc:
                failures.append(f"{info.name}: {exc}")
            except LLMError as exc:
                failures.append(f"{info.name}: {exc}")
        raise LLMError("all free LLM providers failed: " + "; ".join(failures))
