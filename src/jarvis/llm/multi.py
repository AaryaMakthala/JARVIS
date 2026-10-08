"""Composite :class:`LLMClient` that falls through free providers per call.

Wraps the eligible free providers (configured order).  For every operation it
walks the list and:

* skips providers whose registry entry lacks the capability the operation
  needs (structured output + tool planning for ``structured()``; plain text
  generation for ``text()``) — checked against the model **for the requested
  role**, not always the planner;
* orders candidates by :class:`~jarvis.llm.health.ProviderHealth`, so a provider
  that just 503'd is deprioritised for the next interaction without being
  removed from the configuration.  Candidates are walked **provider-major**:
  providers in health order, each with its key slots in slot order;
* on any failure records a :class:`~jarvis.llm.failures.LLMCallFailure`, applies
  a finite cooldown, then follows the B8 rotation rule (see below) - either
  advancing to the provider's next key slot or moving to the next provider;
* when every provider has failed, raises :class:`LLMError` with a joined set of
  safe reasons (provider/model/category only, never credentials).

B8 KEY ROTATION (availability only)
-----------------------------------
* Availability failures - ``AUTH``, ``SERVER_ERROR``, ``CONNECTION``, ``DNS``,
  ``TLS``, ``TIMEOUT``, ``UNEXPECTED`` - advance to the provider's *next key
  slot*; a rotated credential may succeed where the current one cannot.
* ``RATE_LIMIT`` cools only the failing ``(provider, slot, role)`` and does
  **not** rotate keys by default: it falls through to the next provider, and
  logs it plainly.  Set ``llm.rotate_on_rate_limit = true`` to rotate (some
  provider terms forbid evading 429s by account rotation).
* Model-scoped failures (no structured output, unknown model, context length,
  unrepairable output) are recorded once on the provider-wide pseudo-slot 0
  and go to the next provider: a different key cannot fix a model/request
  problem, and one probe cools every slot of that provider for the role.
* Slot 0 is never a callable candidate; it exists only in health.

**No permanent blacklisting.**  A cooldown expiring makes the provider eligible
again, and the next success resets its failure count, so a rotated key or a
re-published model recovers without restarting the daemon.  If *every* provider
is cooling, the cooling ones are still tried in health order, because refusing
to call anything would fail the interaction outright.

The loop is bounded by the number of eligible providers, so fallback can never
spin forever and never reaches a paid model (ineligible providers never enter
this client).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from pydantic import BaseModel

from jarvis.llm import models as llm_models
from jarvis.llm.client import CallMeta, LLMError, Usage
from jarvis.llm.failures import (
    MODEL_SCOPED_CATEGORIES,
    ROTATION_CATEGORIES,
    FailureCategory,
    LLMCallFailure,
    failure_from_exception,
)
from jarvis.llm.health import ProviderHealth
from jarvis.llm.models import ModelSpec
from jarvis.llm.provider import ProviderInfo

Candidate = tuple[ProviderInfo, ModelSpec, Any]


def _role_model(client: Any, role: str) -> str:
    """Best-effort model label the client will use for ``role``.

    The concrete clients resolve roles differently (a settings override on
    Groq, a per-role constructor argument elsewhere), so ask the client rather
    than trusting the provider-level planner label.  Returns ``""`` when the
    client cannot answer, which makes the caller fall back to the configured
    spec instead of guessing.
    """
    for attr in ("resolve_model", "_model_for"):
        resolver = getattr(client, attr, None)
        if callable(resolver):
            try:
                return str(resolver(role))
            except Exception:  # noqa: BLE001 - an unknown role is not fatal here
                return ""
    return ""


def _spec_for_role(provider: str, candidate_spec: ModelSpec, client: Any, role: str) -> ModelSpec:
    """The :class:`ModelSpec` that actually serves ``role`` for this client.

    Fixes a real bug: the capability gate used the *planner* spec for every
    operation, so a ``text()`` call was refused because the planner model could
    not do structured output.

    Falls back to ``candidate_spec`` when the role's model is not a registered
    entry — ``model_spec`` returns a synthetic all-false "unknown" spec in that
    case, and gating on it would refuse a perfectly good client (a test double,
    or a model published after this registry was written).
    """
    model = _role_model(client, role)
    if not model:
        return candidate_spec
    try:
        spec = llm_models.model_spec(provider, model)
    except Exception:  # noqa: BLE001 - unknown model -> keep the given spec
        return candidate_spec
    if spec.pricing_mode == "unknown":
        return candidate_spec
    return spec


class MultiProviderClient:
    """Fallback LLMClient over an ordered list of eligible free clients."""

    def __init__(
        self,
        candidates: list[Candidate],
        logger: logging.Logger | None = None,
        *,
        health: ProviderHealth | None = None,
        clock: Callable[[], float] = time.monotonic,
        rotate_on_rate_limit: bool = False,
    ) -> None:
        if not candidates:
            raise ValueError("MultiProviderClient needs at least one candidate")
        self._candidates = list(candidates)
        self._logger = logger or logging.getLogger("llm.multi")
        self._health = health if health is not None else ProviderHealth(clock=clock)
        self._rotate_on_rate_limit = rotate_on_rate_limit
        self._last_call = CallMeta(provider="", model="", role="")

    # ── observability ────────────────────────────────────────────────────

    @property
    def health(self) -> ProviderHealth:
        """The live health memory (for ``doctor`` and voice reporting)."""
        return self._health

    @property
    def last_call(self) -> CallMeta:
        """Metadata for the most recent call: who answered, and how slowly."""
        return self._last_call

    @property
    def providers(self) -> list[str]:
        """Provider names in configured fallback order (for doctor/status)."""
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

    def routing_report(self) -> str:
        """One line, secret-free, describing who is preferred right now."""
        parts = []
        for candidate in self._ordered("planner"):
            info, _spec, _client = candidate
            parts.append(self._label(info, "planner"))
        return ", ".join(parts) or "none"

    def health_snapshot(self) -> list[dict[str, Any]]:
        """Safe diagnostic view of every tracked ``(provider, slot, role)`` triple."""
        return self._health.snapshot()

    # ── operations ───────────────────────────────────────────────────────

    def structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[BaseModel],
        model_role: str = "planner",
        temperature: float = 0.0,
    ) -> tuple[BaseModel, Usage]:
        """First provider that can produce ``schema`` for ``model_role`` wins."""
        failures: list[str] = []
        for group in self._groups(model_role):
            for info, _spec, client in group:
                spec = _spec_for_role(info.name, _spec, client, model_role)
                caps = spec.capabilities
                if not (caps.supports_structured_output and caps.supports_tools):
                    failures.append(
                        f"{info.name}: skipped ({spec.model} lacks "
                        "structured-output/tool-planning support)"
                    )
                    continue
                try:
                    result = client.structured(
                        system=system,
                        user=user,
                        schema=schema,
                        model_role=model_role,
                        temperature=temperature,
                    )
                except LLMError as exc:
                    failure = self._note_failure(
                        info, model_role, client, exc, failures, structured=True
                    )
                    if self._advance_after_failure(failure):
                        continue
                    break
                self._note_success(info, model_role, client)
                return result
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
        """First healthy provider that can answer plain text for ``model_role``."""
        failures: list[str] = []
        for group in self._groups(model_role):
            for info, _spec, client in group:
                try:
                    result = client.text(
                        system=system,
                        user=user,
                        model_role=model_role,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                except LLMError as exc:
                    failure = self._note_failure(
                        info, model_role, client, exc, failures, structured=False
                    )
                    if self._advance_after_failure(failure):
                        continue
                    break
                self._note_success(info, model_role, client)
                return result
        raise LLMError("all free LLM providers failed: " + "; ".join(failures))

    def close(self) -> None:
        """Close every wrapped client that owns a transport."""
        for _info, _spec, client in self._candidates:
            closer = getattr(client, "close", None)
            if callable(closer):
                closer()

    # ── internals ────────────────────────────────────────────────────────

    def _ordered(self, role: str) -> list[Candidate]:
        """Candidates for ``role`` in current health order (stable)."""
        return self._health.order(
            self._candidates,
            key=lambda item: (item[0].name, item[0].slot, role),
        )

    def _groups(self, role: str) -> list[list[Candidate]]:
        """Candidates grouped per provider, slots in slot order (B8 walk).

        Provider order comes from the flat health order (first appearance of
        any of the provider's slots); within a group slots ascend, so the
        walk visits key slot 1, 2, ... in order.  The provider-wide slot-0
        entry never appears here: it is health state, not a callable
        candidate.  Every candidate appears exactly once, so the walk stays
        bounded by the candidate count.
        """
        groups: dict[str, list[Candidate]] = {}
        for candidate in self._ordered(role):
            groups.setdefault(candidate[0].name, []).append(candidate)
        for members in groups.values():
            members.sort(key=lambda item: item[0].slot)
        return list(groups.values())

    def _advance_after_failure(self, failure: LLMCallFailure) -> bool:
        """B8: whether the walk may try this provider's *next key slot*.

        ``True`` = advance within the provider group; ``False`` = leave the
        group and move on to the next provider.  Model-scoped failures were
        already recorded provider-wide on slot 0 by :meth:`_note_failure`.
        """
        if failure.category in MODEL_SCOPED_CATEGORIES:
            return False  # a different key cannot fix a model/request problem
        if failure.category is FailureCategory.MALFORMED_REQUEST:
            return False  # a different key cannot fix a rejected payload
        if failure.category is FailureCategory.RATE_LIMIT:
            if self._rotate_on_rate_limit:
                self._logger.warning(
                    "%s rate limited; rotating to the next key slot", failure.provider
                )
                return True
            self._logger.warning(
                "%s rate limited; not rotating key, trying next provider", failure.provider
            )
            return False
        return failure.category in ROTATION_CATEGORIES

    def _label(self, info: ProviderInfo, role: str) -> str:
        cooling = self._health.is_cooling(info.name, role, info.slot)
        return f"{info.name} ({'cooldown' if cooling else 'ready'})"

    def _note_failure(
        self,
        info: ProviderInfo,
        role: str,
        client: Any,
        exc: LLMError,
        failures: list[str],
        *,
        structured: bool,
    ) -> LLMCallFailure:
        """Record the failure, apply its cooldown, and queue a safe reason.

        Model-scoped categories are re-attributed to the provider-wide
        pseudo-slot 0 before recording, so one probe cools every key slot of
        that provider for this role instead of burning each key in turn.
        Returns the recorded failure so the walk can apply the B8 rule.
        """
        meta = getattr(client, "last_call", None)
        model = _role_model(client, role)
        failure = failure_from_exception(
            exc,
            provider=info.name,
            model=model or (meta.model if isinstance(meta, CallMeta) else ""),
            role=role,
            slot=info.slot,
            latency_ms=meta.latency_ms if isinstance(meta, CallMeta) else 0,
        )
        if failure.category in MODEL_SCOPED_CATEGORIES:
            failure = replace(failure, slot=0)
        applied = self._health.record_failure(failure)
        self._last_call = CallMeta(
            provider=info.name,
            model=failure.model,
            role=role,
            latency_ms=failure.latency_ms,
            failure_category=failure.category.value,
            ok=False,
        )
        failures.append(f"{info.name}: {failure.category.value} ({exc})")
        kind = "structured output" if structured else "text"
        if failure.retryable:
            self._logger.warning(
                "provider %s %s transient failure: %s; cooling that role for %.0fs",
                info.name,
                kind,
                failure.category.value,
                applied,
            )
        else:
            self._logger.warning(
                "provider %s %s failed: %s; cooling that role for %.0fs",
                info.name,
                kind,
                failure.category.value,
                applied,
            )
        return failure

    def _note_success(self, info: ProviderInfo, role: str, client: Any) -> None:
        """Clear the cooldown and publish who answered, for the voice reporter."""
        self._health.record_success(info.name, role, info.slot)
        meta = getattr(client, "last_call", None)
        if isinstance(meta, CallMeta):
            self._last_call = CallMeta(
                provider=info.name,
                model=meta.model or info.model,
                role=role,
                latency_ms=meta.latency_ms,
                ok=True,
            )
        else:
            self._last_call = CallMeta(provider=info.name, model=info.model, role=role, ok=True)
