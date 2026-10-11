"""Slice 5: ToolSpec resolver hook + AppContext transient binding store.

The resolver hook is *optional* and purely observational: it never runs during
registration or ``execute()``, it returns identity metadata (or ``None``) and
never authorizes anything.  ``AppContext`` gains a transient, in-memory
``TargetBindingStore`` that is absent from checkpointed state.
"""

from __future__ import annotations

from typing import Any

from jarvis.agent.context import make_app_context
from jarvis.agent.target_binding import TargetBindingStore
from jarvis.config import Settings
from jarvis.policy.targets import ResolvedTarget
from jarvis.tools.base import ToolContext, ToolResult, ToolSpec, resolve_target
from support import TextArgs, make_spec, registry_with


class _FakeUnlock:
    def is_unlocked(self) -> bool:
        return True


def _ctx() -> ToolContext:
    return ToolContext(settings=Settings())


def _target() -> ResolvedTarget:
    return ResolvedTarget(kind="uia", ref="r1", name="Save", role="button")


def _spec_with_resolver(resolver: Any, record: list[str] | None = None) -> ToolSpec:
    def run(args: TextArgs, ctx: ToolContext):
        if record is not None:
            record.append("ran")
        return ToolResult(ok=True, output="ok")

    return ToolSpec(
        name="fake_resolve",
        description="fake resolver tool",
        args_model=TextArgs,
        base_tier=0,
        run=run,
        resolve=resolver,
    )


# ── ToolSpec construction compatibility ─────────────────────────────────────


def test_constructor_compatible_without_resolver() -> None:
    spec = ToolSpec(name="plain", description="d", args_model=TextArgs, base_tier=0)
    assert spec.resolve is None
    registry = registry_with(spec, make_spec("other", base_tier=0, record=[]))
    assert registry.count() == 2


def test_registration_and_execute_never_trigger_resolver() -> None:
    called: list[str] = []

    def resolver(args: TextArgs, ctx: ToolContext) -> ResolvedTarget:
        called.append("resolve")
        return _target()

    spec = _spec_with_resolver(resolver)
    registry = registry_with(spec)
    assert registry.count() == 1
    assert called == []

    result = spec.execute(TextArgs(text="hi"), _ctx())  # no resolve side effects
    assert called == []  # execution must not resolve
    assert result.ok and result.data["duration_ms"] >= 0


def test_resolver_called_explicitly_with_validated_args_and_context() -> None:
    seen: dict[str, Any] = {}

    def resolver(args: TextArgs, ctx: ToolContext) -> ResolvedTarget:
        seen["args"] = args
        seen["ctx"] = ctx
        return _target()

    spec = _spec_with_resolver(resolver)
    args = TextArgs(text="hello")
    ctx = _ctx()
    result = resolve_target(spec, args, ctx)
    assert result == _target()
    assert isinstance(seen["args"], TextArgs) and seen["args"].text == "hello"
    assert seen["ctx"] is ctx


# ── resolver failure semantics (fail closed) ────────────────────────────────


def test_no_resolver_returns_none() -> None:
    spec = ToolSpec(name="plain", description="d", args_model=TextArgs, base_tier=0)
    assert resolve_target(spec, TextArgs(text="x"), _ctx()) is None


def test_resolver_returning_none_is_safe() -> None:
    spec = _spec_with_resolver(lambda args, ctx: None)
    assert resolve_target(spec, TextArgs(text="x"), _ctx()) is None


def test_resolver_raising_fails_closed_to_none() -> None:
    def broken(args: TextArgs, ctx: ToolContext) -> ResolvedTarget:
        raise RuntimeError("UI automation failed")

    spec = _spec_with_resolver(broken)
    assert resolve_target(spec, TextArgs(text="x"), _ctx()) is None  # never crashes


def test_broken_resolver_does_not_affect_execute() -> None:
    spec = _spec_with_resolver(lambda args, ctx: (_ for _ in ()).throw(RuntimeError("boom")))
    assert resolve_target(spec, TextArgs(text="x"), _ctx()) is None
    result = spec.execute(TextArgs(text="x"), _ctx())
    assert result.ok


# ── AppContext exposes the transient store ──────────────────────────────────


def test_app_context_has_transient_binding_store() -> None:
    app = make_app_context()
    assert isinstance(app.target_bindings, TargetBindingStore)
    binding = app.target_bindings.put(
        "t1", "s1", action_hash="a", fingerprint="f", origin="o", scope_confirmation_id="c"
    )
    assert app.target_bindings.get("t1", "s1") == binding
    assert app.target_bindings.consume(
        "t1", "s1", action_hash="a", fingerprint="f", origin="o", scope_confirmation_id="c"
    ).ok


def test_each_context_gets_its_own_store() -> None:
    assert make_app_context().target_bindings is not make_app_context().target_bindings


def test_make_app_context_injects_store() -> None:
    store = TargetBindingStore()
    app = make_app_context(target_bindings=store)
    assert app.target_bindings is store


def test_store_isolated_from_checkpointed_fields() -> None:
    app = make_app_context()
    assert hasattr(app, "target_bindings")
    # The store is a plain in-memory object, not part of AgentState; nothing in
    # AppContext serializes it (it carries no secrets and is never checkpointed).
    assert isinstance(app.target_bindings, TargetBindingStore)


def test_unlock_handling_unaffected() -> None:
    fake = _FakeUnlock()
    app = make_app_context(unlock=fake)
    assert app.unlock is fake
    assert app.policy_ctx is not None and app.policy_ctx.unlock is fake
    assert make_app_context().unlock is None
