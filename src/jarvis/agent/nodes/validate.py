"""validate node: deterministic plan checks before anything executes.

Rejects unknown tools, invalid arguments, and plans over the step cap.  A
rejected *initial* plan goes back to ``plan`` once (one repair retry); a
rejected *replacement* plan goes back to ``replan`` once.  A second rejection
of either halts with an honest explanation.

When it validates a plan produced by :mod:`replan` (``pending_replan`` set),
the already-completed results and approved hashes are preserved and the
TOCTOU path snapshot is recomputed for the new steps.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from jarvis.agent.context import AppContext
from jarvis.agent.nodes.util import policy_context_for
from jarvis.agent.runtime_args import mark_runtime_args
from jarvis.tools.default_dirs import resolve_location

#: Tools whose location args get a default/alias resolved at plan time.  Every
#: other path-taking tool (append_file, read_file, list_dir, dictation) is left
#: exactly as it was.
_LOCATION_ARGS: dict[str, tuple[str, ...]] = {
    "create_file": ("path",),
    "delete_path": ("paths",),
}

#: Asked (deterministically, never guessed) when a location cannot be resolved.
_CLARIFY_QUESTION = "Which folder should I use?"


def validate(state: dict[str, Any], ctx: AppContext) -> dict[str, Any]:
    """Check the plan; return updates that route to plan/replan/policy_gate/respond."""
    plan = state.get("plan")
    if plan is None:
        return _reject(state, "The planner produced no plan.")

    # Mark steps whose arguments depend on runtime output.  This happens
    # before the arg-validation loop and before any engine hashing, so the
    # flag reflects the plan as validate sees it.  It does not change args,
    # tiers, hashes, or decisions.
    mark_runtime_args(plan)

    if plan.needs_clarification:
        return {
            "plan": plan,
            "error": None,
            "validate_attempts": 0,
            "pending_replan": False,
        }

    # Resolve default/alias locations to absolute paths *before* the arg checks
    # and before the engine hashes anything, so the action_hash is over the
    # resolved path (I7).  An unresolvable location is asked about, never
    # guessed.  Runs for the initial plan and for replans alike.
    if not _resolve_locations(plan, ctx.settings):
        plan.needs_clarification = True
        plan.clarification_question = _CLARIFY_QUESTION
        plan.steps = []
        return {
            "plan": plan,
            "error": None,
            "validate_attempts": 0,
            "pending_replan": False,
        }

    errors: list[str] = []
    for step in plan.steps:
        spec = ctx.registry.get_optional(step.tool)
        if spec is None:
            errors.append(f"step {step.id}: tool {step.tool!r} is not available")
            continue
        try:
            spec.args_model.model_validate(step.args)
        except ValidationError as exc:
            errors.append(
                f"step {step.id}: invalid args for {step.tool}: {exc.error_count()} error(s)"
            )

    if len(plan.steps) > ctx.settings.agent.max_steps:
        errors.append(f"plan has {len(plan.steps)} steps (max {ctx.settings.agent.max_steps})")

    if errors:
        return _reject(state, "; ".join(errors))

    # A replanned plan has new steps (possibly reusing ids) whose paths must be
    # re-snapshotted; the original plan's snapshot is kept otherwise (resume).
    replaying = bool(state.get("pending_replan"))
    gated = {} if replaying else state.get("gated_resolved_paths")

    if gated is None:
        # Pre-compute and stash the resolved paths for each step so that
        # policy_gate can detect TOCTOU changes after interrupt/resume.  The same
        # enrichment the gate uses is applied here, so the snapshot it records
        # is the one the gate will compare against.
        gated = {}
        pctx = policy_context_for(state, ctx.policy_ctx)
        for step in plan.steps:
            decision = ctx.engine.decide(step, pctx)
            gated[step.id] = list(decision.resolved_paths)

    if replaying:
        results = list(state.get("results") or [])
        approved_hashes = list(state.get("approved_hashes") or [])
    else:
        results = []
        approved_hashes = []

    return {
        "plan": plan,
        "error": None,
        "validate_attempts": 0,
        "step_index": 0,
        "retry_count": 0,
        "retry_now": False,
        "replan_now": False,
        "decisions": {},
        "approved_hashes": approved_hashes,
        "results": results,
        "gated_resolved_paths": gated,
        "halted_reason": None,
        "pending_replan": False,
    }


def _resolve_locations(plan: Any, settings: Any) -> bool:
    """Rewrite location args to resolved absolute paths; ``False`` to ask.

    Deterministic and side-effect free apart from the plan's own args (and a
    read-only existence check inside :func:`resolve_location`).  Never guesses,
    never creates a folder, never touches ``allowed_roots`` -- the policy engine
    still decides whether the resolved path is inside them.
    """
    for step in plan.steps:
        fields = _LOCATION_ARGS.get(step.tool)
        if not fields:
            continue
        for field in fields:
            resolved = _resolve_field(step.args.get(field), settings)
            if resolved is None:
                return False
            step.args[field] = resolved
    return True


def _resolve_field(value: Any, settings: Any) -> str | list[str] | None:
    """Resolve one arg value; ``None`` means the user must be asked which folder."""
    if isinstance(value, str):
        result = resolve_location(value, settings)
        if result.kind != "resolved" or result.path is None:
            return None
        return str(result.path)
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if not isinstance(item, str):
                return None
            result = resolve_location(item, settings)
            if result.kind != "resolved" or result.path is None:
                return None
            out.append(str(result.path))
        return out
    return value  # non-string: leave it for the args model to reject


def _reject(state: dict[str, Any], reason: str) -> dict[str, Any]:
    """Return rejection updates; a second rejection halts the task.

    The next planning stage depends on whether the rejected plan was a
    replacement (``pending_replan`` set) or the initial one.
    """
    attempts = int(state.get("validate_attempts") or 0) + 1
    if attempts >= 2:
        return {
            "validate_attempts": attempts,
            "error": reason,
            "halted_reason": f"Could not produce a valid plan (after 2 attempts): {reason}",
        }
    return {"validate_attempts": attempts, "error": reason, "halted_reason": None}
