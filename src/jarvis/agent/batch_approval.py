"""Plan-level (batch) approval for Tier 1 steps whose args are fully known.

Pure in the agent layer: no I/O, no new tier logic, no new hash semantics.
It only decides *when* policy_gate raises one interrupt for several steps
instead of one interrupt per step, and describes *what* the resume answer must
carry.

Batch approval is controlled entirely here and in policy_gate's thin call-in.
Nothing outside this module may approve a step; act.py still enforces
``action_hash in approved_hashes`` and the TOCTOU path compare.

Invariant this module must never violate:
* it never lowers a tier (tier still comes from the engine);
* it never batch-approves Tier 2 (eligibility is Tier 1 only);
* it never batch-approves a runtime-derived step;
* it never approves a step whose recomputed decision no longer matches;
* it only ever batches a **contiguous** run of steps that starts at the step the
  gate is standing on, and never one whose hash is already approved;
* it never lets a non-approving answer (timeout or rejection) fall through to
  act: the resume halts with honest wording instead of a silent empty result;
* it never touches the serde allowlist.
"""

from __future__ import annotations

from typing import Any

from jarvis.agent.plan_summary import summarise
from jarvis.daemon.confirmations import plan_hash
from jarvis.policy.refusal import refusal_text

#: Interrupt payload ``type`` for a plan-level approval.
TYPE_PLAN_APPROVAL = "plan_approval"

#: Key used to carry the ordered list of eligible (step_id, action_hash) pairs
#: inside the confirmation payload.  It is a plain dict key on the payload the
#: daemon already forwards, so it is safe on the wire and on resume; it is not
#: a new Pydantic model field on the allowlisted state models.
KEY_ELIGIBLE = "eligible"


def decide_and_build_payload(
    plan: Any,
    decisions: dict[str, Any],
    *,
    step_index: int = 0,
    approved_hashes: Any = (),
    summary_budget: int = 600,
) -> dict[str, Any]:
    """Return the interrupt payload policy_gate should use for the current step.

    ``decisions`` must cover **every** remaining step (see
    :func:`compute_all_decisions`): a partial map can only ever describe the one
    step the gate is on, which made a plan-level batch unreachable in the real
    graph.

    When **2 or more contiguous** batch-eligible Tier 1 steps start at
    ``step_index``, returns a single ``plan_approval`` payload naming all of
    them.  Otherwise returns the existing per-step ``confirm`` payload for the
    *current* step, unchanged (behaviour preserved for 0-or-1 eligible steps).
    """
    eligible = _eligible_steps(
        plan, decisions, step_index=step_index, approved_hashes=approved_hashes
    )
    if len(eligible) < 2:
        return _single_step_payload(
            _current_decision(plan, decisions, step_index), _step_at(plan, step_index)
        )

    ordered_hashes = [action_hash for _, action_hash in eligible]
    return _plan_approval_payload(plan, decisions, eligible, ordered_hashes, summary_budget)


def compute_all_decisions(plan: Any, pctx: Any, engine: Any) -> dict[str, Any]:
    """Decide **every** step of ``plan`` with the one policy engine.

    ``policy_gate`` decides only the step it is standing on, so ``state["decisions"]``
    holds one entry per step *visited so far* and batch eligibility could never
    see a second step.  This helper asks the SAME
    :meth:`~jarvis.policy.engine.PolicyEngine.decide` the gate uses - with the
    same ``PolicyContext`` - for each step, so there is exactly one policy path
    and no tier arithmetic anywhere in this module.

    Read-only, like ``validate``'s own pre-computation of
    ``gated_resolved_paths``: the engine resolves paths and reads rules/settings,
    it never acts.  A step whose tool/args the engine rejects comes back as a
    blocked ``Decision`` (``allowed=False``), not an exception, which
    :func:`_batch_eligible` then refuses.
    """
    out: dict[str, Any] = {}
    for step in _steps(plan):
        out[str(getattr(step, "id", "") or "")] = engine.decide(step, pctx)
    return out


# ── eligibility ──────────────────────────────────────────────────────────


def _eligible_steps(
    plan: Any,
    decisions: dict[str, Any],
    *,
    step_index: int = 0,
    approved_hashes: Any = (),
) -> list[tuple[str, str]]:
    """Ordered (step_id, action_hash) pairs that may be batch-approved.

    A step is batch-eligible ONLY when ALL of these are true:
    * decision.tier == 1
    * decision.needs_confirm
    * not decision.needs_unlock
    * step.resolved_from_runtime is False
    * decision.allowed (not blocked / refused)
    * its action_hash is **not** already in ``approved_hashes``
    Everything else is gated individually, exactly as today.

    Only steps at index ``>= step_index`` are considered, and only as one
    **contiguous run starting at ``step_index``**: collection stops at the first
    step that is not eligible.  A batch therefore never spans a step the user
    has to look at on its own (Tier 2, Tier 3, runtime-derived, already
    approved), and never re-offers a batch for steps that are already done.
    Fail closed: a missing decision also ends the run.
    """
    already = {str(h) for h in (approved_hashes or ())}
    out: list[tuple[str, str]] = []
    for step in _steps(plan)[max(int(step_index), 0) :]:
        decision = _decision(decisions, step)
        if decision is None:
            break
        if not _batch_eligible(decision, step, already):
            break
        out.append((str(step.id), str(decision.action_hash)))
    return out


def _batch_eligible(decision: Any, step: Any, already_approved: Any = ()) -> bool:
    """Fail closed: return False unless every condition is clearly met."""
    if not getattr(decision, "allowed", False):
        return False
    if getattr(decision, "tier", 0) != 1:
        return False
    if not getattr(decision, "needs_confirm", False):
        return False
    if getattr(decision, "needs_unlock", False):
        return False
    if getattr(step, "resolved_from_runtime", False):
        return False
    already = {str(h) for h in (already_approved or ())}
    return str(getattr(decision, "action_hash", "")) not in already


# ── payload builders ──────────────────────────────────────────────────────


def _single_step_payload(decision: Any, step: Any = None) -> dict[str, Any]:
    """The existing per-step payload, unchanged.

    ``step`` (optional, for callers that build one without a plan) adds the
    Stage 3 fields the daemon and the voice layers need to *re-verify* a Tier 2
    approval instead of guessing: ``tool``, ``allowed`` (the engine's own
    verdict, so a consumer can tell "blocked" from "Tier 1") and, for a Tier 2
    step that needs an unlock, ``readback_args`` - the validated tool args, which
    are what makes an exact readback possible (a WhatsApp message has to be
    stated in full).  Those args are already in ``state["plan"]`` in the same
    checkpoint, so this adds no new exposure; it is not on the serde allowlist.

    The *authorisation* stays in :mod:`jarvis.voice.tier2_voice`: carrying the
    fields here grants nothing by itself.
    """
    payload = {
        "type": "confirm",
        "step_id": str(getattr(decision, "step_id", "")),
        "tier": int(getattr(decision, "tier", 0) or 0),
        "summary": str(getattr(decision, "summary", "") or ""),
        "needs_unlock": bool(getattr(decision, "needs_unlock", False)),
        "typed_confirmation": getattr(decision, "needs_typed_confirmation", None),
        "resolved_paths": list(getattr(decision, "resolved_paths", []) or []),
        "action_hash": str(getattr(decision, "action_hash", "")),
        "untrusted": bool(getattr(decision, "warn_untrusted", False)),
        "tool": str(getattr(step, "tool", "") or "") if step is not None else "",
        "allowed": bool(getattr(decision, "allowed", False)),
    }
    if (
        payload["tier"] == 2
        and payload["needs_unlock"]
        and isinstance(getattr(step, "args", None), dict)
    ):
        payload["readback_args"] = dict(step.args)
    return payload


def _plan_approval_payload(
    plan: Any,
    decisions: dict[str, Any],
    eligible: list[tuple[str, str]],
    ordered_hashes: list[str],
    summary_budget: int,
) -> dict[str, Any]:
    """One interrupt covering the batch of eligible Tier 1 steps."""
    summary = summarise(plan, decisions, budget=summary_budget)
    return {
        "type": TYPE_PLAN_APPROVAL,
        "tier": 1,
        "summary": summary,
        "action_hash": _batch_action_hash(ordered_hashes),
        "eligible": eligible,
        "plan_hash": plan_hash(ordered_hashes),
        "resolved_paths": [],  # batch carries per-step hashes, not one path
    }


def _batch_action_hash(ordered_hashes: list[str]) -> str:
    """A single binding hash for the whole batch.

    Computed from the ordered eligible hashes.  The resume answer must carry
    this exact value; a mismatch falls back to individual gating (see
    ``recompute_and_resolve_batch``).
    """
    return plan_hash(ordered_hashes)


# ── resume: recompute + resolve ───────────────────────────────────────────


def recompute_and_resolve_batch(
    plan: Any,
    decisions: dict[str, Any],
    state: dict[str, Any],
    answer: Any,
) -> tuple[dict[str, Any], list[str]]:
    """On resume, recompute each eligible step's decision and decide per step.

    Returns ``(updates, approved_hashes)`` where:
    * ``updates`` is the dict policy_gate should return (may include a fresh
      ``halted_reason`` when the batch cannot be approved as a whole — a
      confirmation timeout, an explicit rejection, or a failed check);
    * ``approved_hashes`` is the list of action hashes to append to
      ``approved_hashes`` (only the steps that still pass every check).

    Any step whose recomputed decision is no longer Tier 1 / needs_confirm / not
    unlock / runtime-derived / allowed / still-unapproved, or whose recomputed
    action_hash differs from the one the batch was built on, is NOT
    batch-approved: it is left for its own individual interrupt (the caller
    handles that by returning a per-step payload for that step_index).

    The eligible run is recomputed with **the same** ``step_index`` and
    ``approved_hashes`` the interrupt was built from, so a resume can never widen
    the batch beyond what the user actually saw.
    """
    if not isinstance(answer, dict) or answer.get("approved") is not True:
        # Neither a timeout nor an explicit rejection may fall through to act,
        # where the missing hash would surface as a misleading "the plan was
        # likely altered after confirmation" refusal.  Halt here with the same
        # honest wording the single-step confirmation path uses (policy_gate).
        if isinstance(answer, dict) and answer.get("timed_out") is True:
            return {
                "decisions": decisions,
                "halted_reason": "Confirmation timed out. I did not perform the action.",
            }, []
        return {"decisions": decisions, "halted_reason": refusal_text(answer)}, []

    provided_hash = answer.get("action_hash")
    if not isinstance(provided_hash, str) or not provided_hash:
        return {
            "decisions": decisions,
            "halted_reason": "Refused: the plan approval did not carry an action hash.",
        }, []

    eligible = _eligible_steps(
        plan,
        decisions,
        step_index=int(state.get("step_index") or 0),
        approved_hashes=state.get("approved_hashes") or (),
    )
    if not eligible:
        return {"decisions": decisions}, []

    ordered_hashes = [action_hash for _, action_hash in eligible]
    expected_batch_hash = plan_hash(ordered_hashes)
    if provided_hash != expected_batch_hash:
        return {
            "decisions": decisions,
            "halted_reason": "Refused: the plan approval's action hash does not match this plan.",
        }, []

    # Recompute each eligible step on resume and TOCTOU-compare resolved paths,
    # exactly as the single-step path does.  Only steps that still pass every
    # check get their hash appended.
    updates: dict[str, Any] = {"decisions": decisions}
    approved_hashes: list[str] = []
    gated = dict(state.get("gated_resolved_paths") or {})

    for step_id, original_hash in eligible:
        step = _step_by_id(plan, step_id)
        if step is None:
            continue
        decision = _decision(decisions, step)
        if decision is None:
            continue
        if not _batch_eligible(decision, step):
            # This step is no longer batch-eligible on resume: leave it for its
            # own interrupt.  Do not append its hash.
            continue
        if str(decision.action_hash) != str(original_hash):
            # The action itself changed (args / tool) since the batch was built.
            # Do not batch-approve it.
            continue

        # TOCTOU: compare the stashed pre-interrupt paths against the fresh ones.
        prior_paths = gated.get(step_id)
        gated[step_id] = list(decision.resolved_paths)
        if prior_paths is not None and prior_paths != decision.resolved_paths:
            updates["gated_resolved_paths"] = gated
            updates["halted_reason"] = (
                "Refused: a file path changed after you confirmed — not acting on it."
            )
            return updates, approved_hashes

        approved_hashes.append(str(decision.action_hash))

    if not approved_hashes:
        return {
            "decisions": decisions,
            "gated_resolved_paths": gated,
            "halted_reason": "Refused: none of the planned steps could be approved as a batch.",
        }, []

    updates["gated_resolved_paths"] = gated
    updates["approved_hashes"] = list(
        {str(h) for h in (state.get("approved_hashes") or []) + approved_hashes}
    )
    return updates, approved_hashes


# ── tiny helpers ──────────────────────────────────────────────────────────


def _steps(plan: Any) -> list[Any]:
    return list(getattr(plan, "steps", None) or [])


def _decision(decisions: dict[str, Any], step: Any) -> Any | None:
    return decisions.get(str(getattr(step, "id", "") or "") or "")


def _step_by_id(plan: Any, step_id: str) -> Any | None:
    for s in _steps(plan):
        if str(getattr(s, "id", "") or "") == str(step_id):
            return s
    return None


def _step_at(plan: Any, step_index: int) -> Any:
    """The step the gate is standing on, or ``None`` (fail closed)."""
    steps = _steps(plan)
    idx = max(int(step_index), 0)
    return steps[idx] if 0 <= idx < len(steps) else None


def _current_decision(plan: Any, decisions: dict[str, Any], step_index: int) -> Any:
    """The decision for the step the gate is standing on.

    Deliberately **not** step 0: the single-step payload must name the step being
    gated, or a Tier 2 step reached at index 2 would be confirmed under an earlier
    step's hash.  Fail closed: no plan, an out-of-range index, or a missing
    decision yields the inert ``_no_step()`` stub (an empty, un-approvable
    ``confirm`` payload) instead of another step's decision.
    """
    steps = _steps(plan)
    if not steps:
        return _no_step()
    idx = max(int(step_index), 0)
    if idx >= len(steps):
        return _no_step()
    decision = _decision(decisions, steps[idx])
    return decision if decision is not None else _no_step()


def _no_step() -> Any:
    """A stub Decision for the degenerate 0-step case (should notnormally happen)."""
    return type(
        "._NoStep",
        (),
        {
            "step_id": "",
            "tier": 0,
            "summary": "",
            "needs_unlock": False,
            "needs_typed_confirmation": None,
            "resolved_paths": [],
            "action_hash": "",
            "warn_untrusted": False,
            "allowed": False,
            "needs_confirm": False,
        },
    )()
