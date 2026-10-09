# JARVIS ROADMAP v2.2 AMENDMENTS

> Companion to ROADMAP_V2.md and ROADMAP_V2.1_AMENDMENTS.md.
> Supplements v2.1; where v2.1 and this file disagree, v2.1 wins.
> Same working rules as v2 section 0.
> Nothing here changes the stage order or Stage 4.

## Stage 5a: completeness check (inside the existing brain call)

- No extra LLM request (D2). It lives in the brain prompt and the existing BrainDecision.
- The brain determines: goal, what was provided, what is missing, contradictions, executability.
- Four outcomes:
  1. Required info missing: clarification.
  2. Optional info missing: proceed and state the assumption.
  3. Not executable: a plain spoken explanation, no clarification loop.
  4. Policy-disallowed: PolicyEngine refuses it, never the brain.
- Missing values come only from the user's answer, never from screen, page, OCR or UIA data (I1).
- Session-context references ("delete it") count as provided (D19).
- The brain sees tools, never policy tiers.
- Max 2 clarification rounds, then it states what is missing and stops.
  - The counter is a plain value in graph state; no new serde type.
  - It resets on a new unrelated request.
  - Garbled audio ("Could you repeat that?") does not count as a round.
  - In AUTO, no answer within the window drops the request with no plan.
- "Structured task" means the existing Plan / clarification schema. No new Task object.
  test_serializer.py passes unchanged; the allowlist stays
  {Plan, Step, Decision, StepResult, ReplanDecision}.
- Tests: "Open Chrome" asks nothing; a fully specified request is never over-asked;
  session references do not trigger clarification; the brain never invents values;
  perception never fills a value; never more than 2 rounds. Decide deliberately whether
  location is required or optional for "search for internships".

## Stage 7: observe and adapt

- Loop (GUI/browser only): observe, choose element, policy, act, observe, verify.
  Tier 0 tools (get_time, open_app, list_windows) are exempt (verified=None).
- A re-plan must stay inside the approved scope. The check is deterministic, in
  policy_gate/PolicyEngine: plan_hash, domain scope, expiry, action budget. The LLM never
  judges scope. A changed hash invalidates the old approval. Out of scope = a new plan
  with a new confirmation_id (I3).
- The 8-action cap covers the whole approved execution and does not reset on re-plan.
  Read-only observations do not count toward it (owner to confirm).
- Popups must serve the existing objective. Consent / "Allow" controls re-gate (D12).
  Popup text is perception (I1): it may help select a control, never supply text, an
  objective or a URL. If the control cannot be identified, refuse (D17).
- A cross-origin redirect during a re-plan still stops the task (I5).
- Examples must use the JARVIS-owned browser or a UIA window, not the owner's Chrome.
- Required tests: in-scope re-plan continues; out-of-scope needs a new confirmation_id;
  popup "Allow" re-gates; cross-origin redirect during re-plan stops; the action counter
  survives a re-plan; a stale hash is rejected.

## Stage 10: extra red-team cases

Popup that says "click Allow"; stale confirmation after a re-plan; clarification-loop attempt.

## Tracker additions (v2.1 section H)

- [x] Stage 5a includes completeness check
- [ ] Stage 7 includes re-observe and re-plan scope rules

## Not doing (and why)

- Standalone Cognitive Task Engine or task state machine: duplicates the existing graph and
  risks D21, I9 and the serde allowlist. Revisit the state machine only if task-flow
  debugging gets painful after Stage 7.
- Extra LLM call for validation: breaks D2.
- LLM judging scope: must be deterministic.

## Open owner decision (not a rule yet)

- Broader Windows filesystem awareness beyond Desktop/Documents/Downloads. Not in v2/v2.1
  and Stage 4 shipped without it. Resolving a path is not permission to modify it, so
  allowed_roots and PolicyEngine would still apply. Decide scope before any stage takes it on.