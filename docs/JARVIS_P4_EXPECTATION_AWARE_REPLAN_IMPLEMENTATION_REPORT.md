# JARVIS P4 Privacy Verification and Expectation-Aware Replanning Report

Date: 2026-11-20
Branch: `main`
Status: implemented and unit-tested; no live provider, browser, voice or desktop run performed.

## Part 1 — PreferenceStore privacy

The reported concern is no longer valid at this revision. The central write path is
`PreferenceStore.set` in `src/jarvis/memory/prefs.py:137`: it normalises the key, rejects
credential-shaped keys via `normalise_key`, and rejects credential-shaped values via the
shared `looks_secret` detector before any SQLite write. That detector also includes
standalone vendor token shapes through `looks_like_token`, so a raw token under a harmless
key is refused.

The other persistence entrypoint, `SqliteMemory.remember`, delegates every preference
record to `self.preferences.set` in `src/jarvis/memory/store.py:109`; there is no second
write path that bypasses the guard. The conversational path uses the same shared detector,
so store and chat behaviour cannot silently diverge. Rejection messages are constant and do
not contain the supplied value.

Focused verification:

```powershell
& .\.venv\Scripts\python.exe -m pytest -o addopts="" -q tests/unit/test_memory_secret_tokens.py tests/unit/test_memory.py::test_preference_round_trip tests/unit/test_memory.py::test_preference_set_overwrites tests/unit/test_memory.py::test_preference_rejects_unusable_keys tests/unit/test_memory.py::test_preference_rejects_secret_values tests/unit/test_memory.py::test_preference_delete tests/unit/test_invariants.py -k "secret or credential or preference or token"
```

Result: `44 passed, 34 deselected in 3.31s`.

No privacy code was changed; adding another validation layer would duplicate the existing
central guard.

## Part 2 — Orchestration improvement

### Actual limitation

`Step.expect` is described as the planner's checkable success condition
(`src/jarvis/agent/state.py:47`) and is written by the brain, but before this change no
verify or replan node read it. The replanner's only observation view was `ToolCall`, which
had no expectation field, and the prompt did not show what work remained after the failing
step. A repair LLM could therefore propose a replacement plan from tool output alone,
without the original success criteria or the unfinished part of the plan.

### Implemented change

- Added `expect` to the transient `ToolCall` boundary model
  (`src/jarvis/agent/schemas.py:74`) and populated it from the matching plan step in
  `src/jarvis/agent/nodes/replan.py`.
- Added a bounded `_remaining_steps` view of unfinished work: step id, tool, rationale,
  expectation and argument *keys* only, never argument values. It is capped at eight steps
  and 120 characters per text field.
- Passed that view through `compose_replan_context` into `replan_user_prompt`, inside a
  separate `<untrusted_data>` block rendered with the existing escaping helper, so plan
  text cannot close the block or issue instructions.
- `verify` now attaches a constant, non-sensitive note when a step has a stated expectation
  but the tool's verification returns `None`: `Planned success condition was not
  independently checked.` The expectation itself is never copied into state, logs or the
  final answer.

No graph edges, budgets, tool registry entries, tiers, confirmation hashes, approvals,
target bindings or policy checks were changed. No extra LLM call was added; the existing
replan call simply receives better information.

### What JARVIS can now do better

After a failed step exhausts its retries, the replanner can see both what each earlier step
was supposed to achieve and what was still pending when the failure occurred. It can avoid
repeating the same approach, preserve the completed prefix, and produce a more targeted
replacement plan or an honest stop. Successful-but-unverifiable steps with an explicit
expectation are reported without claiming that the expectation was proven.

## Files changed

- `src/jarvis/agent/schemas.py`
- `src/jarvis/agent/nodes/verify.py`
- `src/jarvis/agent/nodes/replan.py`
- `src/jarvis/agent/context_composer.py`
- `src/jarvis/llm/prompts.py`
- `tests/unit/test_replan.py`
- `tests/unit/test_schemas.py`
- `PROGRESS.md`
- `docs/JARVIS_P4_EXPECTATION_AWARE_REPLAN_IMPLEMENTATION_REPORT.md`

The pre-existing untracked audit docs were left untouched.

## Verification

Commands executed with `.venv\Scripts\python.exe` from the project root:

- Focused orchestration set: `pytest -o addopts="" -q tests/unit/test_replan.py
  tests/unit/test_schemas.py tests/unit/test_compose_observations.py
  tests/unit/test_agent_architecture.py` — `81 passed in 3.57s`.
- Neighbouring regression set: `pytest -o addopts="" -q tests/unit/test_agent.py
  tests/unit/test_agent_architecture.py tests/unit/test_replan.py
  tests/unit/test_verify_nonretryable.py tests/unit/test_compose_observations.py
  tests/unit/test_b9_context_discovery.py tests/unit/test_brain_prompt.py
  tests/unit/test_schemas.py tests/unit/test_act_node.py tests/unit/test_batch_graph.py
  tests/unit/test_policy_gate_batch.py tests/unit/test_specialist_loop_e2e.py
  tests/unit/test_invariants.py tests/integration/test_confirmation_resume.py` —
  `252 passed in 9.79s`.
- `mypy src/jarvis/policy` — no issues in 11 source files.
- `ruff check` on the seven changed Python files — `All checks passed!`.
- `ruff format --check` on the seven changed Python files — `7 files already formatted`.

## Errors found

No error was introduced by the implementation. A broad memory test selection surfaced the
unrelated existing failure
`tests/unit/test_memory.py::test_retrieve_returns_skills_first_then_preferences_then_failures`
(`assert any(r.kind == "preference" for r in records)`); it is outside this privacy and
orchestration scope and was not changed.

## Known limits

- Expectations remain human-readable prose. JARVIS can report that one was not checked; it
  cannot prove arbitrary prose expectations without a dedicated tool verifier.
- Remaining-work context is bounded and contains argument names, not values, so a replanner
  cannot infer dataflow from earlier tool output.
- No general per-step LLM decision loop was added. This extends replanning only; the browser
  specialist remains the observation-driven subsystem for browsing tasks.
- All verification is unit/integration-level with fakes. Live multi-step behaviour with a
  real provider or desktop tools remains unverified.
