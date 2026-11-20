# JARVIS Browser Specialist Loop Guard Implementation Report

Date: 2026-11-20
Branch: `main`
Status: implemented and unit-tested; no live browser/provider run performed.

## Milestone and objective

The browser specialist loop was already active in the real LangGraph flow, but its
repeat guard only compared the last two history entries. Automatic page-refresh
steps and intervening actions meant a cycle such as `search -> scroll -> search`
could consume the full `specialist_max_calls` budget before stopping.

The objective was to make repetitive specialist behaviour fail closed earlier
without changing the policy engine, tool registry, confirmation flow, or the
legitimate browsing-grant behaviour.

## What changed

- `src/jarvis/agent/nodes/specialist.py`
  - `decide()` now reads the full bounded `specialist_history` and checks for a
    repeated action signature before asking the LLM. Non-consecutive cycles halt
    with `Specialist stuck in a loop.`
  - After the LLM proposes a `browser_*` action, the same signature check runs
    against prior history before a step is returned to `validate`/`act`, so an
    A→B→A repetition does not execute again.
  - Signatures are `(tool, args-summary)` over the last 12 entries. Automatic
    refresh steps are marked `refresh` when recorded and excluded. Decision
    entries covered by a browsing grant are marked `grant` from the recorded
    policy decision.
  - Grant-covered `browser_goto` is excluded from repeat counting: repeated
    navigation to an already-approved URL is a legitimate refresh pattern and
    remains bounded by the grant action cap and `specialist_max_calls`. All
    other grant-covered tools still count, so a granted `browser_search` cycle
    is stopped.
  - `_record_history()` writes the plain `refresh`/`grant` markers into the
    existing `list[dict[str, Any]]` state field; no new state type, database,
    policy rule, or tool was added.

- `tests/unit/test_specialist.py`
  - Added direct tests for non-consecutive repeat detection, proposed-repeat
    blocking before execution, refresh exclusion, grant-covered navigation
    exclusion, and grant-covered non-navigation still counting as a repeat.

- `tests/unit/test_specialist_loop_e2e.py`
  - Added an end-to-end specialist cycle test: `search -> scroll -> search`
    halts before the repeated search executes. The assertion filters generated
    `r<call>` refresh steps so it checks the user-visible decision sequence.

## What was already present

- The specialist loop, call cap, recoverable gate refusals, validate rejection
  recovery, exactly-once step execution, browser confirmation policy, browsing
  grants, taint marking, and task telemetry were already implemented.
- The original guard stopped only adjacent duplicate actions. This change adds
  bounded non-consecutive detection and an early pre-execution check; it does
  not weaken or bypass any policy, tier, confirmation, or target-binding rule.
- Grant-covered repeated `browser_goto` was already a supported real flow;
  `test_first_prompt_text_and_grant_covers_the_next_step` proved that before
  this change and exposed the regression during this session.

## Verification

Commands run with `.venv\Scripts\python.exe` from the project root:

- Focused milestone tests:
  `pytest -o addopts="" -q tests/unit/test_specialist.py tests/unit/test_specialist_loop_e2e.py`
  Result: 18 passed.
- Neighbour regression set:
  `pytest -o addopts="" -q tests/unit/test_specialist.py tests/unit/test_specialist_loop_e2e.py tests/unit/test_browse_grant_wiring.py tests/unit/test_browse_grant.py tests/unit/test_agent_architecture.py tests/unit/test_compose_observations.py tests/unit/test_replan.py tests/unit/test_invariants.py`
  Result: 140 passed.
- `mypy src/jarvis/policy`: no issues in 11 source files.
- `ruff check` on the three changed Python files reports only findings that
  also exist at HEAD in `tests/unit/test_specialist.py`; the changed source
  file has no findings. `ruff format --check` flags only pre-existing formatting
  drift in the two test files; the added hunks and the source file are clean.
- No live provider, browser, WhatsApp, voice, or desktop integration test was
  performed. The end-to-end specialist test uses `FakeLLM`, a fake browser
  session and a real SQLite checkpointer; it verifies graph wiring, not live
  browser behaviour.

## Interrupted full-suite run

`pytest` was interrupted at roughly 83% and produced no final summary. The
cached failure list at `.pytest_cache/v/cache/lastfailed` was written at
2026-11-20 17:32:24 and contains 145 exact node IDs. None is in the two
specialist test files.

A representative subset was checked against the current tree. These cached IDs
do not resolve now, so the cache is stale relative to the current test tree:
`test_browse_grant_wiring.py::test_enter_after_cross_host_navigation_is_not_covered_by_act`,
`test_agent_architecture.py::test_conversation_blank_answer_halts_without_calling_fast_model`,
`test_compose_observations.py::test_failed_result_outranks_a_brain_final_answer_everywhere`,
`test_b8_capability.py::test_configured_registry_models_satisfy_their_own_role[fast]`,
and `test_taskstore_privacy.py::test_taskstore_import_does_not_import_memory`.

One real regression was found by the neighbour run, not by the stale cache:
`tests/unit/test_browse_grant_wiring.py::test_first_prompt_text_and_grant_covers_the_next_step`
failed with `Specialist repeated an action.`. It was fixed by excluding only
grant-covered `browser_goto` from repeat counting and now passes. No other
cached failure was reproduced or repaired; the remaining identifiers are
classified as unrelated or undetermined, not as proven pre-existing failures.

## Known limits

- The detection window is 12 history entries and uses canonical JSON argument
  summaries; semantically equivalent actions with different argument text are
  not treated as repeats.
- Grant-covered navigation can repeat until the grant action cap or call cap is
  reached; it is deliberately not a loop error.
- All verification is unit/integration-style with fakes. Live browser behaviour
  remains unverified.
