# JARVIS P5 Evidence-Based Verification — Implementation Report

## 1. Milestone

Implemented **closed, deterministic evidence-based step verification integrated with expectation-aware replanning** as the next P5 improvement. This is a focused execution-path change, with no audit or planning work, no policy changes, and no new dependencies.

Implementation commit: `f15638db9d9b55ab2f04e565cfe45a483383619f` (`f15638d`) — `feat(agent): add evidence-based step verification`. Parent milestone: `a8ac351` (`feat(agent): make replanning expectation-aware`).

## 2. Plan followed

1. Inspected `verify.py`, `replan.py`, `brain.py`, `state.py`, `schemas.py`, `prompts.py`, `llm/prompts.py` and their tests.
2. Chose a backward-compatible evidence contract that can be checked from existing `ToolResult` fields only.
3. Added a small deterministic evaluator module and wired it into the existing verify → replan/fail-closed path.
4. Exposed the contract to planner/replanner prompts without exposing secrets or new serialized types.
5. Added focused unit, schema, prompt, non-authorization, state/serde and graph-level tests.
6. Ran focused regressions, Ruff, mypy, reviewed the diff, committed implementation, then recorded status here.

## 3. Files changed

### Created

- `src/jarvis/agent/evidence.py`
- `tests/unit/test_evidence_verification.py`

### Modified

- `src/jarvis/agent/state.py`
- `src/jarvis/agent/schemas.py`
- `src/jarvis/agent/nodes/verify.py`
- `src/jarvis/agent/nodes/brain.py`
- `src/jarvis/agent/nodes/replan.py`
- `src/jarvis/agent/prompts.py`
- `src/jarvis/llm/prompts.py`
- `tests/unit/test_schemas.py`

### Documentation

- `PROGRESS.md`
- `docs/JARVIS_P5_EVIDENCE_VERIFICATION_IMPLEMENTATION_REPORT.md` (this report)

## 4. Implementation details

### Closed evidence contract

`EvidenceKind` and `EvidenceStatus` are plain `Literal` strings in `src/jarvis/agent/state.py:20` and `src/jarvis/agent/state.py:29`, so the existing checkpoint serializer allowlist is unchanged.

| Identifier | Meaning | Deterministic check |
|---|---|---|
| `action` | Execution itself is the expectation | Tool executed successfully |
| `resource` | A file/artifact/result must exist | Non-empty structured result keys, or tool-reported `verified` |
| `information` | Text/data must be retrieved | Non-empty `output`, or non-empty structured result keys |
| `state` | A tool post-condition proves the outcome | The tool's existing `spec.apply_verify(...)` result |
| `unverifiable` | No independent check exists | Always inconclusive with a constant honest note |

`EVIDENCE_KINDS`, `EvidenceVerdict`, `normalise_kind()` and `evaluate_evidence()` live in `src/jarvis/agent/evidence.py` (`EVIDENCE_KINDS` at line 27, `evaluate_evidence()` at line 87). Unknown values are normalised to `""`; the planner/replanner can only select from the closed vocabulary because `ActionIntent.evidence` and `Step.evidence` are Pydantic-validated.

### Execution path

- `brain._project()` copies `ActionIntent.evidence` into `Step.evidence` before validation/approval.
- `verify()` evaluates the declared contract **only after** the existing tool `verify()` logic, using `last.model_copy(update={"verified": verified, "verify_note": note})`; it never re-runs a side-effecting tool.
- Verdict mapping: `confirmed -> verified=True`, `contradicted -> verified=False`, `inconclusive -> verified=None`.
- Contradicted evidence sets `non_retryable=True`, so the graph skips a blind identical retry and routes to replan/fail-closed through the existing `route_after_verify()` logic. The specialist path still observes before replanning, which is unchanged.
- Inconclusive evidence adds only the constant string `"The expected information could not be independently observed."`; no expectation text, output text or data value is copied into state notes or answers.
- When a tool's own `verify()` reports `False` but declared evidence is `confirmed`, the verdict is forced to `contradicted` with the constant note `"An independent check of the expected outcome failed."`.
- `StepResult.evidence` and `StepResult.evidence_status` are written by `verify()` and included in `ToolCall` within `replan._toolcalls()`; `_remaining_steps()` also includes each pending step's evidence kind. The replan prompt therefore sees the verdict as untrusted context but cannot change authorization.

### One check per step, trust and safety

- Evidence is evaluated exactly once per verified attempt, inside `verify()`, before replanning and before final answering.
- Evidence is LLM-proposed metadata only. The deterministic evaluator decides the verdict; the PolicyEngine still makes every authorization decision.
- No approval hash, permission decision, target binding, confirmation state or policy tier is read or modified by the evidence path.
- Prompts explicitly forbid passwords, keys and phone numbers in `expect`, and no tool argument values were added to prompts by this milestone.
- No new CLI, database, tool, dependency, state type, LLM call, graph node or edge was introduced.

## 5. New tests

`tests/unit/test_evidence_verification.py` contains 16 tests:

- `test_evaluate_evidence_action_confirmed`
- `test_evaluate_evidence_resource_confirmed_from_data`
- `test_evaluate_evidence_resource_contradicted_by_empty_data`
- `test_evaluate_evidence_information_confirmed_from_output`
- `test_evaluate_evidence_state_uses_tool_verify`
- `test_evaluate_evidence_state_contradicts_a_tool_verify_failure`
- `test_evaluate_evidence_unverifiable_is_inconclusive`
- `test_evaluate_evidence_ignores_unknown_kinds_and_invalid_results`
- `test_verify_records_confirmed_evidence`
- `test_verify_records_contradicted_evidence_and_skips_retry`
- `test_verify_records_inconclusive_evidence_without_blaming_the_user`
- `test_verify_keeps_legacy_expectation_note_when_evidence_is_absent`
- `test_evidence_does_not_change_policy_or_approval_state`
- `test_evidence_fields_survive_checkpoint_serde`
- `test_graph_reports_failure_when_a_successful_tool_has_no_expected_resource`
- `test_graph_replans_once_instead_of_repeating_a_contradicted_action`

`tests/unit/test_schemas.py` was updated so `test_action_intent_defaults_and_extra_forbid` checks the new `evidence` field and its `None` default.

## 6. Verification commands and results

Commands were run from the repository root with `.venv/Scripts/python.exe` because `python` is not on PATH.

| Command | Result |
|---|---|
| `.venv/Scripts/python.exe -m pytest -o addopts="" -q tests/unit/test_evidence_verification.py` | `16 passed in 2.18s` |
| `.venv/Scripts/python.exe -m pytest -o addopts="" -q tests/unit/test_evidence_verification.py tests/unit/test_replan.py tests/unit/test_verify_nonretryable.py tests/unit/test_schemas.py tests/unit/test_compose_observations.py tests/unit/test_outcome_agreement.py` | `96 passed in 2.56s` |
| `.venv/Scripts/python.exe -m pytest -o addopts="" -q tests/unit/test_agent.py tests/unit/test_agent_architecture.py tests/unit/test_plan_step_ids.py tests/unit/test_invariants.py tests/unit/test_b9_context_discovery.py tests/unit/test_plan_summary.py tests/unit/test_replan_scope.py tests/unit/test_specialist_loop_e2e.py tests/integration/test_confirmation_resume.py` | `151 passed in 9.09s` |
| `.venv/Scripts/ruff.exe check <10 changed Python files>` | `All checks passed!` |
| `.venv/Scripts/ruff.exe format --check <10 changed Python files>` | `10 files already formatted` |
| `.venv/Scripts/mypy.exe src/jarvis/policy` | `Success: no issues found in 11 source files` |
| `.venv/Scripts/mypy.exe src/jarvis/agent/evidence.py src/jarvis/agent/nodes/verify.py --follow-imports=silent` | `Success: no issues found in 2 source files` |

Total pytest results reported above: **16 new + 80 focused neighbours + 151 broader regressions = 247 passed**, zero failed in the executed commands.

## 7. Review results

- **Security**: no policy/tier/confirmation/target-binding code changed; evidence cannot lower a tier or authorize an action. Prompt context is escaped via the existing replan composer; notes are constants; no secrets enter state or logs.
- **State compatibility**: all new Pydantic fields are optional with defaults; a legacy `Step.model_validate(...)` and checkpoint serde round trip are covered by tests.
- **Flow compatibility**: existing `verify_enabled` ablation, retry limits, `max_replans`, specialist observation, fail-closed halting and final answering remain unchanged.
- **Diff**: source changes are additive and small; no stale test expectations were weakened to make failures pass.

## 8. Manual verification commands

Run these on the development PC:

```powershell
.venv\Scripts\python.exe -m pytest -o addopts="" -q tests\unit\test_evidence_verification.py
.venv\Scripts\python.exe -m pytest -o addopts="" -q tests\unit\test_evidence_verification.py tests\unit\test_replan.py tests\unit\test_verify_nonretryable.py tests\unit\test_schemas.py tests\unit\test_compose_observations.py tests\unit\test_outcome_agreement.py
.venv\Scripts\ruff.exe check src\jarvis\agent\evidence.py src\jarvis\agent\state.py src\jarvis\agent\schemas.py src\jarvis\agent\nodes\brain.py src\jarvis\agent\nodes\replan.py src\jarvis\agent\nodes\verify.py src\jarvis\agent\prompts.py src\jarvis\llm\prompts.py tests\unit\test_evidence_verification.py tests\unit\test_schemas.py
.venv\Scripts\mypy.exe src\jarvis\policy
jarvis doctor
jarvis chat
```

Suggested smoke prompt in `jarvis chat`: `Create a short note file with the Python installation steps, then tell me only if the file was actually created.` The LLM may declare `evidence="resource"`; a successful tool call with an empty result must be reported as failure/inconclusive rather than silently claimed as success. No live provider, Windows GUI, browser, WhatsApp or voice verification was performed here.

## 9. Known limits

- Evidence checks are generic and based on existing `ToolResult` shape (`ok`, `verified`, `output`, `data`); they do not prove arbitrary real-world semantics (for example, correct file contents).
- `resource` checks treat empty structured results as contradiction, which is correct for "must exist" expectations but not for tools whose legitimate result is an empty list; the planner should choose `action` or `information` in that case.
- The LLM still chooses the contract from the closed vocabulary; a deliberately wrong choice can weaken the check for that step, but cannot authorize, modify policy or hide a failed tool.
- `unverifiable` is honest but does not improve task success; it only prevents false confidence.
- No per-tool typed evidence adapters, benchmark ablation or live end-to-end desktop run was added.

## 10. Commit hash and message

- Implementation: `f15638db9d9b55ab2f04e565cfe45a483383619f` — `feat(agent): add evidence-based step verification` (10 files, 699 insertions, 13 deletions).
- Documentation/status commit: `docs(progress): record P5 evidence verification` (this report and `PROGRESS.md`).

## 11. Next milestone

Add typed, per-tool evidence adapters with explicit result keys (for example `file_written`, `rows_returned`, `window_found`) and benchmark the evidence-enabled path against the `verify_enabled=false` ablation, before moving to the next P5 memory/skills milestone.

## 12. Final status

Implemented a closed evidence contract and wired it into execution, replanning and prompts without touching policy, approvals, tools or state serialization. Tested 247 passing unit/integration checks plus clean Ruff and mypy on relevant files. Next step is typed per-tool evidence adapters and an ablation measurement.
