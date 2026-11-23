# JARVIS P8 Orchestration Guard Implementation Report

## 1. Scope and outcome

P8 adds a deterministic orchestration layer that checks feasibility **before** the planner is called and makes terminal reporting honest when retries or partial requests are involved. The LLM still proposes; the closed registry, deterministic validation and existing PolicyEngine remain the only execution authorities.

The work was already present in the working tree when this session resumed. This report records the completed implementation, the static review, the limited verification permitted by low-resource constraints, and the remaining unverified areas.

## 2. What was implemented

### Pre-planning feasibility gate (`src/jarvis/agent/feasibility.py`)

- `assess_request()` classifies the current user request as `SUPPORTED`, `PARTIALLY_SUPPORTED`, `UNSUPPORTED` or `NEEDS_CLARIFICATION`.
- Conservative closed patterns stop known Tier-3-class requests before any model call: flight/hotel booking, payments, password changes, and disabling Defender/firewall/antivirus.
- Runtime capability checks use the real `ToolRegistry` and the already-wired `AppContext`; no check launches a browser, opens WhatsApp or reads a credential.
- `tool_availability()` fails closed for unregistered tools, Windows-only tools on non-Windows hosts, `browser_*` tools without a managed browser, and WhatsApp tools without the isolated WhatsApp Web session.
- Missing referents such as “send this message” are converted into a bounded clarification before planning.
- Mixed requests produce a partial plan constraint rather than a blanket refusal when a useful supported part remains.

### Brain and prompt integration

- `brain()` runs the feasibility check only after the deterministic fast paths and the no-LLM guard, so existing fast paths are preserved.
- Unsupported requests halt with `request_kind="unsupported"` and no LLM call.
- Clarifications reuse the bounded `_clarification_stop()` path.
- Partial requests add a deterministic rule to the system prompt through `compose_brain_context()` and `brain_system_prompt()`.
- `_project()` attaches the short limitation note only to a non-empty executable tool plan; conversation, clarification, unsupported and browse routes are unchanged.

### Validation integration

- `validate()` now calls `tool_availability()` for every registered tool before args validation and policy hashing. This adds a rejection path only; it never weakens registry, tier, path or policy enforcement.
- Unknown tools are still rejected first by the existing registry lookup.

### Effective-result handling

- `src/jarvis/agent/results.py` provides `effective_results()`, which keeps the latest `StepResult` per step id in first-seen order.
- `compose_rules`, `respond`, `memory_save` and `replan` now interpret retries through that helper, so a failed attempt followed by a success is not reported as a failure or archived as a dead end.
- Replacement plans remain safe because `validate._check_step_ids()` rejects reused completed step ids when `pending_replan` is set; this prevents collapsed identity across plan generations.

### Honest partial completion

- `partial_note()` now emits only the limitation itself: `I can't do this part here: ...`.
- The earlier wording `I did the rest.` was removed after static review because it could appear after a failed final answer and falsely imply success.
- `classify_outcome()`, `task_status_for()`, `spoken_answer()` and `respond._answer_for()` use the tool plan's `dialog_answer` limitation consistently.

## 3. Security and architecture review

Reviewed against the P8 diff and surrounding source:

- No change to `src/jarvis/policy/**`; PolicyEngine remains the sole authorization authority.
- No change to confirmation hashing, policy-gate interrupts, scope grants, taint propagation or target-binding stores.
- No generic shell, `eval`, `exec`, payment or password tool was introduced.
- Feasibility checks are advisory to the planner and validator; they do not lower tiers or authorize tools.
- Availability checks fail closed when required integrations are absent.
- Unregistered tools still cannot enter a validated executable plan.
- Fast-path and no-LLM behaviour remains ahead of the new gate.
- Step-ID collision protection was checked as the guard that keeps retry collapsing safe across replans.
- No secrets, phone numbers, message bodies or API keys are written to state or logs by the new code.

## 4. Baseline investigation status

- The baseline snapshot at `AppData/Local/Temp/opencode/jarvis-head-baseline` was verified against all 452 tracked files at current committed `HEAD`: 0 missing, 0 mismatched, 0 extra non-artifact files.
- Import probes with `WSLENV=PYTHONPATH/p` confirmed that baseline tests load the baseline source and current tests load the working-tree source.
- The baseline full-suite command was manually aborted and produced no final summary. It was not resumed.
- No baseline-versus-working-tree failure comparison was produced. No failure is claimed to be pre-existing or newly introduced on the basis of that aborted run.

## 5. Verification evidence

### Previously recorded evidence reused, not rerun

- `tests/unit/test_feasibility_gate.py`: **8 passed**.
- Nine-file focused P8-related test group: **passed**.

These results were supplied as prior successful evidence and were not repeated except where the two node IDs below were affected by the new wording fix.

### Minimal tests run in this session

One test process, two individual node IDs, no parallel workers:

| Command | Result |
|---|---|
| `.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider --tb=short tests/unit/test_feasibility_gate.py::test_partial_request_adds_a_bounded_prompt_constraint_and_plan_note tests/unit/test_feasibility_gate.py::test_partial_limitation_never_claims_success_on_failure` | exit 0, `.. [100%]` (2 passed; duplicated `-q` suppressed the summary line) |

The first node verifies that a partial request still adds the prompt constraint and a bounded plan limitation. The second was added in this session to verify that a failed partial task never claims “I did the rest”.

### Lightweight static checks

| Command | Result |
|---|---|
| `.venv/Scripts/python.exe -m ruff check <12 changed Python files>` | `All checks passed!`, exit 0 |
| `.venv/Scripts/python.exe -m ruff format --check <12 changed Python files>` | `12 files already formatted`, exit 0 |
| `.venv/Scripts/python.exe -m mypy --follow-imports=silent src/jarvis/agent/feasibility.py src/jarvis/agent/results.py` | `Success: no issues found in 2 source files`, exit 0 |
| `.venv/Scripts/python.exe -m mypy --follow-imports=silent src/jarvis/agent/feasibility.py src/jarvis/agent/results.py src/jarvis/agent/nodes/validate.py` | 1 issue: pre-existing `no-any-return` at `validate.py:225`, outside every P8 diff hunk |

`validate.py:225` returns an existing untyped value in the location-resolution helper. It was not introduced or touched by P8 and was left unchanged to avoid unrelated edits.

## 6. Deliberately not tested

To keep the Windows host responsive, this session did **not** run:

- the complete test suite or complete `tests/unit` directory;
- a new baseline test run or a full baseline-versus-P8 comparison;
- repeated integration or end-to-end suites;
- parallel pytest workers, stress tests or benchmarks;
- live Groq/Gemini provider calls;
- live Windows desktop, browser or WhatsApp Web exercises;
- voice/microphone verification;
- repository-wide Ruff or mypy.

No comprehensive regression claim is made.

## 7. Remaining limitations

- Full-suite regression status is unknown because the controlled baseline run was aborted and no replacement comparison was permitted.
- Feasibility classification is deliberately lexical and conservative; unusual wording may be treated as supported and still rely on the existing validator/policy layers.
- Runtime availability is checked only against context wiring and platform flags. It does not prove that a live browser profile, WhatsApp session or Windows integration is healthy.
- Partial plans carry a limitation note, but the supported part can still fail; the neutral wording now avoids a false completion claim.
- Retry collapsing depends on stable step ids and the existing replacement-plan ID collision check.
- Live Windows, WhatsApp, browser and voice behaviour requires Phase 10 real-world verification.
- Phase 10 packaging, autostart, upgrade and hardening remain outside this implementation.

## 8. Phase 10 manual verification list

On the Windows PC, with the configured provider available:

1. `jarvis chat` → “Explain Python decorators.” — conversational answer, no tool call.
2. `jarvis chat` → “Please book an airline ticket to Delhi.” — immediate unsupported halt, no provider call.
3. `jarvis chat` → “Open Notepad and book a flight.” — Notepad opens, final answer includes the limitation, no booking attempt.
4. `jarvis chat` → “Send a WhatsApp message to mom.” with no isolated WhatsApp profile — honest unavailable refusal.
5. `jarvis chat` → “Open Notepad.” — fast path remains active and does not invoke the planner.
6. Exercise a verified tool retry, then a replan, and confirm the final answer reflects the recovered result and does not duplicate execution.
7. Confirm a Tier 1 action still asks for confirmation and that TOCTOU path checks remain active.

## 9. Files changed

### Created

- `src/jarvis/agent/feasibility.py`
- `src/jarvis/agent/results.py`
- `tests/unit/test_feasibility_gate.py`
- `docs/JARVIS_P8_ORCHESTRATION_IMPLEMENTATION_REPORT.md`

### Modified

- `src/jarvis/agent/compose_rules.py`
- `src/jarvis/agent/context_composer.py`
- `src/jarvis/agent/nodes/brain.py`
- `src/jarvis/agent/nodes/memory_save.py`
- `src/jarvis/agent/nodes/replan.py`
- `src/jarvis/agent/nodes/respond.py`
- `src/jarvis/agent/nodes/validate.py`
- `src/jarvis/agent/prompts.py`
- `src/jarvis/agent/state.py`
- `PROGRESS.md`

Unrelated untracked audit documents were intentionally not staged.
