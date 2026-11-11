# B8 — Post-Implementation Audit

Date: 2026-11-11 · Branch: `feature/b1-task-event-store`

This audit treats the original B8 report (`docs/B8_IMPLEMENTATION_REPORT.md`) as a claim and
checks it against the source, the call graph and fresh test runs. It records what was already
correct, what was wrong, what was fixed, and what is still only mock-tested.

## 1. Repository integrity

| Item | Value |
|---|---|
| Starting branch / HEAD | `feature/b1-task-event-store` / `70252811f7c251a51c87159d651cc3ca3ee8df18` |
| Starting `git status --short` | `?? docs/JARVIS_BACKEND_AUDIT.md` |
| Final `git status --short` | `?? docs/JARVIS_BACKEND_AUDIT.md` (unchanged, never staged) |
| History rewritten | No - no reset, rebase, amend, stash or force-push |
| Pushed | Nothing |
| User-owned files | `docs/JARVIS_BACKEND_AUDIT.md` and the original `docs/B8_IMPLEMENTATION_REPORT.md` left untouched |

New commits (staged by explicit path only):

| Hash | Message |
|---|---|
| `33eb37e` | `feat(telemetry): record every real LLM attempt against its task` |
| `1268a43` | `test(agent): correct the four-provider premise and the arithmetic coverage` |
| this commit | `docs(backend): add the post-B8 implementation audit` |

## 2. Findings

### Correct in the original B8 (verified in source, not taken on trust)

* `src/jarvis/llm/capability.py` requirements are closed and declarative
  (`CONVERSATION_REQUIREMENT`, `PLANNER_REQUIREMENT`, `VISION_REQUIREMENT`); `satisfies()` reads
  only `ModelSpec.capabilities`, and refuses `pricing_mode == "unknown"` outright, so a model is
  never judged by its name.
* `provider._guard_capability` runs inside both `build_provider_client` and `_build_provider_slots`,
  i.e. before any provider client is constructed; `MultiProviderClient.structured` and
  `.describe_image` re-check per role during the fallback walk.
* Free-only / strict-zero-cost qualification still decides membership *before*
  `MultiProviderClient` exists, so no fallback can reach a paid model. Provider order is preserved
  by `_groups()` (first-appearance order, slots ascending).
* The brain call already emitted one `LLM_CALL` per real attempt, with a fallback visible as two
  attempts, per-request `LLMAttemptMeta` (no global task id) and the allow-listed detail fields only.

### Confirmed issues (all fixed)

1. **Telemetry coverage was genuinely incomplete.** Only `agent/nodes/brain.py` passed
   `attempt_meta`. Five other real provider calls produced no event:
   `agent/nodes/replan.py`, `agent/nodes/specialist.py`, `tools/web.py` (`web_answer`),
   `tools/whatsapp_web.py` (`_summarize_messages`) and `messaging/grounding.py` (`_interpret`).
   The last one was not even listed in the original report's inventory, although it is a real
   `llm.text(...)` call during an action.
2. **The tool layer had no task metadata at all.** `ToolContext` carried settings, browser and
   approval bindings but no task identity, and `act` built every context without one. Even a tool
   that wanted to record an attempt could not have attributed it.
3. **Stale test premise.** `test_freeonly.py::test_langgraph_uses_four_provider_multi_client` sent
   "What is 2 plus 2?", which the deterministic arithmetic fast path answers before any provider is
   consulted; the assertion expected the scripted provider's reply. The test's purpose is the
   four-provider walk, so its input - not its assertions - was wrong.

### Fixed, and how

* `llm/telemetry.py` gained two shared, cycle-free helpers: `build_attempt_meta(client, task_id,
  task_class, store)` (returns `None` when the client does not advertise `supports_attempt_meta`,
  the store is absent, or the task id is empty) and `attempt_kwargs(client, meta)` (the keyword only
  for clients that advertise support). `agent/nodes/telemetry.py::llm_attempt_meta` now delegates to
  the same builder, so graph nodes and tools cannot drift apart.
* `ToolContext.attempt_meta` was added (default `None`), `AppContext.tool_context(*, attempt_meta)`
  accepts it, and `act` builds every tool context through one `_tool_ctx()` helper that carries
  `llm_attempt_meta(state, ctx, "action")`. No global state, no per-process "current task".
* `web_answer`, the WhatsApp summariser and contact grounding now pass
  `attempt_kwargs(...)`; `ground_brain_update` builds the metadata from `state["task_id"]` and
  `ctx.tasks` and forwards it through `ground_contacts` → `_interpret`.
* `replan` and the specialist loop pass the same metadata to `structured()`.
* `test_freeonly.py::test_langgraph_uses_four_provider_multi_client` now sends a conversational
  request ("hello there") and keeps its original four-provider assertions
  (`structured_calls == 1` for every client, `text_calls == 0`). The deterministic-arithmetic
  guarantee is owned by `tests/unit/test_b8_acceptance.py::test_arithmetic_uses_the_safe_local_route`,
  which now asserts **both** `structured_calls == 0` and `text_calls == 0` and that the answer is
  `48`.

### Intentionally unchanged (with evidence)

* **The arithmetic fast path itself** - it is accepted product behaviour (`agent/fastpath.py`), not a
  B8 regression. Only the test that assumed otherwise was corrected.
* **`cli.py:419`** - `_live_provider_check` is the `jarvis doctor` connectivity probe. It builds a
  single-provider client deliberately and runs outside any task, so it has no task id to attribute
  and must not fabricate one; it correctly produces no `LLM_CALL` event.
* **Local OCR** (`perception/ocr.py`, `tools/screen.py`) is untouched and still PaddleOCR on local
  CPU. `describe_image` exists in the LLM layer but has **no caller in the agent graph**; the OCR
  path never requests the `vision` role.
* **Free-only / pricing / provider ordering / planner topology** - not modified by this audit.

## 3. LLM telemetry coverage

| Caller / path | Role | Provider invocation | Task metadata | Status |
|---|---|---|---|---|
| `agent/nodes/brain.py` | `fast` or `planner` (from `route_request`) | `MultiProviderClient.structured` | `llm_attempt_meta(state, ctx, route.task_class)` | recorded (pre-existing, verified) |
| `agent/nodes/replan.py` | `planner` | `structured` | `llm_attempt_meta(state, ctx, "action")` | recorded (new) |
| `agent/nodes/specialist.py` | `planner` | `structured` | same | recorded (new) |
| `tools/web.py::_run_web_answer` | `fast` | `text` | `ToolContext.attempt_meta` set by `act` | recorded (new) |
| `tools/whatsapp_web.py::_summarize_messages` | `fast` | `text` | `ToolContext.attempt_meta` | recorded (new) |
| `messaging/grounding.py::_interpret` | `fast` | `text` | built in `ground_brain_update`, forwarded explicitly | recorded (new) |
| `cli.py::_live_provider_check` (`jarvis doctor`) | `fast` | direct single-provider client | none (not a task) | intentionally not recorded |

Each real provider attempt yields exactly one event; a fallback yields one event per attempt
(index 0 failure, index 1 success) because the emission sites are inside the walk, not around it.
Capability skips happen before the call and produce no event. The detail payload is still an
allow-list: provider, model, role, outcome, duration, fallback index, cancelled/timed-out flags,
attempt id, task class, and tokens only when the provider reported them.

## 4. Capability correctness

* **Registration** - capabilities come only from `llm/models.py`; `satisfies()` never infers from a
  model name, and an unregistered pair (`pricing_mode == "unknown"`) is refused.
* **Context window** - `ModelSpec.context_tokens is None` is only fatal when a requirement sets
  `min_context_tokens`; none of the three shipped requirements does today, so an unknown window is
  not silently treated as sufficient *for a demand that does not exist* and would fail closed if one
  were added.
* **Role guards** - `fast` and `planner` are gated in the provider factory and re-checked in the
  composite walk. `text()` is deliberately ungated (a plain completion implies no capability) - the
  tool-layer consumers above all use `text()` on `fast`, which a free-tier model can serve.
* **Fallback eligibility** - eligibility and free-tier qualification are evaluated per candidate
  inside the walk, so a fallback cannot select a model that failed the requested capability or the
  pricing policy.
* **Local-only OCR** - `route_request(..., has_image=True)` → `local_ocr`, `planner` role;
  `requirement_for_task_class(local_ocr)` demands nothing (`vision=False`) and
  `PLANNER_REQUIREMENT.vision` is `False`. A new test asserts this, so the selector cannot route
  image text to a cloud vision model. Nothing in this audit added cloud vision.

## 5. Acceptance and regression results

All runs: `.venv/Scripts/python.exe`, Python 3.11, Windows, repo root. Nothing used a live provider,
network, browser, WhatsApp session or voice loop.

| Exact command | Passed | Failed | Runtime | Kind |
|---|---|---|---|---|
| `pytest tests/unit/test_b8_capability.py -q` | 11 | 0 | 3 s | unit, mocked |
| `pytest tests/unit/test_b8_llm_telemetry.py -q` | 9 | 0 | 3 s | unit, mocked |
| `pytest tests/unit/test_b8_acceptance.py -q` | 12 | 0 | 6 s | graph-level, scripted provider + temp store |
| `pytest tests/unit/test_b8_telemetry_coverage.py -q` | 9 | 0 | 4 s | unit + tool-level, mocked |
| `pytest tests/unit/test_freeonly.py -q` | 35 | 0 | 5 s | unit, mocked (includes the corrected four-provider walk) |
| `pytest tests/unit/test_web_answer.py -q` | 22 | 0 | 3 s | unit, mocked |
| `pytest tests/unit/test_invariants.py -q` | 37 | 0 | not separately measured | unit, mocked |
| Combined focused regression: `pytest test_b8_*.py test_freeonly.py test_agent.py test_agent_architecture.py test_taskstore_privacy.py test_brain_prompt.py test_act*.py test_grounding*.py test_whatsapp*.py test_context*.py test_tools_web*.py test_replan*.py test_specialist*.py -q` | all passed (0 failures, 100% bar) | 0 | completed inside the 180 s cap | mixed unit/integration, mocked |
| `ruff check <13 changed files>` | - | 0 | <1 s | lint |
| `ruff format --check <13 changed files>` | 13 already formatted | 0 | <1 s | format |

Notes:

* The combined regression invocation's exact test count was not captured (output was tailed); its
  exit status was 0 with no failure lines.
* The corrected test `test_freeonly.py::test_langgraph_uses_four_provider_multi_client` passes in the
  35-test `test_freeonly.py` run. The separate zero-call arithmetic test
  (`test_b8_acceptance.py::test_arithmetic_uses_the_safe_local_route`) passes in the 12-test
  acceptance run.
* Required telemetry cases and where they live: fallback attempts and no-duplicate emission
  (`test_b8_llm_telemetry.py`), capability rejection produces no event
  (`test_capability_skip_is_not_recorded_as_an_attempt`), token omission when unreported and
  per-task isolation (`test_b8_telemetry_coverage.py`), privacy allow-list
  (`test_privacy_policy_rejects_prompt_like_fields_on_llm_call`).

## 6. Remaining limitations

**Not implemented**

* No prompt-budget composer was added; context composition is the existing prompt-trim, history and
  memory-block implementation (unchanged, still opt-in for tool shortlisting).
* No new browser/WhatsApp/voice behaviour; those subsystems were only checked for regression.

**Implemented but not tested**

* Nothing in this audit's changed code is untested at unit level.

**Tested with mocks only**

* Every result in section 5. The graph tests use a scripted provider and a temporary SQLite task
  store; tool tests patch the search provider.

**Verified live**

* Nothing. No provider key was used, no browser was opened, no WhatsApp session attached, no audio
  captured. The capability/selection/fallback decisions are therefore verified against the registry
  and scripted clients, not against provider reality.

**Known remaining risk**

* `text()` is capability-ungated by design. A configured `fast` model that cannot do plain
  completions would fail at runtime rather than being filtered; every registered free model supports
  completions, so this is theoretical but worth remembering when a provider changes its API.

## 7. Final recommendation

**B8 complete with limitations.** All confirmed defects found by this audit were fixed and covered by
tests; the earlier report's main understatement (telemetry "brain only") is resolved, and its
inventory omission (contact grounding) is now recorded. The only outstanding items are live
verification, which needs a configured zero-cost provider key and a real browser/WhatsApp session,
and the unchanged prompt-budget/shortlisting scope. No concrete blocker remains in the backend code.
