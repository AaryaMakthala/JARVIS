# B9 — Context-Aware Intelligence, Dynamic Tool Discovery & Prompt Optimization

Date: 2026-11-11 · Branch: `feature/b1-task-event-store`

## 1. Repository state

| Item | Value |
|---|---|
| Starting branch | `feature/b1-task-event-store` |
| Starting HEAD | `506c4e439718678e1aa83704c6dfe35c985ceab4` |
| Starting status | clean except untracked `docs/JARVIS_BACKEND_AUDIT.md` and `docs/PYTHON_PREFERENCE_LIVE_VERIFICATION.md` |
| Final HEAD before this report | `1a665b5` |
| Final status | the same two untracked docs remain untouched; no unrelated file was staged |
| Pushed | **nothing**; no reset, rebase, amend or force-push; explicit `git add` paths only |

The requested starting branch and HEAD matched the repository exactly. The production preference record
`fact.preference.my_preferred_coding_language_0dff4061f1` and the existing SQLite databases were not
modified, deleted or migrated by B9.

## 2. Commits

| Hash | Message | Purpose |
|---|---|---|
| `8845b95` | `feat(agent): add compact tool capability index` | Registry-derived descriptors, categories, keywords and bounded lexical discovery |
| `d213a21` | `feat(agent): compose bounded task-specific model context` | Shared composer, compact/full-schema prompt mode, explicit budget and trimming |
| `9dc7b57` | `feat(agent): route brain and replanner through discovery` | Brain/replan wiring, one bounded schema-resolution retry, context telemetry, memory relevance |
| `1a665b5` | `test(agent): cover B9 context discovery and budgets` | 26 mocked acceptance tests |
| docs commit | `docs(agent): report B9 context discovery implementation` | This report and PROGRESS |

## 3. Source files changed and why

| File | Change |
|---|---|
| `src/jarvis/agent/capability_index.py` (new) | `ToolDescriptor`/`CapabilityIndex`, compact directory rendering, deterministic shortlist, full canonical schemas for selected names only |
| `src/jarvis/agent/context_composer.py` (new) | One composer for brain and replan; budget, trimming order, statistics, context-length refusal |
| `src/jarvis/agent/tool_groups.py` | Public `group_keywords()` view reused by discovery; group table is unchanged |
| `src/jarvis/agent/prompts.py` | Optional compact directory; `tool_schema` discovery rule; explicit “do not guess a personal fact” memory instruction |
| `src/jarvis/agent/schemas.py` | `BrainDecision.request_type="tool_schema"` and `requested_tool` (transient, never checkpointed) |
| `src/jarvis/agent/context.py` | `AppContext.capability_index` built from the canonical registry |
| `src/jarvis/config.py` | `prompt_trim_enabled = true` by default plus explicit context-budget and candidate limits |
| `src/jarvis/agent/nodes/brain.py` | Uses the composer; one bounded schema-resolution retry; safe context log line |
| `src/jarvis/agent/nodes/replan.py` | Uses the composer for a bounded, shortlisted replanner prompt |
| `src/jarvis/llm/prompts.py` | Replanner accepts the compact directory and shortlist |
| `src/jarvis/llm/telemetry.py`, `src/jarvis/llm/multi.py`, `src/jarvis/agent/nodes/telemetry.py`, `src/jarvis/taskstore/privacy.py` | Safe context counts on each real LLM attempt, allow-listed; no DB migration |
| `src/jarvis/memory/store.py` | Preference/fact retrieval is query-relevant; an unrelated fact is no longer injected |
| `tests/unit/test_b9_context_discovery.py` (new) | Acceptance A–J coverage |
| `tests/unit/test_memory_b4.py` | Reservation test now uses a query that is actually about the stored fact |

No new dependency, vector database, cloud router, orchestrator, tool or generic shell path was added.
LangGraph, the registry, Pydantic schemas and the PolicyEngine keep their authority.

## 4. Actual runtime flow

The existing graph topology is unchanged:

```
START -> intake -> context_resolve -> memory_retrieve -> brain
                                                            |
          +--------------------------+----------------------+-------------------+
          | conversation             | action               | clarification     |
          v                          v                      v                   v
       compose <- respond         validate               clarify            specialist (optional)
                                     |
                               prepare_readback -> policy_gate -> act -> verify
                                     |                              |
                                     +<- replan <- failure <--------+
```

B9 changes only **what context `brain` and `replan` receive** and how a missing schema is handled.
`act`, `verify`, `policy_gate`, confirmations, target binding, action hashes and the task stores are
untouched.

## 5. How the brain selects conversation vs tool execution

* The brain still makes exactly one structured `BrainDecision` call for a normal request. There is
  no separate router call.
* `conversation`, `action`, `clarification`, `unsupported`, `browse` and the new bounded
  `tool_schema` continuation are distinct values in the same schema.
* `jarvis.agent.routing.route_for_brain_result` still maps the projected state to the next node;
  the model never chooses the graph edge.
* Unknown tool names are never expanded or executed. `validate` rejects them and the existing
  one-repair path applies.
* A registered tool whose full schema was not sent can be named with `request_type="tool_schema"` and
  `requested_tool`. The brain verifies the name in the canonical registry, re-asks **once** with that
  schema, and if the model still cannot decide it returns a bounded clarification. This is the only
  extra call B9 can add.

Efficiency contract (measured paths):

| Request | Calls |
|---|---|
| Ordinary chat | One structured call (fast role when the existing router classifies it as conversation) |
| Supported time/date/arithmetic fast path | Zero LLM calls |
| Clear single-tool request | One structured call; deterministic execution adds none |
| Memory question | Local retrieval + the normal response call |
| Multi-step task | One planner call; executor/verifier are deterministic; replan only after a failure |
| Registered tool missing from the shortlist | At most one extra structured schema-resolution call |

## 6. Compact descriptors and full schemas

`build_capability_index(registry)` reads the canonical `ToolSpec` list and produces one descriptor per
tool: name, short description, category, keywords/aliases, read-only or changes-state, declared base
tier and “normally requires confirmation”. The numeric tier and confirmation flag are internal
ranking/observability metadata and are **not rendered**; docs/04 keeps the planner out of permission
reasoning.

`render_capability_directory(index)` is a names-and-short-descriptions view of all 57 tools.
`discover_candidates(...)` scores the user's words lexically against names, curated aliases, group
keywords and descriptions; it returns at most 5 candidates. `catalogue_entries(...)` then loads the
complete canonical JSON schemas for the selected names only (default 4). Nothing in the compact index
is an executable definition.

A registry of 8 tools or fewer is described in full; this keeps small test/embedded registries
byte-simple and avoids a pointless shortlist of everything.

## 7. Notepad, browser and memory selection

* **Notepad**: “Open Notepad.” selects `open_app` first; its complete canonical schema is in the
  prompt. `open_in_app`, `open_path` and `list_windows` may also be shortlisted as related desktop
  capabilities. No `browser_*` or `whatsapp_*` schema is supplied. Policy still decides whether
  `open_app` may run.
* **Browser**: “Search the web for recent AI research.” shortlists `web_answer`, `browser_search`,
  `google_search` and `browser_open_result`; no desktop application schema is supplied. Origin,
  domain, grant, target-binding and confirmation rules are unchanged and still enforced by the gate.
* **Memory**: `memory_retrieve` still returns bounded, secret-filtered records. B9 adds query
  relevance to preference/fact retrieval: the explicit Python fact is returned for “preferred coding
  language”, while an unrelated decorators question does not receive it. The prompt says to answer
  only from explicit records and to say it does not know when the fact is absent. Remember/forget
  remains explicit and deterministic; the LLM cannot silently store a guessed fact.

## 8. Multi-step planning and step schemas

* The planner is the same brain call. It receives the compact directory plus full schemas for the
  shortlisted candidates, not all 57 schemas.
* Execution is deterministic: `act` loads the canonical Pydantic model from the registry, re-validates
  arguments, re-decides policy and re-observes the target. There is no per-step executor LLM call to
  optimize or to give a step-specific schema.
* `verify` uses the real mocked/recorded `StepResult`, not the model's claim. Replanning is bounded by
  `max_replans` and now receives only relevant schemas, bounded memory and bounded observations.
* The replanner cannot authorize future steps; any replacement still goes through `validate`,
  `policy_gate` and `act`.

## 9. Context budget and prompt trimming

New `[agent]` settings (defaults):

```
prompt_trim_enabled = true
context_budget_tokens = 24000
context_reserved_output_tokens = 1024
context_safety_margin = 0.15
tool_candidate_limit = 5
tool_schema_limit = 4
```

The effective input budget is `24000 × 0.85 − 1024 = 19,376` estimated tokens. This is explicitly an
application request budget, **not** a claim about any model's context window. `ModelSpec.context_tokens`
remains `None` for every registered model and dynamic aliases remain unknown.

`estimate_tokens` is `ceil(chars / 4)` and is always labelled estimated. Provider-reported
`prompt_tokens`/`completion_tokens` remain separate in LLM telemetry.

Trimming order (first fit wins):

1. older conversation history (`6 → 4 → 2 → 0` turns);
2. low-relevance memory candidates (`3 → 2 → 1 → 0`);
3. low-relevance candidate schemas (never below the required/pinned schema);
4. replan observations (`8 → 4 → 2 → 0`);
5. if even the minimum prompt exceeds the budget, fail closed with
   `CONTEXT_BUDGET_EXCEEDED` — safety instructions, the user request, typed output constraints and
   selected JSON schemas are never truncated.

A provider context-length error is classified as a request that cannot fit; the brain does not
retry the identical oversized request. The existing provider fallback/health policy is unchanged.

## 10. Before/after context sizes for representative requests

Measured on the real 57-tool registry, terminal source, no conversation history, no memory. Sizes are
characters; token columns are estimates (`chars / 4`), not provider counts.

| Request | Before chars | Before est. tokens | After system chars | After user chars | After total chars | After est. tokens | Candidates | Schemas |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Explain Python decorators | 48,700 | 12,175 | 12,717 | 519 | 13,236 | 3,310 | 0 | 0 |
| Open Notepad | 48,668 | 12,167 | 15,288 | 456 | 15,744 | 3,936 | 5 | 4 |
| Search the web for recent AI research | 48,693 | 12,174 | 15,949 | 481 | 16,430 | 4,109 | 5 | 4 |
| What is my preferred coding language? | 48,692 | 12,173 | 12,717 | 480 | 13,197 | 3,300 | 0 | 0 |
| Open Notepad and write/save a file | 48,711 | 12,178 | 16,100 | 635 | 16,735 | 4,184 | 5 | 4 |

Component sizes: full legacy catalogue 33,959 chars (8,490 est. tokens); compact directory 8,714 chars;
shortlisted full schemas 2,573–3,385 chars; site catalogue 737 chars. Reduction is roughly **73%** in
characters and **64–73%** in estimated tokens for these requests.

## 11. Estimated versus actual token measurements

* **Old prompt actual**: the live B8 acceptance run reported about **11,908 provider-reported input
  tokens** for the oversized prompt. This number came from the provider, not from this session.
* **Old prompt estimate**: 12,167–12,178 tokens for the same five requests.
* **New prompt actual**: **not measured**. No live provider call was made in B9, so no new
  provider-reported count exists.
* **New prompt estimate**: 3,300–4,184 tokens for the same requests. This is an estimate and is
  labelled as such; it is not presented as actual usage.

## 12. Provider and fallback behaviour

* Model selection, provider order, free-only/strict-zero-cost qualification and the composite
  client's health/cooldown/fallback walk are unchanged.
* The brain still uses the existing deterministic route role (`fast` for conversation, `planner` for
  action/reasoning/coding/tool-args); the schema-resolution retry uses `planner`.
* Every real provider attempt still produces exactly one `LLM_CALL` event. B9 adds only safe context
  counts to that event; it does not add a fake event, merge attempts, or skip fallback.
* Context-length failures are not retried identically by the brain; they become the controlled
  budget refusal. Existing provider-level fallback rules are untouched.

## 13. Telemetry and privacy

Added to the existing `llm_call` allow-list (no schema/version migration, existing rows preserved):

`estimated_input_tokens`, `candidate_tools`, `schema_tools`, `memory_candidates`, `history_turns`,
`context_trimmed`.

These are counts/booleans only. Never logged or stored: prompts, model responses, credentials,
conversation text, memory contents, tool arguments/results, browser/OCR text, screenshots or WhatsApp
messages. The safe context log line in `brain` contains only task id, counts, an estimate and a
boolean. Task ids and task classes are the existing opaque identifiers.

## 14. Tests, commands, counts and runtimes

| Command | Result | Wall time |
|---|---|---:|
| `pytest -q tests/unit/test_b9_context_discovery.py` | **26 passed, 0 failed** | 8 s |
| Focused regression: brain prompt, agent architecture, memory B4, taskstore privacy, replan, conversation, notepad flow, routing, schemas, B8 acceptance/capability/contracts, agent, tool resolver, validate paths | **229 passed, 0 failed** | 18 s |
| Security/B8 batch: B8 telemetry + coverage, freeonly, provider, invariants, policy batch/target, browse grant, web_answer, whatsapp, taskstore runtime/telemetry, memory graph/save, fastpath, task runtime | **330 passed, 1 failed** | 22 s |
| `ruff check` on every changed file | **clean** | <1 s |
| `ruff format` on every changed file | **clean** | <1 s |
| `mypy src/jarvis/policy` | **Success, 11 files** | 7 s |
| `ruff check .` | **fails on pre-existing findings outside B9 files** (e.g. `browser/tools.py`, `agent/referents.py`, benchmarks) | — |
| `ruff format --check .` | **68 pre-existing files would be reformatted**; changed files are formatted | — |
| Full repository suite | **not run**, per the B9 instructions | — |

The single security-batch failure is
`tests/unit/test_provider.py::test_registered_vision_model_allows_build`. B9 did not modify
`llm/provider.py`, `llm/capability.py` or `llm/models.py`; the test configures the non-vision Groq
`openai/gpt-oss-120b` as the `vision` role, which B8's `VISION_REQUIREMENT` correctly refuses. It is
recorded as a pre-existing test/spec inconsistency, not as a B9 regression.

## 15. Known limitations

1. **No live verification in this session.** No real provider, desktop application, browser, WhatsApp,
   voice microphone or live preference-recall run was used. The 26 B9 tests are mocked.
2. New token counts are estimates. The old 11,908 figure is the only provider-reported number and
   comes from the earlier live B8 run; the new prompt has no provider-reported count yet.
3. `ModelSpec.context_tokens` is still `None` for every model. The application budget is conservative
   but not a model window; dynamic free-route aliases remain unknown.
4. Candidate discovery is lexical, not semantic. It can miss a relevant tool; the single bounded
   schema-resolution retry handles registered misses, but a second miss becomes a clarification.
5. The compact directory is still about 8.7 KB on every request. It is much smaller than 57 full
   schemas but is not zero-cost.
6. Shortlists can include related-but-extra desktop tools (for Notepad: `open_in_app`, `open_path`,
   `list_windows`). “Open Notepad and write…” may not include `type_text` in the four schemas; the
   model can request it and pay the bounded one-time expansion.
7. The replanner has no `tool_schema` continuation. It is told to prefer a shown schema; an unseen
   registered tool cannot be executed because `validate` rejects it and the replan budget is bounded.
8. The specialist browser path already used only browser tools, so B9 did not need to change it; its
   full browser schema set is still larger than the brain's shortlist.
9. Memory relevance is lexical with a small synonym normalisation, not learned semantics. It fixes the
   observed unrelated-fact leak and preserves the Python preference case, but unusual phrasings may
   still miss a relevant fact.
10. `ruff check .` / `ruff format --check .` are not repository-clean; the findings are pre-existing
    and outside the B9 changed-file set. `mypy src/jarvis/policy` passes.
11. The full repository suite was not run. The one focused failure is the pre-existing provider vision
    test described above.

## 16. New commit hashes

```
8845b95  feat(agent): add compact tool capability index
d213a21  feat(agent): compose bounded task-specific model context
9dc7b57  feat(agent): route brain and replanner through discovery
1a665b5  test(agent): cover B9 context discovery and budgets
```

The report/PROGRESS commit is the docs commit immediately after `1a665b5`; its hash is reported in the
final session message and in `git log -1`.

## 17. What was mocked vs real

| Area | Mocked | Real |
|---|---|---|
| LLM provider | Yes — scripted `BrainDecision`/`ReplanDecision` and context-length failure | No live provider call |
| Tool execution | Yes — fake Tier-0 specs and real canonical registry metadata | No desktop/browser/WhatsApp action |
| PolicyEngine | No — real engine exercised for decisions/blocking | Yes |
| Memory | No — real in-memory SQLite preferences/skills through the real store | Yes (in-memory only) |
| Task checkpoint | No — real SQLite checkpointer in single/multi-step tests | Yes (in-memory) |
| Task-event telemetry | Safe record + allow-list tested directly | No live daemon/task-store run |
| Browser/WhatsApp restrictions | Existing focused suites re-run | No live session |
| Voice | Not exercised | No microphone/TTS |

## 18. Verdict

**B9 COMPLETE WITH LIMITATIONS** — the required architecture is implemented and the focused mocked
acceptance criteria pass: compact discovery, task-specific schemas, one-call conversation/single-tool
paths, bounded planning/execution, memory relevance, schema-miss handling, explicit budgeting, safe
telemetry and unchanged policy authority. The limitations are the absence of live provider/desktop/
browser/WhatsApp verification, the still-conservative lexical discovery, unknown model context
windows, and the pre-existing repository-wide lint/provider-test issues listed above.
