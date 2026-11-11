# B8 — Intelligent Backend Implementation Report

Date: 2026-11-11 · Branch: `feature/b1-task-event-store`

## A. Repository state

| Item | Value |
|---|---|
| Branch | `feature/b1-task-event-store` |
| Starting HEAD | `19f47b51d06ef9eac8b8364106f8aa767162a00a` |
| Starting status | clean except untracked `docs/JARVIS_BACKEND_AUDIT.md` |
| Final HEAD | last commit on this branch (implementation commits `1ff173e`, `fc0d840`, `79857f7`, then the docs commit) |
| Working tree after | `docs/JARVIS_BACKEND_AUDIT.md` still untracked and untouched |
| Pushed | **nothing** was pushed; no reset/rebase/amend/force-push; staging used explicit paths only |

The requested starting HEAD matched the actual repository, so no adaptation was needed.

## B. Commit history

| Hash | Message | Files changed | Purpose |
|---|---|---|---|
| `1ff173e` | `feat(llm): add capability-aware model selection` | `src/jarvis/llm/capability.py` (new), `src/jarvis/llm/models.py`, `src/jarvis/llm/provider.py`, `tests/unit/test_b8_capability.py` (new) | Requirement-driven, fail-closed capability gate used by the factory and the composite client |
| `fc0d840` | `feat(taskstore): persist real LLM attempt telemetry` | `src/jarvis/llm/telemetry.py` (new), `src/jarvis/llm/multi.py`, `src/jarvis/agent/nodes/telemetry.py`, `src/jarvis/agent/nodes/brain.py`, `src/jarvis/taskstore/privacy.py`, `src/jarvis/taskstore/vocabulary.py`, `tests/unit/test_b8_llm_telemetry.py` (new) | Gives `TaskEventType.LLM_CALL` real producers with safe, per-request metadata |
| `79857f7` | `test(backend): add B8 end-to-end acceptance coverage` | `tests/unit/test_b8_acceptance.py` (new) | Real graph-path acceptance for routes A/B/C, registry refusal, per-task attempt isolation |
| docs commit | `docs(backend): document completed B8 architecture` | `docs/B8_IMPLEMENTATION_REPORT.md`, `PROGRESS.md` | This report plus session status |

Existing components were **extended, not rebuilt**: LangGraph topology, Plan/Step/BrainDecision
schemas, policy engine, tool registry, fast paths, memory, browser and WhatsApp code are unchanged.

## C. Implementation status

| Stage | Implemented | Unit tested | Integration tested | Live tested | Remaining |
|---|---|---|---|---|---|
| B8.1 capability-aware selection | Yes | Yes (11 tests) | Yes (graph runs with the composite client) | No | Live provider capability probing |
| B8.2 planner / bounded step execution | Already existed; unchanged | Existing suites (subset re-run) | Partly (acceptance Route C) | No | Not extended in B8 |
| B8.3 task-specific tool discovery | Already existed (`agent/tool_groups.py`, opt-in) | Existing suites (not re-run) | No new coverage | No | Not exercised end-to-end this session |
| B8.4 bounded context composition | Already existed (prompt trim, history, memory blocks) | Existing suites (not re-run) | No new coverage | No | No new composer added |
| B8.5 LLM telemetry producers | Yes, for the **brain** call | Yes (9 tests) | Yes (graph → task store) | No | `replan`, `specialist`, `web_answer`, WhatsApp summariser call sites not yet instrumented |
| B8.6 grounded browser execution | Already existed; unchanged | Existing browser suites (not re-run) | No new coverage | No | Live browser verification |
| B8.7 memory integration | Already existed (B4); unchanged | Existing memory suites (not re-run) | No new coverage | No | Nothing added in B8 |
| B8.8 acceptance testing | Partly (see I) | Yes | Yes | No | Live provider smoke test, browser fixtures |

## D. Model routing

* **Task classification** — `src/jarvis/llm/routing.py::route_request` is deterministic (closed word
  sets + prefixes). It returns `RouteDecision(task_class, model_role, reason_code)`, e.g.
  `conversation → fast`, `action/reasoning/coding/tool_args → planner`, `local_ocr → planner`.
  `brain.py` records the decision with `record_route(...)` (closed vocabulary).
* **Model roles** — exactly three: `fast`, `planner`, `vision` (`settings.llm.models.<provider>.<role>`).
  There is no separate summarisation or OCR role: local OCR never leaves the machine and
  summarisation is served by the planner requirement.
* **Capability checks** — `src/jarvis/llm/capability.py`:
  `CONVERSATION_REQUIREMENT` (structured output), `PLANNER_REQUIREMENT` (structured + tool planning),
  `VISION_REQUIREMENT` (vision). `satisfies()` reads only the registry's `Capabilities`; an
  unregistered model (`pricing_mode == "unknown"`) is *unknown capability* and is refused. An unknown
  context window (`ModelSpec.context_tokens is None`) only fails when a minimum is demanded.
  `provider.py::_guard_capability` enforces this before any client is constructed;
  `multi.py` re-checks per role and skips a model that cannot serve the operation.
* **Provider selection** — `provider.py::build_llm_client` walks `settings.llm.provider_order`
  (default `groq, openrouter, nvidia, gemini`), one client per configured key slot that has a
  credential, and keeps the configured order. Health ordering inside the composite client is
  preference-only and finite.
* **Free-tier restriction** — `llm/models.qualify()` still decides eligibility from
  `free_only` / `strict_zero_cost`; ineligible providers never enter `MultiProviderClient`, so no
  fallback can reach a paid model. B8 did not change this matrix.
* **Fallback** — `MultiProviderClient` walks provider groups in health order, each group's key slots
  in slot order. Failure categories decide whether a retry/rotation can help
  (`llm/failures.py`): rate limit cools the (provider, slot, role) triple and moves to the next
  provider (no key rotation by default), auth/server/transport rotate keys, model-scoped failures
  cool the provider-wide pseudo-slot 0. The walk is bounded by candidate count and by one overall
  deadline.
* **LLM calls per request** — deterministic routes: **0**. Simple conversation: **1** (`fast` role,
  verified by acceptance test). Single-tool request: **1** brain call, no extra executor call.
  A complex task can use one planner call plus bounded replans; the existing per-node budgets
  (validate attempts, retries, replan count) are unchanged and cannot be raised by an LLM.
* **Sanitised example** — request "hello there" → class `conversation`, role `fast`, provider `groq`,
  model `openai/gpt-oss-20b`; stored event: `{provider: groq, model: <planner label actually used>,
  role: fast, outcome: success, fallback_index: 0, task_class: conversation, prompt_tokens: 5}`.

## E. Planner and executor

Existing design, unchanged: one `BrainDecision` → projected `Plan(kind, goal, steps=[Step(id, tool,
args, rationale, expect, depends_on_untrusted)])` → `validate` (registry membership + Pydantic args
model) → `prepare_readback` → `policy_gate` (tier, confirmation interrupt, action hash, target and
content-hash binding) → `act` → `verify` → `replan` or `respond`.

Structured plan example (shape produced by `brain._project`):

```json
{"kind": "tool", "goal": "open notepad",
 "steps": [{"id": "s1", "tool": "open_app", "args": {"name": "notepad"},
            "rationale": "user asked to open notepad", "expect": "Notepad window is visible"}]}
```

There is no arbitrary code/command path: an unregistered tool name is refused by `validate`, and
`tests/unit/test_b8_acceptance.py` proves the requested `made_up_tool` never executes. Verification
is by real tool post-conditions (`verify()` per tool), not by the model's own claim; replanning is
bounded by the pre-existing counters.

## F. Browser automation

Unchanged in B8 (existing implementation already follows observe → ground → authorize → act → verify):
`browser_read` returns a bounded accessibility observation with transient `e1..eN` handles;
`browser_click`/`browser_type` re-validate element identity immediately before acting and refuse
stale, ambiguous or invisible targets; navigation refuses non-canonical URLs and cross-origin
redirects; domain grants, origin checks, policy tiers, confirmations, action hashes and the
`browser_final_click` readback remain the only authorization path. Verification is a fresh
observation after the action, never the model's statement.

Evidence: no new browser code or tests were written in this session, and **no live browser
verification was performed**. The pre-existing browser suites were not re-run.

## G. Databases

| Database | Owner | Purpose |
|---|---|---|
| `checkpoints.db` | LangGraph `SqliteSaver` with the strict allow-listed serde | graph state |
| `task_events.db` | `jarvis.taskstore` (`TaskStore`) | lifecycle, stages, routes, **LLM attempts**, tools |
| `memory.db` | `jarvis.memory` | explicit facts/preferences |

No new database and **no schema migration** was needed: `llm_call` already existed as an event type
and only its allow-listed detail keys were extended, so existing databases remain valid and
versioned as before (`taskstore/schema.py` untouched).

Stored per attempt (allow-list in `taskstore/privacy.py`): provider, model, role, task class,
attempt id (UUID), fallback index, outcome (`success|error|cancelled|timeout`), duration_ms,
prompt/completion tokens when reported, failure category (`llm|timeout|cancelled`), cancelled and
timed-out flags. Prompts, responses, tool arguments, file paths, screenshots, OCR text, message
bodies and credentials are **not** representable: unknown detail keys are dropped (non-strict) or
rejected (strict); a dedicated test asserts a payload containing `prompt`, `tool_args` and
`response` reduces to `{provider, model, role}`.

Restart recovery is unchanged (`fix(taskstore): repair daemon startup reconciliation`); attempt
events are append-only and never replayed.

## H. Memory

Existing B4 implementation, unchanged: explicit remember/forget commands are parsed deterministically
in `brain._memory_command` (secrets and oversized text are refused, dry-run stores nothing), writes
happen only in `memory_save`, retrieval happens in `memory_retrieve`, provenance is preserved, and
unrelated preferences are not injected into every prompt. Conversation separation and daemon-restart
persistence come from the SQLite store, not from graph state.

No new memory code or tests were added in B8; the existing memory acceptance suite was not re-run in
this session (see limitations).

## I. Verification

Exact commands run (Git Bash, repo root, `.venv/Scripts/python.exe`, Python 3.11). Per-command wall
time was not instrumented; each command completed inside the 180 s budget.

| Command | Passed | Failed | Skipped |
|---|---|---|---|
| `python -m pytest tests/unit/test_b8_capability.py tests/unit/test_b8_llm_telemetry.py tests/unit/test_b8_acceptance.py -q` | 32 | 0 | 0 |
| `python -m pytest tests/unit/test_llm_models.py tests/unit/test_llm_bounded.py tests/unit/test_llm_rotation.py tests/unit/test_llm_health.py tests/unit/test_llm_client.py tests/unit/test_taskstore_privacy.py tests/unit/test_agent.py tests/unit/test_brain_prompt.py tests/unit/test_agent_architecture.py -q` | 171 | 0 | 0 |
| `python -m ruff check <changed files>` | — | 0 | — |
| `python -m ruff format --check <changed files>` → `ruff format` → re-check | — | 0 | — |

Ruff: `All checks passed!` after formatting. `mypy src/jarvis/policy` was not run because no policy
file was touched.

**Pre-existing failure, deliberately not "fixed"**: `tests/unit/test_freeonly.py::
test_langgraph_uses_four_provider_multi_client` fails identically on the unmodified tree (verified by
stashing this session's changes and re-running it): the arithmetic fast path answers
"What is 2 plus 2?" with `4` before the LLM is consulted, while the test still expects the scripted
provider's `"Four."`. That is a stale test premise outside B8's scope; changing it would alter
accepted fast-path behaviour.

## J. Known limitations

**Not implemented in this session**

* `LLM_CALL` producers exist for the brain call only. `replan`, `specialist`, `web_answer` and the
  WhatsApp summariser still call the client without `attempt_meta`, so their attempts are not stored.
* No prompt-budget composer was added; context composition remains the existing prompt-trim,
  conversation-history and memory-block implementation.
* No new tool-shortlisting logic was added; `agent/tool_groups.py` is unchanged and still opt-in via
  `agent.prompt_trim_enabled`.
* No browser-recovery changes; the existing bounded recovery is untouched.

**Implemented but not tested live**

* Capability selection, free-only enforcement and fallback are unit/integration tested with scripted
  clients; no real provider was called (no API key was used, no live smoke test run).
* Browser, WhatsApp, voice and memory paths are mock/unit tested by their pre-existing suites only;
  those suites were not re-run in this session.

**Verified by mocks/fixtures only**

* Everything under section I (fake provider, temporary SQLite databases, no network).

**Known pre-existing issue**

* `test_freeonly.py::test_langgraph_uses_four_provider_multi_client` (stale arithmetic assumption).

## K. Next recommended milestone

Instrument the remaining real call sites (`replan.py`, `specialist.py`, `tools/web.py`,
`tools/whatsapp_web.py`) with the same `llm_attempt_meta(...)` helper so the whole task timeline is
complete, then run one tiny live smoke test against a configured zero-cost provider to confirm the
selection/fallback decisions and stored events match what the provider actually did.
