# Graph reference

The compiled graph (`src/jarvis/agent/graph.py`, `build_graph`) is a LangGraph
`StateGraph` over `AgentState`. **The LLM proposes; the deterministic code
routes.** No router asks a model where to go, and every router is a pure
function of state.

## Nodes

| Node | File | Deterministic? | Purpose |
|---|---|---|---|
| `intake` | `nodes/intake.py` | yes | Validate/normalise the input, seed `task_id`/`conversation_id`, open the `task_log` row, reset per-task keys. |
| `context_resolve` | `nodes/context_resolve.py` | yes | Resolve deictic references ("the first result", "that photo") into `resolved_refs`. No LLM. |
| `memory_retrieve` | `nodes/memory_retrieve.py` | yes | Retrieve bounded skills/failures/preferences as *examples* (`memory_context`). |
| `brain` | `nodes/brain.py` | **LLM** | The single classification/planning call; projects the transient `BrainDecision` onto `Plan`/`Step`. |
| `clarify` | `nodes/clarify.py` | yes | Interrupt and ask the human one question (round budget in code). |
| `validate` | `nodes/validate.py` | yes | Step cap, tool existence, per-tool Pydantic args, availability of the plan's tools. One repair attempt, then halt. |
| `policy_gate` | `nodes/policy_gate.py` | yes | The only authorization point: tiers, Tier 3 block, path resolution, target binding, confirmations (interrupts), batch approval. |
| `act` | `nodes/act.py` | yes | Re-check the gate's decision and hash, then execute exactly one step. |
| `verify` | `nodes/verify.py` | yes | Post-condition check via the tool's `verify()`; decides retry / replan / next step. |
| `replan` | `nodes/replan.py` | **LLM** | One bounded replan call when a step failed and the budget allows. |
| `specialist_decide` | `nodes/specialist.py` | **LLM** | Flagged (`agent.use_specialists`): one browser step per call inside the browse loop. |
| `specialist_observe` | `nodes/specialist.py` | yes | Flagged: read the page / record the outcome in `specialist_history`. |
| `compose` | `nodes/compose.py` | yes | Write `spoken_answer` (≤2 sentences) and `screen_answer` for the outcome. No LLM, no tools. |
| `respond` | `nodes/respond.py` | yes | The single sanitizer + task-log writer: `safe_final_answer`, `record_finished`. |
| `memory_save` | `nodes/memory_save.py` | yes | Exactly one memory write attempt per task (verified plan or failure). |

## Edges

Unconditional:

| From | To |
|---|---|
| `START` | `intake` |
| `context_resolve` | `memory_retrieve` |
| `memory_retrieve` | `brain` |
| `compose` | `respond` |
| `respond` | `memory_save` |
| `memory_save` | `END` |

Conditional (the map's **right-hand side** is the node; routers still return the
route keys shown on the left, so `"respond"` is translated to `"compose"`):

| From | Router | Route keys -> nodes |
|---|---|---|
| `intake` | `route_after_intake` | `memory_retrieve` -> `context_resolve`; `respond` -> `compose` |
| `brain` | `route_after_brain` | `clarify` -> `clarify`; `validate` -> `validate`; `respond` -> `compose`; `specialist_decide` -> `specialist_decide` *(flag)* |
| `clarify` | `route_after_clarify` | `brain` -> `brain`; `respond` -> `compose` |
| `validate` | `route_after_validate` | `brain` -> `brain`; `replan` -> `replan`; `policy_gate` -> `policy_gate`; `respond` -> `compose`; `specialist_decide` *(flag)* |
| `policy_gate` | `route_after_policy_gate` | `act` -> `act`; `respond` -> `compose`; `specialist_decide` *(flag)* |
| `act` | `route_after_act` | `verify` -> `verify`; `respond` -> `compose` |
| `verify` | `route_after_verify` | `act` -> `act`; `replan` -> `replan`; `policy_gate` -> `policy_gate`; `respond` -> `compose`; `specialist_observe` *(flag)* |
| `replan` | `route_after_replan` | `validate` -> `validate`; `respond` -> `compose` |
| `specialist_decide` *(flag)* | `route_after_specialist_decide` | `validate` -> `validate`; `respond` -> `compose` |
| `specialist_observe` *(flag)* | `route_after_specialist_observe` | `act` -> `act`; `decide` -> `specialist_decide` |

*(flag)* = edge and node exist only when `agent.use_specialists = true`.

## Brain route table (`agent/routing.py`)

`route_for_brain_result(state)` is the single source of truth; the docstring
table is repeated here.

| `request_type` | Required fields (validated on `BrainDecision`) | Next node |
|---|---|---|
| `conversation` | `response_text` non-blank, `actions` empty | `respond` (via `compose`) |
| `action` | at least one action | `validate` |
| `clarification` | `clarification_question` non-blank, `actions` empty | `clarify` |
| `browse` | `goal` non-blank, `actions` empty | `specialist_decide` (or `respond` when the specialist flag is off) |
| `unsupported` | `actions` empty; answer = the brain's text, else "I can't do that with the tools I have." | `respond` (via `compose`) |

Precedence before the table (first match wins): `halted_reason` -> `respond`;
`request_kind == "unsupported"` -> `respond`; specialist waiting to decide ->
`specialist_decide`; no plan -> `respond`; `plan.needs_clarification` ->
`clarify`; `plan.kind == "conversation"` -> `respond`; otherwise `validate`.

A `BrainDecision` that breaks the table (e.g. `conversation` with actions) is
rejected by its Pydantic validator, which the LLM clients answer with their
existing single repair round - never a crash.

## Loop caps

Every loop is bounded in code. "Not configurable" is deliberate: a value that
needs tuning is a config key, a safety bound is a constant.

| Loop | Cap | Config key / constant | Enforced in |
|---|---|---|---|
| Plan length | 12 steps | `agent.max_steps` | `validate` |
| Step retries | 2 per step | `agent.max_retries_per_step` | `verify` |
| Replans | 2 per task | `agent.max_replans` | `replan`, `verify` |
| Plan repair attempts | 2 | constant `2` | `validate` |
| Clarification rounds | 2 | `MAX_CLARIFICATIONS` | `clarify`, `brain` |
| Specialist calls | 12 | `agent.specialist_max_calls` | `specialist.decide` |
| Specialist call timeout | 12.0 s | `agent.specialist_call_timeout_s` | `specialist.decide` |
| LLM retries per provider | 1 | `llm.max_retries` | `llm/client.py` |
| LLM per-call timeout | 30.0 s | `llm.timeout_seconds` | `llm/client.py` |
| LLM overall deadline (whole fallback walk) | 30.0 s | `llm.total_deadline_seconds` | `llm/multi.py` |
| LLM overall deadline, voice path | 12.0 s | `llm.voice_deadline_seconds` | `llm/multi.py` |
| Provider walk | one pass over eligible candidates | `llm.provider_order`, `llm.free_only`, `llm.strict_zero_cost` | `llm/multi.py` |
| Browsing grant | 30 actions / 600 s | `browser.task_grant_max_actions`, `browser.task_grant_ttl_s` | `policy/browse_grant.py` |
| Memory caps | 500 skills / 200 failures / 50 prefs / 500 task rows | `memory.*` | `memory/*` |
| Step wall-clock timeout | 30 s | `agent.step_timeout_seconds` | **declared, never read** (no node enforces it - see `PROGRESS.md`) |

## Serializer

The checkpointer (`build_secure_serde`) allowlists exactly `Plan`, `Step`,
`Decision`, `StepResult` and `ReplanDecision`, with no pickle fallback. New
state is plain data (`str`/`int`/`bool`/`list`/`dict`), so the allowlist has not
changed for prompt trimming, the node trace or the composed answers.

## Prompt shaping (not routing)

`agent/tool_groups.py` chooses which tools/sites the brain prompt *describes*
when `agent.prompt_trim_enabled = true` (default false). It cannot change a
route, a tier or a decision: the registry stays closed and `validate` /
`policy_gate` / `act` judge every step as before. An unassigned tool or a
request that matches no group sends the full catalogue (fail open).
