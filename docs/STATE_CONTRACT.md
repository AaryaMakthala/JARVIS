# AgentState contract

`AgentState` (in `src/jarvis/agent/state.py`) is the shared, **checkpointed**
state of one task. Every key is optional (`TypedDict, total=False`), and the
whole dict is written to SQLite in plaintext, so the rules are strict:

1. **No secrets.** Never a password, API key, token or raw audio. Passwords are
   verified in the daemon layer, before the graph is resumed.
2. **Plain data only for new keys.** `str` / `int` / `bool` / `list` / `dict`
   of those. The checkpoint serializer allowlist
   (`graph._ALLOWED_MSGPACK_TYPES`) stays unchanged: `Plan`, `Step`,
   `Decision`, `StepResult` and `ReplanDecision` only.
3. **Failures are facts, not opinions.** `halted_reason`, `error` and
   `results` are how a terminal path proves what happened; nothing may claim
   success that these do not support.

Writers/readers below are per module (`agent/nodes/*` unless stated).

## Identity and input

| Key | Writers | Readers | Notes |
|---|---|---|---|
| `task_id` | `runner` (seeds), `intake` (mints if absent) | `brain`, `act`, `clarify`, `policy_gate`, `replan`, `util`, `telemetry`, `compose_rules`, `fastpath`, `daemon`, `memory.tasklog`, `benchmark` | One id per task; used as the `task_log` primary key and the thread id. |
| `source` | `runner`, `intake` | `brain`, `policy_gate`, `replan`, `telemetry` | `terminal` / `voice` / `benchmark`; voice gets the shorter LLM budget. |
| `user_input` | `intake` | `brain`, `fastpath`, `memory_retrieve`, `context_resolve`, `replan`, `util`, `telemetry`, `tool_groups` | Third-party text: truncated and redacted before it is stored in `task_log`. |
| `conversation_id` | `intake`, `act`, `context_resolve` | `context_resolve`, `act` | Keys the per-conversation referent store. |
| `memory_context` | `memory_retrieve` | `brain`, `replan` | Retrieval *examples*, never instructions; capped in the node. |

## Planning (**multi-writer**)

| Key | Writers | Readers | Notes |
|---|---|---|---|
| `plan` | `brain`, `replan`, `validate` (resets/repairs), `fastpath`, `specialist.decide` | `routing`, `validate`, `policy_gate`, `act`, `verify`, `replan`, `respond`, `telemetry`, `memory_save`, `specialist`, `answer`, `runner` | **Multi-writer.** The only allowlisted model carrying steps; `kind` decides conversation vs. tool. `Plan.assumption` is surfaced by `respond` as `Assumption: ...`. |
| `step_index` | `intake` (0), `validate`, `act`, `verify`, `replan` | `policy_gate`, `act`, `verify`, `specialist`, `batch_approval` | Which step the graph is on; reset (not decremented) on replan. |
| `retry_count` | `act`, `verify`, `validate`, `replan` (reset) | `verify` | Per-step retries, capped by `agent.max_retries_per_step`. |
| `replan_count` | `replan` | `verify`, `telemetry` | Capped by `agent.max_replans`. |
| `validate_attempts` | `validate` | `brain`, `replan` | Attempts to obtain a *valid* plan (cap 2, checked in `validate`). |
| `pending_replan` | `replan`, `validate` | `nodes.route_after_validate` | "This plan came from replan"; affects only routing. |
| `replan_now` | `verify`, `act`, `replan`, `validate` | `nodes.route_after_verify` | Ask the replanner for a fresh plan. |
| `retry_now` | `verify`, `act`, `replan`, `validate` | `nodes.route_after_verify` | Retry the same step. |
| `replan_decision` | `replan` | `replan` | Last `ReplanDecision` (allowlisted); kept so the final answer stays honest. |
| `error` | `validate`, `brain` (clears), `replan`, `fastpath` | `nodes.route_after_validate`, `brain`, `runner`, `benchmark` | Repair cue for the next LLM call; `None` means no pending repair. |

## Execution (**multi-writer**)

| Key | Writers | Readers | Notes |
|---|---|---|---|
| `decisions` | `policy_gate`, `validate` (reset), `intake` (reset), `specialist` (reset) | `act`, `batch_approval`, `runner`, `benchmark` | **Multi-writer.** `step_id -> Decision`; the policy engine's output, recomputed rather than trusted from a checkpoint. |
| `approved_hashes` | `policy_gate` (approval), `validate`/`intake` (reset) | `act`, `policy_gate`, `batch_approval` | Confirmation binds to `sha256(tool + canonical(args))`; `act` re-checks before dispatch. |
| `results` | `act` (appends), `verify` (marks verified), `replan`, `validate` (preserves in a specialist iteration) | `respond`, `telemetry`, `replan`, `memory_save`, `specialist`, `compose_rules`, `runner`, `verify` | **Multi-writer.** Append-only evidence; `respond`, `compose` and `telemetry` all read the same list so the answer and the log cannot disagree. |
| `gated_resolved_paths` | `policy_gate`, `act`, `validate` | `act`, `batch_approval` | TOCTOU guard: paths resolved *before* the confirmation interrupt. |
| `gated_content_hashes` | `policy_gate`, `act` | `act`, `policy_gate` | Content hashes for tools with `bind_content_hash`. |
| `halted_reason` | `brain`, `intake`, `validate`, `policy_gate`, `act`, `verify`, `replan`, `specialist`, `clarify` | `routing`, `respond`, `telemetry`, `memory_save`, `compose_rules`, `runner` | The stop signal + the user-facing reason. Never reworded by `compose`. |
| `cancelled` | `intake`, `clarify` | `routing`, `respond`, `telemetry`, `memory_save`, `compose_rules` | The user stopped it. |
| `api_calls` / `tokens` | `brain`, `replan`, `fastpath`, `intake` (reset) | `telemetry`, `benchmark` | Counters only; final by `respond`. |
| `resolved_refs` | `context_resolve` | `brain`, `tool_groups` | Summary of referents (page URL, result positions); untrusted, wrapped in delimiters in prompts. |

## Specialist loop (**multi-writer**)

| Key | Writers | Readers | Notes |
|---|---|---|---|
| `specialist_goal` | `brain` | `specialist`, `routing`, `tool_groups` | Set when a browse request starts the loop. |
| `specialist_calls` | `brain` (reset), `specialist.decide` | `specialist` | Capped by `agent.specialist_max_calls`. |
| `specialist_history` | `brain` (reset), `specialist`, `validate`, `policy_gate` | `specialist` | Bounded list of dicts; page-derived text stays delimited as untrusted. |
| `specialist_phase` | `brain`, `specialist`, `validate`, `policy_gate`, `util` | `nodes.route_after_*`, `routing`, `specialist` | `idle` / `decide` / `observe`; routing reads the phase, never a stale plan. |

## Clarification loop

| Key | Writers | Readers | Notes |
|---|---|---|---|
| `clarification_question` | `brain`, `clarify`, `intake` | `brain`, `respond`, `compose_rules` | The newest question (`Plan.clarification_question` mirrors it). |
| `clarification_answer` | `clarify`, `brain` (clears), `intake` | `brain`, `fastpath`, `tool_groups` | Validated free text from the human. |
| `clarification_count` | `brain`, `clarify`, `intake` | `brain`, `benchmark` | Rounds used (cap `MAX_CLARIFICATIONS = 2` in `clarify.py`, checked in code). |
| `clarification_history` | `brain`, `clarify`, `intake` | `brain` | Every answered round, oldest first, as plain `{"question", "answer"}` dicts. |

## Answers

| Key | Writers | Readers | Notes |
|---|---|---|---|
| `final_answer` | `respond` (and `brain` for a conversation) | `runner`, `daemon` (voice fallback), `cli`, `benchmark`, `voice` | The single sanitized answer (`safe_final_answer`). Unchanged by `compose`. |
| `screen_answer` | `compose` | `respond`, `runner`, `cli` | **New.** The full answer the terminal shows; built from `respond.base_answer`, so it equals the pre-compose text. Plain `str`, nothing new in the serializer. |
| `spoken_answer` | `compose` | `runner`, `daemon/server.py` (voice) | **New.** At most two sentences, cut at a sentence boundary. Failures/refusals are the factual text shortened, never a template. Plain `str`. |

## Memory outcome keys

Written by `memory_save`, which runs after `respond`. They now carry real
values; no node reads them yet (the reader would belong in the runner's
outcome, since `respond` runs *before* `memory_save`), which is why they are
listed here rather than in the answers table.

| Key | Written by | Read by | Values |
|---|---|---|---|
| `memory_saved` | `memory_save` | - | `True` only when a **skill** was stored; a failure record does not count. |
| `memory_saved_reason` | `memory_save` | - | `"name_saved"` when an explicitly stated name was stored as the `user_name` preference; otherwise why nothing was stored (`"cancelled by the user"`, `"task halted: ..."`, `"not a tool plan"`, `"at least one step failed"`, `"dry-run: nothing is written"`, `"memory write failed"`, ...) or `None`. |
| `memory_failure_logged` | `memory_save` | - | `True` when a failure example was appended to the failure log instead of a skill. |

Name learning writes only the preference store: the turn that states the name
is already answered, so the name applies from the **next** turn (`compose`
reads `user_name` before `memory_save` runs). See `docs/MEMORY.md`.

## What may never be added here

- Passwords, API keys, tokens, raw audio, full prompts, or provider payloads.
- New serialized *types* (the allowlist in `graph.py` is deliberately closed).
- Policy decisions as a source of truth: `decisions` is a cache of what the
  engine computed for this exact state; `act` and `policy_gate` recompute.
- Anything the LLM wrote verbatim into a navigational field (`tier`,
  `allowed`, `action_hash`) - those come from code.
