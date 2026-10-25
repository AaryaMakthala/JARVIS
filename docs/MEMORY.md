# Memory design

One SQLite file (`jarvis.config.memory_db()`, normally
`%LOCALAPPDATA%/JARVIS/memory.db`) holds everything JARVIS remembers. There is
no second store and no cache: the CLI, the graph and the daemon all go through
`jarvis.memory.store.SqliteMemory`.

Memory is **advisory**. A stored example is text handed to the planner; the
planner must still produce a plan that passes `validate` and `policy_gate`, and
nothing stored here can be executed, replayed or approved automatically.

## What is stored

| Table | Columns (essentials) | Written by | Cap |
|---|---|---|---|
| `skills` | `goal_text`, `plan_json`, `embedding`, `tools_used`, `success_count`, `fail_count`, `created_at`, `last_used_at` | `memory_save` → `SqliteMemory.remember` → `SkillStore.save` | `memory.max_skills` 500 |
| `failures` | `goal_text`, `step_json`, `error` (≤300 chars), `created_at`, `step_state` (`ok`/`verified`/`tainted`), `quarantined` | `memory_save` → `FailureStore.record` | `memory.max_failures` 200 |
| `preferences` | `key` (≤64), `value` (≤400), `updated_at` | `SqliteMemory.remember(kind="preference")`, `memory_save` (name learning) | `memory.max_preferences` 50 |
| `task_log` | counters, status, redacted `user_input`, node path | `telemetry` (`intake` / `respond`) | `memory.max_task_log` 500 |

`task_log` is telemetry for the benchmark, not learned memory: `jarvis memory
clear` deliberately leaves it alone.

## Retrieval

`memory_retrieve` asks `SqliteMemory.retrieve(query, limit)` once per task;
the backend scores every source against the user's text with the deterministic
offline lexical embedding (`jarvis.memory.embeddings`):

* skills: cosine ≥ `memory.similarity_threshold` (0.75), best first;
* preferences: all of them, as stable context, until the limit is reached;
* failures: cosine ≥ `memory.failure_similarity_threshold` (0.45) - lower on
  purpose, because missing a real failure is worse than showing a loose one and
  unrelated goals score below 0.2.

Hard caps: `memory.retrieval_limit` (3) records per task; each item is truncated
to 4000 chars by `nodes/util.truncate` before it enters state.

The records are injected into the brain prompt as `memory_context` (a
`list[dict]` of plain primitives: `kind`, `text`, optional `meta`), serialised
as JSON under "Relevant past examples". They are reference data only - the
brain system prompt treats stored text as untrusted, never as instructions.

**Failure behaviour:** any exception during retrieval is swallowed, logged as
one line without content, and the task continues with an empty
`memory_context` (fail **open** for retrieval). There is no timeout today: a
slow SQLite call would block the task, and the retrieval is small enough that
this has not been hit - a bounded timeout is tracked as a follow-up, not
implemented.

## Save policy (`memory_save`, after `respond`)

One record per task, decided in code:

A **skill** is stored only when *all* hold: not cancelled, no
`halted_reason`, `plan.kind == "tool"` with at least one step, results match the
steps one-to-one, every result `ok`, no result `verified is False`, no result
`verified is None` (unverifiable is not a failure but is not proof either), no
tainted result, no step `depends_on_untrusted`, and `memory.save_skills` on.
Text stored: the plan goal; args pass through `safe_value`; the row is refused
if it still looks like it carries a credential.

A **failure** is stored for the first *bad* step (failed, failed-verification or
tainted) of a task that could not become a skill. Text stored: the step error,
redacted to 300 chars; a tainted step's output is dropped. `FailureStore`
additionally refuses any row whose provenance says the step actually succeeded.
No page text, tool output, file contents or plan args are stored for failures.

Dedupe: `SkillStore.save` folds a near-identical goal into the existing row
(incrementing `success_count`, updating `last_used_at`); failures append.
Retention: each save prunes its own table back to the cap, oldest/least-used
rows first, in the same call.

The node never raises and never blocks the answer; a memory error is a one-line
warning. It sets the three state keys (`memory_saved`,
`memory_saved_reason`, `memory_failure_logged`) - see
`docs/STATE_CONTRACT.md`.

## Preferences and the user's name

Preference keys are validated on write: lower-case `[a-z0-9][a-z0-9_.-]{0,63}`,
values ≤400 chars, a credential-looking key *or* value is refused.

`memory_save` learns a name from an explicit statement in the user's own words -
`"my name is X"`, `"call me X"`, `"you can call me X"`, `"I'm called X"` - with
no LLM involved. The rules: one or two words of Unicode letters, apostrophes and
hyphens, 2-40 characters; digits, URLs, stop words, command verbs and
credential-ish words are refused; the value must pass the secret check. It only
runs on a plain conversation turn (no tool steps, no halt, not cancelled) and
only when `memory.learn_name` is on.

The name is stored as the `user_name` preference through the *existing*
`PreferenceStore`. An existing name is replaced only by the explicit "my name is"
/ "call me" phrasings; "you can call me" / "I'm called" only fill an empty slot.
`memory_saved_reason` becomes `"name_saved"` when a name is written.

**The name takes effect from the next turn**: `memory_save` runs after
`respond`, and `compose` reads `user_name` before it, so the turn that teaches
the name is answered without it. There is deliberately no write path for the
name other than this one statement detector (and the raw `preferences` table).

## CLI (`jarvis memory ...`)

| Command | Effect |
|---|---|
| `jarvis memory list [--limit N]` | Up to 50 rows: kind, id, date, 60-char text. Failures held back by the unverifiable-success fix are marked `!`. |
| `jarvis memory forget <id> [--kind auto\|skill\|failure]` | Deletes one row. `auto` refuses an id that exists in both tables. |
| `jarvis memory clear [CLEAR]` | Deletes every skill, failure and preference (asks first, or requires the word `CLEAR`). `task_log` is kept. |
| `jarvis memory prefs` | Preference keys; values truncated to 40 chars, credential-looking ones shown as `[hidden]`. |

`jarvis skills list|show|delete|clear` remains the richer view of learned plans.

## Configuration

| Key | Default | Effect |
|---|---|---|
| `memory.enabled` | `true` | Off = `NullMemory`: nothing read, nothing written. |
| `memory.retrieval_limit` | `3` | Records handed to the planner per task. |
| `memory.save_skills` | `true` | Off = still read examples, never record new ones. |
| `memory.learn_name` | `true` | Learn an explicitly stated name into `user_name`. |
| `memory.similarity_threshold` | `0.75` | Cosine floor for a skill to count as relevant. |
| `memory.failure_similarity_threshold` | `0.45` | Same floor for remembered failures. |
| `memory.max_skills` | `500` | Skill rows retained. |
| `memory.max_failures` | `200` | Failure rows retained. |
| `memory.max_preferences` | `50` | Preference keys retained. |
| `memory.max_task_log` | `500` | Task-log rows retained. |

## Not implemented yet (planned, do not assume)

Bounded retrieval hardening is **not** in the code today: per-item char cap
(300), total prompt cap (1200), `memory.min_score`, dedupe of near-identical
items, excluding secret-looking records and stale failures (older than 30 days)
at read time, and a hard retrieval timeout. `retrieval_limit` is the only
retrieval cap; the item truncation is 4000 chars from `nodes/util.truncate`.
The save side likewise has no `memory.max_records` /
`memory.max_failure_records` keys - retention uses the `max_*` keys above.
