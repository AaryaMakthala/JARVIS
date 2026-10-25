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

Bounded retrieval (`memory_retrieve`, M2):

* `memory.retrieval_limit` (3) records per task - the hard count cap;
* each item's text is cut to `memory.item_max_chars` (300) and the whole
  injected block to `memory.context_max_chars` (1200), so a single prompt can
  never be flooded by a large plan or goal;
* an item near-identical to one already kept is dropped (normalised-text
  comparison plus the lexical embedding's cosine), so three paraphrases of the
  same skill cost one slot, not three;
* an item that trips the shared secret check (`memory.db.looks_secret`) is
  skipped, and the fact is logged without the text;
* a failure example older than `memory.failure_max_age_days` (30) is skipped -
  stale advice about a world that has moved on. Preferences and skills are not
  age-filtered. `0` disables the age filter.

The store call runs in a worker thread and is given
`memory.retrieve_timeout_seconds` (2.0) via `future.result(timeout)`; on
timeout the task continues immediately. The SQLite connection is opened per
`SqliteMemory` and only ever touched by one thread at a time (`check_same_thread`
stays on, `_lock` covers writes), so the worker borrows the store, does one read
and returns - it never writes and is never shared with a second caller.

The records are injected into the brain prompt as `memory_context` (a
`list[dict]` of plain primitives: `kind`, `text`, optional `meta`), serialised
as JSON inside a delimited block (see `agent/prompts.py`) that is labelled
"past tasks, reference only": never instructions, never replayed, and anything
inside it is untrusted data.

**Failure behaviour:** any exception during retrieval, and a timeout, are
swallowed, logged as one line without content, and the task continues with an
empty `memory_context` (fail **open** for retrieval).

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
tainted) of a task that could not become a skill, and only once the task is
terminal (halted, replan budget spent, or no retry/replan pending) - a retry
still in flight is never archived as a dead end. Text stored: the **tool name
plus a short category** (`tool_failed`, `verify_failed`, `untrusted_output`,
`unknown`), never the tool's own error string, page text, tool output, file
contents or plan args. `FailureStore` additionally refuses any row whose
provenance says the step actually succeeded, and redacts a goal that carries a
credential.

Dedupe: `SkillStore.save` folds a near-identical goal into the existing row
(incrementing `success_count`, updating `last_used_at`); a failure with the
same goal, tool and category refreshes the existing row's `created_at` instead
of inserting a duplicate (`failures` has no repeat counter - that would need a
schema migration, which this pass does not do - so "how many times" is not
recorded).

Retention: each save prunes its own table back to the cap in the same call,
oldest/least-used rows first, deleting **at most 50 rows per call**
(`memory.db.MAX_PRUNE_DELETIONS`). Lowering a cap therefore takes a few saves
instead of one long transaction, and a save can never stall on a table that
grew while the cap was smaller.

The node never raises and never blocks the answer; a memory error is a one-line
warning. Every return path sets all three state keys, and
`memory_saved_reason` is one of the codes in `memory_save.REASONS`:

| `memory_saved_reason` | Meaning |
|---|---|
| `saved` | A skill was stored (`memory_saved` is true). |
| `skipped_cancelled` | The user cancelled the task. |
| `skipped_halted` | The task halted (`halted_reason` set). |
| `skipped_unverified` | Not provably good: a step failed, failed verification, was unverifiable, or was tainted. |
| `skipped_sensitive` | The turn carried a credential-looking value; nothing was stored at all. |
| `skipped_conversation` | Not a tool plan (no steps to remember). |
| `name_saved` | A stated name was written to `user_name` (see below). |
| `error` | Nothing recorded for a reason outside the task: no backend/disabled memory, `--dry-run`, `save_skills` off, or a write that raised. |

`memory_failure_logged` is true only when a failure row was written. See
`docs/STATE_CONTRACT.md` for the state keys themselves.

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
| `memory.item_max_chars` | `300` | Char cap for one retrieved item. |
| `memory.context_max_chars` | `1200` | Char cap for the whole `memory_context` block. |
| `memory.failure_max_age_days` | `30` | Failure examples older than this are not retrieved (`0` = no age filter). |
| `memory.retrieve_timeout_seconds` | `2.0` | Hard timeout for one retrieval call; on expiry the task continues with no memory. |

Old config files keep loading: the four keys above are new fields with
defaults, so a `config.toml` written before this pass is still valid.

## Known limits (deliberate, not bugs)

* A failure row records *that* the approach failed, not how many times: no
  repeat counter (would be a migration).
* Pruning is bounded, so a table can sit above its cap for a few saves after
  `max_*` is lowered.
* The name takes effect from the next turn (see above).
* Retrieval scores with the offline lexical encoder, not a semantic model:
  synonyms that share no words can be missed.
