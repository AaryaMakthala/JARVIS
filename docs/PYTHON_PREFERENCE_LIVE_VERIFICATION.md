# JARVIS — Live Python Preference Save, Recall & Restart Verification

Date: 2026-11-11
Environment: Windows, `.venv\Scripts\python.exe` (Python 3.11.15)
Branch: `feature/b1-task-event-store`

**Verdict: PASS** — the preference was saved through the real chat flow, recalled in the same
session, and recalled again in a brand-new conversation after a clean daemon restart.

---

## 1. Starting and ending branch / HEAD

| | |
|---|---|
| Branch (start and end) | `feature/b1-task-event-store` |
| HEAD (start and end) | `506c4e439718678e1aa83704c6dfe35c985ceab4` |
| Code changes | none — no source file was modified |
| Commits | none — this was an operational test |

## 2. Git status

Initial `git status --short`:

```
?? docs/JARVIS_BACKEND_AUDIT.md
```

Final `git status --short`:

```
?? docs/JARVIS_BACKEND_AUDIT.md
?? docs/PYTHON_PREFERENCE_LIVE_VERIFICATION.md
```

The pre-existing audit file was left untouched; this report is deliberately uncommitted.

## 3. Preflight memory state (read-only)

`memory.db` opened with `mode=ro`:

| Check | Before |
|---|---|
| `PRAGMA quick_check` | `ok` |
| skills | 11 |
| failures | 40 |
| preferences | **0** (no rows) |
| SHA-256 of `memory.db` (first 16 hex) | `123f11e39b0a71fd` |

The previous acceptance report's claim of zero preferences was confirmed, not assumed.

## 4. Chat save

Daemon started with:

```
.venv\Scripts\python.exe -m jarvis daemon --foreground
```

Ready line from `jarvis.jsonl`:

```
2026-11-11T08:14:01Z daemon listening on 127.0.0.1:63532 (pid=6868)
2026-11-11T08:14:01Z daemon ready — serving until shutdown is requested
```

Chat session conversation id: `4d9c5248fd394e9394d18eb7efbc372d`.

Exact message submitted:

```
Remember that my preferred coding language is Python.
```

Actual terminal output:

```
you>: started
Noted, sir. I'll remember that.
```

Task `bbe39230-8b1e-4f02-9dc8-7e76f46f95eb`:

| Field | Value |
|---|---|
| status | `answered` |
| duration | 15.7 ms (08:14:27.333 → 08:14:27.349) |
| `request_kind` | `conversation` |
| `memory_saved` | `True` |
| `memory_saved_reason` | `fact_saved` |
| events | `task_created`, `task_started`, `stage_changed` (recalling, planning, responding), `task_finished` |
| `llm_call` events | **none** — the explicit remember command is deterministic and never calls a model |
| log line | `agent event=memory_command task_id=bbe39230… action=remember reason=accepted` |

Stored row (read-only query):

```
key        fact.preference.my_preferred_coding_language_0dff4061f1
value      my preferred coding language is Python
updated_at 2026-11-11T08:14:27+00:00
```

No other preference row exists, and skills/failures stayed at 11/40. The value contains only the
stated preference; no credential-shaped text and no unrelated data were written.

**Scope:** this JARVIS build is single-user — the preferences table has no session or account
column. The row is scoped to the local Windows profile's `memory.db`
(`%LOCALAPPDATA%\jarvis\jarvis\memory.db`) under the `fact.preference.` namespace.

## 5. Same-session recall

Exact question submitted next, in the same chat process:

```
What is my preferred coding language? Answer only from my saved persistent preferences. Do not guess.
```

Actual answer:

```
Your preferred coding language is Python, according to your saved persistent
preferences.
```

Task `16231a09-782f-4942-8a8a-d59924916774`, conversation `4d9c5248fd394e9394d18eb7efbc372d`.

Checkpoint state for that task contained exactly:

```json
"memory_context": [
  {
    "kind": "preference",
    "text": "Remembered fact (preference): my preferred coding language is Python",
    "meta": {
      "key": "fact.preference.my_preferred_coding_language_0dff4061f1",
      "category": "preference",
      "provenance": "user_explicit",
      "fact": true
    }
  }
]
```

Only that one item was retrieved; no unrelated memory was exposed to the prompt.

## 6. Daemon restart

```
.venv\Scripts\python.exe -m jarvis stop
```

Log:

```
2026-11-11T08:16:57Z daemon shut down cleanly
2026-11-11T08:16:57Z daemon: stopped cleanly
```

`daemon.json` (the discovery file) was removed; no `python -m jarvis` process remained. The
zero-byte `daemon.lock` file is a pre-existing persistent lock file (dated 2026-11-06), not a
stale daemon.

Second daemon start:

```
2026-11-11T08:17:22Z daemon listening on 127.0.0.1:57821 (pid=2204)
2026-11-11T08:17:22Z daemon ready — serving until shutdown is requested
```

## 7. Fresh-session recall after restart

A new `jarvis chat` process, therefore a new conversation id:
`65986a30604b4fada8de562fbd0f841b` (different from the first session). Same question:

```
What is my preferred coding language? Answer only from my saved persistent preferences. Do not guess.
```

Actual answer:

```
Your preferred coding language is Python.
```

Task `330a1a64-f646-4351-a70a-fed9630d3739`, served by the new daemon pid 2204.

Its checkpoint `memory_context` again contained the single stored fact:

```json
[{ "kind": "preference",
   "text": "Remembered fact (preference): my preferred coding language is Python",
   "meta": { "key": "fact.preference.my_preferred_coding_language_0dff4061f1",
             "category": "preference", "provenance": "user_explicit", "fact": true } }]
```

This is the decisive evidence that retrieval came from persistent storage: the daemon process and
all in-memory per-conversation history were gone, the conversation id was new, and the fact was
still loaded by `memory_retrieve` from `memory.db` before the model was called.

## 8. Model execution evidence

The recall path is **not** deterministic: the question is not a `remember`/`forget` command, so it
goes through `memory_retrieve → brain` and a real LLM call. Both recall tasks produced exactly two
`llm_call` events — one per provider attempt, no duplicates.

| Task | Attempt | Provider / model | Outcome | Duration | Tokens |
|---|---|---|---|---|---|
| `16231a09…` | 0 | `groq` / `openai/gpt-oss-120b` | error (`failure_category: llm`) | 313 ms | — |
| `16231a09…` | 1 | `openrouter` / `openrouter/free` | success | 6,343 ms | 11,806 prompt / 601 completion |
| `330a1a64…` | 0 | `groq` / `openai/gpt-oss-120b` | error (`failure_category: llm`) | 266 ms | — |
| `330a1a64…` | 1 | `openrouter` / `openrouter/free` | success | 1,983 ms | 11,638 prompt / 218 completion |

Task attribution is correct: every event carries the matching `task_id`, and the route event names
the planner role (`task_class: coding`, `reason_code: coding_keyword`). The event payloads contain
only provider, model, role, outcome, duration and token counts — no prompt text and no response text.

Underlying cause of the Groq errors, from `jarvis.jsonl` for both tasks:

```
HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 400 Bad Request"
model openai/gpt-oss-120b rejected json_schema ...; using JSON mode for the rest of the session
HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 413 Payload Too Large"
provider groq structured output transient failure: context_length; cooling that role for 300s
HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
```

The same prompt-size limitation documented earlier (the ~11.6–11.8k-token tool catalogue) caused
the 413; the existing fallback moved the call to OpenRouter, which answered. This test does not
change or reinterpret the earlier OpenRouter model test — it is separate evidence.

## 9. Database integrity after the test

| Check | Before | After |
|---|---|---|
| readable (`SELECT COUNT(*)`) | yes | yes |
| `PRAGMA quick_check` | `ok` | `ok` |
| skills | 11 | 11 |
| failures | 40 | 40 |
| preferences | 0 | **1** |
| stored value | — | `my preferred coding language is Python` |
| SHA-256 of `memory.db` (first 16) | `123f11e39b0a71fd` | `10700c08d94c8e50` |

The preference count is exactly one higher, as required. The hash change is expected: the write was
in the WAL during the session and was checkpointed into the main file by the clean shutdown. The
`-wal` and `-shm` files were not deleted; the WAL is 0 bytes after shutdown.

After the second `jarvis stop`, the log again shows `daemon shut down cleanly` / `daemon: stopped
cleanly`, `daemon.json` is absent, and a process scan returned no `python -m jarvis` process.

## 10. Ids

| Purpose | Task id | Conversation id |
|---|---|---|
| Remember | `bbe39230-8b1e-4f02-9dc8-7e76f46f95eb` | `4d9c5248fd394e9394d18eb7efbc372d` |
| Recall (same session) | `16231a09-782f-4942-8a8a-d59924916774` | `4d9c5248fd394e9394d18eb7efbc372d` |
| Recall (after restart) | `330a1a64-f646-4351-a70a-fed9630d3739` | `65986a30604b4fada8de562fbd0f841b` |

## 11. Errors, limitations and honest caveats

- Groq rejected both recall prompts with `413 Payload Too Large` after a `400` json_schema
  rejection; the configured fallback to OpenRouter answered. This is a known prompt-size issue,
  not a preference-recall failure.
- The recall answer is produced by an LLM, not by a hard-coded lookup. The model was given the
  stored fact in `memory_context`; the checkpoint proves the retrieval populated before the call,
  but the wording is the model's. The post-restart run is what rules out reliance on chat history.
- In the same session, recent-turn context and retrieved memory both exist; only the restart run
  isolates persistent storage. Both checkpoints nevertheless show the fact in `memory_context`.
- Preferences are global to the local user profile; there is no per-account scoping to verify.
- No source code, schema, skill, failure or unrelated preference was changed; the production
  database was never reset, migrated or written directly.
