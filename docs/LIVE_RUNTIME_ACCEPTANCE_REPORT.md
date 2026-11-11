# JARVIS — Live Backend, Database, Memory & Model Acceptance Report

Date: 2026-11-11
Host: Windows, repository `C:\Users\aarya\OneDrive\Desktop\Jarvis`
Interpreter: `.venv\Scripts\python.exe` (CPython 3.11.15, 64-bit)
Scope: runtime acceptance test only — no architecture change, no development work.

---

## A. Repository state

| Item | Value |
|------|-------|
| Branch | `feature/b1-task-event-store` |
| HEAD (initial) | `f7b4ee72ba653f2f4d97ede6c3ff9f59bac26fff` |
| HEAD (final) | `f7b4ee72ba653f2f4d97ede6c3ff9f59bac26fff` (unchanged) |
| `git status --short` (initial) | `?? docs/JARVIS_BACKEND_AUDIT.md` |
| `git status --short` (final) | `?? docs/JARVIS_BACKEND_AUDIT.md` (plus this new report, untracked) |

* No reset, clean, rebase, amend, history rewrite or push was performed.
* `docs/JARVIS_BACKEND_AUDIT.md` was never opened for writing or modified: it is untracked both before
  and after the test, timestamp `Nov 6 19:33`, size 53,531 bytes.
* No source file was edited during the acceptance test.
* No runtime log, temporary database or test artifact was written inside the repository. Daemon
  stdout/stderr and scratch output were captured under the system temp directory
  (`%LOCALAPPDATA%\Temp`), outside the working tree.

---

## B. Startup, chat connection and shutdown

### Exact commands

```powershell
.\.venv\Scripts\python.exe -m jarvis daemon --foreground    # Terminal A
.\.venv\Scripts\python.exe -m jarvis chat                   # Terminal B
```

The interactive REPL was driven by piping each prompt on stdin (the CLI reads `Prompt.ask`); EOF
after the line produced the normal `bye` exit. This is the real daemon-backed chat path
(`_chat_with_daemon`), not `--no-daemon` and not `jarvis run --dry-run`.

### Startup result (first instance)

Configuration loaded from `C:\Users\aarya\AppData\Local\jarvis\jarvis\config.toml`; the daemon became
genuinely ready about 12 s after launch. Log evidence (`logs/jarvis.jsonl`):

```
07:25:36.024  jarvis.daemon.server  task recovery found no stale tasks
07:25:36.214  jarvis.daemon.server  llm provider available name=groq slots=1 model=openai/gpt-oss-120b mode=free_tier
07:25:36.230  jarvis.daemon.server  llm provider available name=openrouter slots=1 model=openrouter/free mode=zero_cost_endpoint
07:25:36.246  jarvis.daemon.server  llm provider available name=nvidia slots=1 model=nvidia/nemotron-3-super-120b-a12b mode=zero_cost_endpoint
07:25:36.259  jarvis.daemon.server  llm provider available name=gemini slots=1 model=gemini-3.8-flash mode=free_tier
07:25:36.268  jarvis.daemon.server  voice bootstrap: enabled=True service_available=True
07:25:47.310  jarvis.voice.tts     piper backend requested but no model_path provided; falling back to SAPI
07:25:47.367  jarvis.voice.tts     TTS sapi backend initialized (win32com SpVoice, stoppable)
07:25:47.622  jarvis.daemon.server  daemon listening on 127.0.0.1:59756 (pid=14796)
07:25:47.622  jarvis.daemon.server  daemon ready — serving until shutdown is requested
07:25:47.710  jarvis.voice.status  voice event=ready voice_state=READY ... wake_word=hey_jarvis
```

`jarvis status` reported `daemon: running`, `voice: on`, `mode: NORMAL`, `version: 0.1.0`.
`daemon.json` contained `{"pid": 14796, "port": 59756, ...}`. Task-store and memory-store opens are
not logged individually; both were exercised by the chat tasks below (the memory context is produced
by `memory_retrieve` and `task_log` rows were written, so both stores were live).

The only startup warning is the optional voice TTS fallback (Piper model path not configured → SAPI).
That is a voice-subsystem configuration gap, not a core daemon failure. No exception, no traceback.

The stale zero-byte `daemon.lock` from Nov 6 is not a stale lock: the daemon started twice while it
existed, and it is removed/reused by the lock protocol rather than by liveness of that file.

### Chat connection

Every session printed `JARVIS chat via daemon (Ctrl+C to exit)`, then `started` (the submission ACK),
then the answer, then `bye`. Connections are authenticated in the daemon log
(`client authenticated: 127.0.0.1:...`), and each REPL session received its own conversation id
(read from the checkpoint state, see section D).

### Graceful shutdown

Because the daemon was launched as a non-interactive background process (no console to deliver a
literal Ctrl+C to), shutdown used the documented IPC shutdown path:

```powershell
.\.venv\Scripts\python.exe -m jarvis stop    # -> "shutdown signal sent"
```

Result, twice (before and after the restart test):

* `daemon.json` removed; `jarvis status` → `daemon not running`.
* Daemon log ends with `microphone closed` / `[VOICE] STOPPING`; no traceback.
* No python process remained (`Get-CimInstance Win32_Process` empty).
* No unrelated process was terminated.

### Restart

A second foreground daemon started cleanly: `task recovery found no stale tasks`, all four providers
reported available, `daemon listening on 127.0.0.1:54422 (pid=7712)`, new `instance_id`. Chat
reconnected successfully and ran a new task.

---

## C. Database integrity (read-only checks)

Opened with `sqlite3` URI `mode=ro` (never `create=True`, never migrated, never repaired). WAL/SHM
files were left untouched; no checkpoint/cleanup command was run.

| Database | Path | `quick_check` | Schema version | Record counts (initial → final) |
|----------|------|---------------|----------------|----------------------------------|
| Checkpoints | `C:\Users\aarya\AppData\Local\jarvis\jarvis\checkpoints.db` | `ok` | `user_version=0` (LangGraph-managed) | `checkpoints` 3682 → 3727; `writes` 22054 → 22388 |
| Task events | `C:\Users\aarya\AppData\Local\jarvis\jarvis\task_events.db` | `ok` | `user_version=3` | `tasks` 22 → 27; `task_events` 186 → 226 |
| Memory | `C:\Users\aarya\AppData\Local\jarvis\jarvis\memory.db` | `ok` | `user_version=2` | `skills` 11 → 11; `failures` 40 → 40 (39 planner-visible, 1 quarantined); `preferences` 0 → 0; `task_log` 283 → 288 |

All three databases are in WAL mode. `task_events.db` had a 0-byte `-wal` and a 32 KB `-shm` before
the test; after clean shutdown all `-wal` files are 0 bytes, i.e. no recovery is pending. Every
database was opened and read successfully before, during and after the run; none is missing or
inaccessible.

The `task_log` growth (+5) is operational telemetry written by `respond` for the five tested tasks,
not a change to user memory. `skills`, `failures` and `preferences` are unchanged.

---

## D. Persistent memory and preferences

### What actually exists (verified with the real `SqliteMemory` API on the production database)

| Kind | Count | Inventory (non-sensitive) |
|------|-------|---------------------------|
| Preferences / explicit facts | **0** | none. `jarvis memory prefs` → `no preferences stored.`; `preferences.get("user_name")` → `None`. There are no `fact.*` rows either. |
| Learned skills | 11 | Open Chrome; Open Calculator; Open Notepad; fastpath: open notepad; Create a file text.txt with hello; Create chattest.txt with "hello" on the Desktop; Delete chattest.txt; Undo the most recent delete operation; Create a hello.py hello-world script on Desktop; Take a photo of the user (×2 phrasings) |
| Failure records | 40 (39 planner-visible, 1 quarantined) | Older failed attempts (browser page reads, WhatsApp contact resolution, `take_photo` OpenCV missing, etc.) |

**There is no stored user preference in the production database.** The preferences and explicit-fact
tables are empty, so no preference could be retrieved, and none was created: creating one would have
meant inventing memory, which this test forbids.

### Retrieval path, exercised against the real store (read-only connection)

`SqliteMemory.retrieve(query, 3)` — the exact method `memory_retrieve` calls:

| Query | Records returned |
|-------|------------------|
| `"open notepad"` | 3 (skill "Open Notepad"; two older Notepad typing failures) |
| `"create a text file"` | 1 (failure: `create_file` reported file exists) |
| `"What is 19 multiplied by 23?"` | 0 |
| `"have I asked you to open notepad before"` | 0 (skills/failures are similarity-scoped; preferences are not) |

So the retrieval machinery works on real rows, and it is query-scoped: an unrelated request retrieves
nothing. Preferences, when present, are query-independent and are always offered (with one reserved
slot for explicit facts), which is why a stored preference would have been visible to every prompt.

### What happened through `jarvis chat`

| Task | Question | `memory_context` in graph state | Answer |
|------|----------|--------------------------------|--------|
| `29672921` (pre-restart) | "What is my saved preference for my coding language? Only use information explicitly stored in my persistent memory. Do not guess." | `[]` | "I don't have a supported tool to read your persistent memory in this session, so I can't tell you your saved coding language preference without guessing. …" |
| `319f5d89` (post-restart) | same question | `[]` | "I can't access your persistent memory with the tools available here, so I can't read or confirm a saved coding language preference. I won't guess." |

Both answers are honest: JARVIS did **not** invent a coding-language preference, and the memory
context really was empty. The wording ("no tool to read memory") is the model's own framing; in this
architecture memory is injected into the prompt, not exposed as a tool.

Distinction requested by the test:

* **Actually persisted:** 11 skills and 40 failure examples; **0 preferences**.
* **Retrieved by JARVIS in a live task:** no preference (none exists); `memory_context` was empty for
  every tested prompt, which is correct given the queries and the empty preference table.
* **Available in the database but not retrieved:** none that is relevant to the tested prompts (the
  store-level retrieval above shows what would be retrieved for a Notepad query).
* **Not stored:** every user preference — including a coding-language preference and a `user_name`.

### Restart persistence

The memory database was closed by a clean daemon shutdown, passed `quick_check` again, and was
re-opened by the second daemon with identical learned-memory counts (11 skills, 40 failures, 0
preferences). Skills/failures therefore survive a daemon restart. Preference recall after restart
could not be demonstrated because no preference exists; the post-restart answer was nevertheless
consistent with the pre-restart answer (same refusal, no invention).

Note: the first post-restart attempt did not reach the model (see section E, task `ee44e73c`); it was
retried once after the provider cooldown, which produced the answer above.

### Isolated lifecycle test (disposable database, never the production file)

`tempfile.mkdtemp()` + `open_memory(temp/memory.db)` using the real B4 API, then the temporary
directory was deleted. Observed:

| Step | Result |
|------|--------|
| Explicit remember (`fact.preference.coding_language_ab12` = `Python`, same shape as `memory_save`) | stored, `preferences.count() == 1` |
| Retrieval through `memory.retrieve("What is my saved preference for my coding language?", 3)` | `['Remembered fact (preference): Python']` |
| Update the same fact to `Rust` | one row, value `Rust` (overwrite, not duplicate) |
| Close and reopen the database | `Rust` still present; retrieval returns it |
| Explicit forget (`preferences.delete`) | `True`; `get()` → `None`; count 0; retrieval `[]` |
| Reopen after forget | value `None`, count 0 — absent from disk |
| Secret rejection (key name, e.g. `api_token` / `login_password`) | `ValueError: preference key looks like a credential; refusing to store it` |
| Secret rejection through `remember(meta=...)` with a credential key | not persisted (logged, count unchanged) |
| `kind="session"` record ("browser page said: remember …") | not persisted (count unchanged) |

Existing focused B4 tests were also run (mocked LLM, temporary SQLite):
`pytest -q tests/unit/test_memory_b4_acceptance.py tests/unit/test_memory_b4.py` → **16 passed**.
They cover the canonical remember/ask/update/forget conversation pair, "ordinary conversation
creates no preference", and taint handling in `memory_save`.

**One limitation found (not fixed, reported):** `PreferenceStore.set` claims to reject
credential-like *values*, but its check (`looks_secret`) only matches `<credential-ish name><:|=><value>`
shapes. Standalone token-shaped values (`sk-live-…`, `AKIAIOSFODNN7EXAMPLE`, `ghp_…`) were accepted
when supplied directly to the store API under a harmless key. This is not reachable through the chat
memory path, which is stricter (`_SENSITIVE_FACT` + `_TOKEN_SHAPE`):

| Chat-level command (pure parse, nothing written) | Verdict |
|--------------------------------------------------|---------|
| `remember that my coding language is Python` | accepted (`fact.personal.my_coding_language_…`) |
| `remember my api key is sk-live-1234567890abcdef` | `error="sensitive"` |
| `remember sk-live-1234567890abcdef` | `error="sensitive"` |
| `remember my card number is 4111 1111 1111 1111` | `error="sensitive"` |

Recommendation: extend `looks_secret`/`PreferenceStore.set` with the same token-shape detector used by
`jarvis.memory.intent`, so a direct API caller cannot store a raw token either.

---

## E. Live model result

**Question submitted (exact):**

```
Explain the difference between a Python list and a tuple. Cover mutability, ordering, and one important caveat about tuples containing mutable objects. Keep it concise.
```

| Item | Value |
|------|-------|
| Task id | `4a427b12-05ce-4099-8b99-5828c6fde241` |
| Conversation id | `6abd49363b684d928046842a389efba3` |
| Route | `task_class=coding`, `model_role=planner`, `reason_code=coding_keyword` |
| Attempt 1 (event 192) | provider `groq`, model `openai/gpt-oss-120b`, role `planner`, outcome **error**, 391 ms, `fallback_index=0` |
| Attempt 2 (event 193) | provider `openrouter`, model `openrouter/free`, role `planner`, outcome **success**, 10,077 ms, `prompt_tokens=11908`, `completion_tokens=336`, `fallback_index=1` |
| Task finish (event 195) | `task_finished`, status `answered`, outcome `success` |
| Fallback used | **Yes** — groq first, then openrouter. No paid model was selected. |
| Provider failure cause (daemon log) | `groq` returned HTTP 400 then **413 Payload Too Large** → `provider groq structured output transient failure: context_length; cooling that role for 300s` |

All eight required evidence points hold: a complete answer appeared in the chat terminal; a real
provider was contacted; the text came from the selected model (376-character `final_answer` in graph
state, not a canned fallback); the task has its own `LLM_CALL` events (one per real attempt, no
duplicates); each event names provider, model, role, outcome and duration; token usage is recorded on
the successful attempt; the event detail contains only allow-listed operational metadata (no prompt,
response, credential or private data); and the answer is factually correct.

**Answer as displayed:**

> Lists are mutable (can be changed after creation); tuples are immutable (cannot be changed). Both
> are ordered sequences that preserve insertion order.
> Important caveat: a tuple containing a mutable object (like a list) cannot be changed itself, but
> the mutable object inside it can still be modified — e.g., `t = ([1, 2], 3); t[0].append(99)` works
> and changes the inner list.

**Factual assessment:** correct on all four required points — lists mutable, tuples immutable as
containers, both ordered, and the contained-mutable-object caveat with a valid example. No fabricated
capability, no claim of having performed an action. It is slightly terser on "ordering" than asked,
but it states ordering explicitly for both. Quality: good. Latency: ~10 s for the successful planner
call, ~14 s wall clock for the turn.

### Provider failure behaviour observed separately

A later memory question (task `ee44e73c-5c04-41a8-8b96-6001a01071d3`) showed the failure path honestly:

* event 215: groq `openai/gpt-oss-120b` → outcome `error`, 312 ms, `failure_category="llm"`
* event 216: openrouter `openrouter/free` → outcome `timeout`, 30,594 ms, `failure_category="timeout"`, `timed_out=true`
* event 218: `task_finished`, `outcome=blocked`, `failure_code=halted`
* Chat displayed: "JARVIS couldn't get a response from the AI backend in time."

Exactly one event per real attempt, no paid model, no silent fallback message. The identical question
succeeded on retry (task `319f5d89`, one openrouter success, 13,984 ms). This is a provider
capacity/deadline limitation, reported rather than hidden.

---

## F. Deterministic route

| Item | Value |
|------|-------|
| Input | `What is 19 multiplied by 23?` |
| Output | `437` (chat displayed only `437`; `final_answer` length 3 in graph state) |
| Task id | `d2ed19c5-ed81-4851-b693-cf567b094996` |
| Conversation id | `34992e182c5942b49cec7e9b4cf3a119` |
| Events | 6: `task_created`, `task_started`, `stage_changed recalling`, `stage_changed planning`, `stage_changed responding`, `task_finished answered` |
| `LLM_CALL` events | **0** (verified by query) |
| Wall clock | 4 s |

No routing/planning event was emitted and no provider was contacted, confirming the deterministic
fast path. The answer matches the recorded outcome.

---

## G. Test evidence summary

| # | Command / interaction | Result | Runtime |
|---|----------------------|--------|---------|
| 1 | `git rev-parse HEAD`, `git status --short` | expected branch/HEAD; only the untracked audit file | — |
| 2 | `.venv\Scripts\python.exe -m jarvis --help` (+ `daemon/chat/memory/tasks/status/doctor --help`) | CLI and syntax confirmed | <1 s each |
| 3 | `.venv\Scripts\python.exe -m jarvis doctor` | all core checks PASS; strict_zero_cost WARN (disabled), Piper OCR/vision WARN | ~2 s |
| 4 | `.venv\Scripts\python.exe -m jarvis status` | `daemon not running` before launch; `running` after | <1 s |
| 5 | Read-only SQLite `quick_check` + counts on all three databases | `ok` before, during and after; counts consistent | <1 s |
| 6 | `jarvis daemon --foreground` (live Windows runtime) | ready in ~12 s, pid 14796 / port 59756 | 12 s |
| 7 | `jarvis chat` — live model question | correct answer via groq→openrouter fallback | 14 s |
| 8 | Event query for task `4a427b12` | 9 events, 2 `llm_call`, tokens recorded, no duplicates | <1 s |
| 9 | `jarvis chat` — `19 × 23` | `437`, 0 provider events | 4 s |
| 10 | `jarvis chat` — memory preference recall (pre-restart) | honest refusal, `memory_context=[]`, 1 real `llm_call` | 16 s |
| 11 | `jarvis stop` → restart → `jarvis chat` | clean stop, clean start, new pid/port/instance, recall retried | ~45 s |
| 12 | Isolated temp-DB lifecycle (remember/retrieve/update/reopen/forget/secrets) | all steps pass; temp dir deleted | ~3 s |
| 13 | `pytest -q tests/unit/test_memory_b4_acceptance.py tests/unit/test_memory_b4.py` | 16 passed (mocked LLM, temp SQLite) | ~5 s |
| 14 | `jarvis stop`, final DB + git checks | clean exit, DBs readable, tree unchanged | 6 s |

Evidence classes:

* **Live Windows runtime:** items 6, 7, 9, 10, 11, 14 (real daemon process, real IPC, real voice boot).
* **Live provider request:** items 7 and 10/11 post-restart (groq and openrouter over HTTPS; logged
  HTTP status lines).
* **Database checks:** items 5, 8, 14 (read-only SQLite).
* **Mocked/unit tests:** item 13 (`FakeLLM`, no network, no production database).
* **Temporary isolated database:** item 12 (system temp directory, deleted afterwards).

Telemetry was checked without dumping content: `detail_json` for every event contains only fields such
as `source`, `stage`, `provider`, `model`, `role`, `outcome`, `duration_ms`, `prompt_tokens`,
`completion_tokens`, `fallback_index`, `task_class`, `timed_out`, `failure_code`. No prompt, answer,
memory text or tool argument is stored there.

---

## H. Additional observations

1. **No preferences exist**, so no preference could be stored/retrieved live. JARVIS answered honestly
   instead of guessing, which satisfies the "do not invent" requirement but leaves preference recall
   unproven on production data.
2. **Provider reliability:** the groq planner path fails with `context_length` (HTTP 413) on this
   prompt size, so the system runs on the openrouter free fallback; one openrouter call exceeded the
   30 s overall deadline. The fallback and the honest halt both worked as designed.
3. **Store-level secret guard gap** described in section D (`looks_secret` does not recognise raw
   token shapes). The chat path is protected; the direct API is not.
4. **Voice:** initialized and ready with a Piper→SAPI fallback because no Piper model path is
   configured. The microphone picked up ambient/TTS audio once (`STT EMPTY — returning to READY`);
   no voice-sourced task was created (`voice-source tasks today: 0`).
5. `jarvis daemon` still prints `[DIAG]` debug lines to stdout (pre-existing, harmless in foreground
   dev mode; the daemon normally runs under `pythonw`).
6. Task classification maps the model's `unsupported`/refusal answer to `halted` / `blocked`. The text
   the user saw matched the state, so the outcome is honest, though "halted" is a coarse label for a
   polite refusal.

---

## I. Final verdict

**PARTIAL — core runtime works, but the memory/preference acceptance requirement remains unverified.**

Reasons:

* **Passed:** the real foreground daemon starts, reconciles, serves, restarts and shuts down cleanly;
  `jarvis chat` connects over IPC, accepts input, shows complete answers, assigns task and conversation
  ids, and exits gracefully; a real eligible free provider produced a correct, complete answer with
  correct per-attempt `LLM_CALL` telemetry, token usage and a genuine fallback (groq `context_length`
  → openrouter `openrouter/free`, never a paid model); the arithmetic question returned `437` with zero
  provider events; all three databases pass `quick_check` and were never migrated or modified; the
  isolated B4 lifecycle (remember → retrieve → update → reopen → forget, secret rejection, no
  session/untrusted persistence) passes against a disposable database.
* **Not verified / failed:**
  1. No explicit preference or fact exists in `memory.db` (`preferences = 0`), so storing and
     retrieving a real user preference through chat could not be demonstrated, and restart
     persistence of a recalled preference is therefore also unproven. JARVIS correctly refused to
     invent one.
  2. One post-restart planner call timed out after groq failed (`ee44e73c`); the task halted honestly
     and the retry succeeded, so provider availability is variable at this prompt size.
  3. `PreferenceStore.set` accepts raw token-shaped values from a direct caller; the chat guard
     rejects them. Low severity in the current product path, but the stated invariant ("a value that
     looks credential-ish is rejected") is not fully met at the store boundary.
