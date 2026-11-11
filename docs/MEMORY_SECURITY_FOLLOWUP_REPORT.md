# JARVIS — Memory Secret-Boundary Fix & Preference Preparation (follow-up)

Date: 2026-11-11
Branch: `feature/b1-task-event-store`
Parent evidence: `docs/LIVE_RUNTIME_ACCEPTANCE_REPORT.md` (kept unchanged)
Scope: close the `PreferenceStore.set` credential gap; prepare — not create — real user preferences.

---

## 1. Repository state at start

| Item | Value |
|------|-------|
| Branch | `feature/b1-task-event-store` |
| HEAD before the work | `f7b4ee72ba653f2f4d97ede6c3ff9f59bac26fff` |
| `git status --short` before | `?? docs/JARVIS_BACKEND_AUDIT.md`, `?? docs/LIVE_RUNTIME_ACCEPTANCE_REPORT.md` |
| Interpreter | `.venv\Scripts\python.exe` — CPython 3.11.15 |

Memory source inspected: `memory/db.py`, `memory/prefs.py`, `memory/intent.py`, `memory/store.py`,
`memory/skills.py`, `memory/failures.py`, `memory/tasklog.py`, `memory_cli.py`,
`agent/nodes/memory_save.py`, `agent/nodes/memory_retrieve.py`, `agent/nodes/brain.py`.
Memory tests inspected: `test_memory*.py`, `test_memory_save_node.py`, `test_name_preference.py`,
`test_taskstore_privacy.py`, `test_task_log.py`, `test_secrets.py`.

Nothing was reset, cleaned, rebased, amended, pushed or migrated. `docs/JARVIS_BACKEND_AUDIT.md` was
never staged or opened for writing.

---

## 2. The defect and the exact correction

### Defect

`PreferenceStore.set()` validated its value with `looks_secret()`, which only matched
`<credential-ish name><: | =><value>` pairs. A bare vendor token has no such pair, so:

```python
store.set("favourite_note", "sk-proj-…")   # accepted before the fix
```

was persisted verbatim. `SqliteMemory.remember()` reaches the same code, and a token used as a *key*
(`store.set("sk-proj-…", "x")`) also passed `normalise_key()`.

The conversational path was already safe because `jarvis.memory.intent` carried its own token regex
(`_TOKEN_SHAPE`) and a credential-word regex (`_SENSITIVE_FACT`) — two detectors that could drift
apart. That divergence is what let the defect survive.

### Correction (3 source files)

| File | Change |
|------|--------|
| `src/jarvis/memory/db.py` | Added `TOKEN_SHAPE` (vendor token formats) and `looks_like_token()`. `looks_secret()` now returns `True` for a bare token as well as for `name=value` pairs. `redact_secrets()` replaces bare tokens with `[redacted]` after redacting pairs. Both new names exported. |
| `src/jarvis/memory/intent.py` | Removed the private duplicate `_TOKEN_SHAPE`; `_is_sensitive()` now relies on `looks_secret()`, so the chat guard and the store guard use one detector and cannot diverge again. |
| `src/jarvis/memory/prefs.py` | `normalise_key()` also rejects `looks_like_token(value)`, closing the token-as-key variant. `set()` needed no structural change: it already validates before the `INSERT`, so both insert and the `ON CONFLICT … DO UPDATE` path are now covered. |

Design points that satisfied the brief:

* **Before any write** — validation runs before `conn.execute`; a rejected update leaves the previous
  row untouched (proved by test).
* **Whole operation rejected** — `ValueError` is raised; nothing is stripped or partially stored.
* **No disclosure** — the exception text is the constant "preference value looks like a credential;
  refusing to store it". The store's caller logs `exc` only, and a test asserts the token is absent
  from the exception, the log capture and the stored rows.
* **No blanket heuristics** — no entropy, length or "looks random" rule. Only explicit vendor prefixes
  with minimum body lengths, so ordinary words, paths and code are unaffected.

---

## 3. Supported patterns and false-positive controls

| Pattern | Shape matched | Minimum body |
|---------|---------------|--------------|
| `sk-…` | OpenAI / OpenRouter style keys | 8 chars of `[A-Za-z0-9_-]` |
| `gh[pousr]_…` | GitHub classic tokens (`ghp_`, `gho_`, `ghu_`, `ghs_`, `ghr_`) | 8 chars of `[A-Za-z0-9]` |
| `github_pat_…` | GitHub fine-grained token | 8 chars of `[A-Za-z0-9_]` |
| `AKIA…` | AWS access-key id | 12 chars of `[0-9A-Z]` after `AKIA` |
| `AIza…` | Google API key | 20 chars of `[A-Za-z0-9_-]` |
| `eyJ….….` | JWT (`header.payload.` prefix) | 10 + 10 chars |

All alternatives require a `\b` word boundary and the vendor prefix, so these ordinary strings are
accepted (and are pinned by tests): `Python`, `VS Code`, `Git Bash`, `4 spaces`, `dark`,
`scikit-learn`, `task-queue`, `print('hello world')`, `https://github.com/aarya/jarvis`,
`C:/Users/aarya/dev/jarvis`, `Windows 11 + CPython 3.11`.
`sk-learn` is too short after `sk-` to match; `jwt-notes` does not start with `eyJ`.

Known, accepted trade-off: a kebab-case identifier starting `sk-` with 8+ body characters (e.g.
`sk-image-pipeline`) is treated as a credential. That is the same rule the chat path already used, and
a false positive only refuses one memory write.

---

## 4. Tests and exact results

New file: `tests/unit/test_memory_secret_tokens.py` — **32 tests**, all passing.

| # | Requirement | Test |
|---|-------------|------|
| 1 | rejects synthetic OpenAI-style token | `test_set_rejects_token_values[sk-…]` |
| 2 | rejects synthetic AWS access-key id | `test_set_rejects_token_values[AKIA…]` |
| 3 | rejects synthetic GitHub tokens (`ghp_`, `github_pat_`) | `test_set_rejects_token_values[ghp_…/github_pat_…]` |
| 4 | nothing written on rejection (row query, not just count) | `test_set_rejects_token_values`, `test_remember_rejects_token_without_writing` |
| 5 | update rejected without corrupting the previous value | `test_rejected_update_keeps_previous_value` |
| 6 | ordinary preferences still work; no false positives | `test_ordinary_preferences_still_work`, `test_detector_keeps_ordinary_text` |
| 7 | chat guard and store agree | `test_chat_and_store_reject_the_same_token`, `test_chat_accepts_ordinary_fact_that_store_accepts` |
| 8 | no token in exceptions, logs or rows | `test_rejection_does_not_leak_or_log_the_token`, `test_redaction_removes_bare_tokens_but_keeps_prose` |
| 9 | existing behaviour intact | focused B4 suites below, plus `test_remember…`/`test_ordinary…` |

Also covered: token-shaped **key** rejection, detection of all four formats, and redaction.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_memory_secret_tokens.py
# 32 passed in 1.36s

.\.venv\Scripts\python.exe -m pytest tests/unit/test_memory_secret_tokens.py `
  tests/unit/test_memory_b4_acceptance.py tests/unit/test_memory_b4.py `
  tests/unit/test_memory.py tests/unit/test_memory_save_node.py `
  tests/unit/test_memory_graph.py tests/unit/test_name_preference.py `
  tests/unit/test_taskstore_privacy.py tests/unit/test_task_log.py tests/unit/test_secrets.py
# 328 passed in 7.12s

.\.venv\Scripts\python.exe -m ruff check  <3 source files + new test>
# All checks passed!
.\.venv\Scripts\python.exe -m ruff format --check <same>
# 4 files already formatted
```

The full test suite was **not** run, per instruction. All tokens used in tests are clearly synthetic
(`SYNTHETIC` in the body) and were never issued by a provider.

---

## 5. Production memory was not modified

Checked read-only (`mode=ro`) before and after the lifecycle work:

| Check | Before | After |
|-------|--------|-------|
| `PRAGMA quick_check` | `ok` | `ok` |
| `skills` | 11 | 11 |
| `failures` | 40 | 40 |
| `preferences` | 0 | 0 |
| `memory.db` SHA-256 (first 16) | `123f11e39b0a71fd` | `123f11e39b0a71fd` |

The file is byte-identical. No preference was created, no row was deleted, no schema change, no
WAL/SHM cleanup. The candidate statements in section 7 were only *parsed* (`detect_memory_intent`),
which is a pure function and writes nothing.

---

## 6. Temporary memory lifecycle

Disposable `tempfile.mkdtemp()` database via the real `open_memory()`; directory deleted afterwards:

| Step | Observed |
|------|----------|
| Remember | `VS Code`, 1 row |
| Retrieve (`"what editor do I prefer"`) | `Remembered fact (preference): VS Code` |
| Update | `Neovim`, still 1 row (overwrite, no duplicate) |
| Close + reopen | `Neovim`; retrieval returns `Neovim` |
| Forget | `None`, 0 rows |
| Reopen after forget | `None`, 0 rows |
| Secret rejection ×4 (`sk-`, `AKIA`, `ghp_`, `github_pat_`) | `ValueError` each time; `leaks? False` |
| Temp directory | removed |

---

## 7. Candidate preferences — for the user to approve and enter (not saved)

The production store still has **zero** preferences. The commands below are phrased exactly as
`detect_memory_intent` expects (imperative, under 400 characters) and each was verified to parse to a
stable, non-sensitive fact key. **Nothing has been written.**

| # | Command to type in `jarvis chat` | Parsed key | Category |
|---|----------------------------------|------------|----------|
| 1 | `remember that I prefer Windows with a project-local Python 3.11 .venv for JARVIS development` | `fact.preference.i_prefer_windows_with_a_project_lo_086134a259` | preference |
| 2 | `remember that I prefer focused, bounded tests over routinely running the full test suite` | `fact.preference.i_prefer_focused_bounded_tests_ove_e0a192c638` | preference |
| 3 | `remember that you should stage files by explicit path and make local commits only; never push without my authorization` | `fact.personal.you_should_stage_files_by_explicit_6f285b4ee9` | personal |
| 4 | `remember that JARVIS must stay free-tier and zero-cost only; never switch to a paid model` | `fact.project.jarvis_must_stay_free_tier_and_zer_e3296b7638` | project |
| 5 | `remember that I prefer controlled implementation scope and no unnecessary refactors` | `fact.preference.i_prefer_controlled_implementation_fa889c44fc` | preference |
| 6 | `remember that I want a short implementation plan before you start unless I explicitly authorize implementation` | `fact.personal.i_want_a_short_implementation_plan_abd0d20245` | personal |

Each is grounded in an explicit instruction in this project's context (the AGENTS.md workflow rules and
the user's own acceptance-test instructions), not in inference. Review, edit or drop any of them before
running; they are proposals, not memories.

**Preference recall is still unverified.** It becomes verified only when: the user runs these commands
in `jarvis chat`, `jarvis memory prefs` shows the rows, the daemon is restarted, and a *fresh* chat
session answers a recall question from the stored text. Do not treat section 7 as evidence of recall.

---

## 8. Provider context-size investigation (no routing change)

### Measurement

Built the real planner request locally with the production registry and the same prompt builders, for
the same question that produced the live 11,908 prompt-token count:

| Component | Size | ≈ tokens (chars/4) | Share |
|-----------|------|--------------------|-------|
| Tool catalogue (57 tools, name + description + args JSON schema) | 44,157 chars | 11,039 | **92%** |
| System instructions (excluding catalogue) | 3,526 chars | 882 | 7% |
| User prompt (question only) | 183 chars | 46 | <1% |
| **Messages total** | 47,866 chars | **11,967** | 100% |
| `BrainDecision` JSON schema, sent separately as `response_format` | 3,364 chars | 841 | (not in prompt tokens) |

The estimate is within 0.5% of the provider's reported 11,908 prompt tokens, so the size is explained:
it is the **full tool catalogue**, not memory, history or the user's words. For that task
`memory_context` was `[]` and the conversation was fresh, so neither contributed anything. The largest
single schemas are `whatsapp_read_messages` (1,113 chars), `whatsapp_verify_message` (1,108),
`whatsapp_open_chat` (903) and `whatsapp_send_message` (850).

### Budget comparison

* No request-budget calculation exists: the brain does not estimate tokens and passes no `max_tokens`.
* `ModelSpec.context_tokens` is `None` for every registered model, including
  `groq/openai/gpt-oss-120b` and `openrouter/free` — the registry deliberately says "not known".
* `PLANNER_REQUIREMENT` has `min_context_tokens=None`, so capability selection never rejects a model
  for an unknown window, and nothing compares the prompt to a documented limit.
* Therefore there is **no demonstrable incorrect calculation or eligibility decision** to fix. Groq's
  HTTP 413 is a provider-side rejection of an ~11.9k-token request for that model on that account; the
  client had no registered limit to pre-check and correctly fell through to the free OpenRouter route.

### Known optimization (documented, not applied)

1. The catalogue is 92% of every planner prompt. `agent.prompt_trim_enabled` already exists (default
   `false`) and selects a smaller catalogue via `tool_groups.select_catalogue`; enabling it would cut
   the prompt substantially without touching the registry, validation or policy.
2. Registering a verified per-request/context cap for groq models would let capability selection skip
   them before the failing call — but that is a routing/eligibility change, so it was not made here.

Neither change was applied: provider ordering, free-only qualification, model preferences and
capability rules are unchanged.

---

## 9. Commits, HEAD and working tree

| Commit | Message | Contents |
|--------|---------|----------|
| `5b5f108` | `fix(memory): reject bare vendor tokens at the persistence boundary` | `src/jarvis/memory/{db,intent,prefs}.py`, `tests/unit/test_memory_secret_tokens.py` |
| documentation commit | `docs(test): record the memory security follow-up` | this report and the live acceptance report |

No push. No temporary database, log or credential is committed. `docs/JARVIS_BACKEND_AUDIT.md` remains
untracked and untouched.

Final HEAD is the documentation commit that carries this file (a file cannot contain its own hash);
`git log -1 --oneline` shows it. Verification after that commit: `git status --short` shows no modified
source files — only `?? docs/JARVIS_BACKEND_AUDIT.md` if the audit file is still untracked, which is
expected and intentionally left alone.
