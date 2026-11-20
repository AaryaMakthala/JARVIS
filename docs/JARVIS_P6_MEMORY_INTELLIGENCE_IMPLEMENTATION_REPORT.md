# JARVIS P6 Memory Intelligence — Implementation Report

## 1. Problem selected

Chosen improvement: **relevance-ranked preference and fact recall** in the existing SQLite memory path.

The concrete limitation was that `SqliteMemory._preference_records()` computed a relevance score but then discarded it. Matching preferences and facts were emitted in `PreferenceStore.all()` order (most recently updated first) and truncated only afterwards. A newer partial match could therefore occupy a bounded memory slot ahead of the older stored fact that actually answered the user's request. Separately, verbose requests diluted the old score because overlap was divided by the number of query words, and fact-key hash suffixes counted as stored words.

## 2. Why it mattered

Memory is advisory text injected into the planner. If the wrong preference wins a three-slot budget, JARVIS reasons with irrelevant user context and may answer as if the wrong fact were current. This is exactly the "use memory to improve decisions, not merely store facts" requirement. The fix keeps retrieval bounded, deterministic, local and dependency-free.

## 3. What JARVIS can now do better

- Recall the **best-matching** stored preference/fact first, not merely the newest one.
- Keep recalling the right fact in a **long, chatty request** ("please tell me ... and then open the browser") instead of losing it to score dilution.
- Preserve the intended B9 behaviour: unrelated requests still receive **no** preference memory.
- Carry a small `relevance` score in memory metadata, so the bounded `memory_context` is ordered with the least relevant item last.
- Continue to retrieve correctly after a **database close/reopen** (daemon restart simulation).
- Continue to reject sensitive values through the existing `PreferenceStore` guard.

## 4. How the implementation works

### Store ranking (`src/jarvis/memory/store.py`)

- `_preference_relevance()` now:
  - measures overlap against the **smaller** of the query and stored token sets, so extra question words do not dilute a strong match;
  - requires two shared content words for multi-word queries, preserving the rule that one common word in a longer request is not enough;
  - keeps one-word queries working exactly as before;
  - ignores fact-key hash suffixes by using `fact_details(key)[0]` (the category) instead of the raw key for `fact.*` rows.
- `_preference_records()` now:
  - scores every matching preference/fact;
  - sorts by descending score with a stable sort, so recency and key order from `PreferenceStore.all()` remain the deterministic tie-break;
  - truncates **after** sorting;
  - adds `meta["relevance"] = round(score, 3)`, a plain JSON/checkpoint-safe float.

### Context selection (`src/jarvis/agent/nodes/memory_retrieve.py`)

- `_select()` now skips an item that would exceed `context_max_chars` and continues evaluating later items, instead of stopping at the first oversized item.
- Existing per-item cap, total cap, secret check, stale-failure filter, dedupe, timeout and untrusted-data prompt wrapper are unchanged.

### Compatibility and safety

- No schema change, new dependency, new LLM call, new graph node/edge, policy change, tool change or serializer type change.
- Existing APIs (`SqliteMemory.retrieve`, `remember`, `PreferenceStore.set/get/all/delete`) keep their signatures.
- Stored rows are unchanged; only their retrieval order and one metadata field changed.
- Privacy guards in `normalise_key()`, `PreferenceStore.set()`, `detect_memory_intent()` and `memory_retrieve._is_secret()` were not weakened.

## 5. Files modified and created

### Created

- `tests/unit/test_memory_relevance_ranking.py` — 8 focused tests.

### Modified

- `src/jarvis/memory/store.py` — scoring and ranked truncation.
- `src/jarvis/agent/nodes/memory_retrieve.py` — skip oversized items.
- `tests/unit/test_memory.py` — two stale B9-era expectations corrected to use actually relevant preferences.

### Documentation

- `PROGRESS.md`
- `docs/JARVIS_P6_MEMORY_INTELLIGENCE_IMPLEMENTATION_REPORT.md` (this report)

## 6. Focused test results

Commands run from the repository root with `.venv/Scripts/python.exe` (`python` is not on PATH).

| Command | Result |
|---|---|
| `.venv/Scripts/python.exe -m pytest -o addopts="" -q tests/unit/test_memory_relevance_ranking.py` | `8 passed in 2.00s` |
| `.venv/Scripts/python.exe -m pytest -o addopts="" -q tests/unit/test_memory_relevance_ranking.py tests/unit/test_memory.py tests/unit/test_memory_b4.py tests/unit/test_memory_b4_acceptance.py tests/unit/test_memory_secret_tokens.py tests/unit/test_memory_save_node.py tests/unit/test_memory_graph.py tests/unit/test_name_preference.py tests/unit/test_b9_context_discovery.py tests/unit/test_invariants.py` | `360 passed in 7.94s` |
| `.venv/Scripts/python.exe -m pytest -o addopts="" -q tests/unit/test_agent.py tests/unit/test_agent_architecture.py tests/unit/test_plan_step_ids.py tests/unit/test_b9_context_discovery.py tests/unit/test_plan_summary.py tests/unit/test_replan_scope.py tests/unit/test_specialist_loop_e2e.py tests/unit/test_compose_observations.py tests/unit/test_outcome_agreement.py tests/integration/test_confirmation_resume.py` | `133 passed in 6.33s` |

The first regression command includes the 8 new tests; the second shares `test_b9_context_discovery.py` with it. Tests cover best-match ordering, restart persistence, verbose-query recall, unrelated-request suppression, update replacement, configured limits, sensitive-value rejection, oversized-item skipping and a real `run_task` conversation receiving the ranked fact inside `<untrusted_data>`.

## 7. Ruff and mypy results

| Command | Result |
|---|---|
| `.venv/Scripts/ruff.exe check src/jarvis/memory/store.py src/jarvis/agent/nodes/memory_retrieve.py tests/unit/test_memory_relevance_ranking.py tests/unit/test_memory.py` | `All checks passed!` |
| `.venv/Scripts/ruff.exe format --check <same four files>` | `4 files already formatted` |
| `.venv/Scripts/mypy.exe src/jarvis/policy` | `Success: no issues found in 11 source files` |
| `.venv/Scripts/mypy.exe src/jarvis/memory/store.py src/jarvis/agent/nodes/memory_retrieve.py --follow-imports=silent` | `Success: no issues found in 2 source files` |

## 8. Actual bugs encountered and fixes

- **Verbose-query tests failed after the first scoring change.** The fact key's 10-character SHA suffix (for example `0dff4061f1`) was being tokenised as a stored word, inflating the stored set and dropping a strong match below threshold. Fixed by scoring fact rows from their category instead of the raw hash-bearing key.
- **Two pre-existing stale tests failed** (`test_retrieve_returns_skills_first_then_preferences_then_failures`, `test_retrieve_respects_the_limit`). They used preferences with zero overlap with the query and nevertheless expected them to be retrieved, contradicting the B9 relevance filter already present at HEAD. Fixed the fixtures to use genuinely relevant preferences instead of weakening relevance filtering.
- No policy, privacy, graph or serialization bug was found in the changed flow.

## 9. Known limitations

- Relevance is still lexical with a small synonym map, not semantic; a paraphrase sharing no content words can be missed.
- The score uses token overlap only; it does not prove the stored fact is still factually true. Old information is not automatically downgraded by age.
- Rephrased subjects still produce different fact keys, so "my language" and "my preferred coding language" can coexist as separate rows. Same-subject updates overwrite correctly.
- `meta["relevance"]` is currently emitted but not used to reorder skills/failures or to drive composer trimming beyond natural list order.
- The replan prompt's memory block is still not wrapped in `<untrusted_data>`; the active brain prompt is wrapped. This was left unchanged to keep P6 scoped.
- Implemented and unit-tested here; **not live-verified** with a real LLM provider, Windows daemon restart, GUI, browser, WhatsApp or voice. No full test suite was run without a concrete reason.

## 10. Commit hash and message

- Implementation: `ed77da9f072d0e3d0e3fa98815ca1b10df31133c` (`ed77da9`) — `feat(memory): rank recalled preferences by relevance` (4 files, 202 insertions, 21 deletions).
- Documentation/status commit: `docs(progress): record P6 memory intelligence` (this report and `PROGRESS.md`).

## 11. Branch and working-tree state

- Branch: `main`, tracking `origin/main`, ahead by 11 after both P6 commits.
- Nothing pushed, fetched, reset, stashed or amended.
- Untracked audit documents preserved and untouched: `docs/JARVIS_BACKEND_AUDIT.md`, `docs/JARVIS_COMPREHENSIVE_STATUS_AUDIT.md`.
- No unrelated browser, WhatsApp, voice, frontend or provider changes.

## 12. Next milestone

Add **typed per-tool evidence adapters** (P5 follow-up) or, staying in memory, add **subject-alias/supersede handling** so rephrased corrections overwrite the same durable fact and old values are explicitly marked stale. The latter is the next most valuable memory step because duplicate facts are the remaining main source of contradictory recall.

## 13. Final status

Implemented relevance-ranked preference/fact recall and oversized-item skipping in the real SQLite memory flow. Unit-tested 8 new cases plus the reported 360-result and 133-result focused/neighbouring sets, with Ruff and mypy clean. Live behaviour remains unverified; next step is alias-aware fact supersession.
