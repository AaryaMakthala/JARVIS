# JARVIS P7 Alias-Aware Memory Supersession — Implementation Report

## 1. Problem selected

Chosen improvement: **alias-aware identity and safe fact supersession** in the existing SQLite memory path.

P6 left one important contradiction unresolved: "Remember that my preferred coding language is Java." and "Actually, I prefer Python now." were stored as two different facts because the subject phrasing differed. A later correction could therefore coexist with the stale value, and a bounded recall could show both. The lexical relevance ranking from P6 made the best-matching row more likely to appear, but it could not decide that two phrasings describe the same preference.

## 2. Why it mattered

Memory is advisory text injected into the planner. Contradictory user facts make JARVIS reason with stale context and answer inconsistently. The fix keeps correction handling deterministic, local, bounded and testable: no embedding model, no vector database, no extra LLM call, no new tool and no policy change. Identity is based on an explicit closed alias table, so unrelated facts that merely share a word such as "language" are not merged.

## 3. What JARVIS can now do better

- Resolve a few high-value topics across exact phrasings: `programming_language`, `code_editor`, `operating_system`, `response_language`.
- Resolve a correction from a well-known value alone: "Python" -> programming language, "PyCharm" -> code editor, "Hindi" -> response language.
- Supersede an existing fact after corrections such as "Actually, I prefer Python now." or "I have switched to PyCharm."
- Remove a fact with "I no longer use Java." without storing a replacement.
- Archive the old value under a stable non-retrievable key, preserving history in the existing SQLite table.
- Store replacements as self-describing facts, so P6 relevance can still recall them after a daemon restart.
- Refuse vague corrections ("My preference has changed."), questions, additive statements ("I also use Linux at home."), sensitive values, overlong values and topic conflicts without writing memory.
- Keep ordinary new facts and explicit "remember"/"forget" commands on the existing B4 path.

## 4. How the implementation works

### Alias identity (`src/jarvis/memory/aliases.py`)

- `_TOPIC_PHRASES` contains short exact phrases, matched only when all normalised words appear contiguously.
- `_VALUE_PHRASES` contains an unambiguous known-value lexicon for languages, editors and operating systems.
- `topic_from_text()` and `topic_from_value()` return a canonical topic only when the match is exact and unambiguous; otherwise they return `None`.
- `fact_key_for_topic()` returns the stable key `fact.preference.<topic>`, preserving the existing `fact_details()` category without a schema change.
- `canonical_value()` renders replacements such as `my preferred programming language is Python`. This keeps the stored value self-describing for P6 lexical recall and avoids a bare token losing to unrelated text.

### Correction parsing (`src/jarvis/memory/correction.py`)

- `detect_correction()` recognises a deliberately small set of correction forms: explicit removal, response-language statements, "switched/changed ... to ...", "from now on ...", and "actually/I prefer|use|want ... now/instead".
- It first rejects questions, additive statements, missing correction markers, ambiguous multi-value statements and overlong/sensitive values.
- `CorrectionIntent` is immutable and carries no side effects. It exposes `valid`, with `error` as a closed set of safe codes: `ambiguous`, `sensitive`, `too_long`.
- `is_sensitive_fact()` is now the shared public guard used by both B4 memory intent parsing and the correction parser.

### Supersession (`src/jarvis/memory/supersede.py`)

- `apply_correction()` resolves the topic from the stated alias and the known value; a conflicting pair is ambiguous and is not written.
- It reads only active rows from `PreferenceStore.active()` and identifies existing rows from their value first, then their key. This catches old phrasings such as "I use VS Code." as well as canonical keys.
- Old matching active rows are archived and deleted. A replace then writes the canonical key/value; a remove writes no replacement.
- `SupersessionResult` contains only a status, closed topic and archived count. User values never enter logs, state or result payloads.

### Storage guards (`src/jarvis/memory/prefs.py`, `src/jarvis/memory/store.py`)

- `SUPERSEDED_PREFIX = "superseded."` marks archived rows. `archive_key_for()` uses a truncated SHA-256 of the source key, so repeated corrections reuse one archive row and cannot grow the table without bound.
- Archived rows keep the old timestamp `1970-01-01T00:00:00+00:00`, which keeps them out of `set()` pruning order and out of active retrieval.
- `PreferenceStore.all()` still includes archived rows for inspection; new `active()` excludes them.
- `_validated_value()` centralises the existing blank/length/secret checks for both `set()` and `archive()`.
- `_preference_records()` now reads active rows; `_preference_relevance()` also skips any superseded key defensively.

### Graph write path (`src/jarvis/agent/nodes/memory_save.py`)

- `memory_save()` tries explicit B4 intent first, then deterministic correction parsing.
- Corrections are accepted only for a non-dry-run, non-cancelled, non-halted conversation plan with no steps and no pending clarification.
- Writes happen after `respond`; the answer to the correcting turn is unchanged.
- New closed reason codes: `fact_superseded`, `fact_unchanged`, `fact_ambiguous`.
- Logs use only a closed topic identifier and archived count. No value, phone number, password or message body is logged.

## 5. Files modified and created

### Created

- `src/jarvis/memory/aliases.py` — closed alias/value tables and canonical topic helpers.
- `src/jarvis/memory/correction.py` — deterministic correction parser and immutable intent.
- `src/jarvis/memory/supersede.py` — conservative archive/replace/remove logic.
- `tests/unit/test_memory_supersession.py` — 13 focused tests.

### Modified

- `src/jarvis/memory/intent.py` — exported shared `is_sensitive_fact()` guard.
- `src/jarvis/memory/prefs.py` — archived-history support and centralised value validation.
- `src/jarvis/memory/store.py` — active-only preference retrieval and defensive superseded-key skip.
- `src/jarvis/agent/nodes/memory_save.py` — correction write path and new closed reasons.

### Documentation

- `PROGRESS.md`
- `docs/JARVIS_P7_MEMORY_SUPERSESSION_IMPLEMENTATION_REPORT.md` (this report)

## 6. Focused test results

Commands run from the repository root with `.venv/Scripts/python.exe` (`python` is not on PATH).

| Command | Result |
|---|---|
| `.venv/Scripts/python.exe -m pytest -o addopts="" -q tests/unit/test_memory_supersession.py` | `13 passed in 2.35s` |
| `.venv/Scripts/python.exe -m pytest -o addopts="" -q tests/unit/test_memory_supersession.py tests/unit/test_memory_relevance_ranking.py tests/unit/test_memory.py tests/unit/test_memory_b4.py tests/unit/test_memory_b4_acceptance.py tests/unit/test_memory_secret_tokens.py tests/unit/test_memory_save_node.py tests/unit/test_memory_graph.py tests/unit/test_name_preference.py tests/unit/test_b9_context_discovery.py tests/unit/test_invariants.py` | `373 passed in 8.52s` |
| `.venv/Scripts/python.exe -m pytest -o addopts="" -q tests/unit/test_agent.py tests/unit/test_agent_architecture.py tests/unit/test_plan_step_ids.py tests/unit/test_plan_summary.py tests/unit/test_replan_scope.py tests/unit/test_specialist_loop_e2e.py tests/unit/test_compose_observations.py tests/unit/test_outcome_agreement.py tests/integration/test_confirmation_resume.py` | `86 passed in 5.59s` |

The first regression command includes the 13 new tests. Coverage includes alias resolution, value-only resolution, correction parser acceptance/refusal, rephrased language supersession, editor switching, response-language correction, unrelated/additive statements, duplicate corrections, removal, ambiguity, sensitive values, SQLite close/reopen persistence, and the P6 relevance-ranking regression.

## 7. Ruff and mypy results

| Command | Result |
|---|---|
| `.venv/Scripts/ruff.exe check src/jarvis/memory/aliases.py src/jarvis/memory/correction.py src/jarvis/memory/supersede.py src/jarvis/memory/intent.py src/jarvis/memory/prefs.py src/jarvis/memory/store.py src/jarvis/agent/nodes/memory_save.py tests/unit/test_memory_supersession.py` | `All checks passed!` |
| `.venv/Scripts/ruff.exe format --check src/jarvis/memory/aliases.py src/jarvis/memory/correction.py src/jarvis/memory/supersede.py src/jarvis/memory/intent.py src/jarvis/memory/prefs.py tests/unit/test_memory_supersession.py` | `7 files already formatted` |
| `.venv/Scripts/mypy.exe src/jarvis/policy` | `Success: no issues found in 11 source files` |
| `.venv/Scripts/mypy.exe src/jarvis/memory/aliases.py src/jarvis/memory/correction.py src/jarvis/memory/supersede.py src/jarvis/memory/prefs.py src/jarvis/memory/store.py src/jarvis/agent/nodes/memory_save.py --follow-imports=silent` | `Success: no issues found in 6 source files` |

`memory_save.py` has pre-existing Ruff format drift on untouched lines (the name-statement regexes and candidate selection) and was deliberately not reformatted. Ruff check passes on it. Its pre-existing untyped `_safe_plan() -> dict` was annotated as `dict[str, Any]` so targeted mypy could pass without weakening anything.

## 8. Actual bugs encountered and fixes

- **First focused run: 4 failures.** Bare canonical values such as `"Python"` did not match P6 relevance for "What is my preferred coding language?". Fixed by storing self-describing canonical values and updating expectations.
- **Existing editor value was invisible.** `_pref_topic()` only checked textual aliases and the key, so "I use VS Code." was not recognised as `code_editor`. Fixed by also calling `topic_from_value()` on the stored value.
- **Ambiguous corrections were mislabelled.** "My preference has changed." fell through to the generic fallback and returned `skipped_sensitive`. Fixed by mapping `intent.error == "ambiguous"` to `fact_ambiguous` before the store call.
- **Restart expectation was stale.** The persistence test still expected a bare `"Python"`. Updated to the canonical self-describing value.
- No policy, privacy, graph-edge, serializer or schema bug was found in the changed flow.

## 9. Compatibility and privacy preserved

- No schema migration, new dependency, new tool, new graph node/edge, policy change, tier change, confirmation change or extra LLM call.
- Existing `PreferenceStore` APIs keep their signatures; `all()`, `set()`, `get()` and `delete()` still work. `active()` is additive.
- Explicit B4 "remember"/"forget" commands still run first and keep their old behaviour when no correction is detected.
- Unknown, vague or conflicting corrections write nothing; fail-closed is preferred over guessing.
- Credential-shaped values are rejected by both the parser and the storage layer. `PreferenceStore.archive()` uses the same validation as `set()`.
- Archived rows are inspectable through `jarvis memory prefs`, but never enter retrieval or the planner prompt.
- Tests are unit-level with fake LLM/context objects only; no live provider, desktop, browser, WhatsApp or voice run was performed.

## 10. Known limitations

- The alias table covers four topics and a small value lexicon only. It is not a general ontology or semantic similarity system.
- Corrections require an explicit marker ("actually", "now", "instead", "switched", "no longer", "from now on"). A plain assertion like "Java is my language" is not treated as a correction.
- A value not in the lexicon needs an explicit topic phrase; otherwise the correction is left as `fact_unchanged` or `fact_ambiguous` rather than guessed.
- Archived rows keep one value per source topic. A long sequence of changes to the same fact overwrites the prior archive row rather than keeping an unbounded chain.
- `PreferenceStore.all()` and `jarvis memory prefs` show archived rows; this is intentional for inspection but requires reading the `superseded.` prefix.
- There is no CLI command to promote an archived value back to active; use an explicit new fact or correction instead.
- No full test suite was run without a concrete reason. Live Windows restart and real LLM behaviour remain unverified.

## 11. Manual smoke tests for the dev PC

These are not automated here. Run on the Windows PC with a configured zero-cost LLM:

```powershell
.venv\Scripts\activate
jarvis chat
```

Then try:

1. `Remember that my preferred coding language is Java.`
2. `What is my preferred coding language?` — expect Java.
3. `Actually, I prefer Python now.` — expect a short acknowledgement, no clarification.
4. `What is my preferred programming language?` — expect Python, not Java.
5. `Remember that I use VS Code.` followed by `I have switched to PyCharm.` — recall should show PyCharm.
6. `I want responses in Hindi now.` followed by `What response language do I prefer?` — expect Hindi.
7. `I no longer use Java.` — the Java row should become non-retrievable.
8. `My preference has changed.` — expect no write and a safe clarification/refusal.
9. `I also use Linux at home.` — expect additive statement not to overwrite an existing preference.
10. `jarvis memory prefs` — expect one active `fact.preference.*` row per topic and `superseded.<hash>` rows for old values only.

## 12. Commit hashes and branch state

- Implementation: `45b1870bf50450eae36e35870fa0e170a43e63ae` (`45b1870`) — `feat(memory): supersede corrected facts across aliases` (8 files, 778 insertions, 21 deletions).
- Documentation/status commit: `docs(progress): record P7 memory supersession` (this report and `PROGRESS.md`).
- Branch: `main`, tracking `origin/main`, ahead by 14 after both P7 commits.
- Nothing pushed, fetched, reset, stashed or amended.
- Untracked audit documents preserved and untouched: `docs/JARVIS_BACKEND_AUDIT.md`, `docs/JARVIS_COMPREHENSIVE_STATUS_AUDIT.md`.
- No unrelated browser, WhatsApp, voice, frontend or provider changes.

## 13. Next milestone

Possible next steps, in priority order:

1. Live-verify the smoke-test list above on the Windows dev PC with a real LLM and a restarted daemon.
2. Extend the closed alias table with project-specific topics only after observing real correction phrasing.
3. Return to the P5 follow-up: typed per-tool evidence adapters and an ablation measurement.
4. Address the P6 replan memory wrapper gap if memory starts reaching replanning with stale or contradictory text.

## 14. Final status

Implemented conservative alias-aware fact supersession in the real SQLite memory path. New tests pass (13 focused, 373 memory/context/invariants, 86 agent/compose), Ruff check and targeted mypy are clean. Live behaviour remains unverified; the next required action is the manual smoke-test list above.
