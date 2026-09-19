# PROGRESS.md

> Maintained by the coding agent. Update at the END of every session. Keep it short and factual.

## Current phase
Phase 1 — DONE (confirmation/resume flakiness fixed, secure checkpoint serialization, full test coverage)

## Phase checklist
- [x] Phase 0: Skeleton, config, secrets, `jarvis init`, `jarvis doctor`
- [x] Phase 1: LangGraph planner + policy engine + confirmation (interrupt) + basic verify
- [ ] Phase 2: File tools, Recycle Bin delete, undo log, JARVIS password/unlock
- [ ] Phase 3: Daemon, socket client, Task Scheduler autostart, on/off
- [ ] Phase 4: Google open/search, web answers with sources
- [ ] Phase 5: Voice (wake word, STT, TTS), Notepad dictation
- [ ] Phase 6: WhatsApp with contacts and confirmation
- [ ] Phase 7: `jarvis audit`
- [ ] Phase 8: Skill memory, risk classifier, advanced verification/replan
- [ ] Phase 9: Benchmark, ablation, report, demo
- [ ] Phase 10: Packaging, deployment (wheel + pipx, autostart XML, uninstall), clean-machine test
- [ ] Dependency freeze (`requirements.lock`, `verify_env.py --all --live` green)

## What works (verified), Phase 1 additions
- `agent/` Phase 1 graph: `intake → memory_retrieve → plan → validate → policy_gate(interrupt)
  → act → verify → respond` compiled on a file-backed `SqliteSaver`; `run_task`/`resume_task`
  with per-task thread ids; `extract_interrupt` surfaces the `confirm` payload.
- `policy/engine.py` decides deterministically: tier = max(base, rules, classifier[, taint]);
  unknown tool / invalid args / `matches_blocked` (e.g. `disable_defender`) all return a
  Tier-3 refused Decision carrying a recorded reason; summary comes from `spec.describe(canonical args)`;
  `action_hash = sha256(tool + canonical json args)`.
- Secure checkpoint serialization: `graph.build_secure_serde()` = `JsonPlusSerializer` with an
  explicit allowlist of exactly `Plan/Step/Decision/StepResult`, `pickle_fallback=False`,
  `allowed_msgpack_modules` is a set (never `True`). `open_sqlite_checkpointer` uses it; the CLI
  `jarvis chat` routes through it too. `tests/conftest.py` sets `LANGGRAPH_STRICT_MSGPACK=true`
  so any unmanaged serde path becomes loud.
- Confirmation/resume e2e proven (integration suite): approved runs+verifies+reports;
  denied / tampered-hash / plan-mutated-after-approval all fail closed with provider different
  message and tool never runs; unknown tool and Tier-3 hard block refuse before any side effect;
  verification failure is reported honestly, never claimed as success; fresh-connection resume
  (simulated daemon restart) works.
- Runner state access fixed/regression-tested: runner reads state via `graph.get_state(config)`,
  never `checkpointer.get_state` (SqliteSaver has no such method). Decisions appear in state
  only after resume (they are written after `interrupt()` returns) — documented in a test.
- Tests: added `tests/conftest.py`, `tests/support.py` (fake tool harness), `tests/unit/test_serializer.py`,
  `tests/unit/test_agent.py`, `tests/unit/test_invariants.py`,
  `tests/integration/test_confirmation_resume.py`. Full `pytest -q` green (`scripts/check.ps1`:
  ruff format/check, mypy strict on policy, pytest `not windows_only/voice/slow`, verify_env --phase 1).
- Phase 0 items unchanged and still green.

## Known fragile areas
- Groq JSON-schema `strict: True` mode is unverified against a live model (no API key
  configured on this machine yet) — unit tests cover the fallback path via a fake transport.
- Model names are empty placeholders; must be chosen from current Groq/Gemini docs.
- typer 0.27.2 `no_args_is_help` exits with code 2 (not 0) when run with no args; help text
  is still shown. Accepted as upstream behaviour.
- `verify_env.py` phase tags/skips are per-phase; the "all Phase 0 checks passed" banner in
  `check.ps1` was renamed to "all JARVIS checks passed".

## Known bugs
- none in this session (see Decisions log for the two real bugs found & fixed: engine
  hard-block tier/reasons, respond claiming success on verification failure).

## Decisions log
| Date | Decision | Reason |
|------|----------|--------|
| 2026-09-19 | Phase 0 marketed "done"; single Phase-0 session following docs/06_OPENCODE_PROMPTS.md | AGENTS.md §7.1: one phase per session |
| 2026-09-19 | Groq retries/backoff implemented in `GroqClient._complete` with SDK `max_retries=0` | avoid double-retry with the SDK's internal retries; deterministic behaviour + testable |
| 2026-09-19 | `--version` needs eager option callback (not inside command callback) | typer/click resolves commands before invoking the callback; the eager pattern short-circuits first |
| 2026-09-19 | `SecretStore.check_store_access()` method added (module-level fn kept) | duck-typing lets `run_doctor` swap in fake stores for tests |
| 2026-09-19 | `scripts/check.ps1` created and used as the per-change gate | gives one command the human can run in the viva |
| 2026-09-19 | Phase 1 checkpoint serde: explicit allowlist of the 4 State models; no pickle fallback; `LANGGRAPH_STRICT_MSGPACK=true` in tests | reservations: block arbitrary type revival from checkpoint bytes (docs/03 §5); strict env makes future warnings loud for the whole suite |
| 2026-09-19 | `graph.update_state(config, {"plan": ...})` can mutate a paused task's plan; policy_gate recomputes the action_hash and refuses | used by the "modified after approval" e2e; deterministic, no code change needed |
| 2026-09-19 | Hard-blocked steps (`matches_blocked`) now go through `_blocked()` → tier 3 + recorded reason | engine previously returned tier 0, empty reasons → misleading "(blocked by policy)" refusal |
| 2026-09-19 | `respond` treats `verified is False` as a hard failure, never claims success | action executed but post-condition check failed (docs/02 §11) |
| 2026-09-19 | langgraph **persists the raw resume payload** in its internal `__resume__` write channel | earlier smoke test reading only `checkpoints` was wrong; consequence: the daemon must never put secrets in resume values (docs/03 invariant 8 — password verified in daemon layer before resume). State/channel_values stay clean; invariant 7 test asserts that exact boundary |

## Manual tests to run on the PC
```powershell
.venv\Scripts\activate
scripts\check.ps1                  # full gate (expect "all JARVIS checks passed")
jarvis doctor                       # plan PASS/WARN/FAIL; missing groq key expected
jarvis init --non-interactive      # used API; CREDENTIAL STORE read/write flow
jarvis init --groq-api-key ...     # (optional) store your real Groq key, then `jarvis doctor`
jarvis --version ; jarvis status ; jarvis password set   # stubs print their phase
```
`jarvis doctor --live` requires a real Groq key + a model name in `config.toml [llm]`.

## Next step
Phase 2 prompt from `docs/06_OPENCODE_PROMPTS.md` (file tools — create_file verified on
Windows, Recycle Bin delete + undo log, JARVIS password/unlock gate). Do not modify Phase 2
files before then.