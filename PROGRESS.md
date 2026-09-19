# PROGRESS.md

> Maintained by the coding agent. Update at the END of every session. Keep it short and factual.

## Current phase
Phase 0 — DONE

## Phase checklist
- [x] Phase 0: Skeleton, config, secrets, `jarvis init`, `jarvis doctor`
- [ ] Phase 1: LangGraph planner + policy engine + confirmation (interrupt) + basic verify
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

## What works (verified)
- `pip install -e ".[dev]"` on `.venv` (Python 3.11.15 64-bit via uv); `pip check` clean.
- `src/` package skeleton created; every subpackage/module from `docs/02_ARCHITECTURE.md`
  imports on any OS (Phase 1+ modules are docstring-only placeholders).
- `config.py`: nested `Settings` (pydantic-settings, `JARVIS_` + `__` delimiter), validates
  TOML + env precedence against real pydantic-settings 2.15.0 API.
- `secrets.py`: `SecretStore` over keyring → Windows Credential Manager
  (`WinVaultKeyring`); probe round-trip `check_store_access()` returns backend name.
- `logging_setup.py`: redacts registered secrets + key prefixes + sensitive fields +
  phones (keeps last 2 digits); JSONL rotating handler (5 MB × 5) on the root logger.
- `platform_guard.py`: `is_windows`, `require_windows`, `@windows_only`, thread-safe
  `LazyImport`.
- `llm/client.py`: `LLMClient` protocol, `GroqClient` (JSON-schema mode → JSON mode
  fallback on HTTP 400 + Pydantic validation + one repair retry; 429/5xx backoff;
  Usage counters), `GeminiClient` stub, scripted `FakeLLM` used by graph tests.
- `cli.py`: `jarvis init` (writes config.toml + stores keys/IPC token in credential
  store), `jarvis doctor` (PASS/WARN/FAIL table naming missing keys; `--live` real call),
  `status` / `password set|change` stubs (explicit Phase 2/3), `--version`.
- `scripts/check.ps1` runs: ruff format --check, ruff check, mypy (policy strict),
  pytest (`not windows_only/voice/slow`), `verify_env.py --phase 1`.
- `scripts/verify_env.py --phase 1`: 29 pass, 0 warn, 0 fail, 21 skipped. (moved from repo
  root to `scripts/` to match docs).
- Tests: 109 unit/integration tests green (`pytest -q`); `tests/windows_only` covered by
  skip-not-run on non-Windows; real-keyring test passes on this dev PC.

## Known fragile areas
- Groq JSON-schema `strict: True` mode is unverified against a live model (no API key
  configured on this machine yet) — unit tests cover the fallback path via a fake transport.
- Model names are empty placeholders; must be chosen from current Groq/Gemini docs.
- typer 0.27.2 `no_args_is_help` exits with code 2 (not 0) when run with no args; help text
  is still shown. Accepted as upstream behaviour.

## Known bugs
- none

## Decisions log
| Date | Decision | Reason |
|------|----------|--------|
| 2026-09-19 | Phase 0 marketed "done"; single Phase-0 session following docs/06_OPENCODE_PROMPTS.md | AGENTS.md §7.1: one phase per session |
| 2026-09-19 | Groq retries/backoff implemented in `GroqClient._complete` with SDK `max_retries=0` | avoid double-retry with the SDK's internal retries; deterministic behaviour + testable |
| 2026-09-19 | `--version` needs eager option callback (not inside command callback) | typer/click resolves commands before invoking the callback; the eager pattern short-circuits first |
| 2026-09-19 | `SecretStore.check_store_access()` method added (module-level fn kept) | duck-typing lets `run_doctor` swap in fake stores for tests |
| 2026-09-19 | `scripts/check.ps1` created and used as the per-change gate | gives one command the human can run in the viva |

## Manual tests to run on the PC
```powershell
.venv\Scripts\activate
scripts\check.ps1                  # full Phase 0 gate (expect "all Phase 0 checks passed")
jarvis doctor                       # plan PASS/WARN/FAIL; missing groq key expected
jarvis init --non-interactive      # used API; CREDENTIAL STORE read/write flow
jarvis init --groq-api-key ...     # (optional) store your real Groq key, then `jarvis doctor`
jarvis --version ; jarvis status ; jarvis password set   # stubs print their phase
```
`jarvis doctor --live` requires a real Groq key + a model name in `config.toml [llm]`.

## Next step
Run the Phase 1 prompt from `docs/06_OPENCODE_PROMPTS.md` (LangGraph planner + policy engine + confirmation interrupt + basic verify). Do not modify Phase 1 files before then.