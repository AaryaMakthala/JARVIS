# PROGRESS.md

> Maintained by the coding agent. Update at the END of every session. Keep it short and factual.

## Current phase
Phase 4 — IN PROGRESS (web answers, untrusted-data handling, taint propagation)
All 354 automated tests pass (`pytest -q` — 0 failures, 0 errors).
No real LLM provider/key is configured in this development environment, so real-LLM
chat, file creation, and daemon end-to-end behaviour are **not** manually validated.
Manual testing with a real LLM is deferred until after Phase 5 when voice and
live-provider integration are in place.

## Phase checklist
- [x] Phase 0: Skeleton, config, secrets, `jarvis init`, `jarvis doctor`
- [x] Phase 1: LangGraph planner + policy engine + confirmation (interrupt) + basic verify
- [x] Phase 2: File tools, Recycle Bin delete, undo log, JARVIS password/unlock
- [x] Phase 3: Daemon, socket client, Task Scheduler autostart, on/off
- [x] Phase 4: Google open/search, web answers with sources
- [ ] Phase 5: Voice (wake word, STT, TTS), Notepad dictation
- [ ] Phase 6: WhatsApp with contacts and confirmation
- [ ] Phase 7: `jarvis audit`
- [ ] Phase 8: Skill memory, risk classifier, advanced verification/replan
- [ ] Phase 9: Benchmark, ablation, report, demo
- [ ] Phase 10: Packaging, deployment (wheel + pipx, autostart XML, uninstall), clean-machine test
- [ ] Dependency freeze (`requirements.lock`, `verify_env.py --all --live` green)

## What works (verified), Phase 4 additions
- **`web_answer` tool** (`tools/web.py`): Tier 0; searches via `ddgs`, generates a cited
  answer via LLM, verifies citations against real results, flags output as `tainted=True`.
  Citation verification: out-of-range or fabricated `[n]` references → `verified=False`.
  No results or network error → clear error message.
- **Deterministic taint detection** (`policy/rules.py` `args_overlap_taint`): independent
  of LLM's `depends_on_untrusted` flag. Engine checks ≥12-char substring overlap between
  step args and tainted fragments from previous results. LLM omission cannot clear taint.
- **`Decision.warn_untrusted`** (`agent/state.py`): engine-side flag computed deterministically;
  `policy_gate` uses this for the confirmation banner instead of the LLM-controlled flag.
- **`PolicyContext.tainted_fragments`** (`policy/engine.py`): enriched by `policy_gate` from
  `state["results"]` where `tainted=True`; passed to `engine.decide()` for overlap check.
- **URL scheme allowlist** (`tools/web.py` `is_allowed_http_url`): `urllib.parse.urlsplit`-based;
  rejects `file:`, `javascript:`, `data:`, `ms-settings:`, credentials in URL.
- **Invariant #8** added to `test_invariants.py`: injection in web result yields tainted
  output and engine independently forces Tier ≥ 1.
- **Red-team cases** (`benchmarks/redteam.yaml`): 17 cases covering injection (4),
  URL attacks (5), citation forgery (2), taint bypass (3), mixed attacks (1),
  and edge cases (2).
- Tests: `test_web_answer.py` (22 tests), `test_untrusted.py` (18 tests),
  invariant #8 in `test_invariants.py` (1 test). Total suite: 354 tests.

## What has NOT been verified (deferral)
- Real LLM provider call (no Groq/Gemini API key in this env)
- `jarvis daemon --foreground` + `jarvis chat` end-to-end with a real LLM
- File creation/deletion through the daemon with a real planner
- `jarvis autostart` on reboot/logon (manual; deferred)
- `jarvis doctor --live` (requires a real Groq key + model name)
- Live web search via `ddgs` (all tests use mocks)

## What works (verified), Phase 3 additions
- **Daemon server** (`daemon/server.py`): asyncio TCP on `127.0.0.1:<random port>`, NDJSON protocol.
  - Token auth via `hmac.compare_digest` — first message must be `auth` with the IPC token from keyring.
  - Task queue: 1 active + up to 5 queued; new `chat` messages get queued when busy.
  - Worker thread runs LangGraph synchronously; `threading.Event` bridges interrupt→confirm→resume flow.
  - Stale confirmation cleanup every 10s (60s timeout).
  - Single-instance enforcement via `daemon.json` pid check.
  - `pythonw` stdio redirect to log file + `sys.excepthook`/`threading.excepthook` installed.
  - Graceful shutdown on SIGTERM/SIGINT and via IPC `shutdown` message.
- **Daemon client** (`daemon/client.py`): `DaemonClient` with `connect`, `send_chat`, `send_confirm`,
  `send_cancel`, `send_shutdown`, `get_status`, `wait_for_event`.  Uses `_get_or_create_loop()` helper
  for Python 3.10+ asyncio compatibility.
- **Protocol** (`daemon/protocol.py`): 12 Pydantic message models for client↔server NDJSON exchange.
  Max message size 64 KB enforced server-side.
- **Autostart** (`daemon/autostart.py`): Task Scheduler XML generation, `schtasks /Create /XML`,
  `schtasks /Delete`, `schtasks /Query`.  Task name `JarvisAgent`, At log-on trigger with 20s delay,
  least-privilege, restart on failure (1min × 3).
- **CLI commands** (`cli.py`):
  - `jarvis daemon --foreground` — starts the daemon in the foreground (dev mode).
  - `jarvis status` — queries the daemon over IPC.
  - `jarvis stop` — sends shutdown signal to the daemon.
  - `jarvis chat` (default) — connects to the daemon for interactive chat.
  - `jarvis chat --no-daemon` — in-process agent (Phase 1 mode, unchanged).
  - `jarvis autostart enable|disable|status` — manage Task Scheduler autostart.
- **Confirmation over IPC**: Client collects password (via `getpass`), sends it in `confirm_response`,
  daemon verifies with `UnlockManager` before resuming the graph.  Password never enters the graph
  (docs/03 invariant 8).
- **Invariant #10**: IPC requires token — tested with real TCP socket: wrong token rejected,
  missing auth rejected, correct token grants access.
- Tests: `test_daemon_protocol.py` (43 tests), `test_daemon_server.py` (11 tests),
  `test_ipc_auth.py` (5 tests), invariant #10 added to `test_invariants.py`.
  Regression suite in `test_daemon_wake.py` (11 tests): wake mechanism (3),
  confirm flow ≤2s (3), direct-completion Tier 0 (1), queue promotion (1),
  status-while-active (1), denied/tampered hash (2).
  Full `pytest -q` green (313 tests).

## What has NOT been verified (deferral)
- Real LLM provider call (no Groq/Gemini API key in this env)
- `jarvis daemon --foreground` + `jarvis chat` end-to-end with a real LLM
- File creation/deletion through the daemon with a real planner
- `jarvis autostart` on reboot/logon (manual; deferred)
- `jarvis doctor --live` (requires a real Groq key + model name)

## What works (verified), Phase 2 additions
- File tools: `list_dir`, `read_file`, `create_file` (overwrite handling), `append_file`,
  `delete_path`, `undo_last_delete` with undo log (JSONL).
- `paths.py`: `resolve_safe`, protected list, allowed roots, junction/UNC/ADS blocking.
  Hypothesis property tests for path traversal.
- `policy/unlock.py`: Argon2id password hashing, TTL-based unlock, lockout with exponential backoff.
- `jarvis password set|change`, `jarvis lock|unlock`.
- Typed folder-name confirmation for folder deletes; delete preview with item count and total size.
- Tier 2 flow in the terminal loop (yes/no + password when locked).

## Known fragile areas
- Groq JSON-schema `strict: True` mode is unverified against a live model (no API key
  configured on this machine yet) — unit tests cover the fallback path via a fake transport.
- Model names are empty placeholders; must be chosen from current Groq/Gemini docs.
- typer 0.27.2 `no_args_is_help` exits with code 2 (not 0) when run with no args; help text
  is still shown. Accepted as upstream behaviour.
- Daemon worker thread uses `threading.Event` for interrupt→resume bridging; edge case:
  if the daemon crashes between storing `confirm_payload` and the client sending the response,
  the confirmation is lost (acceptable for v1 — stale confirmation cleanup handles this).
- `pythonw` daemon has no console; all output goes to log file.  If log file path is broken,
  stdout/stderr redirect fails silently.
- `schtasks` autostart uses XML with escaped paths; untested on machines with unusual Python
  install paths (e.g. non-ASCII characters in path).

## Known bugs
- None in this session.

## Decisions log
| Date | Decision | Reason |
|------|----------|--------|
| 2026-09-21 | Phase 3 regression: added direct-completion (Tier 0) daemon test | coverage gap: every prior daemon test used the blocking-worker (Tier 1+ confirmation) path only |
| 2026-09-21 | Deterministic taint detection: engine checks arg overlap independently of LLM flag | LLM omission (depends_on_untrusted=False) must not suppress taint escalation (docs/03 §8) |
| 2026-09-21 | Added Decision.warn_untrusted + PolicyContext.tainted_fragments | engine-side taint flag replaces LLM-controlled step.depends_on_untrusted in confirmation payload |
| 2026-09-21 | Citation verification rejects out-of-range [n] references | fabricated citations must not be treated as verified (docs/04 §2.1 web_answer) |
| 2026-09-21 | `test_invariant_10` shutdown fixed: `call_soon_threadsafe` instead of `task.cancel()` | `task.cancel()` left a `CancelledError` unhandled in the server thread; thread-safe wake is the correct pattern (same as `_ServerHarness.stop()`) |
| 2026-09-21 | Removed stale `noqa` directives and unused imports from `scripts/phase3_verify*.py` | ruff flagged 21 fixable lint issues accumulated from Phase 3 scripts |
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
| 2026-09-19 | Daemon uses `threading.Event` to bridge LangGraph interrupt→confirm→resume across threads | LangGraph's `interrupt()` is synchronous; the worker thread blocks on an Event while the asyncio loop sends the confirmation to the client and waits for the response |
| 2026-09-19 | `DaemonClient` uses `_get_or_create_loop()` helper instead of `asyncio.get_event_loop()` | `get_event_loop()` raises `RuntimeError` on Python 3.10+ when no loop exists in the current thread; the helper creates one lazily |
| 2026-09-19 | Autostart uses Task Scheduler XML (not Service) | Services run in session 0 and cannot access user desktop/audio; Task Scheduler "At log on" runs in the user session |
| 2026-09-19 | `jarvis chat` (default) connects to daemon; `--no-daemon` runs in-process | Phase 3 adds daemon mode; `--no-daemon` preserved for testing and development |

## Manual tests to run on the PC
```powershell
.venv\Scripts\activate
scripts\check.ps1                  # full gate (expect "all JARVIS checks passed")
jarvis doctor                       # PASS/WARN/FAIL table
jarvis init --non-interactive      # config + API keys + ipc_token
jarvis --version

# Daemon tests (requires two terminals)
jarvis daemon --foreground          # Terminal 1: start daemon
jarvis status                       # Terminal 2: query daemon (should show "running")
jarvis chat                         # Terminal 2: interactive chat via daemon
#   Type: "open notepad" → should auto-execute (Tier 0)
#   Type: "create file test.txt with hello" → should show confirmation
#   Say yes → file created
#   Type: "delete file test.txt" → should show confirmation + password prompt (Tier 2)
# Press Ctrl+C in Terminal 2 → "bye"
jarvis stop                         # Terminal 2: send shutdown to daemon

# Autostart tests
jarvis autostart enable             # Create JarvisAgent task
jarvis autostart status             # Should show "enabled"
jarvis autostart disable            # Remove task

# Password + unlock
jarvis password set                 # Set JARVIS password
jarvis lock                         # Lock session
jarvis unlock                       # Unlock session
```

`jarvis doctor --live` requires a real Groq key + a model name in `config.toml [llm]`.

## Next step
Phase 4: Google open/search, web answers with sources.  Do not modify Phase 3 files unless
a Phase 4 dependency requires it.
