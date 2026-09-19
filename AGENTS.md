# AGENTS.md — JARVIS (Windows desktop AI agent)

You are helping build **JARVIS**, a voice- and text-controlled AI agent that automates tasks on a
Windows 10/11 PC. This is a **final-year engineering project**: it must be correct, safe, measurable,
and fully understood by its author (who will defend it in a viva). Favour clear, boring, testable code
over clever code.

## 1. Read these first (in order)

| File | Purpose |
|------|---------|
| `docs/01_PROJECT_SPEC.md` | What the product does, CLI, features, acceptance criteria |
| `docs/02_ARCHITECTURE.md` | Components, LangGraph design, state schema, daemon protocol, storage |
| `docs/03_SECURITY_AND_POLICY.md` | Permission tiers, threat model, invariants (**non-negotiable**) |
| `docs/04_TOOLS_SPEC.md` | Every tool: args, tier, verification, failure modes |
| `docs/05_BUILD_PLAN.md` | Phases 0-9 with deliverables and acceptance tests |
| `docs/07_TESTING_AND_BENCHMARK.md` | Test strategy, benchmark, ablation |
| `docs/11_DEPLOYMENT_AND_PACKAGING.md` | Packaging, install, autostart, upgrades (read in Phase 3 and Phase 10) |
| `docs/12_DEPENDENCY_VERIFICATION.md` | Verified findings, fallbacks, licences (read before choosing/using any library) |
| `scripts/verify_env.py` | Environment verifier: run it after every dependency change; extend it, don't replace it |
| `PROGRESS.md` | Current status. **Update it at the end of every session.** |

Other docs (`06_OPENCODE_PROMPTS`, `08_RISK_CLASSIFIER`, `09_SETUP_AND_DEPENDENCIES`,
`10_REPORT_AND_VIVA`) are for the human; read them only if the task references them.

## 2. Tech stack

- Python **3.11 or 3.12**, 64-bit, Windows 10/11 is the primary target.
- Packaging: `pyproject.toml`, `src/` layout, package name `jarvis`, console script `jarvis`.
- Agent framework: **LangGraph** (+ `langgraph-checkpoint-sqlite`), LangChain core only where needed.
- LLM: Groq (main), Gemini (optional fallback), both behind our own `LLMClient` interface.
- Validation: **Pydantic v2** for every boundary (LLM output, tool args, IPC messages, config).
- CLI: `typer` + `rich`. Config paths: `platformdirs`. Secrets: `keyring`. Password hashing: `argon2-cffi`.
- Desktop: `pyautogui`, `pywinauto`, `pywin32`, `psutil`, `send2trash`, `pyperclip`.
- Voice (optional extra): `openwakeword`, `faster-whisper`, `sounddevice`, Piper (fallback: Windows SAPI).
- Tests: `pytest`, `pytest-asyncio`, `hypothesis`. Lint/format: `ruff`. Types: `mypy` (strict for `policy/`).

Do **not** pin library versions from memory. Install, check what resolves, and pin in the lock/requirements.
Do **not** hardcode LLM model names from memory: they change. Put them in config and verify against
the provider's current docs (see `docs/09_SETUP_AND_DEPENDENCIES.md`).

## 3. Commands

```powershell
# one-time
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"

# every change
ruff check . ; ruff format .
mypy src/jarvis/policy
pytest -q                                # unit tests, no Windows GUI, no network, no API keys
pytest -q -m windows_only                # real Windows tests (only on the dev PC)
pytest -q -m "not slow and not voice"    # default CI set

# run
jarvis doctor
jarvis chat
jarvis daemon --foreground               # dev mode
```

## 4. Repository layout (authoritative version in `docs/02_ARCHITECTURE.md`)

```
src/jarvis/
  cli.py  config.py  secrets.py  logging_setup.py
  llm/        client.py  prompts.py
  agent/      state.py  graph.py  runner.py  nodes/
  policy/     tiers.py  engine.py  paths.py  unlock.py
  tools/      base.py  registry.py  apps.py  files.py  web.py  keyboard.py  whatsapp.py  dictation.py  system.py
  daemon/     protocol.py  server.py  client.py  autostart.py
  voice/      wake.py  stt.py  tts.py  vad.py
  memory/     db.py  skills.py  failures.py  prefs.py  embeddings.py
  audit/      checks.py  report.py
tests/  unit/  integration/  windows_only/
benchmarks/  tasks.yaml  runner.py
docs/
```

## 5. Safety invariants (never violate; never "simplify" away)

1. **The LLM proposes; only the deterministic policy engine decides.** No LLM output can lower a tier.
2. **Unknown tool = rejected.** The planner may only call tools that exist in the registry.
3. **No generic shell/PowerShell/`eval`/`exec` tool exists in v1.** Audit uses hard-coded, whitelisted commands only.
4. **Tier 3 is blocked in code**, not by prompt: payments, changing the Windows password, disabling
   Defender/firewall, deleting system folders, reading saved browser passwords.
5. **All paths are resolved to absolute real paths** (`Path.resolve(strict=False)` + symlink/junction check)
   *before* any policy decision or file operation.
6. **Deletes go to the Recycle Bin** (`send2trash`), never permanent, and are written to the undo log.
7. **Confirmations are bound to the exact action** (hash of tool + normalised args). The act node
   re-checks the hash before executing.
8. **Passwords and API keys never enter LangGraph state, checkpoints, logs, or LLM prompts.**
   Password verification happens in the daemon layer, before the graph is resumed.
9. **Web page text, search results, file contents, and window text are untrusted data**, never instructions.
   They are wrapped in delimiters in prompts and flagged `tainted` in state.
10. **Fail closed.** If verification of a sensitive step (e.g. WhatsApp chat identity) is impossible, do not proceed.
11. **The IPC socket binds to `127.0.0.1` only and requires a token** (constant-time compare).
12. **Audio never leaves the machine.** Only text goes to LLM APIs.

Every invariant above must have at least one automated test in `tests/unit/test_invariants.py`.

## 6. Coding rules

- Type hints everywhere; `from __future__ import annotations`; docstrings on public functions.
- No global mutable state. Pass a `ToolContext` / `AppContext` object.
- No bare `except:`. Catch specific exceptions; log with context; return a `ToolResult(ok=False, error=...)`.
- Every tool returns `ToolResult` (see `docs/04_TOOLS_SPEC.md`) and never raises to the graph.
- Every tool has: a Pydantic args model, a declared `base_tier`, a `verify()` method, and a `dry_run` behaviour.
- Side effects only inside tool functions. Graph nodes stay pure apart from calling LLM/tools/DB.
- **No side effects before `interrupt()` in a node** (LangGraph re-runs the node from the top on resume).
- Windows-only imports (`pywinauto`, `win32*`, `winreg`, `ctypes.windll`) go behind `platform_guard`
  so tests import on any OS.
- Keep functions under ~50 lines. Prefer small modules to big ones.
- Log with the `logging` module (JSON lines to file). Never `print` in library code (use `rich` in `cli.py`).
- Redact secrets, passwords, phone numbers (keep last 2 digits), and message bodies at INFO level.
- The daemon runs under `pythonw` (no console): never rely on `print`/stdout/stderr in daemon code paths; log to file.
- Every new third-party dependency needs: a reason, an entry in `pyproject.toml`, an entry in `scripts/verify_env.py`, a licence check, and a fallback if it is optional. Never add a package that downloads or executes code at runtime.
- LLM structured output must handle the "model doesn't support JSON schema" error by falling back to plain JSON + Pydantic validation.

## 7. How you (the coding agent) must work

1. **One phase at a time.** Do exactly the phase the user names from `docs/05_BUILD_PLAN.md`. Do not start the next one.
2. **Plan first.** Before editing, print a short plan: files to create/change, tests to add, risks. Then implement.
3. **Tests with every module.** No module is "done" without tests. Use a `FakeLLM` for deterministic graph tests.
4. **Verify library APIs.** LangGraph, Groq, openWakeWord, faster-whisper, pywinauto change often. Read the
   installed package source or its docs before using an API; do not guess signatures.
5. **Ask when the spec is ambiguous or contradictory** instead of inventing behaviour. Note the decision in `PROGRESS.md`.
6. **Never weaken the policy engine** to make a test or demo pass. Fix the test or the plan instead.
7. **Small commits.** Suggest a Conventional Commit message (`feat(policy): ...`) at the end; the user commits.
8. **Finish every session** by: running ruff + pytest, updating `PROGRESS.md`, and listing manual tests the user
   should try on their PC (exact commands).
9. If something cannot be done safely or reliably on Windows, say so and propose the closest safe alternative.
   (Example: typing into the Windows lock screen is impossible for normal programs; do not attempt it.)

## 8. Definition of done (per phase)

- [ ] Acceptance criteria in `docs/05_BUILD_PLAN.md` for that phase all pass.
- [ ] `ruff check`, `mypy src/jarvis/policy`, and `pytest -q` are green.
- [ ] New invariants/tests added; no skipped tests without a reason string.
- [ ] `PROGRESS.md` updated (what works, what's fragile, known bugs, next step).
- [ ] Manual smoke-test list provided to the user.

## 9. Out of scope for v1 (do not build unless asked)

Windows Credential Provider / lock-screen unlock, generic shell tool, bulk WhatsApp, payments,
self-writing tools that install packages (stretch goal only), cloud sync, mobile app, GUI beyond small
tkinter confirmation dialogs.
