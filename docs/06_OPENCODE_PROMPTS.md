# 06 — OpenCode Prompts (copy and paste)

## How to work with OpenCode

1. Open the project folder in VS Code, run `opencode` in the terminal. It loads `AGENTS.md` automatically.
2. Start each phase with **planning first** (OpenCode's plan mode, if your version has it — usually toggled with `Tab`), read the plan, fix mistakes, then let it build.
3. **One phase per session.** Start a fresh session for each phase to keep context clean; `AGENTS.md` and `PROGRESS.md` carry the memory.
4. After each session: run `scripts/check.ps1` yourself, try the manual tests, read the diff, then `git commit`.
5. If OpenCode goes wrong, use its undo feature or `git restore`. Never let it "fix" a failing safety test by weakening the policy.
6. Ask it to explain anything you don't understand (`Explain policy/engine.py line by line`). You must be able to defend it.

Keep API keys out of prompts and out of files. Keys go through `jarvis init` into Windows Credential Manager.

---

## Prompt 0 — Phase 0 (skeleton)

```
Read AGENTS.md, PROGRESS.md, docs/01_PROJECT_SPEC.md, docs/02_ARCHITECTURE.md, docs/09_SETUP_AND_DEPENDENCIES.md and the "Phase 0" section of docs/05_BUILD_PLAN.md.

Task: implement Phase 0 only.
First, print a short plan (files to create, dependencies, tests, risks) and wait for my "go".
Then implement: pyproject.toml (src layout, console script "jarvis", extras dev/voice/browser/ml), config.py, secrets.py (keyring wrapper), logging_setup.py with redaction filter, platform_guard.py, cli.py (init wizard, doctor, status stub), llm/client.py (LLMClient protocol, GroqClient, FakeLLM, usage tracking), and empty module skeletons with docstrings for the rest of the layout in docs/02_ARCHITECTURE.md.
Do not hardcode model names: read them from config with placeholders, and tell me where I must set them after checking the provider docs.
scripts/verify_env.py already exists: run it (python scripts/verify_env.py --phase 1) and tell me what fails; extend it only when you add new dependencies.
The LLM client must fall back from JSON-schema structured output to plain JSON + Pydantic validation + one repair retry, and log which mode was used.
Verify the Groq SDK and keyring APIs from the installed packages before using them.
Add tests (redaction test with a planted key, config loading, FakeLLM). Windows-only tests must be skipped on other OSes.
Finish by running ruff and pytest, updating PROGRESS.md, and listing manual tests for me.
```

## Prompt 1 — Phase 1 (planner, policy, confirmation)

```
Read AGENTS.md, PROGRESS.md, docs/02_ARCHITECTURE.md (sections 4, 5, 7), docs/03_SECURITY_AND_POLICY.md, docs/04_TOOLS_SPEC.md (sections 1, 2.1, 2.2 create_file, 3) and the "Phase 1" section of docs/05_BUILD_PLAN.md.

Task: implement Phase 1 only.
Plan first, wait for "go". Then build: tool framework (ToolSpec, ToolResult, registry with the forbidden-name check), policy engine (tiers, rules, first version of paths.py), tools open_app, open_url, google_search, system_info, create_file (each with verify and dry-run), agent state/nodes/graph with SqliteSaver, planner prompt with structured output and one repair retry, validate node, policy_gate using interrupt(), act with action_hash check, basic verify with retries, runner, and `jarvis chat --no-daemon`.
Check the installed LangGraph version's interrupt/Command/checkpointer API before coding; do not guess.
No side effects before interrupt() in any node.
Tests: FakeLLM graph tests, invariant tests 1, 2, 3, 6, 12, 13, and a checkpoint-resume integration test.
Finish with ruff, mypy on policy, pytest, PROGRESS.md update, and manual test commands.
```

## Prompt 2 — Phase 2 (files, password)

```
Read AGENTS.md, PROGRESS.md, docs/03_SECURITY_AND_POLICY.md (sections 3-7), docs/04_TOOLS_SPEC.md (section 2.2) and "Phase 2" in docs/05_BUILD_PLAN.md.

Task: implement Phase 2 only.
Plan first. Then: complete policy/paths.py (resolve_safe, protected paths, allowed roots, junction/reparse-point handling, UNC/device/ADS rejection), file tools (list_dir, read_file, create_file overwrite handling, append_file, delete_path with send2trash and undo log, undo_last_delete), policy/unlock.py (Argon2id, TTL, lockout with backoff, injectable clock for tests), CLI `password set|change`, `lock`, `unlock`, typed folder-name confirmation, delete preview text, Tier 2 flow in the terminal loop.
Hypothesis property tests for path safety (.., mixed slashes, case, trailing dots, symlinks/junctions).
Invariant tests 4, 5, 7, 11, 14. Verify that no password or key appears in checkpoints or logs by grepping the DB and log file in a test.
Never use os.remove or shutil.rmtree in delete code paths.
Finish with checks, PROGRESS.md, manual tests.
```

## Prompt 3 — Phase 3 (daemon)

```
Read AGENTS.md, PROGRESS.md, docs/02_ARCHITECTURE.md (sections 6, 8), docs/03_SECURITY_AND_POLICY.md (sections 2, 7) and "Phase 3" in docs/05_BUILD_PLAN.md.

Task: implement Phase 3 only.
Plan first. Build daemon/protocol.py (Pydantic messages, 64 KB limit), daemon/server.py (asyncio TCP on 127.0.0.1, random port, token auth with hmac.compare_digest, auth-failure throttling, task queue max 5, worker thread for graph runs, cancel token), daemon/client.py, CLI commands chat/run/status/on/off/stop, IPC token generated at init and stored in keyring, confirmation over IPC including password (terminal getpass; tkinter dialog for voice-initiated confirmations), and daemon/autostart.py using a Task Scheduler "At log on" task running pythonw -m jarvis daemon (NOT a Windows Service; explain why in a comment).
Password is verified by the daemon via UnlockManager and never passed to the graph.
Tests: protocol validation, auth failure, token missing, queue/cancel, invariant test 10, and an integration test with an in-process server and FakeLLM.
Finish with checks, PROGRESS.md, manual tests (including reboot autostart test).
```

## Prompt 4 — Phase 4 (web)

```
Read AGENTS.md, PROGRESS.md, docs/03_SECURITY_AND_POLICY.md (section 8, 12), docs/04_TOOLS_SPEC.md (section 2.1) and "Phase 4" in docs/05_BUILD_PLAN.md.

Task: implement Phase 4 only.
Plan first. Build the web_answer tool (ddgs search, optional Tavily if a key exists), untrusted-data wrapping, citation verification, tainted flags on results, taint propagation (depends_on_untrusted overlap check) and the warning banner in confirmations, URL scheme allowlist for open_url, and start benchmarks/redteam.yaml with at least 10 cases.
Verify the current ddgs package API before using it.
Tests use canned search results, including injection strings. Invariant test 8.
Finish with checks, PROGRESS.md, manual tests.
```

## Prompt 5 — Phase 5 (voice)

```
Read AGENTS.md, PROGRESS.md, docs/01_PROJECT_SPEC.md (F6, F10), docs/04_TOOLS_SPEC.md (section 2.3, 2.5) and "Phase 5" in docs/05_BUILD_PLAN.md.

Task: implement Phase 5 only.
Plan first. Build voice/wake.py (openWakeWord, ONNX backend), vad.py, stt.py (faster-whisper tiny/base int8 CPU), tts.py (Piper with a Windows SAPI fallback), loop.py, jarvis on/off, listening indicator, dictation tools, focus-verified type_text, voice confirmation for Tier 1 only (Tier 2 must use terminal/dialog).
Hide all audio and window access behind interfaces so tests use fakes; real-hardware tests get @pytest.mark.voice.
Check installed package APIs for openWakeWord, faster-whisper and Piper before coding; note any Windows install problems in PROGRESS.md and use the fallback.
Invariant test 15. Finish with checks, PROGRESS.md, manual tests including false-trigger measurement steps.
```

## Prompt 6 — Phase 6 (WhatsApp)

```
Read AGENTS.md, PROGRESS.md, docs/04_TOOLS_SPEC.md (section 2.6), docs/03_SECURITY_AND_POLICY.md and "Phase 6" in docs/05_BUILD_PLAN.md.

Task: implement Phase 6 only.
Plan first. Build `jarvis contacts add|list|remove` with number validation, whatsapp_send with the exact flow in the spec (single recipient, masked number in summary, verify chat before Enter, fail closed if unverifiable, rate limit), and a WindowProvider interface so tests can fake the WhatsApp window.
Playwright WhatsApp Web fallback is optional: implement only if the desktop route works and I ask for it.
Invariant test 9. Finish with checks, PROGRESS.md, manual tests (only to my own test contact).
```

## Prompt 7 — Phase 7 (audit)

```
Read AGENTS.md, PROGRESS.md, docs/04_TOOLS_SPEC.md (sections 2.4, 2.7) and "Phase 7" in docs/05_BUILD_PLAN.md.

Task: implement Phase 7 only.
Plan first. Build audit/checks.py (each check returns Finding objects; per-check timeouts; graceful "needs admin"), audit/report.py (Markdown report), `jarvis audit`, and tools audit_run, defender_status, defender_quick_scan.
All PowerShell/winget commands must be module-level constants run without shell=True; no argument may come from the LLM or user text.
Tests use recorded fixture outputs so they run on any OS, plus a test that all commands come from the whitelist and none is a write operation.
Finish with checks, PROGRESS.md, manual tests.
```

## Prompt 8 — Phase 8 (memory, classifier, replan)

```
Read AGENTS.md, PROGRESS.md, docs/02_ARCHITECTURE.md (sections 5, 9), docs/03_SECURITY_AND_POLICY.md (sections 4, 8), docs/08_RISK_CLASSIFIER.md and "Phase 8" in docs/05_BUILD_PLAN.md.

Task: implement Phase 8 only (the classifier training happens in Colab with me; you only build the inference wrapper and integration, plus dataset/training scaffolding under training/).
Plan first. Build memory/* (local ONNX embeddings via fastembed or sentence-transformers, skills save/retrieve, failures, preferences), memory_retrieve/memory_save nodes, `jarvis skills`, the replan node (caps enforced), and ml/ risk classifier wrapper that can only RAISE a tier.
Skills are examples for the planner, never auto-executed; save only fully verified, non-tainted plans.
Tests: retrieval ranking, skill rules, replan caps, and a test proving the classifier cannot lower a tier.
Finish with checks, PROGRESS.md, manual tests.
```

## Prompt 9 — Phase 9 (benchmark)

```
Read AGENTS.md, PROGRESS.md, docs/07_TESTING_AND_BENCHMARK.md, docs/10_REPORT_AND_VIVA.md and "Phase 9" in docs/05_BUILD_PLAN.md.

Task: implement Phase 9 only.
Plan first. Build benchmarks/runner.py (sandboxed folder, dry-run and real modes, repeats, config flags for ablations), analyze.py (CSV summaries, mean and 95% CI, charts as PNG), populate tasks.yaml (>= 50 tasks) and redteam.yaml (>= 30 cases) following the categories in the doc, and `jarvis benchmark run|report`.
Record API calls, tokens, latency, peak RAM (psutil), confirmations, success, first-attempt success, unsafe actions.
Never run destructive benchmark tasks outside the sandbox folder.
Finish with checks, PROGRESS.md, and the exact commands to reproduce every table in the report.
```

---

## Prompt 10 — Phase 10 (packaging and deployment)

```
Read AGENTS.md, PROGRESS.md, docs/11_DEPLOYMENT_AND_PACKAGING.md, docs/12_DEPENDENCY_VERIFICATION.md and "Phase 10" in docs/05_BUILD_PLAN.md.

Task: implement Phase 10 only.
Plan first. Build: wheel packaging with package data and `jarvis --version`; scripts/install.ps1 and uninstall.ps1 (idempotent, no admin, verbose; verify current pipx and extras syntax from docs before writing); autostart via a Task Scheduler XML definition (current user, least privilege, restart on failure, ignore new instance) with `jarvis autostart enable|disable|status`; pythonw-safe startup (redirect stdout/stderr to the log before noisy imports, sys.excepthook and threading.excepthook, single-instance mutex); stale-confirmation cleanup; DB and config migrations with backups; `jarvis models download`, `logs`, `backup`, `reset`, `uninstall`; voice loop resilience for device loss and sleep/resume.
Tests: migrations, XML generation (validate it, do not run schtasks in default tests), mutex/single instance, stale-confirmation expiry, uninstall dry-run. Windows-only tests for the real task get @pytest.mark.windows_only.
Give me the exact clean-machine test steps (Windows Sandbox or second account) and the release checklist commands. Update PROGRESS.md.
```

---

## Utility prompts

**Session start (any phase)**
```
Read AGENTS.md and PROGRESS.md. Summarise where the project stands and what the next unchecked item in docs/05_BUILD_PLAN.md is. Do not change any files yet.
```

**Security review (run after Phases 2, 4, 6, 8)**
```
Act as a security reviewer. Read docs/03_SECURITY_AND_POLICY.md and all code under src/jarvis/policy, tools, daemon and agent. Look for: ways an LLM output or untrusted text could lower a tier or skip confirmation, path-resolution gaps, secrets in logs/state, TOCTOU between confirmation and execution, unauthenticated IPC access, and any place that uses shell=True, eval, exec, os.remove or shutil.rmtree. Report findings with file and line, severity, and a proposed fix. Do not edit files.
```

**Bug fix**
```
Bug: <describe> . Steps to reproduce: <commands>. Expected: <..>. Actual: <..>. Logs: <paste redacted lines>.
First write a failing test that reproduces it, then fix the code, then run the full checks. Do not weaken any safety rule to fix it.
```

**Explain for viva**
```
Explain <file or module> as if preparing me for a viva: purpose, data flow, key design decisions and alternatives rejected, failure modes, how it is tested, and 5 likely examiner questions with strong answers.
```

**Refactor guard**
```
Refactor <module> for readability only. Behaviour must not change. Run all tests before and after and show that the results are identical.
```
