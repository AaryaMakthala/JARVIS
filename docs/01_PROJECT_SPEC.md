# 01 — Project Specification

## 1. Vision

A personal AI agent for Windows that a user can command by **voice or text** to automate everyday PC tasks,
and that tells the user what it is doing and what happened. It is **safe by construction**: an LLM plans,
but a deterministic policy engine decides what may run, and the user confirms anything sensitive.

**Platform:** Windows 10/11 (64-bit), Python 3.11/3.12. Single user, single machine.

## 2. Research claims (what the report defends)

| # | Claim | How it is measured |
|---|-------|--------------------|
| C1 | A deterministic policy engine separated from the LLM prevents unsafe actions even under prompt injection, where LLM-only self-policing fails. | Red-team suite (30+ injection/abuse cases): unsafe-action rate, false-block rate. Compare: LLM-self-policing vs policy engine vs engine + classifier. |
| C2 | A verify → retry → replan loop raises task success rate over plan-once execution. | Ablation on 50-task benchmark: success rate, first-attempt success, extra API calls. |
| C3 | Skill memory (retrieving past successful plans as examples) reduces API calls and latency over repeated use. | Run benchmark 3 rounds; plot calls/task and latency per round with and without memory. |
| C4 | A small trained risk classifier adds recall on sensitive/dangerous actions beyond rules alone. | Precision/recall/F1 per class on held-out set; end-to-end effect on red-team suite. |

## 3. User experience

```powershell
pipx install jarvis-agent   # or pip install -e . in dev
jarvis init                 # wizard: API keys, JARVIS password, voice choice, contacts, autostart
jarvis on                   # start listening (mic + wake word)
jarvis chat                 # terminal chat to the running daemon
jarvis off                  # stop listening (daemon stays for terminal use)
jarvis stop                 # stop the daemon completely
```

After `jarvis init` + autostart, JARVIS starts at every Windows login (hidden, in the user's session).
It answers in text (terminal) and by voice (if enabled) and shows a visible indicator when the mic is live.

### CLI reference

| Command | Behaviour |
|---------|-----------|
| `jarvis init` | Interactive setup wizard (idempotent, can be re-run). Stores keys in Windows Credential Manager via `keyring`. |
| `jarvis doctor` | Checks Python version, OS, keyring access, API key presence, LLM reachability, mic, TTS, Playwright, WhatsApp desktop app, Defender availability. Prints PASS/WARN/FAIL with fixes. |
| `jarvis daemon [--foreground]` | Runs the background service (used by autostart and dev). |
| `jarvis chat` | Terminal client to the daemon; supports confirmations and password prompts. |
| `jarvis run "<command>"` | One-shot command via daemon (or in-process with `--no-daemon`). |
| `jarvis on` / `off` / `stop` / `status` | Control mic/wake word, daemon lifecycle, show state. |
| `jarvis password set` / `change` | Set/change the JARVIS password (Argon2id hash in keyring). |
| `jarvis lock` / `unlock` | Manually lock/unlock the Tier-2 session. |
| `jarvis contacts add|list|remove` | Manage WhatsApp contacts (`name → +countrycode number`). |
| `jarvis autostart enable|disable|status` | Task Scheduler "At log on" task. |
| `jarvis audit [--fix-suggestions]` | Read-only security/performance/updates/self report to Markdown. |
| `jarvis skills list|show|delete` | Inspect skill memory. |
| `jarvis undo` | Restore the most recent deleted files from the undo log/Recycle Bin (best effort). |
| `jarvis benchmark run|report` | Run benchmark tasks and produce metrics (dev/eval use). |
| `jarvis --version` | Print installed version. |
| `jarvis models download` | Download first-run assets (whisper, wake word, Piper voice, embeddings, classifier) with checksums; safe to re-run. |
| `jarvis logs tail|path` | Show recent redacted log lines / log location. |
| `jarvis backup` | Zip contacts, config, and memory DB (never secrets). |
| `jarvis reset --memory|--all` | Clear learned memory or all local data (with confirmation). |
| `jarvis uninstall` | Stop daemon, remove autostart, optionally remove keyring entries and data. |

Global flags: `--dry-run` (tools describe what they would do, execute nothing), `--verbose`.

## 4. Features and acceptance criteria

### F1 Command understanding and planning
- Natural-language command → validated `Plan` (list of steps, each: tool, args, rationale, expectation).
- Planner can only use registered tools. Invalid plans are rejected and retried once with the validation error.
- If the request is ambiguous ("delete the file"), the planner returns a clarification question instead of guessing.
- **Accept:** 20 sample commands produce valid plans; 0 reference non-existent tools (with `FakeLLM` and with the real LLM).

### F2 Safety and confirmation
- Four tiers (see `03_SECURITY_AND_POLICY.md`). Confirmation shows *exactly* what will happen.
- Confirmation by terminal (yes/no), by voice (Tier 1 only), or tkinter dialog when triggered by voice.
- **Accept:** every Tier ≥1 step pauses for confirmation (LangGraph `interrupt`); Tier 3 never runs; all invariants tested.

### F3 File operations
- Create/overwrite/append/read/list/delete within allowed roots; Recycle-Bin delete with preview; folders require typing the folder name; undo log.
- **Accept:** path-traversal, symlink, and protected-path tests all pass; deleting 3 files shows a preview and requires Tier 2 unlock.

### F4 JARVIS password and unlock session
- Argon2id hash; unlock TTL (default 5 min); relock on idle, on `jarvis lock`, or spoken "lock Jarvis".
- Brute-force protection (lockout with backoff). Password entered only in terminal or tkinter dialog, never by voice, never sent to the LLM.
- **Accept:** wrong password ×5 → lockout; unlock expires; password absent from logs/checkpoints (tested by grep).
- **Note:** JARVIS cannot unlock the Windows lock screen (secure desktop). For that use Windows Hello / Dynamic Lock. `lock_computer` calls `LockWorkStation`.

### F5 Google search and current information
- "Open Google and search X" → opens `https://www.google.com/search?q=…` in the default browser (no automation).
- "What's the latest on X?" → web search (DuckDuckGo via `ddgs`, or Tavily if key set) → LLM answer with numbered source links. Results treated as untrusted.
- **Accept:** answer contains ≥1 source URL taken from the results; an injected instruction inside a result does not change the plan.

### F6 Notepad dictation
- "Take notes" → opens Notepad, starts loop: mic → VAD → faster-whisper → paste into Notepad (verified as foreground).
- Voice commands inside dictation: "new line", "new paragraph", "save as <name>", "stop dictation".
- **Accept:** dictating 5 sentences produces the right text; if Notepad loses focus, dictation pauses and warns.

### F7 WhatsApp single message
- Contacts from `contacts.json` only. Confirmation shows `To: Rahul (+91…)  Message: "…"`.
- Opens `whatsapp://send?phone=<digits>&text=<urlencoded>`, verifies the chat window, presses Enter. If the chat cannot be verified → **do not send**; leave the draft and tell the user.
- Fallback: WhatsApp Web via Playwright persistent profile (QR once).
- **Accept:** unknown contact is refused; message is never sent without Tier 2 confirmation; no bulk sends (max 1 recipient per step, rate limit per hour).

### F8 Threat check and audit
- `jarvis audit` produces a Markdown report: Security (Defender, firewall, BitLocker, updates, listening ports + processes, startup entries, scheduled tasks), Performance (RAM, disk, heavy startup apps), Updates (winget), and JARVIS self-review (from its own logs/failures).
- Optional VirusTotal hash lookup (hash only).
- Read-only. Proposed fixes are text only in v1; any later "apply fix" is Tier 2 from a whitelist.
- **Accept:** report generated on a normal (non-admin) user account, with graceful "needs admin" notes.

### F9 Learning
- Successful plans → `skills` table with embeddings; retrieved by similarity as **few-shot examples** for the planner (never auto-executed).
- Failures logged; planner receives "avoid" hints.
- User preferences (default browser, favourite apps) stored in `preferences`.
- **Accept:** second run of the same command uses ≤ the API calls of the first (measured in benchmark).

### F10 Voice
- openWakeWord ("hey jarvis") → faster-whisper (tiny/base, CPU) → LangGraph → Piper (fallback SAPI) speech.
- Visible listening indicator; mic can be disabled instantly (`jarvis off`, spoken "stop listening").
- **Accept:** wake word triggers < 1 false trigger per hour in normal use; end-to-end voice command works.

## 5. Non-functional requirements

| Area | Target |
|------|--------|
| Latency | Tier 0 command with cached plan < 3 s excluding LLM; typical LLM planning < 5 s |
| Resource use | Idle daemon (no voice) < 200 MB RAM; with voice models < 800 MB; CPU idle < 3% |
| Reliability | ≥ 90% success on benchmark (report the real number honestly) |
| Privacy | Audio never leaves the PC; only text prompts and tool results go to the LLM API; no screenshots in v1 |
| Observability | JSON-lines log, task log table, per-task API-call and token counts |
| Robustness | Daemon survives tool crashes; each task cancellable; one task at a time (queue others) |
| Portability | Non-Windows tests run on any OS (mock Windows APIs) |

## 6. Non-goals (v1)

Lock-screen unlock, generic shell access, bulk messaging, payments, browser password access, self-installing
tools, cloud sync, multi-user, GUI beyond confirmation dialogs.

## 7. Stretch goals (only after everything above works and is benchmarked)

- **Tool builder** node: when no tool exists, the LLM writes code + test + dependency list → AST static check →
  isolated venv install from allowlist → subprocess test with timeout → user approval → saved to `tools/generated/`.
  (Everything generated is Tier 2 at minimum.)
- **MCP layer**: expose tools via Model Context Protocol (`langchain-mcp-adapters`).
- **Vision fallback**: screenshot → Gemini for UI cases accessibility APIs can't handle.
- Custom "Hey Jarvis" wake word trained via openWakeWord.
