# 02 — Architecture

## 1. Big picture

```
Windows logon ──► Task Scheduler (At log on) ──► pythonw -m jarvis daemon   (hidden, user session)
                                                        │
              ┌─────────────────────────────────────────┼──────────────────────────────┐
              │                                         │                              │
      voice loop (optional)                   IPC server 127.0.0.1:<port>      Task queue + runner
  wake → VAD → STT → text ─────────────►      NDJSON + token auth  ◄────────── jarvis chat / run / on / off
              │                                         │
              └────────────────────────────────────────►│
                                                        ▼
                                     LangGraph agent (SQLite checkpointer)
     intake → memory_retrieve → plan → validate → policy_gate ⇄ (interrupt: confirm)
                                                        │
                                                        ▼
                                  act → verify ─┬─► next step ─► act …
                                                ├─► retry
                                                ├─► replan → validate → policy_gate …
                                                └─► respond (text + TTS) → memory_save
```

Why a **daemon** and not a Windows Service: services run in session 0 and cannot control the user's desktop,
mouse, windows, or audio. A Task Scheduler "At log on" task runs inside the user's session.

## 2. Repository layout (authoritative)

```
jarvis/
├─ AGENTS.md  README.md  PROGRESS.md  pyproject.toml  .gitignore  .env.example
├─ docs/
├─ src/jarvis/
│  ├─ __init__.py  __main__.py
│  ├─ cli.py                 # typer app
│  ├─ config.py              # Settings (pydantic-settings), paths (platformdirs)
│  ├─ secrets.py             # keyring wrapper: get/set/delete; never logs values
│  ├─ logging_setup.py       # JSONL file logging + redaction filter
│  ├─ platform_guard.py      # is_windows(), require_windows(), lazy imports
│  ├─ llm/
│  │   ├─ client.py          # LLMClient protocol, GroqClient, GeminiClient, FakeLLM (tests)
│  │   └─ prompts.py         # planner/replanner/answer prompts (templates)
│  ├─ agent/
│  │   ├─ state.py           # AgentState + Pydantic models (Plan, Step, StepResult, Decision)
│  │   ├─ graph.py           # build_graph(ctx) → compiled graph
│  │   ├─ runner.py          # run_task(), resume_task(), cancel_task()
│  │   └─ nodes/             # intake.py memory_retrieve.py plan.py validate.py policy_gate.py
│  │                         # act.py verify.py replan.py respond.py memory_save.py
│  ├─ policy/
│  │   ├─ tiers.py           # Tier enum, Decision model
│  │   ├─ engine.py          # PolicyEngine.decide(step, ctx) → Decision
│  │   ├─ paths.py           # resolve_safe(), protected roots, allowed roots
│  │   ├─ unlock.py          # UnlockManager (Argon2id, TTL, lockout)
│  │   └─ rules.py           # declarative rules table (tool/arg patterns → tier)
│  ├─ tools/
│  │   ├─ base.py            # ToolSpec, ToolResult, ToolContext, verify helpers
│  │   ├─ registry.py        # ToolRegistry (register, get, catalogue_for_llm)
│  │   ├─ apps.py            # open_app
│  │   ├─ files.py           # create/append/read/list/delete/undo
│  │   ├─ web.py             # open_url, google_search, web_answer
│  │   ├─ keyboard.py        # type_text (focus-verified)
│  │   ├─ dictation.py       # start/stop dictation
│  │   ├─ whatsapp.py        # whatsapp_send
│  │   └─ system.py          # lock_computer, system_info, audit tools
│  ├─ daemon/
│  │   ├─ protocol.py        # Pydantic message models (client↔server)
│  │   ├─ server.py          # asyncio TCP server, auth, task queue
│  │   ├─ client.py          # DaemonClient used by CLI
│  │   └─ autostart.py       # Task Scheduler create/delete/status
│  ├─ voice/                 # wake.py stt.py tts.py vad.py loop.py
│  ├─ memory/                # db.py skills.py failures.py prefs.py embeddings.py
│  ├─ audit/                 # checks.py report.py
│  └─ ml/                    # risk_classifier inference wrapper (Phase 8)
├─ tests/  unit/  integration/  windows_only/
├─ benchmarks/  tasks.yaml  redteam.yaml  runner.py  analyze.py
├─ training/                 # Colab notebooks + dataset for the risk classifier (Phase 8)
└─ scripts/                  # helper scripts (dev only)
```

## 3. Data locations (via `platformdirs`; default `%LOCALAPPDATA%\jarvis`)

| Item | Path | Notes |
|------|------|-------|
| Settings | `config.toml` | Non-secret only (models, TTL, ports, allowed roots, app map) |
| Contacts | `contacts.json` | `{ "rahul": {"display": "Rahul", "phone": "+91XXXXXXXXXX"} }` |
| Memory DB | `memory.db` (SQLite) | skills, failures, preferences, task_log |
| Checkpoints | `checkpoints.db` (SQLite) | LangGraph state. **Contains no secrets.** |
| Logs | `logs/jarvis.jsonl` (rotating) | Redacted |
| Undo log | `undo_log.jsonl` | Delete operations |
| Reports | `reports/audit-YYYYMMDD-HHMM.md` | |
| Playwright profile | `browser_profile/` | WhatsApp Web fallback |
| Secrets | Windows Credential Manager via `keyring` | service name `jarvis`: `groq_api_key`, `gemini_api_key`, `tavily_api_key`, `ipc_token`, `password_hash`, `virustotal_api_key` |
| Daemon runtime | `daemon.json` | `{pid, port, started_at}` (no token here) |

## 4. State schema (`agent/state.py`)

```python
from __future__ import annotations
from typing import Any, Literal, TypedDict
from pydantic import BaseModel, Field


class Step(BaseModel):
    id: str  # "s1", "s2", ...
    tool: str  # must exist in registry
    args: dict[str, Any]  # validated against the tool's args model in `validate`
    rationale: str  # short, user-visible
    expect: str = ""  # human-readable success condition (used by verify + LLM replan)
    depends_on_untrusted: bool = False  # set by the plan node if args derive from web/file text


class Plan(BaseModel):
    goal: str
    steps: list[Step] = Field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: str | None = None


class Decision(BaseModel):
    step_id: str
    tier: int  # 0..3
    allowed: bool  # False when Tier 3 or hard rule hit
    needs_confirm: bool
    needs_unlock: bool
    needs_typed_confirmation: str | None = None  # e.g. folder name to type
    reasons: list[str]
    summary: str  # exact text shown to the user
    action_hash: str  # sha256(tool + canonical(args)); binds confirmation to this action


class StepResult(BaseModel):
    step_id: str
    ok: bool
    output: str = ""  # short text for the LLM/user (truncated)
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    verified: bool | None = None
    tainted: bool = False  # output contains untrusted external text
    duration_ms: int = 0


class AgentState(TypedDict, total=False):
    task_id: str
    source: Literal["terminal", "voice", "benchmark"]
    user_input: str
    memory_context: list[dict]  # retrieved skills/failures/preferences (examples only)
    plan: Plan | None
    step_index: int
    decisions: dict[str, Decision]  # by step_id
    approved_hashes: list[str]  # action hashes the user approved (this task only)
    results: list[StepResult]
    retry_count: int  # for current step
    replan_count: int
    api_calls: int
    tokens: int
    cancelled: bool
    final_answer: str | None
    error: str | None
```

**Never** add password, API keys, or raw audio to state. State is checkpointed to SQLite in plaintext.

## 5. Graph nodes (`agent/nodes/`)

| Node | Responsibility | Notes |
|------|----------------|-------|
| `intake` | Normalise text, assign `task_id`, handle meta-commands ("cancel", "stop") | Cheap; no LLM |
| `memory_retrieve` | Embed request, fetch top-k similar skills + recent failures + preferences | Examples only, never executed directly |
| `plan` | LLM call → `Plan` (structured output, one repair retry) | Prompt includes tool catalogue + memory examples |
| `validate` | Check tools exist, args validate against Pydantic models, step count ≤ max (default 12), no duplicates of Tier 3 | Failure → back to `plan` once with error text |
| `policy_gate` | For **next step only**: `PolicyEngine.decide()`; if Tier 3 → skip with refusal; if confirm needed → `interrupt(payload)`; on resume re-check unlock state and hash | **No side effects before `interrupt()`** |
| `act` | Verify `action_hash ∈ approved_hashes` (or Tier 0), run tool via registry (respect `--dry-run`, timeout, cancel flag) | Appends `StepResult` |
| `verify` | Tool-specific `verify()` + generic checks; decide `next` / `retry` (max 2) / `replan` (max 2) / `fail` | Failures → `failures` table |
| `replan` | LLM call with plan so far, results, error; produce remaining steps | Goes through `validate` + `policy_gate` again |
| `respond` | Compose final text; TTS if voice; task_log row | For `web_answer`, includes sources |
| `memory_save` | If all steps ok and verified → save/update skill; bump counters | Skip if any step tainted-and-sensitive |

Conditional edges: `policy_gate → act | respond(refused) | plan(clarify)`; `verify → act(next) | act(retry) | replan | respond`.

### Interrupt / resume contract

```python
# in policy_gate node
payload = {"type": "confirm", "step_id": d.step_id, "tier": d.tier, "summary": d.summary,
           "needs_unlock": d.needs_unlock, "typed_confirmation": d.needs_typed_confirmation,
           "action_hash": d.action_hash}
answer = interrupt(payload)          # blocks; graph pauses; checkpoint saved
# answer = {"approved": bool, "action_hash": str}   <- NO password here
if not answer["approved"] or answer["action_hash"] != d.action_hash: → refuse step
if d.needs_unlock and not ctx.unlock.is_unlocked(): → refuse step ("locked")
```

The **daemon** (not the graph) collects the password, verifies it with `UnlockManager.unlock(password)`, and only
then resumes the graph with `Command(resume={"approved": True, "action_hash": ...})`. Resume with
`graph.invoke(Command(resume=...), config={"configurable": {"thread_id": task_id}})`.
Verify the exact LangGraph API in the installed version before coding (it has changed across releases).

## 6. Daemon and IPC protocol (`daemon/protocol.py`)

Transport: TCP on `127.0.0.1` (random free port written to `daemon.json`), newline-delimited JSON (NDJSON), UTF-8.
First message must be `auth` with the token from keyring (constant-time compare via `hmac.compare_digest`).
Connections that fail auth are closed after a short delay; more than 5 failures → temporary ban of that connection source.
Max message size 64 KB.

Client → server:
```json
{"type":"auth","token":"…"}
{"type":"chat","id":"c1","text":"open notepad","source":"terminal"}
{"type":"confirm_response","task_id":"t1","approved":true,"action_hash":"…","password":"…optional…"}
{"type":"cancel","task_id":"t1"}
{"type":"voice","enabled":true}
{"type":"lock"}
{"type":"status"}
{"type":"shutdown"}
```
Server → client:
```json
{"type":"auth_ok","version":"0.1.0"}
{"type":"event","task_id":"t1","kind":"plan|step_start|step_result|log","data":{…}}
{"type":"confirm_request","task_id":"t1","tier":2,"summary":"Delete 3 files …","needs_password":true,"typed_confirmation":null,"action_hash":"…"}
{"type":"final","task_id":"t1","text":"Done. …"}
{"type":"error","code":"…","message":"…"}
{"type":"status","daemon":"running","voice":"on","unlocked":false,"queue":0,"active_task":null}
```
Rules: password field is used once and discarded; never logged; never forwarded to the graph.
Only one active task at a time; further `chat` messages are queued (max 5). Voice-initiated confirmations
that need a password open a small tkinter dialog on the user's desktop (topmost) and never accept spoken passwords.

## 7. LLM abstraction (`llm/client.py`)

```python
class LLMClient(Protocol):
    def structured(
        self, *, system: str, user: str, schema: type[BaseModel], model_role: str = "planner"
    ) -> tuple[BaseModel, Usage]: ...
    def text(self, *, system: str, user: str, model_role: str = "fast") -> tuple[str, Usage]: ...
```
- Roles map to model names in `config.toml` (`planner`, `fast`, `vision`). Defaults must be verified against current provider docs.
- Structured output: try the provider's JSON-schema mode first. Support is **per model** and an unsupported model returns HTTP 400, so on that error fall back to plain JSON mode ("respond with JSON only") + Pydantic parse + **one** repair retry, and log which mode was used. Cache the per-model result for the session.
- Usage (calls, tokens) is returned and accumulated in state for the benchmark.
- Timeouts (30 s), retries with backoff for 429/5xx (max 3), provider fallback Groq → Gemini if configured.
- `FakeLLM` returns scripted responses keyed by test scenario; all graph tests use it.

## 8. Concurrency model

- Daemon: `asyncio` event loop for IPC. The LangGraph run and blocking tool calls execute in a worker thread
  (`asyncio.to_thread`) so the loop stays responsive to `cancel` and `status`.
- A `CancelToken` (threading.Event) is checked between steps and inside long tools (dictation, audit).
- Voice loop runs in its own thread, pushes recognised text onto the same task queue with `source="voice"`.
- PyAutoGUI `FAILSAFE = True` (mouse to top-left corner aborts). Add small `PAUSE` (0.05–0.1 s) for reliability.
- **Stale confirmations:** LangGraph does not expire paused threads. The daemon marks tasks waiting on a confirmation longer than the timeout as `expired` (never resumed), on start and periodically; on version change all pending checkpoints are discarded.
- **Hidden daemon (`pythonw`)** has no console (`sys.stdout`/`sys.stderr` are `None`): redirect both to the log file at startup, install excepthooks, enforce a single instance. Details in `11_DEPLOYMENT_AND_PACKAGING.md §4`.

## 9. Memory schema (`memory/db.py`)

```sql
CREATE TABLE IF NOT EXISTS skills (
  id INTEGER PRIMARY KEY, goal_text TEXT NOT NULL, plan_json TEXT NOT NULL,
  embedding BLOB NOT NULL, tools_used TEXT NOT NULL,
  success_count INTEGER DEFAULT 1, fail_count INTEGER DEFAULT 0,
  created_at TEXT NOT NULL, last_used_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS failures (
  id INTEGER PRIMARY KEY, goal_text TEXT, step_json TEXT, error TEXT, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS preferences (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS task_log (
  task_id TEXT PRIMARY KEY, source TEXT, user_input TEXT, status TEXT,
  api_calls INTEGER, tokens INTEGER, steps INTEGER, replans INTEGER,
  started_at TEXT, finished_at TEXT, duration_ms INTEGER, peak_rss_mb REAL
);
```
Embeddings: local ONNX model via `fastembed` (small, CPU) or `sentence-transformers` MiniLM; stored as float32 bytes;
cosine similarity in NumPy (fine for < 10k skills). Retrieval threshold configurable (default 0.75). Skills whose
`fail_count` exceeds `success_count` are ignored.

## 10. Configuration (`config.py`)

`pydantic-settings` reads `config.toml` then env vars prefixed `JARVIS_`. Key settings:

```toml
[llm]
planner_model = "<verify current Groq model>"
fast_model    = "<verify current Groq model>"
vision_model  = "<verify current Gemini model>"   # optional
provider_order = ["groq", "gemini"]

[agent]
max_steps = 12
max_retries_per_step = 2
max_replans = 2
step_timeout_seconds = 30

[policy]
unlock_ttl_seconds = 300
password_max_failures = 5
lockout_seconds = 60
allowed_roots = ["~/Documents/JarvisWorkspace", "~/Desktop", "~/Downloads"]   # file tools operate only here
delete_preview_threshold = 1            # always preview
typed_confirmation_for_folders = true

[apps]     # allowlist: name -> command
notepad = "notepad.exe"
calculator = "calc.exe"
chrome = "chrome.exe"
# … extended by user

[daemon]
host = "127.0.0.1"
port = 0            # 0 = random free port

[voice]
enabled = false
wake_word = "hey_jarvis"
stt_model = "base"          # tiny|base|small
tts_backend = "piper"       # piper|sapi
```

## 11. Error handling philosophy

- Tools never raise into the graph; they return `ToolResult(ok=False, error=…)`.
- Verify decides retry vs replan. After max retries/replans → `respond` with an honest failure summary and the
  log path. Never claim success without verification.
- Unhandled exceptions in a task are caught at the runner, logged with traceback, and reported as `error` to the client;
  the daemon keeps running.
