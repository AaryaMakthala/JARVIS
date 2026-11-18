# JARVIS P0 Task Event Store Implementation Report

## Scope
P0 completion for task lifecycle/event observability in `C:\Users\aarya\OneDrive\Desktop\Jarvis`: P0.1 confirmation/clarification events, P0.2 tool lifecycle events, P0.3 cancellation persistence, P0.4 CLI task event timeline, P0.5 lifecycle/documentation/acceptance updates.

No push performed. No live integration claims are made (no live Groq/Gemini, no real desktop/browser/WhatsApp/voice runs).

## Implementation

### P0.1 - confirmation and clarification events (`4f7f01f`)
- Records `CONFIRMATION_REQUESTED` and `CLARIFICATION_REQUESTED` pause events via task-store telemetry when the graph pauses for user input.
- Preserves privacy/allow-list constraints (no question/answer text, no secrets/messages/tool args/raw output).
- Commit: `4f7f01f feat(taskstore): record confirmation and clarification events`.

### P0.2 - tool lifecycle events (`c33375d`)
- `src/jarvis/agent/nodes/telemetry.py`: fail-soft `record_tool_started` / `record_tool_finished`.
- `src/jarvis/agent/nodes/act.py`: emits `TOOL_STARTED` immediately before actual non-dry-run tool execution and `TOOL_FINISHED` immediately after return/failure; dry-run and refusal-before-execution paths emit none.
- Event details are sanitized/allow-listed: start `{tool, step_id}`; finish `{tool, step_id, ok, verified: bool|None, duration_ms}`. No args, user messages, secrets, tool output, or exception messages.
- Task-store projections for `active_tool` / `active_step_id` are handled by existing taskstore event handling; writers are fail-soft.
- Commit: `c33375d feat(taskstore): record tool lifecycle events`.

### P0.3 - cancellation persistence (`bf5c11f`)
- `src/jarvis/agent/runner.py`: `run_task()` / `resume_task()` catch `Cancelled` escaping `graph.invoke()`, best-effort persist terminal `CANCELLED` via `_record_cancelled` (ensures task on run path when source known; finishes with `TaskStatus.CANCELLED`, `failure_category="cancelled"`, `failure_code="cancelled"`), then re-raise. Persistence is centralized in runner (not daemon-only) and fail-soft.
- Avoids contradictory terminal writes by relying on taskstore idempotent finish behavior for same terminal status/metadata; daemon slot/queue/cancellation bookkeeping preserved.
- Commit: `bf5c11f feat(taskstore): persist task cancellation`.

### P0.4 - CLI task event timeline (`e8453ad`)
- `src/jarvis/cli.py`: `jarvis tasks --task <full-task-id>` displays sanitized timeline (seq, timestamp, event type, stage/status where available, sanitized details JSON) using existing task-store read APIs; existing recent-tasks view and plain piped output behavior preserved.
- Handles missing store/file, unavailable/unreadable store, unknown task, and read errors gracefully without traceback (`task history unavailable` or explicit not-found/read error + exit code 1 where appropriate).
- `tests/unit/test_cli.py`: added timeline tests for sanitized event display, unknown task, unreadable store, and missing store.
- Commit: `e8453ad feat(cli): display task event timeline`.

### P0.5 - lifecycle/documentation/acceptance (this commit)
- Lifecycle mapping documented (no new statuses/schema migrations):
  - `COMPLETED` = `completed`
  - `WAITING_FOR_USER` = `awaiting_clarification`
  - `WAITING_FOR_APPROVAL` = `awaiting_confirmation`
  - `RETRYING` = `running` with `retry_count` / replan information (`note` event with `event_code=replan`)
- `PROGRESS.md`: corrected registered tool count from 40 to 57 (dynamic registry count: 57) and documented P0 task observability/lifecycle mapping and timeline.
- `docs/JARVIS_P0_IMPLEMENTATION_REPORT.md`: this report.

## Tests and Verification

Focused verification executed in this workspace (`.venv\Scripts\python`):

| Command | Result |
|---|---|
| `pytest tests/unit/test_taskstore_p0_events.py tests/unit/test_taskstore_tool_lifecycle.py tests/unit/test_taskstore_telemetry.py -q` (P0.1/P0.2 context) | 20 passed |
| `pytest tests/unit/test_taskstore_p0_cancellation.py tests/unit/test_taskstore_p0_events.py tests/unit/test_taskstore_tool_lifecycle.py -q` (P0.3 context) | 14 passed |
| `pytest tests/unit/test_cli.py -k "tasks" -q` (P0.4) | 8 passed |
| `pytest tests/unit/test_taskstore.py -q` (taskstore regression) | 64 passed |
| `pytest tests/unit/test_taskstore_p0_events.py tests/unit/test_taskstore_tool_lifecycle.py tests/unit/test_taskstore_p0_cancellation.py tests/unit/test_cli.py -k tasks tests/unit/test_taskstore.py tests/unit/test_taskstore_telemetry.py tests/unit/test_taskstore_runtime.py tests/unit/test_taskstore_privacy.py -q` (focused final set) | 100 passed |

Ruff:
- `ruff check src/jarvis/cli.py` after P0.4 fixes reports 2 remaining pre-existing findings in unchanged code (`S110`/`BLE001` at `src/jarvis/cli.py:1232`, `try-except-pass` in existing confirmation helper); no new Ruff errors introduced by P0.4 changes.
- `ruff check tests/unit/test_cli.py` clean for P0.4 test additions in this run context.
- P0.2/P0.3 changed Python files were Ruff-clean at commit time (per implementation sessions).

Mypy:
- Not run as part of P0 final verification (no policy-module changes; repo guidance requires `mypy src/jarvis/policy` for general changes and it was not part of this scoped P0 closeout).

## Commits (local, not pushed)

| Hash | Message |
|---|---|
| `4f7f01f` | `feat(taskstore): record confirmation and clarification events` |
| `c33375d` | `feat(taskstore): record tool lifecycle events` |
| `bf5c11f` | `feat(taskstore): persist task cancellation` |
| `e8453ad` | `feat(cli): display task event timeline` |
| (this commit) | `docs(taskstore): document lifecycle and acceptance` |

## Final State
- Branch: `main`
- Status before P0.5 commit: modified `PROGRESS.md`, added `docs/JARVIS_P0_IMPLEMENTATION_REPORT.md`; untracked audit docs preserved (`docs/JARVIS_BACKEND_AUDIT.md`, `docs/JARVIS_COMPREHENSIVE_STATUS_AUDIT.md`)
- Nothing pushed.
