# PROGRESS.md

> Maintained by the coding agent. Update at the END of every session. Keep it short and factual.

## Current phase
Phase 6 — DONE (WhatsApp contacts + whatsapp_send, verification-before-send, rate limits; fail-closed invariant #9).
Phase 2 LLM-driver — DONE this session (see "LLM-driver architecture" below).
Native Windows suite: **601 tests, 0 errors, 0 failures, 1 skipped** (`pytest tests
-m "not slow and not voice"`, includes the 5 windows_only tests and the NTFS ADS path
test). `scripts/check.ps1` green (29 pass, 0 fail). WSL hermetic run: 596 collected,
1 known failure (NTFS ADS test — passes only on real Windows, as before).
Phase 5 security review (13 findings — F1..F13) — **all fixed & tested**, see below.
- **Config regression fixed (Phase 5)**: real `config.toml` `[voice] enabled=true`
  was previously ignored — ``load_settings`` silently returned defaults because
  pydantic-settings ≥2.9 no longer auto-registers a TOML source; the ``_toml_file``
  init kwarg and ``toml_file`` model_config key are both inert there, so the file
  was never read while env overrides (unaffected source) still worked.  Now wired
  via ``settings_customise_sources`` -> ``TomlConfigSettingsSource`` in
  ``config.py``; CLI and daemon both call ``load_settings`` so each now reads the
  file.  Verified on the user's real runtime config: file loaded, `voice.enabled`
  now `True`. 12 new/updated tests; suite still **472 pass / 0 fail**.
No real LLM provider/key is configured, so voice→LLM→TTS end-to-end is
**not** manually validated.  No voice deps (`openwakeword`, `faster-whisper`,
`piper-tts`, `pyttsx3`, `sounddevice`) are installed in this env — all real
implementations lazy-import gracefully and return `None` from their `create()`
factory functions.

## Session: LLM-driver architecture (understand/converse/replan) + provider factory, memory, catalogue — DONE (no commit)

`pytest --ignore=tests/windows_only` = **707 passed, 8 skipped, 1 failed** (the
1 failure is the pre-existing NTFS-ADS Windows-only `test_paths.py`).  `ruff
check`/`format` clean.  `mypy src/jarvis/policy` still not runnable under WSL
Python 3.14 (numpy stubs reject 3.12-only `type` syntax — env artifact; run on
the dev PC).  Baseline was 596 collected; +110 collected, all green.

### What changed (target architecture, docs/02 §5)
- **Request routing**: `plan` node now also classifies `Plan.kind`
  (`tool`|`conversation`, default `tool` — zero churn for existing tests).  New
  deterministic `understand_request` node records `request_kind` and routes:
  `conversation → converse → respond`; `needs_clarification → respond`;
  `tool → validate`.  Only `kind=="tool"` can ever reach `validate`/`act`.
  `understand_request` never clobbers an existing halt.  `converse` node asks
  the fast LLM for the spoken answer, or honours a `dialog_answer` the planner
  already wrote (no redundant call).
- **Replanning** (`replan` node): budget (`agent.max_replans`, default 2) is
  enforced **in code**, never by prompt.  LLM returns a structured
  `ReplanDecision` (`continue`+replacement `Plan` | `stop`+message);
  `continue` resets step bookkeeping, keeps completed successes, drops the
  failed attempt, goes back through `validate`+`policy_gate`; `stop`/budget/LLM
  failure halts fail-closed with an honest message.  `verify` sets `replan_now`
  when retries are exhausted (only if budget remains), else fails closed.
  `validate` on a *replacement* plan preserves results/approved hashes and
  re-snapshots TOCTOU paths; rejections route initial→`plan` (1 repair),
  replacement→`replan` (1 repair).
- **Boundary schemas** (`agent/schemas.py`, all `extra="forbid"`): `UserRequest`,
  `ToolCall`, `ReplanDecision` (`continue` requires a `plan`), `ConfirmationRequest`,
  `AgentResponse` (`kind` mirrors `Plan.kind`), plus `PlanStep`/`PolicyDecision`
  aliases.  `ReplanDecision` is stored in state → added to the msgpack
  allowlist (serializer tests updated to the 5-entry list).
- **Provider factory** (`llm/provider.py`): `build_llm_client(settings,
  store)` walks `llm.provider_order` (groq→gemini; gemini "not implemented yet")
  returning the first buildable client or `client=None` + human-safe `reasons`
  — the graph then halts cleanly with "No LLM backend configured (run `jarvis
  init`)." instead of crashing.  Keys never leave keyring, never logged/returned.
  `provider_status()` feeds `doctor`.  Wired into `cli._chat_no_daemon` and
  `daemon._build_default_context`.
- **Memory light integration** (`memory/base.py`): `MemoryBackend` protocol +
  `MemoryRecord` + deterministic `NullMemory`; `AppContext.memory` flows to
  `memory_retrieve`; results capped at `[memory] retrieval_limit` (default 3)
  and truncated.  A memory failure never blocks a task.  Real embeddings stay
  Phase 8.
- **Registry**: `catalogue_with_policy()` (internal tier/windows_only view;
  never sent to the LLM).
- `TaskOutcome.agent_response` → typed `AgentResponse` for the daemon.

### New tests (5 files, ~60 tests)
`test_agent_architecture.py` (18 graph scenarios: conversation/tool routing,
no-LLM halt, eager + confirmed + multi-step execution, deny, tampered hash,
clarification, unknown-tool/invalid-args repair, malformed planner, replan
stop/continue/budget/keep-successes, worker resilience, voice source),
`test_replan.py`, `test_provider.py`, `test_schemas.py`, `test_memory.py`.
Three existing tests gained `max_replans=0` to stay focused on their original
claim (verification fail-closed; the replan path is covered by the new suite).

### Fragile / known
- Budget-exhaustion message surfaces when a *replacement* plan is rejected by
  `validate` and there is no budget left; a tool failure after budget is spent
  fails closed directly in `verify` (different, equally honest message).  Both
  are tested.
- `converse` with no LLM and no planner `dialog_answer` halts; with a written
  `dialog_answer` it answers without any LLM call.
- No real LLM key in this env: provider selection paths are unit-tested with a
  memory key store; nothing hits a live Groq/Gemini API.

## Session: "Yes?" then nothing — capture never waits for the command (bug fix, no commit)

User-reported: after wake + "Yes?" the loop stops; no command → STT → response.
Code trace (no hardware here) found the final wedge: `_read_command` only
detected **trailing** silence.  After the "Yes?" ack + mic flush it treated a
normal 1–3 s pause as "utterance ended" → returned ~0.7 s of silence → STT
empty → silent re-arm.  Inverse bug: a noisy room (RMS ≥ threshold) made every
chunk "speech", so capture ran the full `max_segment_s` (30 s), deaf to the
next wake.  Both produce the same visible "Yes?" then nothing / deaf for 30 s.

### Fix (all in `voice/loop.py` unless noted)
1. `_read_command(max_samples)` is now **wait-for-speech → capture**:
   - Phase A waits (bounded by `listen_timeout_s`) for the first speech-energy
     chunk, discarding pre-speech silence but keeping a ~200 ms pre-roll
     (`_CAPTURE_PREROLL_FRAMES`) so whisper still sees the utterance onset.
     No speech by the deadline → returns an empty segment → "no command
     detected; returning to wake".
   - Phase B accumulates chunks until `silence_timeout_s` of trailing silence
     or the `max_segment_s` cap.  `silence_threshold` (config `[voice]`) is now
     wired `config → DaemonServer._build_voice_service → VoiceService →
     VoiceLoop`; `_chunk_is_speech` takes the threshold (was a hardcoded 0.01).
2. Explicit `VOICE state=` INFO transitions (LISTENING/CAPTURING/TRANSCRIBING/
   PROCESSING/SPEAKING/RESETTING) + `VOICE wake_detected`, `VOICE capture_started`,
   `VOICE capture_finished`, `VOICE transcript=N chars` — transcript wording still
   redacted (char count only).
3. New idempotent `_reset_voice_state()` (flush + wake-detector reset + clear
   counters + state logs); `_rearm()` delegates to it.  `_safe_reset()` for
   exception paths.
4. `_run` now wraps each interaction in its own `try/except`: an exception
   escaping every per-phase guard is logged
   (`voice interaction failed; resetting and continuing to listen`), reset, and
   the worker keeps running.

### Tests (`tests/unit/test_voice_loop.py`)
- Updated `test_no_speech_capture_ends_immediately` → `test_no_speech_capture_waits_for_speech_then_gives_up`;
  updated `test_empty_utterance_returns_to_wake_listening` with a short `listen_timeout_s`.
- New: `test_late_command_after_ack_pause_is_captured` (THE regression proof:
  pause after "Yes?" no longer eats the command), `test_three_consecutive_wakes_process_all_commands`,
  `test_tts_response_failure_does_not_wedge_loop`, `test_unexpected_interaction_exception_is_contained`
  (corrupt STT result escapes the per-phase guards — outer `_run` boundary catches it),
  extended INFO-marker test to assert all `VOICE state=` transitions.
- Existing STT/route/capture-failure isolation tests already covered those phases.

**Status (WSL)**: `ruff check .` clean. `pytest tests/unit -m "not slow and not voice"` =
**602 passed, 3 skipped, 1 failed** (the 1 failure is the pre-existing NTFS-ADS/Windows-only
`test_paths.py` test — passes only on real Windows). `mypy` still not runnable under WSL Python 3.14
(numpy stubs) — run on the dev PC. NOT committed.
**REAL-HARDWARE VERIFICATION PENDING** — the acceptance test cannot be reproduced without
microphone/voice deps: on Windows, `jarvis daemon --foreground`, then 3×
"hey jarvis" → "Yes?" → PAUSE one beat → command → audible response, no restart. Also run once with
the command window unused (~`listen_timeout_s`=30 s default) and verify it returns to wake-listening
(tune down `silence_timeout_s` as needed on a noisy mic). Do NOT claim fixed until that passes.

## Session: console-script daemon wrote no INFO + voice-loop wedge proof (bug fix, no commit)

Follow-up to the two-session voice/daemon work. Two concrete defects found and fixed.

### Root cause A — missing INFO in jarvis.jsonl (the daemon ran "deaf" to diagnostics)
- The console script is `jarvis = "jarvis.cli:app"` in `pyproject.toml`, but
  `logging_setup.configure_logging()` was **only** called inside `cli.main()` (cli.py
  line 1041). `main()` runs only via `python -m jarvis` / autostart `pythonw -m jarvis`;
  the installed `jarvis.exe` invokes the Typer `app()` object directly, so **no root file
  handler was ever attached** and every `logger.info(...)` from `jarvis.voice.*` was
  dropped at the inherited WARNING level. Old file entries came from `python -m jarvis` runs.
- **Fix**: (a) `pyproject.toml` script → `jarvis = "jarvis.cli:main"` (so `.exe` reaches
  `main()`); (b) `DaemonServer.run()` now calls `logging_setup.configure_logging()` up front
  (idempotent; takes effect immediately even without a reinstall) and prints
  `[DIAG] server.run(): log file -> <path>` — the daemon always writes INFO no matter the
  entry point; (c) `configure_logging()` logs a self-proof INFO `logging configured: file=...`
  on both the fresh-attach and idempotent paths. Tests never call `.run()`, and the `_main`
  Typer callback was left alone so `CliRunner` tests stay pointed at `cli.app` (no test
  pollution of the real log dir). New test proves a `jarvis.voice.loop` INFO record lands in
  the configured `jarvis.jsonl`.

### Root cause B — two candidate wedges in the voice loop
- `loop.py`: `self._audio.flush()` after the wake ack and `_rearm()` were **unguarded** — a
  mic failure there killed the loop outright (or, worse, left it unable to re-arm, matching a
  single-"Yes?"-then-silence run). Now both are wrapped: a capture-flush failure logs
  `voice interaction failed at capture-flush` and returns to wake-listening; `_rearm()`
  failure is logged and the loop keeps waiting; `_rearm()` emits INFO
  `voice boundary: re-armed for next wake`.
- `server.py` `_voice_submit`: unbounded `while not slot.done` wait (heartbeat only logged, never
  returned) → a single hung worker wedged the voice loop forever. Added `VOICE_SUBMIT_TIMEOUT_S
  = 180.0` deadline (returns `command timed out — try again` → loop speaks an error and re-arms),
  30 s heartbeat now reports real elapsed time.

Per-phase isolation is now *proven* by tests (deterministic fakes, no hardware): first-wake enters
`_run_interaction`, ack speaks before capture starts (event ordering), capture/STT/routing
failures each leave the loop re-armed and answering the NEXT wake, and every boundary INFO marker
is emitted through the `jarvis.voice.loop` logger (caplog-asserted). Transcript wording stays out
of INFO (char count only).

**Status (WSL)**: `ruff check .` clean; hermetically in `/tmp` venv `pytest tests/unit` =
**598 passed, 4 skipped, 1 failed** (the 1 failure is the pre-existing NTFS-ADS/Windows-only
`test_paths.py` test, which only passes on real Windows). `mypy` not runnable under WSL Python 3.14
(numpy stubs) — run on the dev PC. NOT committed.
**REAL-HARDWARE VERIFICATION PENDING — same blocker**: run on Windows `jarvis daemon --foreground`,
do 3 consecutive interactions, then read the log tail. The INFO `logging configured: file=...` line
at daemon start now proves file logging is attached; look for `voice boundary: re-armed for next
wake` between the three interactions. Do NOT claim fixed until 3 consecutive interactions produce
audible speech without a daemon restart.

Manual smoke test (native PowerShell):
1. `pip install -e .` once (so the `jarvis.exe` entry point is `cli.main`), then
   `.venv\Scripts\jarvis.exe daemon --foreground`.
2. Check the first console lines print `[DIAG] server.run(): log file -> ...`.
3. Say "hey jarvis" → expect "Yes?", then speak the command after the ack (one phrase like
   "hey jarvis, open notepad" is fine on a quiet mic but the command audio is flushed with the
   ack buffer — prefer wake → wait for "Yes?" → command).
4. Repeat the interaction 2 more times without restarting the daemon.
5. `Get-Content "$env:LOCALAPPDATA\jarvis\jarvis\logs\jarvis.jsonl" -Tail 300 | Select-String "voice"`

## Session: second-wake-not-detected + foreground daemon exit (bug fix, no commit)

Two user-reported runtime bugfixes, implemented and regression-tested together.

### Root cause 1 — voice: second "Hey Jarvis" never detected
- The wake detector was **never re-armed** after an interaction (only at loop
  start / before confirmations), and command capture read a **fixed
  `listen_timeout_s` (30 s) window**, so the loop was deaf to a second wake
  while capturing after the first command (evidence: the user's two v-task ids
  are ~33 s apart, matching 30 s capture + STT).
- **Fix** (`voice/loop.py`): (a) wake detector is `reset()` + mic `flush()`ed
  after every interaction via new `_rearm()` (called after the extracted
  `_run_interaction()`, so even empty/noise/failed utterances return to
  wake-listening); (b) new `_read_command(max_samples)` captures in 100 ms
  chunks and **ends on `silence_timeout_s` of trailing silence** (RMS gate,
  bounded by `max_segment_s`) instead of a full 30 s window — the loop is never
  deaf for longer than the spoken utterance ~+0.7 s.  Used for both Phase 3
  commands and voice confirmations.  `silence_timeout_s` / `max_segment_s`
  (already in `VoiceSettings`) are wired `settings → VoiceService → VoiceLoop`.

### Root cause 2 — daemon: foreground daemon exits / voice queue wedges
- `_worker_run` created the SQLite checkpointer **outside** its `try`: any
  failure there (locked DB, OneDrive sync on the path) killed the worker
  without marking `slot.done`, so `_voice_submit()` blocked **forever** and the
  voice-loop thread wedged on the first interaction.  `_dispatch_loop` /
  `_stale_confirmation_loop` / `_handle_client` / `_serve` had no per-iteration
  isolation, so one unexpected exception could kill a background task and
  surface as a process exit from `cli.py`.
- **Fix** (`daemon/server.py`): checkpointer moved inside the worker `try`
  (failure → slot failed + voice thread woken); dispatch loop refactored into
  crash-proof `_dispatch_loop` + `_dispatch_once` with a `try/except` per pass
  and `_recover_dispatch_after_error()`; stale-confirmation loop and
  `_handle_client` per-message handling isolated; `_serve` shutdown path wrapped
  in `try/finally` (nothing escapes `asyncio.run`); `cli.py daemon()` now logs
  and exits(1) on any crash instead of a silent exit.

### Verification
- `pytest tests/unit tests/integration`: **607 passed, 1 skipped** (pre-existing
  skip), no failures.  New regression tests: two-consecutive-wakes both submit
  (`test_two_consecutive_wakes_both_process_commands`), empty-utterance returns
  to wake, silence-terminated / max-bounded / immediate capture units, dispatch
  loop survives a bad iteration, bad IPC message doesn't kill the session,
  checkpointer failure cannot wedge `_voice_submit`, and the G1 ordering test
  rewritten for chunked capture (flush immediately precedes the first 1600-frame
  capture read). `ruff check` / `ruff format --check` clean, `mypy
  src/jarvis/policy` green, `git diff --check` clean.
- **Live fix NOT yet proven on real hardware** (no mic/voice deps on this
  machine; probe harnesses with fakes could not reproduce the daemon exit at
  all).  Needs the user's Windows manual smoke tests (below).  Scratch probe
  files removed.

### Manual smoke tests to run on the Windows PC
1. `pip install -e ".[voice]"` if voice deps are not yet installed; reboot the
   daemon: `.venv\Scripts\jarvis daemon --foreground`.
2. **Second-wake**: in another prompt run `jarvis chat`, say/toggle voice on
   (`jarvis on`), then say *"hey jarvis — open notepad"*, wait for the full
   response, then *"hey jarvis — what time is it"* (or any 2nd task).  Both must
   execute; the old 30 s dead window + missing re-arm should be gone.  Repeat a
   third time.
3. Say *"hey jarvis"* then stop talking: the loop must say "Yes?" and return to
   listening within ~1-2 s (capture cut short by trailing silence), NOT a 30 s
   silence.
4. **Daemon stays up**: from `jarvis chat`, submit a task and Ctrl+C out of the
   client mid-task; the daemon must print "result … dropped: client
   disconnected" (normal) and keep running — `jarvis status` must answer.
5. Watch the log file (`%LOCALAPPDATA%\jarvis\logs\jarvis.log`) for
   "dispatch loop iteration failed" / "daemon crashed" — should be absent in
   normal use.

## Phase 5 security review — findings fixed (F1–F13)

- **F1** `tools/keyboard.py`: `type_text` now re-verifies the *foreground*
  window title before pasting (`_verify_focus`, fail-closed: unreadable title
  or mismatch → error, never paste).  Single `_type_text_run`; the old
  duplicated sub-path removed.
- **F2** `voice/loop.py`: dictation pauses (fail closed) as soon as the target
  window loses focus — speech is never forwarded to a wrong window.  Resume
  re-checks focus first.  New real `voice/focus.py` `WindowFocusChecker`
  (Win32) wired into the daemon's voice service.
- **F3** `tools/dictation.py` `save_dictation`: path is resolved + root-checked
  (`resolve_safe` / `roots_from_settings` / `within_any_root`) *before* any
  GUI step and the file is written to the **resolved** path only.  Symlink
  escape inside a root is rejected.
- **F4** `voice/loop.py`: command transcripts are logged at INFO as a char
  count only; the wording only ever appears at DEBUG through `redact()`.
  Dictation logging is len + redacted DEBUG.
- **F5** session limits enforced in the loop: `max_session_s` (1800) and
  `max_dictation_chars` (20000) auto-stop dictation; `idle_timeout_s` stops
  the whole loop.  Config defaults added to `VoiceSettings` + `DEFAULT_CONFIG_TOML`.
- **F6** voice confirmations fail closed: wake word must be re-detected before
  a "yes" is accepted (`_rearm_wake_for_confirmation`), timeout / no detector /
  unrecognised answer are refusals, Tier 2+ and typed-folder confirmations
  always need the terminal (`can_confirm_by_voice`).
- **F7** `jarvis on`/`off` now toggle the **daemon's** voice pipeline via the
  new `voice_toggle` IPC message (`DaemonClient.send_voice_toggle`) instead of
  running a private CLI loop.  Disabled-in-config or missing deps → typed error.
- **F8** voice task bridge in `daemon/server.py`: `_voice_submit()` enqueues a
  task owned by `VOICE_OWNER` and blocks; the worker asks a Tier-1
  confirmation by voice (`_run_voice_confirmation`) and resolves it on the
  worker thread; the final answer is read straight off the slot — no IPC
  socket, no race with the confirmation hash.
- **F9** Tier-2 locking: slot.confirm_tier is recorded at publish time and a
  voice-sourced confirm for Tier 2+ is refused in `_handle_confirm` (on top of
  the loop-side refusal).
- **F10** `VoiceService`/`VoiceLoop` lifecycle hardened: service start/stop
  serialised under a lock (fixed a self-deadlock), `VoiceLoop.stop()` waits on
  a `_stopped` event and never closes audio while the loop thread may read.
- **F11** `allowlist_names()` in `tools/apps.py` uses `model_dump()` so apps
  added through config `extra` entries are honoured by `type_text` and
  `start_dictation` (and tested).
- **F12** `ConfirmResponse` gained `source: Literal["terminal","dialog","voice"]`
  (default terminal) so voice origin is explicit on the wire.
- **F13** `VoiceToolsFacade` on `AppContext`/`ToolContext`; dictation tools
  read the live loop via `ctx.voice` and are *honest* when no loop is running
  (`dictation_active: false`, "no running voice loop") instead of claiming
  success.

## What works (verified), Phase 5 additions

### Voice interfaces and fakes (`voice/interfaces.py`, `voice/fakes.py`)
- Five runtime-checkable protocols: `AudioInput`, `WakeWordDetector`,
  `SpeechToText`, `TextToSpeech`, `WindowFocusChecker`.
- Data classes: `AudioSegment`, `WakeWordResult`, `STTResult`, `VoiceCommand`.
- Full fake implementations: `FakeAudioInput`, `FakeWakeWord`, `FakeSTT`,
  `FakeTTS`, `FakeFocusChecker`, `FakeVoiceCommandParser`.
- All fakes are deterministic, scriptable, and record call history for assertions.

### Real voice backends (lazy-imported, optional)
- **`voice/wake.py`** — `OpenWakeWordDetector` wrapping openWakeWord ONNX backend.
  `create()` returns `None` when `openwakeword` is not installed.  Threshold
  configurable; reset method supported.
- **`voice/vad.py`** — `SoundDeviceVAD` using `sounddevice` InputStream with
  energy-based silence detection.  Configurable silence threshold and timeout.
  `create()` returns `None` when `sounddevice` is not installed.
- **`voice/stt.py`** — `FasterWhisperSTT` using faster-whisper tiny/base model
  with int8 quantisation on CPU.  `create()` returns `None` when
  `faster_whisper` is not installed.
- **`voice/tts.py`** — `PiperTTSEngine` (GPL-3.0, requires `piper-tts`) with
  `SapiTTSEngine` fallback (`pyttsx3` wrapping Windows SAPI5).
  `create()` tries Piper first, falls back to SAPI, returns `None` if neither
  works.
- **`voice/focus.py`** — `WindowFocusChecker`: reads the Win32 foreground
  window title via `ctypes`; fails closed (returns ""/False) when unreadable.

### Voice loop (`voice/loop.py`)
- `VoiceLoop` runs in a background daemon thread.
- Pipeline: wake-word → acknowledge ("Yes?") → capture speech → STT → route.
- Routing: "stop listening" → loop stops; "stop dictation" → ends dictation;
  other text → `submit_task(text, "voice")`.
- Voice confirmation (F6): re-arms the wake word → "Say {wake} to confirm." →
  listen → yes/no.  Tier 2+ / typed folder names fall back to terminal.
- Dictation mode (F2): speech forwarded to `on_dictation` callback; target
  focus re-checked every utterance; focus loss pauses dictation; F5 limits
  (session duration, char count) stop dictation automatically.
- Redaction (F4): transcripts at INFO = char count; wording only at DEBUG via
  `redact()`.
- Graceful error handling: exceptions logged, loop stops, audio stream closed
  (`stop()` waits on `_stopped`; never closes audio mid-read).

### Voice service (`voice/service.py`)
- `VoiceService` manages lifecycle: `start()`, `stop()`, `is_active()`,
  `.loop`.  All transitions under a lock (re-entrancy bug fixed).  Stops the
  daemon voice on shutdown.
- Used by daemon (`jarvis on/off` via IPC, auto-start when `enabled=true`).

### CLI (`cli.py`)
- `jarvis on` — asks the daemon to start listening (DaemonClient
  `send_voice_toggle(True)`); prints "voice activated / already active".
- `jarvis off` — asks the daemon to stop (`send_voice_toggle(False)`).

### Daemon integration (`daemon/server.py`)
- Daemon initialises `VoiceService` on boot when `[voice] enabled = true`
  (`_bootstrap_voice`) and wires `VoiceToolsFacade` into the agent context.
- New `voice_toggle` protocol message → `_handle_voice_toggle`.
- Voice tasks (`_voice_submit` / `VOICE_OWNER`) + worker-side voice
  confirmation (`_run_voice_confirmation`) with hash-bound resume answers.
- `ConfirmResponse.source` (terminal/dialog/voice) gates Tier 2+.
- `StatusResponse.voice` reports `"on"` / `"off"`.

### Voice tools (`tools/keyboard.py`, `tools/dictation.py`)
- **`type_text`** (Tier 1): allowlist check → focus window → **foreground
  re-verification** (F1) → clipboard paste → clipboard restore.  Fails
  closed on any mismatch.
- **`start_dictation`** (Tier 1): opens target app; signals the live voice
  loop via `ctx.voice`; honest fallback when no loop is running (F13).
- **`stop_dictation`** (Tier 0): ends active dictation session.
- **`save_dictation`** (Tier 1): resolves + root-checks the path first (F3),
  reads Notepad via Ctrl+A/Ctrl+C, writes to the *resolved* path, verifies.

### Tool registry (`tools/registry.py`)
- New tools registered: `type_text`, `start_dictation`, `stop_dictation`,
  `save_dictation`.  Total registered: 16 tools.

### Config (`config.py`)
- `VoiceSettings` extended: `silence_threshold`, `silence_timeout_s`,
  `max_segment_s`, `listen_timeout_s`, `idle_timeout_s`, `max_session_s`,
  `max_dictation_chars`.
- `config.toml [voice]` defaults updated.

### Invariant tests (`test_invariants.py`)
- Invariant #15: voice→LLM submission receives only string text, never
  bytes/ndarray/audio.

### Tests (37 new in this round; suite: **472**)
- `test_daemon_voice.py` — 12: toggle protocol + end-to-end toggle,
  voice-source Tier 2/3 rejection, Tier 1 voice acceptance, `_voice_submit`
  worker bridge (approve / refuse / no-loop / queue-full), parse edges.
- `test_voice_loop.py` (extended) — +10: transcript redaction at INFO/DEBUG,
  session + char limits, idle timeout, focus pause/resume, confirmation
  fail-closed (no detector, rearm timeout, unrecognised answer), stop waits
  for thread.
- `test_tools_type_text.py` (extended) — +4: happy path with GUI fakes,
  foreground mismatch, unreadable foreground, config-added app allowlist.
- `test_tools_dictation.py` (extended) — +8: resolved write inside roots,
  outside-roots rejection, symlink escape rejection, honest no-loop start/stop,
  start/stop with live loop facade.
- `test_voice_cli.py` — config defaults for `max_session_s` /
  `max_dictation_chars`.

## What has NOT been verified (deferral)
- Real LLM provider call (no Groq/Gemini API key in this env)
- `jarvis daemon --foreground` + `jarvis chat` end-to-end with a real LLM
- File creation/deletion through the daemon with a real planner
- `jarvis autostart` on reboot/logon (manual; deferred)
- `jarvis doctor --live` (requires a real Groq key + model name)
- Live web search via `ddgs` (all tests use mocks)
- Voice hardware: real mic input, real TTS output, real wake-word detection
- Voice deps not installed: `openwakeword`, `faster-whisper`, `piper-tts`,
  `pyttsx3`, `sounddevice` — all lazy-imported behind interfaces

## What works (verified), Phase 4 additions
- **`web_answer` tool** (`tools/web.py`): Tier 0; searches via `ddgs`, generates a cited
  answer via LLM, verifies citations against real results, flags output as `tainted=True`.
- **Deterministic taint detection** (`policy/rules.py` `args_overlap_taint`): independent
  of LLM's `depends_on_untrusted` flag.
- **Invariant #8** added to `test_invariants.py`.
- **Red-team cases** (`benchmarks/redteam.yaml`): 17 cases.

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
- **Phase 4 taint overlap** (docs/03 §8): the substring-based detection (≥12 chars) can miss
  semantic/paraphrased dependencies.
- **Phase 8 replan** must wrap all tainted `StepResult.output` as `<untrusted_data>` before
  passing results back to the planner (deferred — no replan node exists yet).
- **Voice wake-word over-sensitivity**: openWakeWord pre-trained "hey jarvis" model may
  trigger on partial words; threshold tuning and debounce deferred to manual testing on real
  hardware (docs/12 §1).
- **Piper TTS GPL-3.0**: kept optional; SAPI fallback always available.  If Piper fails to
  install on Windows, `jarvis doctor` will show WARN but voice still works via SAPI.
- **`jarvis on/off` require the daemon** (F7): if no daemon is running, the toggle prints a
  hint and exits 1 — same UX as `jarvis status`.
- **Real focus verification unproven**: `WindowFocusChecker` (Win32 `GetForegroundWindow`) is
  unit-tested via fakes only; its interaction with Notepad/allowlisted apps needs one manual
  pass on real hardware.
- **Voice confirmation re-arm depends on a real wake-word event**: with a fake/fixed detector
  the loop does not emit a "confirm" prompt (only the deny path is fully covered without
  hardware).

## Known bugs
- None in this session.

## Decisions log
| Date | Decision | Reason |
|------|----------|--------|
| 2026-09-21 | Phase 5: all voice code behind protocols + fakes | AGENTS.md §6: tests must run without hardware; `@pytest.mark.voice` for real-hardware tests |
| 2026-09-21 | Tier 2 confirmation rejected by voice | docs/03 §7: "Voice may answer yes/no for Tier 1 only; Tier 2 requires the terminal or dialog" |
| 2026-09-21 | type_text uses clipboard paste + focus verification | docs/04 §2.3: "verify foreground window title/process matches target_app, then paste via clipboard; abort if check fails" |
| 2026-09-21 | VoiceLoop runs in daemon thread, not asyncio | LangGraph runner is synchronous; asyncio event loop cannot block on long STT/TTS calls |
| 2026-09-21 | All voice deps lazy-imported with factory `create()` → `None` | Only `onnxruntime` is installed; `openwakeword`, `faster-whisper`, `piper-tts`, `pyttsx3`, `sounddevice` are not; graceful degradation required |
| 2026-09-21 | `jarvis on`/`off` now toggle the *daemon's* voice pipeline via IPC `voice_toggle` (F7) | Phase 5 security review: a CLI-owned loop bypassed daemon policy (no task bank, no tier gating); daemon-managed voice goes through the normal LangGraph + policy path |
| 2026-09-21 | Voice confirmations re-arm the wake word and fail closed (F6); Tier 2+ only via terminal | Phase 5 security review: a late "yes" must not fire off a prior confirmation; docs/03 §7 tier rules |
| 2026-09-21 | Save only to *resolved* root-checked absolute paths; symlink escape rejected (F3) | Phase 5 security review: a relative/`..`/symlinked path must never be written open |
| 2026-09-21 | save_dictation reads Notepad via Ctrl+A/Ctrl+C | Most reliable cross-version approach; pywinauto read-back varies by Notepad version |
| 2026-09-21 | wake audio converted to int16 before openWakeWord `predict()` | 0.6.0 mel-preprocessor casts input to int16; float [-1,1] truncated to {0,±1} = score 0.000 (proven on hardware) |
| 2026-09-21 | audio `read()` deadline is a stall clock, not wall-clock | absolute 0.5s deadline capped every read (~7680 samples) and would truncate commands |
| 2026-09-21 | WhatsApp reached only via `DesktopWindowProvider` protocol; tests inject a fake window | docs/06: no desktop in CI; unverifiable chat → never press Enter (invariant #9) |
| 2026-09-21 | `whatsapp_send` Tier 2, rate limit enforced in code before any window opens | docs/03 §3, docs/04 §2.6 step 7: recipient count == 1, 10/h, 5s minimum |
| 2026-09-21 | contacts edited only through `jarvis contacts …`; numbers masked in CLI/describe (+last 2 digits) | docs/04 §2.6: no LLM-driven contact writes; docs/03 redaction |
## Audit session (2026-09-22) — full spec/code review delivered, G1 fixed

- Full audit report (sections A–J) delivered: phases 0–6 COMPLETE, 7/8 INCOMPLETE (stubs),
  quality checks all green (pytest 0, ruff 0, mypy policy 0, diff-check 0).
- **G1 fixed**: `audio.flush()` was never called despite its docstring. Now called in
  `voice/loop.py` before the command read and before the confirmation read, so stale
  pre-prompt mic audio cannot be mistaken for a command or an approval. `flush()` added to
  the `AudioInput` protocol + `FakeAudioInput`; ordering tests added. NOT verified on real
  mic hardware (manual pass needed).
- Deferred (P2, out of scope for this session): `jarvis run`, chat Ctrl+C cancel,
  Phase 7 (audit) implementation — next phase.

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

# Voice tests (requires voice deps installed: pip install sounddevice openwakeword faster-whisper piper-tts pyttsx3)
# Voice is toggled through the DAEMON, so Terms 1 & 2 above must be running first.
jarvis on                           # "voice activated"; already-active → "voice is already active" (idempotent)
#   Say "hey jarvis" → should acknowledge "Yes?" (wake re-armed before any confirmation)
#   Say "open notepad" → should open Notepad
#   Say "type hello world in notepad" → type_text: focus verified before paste (fails closed)
#   Say "stop listening" → should stop
jarvis off                          # Stop voice listening
# Without voice deps installed, `jarvis on` prints an error and exits 1 (fail closed, no partial start).

# Dictation test
#   Open Notepad manually
#   Say "hey jarvis, take notes"
#   Dictate 5 sentences
#   Say "stop dictation"
#   Check text appears in Notepad
#   Say "hey jarvis, save as notes.txt" → should save file

# type_text test (via agent)
#   jarvis chat --no-daemon
#   Type: "type hello world into notepad"
#   Should paste "hello world" into Notepad (requires Notepad open)
```

`jarvis doctor --live` requires a real Groq key + a model name in `config.toml [llm]`.

## Next step
Phase 7: audit (tools/audit) with hard-coded whitelisted commands, per-check timeouts,
`jarvis audit` — read-only, graceful degradation without admin, fixtures-based unit tests.
(After the LLM-driver session: also worth a real-provider smoke pass — configure one Groq
key + planner/fast model names, then `jarvis chat` a conversational ask and a tool ask.)

---

## Real voice bug fix + `jarvis voice doctor` + Phase 6 (WhatsApp) — run on native Windows

### Real-hardware voice bug: "voice activated but 'Hey Jarvis' never responds" — FIXED
- **Root cause (proven)**: `wake.py` fed openWakeWord 0.6.0 float32 samples in [-1,1];
  its mel-preprocessor casts to `int16` (`AudioFeatures._get_melspectrogram`), so the
  floats truncated to {0,±1}. Real "hey jarvis" clip through the real mic: float path
  max score **0.000**, int16 path **0.966**. The mic, audio input and TTS were all fine.
- **Fix A** `voice/wake.py`: `detect()` now scales/clips to `int16` before `predict()`.
- **Fix B** `voice/audio_input.py`: `read()` used an absolute wall-clock deadline (0.5s)
  — ANY read was capped at ~0.5s (read(480000) returned 7680). Now a stall clock reset
  per delivered block. This would have truncated spoken commands after wake.
- Regression tests: `test_voice_wake.py` (int16 contract, clipping, threshold, reset,
  create() None paths) + 3 long-window/stall tests in `test_voice_audio_input.py`.
- **Real validation**: fixed detector on the recorded clip → `PIPELINE_WAKE=PASS`
  (6 frames, max 0.974).

### `jarvis voice doctor` (new)
- `jarvis voice doctor` reports audio library / input device / mic open+read / wake
  model / wake detection / STT model / TTS, wired exactly like the daemon's
  `_build_voice_service`. No speech recorded beyond a short sample probe; no raw
  audio/transcripts shown. Flags `--no-mic/--no-stt/--no-tts` skip slow/downloading
  probes. `Check`/`_check` moved to `jarvis/checks.py` (shared with `cli.py`).
- **Real run on this PC: all 8 checks PASS** (mic = Microphone Array, wake model loaded,
  faster-whisper-base cached, SAPI fallback). 11 unit tests.

### Phase 6 WhatsApp — DONE
- `tools/whatsapp.py`: `ContactStore` (contacts.json, E.164-ish validation 8-15 digits,
  masked display +last 2 digits, dedupe), `RateLimiter` (persisted, max/hour + min
  interval), `WhatsAppWindow`/`DesktopWindowProvider` protocols + best-effort
  pywinauto provider (every UI read fails closed), and `whatsapp_send` (Tier 2) with
  the exact §2.6 flow: resolve exactly one contact → rate-limit → open
  `whatsapp://send?phone=…&text=…` → verify chat header (name or digits) → verify
  draft (when readable) → Enter → best-effort read-back. Unverifiable chat → never
  press Enter, leave the draft.
- CLI `jarvis contacts add|list|remove` (numbers masked in list output; raw number
  never printed). Config `[whatsapp]` block (`app_name`, `max_per_hour=10`,
  `min_interval_s=5.0`, `window_timeout_s=10.0`). 17 tools registered.
- **Invariant 9 strengthened**: `test_invariant_9_whatsapp_unverifiable_window_never_sends`
  (window never found → not sent, no Enter, nothing recorded).
- Tests: `test_whatsapp.py` (contacts, rate limit, tool flow with fake window incl.
  invariant-9 paths), `test_contacts_cli.py`. ~35 new Phase 6 tests; suite 535→**601**.
- Manual/real WhatsApp Desktop send deferred (docs/05: own test number only); pywinauto
  reads of the live WhatsApp UI (header/draft) remain unproven on real hardware —
  provider degrades to fail-closed by design (docs/12).

## Manual tests added this session (PC)
```powershell
.venv\Scripts\python.exe -m jarvis voice doctor        # all PASS expected
jarvis contacts add "Test Contact" +<your second number>
jarvis contacts list                                   # masked, +••••••••xx
jarvis contacts remove "Test Contact"
# WhatsApp end-to-end (test number only):
#   jarvis daemon --foreground → jarvis on → "hey jarvis, send message to <contact> …"
#   or in chat: send a WhatsApp message → confirm shows masked number + exact text,
#   unlock, approve; watch the desktop chat verify before Enter.
jarvis on   # re-verify: "hey jarvis" → "Yes?" now (was silent)
```

## Known fragile areas (Phase 6)
- pywinauto may not read the WhatsApp header/draft (docs/12 §1) — the tool then
  reports "Could not verify chat" and leaves the draft; that is correct fail-closed
  behaviour, not a bug, but it needs one manual pass with a real WhatsApp Desktop chat.
- contacts.json / rate-limit file live in the data dir; a corrupt contacts file is
  treated as empty (safe) rather than blocking every tool call.

---

## Post-session update (created by WSL coding run — voice Phase 5, Stage 0 + Stage 1)

**Stage 0 (ground truth)** — verified from source, not docs:
- Loop reads one fixed window `read(listen_timeout_s * 16000)` per utterance (loop.py:243);
  wake detection reads 1280-sample chunks (loop.py:293, 320). **The VAD was never invoked**
  (normal path). `stop()` blocks up to 10 s on `_stopped` (loop.py:125).
- wake.py:49-50 feeds `np.array(segment.samples, dtype=np.float32)`; no int16 scale/clip;
  `reset()` runs at loop start and before confirmations; no cooldown.
- `VoiceService.start()` returns the generic "voice dependencies not available — install
  with: pip install jarvis-agent[voice]" message (service.py:78-80) when deps are missing.
- "lock jarvis" routes to the planner via `_submit_task` (loop.py:395).  Malformed
  `config.toml` raises from pydantic-settings (config.py:264) — no fallback.

**Root-cause fix (the user's real crash)**: `open()` passed `sample_rate=`/`block_size=`
to `sd.InputStream`, whose real kwargs are `samplerate`/`blocksize`. On a machine with
sounddevice installed the strict `__init__` raised `TypeError` → "could not open the
microphone".  **Demonstrated**: old HEAD code raises `VoiceInputError` under a strict
fake stream; new code opens and reads exactly.  (User's pasted `threading.current_event_time`
traceback is from a stale/OneDrive-synced copy — not in this tree.)

**Stage 1 changes** (in `src/jarvis/voice/`):
- `audio_input.py`: protocol-compatible `open(sample_rate=None, channels=None)`;
  stream built via new `_stream_kwargs()` (`samplerate`/`blocksize`/`dtype="int16"`);
  per-instance callback rate-limiting (5 s); module-global `_callback_last_status_log`
  removed; `_read_buffer` surplus so `read(n)` returns **exactly n** (old code dropped
  the surplus tail of each block); `close()` clears `_open` first → unblocks a blocked
  reader in ~0.1 s and clears the buffer; `flush()` resets buffer + dropped counter.
  Targeted-only-noqa added at the four `except Exception` sites.
- `interfaces.py`: `Callable` imported from `collections.abc` (ruff F821).
- New tests: `tests/unit/test_voice_audio_input.py` (~21, strict-fake sounddevice),
  `tests/unit/test_voice_wiring.py` (current server wires an `AudioInput`, never a VAD;
  plus a regression that extracts the **old** `_build_voice_service` from commit `326727b`
  and pins that it produced a VAD with no `read` — the crash), and
  `tests/unit/test_voice_contracts.py` (conformance matrix + **recorded VAD gap**:
  `listen_for_speech` takes no `read_chunk`; Stage 3 makes the VAD pure).

**Status (WSL run, excludes `windows_only`, marker "not slow and not voice")**:
`496 passed, 7 skipped, 1 failed` — the single failure
`test_paths::test_resolve_safe_rejects_alternate_data_streams_and_drive_relative` is
NTFS ADS semantics and passes on real Windows (also failed at the 464-pass baseline).
Ruff: clean across `src` + `tests`. `mypy src/jarvis/policy` clean under `--python-version 3.12`
(numpy's newer `.pyi` uses 3.12-only `type` syntax under the 3.11 target → env artifact).
**Not yet verified on real Windows hardware/mic** — see the manual smoke-test list the
agent reported.  Fixes are untested against a live `config.toml` with `voice.enabled=true`.

---

## Voice Phase 5 Stage 2 (voice state machine + fixed error codes) — DONE (run on native Windows)

**Behavior change**: `VoiceService.start()` now *identifies the missing component* and
moves a state machine `off | starting | on | error`, instead of the old generic
"voice dependencies not available" message. Fixed codes (closed set, invariant-tested):
`no-audio-library`, `mic-open-failed`, `wake-model-missing`, `stt-model-missing`,
`tts-unavailable`, `loop-crashed`. A missing wake detector is now a *hard* `wake-model-missing`
error (the old "voice-activity gate" fallback is gone — voice refuses to start without one).

- Mic open happens synchronously in `start()` (Stage 1 made `open()` idempotent, so the
  loop thread's open is a no-op) → `mic-open-failed` is reported *now*, not silently in the thread.
- `VoiceLoop` gains `on_exit(reason)` — the loop thread reports `mic-open-failed` (open failed)
  or `loop-crashed` (mid-loop) from `finally`; audio is still closed in finally. Clean exits
  (idle timeout / "stop listening" / explicit stop) report `None` → state `off`.
- `StatusResponse` gains optional `voice_reason`; server status/toggle/bootstrap now read the
  service state, so `jarvis status` shows the reason and `jarvis on` prints the reason + hint and
  exits 1 on error; `jarvis on` after an error retries. Boot auto-start failure sets `error`
  without stopping the daemon.
- **Demonstrated old-vs-new**: old HEAD service/loop fail 4/5 contract checks (generic message,
  no state, mic failure "voice activated", unreported crash); new passes all 5.
- Tests: `test_voice_cli.py` (state machine, per-code identification, sync mic-open, retry,
  loop-crash→error, clean-exit→off), `test_voice_loop.py` (on_exit crash reporting),
  `test_daemon_protocol.py` (voice_reason field), `test_daemon_voice.py` (toggle sends
  ErrorMessage with code, retry at daemon level, status reason, boot-failure keeps daemon alive),
  `test_invariants.py` (closed error-code set).

**Status (native Windows venv, Python 3.11.15)**: full suite `-m "not slow and not voice"`
= **535 tests, 0 errors, 0 failures, 1 skipped**. `ruff check src tests` clean. `mypy
src/jarvis/policy` clean. voice+daemon mypy = 21 pre-existing errors (no new ones from this
stage; service/protocol/cli are clean). NOT verified on real mic hardware — manual smoke
tests below.

---

## Daemon self-exit crash (stale Windows PID) — FIXED (WSL run)

**Root cause**: `os.kill(old_pid, 0)` on Windows for a stale pid raised `SystemError:
<class 'OSError'> returned a result with an exception set`, which is NOT caught by
`except OSError` → `_check_single_instance()` threw → daemon self-exited in ~1-2s.
Rechecked with a stale pid: `os.kill(pid,0)` raises `SystemError` (not `OSError`/`ProcessLookupError`).

**Fix** (`src/jarvis/daemon/server.py`):
- `_check_single_instance()` now uses `psutil.pid_exists()` (already a dependency).
- Stale pid file (pid not alive) → `_take_stale_pid_file()`: log warning, unlink daemon.json,
  start fresh.
- Reconciler `_check_foreign_pid()` for a *live* pid: `NoSuchProcess` → stale+delete;
  `AccessDenied` → refuse to start (keep file); `_looks_like_jarvis_daemon(name, cmdline)` →
  refuse; anything else (pid reuse by a non-JARVIS process) → treat as stale+delete.
- New module helper `_looks_like_jarvis_daemon` (python/jarvis name + "jarvis"+"daemon" in cmdline,
  matches autostart's `pythonw -m jarvis daemon`).
- New `tests/unit/test_daemon_single_instance.py` — 10 tests (no pid file / corrupt / own pid /
  dead pid → stale+delete / live JARVIS → refused / NoSuchProcess race / AccessDenied / non-python
  reused pid → stale). All pass. Nothing committed; NOT re-verified on real hardware.
- Re-checking pid 10820 (the stale value that crashed the daemon) now yields "stale, deleted".

---

## Voice interaction not audible during normal use — DIAGNOSTICS ADDED, root cause pending (WSL run)

**User-verified facts**: `jarvis status` = daemon running / voice on / unlocked False / queue 0;
`audio.open()` succeeds; "voice loop started" appears; direct SAPI test is audible (command below);
`stt_model` = `C:\Users\aarya\models\faster-whisper-base`; `tts_backend = "piper"` with no
model_path → SAPI fallback (correct). Piper warning is not the issue. Do NOT replace SAPI /
install Piper / revert the single-instance fix.

**Call-graph traced** (reading notes kept; each file read once):
- `loop.py`: `_run` → `_wait_for_wake` → `_run_interaction` (ack "Yes?" → flush → `_read_command`
  → STT → `_route` → `tts.speak(response)`) → `_rearm`. `_route` → stop/dictation/confirmation or
  `_submit_task` (blocking).
- `server.py`: `_voice_submit` (queue or immediate worker thread; blocks loop thread on
  `slot.event` until `done`) → `_worker_run` → `run_task`/`resume_task`; Tier-1 confirmations
  driven voice-side via `_run_voice_confirmation` → `loop.confirm_by_voice` on the **worker** thread.
- `tts.py`: `create()` piper→SAPI fallback; `SapiTTSEngine.speak` = `pyttsx3` `say`+`runAndWait`
  (lazy init on first use — first call happens on the **voice-loop thread**, not the main thread).
- `stt.py` faster-whisper (`int8`, beam 1, VAD off — our own VAD); `audio_input.py` bounded-queue
  mic (single stream owner); `wake.py` int16-scaled openWakeWord; `service.py` `off|starting|on|error`
  state machine, synchronous `audio.open()` in `start()`.

**Changes made this session** (all uncommitted, alongside the older DIAG instrumentation):
- `loop.py`: per-iteration exception isolation in `_run_interaction` — each phase (wake-ack,
  capture, STT, route, response-TTS) is individually guarded; a failure is `logger.exception`ed
  and the loop keeps listening instead of crashing the whole service (`loop-crashed`). Previously
  any STT/TTS/route error killed voice until `jarvis on` again. Boundary diagnostics added:
  interaction start / ack starting+spoken / capture start+done / capture too short / STT start+done
  (chars+language) / STT empty / routing / response ready / TTS speaking+dones / interaction
  completed (marker `voice boundary:`). Kept the INFO char-count + DEBUG redacted transcript lines
  (redaction invariant). Added `voice boundary: confirmation listening for yes/no` in
  `confirm_by_voice`.
- `server.py`: `_voice_submit` now logs submit/queued/rejected/errored/finished (chars) + a
  30s "still waiting" heartbeat so a wedged worker is visible in `jarvis.jsonl`.
- `cli.py` + `voice/service.py`: fixed 2 pre-existing `UP031` %-format DIAG prints → f-strings.
- Tests: replaced `test_exception_in_loop_does_not_crash` with
  `test_transient_stt_error_keeps_loop_listening` (STT boom → logged "voice interaction failed
  at stt", loop STILL active, re-armed, ack spoken); `test_mid_loop_crash_reports_loop_crashed`
  now crashes the **wake detector** (keeps `loop-crashed` reporting path covered);
  `test_successful_interaction_speaks_and_keeps_listening` (new: ack + response spoken, loop alive,
  re-armed).

### Wake-detection stage: diagnostics added; root cause pending one real run

**Real-hardware finding (2nd run, user)**: daemon starts, mic opens, `voice-loop` thread launches,
but the log contains ONLY `microphone open` + `voice loop started` + `voice auto-start: voice
activated`. There are NO wake/interaction/ack/capture/STT/TTS logs at all → failure is in the
**wake path** (before STT). The old wake code only logged when a wake was DETECTED, so it could not
distinguish "no frames reach the detector" from "frames reach it but scores are silently low".

**Verified against installed openwakeword 0.6.0 source** (why the pipeline SHOULD work):
- Audio: `sd.InputStream` 16 kHz / mono / int16 / blocksize 1280 → float ÷32768 in client →
  detect() scales back to int16 (near-lossless round-trip) → `AudioFeatures` mel pipeline
  (`raw_data_buffer`, melspectrogram, 96-d embedding `feature_buffer`); 1280-sample chunks are the
  documented exact batch size (model input = last `model_inputs` embedding rows).
- `Model.predict` zeroes the first 5 frames after init/reset (~400 ms warm-up — transient, not a bug).
- Model/name key: `wakeword_models=["hey_jarvis"]` → predictions dict key `hey_jarvis` matches
  `model_name` AND `create()` config default. `reset()` clears preprocessor + prediction buffer.
- Config has **no** wake-threshold setting; wake.py hardcodes `_DEFAULT_THRESHOLD = 0.5`
  (settings only expose `wake_word`; no threshold field). Threshold NOT changed pending real scores.

**Changes (uncommitted)**:
- `wake.py`: detector now tracks `frames_seen` / `last_score` / `max_score`; a per-frame DEBUG
  "wake detector invoked" line + a throttled (2 s) INFO "wake detector sample (... last_score=
  ... max_score=... threshold=0.50)" so the log proves whether frames AND scores are flowing
  before any threshold change is made; "wake word detected" now includes threshold; `reset()`
  logs at INFO ("wake detector reset") — also serves as re-arm evidence after every interaction.
- `loop.py` `_wait_for_wake()`: throttled (3 s) INFO heartbeat "voice boundary: waiting for wake
  word (frames_read=... elapsed_s=...)" + DEBUG per-frame "wake frame" lines + "wake trigger;
  entering interaction" on success. **Transient wake-detector exceptions are now caught, logged
  ("voice boundary: wake detector error; re-arming and continuing"), the detector re-armed, and
  the loop keeps listening** (previously any detector exception killed the loop). Mic `read()`
  failures still surface as `loop-crashed`.
- `audio_input.py`: callback increments `_blocks_fed` (bytes never logged — counters only);
  `read()` emits a throttled (5 s) INFO "microphone stream status: blocks_fed=... blocks_dropped=
  ... open=..." (proves the callback→queue→reader path at runtime); `open()` now logs the actual
  stream `samplerate/channels/blocksize/dtype`.
- Tests (all focused on wake path): wake unit tests `test_silence_never_triggers`,
  `test_last_max_and_frames_seen_track_predictions`, `test_reset_clears_last_score_and_forwards_to_model`,
  `test_detector_usable_after_reset`; loop tests `test_frames_reach_detector_but_quiet_audio_never_triggers`
  (exact 1280-frame chunks verified, quiet audio → no wake), `test_transient_detector_error_then_recovers`
  (1st detect raises → re-armed → 2nd detect drives a full interaction, loop stays active, logged).
  `test_mid_loop_crash_reports_loop_crashed` crash source moved from the (now-isolated) wake detector
  to a device `read()` failure.

**Status (WSL)**: `ruff check .` clean; `mypy src/jarvis/policy` clean; full suite =
**629 passed, 1 skipped** (pre-existing WSL symlink skip). NOT committed.
**BLOCKER — one native-Windows run decides the root cause**: the INFO heartbeats above will show
one of (a) `blocks_fed` climbing → queue healthy, (b) `frames_read` climbing but `last_score`≈0 →
frames reach the model but it scores ~0 on real-mic audio (privacy-silence vs low gain vs model),
or (c) neither feeding nor reading → stream/callback problem. Then pick the minimal fix from
evidence (e.g. an explicit score, mic-level check, or a *small justified* threshold step — NOT a
dramatic lowering). Do not claim fixed until 3 consecutive interactions work without a daemon restart.

**Status (WSL)**: `ruff check .` clean; full suite = **623 passed, 1 skipped** (pre-existing WSL
symlink skip in `test_tools_dictation.py:222`). NOT committed. `mypy src/jarvis/policy` unchanged.
**REAL-HARDWARE VERIFICATION PENDING = THE BLOCKER**: need the native PowerShell session to run
`jarvis daemon --foreground`, then 3 consecutive real voice interactions, then read
`%LOCALAPPDATA%\jarvis\jarvis\logs\jarvis.jsonl` for the sequence
wake detected → voice boundary interaction start → ack → capture → STT → routing → response ready
→ TTS speaking → interaction completed → rearm, for each of the 3 utterances. That will pinpoint
the exact stage where TTS is not reached. Do NOT claim fixed until 3 consecutive interactions
produce audible speech without a daemon restart.

Manual smoke test (native PowerShell):
1. `.venv\Scripts\jarvis.exe daemon --foreground`
2. Say "hey jarvis, open notepad" (watch for "Yes?" then the response).
3. Say "hey jarvis, what time is it" twice more without restarting the daemon.
4. `Get-Content "$env:LOCALAPPDATA\jarvis\jarvis\logs\jarvis.jsonl" -Tail 200 | Select-String "voice"`

## Voice state machine hardening — coding complete, hardware verification still pending (WSL run)

**Task** (new user requirement list): make the voice loop converge to LISTENING/STOPPED after every
wake (incl. empty/STT/processing/TTS/unexpected-exception failures and initial trailing silence),
re-arm the wake detector **exactly once per interaction**, never per frame, and expose rate-limited
**audio-level diagnostics** (RMS) in the JSONL log so "mic content is quiet" is distinguishable from
"model is not scoring" at runtime.

**Root-cause decided (do not re-open)**: the historical "wake once-only" behaviour is NOT a reset
bug — evidence in `%LOCALAPPDATA%\jarvis\jarvis\logs\jarvis.jsonl`:
- single 0.910 detection (03:39:49.059) → capture 1.900 s/30400 samples → STT 16 chars → task
  `v1790134794950` started 03:39:54.951 and errored 03:39:55.005 → 36-char speech → re-arm →
  flatline: `last_score` <= 0.017 for 36 s while `blocks_fed`/`frames_read` climbed (audio streamed).
- openwakeword `reset()` cost is <= ~2 s (5 frame warm-up), so the 36 s flatline is mic content /
  room sensitivity (`hey_jarvis` @ threshold 0.5), not a post-reset bug.
- SEPARATE pre-existing issue: "user command produced no response" = agent **planner failure**
  (`validate.py:21` "The planner produced no plan."); runtime `config.toml` has empty `[llm]`
  models. Out of scope here — remains its own fix.

**Changes (uncommitted)**:
- `loop.py`: `_set_state` fixed set now includes `WAKE_DETECTED`; full cycle order is now
  LISTENING → WAKE_DETECTED → "Yes?" (state still WAKE_DETECTED) → CAPTURING → TRANSCRIBING →
  PROCESSING → SPEAKING → RESETTING → LISTENING. `_run` failure path now logs only and relies on
  a **single `_rearm()`** as the one-and-only re-arm point (previously the exception path double
  re-armed via `_safe_reset()` + `_rearm()`; `_safe_reset()` deleted — no test referenced it).
  `_wait_for_wake` sets `WAKE_DETECTED` on detection; its ~3 s heartbeat (new `_WAKE_HEARTBEAT_S`)
  now logs `audio_rms=%.4f`; `_read_command` "waiting for command" heartbeat same. Added
  `_segment_rms()` helper (used by `_chunk_is_speech` for the RMS-threshold comparison).
- `wake.py`: `detect()` computes `audio_rms` from float32 samples and includes `audio_rms=%.4f`
  in the throttled "wake detector sample" INFO line — the distinguishing diagnostic.
- `cli.py`: `jarvis daemon` now prints `logs: <jsonl path>` on start so the JSONL location is
  obvious in the foreground console.
- Tests (`test_voice_loop.py`): full 8-state INFO marker sequence test; exactly-once re-arm
  (startup 1 + one per interaction = 3 for two interactions) on success AND on a failed-then-
  successful interaction; no-per-frame-reset while waiting; capture ends on trailing silence
  (initial silence after "Yes?" does NOT terminate capture, `listen_timeout_s` only caps);
  heartbeat carries `audio_rms`; heartbeat is rate-limited (many frames, few lines).
  (`test_voice_wake.py`): detector sample line includes `audio_rms=0.5000`.

**Status (WSL)**: full suite = **614 passed, 1 skipped** (pre-existing WSL symlink skip in
`test_tools_dictation.py:222`). NOT committed. `ruff`: not available under the WSL-side python3.14;
`ruff check .` stayed clean from the prior native run; `mypy src/jarvis/policy` unchanged (blocked
on numpy stubs under WSL). A flaky pre-existing timing race was found and fixed during this run:
`test_transient_detector_error_then_recovers` asserted `spoken == ["Yes?", "ok"]` as soon as
`submitted` became non-empty, but the response TTS happens on the loop thread a moment later, so
under full-suite load the assert occasionally ran too early; the test now polls for the full
completion signal (`spoken == ["Yes?", "ok"]`). Full suite green after the fix.
**REAL-HARDWARE VERIFICATION STILL PENDING = THE BLOCKER** (unchanged, do not claim fixed): run the
manual smoke test above and read the JSONL for the 8-state sequence + `audio_rms` lines (wake flatline
`audio_rms` ≈ 0 → mic/privacy; `audio_rms` normal but `last_score` ≈ 0 → model sensitivity).
