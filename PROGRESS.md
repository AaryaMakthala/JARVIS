# PROGRESS.md

> Maintained by the coding agent. Update at the END of every session. Keep it short and factual.

## Current phase
Phase 5 — COMPLETE (voice input/output, dictation, focus-verified type_text)
All **472** automated tests pass (`pytest tests -m "not slow and not voice"` —
0 failures, 0 errors), including 5 `windows_only` and 29 integration tests.
`scripts/check.ps1` green.
Phase 5 security review (13 findings — F1..F13) — **all fixed & tested**, see below.
No real LLM provider/key is configured, so voice→LLM→TTS end-to-end is
**not** manually validated.  No voice deps (`openwakeword`, `faster-whisper`,
`piper-tts`, `pyttsx3`, `sounddevice`) are installed in this env — all real
implementations lazy-import gracefully and return `None` from their `create()`
factory functions.

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
Phase 6: WhatsApp with contacts and confirmation.  Do not modify Phase 5 files unless
a Phase 6 dependency requires it.
