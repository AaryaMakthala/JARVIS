# JARVIS ROADMAP v2: Auto mode, screen, clicking, apps, camera, WhatsApp

> Single source of truth for the coding agent. Work **one stage at a time**.
> At the END of every stage append a short dated entry to `PROGRESS.md` and tick the box in section 9.
> Read this whole file before starting any stage, then re-read the stage you are on.

---

## 0. How to work (rules for the agent)

1. **Per stage: ANALYZE, then IMPLEMENT, then SMALL SANITY CHECK.**
   - ANALYZE: re-read the cited symbols (line numbers drift) and state what you will change before changing it. Say "unverified" instead of guessing.
   - SANITY CHECK: ruff on touched files plus *only* the focused tests for the touched area. **Do NOT run the full suite** until Stage 10. Stage 10 is the only full-suite run.
2. **Never weaken a safety invariant silently.** If a stage changes one (see sections 1 and 4), update `docs/03`, `docs/04`, the invariant test, and write the change in `PROGRESS.md`.
3. Keep the **deterministic policy** as the only authority: the LLM proposes, `PolicyEngine` decides, tier = `max(...)`, hash-bound confirmation, no shell execution. Planner never sees tiers.
4. **Free models only.** Do not touch the free-only gate (`llm/models.py` `qualify`). New models must be registered with correct `pricing_mode`.
5. **No new giant files.** `voice/loop.py` (about 1.8k lines), `daemon/server.py` (about 1.6k) and `cli.py` (about 2.2k) are already too big. New logic goes in new modules and the big files only call into them.
6. **Tool naming trap:** `register()` rejects names or descriptions matching `shell|powershell|cmd|exec|eval`, and `policy/rules.py` `BLOCKED_NAME_FRAGMENTS` silently makes a tool Tier 3 if its name contains e.g. `rm`, `registry`, `delete`, `defender`, `cookie`, `payment`. Note `confirm`, `form`, `platform` contain `rm`. Check every new tool name against `matches_blocked` before writing code.
7. Windows-only code stays behind `platform_guard` / lazy imports so the module still imports on WSL/Linux.
8. Voice features are hardware-dependent: for each stage, finish with the **Live test** listed and ask the owner to run it. Do not claim "fixed" without the live result.
9. Do not print or log API keys, transcripts at INFO, screenshots, or camera images.

---

## 1. Decisions already made by the owner

| # | Decision |
|---|---|
| D1 | **Verify what exists live first** (Stage 0) before changing anything. |
| D2 | **Transcript meaning-check = fold into the brain** (no extra LLM call). Brain is told input came from speech and may be garbled and may return `clarification`. Local deterministic gate (`_transcript_problem`) stays as first filter. JARVIS also speaks what it heard before risky actions. |
| D3 | **Modes:** enter full control with *"Jarvis, activate auto mode"*, leave with *"Jarvis, sleep"*. In auto mode JARVIS **repeats what it is going to do** and waits for *"proceed"*. |
| D4 | **Confirmation** must be natural and hands-free (no wake word needed for the yes/no answer). Current confirmation is buggy (Stage 1). |
| D5 | **Tier 2 by voice (delete, WhatsApp send):** in loop/auto mode one spoken plan plus "proceed" covers it. In normal one-by-one mode: "Hey Jarvis, delete X", JARVIS reads it back, owner says "proceed". Implemented as an **opt-in flag** `voice.allow_tier2_by_voice` with readback guardrails (Stage 3). This knowingly relaxes docs/03 section 7 ("voice = Tier 1 only"). |
| D6 | **Screen:** user asks "what's on my screen" and JARVIS tells what site/app is open and the visible text. Free, local-first. |
| D7 | **Clicking websites: use the DOM** (Playwright, JARVIS-owned Chrome profile). Native apps via UI Automation. Vision only as fallback. Best, free, reliable. |
| D8 | **Apps by voice:** Spotify (play a song), Chrome, Notepad (write what I say), Calculator, current time, VS Code (small code only), main camera (take photo, show photo again, delete on command, send on WhatsApp). |
| D9 | **Multiple API keys** (extra Groq, OpenRouter on other accounts): used as **fail-over** AND by an **orchestrator** that gives independent sub-tasks to different models/keys. |
| D10 | Good **user experience** matters: fast replies, audio cues, short spoken status, "I heard: ...". |
| D11 | Testing: small sanity checks per step, full validation only at the end (Stage 10). |

---

## 2. Ground truth from the repo analysis (verified by read-only pass at `99de4c2`)

Voice flow: `VoiceLoop._run` -> `_wait_for_wake` -> `_run_interaction` -> `_read_command` -> `FasterWhisperSTT.transcribe` -> `_transcript_problem` -> `_route` -> `_canonical_command` -> `DaemonServer._voice_submit` -> `_worker_run`/`run_task` -> `tts.speak` -> `_rearm`.

Facts that drive this roadmap:

- **Confirmation needs the wake word again**: `VoiceLoop._rearm_wake_for_confirmation` (`voice/loop.py` about 1642-1669) speaks `"Say {wake_word} to confirm."` using the **raw config name `hey_jarvis`** (underscore likely read aloud), then loops on `detect()`. Fails closed. Callers: `confirm_by_voice` (about 773/799), `capture_free_text` (about 825/842).
- **Voice approves Tier 1 only**: `can_confirm_by_voice` (about 758) and `DaemonServer._handle_confirm` (`daemon/server.py` about 1439-1447, `tier_requires_terminal`).
- **Barge-in does not exist.** Only the in-flight *task* cancel (`watch_for_stop`, "Stopped.") exists. `SapiTTSEngine.stop()` is a no-op, and with no Piper model the TTS is SAPI, so speech is uninterruptible. `tts.stop()` is called only in `VoiceLoop.stop()`.
- **Bare "stop" kills the voice loop**: `_STOP_LISTENING = {"stop listening","stop","turn off","go to sleep"}` (about line 72), handled in `_route` (about 1676). No wake-up path exists.
- Three state vocabularies disagree: `VoicePhase` (`voice/states.py`), `VoiceStatusReporter`, and `VoiceLoop._set_state` free strings (`CAPTURING`, `PROCESSING` not in `VoicePhase`; `_set_state` never calls `check_transition`).
- Dead code: `voice/vad.py`, `VoiceCommand`, `FakeVoiceCommandParser`, `VoiceActivityDetector`, `VoiceLoop._handle_voice_confirmation`, `VoiceLoop._on_confirmation` (unreachable in daemon).
- `ToolSpec.windows_only` is declared but never enforced at runtime.
- 21 tools registered. **Missing:** time, list windows, screenshot/screen read, click, camera, Spotify, VS Code, open-file/viewer.
- `open_app`: allowlist in `[apps]` (`AppSettings`, `extra="allow"`); resolves absolute path / PATH / App Paths; `Popen` without shell; no user argv. `.cmd` shims need an absolute `Code.exe`.
- `type_text`: clipboard paste into an allowlisted focused window, re-verifies foreground title. No keystrokes, no clicks. `docs/04` section 2.3 forbids generic key/click tools.
- `whatsapp_send` (Tier 2): text only, via `whatsapp://send` URI plus pywinauto verification (fragile, unproven on real hardware); contacts via `jarvis contacts`; rate limiter exists.
- LLM: `MultiProviderClient` fails over per call across providers `groq, openrouter, nvidia, gemini`; one key per provider; health/cooldown per `(provider, role)`; roles `planner`/`fast` only; **no vision capability and no image input path**; Gemini and Groq are `free_tier` (need `strict_zero_cost=false`, which the owner has set).
- Observed live: Groq 429 bursts, 10 s backoff, and a 16 s structured-repair retry on OpenRouter. Voice latency is dominated by provider time.
- Deps: `pywinauto`, `pyautogui`, `pyperclip`, `psutil`, `sounddevice`, `openwakeword`, `faster-whisper`, `pyttsx3`, `piper-tts` installed. **Not installed:** `playwright` (declared under `[browser]`), `mss`, `opencv`, any OCR.
- Math is the planner's job. `tests/unit/test_math_phrasings.py` has a structural guard that **fails the build if arithmetic words are hard-coded in `src/jarvis`**. Do not add a calculator tool or phrase table.

---

## 3. Target architecture

```
Mic -> wake / AUTO listen -> STT -> local gate -> BRAIN (speech-aware)
   -> plan -> [announce plan] -> "proceed" -> validate -> policy_gate -> act -> verify -> respond -> TTS

Perception (read-only):  UIA text | DOM snapshot | local OCR | Gemini vision (fallback)
                          -> Element list (ids, names, roles)   [tainted: untrusted]
Action (bounded loop):   observe -> pick element id -> policy -> act -> observe -> verify
                          max 8 steps, max 2 retries per step, then STOP and tell the user

LLM layer: key pool per provider (fail-over) + orchestrator routing
           (planner->strongest, fast->cheapest, vision->Gemini). Parallel ONLY for independent LLM-only sub-tasks.
```

Voice modes: **NORMAL** (wake word each command) / **AUTO** (continuous, plan then "proceed") / voice OFF. "Jarvis sleep" returns AUTO to NORMAL.

---

## 4. Safety changes this roadmap makes (must be documented in docs/03, docs/04)

1. **Voice confirmation without wake word** (Tier 1): short yes/no window opened *after* the prompt finishes (half-duplex, mic flushed). Strict approval words. Fail closed on timeout.
2. **Voice approval of Tier 2** only for `delete_path` (files, not folders needing typed name) and `whatsapp_send`/`whatsapp_send_file`, only when `voice.allow_tier2_by_voice = true`, only after an **exact readback** (target path or recipient + full message/file name), only the word **"proceed"** (plus "cancel"), only within the window. Password path stays for everything else. Known tradeoff: anyone in earshot can say "proceed". Mitigations: readback, short window, strict word, flush, half-duplex, existing rate limits, Recycle Bin undo, default OFF.
3. **Plan-level approval:** one "proceed" approves the hashes of all steps whose args are fully known at plan time. Steps whose args depend on earlier outputs are re-gated individually with a short readback. Hash binding and TOCTOU re-checks in `policy_gate`/`act` must remain intact.
4. **Semantic UI actions** (DOM refs, UIA control names) are allowed; **pixel/coordinate clicks and arbitrary key presses are still not provided.** Amend `docs/04` section 2.3.
5. **Screen/page content is untrusted** (`tainted=True`): any action derived from it escalates to confirmation (existing taint rule).
6. Camera capture and screenshots-to-cloud are privacy-sensitive: Tier 1 with spoken notice.

---

## 5. STAGES

### Stage 0: Live verification (NO code changes)

Owner runs on native Windows; agent records results in `PROGRESS.md` under "Live baseline 2026-10-xx".

Setup: `jarvis doctor --live`, `jarvis voice doctor`, restart `jarvis daemon --foreground` (picks up wake threshold 0.15).

Tests (record seconds, provider, exact spoken text):
1. Wake -> command -> answer, three times in a row without restart.
2. "what is 2 plus 2": seconds from STT line to spoken answer; which provider.
3. "open chrome", "open calculator".
4. Stop during thinking: slow question, then "Hey Jarvis... stop" (expect "Stopped.").
5. Stop during speaking (expect NOT to work); log shows `tts sapi` or `piper`.
6. Confirmation: "create a file test.txt with hello": exact spoken prompt, whether wake word is needed, whether it hears "yes".
7. Bare "stop": "Hey Jarvis" then "stop", then `jarvis status` (suspect voice turns off).
8. Garbled input -> expect "Could you repeat that?".
9. Idle 2+ min, then wake again.

Also attach `Get-Content "$env:LOCALAPPDATA\jarvis\jarvis\logs\jarvis.jsonl" -Tail 200`.

**Done when:** every test has a recorded PASS/FAIL/observation. Failures found here are fixed in Stage 1 before any new feature.

---

### Stage 1: Voice correctness (fix what exists)

Goal: stop/confirmation/TTS behave the way a person expects.

Analyze first: re-read `voice/loop.py` `_route`, `_rearm_wake_for_confirmation`, `confirm_by_voice`, `capture_free_text`, `_run_interaction` speak path; `voice/tts.py`; `daemon/server.py` `_run_voice_confirmation`, `_can_watch_for_stop`.

Implement:
- **1a Spoken wake-word name.** Add a helper that converts `hey_jarvis` to "hey jarvis" for prompts (also for any status text spoken aloud).
- **1b Wake-free confirmation window (Tier 1).** Replace the `detect()` loop in `_rearm_wake_for_confirmation` with: speak prompt (e.g. "Say yes to continue, or no to cancel.") -> `flush()` -> quiet-start drain -> open window `voice.confirm_window_s` (default 8 s) -> capture -> strict yes/no match. Same for `capture_free_text` (clarifications). Keep `_STRICT_YES_WORDS`/`_LOOSE_YES_WORDS` split and fail-closed behaviour. Update the test at `tests/unit/test_voice_loop.py` about line 1004 and the docs/03 section 7.8 text.
- **1c Stop semantics.**
  - "stop" / "cancel" / "never mind" = cancel current speech and task. **Never** exit the loop.
  - "turn off voice" / "stop listening" = voice off.
  - "go to sleep" is reserved for Stage 2 (mode change).
  Edit `_STOP_LISTENING` and its handling in `_route`.
- **1d Interruptible TTS + barge-in.** Make SAPI stoppable (preferred: `win32com` `SAPI.SpVoice` with async speak and purge, instead of per-call `pyttsx3`; pywin32 is installed) or provide a Piper model (better voice, free). Pick in ANALYZE and justify. While SPEAKING, run the wake detector on the mic; on "Hey Jarvis" call `tts.stop()` and go to LISTENING. Half-duplex note: speaker echo can self-trigger; if echo is a problem document "use headphones" and add a short post-speech guard. Add tests for `tts.stop()` mid-speech (fake TTS + fake audio).
- **1e Delete dead code:** `_handle_voice_confirmation`, `_on_confirmation` plumbing, `VoiceCommand`, `FakeVoiceCommandParser`, `VoiceActivityDetector`, `voice/vad.py` (also fix the stale docstring at the top of `loop.py`).

Sanity: ruff on touched files; `pytest -q tests/unit/test_voice_loop.py -k "confirm or stop or route"`, `tests/unit/test_daemon_voice.py`, `tests/unit/test_voice_wiring.py`.

Live test: tests 4-7 from Stage 0 now pass: "yes" without wake word approves a Tier 1 create_file; "stop" during speech silences it; bare "stop" does not turn voice off.

---

### Stage 2: Modes: AUTO and sleep

Goal: "Hey Jarvis, activate auto mode" -> continuous listen/answer loop; "Jarvis sleep" -> back to NORMAL.

Analyze first: `_wait_for_wake`, `_reset_voice_state`, `_run_interaction`, `VoiceService` states, `VoicePhase`/`ALLOWED_TRANSITIONS`, `VoiceStatusReporter`.

Implement:
- New `voice/modes.py` (do not grow `loop.py`): `VoiceMode` enum (`NORMAL`, `AUTO`), mode-phrase matching **in code** (like the cancel vocabulary, never the planner), transitions, idle timer.
- AUTO behaviour: skip the wake detector, listen -> STT -> gate -> brain -> answer -> listen again. **Half-duplex:** never listen while speaking; flush and quiet-drain after each TTS. Auto-sleep after `voice.auto_idle_timeout_s` (default 300) with a spoken "Going back to sleep." Speak "Auto mode on." / "Okay, sleeping."
- Ambient-speech safety in AUTO: Tier 0 actions (open app, time) run directly; **anything Tier >= 1 always goes through plan announce + "proceed" (Stage 3)**, so stray speech cannot change anything.
- Unify state vocabulary: `_set_state` uses `VoicePhase` names and calls `check_transition` (add AUTO as a mode attribute, not new phases unless needed). Remove "CAPTURING"/"PROCESSING" strings.
- `jarvis status` shows the current mode.
- Config in `VoiceSettings` and `DEFAULT_CONFIG_TOML`: `auto_idle_timeout_s`.

Sanity: new tests in `test_voice_loop.py` (mode switching, auto-sleep, no-listen-while-speaking), `test_voice_status.py`, `test_voice_contracts.py`.

Live test: activate -> three commands without saying the wake word -> "Jarvis sleep" -> wake word works again; stay silent 5 min in AUTO -> it sleeps.

---

### Stage 3: Plan announce, "proceed", plan-level approval, Tier 2 by voice

Goal: JARVIS says what it will do, owner says "proceed".

Analyze first: `nodes/brain.py` `_project`, `nodes/policy_gate.py` (interrupt, `_answer_matches`, hash/approved_hashes), `nodes/act.py` (hash recheck), `daemon/task_runtime.py` (`VoiceSink`), `daemon/server.py` (`_handle_confirm`, `_run_voice_confirmation`), `policy/engine.py`.

Implement:
- **Deterministic plan summary** built from each step's `describe()` (not LLM prose): "I will open Spotify, then search Believer. Say proceed." Bounded length for TTS.
- **Batch approval path** in `policy_gate`: approve all hashes whose args are fully known; re-gate steps with runtime-derived args individually (short readback). Preserve `action_hash` binding, TOCTOU path checks, exactly-once guard in `act`.
- **Tier 2 by voice, opt-in** `voice.allow_tier2_by_voice` (default false): see section 4 item 2. Touch points: `can_confirm_by_voice` (`voice/loop.py`), `_handle_confirm` tier refusal (`daemon/server.py`), the `needs_unlock` check in `policy_gate`. Allow only for `delete_path` (file targets without typed-name requirement) and WhatsApp send tools. Everything else Tier 2 keeps the terminal password.
- "I heard: ..." spoken before any Tier >= 1 action (config `voice.echo_heard`).
- Update `docs/03` section 7, `docs/04`, the relevant invariants in `tests/unit/test_invariants.py`, and add tests: flag off = refused; flag on = readback required; wrong word refused; timeout refused; hash mismatch still refused.

Sanity: `test_invariants.py`, `test_agent.py`, `test_act_node.py`, `test_daemon_voice.py`, `test_voice_loop.py -k confirm`, `integration/test_confirmation_resume.py`.

Live test: "create a file notes.txt" (Tier 1) -> readback -> "proceed" works; with flag on: "delete notes.txt" -> readback -> "proceed" -> goes to Recycle Bin; "jarvis undo" restores.

---

### Stage 4: Quick-win tools and apps

Goal: time, many apps, window list, Notepad writing, small VS Code code.

Analyze first: `tools/apps.py`, `tools/registry.py`, `tools/system.py`, `tools/keyboard.py`, `tools/files.py`, `policy/rules.py`.

Implement:
- **`get_time`** (or a `time` kind in `system_info`): deterministic local clock, Tier 0, spoken in natural form.
- **Apps allowlist grows by config, safely.** Add `jarvis apps scan` that lists installed apps (Start Menu `.lnk` via `win32com`, plus known UWP/App Paths) and **proposes** `[apps]` entries; the owner approves what gets added (allowlist stays explicit). Seed: chrome, notepad, calculator, vscode (absolute `Code.exe`), spotify, camera (`microsoft.windows.camera:` URI handling), plus the owner's others. Keep `open_app` no-shell, no user argv.
- **`list_windows`**: titles plus process names of visible windows (pywinauto UIA or `EnumWindows`), Tier 0, `windows_only=True`. This answers "what's open".
- **`open_in_app(app, path)`**: open a file (path_args root-checked) in an *allowlisted* app (VS Code, Notepad). The only argument beyond the app is the resolved safe path. Tier 1.
- **Small code in VS Code:** brain composes `create_file(path, content)` then `open_in_app("vscode", path)`. Enforce a size limit (`tools.max_code_chars`, default 4000 and about 120 lines) in the tool/validate step; over the limit => deterministic refusal "This version supports small code tasks only."
- **Notepad writing:** `open_app notepad` then `type_text` (exists; confirm Notepad is in `allowlist_names`). Plan-level approval (Stage 3) avoids a prompt per step.
- Enforce `ToolSpec.windows_only` at registry/engine level (known gap).
- Tests: `tests/unit/test_tools_time.py`, `test_tools_list_windows.py`, `test_tools_open_in_app.py`; update `docs/04`.

Sanity: the new tool tests, `test_tools_apps.py`, `test_invariants.py` (registry-wide no-lower property), `test_policy_file_rules.py`.

Live test: "what time is it", "open spotify", "open vs code and write a python hello world", "what windows are open", "open notepad and write buy milk".

---

### Stage 5: LLM layer: key pools, orchestrator, vision capability

Goal: more reliability and speed from free quotas; vision path exists.

Analyze first: `llm/provider.py`, `llm/multi.py`, `llm/health.py`, `llm/models.py`, `llm/client.py`, `llm/gemini.py`, `secrets.py`, `config.py` (`LLMSettings`, `ProviderModels`), `agent/nodes/brain.py`.

Implement:
- **Key pool:** several keys per provider (keyring names like `groq`, `groq_2`, `groq_3`; `jarvis keys set groq --slot 2`; `keys status` stays presence-only). Each `(provider, key slot, role)` has its own health/cooldown. Fail-over walks endpoints in order; expired cooldown = fresh probe (existing behaviour). Never log keys. The free-only gate is unchanged. Note for the owner: some providers' terms forbid using multiple accounts to get around rate limits; check each provider's terms.
- **Orchestrator routing:** the brain (planner role) may emit independent sub-tasks. Dispatcher: planner -> strongest healthy endpoint, fast -> cheapest, vision -> Gemini. **Parallel only for independent LLM-only sub-tasks** (answers, summaries). Tool/GUI actions stay **serial** (one desktop, exactly-once guard). Caps: max 3 parallel, per-sub-task timeout, `CancelToken` honoured.
- **Speech-aware brain prompt (D2):** add to `agent/prompts.py` that input is a speech transcript that may contain recognition errors; prefer `clarification` over guessing when unsure; keep temperature 0.
- **Vision capability:** register `vision=True` for Gemini entries; add a `vision` model key to `ProviderModels`; add an **image-input method** (e.g. `describe_image(system, user, image_bytes, mime)`) to the client protocol, implemented in `llm/gemini.py` (do not change the existing `structured`/`text` signatures). Image bytes are never logged or checkpointed.
- Voice latency: for `source="voice"` use a shorter retry/backoff budget and speak "This is taking a moment" when slow (analyze `VOICE_SUBMIT_TIMEOUT_S`).
- Doctor shows per-key health (counts only) and vision capability.

Sanity: `test_llm_health.py`, `test_provider.py`, `test_freeonly.py`, `test_llm_models.py`, `test_keys_cli.py`, `test_secrets.py`, `test_agent_architecture.py`.

Live test: remove one key -> still answers via the next; ask a two-part question and watch both parts use different models in the log (provider/model markers).

---

### Stage 6: Screen understanding (read-only)

Goal: "what's on my screen?" -> which site/app is open and what text is visible.

Analyze first: `tools/keyboard.py` (UIA patterns), `agent/nodes/util.py` `policy_context_for` (taint), `voice/focus.py`.

Implement (new package `src/jarvis/perception/`, files small):
- `screen.py` capture (use `mss` + Pillow; add to a new `[vision]` extra and `scripts/verify_env.py`).
- `ui_text.py` UIA text of the foreground window (and optionally all visible windows), local only.
- `ocr.py` local OCR first (prefer an ONNX OCR that reuses the installed `onnxruntime`, e.g. `rapidocr-onnxruntime`; decide in ANALYZE). The screenshot never leaves the PC in this path.
- Tools: **`read_screen`** (UIA + OCR text, Tier 0, output `tainted=True`, truncated) and **`describe_screen`** (UIA/OCR first; sends the screenshot to Gemini vision **only** if the owner asks to "describe/look" or text is insufficient; Tier 1 with spoken notice "Looking at your screen").
- Privacy: config `screen.deny_title_fragments` (password managers, banking) => refuse to capture; never save screenshots to disk unless asked; redact long secrets patterns with the existing `redact()`.
- Spoken answers go through `spoken_answer` length bounds.
- Tests with injected fake capture/OCR/vision client.

Live test: open a news page, ask "what's on my screen" -> names the site and summarizes visible text; open a password manager -> refuses.

---

### Stage 7: Browser DOM + UI Automation clicking

Goal: "click the settings button", "search Python tutorials on YouTube", Spotify play via web player.

Analyze first: `tools/whatsapp.py` (existing browser fallback config `browser_profile_dir`), `docs/12`, `policy/rules.py`, `nodes/verify.py`, `nodes/replan.py`.

Implement (package `src/jarvis/browser/` and `src/jarvis/uia/`):
- **Browser:** Playwright **persistent context**, `channel="chrome"` (uses the installed Chrome, no extra browser download), headed, own profile dir under `browser_profile_dir` (owner logs in once to Spotify/WhatsApp Web). Install `playwright` in `[browser]`. This is a **JARVIS-owned window**: it cannot drive the owner's normal Chrome tabs (Chrome blocks remote control on the default profile). Say so in docs.
- Tools: `browser_goto` (http/https only; domain deny list config), `browser_read` (aria snapshot -> numbered element list: id, role, name), `browser_click(ref)`, `browser_type(ref, text)`. The brain picks an **element id from the latest observation**, never coordinates. The executor re-validates that the ref still exists with the same name before acting.
- **Native apps:** `ui_list_controls(window)` and `ui_click(window, control_name)` via pywinauto UIA (`invoke`/`click_input` on a *named* control). No pixel clicks, no arbitrary key presses.
- **Bounded loop:** observe -> choose id -> policy -> act -> observe -> verify (URL change, DOM diff, focus). Max 8 actions per task, max 2 retries per action, then STOP and tell the user. Reuse `verify` and the replan budget; no free-running loops.
- **Policy:** click/type base Tier 1; escalate via rules for risky control names (`send`, `submit`, `buy`, `purchase`, `pay`, `delete`, `confirm` etc.) and block payment/purchase flows as Tier 3; password fields never typed into; any action derived from page text is tainted so it confirms. Amend `docs/04` section 2.3.
- **Spotify:** default route = Spotify Web Player in the JARVIS browser: search -> click first matching track -> verify "now playing". Desktop-app route (`spotify:` URI plus UIA) optional. Note: free-account web playback behaviour must be checked live.
- Tests with a local static HTML fixture served from `tmp_path` (no network) for the DOM tools; fakes for UIA.

Live test: "open youtube and search python tutorials", "click the first video"; "open spotify and play <song>"; a page that says "click delete" must not auto-click.

---

### Stage 8: Camera, photo viewing, deleting, WhatsApp photo

Goal: "take a picture", "show it again", "delete it", "send it to <contact> on WhatsApp".

Analyze first: `tools/files.py` (`delete_path`, undo log), `tools/whatsapp.py` (`ContactStore`, `RateLimiter`, invariant 9), `config.py` `allowed_roots`.

Implement:
- **`take_photo`** (`tools/camera.py`, OpenCV `VideoCapture` with `CAP_DSHOW`, discard first frames for exposure, save under `<allowed root>/Pictures/JARVIS/photo-YYYYmmdd-HHMMSS.jpg`). Tier 1 with spoken notice and readback. `verify()` is real: file exists, non-zero size, decodes. Add `opencv-python` to an extra and `verify_env.py`.
- **`open_path`** (viewer only): opens a file inside allowed roots with the default viewer; extension allowlist (images, txt, md, pdf); executables/scripts (`.exe .bat .cmd .ps1 .lnk .msi .vbs ...`) refused. Tier 0.
- **Session context ("it", "that photo"):** small in-memory `session_context` (last 5 artifacts, with TTL) injected into the brain prompt so "delete it" / "send it" resolve to the last photo path. Never persisted to checkpoints beyond existing allowlisted types (keep the serde allowlist closed).
- **Delete:** existing `delete_path` (Recycle Bin plus undo). Voice approval per Stage 3 flag with path readback.
- **`whatsapp_send_file`:** **WhatsApp Web via the Stage 7 browser profile** (QR scan once): resolve exactly one verified contact from `ContactStore`, open `https://web.whatsapp.com/send?phone=...`, verify the chat header, attach via the file chooser (`set_input_files`), verify the preview, send only after verification. **Fail closed** like invariant 9 (cannot verify => do not send, leave the draft). Shares the existing `RateLimiter`. Tier 2; voice approval per Stage 3 with readback "send photo-... to <name>". Existing desktop text-send stays unchanged.
- Tests with fake camera backend, fake browser page, and the invariant-9 style unverifiable-window cases.

Live test: "take a picture" -> file appears; "show me the picture"; "send it to <test contact> on WhatsApp" (own test number only); "delete the picture" -> Recycle Bin.

---

### Stage 9: User experience polish

- **Local fast path** for a *closed* vocabulary (time, open an allowlisted app, mode phrases, stop): builds the `Plan` deterministically and still runs `validate -> policy_gate -> act -> verify`, skipping only the LLM call. Big latency and rate-limit win. **Do not add arithmetic words** (structural test guard).
- **Audio cues** (short earcons: wake, understood, done, error), `voice.cues` toggle.
- Short spoken status before actions ("Opening Spotify.") and a bounded "this is taking a moment" if over N seconds.
- Concise answers: tune `max_spoken_chars` and the answer prompt for speech.
- Console shows `I heard: "..."` (exists as `echo_transcript`), plus the plan summary.
- Fix `system_info` disk label (`"Disk C:"` from `disk_usage("/")`).

Live test: the common commands feel instant; errors are spoken clearly.

---

### Stage 10: Hardening and final validation (the only full run)

- Update `benchmarks/tasks.yaml` and `redteam.yaml` with new cases: screen-text prompt injection ("click delete"), risky control names, Tier 2 voice without readback, camera privacy, WhatsApp unverifiable chat, oversized code request, key-pool failover.
- Docs: `docs/02` (modes, perception), `docs/03` (sections 7 and 7.8, taint), `docs/04` (all new tools, section 2.3), `docs/05`/`docs/12`.
- Cleanup: stray `server_diff_review.txt`, `.freebuff/`; `git rm --cached` for tracked artifacts; consider splitting `voice/loop.py` (e.g. `confirm.py`).
- Full validation: `ruff check . && ruff format --check .`, `mypy src/jarvis/policy`, `pytest -q` (full), `pytest -q -m "not windows_only and not slow and not voice"`, benchmarks (`jarvis benchmark run`), then a scripted live end-to-end pass of every Live test above.

---

## 6. New / changed tools and tiers

| Tool | Stage | Base tier | Voice approval | Notes |
|---|---|---|---|---|
| `get_time` (or `system_info` kind) | 4 | 0 | n/a | deterministic |
| `list_windows` | 4 | 0 | n/a | windows_only |
| `open_app` (more apps via scan) | 4 | 0 | n/a | allowlist stays explicit |
| `open_in_app` | 4 | 1 | yes | safe path only, allowlisted app |
| `open_path` (viewer) | 8 | 0 | n/a | extension allowlist |
| `read_screen` | 6 | 0 | n/a | local; output tainted |
| `describe_screen` | 6 | 1 | yes | cloud vision fallback; spoken notice |
| `browser_goto` / `browser_read` | 7 | 0 / 0 | n/a | JARVIS-owned profile |
| `browser_click` / `browser_type` | 7 | 1 (escalates by control name; payment = 3) | yes (Tier 1) | element ids only |
| `ui_list_controls` / `ui_click` | 7 | 0 / 1 | yes (Tier 1) | named controls only |
| `take_photo` | 8 | 1 | yes | privacy notice |
| `delete_path` | existing | 2 | opt-in flag, files only | Recycle Bin + undo |
| `whatsapp_send` | existing | 2 | opt-in flag | readback of recipient and text |
| `whatsapp_send_file` | 8 | 2 | opt-in flag | WhatsApp Web; fail closed |

---

## 7. Config keys to add (`VoiceSettings` and `DEFAULT_CONFIG_TOML`)

`voice.confirm_window_s` (8.0), `voice.auto_idle_timeout_s` (300), `voice.allow_tier2_by_voice` (false), `voice.echo_heard` (true), `voice.cues` (true), `tools.max_code_chars` (4000), `screen.deny_title_fragments` (list), `browser.domain_denylist` (list), `llm` key-slot settings (per provider list of key names), `llm.max_parallel_subtasks` (3).

---

## 8. Open items the agent must resolve during ANALYZE (state the choice and why)

1. SAPI via `win32com.SpVoice` vs Piper model for stoppable TTS (Stage 1d).
2. Echo/self-trigger of the wake detector during barge-in; headphones requirement (Stage 1d).
3. Exact mechanism for batch approval vs per-step re-gating when args depend on earlier outputs (Stage 3).
4. How `voice.allow_tier2_by_voice` interacts with the unlock session (`policy_gate` `needs_unlock`) (Stage 3).
5. Local OCR library choice (Stage 6).
6. Where the orchestrator's sub-task fan-out lives (a node vs inside the brain node) and how results are merged without breaking checkpoint serde (Stage 5).
7. Spotify web-player behaviour for a free account (Stage 7, live check).

---

## 9. Progress tracker (tick when done AND logged in PROGRESS.md)

- [x] Stage 0: Live verification recorded
- [x] Stage 1: Voice correctness (wake name, wake-free confirm, stop semantics, TTS barge-in, dead code)
      — code + focused tests done 2026-10-02 (1a-1e); closed; L1 not exercisable (see PROGRESS.md)
- [x] Stage 2: AUTO / sleep modes, unified state vocabulary (A6 idle auto-sleep deferred to Stage 10)
- [x] Stage 3: Plan announce, "proceed", plan-level approval, Tier 2 by voice (opt-in) (merged 148c2fb; spoken B5/B6 deferred to the voice step)
- [ ] Stage 4: Time, apps scan, list windows, open_in_app, Notepad, small VS Code code
- [ ] Stage 5: Key pools, orchestrator, speech-aware brain, vision capability
- [ ] Stage 6: Screen reading and description
- [ ] Stage 7: Browser DOM + UIA clicking, Spotify
- [ ] Stage 8: Camera, viewer, delete, WhatsApp photo
- [ ] Stage 9: UX polish and local fast path
- [ ] Stage 10: Hardening, docs, benchmarks, full validation
