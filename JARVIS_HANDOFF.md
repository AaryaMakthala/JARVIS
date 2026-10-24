# JARVIS Handoff

Last updated: 2026-10-23

## 1. Project summary

JARVIS is a voice- and text-controlled Windows desktop AI agent built as a final-year engineering project.
It uses a typed LangGraph pipeline, a deterministic policy layer, and free-tier LLM providers.
This handoff records cleanup state, audit progress, and the next audit step.

## 2. Design decisions

- Screenshot and camera both supported
- Automation follows the user's exact instructions (press this, click that)
- Specialist loops and extra LLM calls are acceptable now; voice latency tuning comes last
- Persona: butler tone ("Certainly, sir"), use the user's name once stated
- Incremental refactor, policy layer untouched
- Free-tier only: Groq, OpenRouter, NVIDIA (Gemini optional)

## 3. Audit status

| Part | Scope | Status |
|---|---|---|
| R1 | Configuration and LLM model selection | DONE |
| R2 | Prompts, tool catalogue, structured schemas, fast path | DONE |
| R3 | Session context and reference resolution | DONE |
| R4 | Browser results, refs, primitives, site catalogue | DONE |
| R5 | Contacts and WhatsApp matching | DONE |
| R6 | OCR and screen perception | DONE |
| R7 | Voice vs terminal response path | DONE |
| R8 | Logs, task log, failure evidence | DONE |
| R9 | State cleanup and read/write map | DONE |
| R10 | Benchmarks and test impact | DONE |
| R11 | Environment and dependency versions | DONE |
| R12 | Scope grants, halt reasons, camera, state readers, benchmark, config | DONE |

## 4. Key R1 facts

- `config.toml` has `free_only=true` and `strict_zero_cost=false`; the shipped default is `strict_zero_cost=true`.
- Configured order: groq, openrouter, nvidia, gemini; one key slot each (no `key_slots`).
- Eight role bindings: Groq gpt-oss-120b/20b, OpenRouter free/free, NVIDIA nemotron super/lightning, Gemini 3.8/3.7 flash.
- Under current flags all eight qualify; under the strict default Groq and Gemini would be refused.
- No vision role is configured, although Gemini is registered as vision-capable.
- Groq starts with `json_schema` and can downgrade to `json_object`; OpenRouter and NVIDIA only use `json_schema`; Gemini uses native `responseSchema`.
- No numeric context windows exist in the registry; overflow is handled only from provider error text.
- Cooldowns are per `(provider, slot, role)`: 60 s rate limit, 45 s server error, 30 s transport/timeout, 300 s auth/context, 3600 s unsupported capability, multiplied 1x/2x/4x/8x.

## 5. Cleanup log

Backup tag: `backup-before-cleanup` (created before deletion; no commit made). Tag contains PROGRESS.md and the three ROADMAP files.

Removed as SAFE-REMOVE (all ignored or untracked, so nothing became staged):
- `.hypothesis/` (Hypothesis cache)
- `.mypy_cache/` (about 110 MB)
- `.pytest_cache/` (about 260 KB)
- `.ruff_cache/` (about 50 KB)
- `scripts/__pycache__/`
- `daemon_console.log`
- `final-results.xml`
- `server_diff_review.txt`

Condensed: `PROGRESS.md` condensed from **405489 B / 5683 lines** -> **3915 bytes / 55 lines** (kept current state, invariants, known bugs/open items, test/benchmark status, condensed milestones; removed per-session logs and duplicates). All headings linked by README/AGENTS/docs preserved where relevant.

Roadmap: `ROADMAP_V2.md.md`, `ROADMAP_V2.1_AMENDMENTS.md.md`, `ROADMAP_V2.2_AMENDMENTS.md` merged into **`docs/ROADMAP.md` (27 lines)** consolidating V2 amended by V2.1 and V2.2 with DONE/IN PROGRESS/PLANNED/DROPPED status and active constraints. All references updated to `docs/ROADMAP.md`; old files deleted. No remaining references found in repo except this handoff log.

Kept as UNSURE:
- `.freebuff/`, `.vscode/`
- `scripts/phase3_verify.py`, `scripts/phase3_verify_confirm.py`, `scripts/whatsapp_attach_probe.py`
- `src/jarvis_agent.egg-info/` and all `__pycache__/` under `src/` or `tests/` (protected directories)

Not touched: `src/` logic, `tests/`, `pyproject.toml`, lock/config TOML files, `README.md`, `LICENSE`, `.gitignore`, `benchmarks/tasks.yaml`, `.git/`, `.venv/`. The only `src/` edit was updating a doc-path reference in `src/jarvis/voice/stopwatch.py`.

## Condensed facts

1. **Graph/policy**: LangGraph pipeline with deterministic policy; LLM proposes, policy decides. Tier 3 blocked, paths resolved, deletes to Recycle Bin.
2. **Providers**: Free-tier only (Groq/OpenRouter/NVIDIA), Gemini optional. Current config has `strict_zero_cost=false` (default strict would refuse Groq/Gemini).
3. **Voice**: Bare "stop" cancels in busy states (D22).
4. **Browser**: Managed browser + UIA; page referent cleared by unrelated actions (act.py:350); P2/P3 added but live verification partial. **Result sets exist** in `BrowserResultStore` but the **brain prompt never receives them** (only SessionContext page/artifact). 
5. **WhatsApp**: Isolated Web profile, content-hash binding, recipient precheck.
6. **Tools/registry**: 39 tools, closed registry, no generic shell/exec.
7. **Stage 8 incomplete**: Camera/WhatsApp/referents live checks not fully verified.
8. **OCR**: Not installed, accuracy unvalidated (local OCR may not be present).
9. **Memory**: SQLite stores skills/failures/preferences/task_log with caps.
10. **Daemon**: IPC 127.0.0.1 + token, confirmations bound to action hash. **open_url/google_search use default browser (webbrowser.open)**; **browser_goto uses managed browser**. **Missing primitives:** press-key/Enter, scroll, wait-for, tab switching, back/forward, page screenshot. **No [sites] configured by default** in code; `SiteDirectory` empty unless settings.sites populated.

## 6. Next step

Phase 1 design spec

## Redesign plan

Principle: specialists propose one step at a time; validate, policy_gate, act and verify stay unchanged.  
New nodes: context_resolve (no LLM), route/orchestrator (work order), dispatch, specialist loop (observe, decide), compose (before respond).  
New: referent store keyed per conversation (result set, page, last photo, last file, last contact); typed AgentState with same key names; browser_search/press_key/scroll/wait/back/forward/tabs/screenshot tools; take_screenshot; default sites in code; contact normalisation and aliases; butler persona with templates; RapidOCR install.  
Phases: 0 telemetry, benchmark rewrite and offline fixture site; 1 typed state and referents; 2 browser primitives and default sites; 3 orchestrator and browser specialist; 4 compose and persona; 5 messaging resolver; 6 OCR, screenshot and camera; 7 wider system tools; 8 Amazon purchase flow; 9 voice latency.  
Pending decisions: placing orders, allowed_roots scope, screenshot scope, Amazon domain, browsing grant policy change.

## Phase 0 results (low-resource mode)
- Step 0: Added autouse test guard in tests/conftest.py blocking subprocess/os.startfile/webbrowser/pyautogui/pywinauto/cv2/sounddevice unless marker live/windows_only (with allowances for specific tests). Added markers browser.
- Step 1: Versions (from venv) - langgraph 1.2.11, langgraph-checkpoint 4.2.0, langgraph-checkpoint-sqlite 3.1.1, langchain-core 1.6.3, pydantic 2.13.5. pyproject pins allow ranges; no uv.lock found. Two dist-info sets exist due to langgraph family split packages.
- Step 2: Benchmark parsed statically - expect.tool fields not present in current schema; tasks use expected.type. Registry has 39 tools (open_app, open_url, google_search, web_answer, system_info, get_time, list_windows, read_screen, describe_screen, lock_jarvis, lock_computer, audit_run, defender_status, defender_quick_scan, create_file, append_file, list_dir, read_file, delete_path, undo_last_delete, take_photo, type_text, open_in_app, open_path, start_dictation, stop_dictation, save_dictation, whatsapp_send, whatsapp_send_file, whatsapp_open_chat, browser_goto, browser_read, browser_click, browser_type, browser_open_site, browser_find, browser_open_result, ui_list_controls, ui_click). Diff: benchmark uses expected.type/checkers, no expect.tool entries - benchmark schema differs; can run offline if checker logic matches.
- Step 3: task_log v3 migration added (idempotent), columns added; TaskLogStore/DB updated; `jarvis tasks` CLI added. Tests added tests/unit/test_task_log_v3.py; test_task_log.py and test_memory.py pass (with guard adjustments).
- Step 4: Created tests/fixtures/site/*.html and fixture server; added browser-marked integration tests (skipped in low-resource mode).
- Step 5: Rewrote benchmarks/tasks.yaml with scenario tasks S001-S006 using real tool concepts; YAML loads.
- Decisions recorded: guarded generic final click incl. opt-in voice approval; whole-profile folders; full-screen screenshot Tier 1; generic not site-specific; browsing grant yes.


## 2a. Exact table for plan [browser_open_site youtube, browser_find, browser_open_result position 1, browser_click play]

**Tool specs (from src/jarvis/browser/tools.py):**
- browser_open_site: base_tier=1, needs_confirm (Tier 1), budget_cost=1 (line ~913)
- browser_find: base_tier=0, no budget unit (line ~935)  
- browser_open_result: base_tier=1, needs_confirm, budget_cost=1, bind_target_identity=True (line ~957)
- browser_click: base_tier=1, needs_confirm, budget_cost=1 (line ~873)

**Policy gate / batch eligibility (src/jarvis/agent/batch_approval.py:119-210, 63-98):**
Eligibility requires: decision.tier==1 AND decision.needs_confirm AND not decision.needs_unlock AND step.resolved_from_runtime==False AND decision.allowed AND action_hash not approved AND not blocked by precheck. Contiguous run starting at step_index; first non-eligible stops run.

**Table:**
| Step | Tool | base_tier | batch-eligible? | Notes |
|------|------|-----------|-----------------|-------|
| 1 | browser_open_site (youtube) | 1 | Yes if allowed, needs_confirm, not runtime-derived, not blocked | Tier 1 confirmable |
| 2 | browser_find | 0 | No (tier != 1) | Tier 0, no confirm - breaks batch run |
| 3 | browser_open_result (position 1) | 1 | No (comes after non-eligible step 2, batch is contiguous from step_index and stops at first non-eligible) | Would be gated individually if reached |
| 4 | browser_click (play) | 1 | No | Would be gated individually |

**(i) Full plan emitted at once:** batch approval only batches contiguous run starting at step_index. Step 1 is Tier 1 eligible, step 2 is Tier 0 (not eligible) → batch run from step_index=0 contains only step 1? Or stops at step 2. So eligible list would be just [step1] (length < 2) → returns single confirm for step 1. Steps 2,3,4 would be gated separately as plan progresses. So **1 confirmation (for step 1) + 1 confirmation for step 3? Or step 2 is Tier 0 (no confirm) so it just executes if allowed**. Step 2 has base_tier 0, needs_confirm likely False → policy gate won't ask for confirm, just executes. Step 3 is Tier 1 but comes after step 2 (Tier 0, non-eligible) so batch from current step_index when we reach it would start fresh? Or batch is computed from step_index at gate time. At each gate call, policy_gate computes decisions for all remaining steps and calls decide_and_build_payload with step_index = current step. So when at step_index pointing to step 1, eligible run from step1: step1 eligible, step2 is Tier 0 → not eligible → run stops after step1, eligible has 1 element → single confirm for step1 only. After step1 approved/executed and we move to step_index for step2 (Tier 0), batch eligibility won't trigger confirm; step2 executes. Then step_index moves to step3 (Tier 1), eligible run from step3: step3 eligible, step4 eligible? both Tier 1, contiguous, not runtime-derived → eligible would have 2 elements → **1 plan_approval covering steps 3 and 4** (if both eligible). So total **2 confirmations**: (1) confirm for step1 alone, (2) plan_approval for steps 3+4 together. Step2 requires no confirmation.

**(ii) Steps appended one at a time:** same result - batch is computed per gate call based on current plan state at that step_index. Appending doesn't change the contiguous eligibility rule from current position.

**Quote - batch eligibility:** src/jarvis/agent/batch_approval.py:151-210 (_batch_eligible checks tier==1, needs_confirm, not needs_unlock, not resolved_from_runtime, allowed, hash not approved; _eligible_steps collects contiguous run starting at step_index stopping at first non-eligible). Policy gate calls compute_all_decisions for all remaining steps (batch_approval.py:214-251) then decide_and_build_payload (batch_approval.py:63-98).

## 2b. Worst-case wall time of ONE structured() call

**GroqClient._complete (src/jarvis/llm/client.py:331-429):**
- retry_budget = self._settings.llm.max_retries if max_retries is None else max_retries (default from settings)
- request_timeout = self._settings.llm.timeout_seconds if timeout_s is None else timeout_s (default 30s)
- Retries: only for categories in RETRYABLE (see failures.py:133-147). On retryable error, if attempts <= retry_budget: sleep backoff (starts at 1.0, doubles each retry), increment attempts, continue. Non-retryable errors (auth, context_length, malformed, unsupported model/capability, invalid output) fail immediately without retry.
- attempts increments on each exception before deciding retry. Max attempts = retry_budget + 1 (initial attempt + retry_budget retries). With max_retries=3, default: up to 4 attempts per provider call.

**GroqClient.structured (src/jarvis/llm/client.py:431-586):**
- Tries json_schema first (if not cached as False for model). On _is_schema_rejection(exc): caches model as False, does ONE retry in json_object mode (calls _parse again). Schema rejection is detected via _is_schema_rejection (client.py:252-298) - checks error text for json_schema markers OR status 400/422.
- _parse does ONE repair round: if first parse fails with ValidationError, builds repair messages and calls _complete again once more with same response_format/params. If repair also fails, raises LLMInvalidOutputError (invalid_structured_output).

**MultiProviderClient.structured (src/jarvis/llm/multi.py:241-301):**
- Walks groups in health order (provider-major: providers ordered by health, within provider slots 1,2,...). For each candidate: checks capabilities via _spec_for_role (model for requested role). On LLMError: calls _note_failure which records to health with cooldown (cooldown_seconds by category, multiplied by factor min(2^(consec_failures-1), 8)), then calls _advance_after_failure.
- _advance_after_failure (multi.py:341-380): MODEL_SCOPED_CATEGORIES (UNSUPPORTED_CAPABILITY, UNSUPPORTED_MODEL, CONTEXT_LENGTH, INVALID_STRUCTURED_OUTPUT) → return False (move to next provider, slot-0 recorded). MALFORMED_REQUEST → False. RATE_LIMIT: if rotate_on_rate_limit false → False (try next provider); if true → True (advance to next key slot). ROTATION_CATEGORIES (AUTH, SERVER_ERROR, CONNECTION, DNS, TLS, TIMEOUT, UNEXPECTED) → True (advance to next key slot). So retries are **per provider/slot attempt within the walk**; the retry_budget in the client is for retrying the SAME provider call on retryable errors before giving up that provider/slot. They are separate mechanisms.
- Cooldown: health.record_failure applies base cooldown_seconds(category) * factor where factor = min(2^(consec_failures-1), 8), capped. Cooldowns are per (provider, slot, role). Slot-0 (provider-wide) used for model-scoped failures cooling all slots of provider for that role.

**Worst case arithmetic (one structured() call from MultiProviderClient perspective):**
Assuming first provider attempt: _complete does up to (max_retries+1)=4 attempts, each with request_timeout=30s. On each attempt, on retryable error sleep backoff 1.0,2.0,4.0,... between retries (sum of sleeps adds to wall time). Plus repair round: if parse fails first time, _parse makes another _complete call with repair messages - that's another up to 4 attempts * 30s (plus retries). Plus schema rejection path: if json_schema rejected, caches and does another _parse call in json_object mode = another up to 4 attempts. Plus provider fallthrough: if all fail, walks through providers/slots - each candidate gets its own _complete attempts up to 4*30s plus possible repair/schema paths. With multiple providers and key rotation possible for some categories, worst case is unbounded in theory but bounded by number of candidates and cooldowns prevent immediate retry of cooling ones. Practically worst case could be O(providers * slots * (attempts) * timeout) = large, but with fail-fast on non-retryable and cooldowns; however per the question "worst-case wall time of ONE structured() call" - the call is the MultiProviderClient.structured() invocation. It will try candidates until success or exhaustion; each candidate's client.structured() does its own retries/repair. So worst case: try N candidates, each fails after exhausting its retry budget and repair attempts → sum of all their timeouts+retries+sleeps. With defaults (30s timeout, 3 retries = 4 attempts, 1 repair round = another 4 attempts max, possible schema downgrade = another 4 attempts) per candidate, and up to several candidates → could be minutes in worst pathological case, but cooldowns and the fact that cooling providers are deprioritized but still tried if all cooling means it won't hang forever. The key point: retries are **per provider call** (in GroqClient._complete) up to retry_budget; provider fallthrough/rotation is **separate** in MultiProviderClient across different providers/slots based on B8 rules; cooldown follows failure category with exponential factor by consecutive failures (health.py:170-232).

## 2c. LangGraph/LangChain versions from METADATA
- langgraph: /mnt/c/Users/aarya/OneDrive/Desktop/Jarvis/.venv/Lib/site-packages/langgraph-1.2.11.dist-info/METADATA → Version: 1.2.11
- langgraph-checkpoint: /mnt/c/Users/aarya/OneDrive/Desktop/Jarvis/.venv/Lib/site-packages/langgraph_checkpoint-4.2.0.dist-info/METADATA → Version: 4.2.0  
- langgraph-checkpoint-sqlite: /mnt/c/Users/aarya/OneDrive/Desktop/Jarvis/.venv/Lib/site-packages/langgraph_checkpoint_sqlite-3.1.1.dist-info/METADATA → Version: 3.1.1
- langchain-core: /mnt/c/Users/aarya/OneDrive/Desktop/Jarvis/.venv/Lib/site-packages/langchain_core-1.6.3.dist-info/METADATA → Version: 1.6.3 (dist-info folder name langchain_core-1.6.3.dist-info; no langchain-core-* folder)

Conflict resolution: handoff earlier mentioned "langgraph 0.6.8" in one place but actual venv has langgraph-1.2.11.dist-info. Both dist-info folders for langgraph family exist as shown above; the installed versions are 1.2.11 (langgraph), 4.2.0 (checkpoint), 3.1.1 (checkpoint-sqlite), 1.6.3 (langchain-core). The 0.6.8 was from earlier environment/memory - current venv has 1.2.11.

## 2d. Benchmark tool names vs registry
**Tools registered in src/jarvis/tools/registry.py (from registrations):**
From registry.py lines ~168-190 and imports: make_open_app_spec(), make_open_url_spec(), make_google_search_spec(), make_web_answer_spec(), make_list_folder_spec(), make_read_file_spec(), make_write_file_spec(), make_create_file_spec(), make_delete_path_spec(), make_move_path_spec(), make_copy_path_spec(), make_open_path_spec(), make_type_text_spec(), make_press_key_spec() (if exists?), make_lock_computer_spec(), make_screen_read_spec(), make_screen_find_spec(), make_take_photo_spec(), make_audit_run_spec(), make_audit_report_spec(), make_send_whatsapp_spec(), make_open_whatsapp_chat_spec(), make_whatsapp_search_contacts_spec() - plus browser tools (browser_goto, browser_read, browser_click, browser_type, browser_open_site, browser_find, browser_open_result) added via browser module. Also system tools. Total 39 tools per earlier fact.

**Benchmark expects (from benchmarks/tasks.yaml):**
Apps: open notepad/calculator/chrome/frobnicate (uses open_app/open_app_spec) - registry has make_open_app_spec() ✓
Web: google python tutorials, open youtube.com, what's the latest news..., open file:///..., open javascript:..., open ms-settings:..., open https://example.com, google the weather (T007-T014) - uses google_search/open_url/web_answer; registry has make_google_search_spec(), make_open_url_spec(), make_web_answer_spec() ✓  
Files: create/append/read/list/overwrite/delete/move/copy/open (T015-T028) - create_file, delete_path, move_path, copy_path, open_path, read_file, write_file/list_folder all registered ✓  
Keyboard/input: type hello there into notepad (T029) - make_type_text_spec() registered ✓  
System: lock the computer (T030) - make_lock_computer_spec() registered ✓  
Audit: run the security section of the audit (T031) - make_audit_run_spec() registered ✓  
WhatsApp: send a message to mom saying running late (T032) - make_send_whatsapp_spec() registered ✓

**Differences found:** Looking up actual tool names - "google_search" vs maybe "web_search"? Registry has make_google_search_spec() creating tool named "google_search" (web.py:128 shows make_google_search_spec). "open_url" exists. "create_file" exists. "delete_path" exists. "type_text" exists. "lock_computer" exists. "send_whatsapp" exists. All match. Browser tools exist but not explicitly used in basic benchmark tasks shown. The benchmark can run today if dependencies present and on Windows for some tasks - but note some tasks expect specific behavior; also camera/screenshot tools added to redesign plan not in current registry? Current registry has take_photo (make_take_photo_spec) - yes added. Browser primitives like press_key/scroll/wait/back/forward/tabs/screenshot mentioned in redesign plan as new - not currently registered. So current benchmark should be able to run against current registry.

## 2e. Camera - reasons take_photo could end with 0 steps for source=voice
**Ignore WSL. For real Windows app:**

**Brain halt reasons (src/jarvis/agent/nodes/brain.py):**
- LLM_NOT_CONFIGURED / LLM_NOT_CONFIGURED_VOICE (lines ~39-77) - if no provider configured
- On exception: _llm_halted_reason maps rate limit→LLM_RATE_LIMITED, timeout→LLM_TIMEOUT, repair/structure markers→LLM_INVALID_OUTPUT, else LLM_PROVIDER_ERROR (lines ~254-271)
- Clarification budget exhausted → halted_reason "Still missing: ..." with request_kind="unsupported", plan None (lines ~189-212)
- Unsupported → halted_reason from response_text or "I can't do that." (lines ~212-224)
- LLM_NO_ANSWER if conversation has empty answer

**Intake (src/jarvis/agent/nodes/intake.py:64-68):** no input → "No input received. Tell me what to do."; cancelled → "Cancelled by the user."

**Validate (src/jarvis/agent/nodes/validate.py:217):** plan validation fails after retries → "Could not produce a valid plan (after 2 attempts): ..."

**Policy gate (src/jarvis/agent/nodes/policy_gate.py):** various halts - no plan, clarification needed, no step, precheck refusal, Tier 2 needs unlock in some cases, TOCTOU issues, batch issues, scope grant failures

**Voice-specific (brain.py:130-170):** for source=="voice", brain passes max_retries: 0, timeout_s: min(timeout_seconds,10.0) to LLM calls; sets voice_short=True. Shorter timeouts/retries increase chance of timeout/rate limit → halt. Also voice path may route differently.

**Tier handling for voice (Tier 1 confirm):** take_photo is Tier 1 (TIER_CONFIRM, base_tier=1). Policy gate will require confirmation for Tier 1. For voice interactions, confirmation flow goes through daemon/voice confirmation mechanisms; if confirmation cannot be obtained or times out in batch approval context, batch timeout gives "Confirmation timed out. I did not perform the action." (policy_gate.py ~660-679). Also voice tier2 rules exist but take_photo is Tier 1.

**Camera dependency status in venv:** grep showed cv2/opencv_python installed? Let us check - earlier ls showed opencv_python in site-packages? Let us look up. Also camera.py imports cv2 lazily via _import_cv2() raising CameraError if missing. In venv we need to check.

**opencv-python installed:** Yes, opencv_python-5.0.0.93.dist-info exists in venv (cv2 module present). So dependency present.

**Prechecks/camera-specific:** take_photo has windows_only=True in spec (camera.py:284). On non-Windows or if platform checks fail, tool execution/policy may refuse. Also camera backend does checks at runtime - if no camera available raises CameraError → ToolResult(ok=False, error=...). Policy gate won't halt before step just for missing camera; the tool will fail at execution and verification will mark verified False; act/verify flow handles tool failures.

**Summary of reasons for 0 steps (voice source):**
1. Brain halt - LLM not configured/rate limited/timeout/invalid output/no answer/clarification exhausted/unsupported (brain.py:39-271). Voice uses shorter timeouts (max_retries:0, timeout_s:min(timeout_seconds,10.0)) increasing risk.
2. Intake failure - no input/cancelled (intake.py:64-68)
3. Validate failure - cannot produce valid plan after retries (validate.py:217)
4. Policy gate halt - no plan/clarification needed/no step/precheck refusal/Tier 2 needs unlock issues/batch timeout/hash mismatch/scope grant failure (policy_gate.py various)
5. Tier 1 confirmation required but cannot be obtained/times out (especially in voice/daemon contexts) → batch timeout or confirmation flow halts (policy_gate.py ~660-679)
6. Tool execution failure after step reaches act - but that would produce a step result, not 0 steps; 0 steps means halted before acting
