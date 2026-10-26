# JARVIS Handoff

Last updated: 2026-10-26

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

Phase 2 design spec (browser primitives)

## Phase 1 results
- Step 0: Conftest guard allows subprocess in test_daemon_single_instance, test_tools_dictation, test_tools_open_in_app, test_tools_apps, and test_memory::test_embed_is_process_stable (5+ files in allowlist noted). Benchmarks YAML total 59, no expect.tool entries found (schema uses expected.type); missing from registry: [] (static). Git status clean at phase start of commits.
- Step 1: Blank input now reaches "No input received" branch without validation exception (intake fixed). Test tests/unit/test_intake_blank.py passes.
- Step 2: Added src/jarvis/agent/referents.py (ReferentStore, TTL 600s, one per kind, thread-safe, in-memory). Wired into act.py - browser_find/open_result/goto/open_site set referents; take_photo/create_file set FileRef; page referent not cleared on unrelated steps (only on nav to different origin/explicit/TTL). SessionContext API preserved. Tests tests/unit/test_referents.py and tests/unit/test_session_context.py pass.
- Step 3: Added referent summary block to brain prompt (<1200 chars) with rule line in system prompt; brain passes resolved_refs summary.
- Step 4: Added context_resolve node (no LLM), wired between intake and memory_retrieve in graph; extended fastpath for referent-aware "open the first/second/third result|link|one" (browser_open_result position) and "open the picture/photo" (open_path). Routing updated in nodes/__init__ and graph.
- Step 5: Added RequestKind and FailureCode enums to state.py; added conversation_id and resolved_refs keys to AgentState. Kept request_kind, replan_decision, Profile. Dead/live table (summary): request_kind written by intake/fastpath/brain/replan paths (live), read by graph routing/respond/telemetry (live). replan_decision written by replan node (live), read by respond/graph (live). Profile defined in state.py and used in benchmark/profiles codebase (separate from AgentState; not stored as AgentState key - largely live in its own domain).
- Decisions: guarded generic final click incl. opt-in voice approval; whole-profile folders; full-screen screenshot Tier 1; generic not site-specific; browsing grant yes.


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


## Phase 2 results

- Step 0: Verified (all passed) - see earlier summary.
- Step A: Removed scratch files (.tmp_pytest.bat, .tmp_run.bat, fix.py, checkbench*.py, checkwrap.py, testimp.py). None committed.
- Step B: Added DEFAULT_SITES in browser/sites.py (Google, YouTube, Amazon amazon.com, Gmail, Google Maps, Wikipedia, GitHub, Reddit, Stack Overflow, LinkedIn, Spotify web, Netflix, X, Flipkart) with aliases and search_url_template where applicable. Config sites override defaults by name. SiteDirectory API/MAX_SITES preserved. render_site_catalogue < 1500 chars. New test tests/unit/test_default_sites.py (passed).
- Step C: Added browser primitives to BrowserSession protocol and PlaywrightBrowser (press_key allowlist, scroll capped, wait capped/load_state, back/forward, list_tabs/switch_tab/close_extra_tab, screenshot to JARVIS pictures folder with prefix screenshot-, current-tab-only). Origin re-checked after actions; refuses if ended on disallowed domain.
- Step D: Added browser_search/press_key/scroll/wait/back/forward/tabs/screenshot tools with typed args, closed registry registration. Tiers: scroll/wait 0; back/forward/tabs/press_key/search/screenshot 1. browser_search resolves via SiteDirectory, builds URL in code from search_url_template (model only supplies site/query), no-template returns error telling planner to use browser_open_site + browser_type + browser_press_key. Records result set like browser_find, returns top 5 titles with positions (tainted). browser_screenshot returns data.artifact {path,type:"screenshot"}. PageRef recorded on navigations; ResultSetRef invalidated on navigation as before. New test tests/unit/test_browser_primitives.py (passed). catalogue_for_llm() JSON size before: ~23479; after: ~23479 (import-only check).
- Step E: Added one rule line to brain system prompt preferring browser_search over google_search/open_url for site searches.
- Step F: Updated benchmarks/tasks.yaml S001-S003 to mention browser_search/browser_open_result. YAML validated.
- Stray files removed: yes. Failing tests left: none from new tests run. 
- NEXT STEP: "Phase 3 design spec (orchestrator and browser specialist)"

## New tool table (Phase 2)
| Tool | base_tier | Notes |
|------|-----------|-------|
| browser_search | 1 | Resolves site, builds URL from template, records results, returns top 5 |
| browser_press_key | 1 | Allowlisted keys only, no modifiers |
| browser_scroll | 0 | Capped 1..10 |
| browser_wait | 0 | Capped 1..10s or load_state |
| browser_back | 1 | Invalidates result set, records PageRef |
| browser_forward | 1 | Invalidates result set, records PageRef |
| browser_tabs | 1 | list/switch/close_extra; bounds checked |
| browser_screenshot | 1 | Saves to pictures/JARVIS with screenshot- prefix, returns artifact |


## Phase 3 results

- Step 0: graph diff was context_resolve wiring -> committed (883fc31). Catalogue: 47 tools, JSON size ~23479. tests/unit/test_tools_registry.py not present; ran daemon_browser_wiring (unrelated failure). Fixture browser test command: pytest tests/integration/test_browser_fixture.py -m browser (not run).
- Step 1: Added browse to BrainDecision, SpecialistDecision (flat), AgentState keys (specialist_goal/calls/history/phase), AgentSettings use_specialists/specialist_max_calls/specialist_call_timeout_s. SpecialistDecision not added to serde allowlist (transient). Commit 7984e45/7cb3d07.
- Step 2: Added agent/nodes/specialist.py (decide with LLM planner role, timeout capped, max_retries 0; observe builds browser_read; guards: call cap, repeat guard, non-browser rejected). Page digest <=2500 wrapped as untrusted. Outputs one-step Plan to reuse validate/policy_gate/act/verify. Routing added when flag true; graph topology updated. Commit 95c4e72.
- Step 3: Brain routes browse to specialist when flag true (sets specialist_goal/phase/clears history); adds prompt rule only if flag true. Commit included in 95c4e72.
- Step 4: tests/unit/test_specialist.py (fakes) - validators, digest bounds/wrapping, call cap, repeat guard, non-browser rejection, flag-off routing unchanged. PASSED.
- How to enable: in config.toml [agent] use_specialists = true (and specialist_max_calls/specialist_call_timeout_s optional).
- Known limit: each Tier 1 browser step still asks for confirmation until browsing grant exists.
- NEXT STEP: "Phase 4 design spec (browsing grant, policy change)"


## Phase 4 results (in progress)

- Task 1 (root cause, not guessed): `specialist.decide` called `ctx.llm.structured(..., role="planner")`;
  the client protocol only accepts `model_role`, so the call raised `TypeError` before the fake recorded it.
  The specialist's `except` handed it to `_llm_halted_reason`, which reported the coding error as
  "I couldn't reach the AI service.". Fixed in ebd5607 (also unpacks the `(model, usage)` tuple, which
  would have raised `AttributeError` next).
- Task 2 (loop fixes, 62824d9): unique step ids `s<specialist_calls>` (plus `r<calls>` for the injected
  read step); `validate` preserves results/approved_hashes for a specialist iteration; `observe` reads
  `Step.tool` correctly (the old `A or B if C else None` expression always yielded `None` and looped
  forever); `observe` records each outcome in `specialist_history`; failed/unverified steps route to
  `specialist_observe`, not the generic replan node; `route_after_specialist_decide` lets `done`/call-cap
  beat the stale plan; `route_after_specialist_observe` routes on the phase, not the leftover plan.
  3 e2e tests pass (three steps once, failure then adaptation, call cap).
- Cleanup/style: a36aac8 (browse projection simplified, no bare except), 4338863 (ruff clean on
  specialist.py and brain.py).
- Decision: the task-scoped browsing grant lives in a **separate** `src/jarvis/policy/browse_grant.py`
  (`BrowseGrant` / `BrowseGrantRegistry`). `ScopeRegistry` is not edited: its grant requires a plan hash
  and its hard cap is 8 actions, neither of which fits a task-scoped browse grant (30 actions, no plan
  hash). The engine, tiers table and Tier 3 blocking stay untouched.
- Authorship fix: 34 commits in `backup-before-cleanup..HEAD` were rewritten to
  `Aarya Makthala <aaryamakthala@gmail.com>` (author and committer); trees are byte-identical. Safety
  branch `backup-pre-author-fix` kept. Rule: commit emails must never be masked (no `***` placeholder).
- Repository identity (global and repo-local) is `Aarya Makthala <aaryamakthala@gmail.com>`; no commit
  trailers or AI attribution are allowed.
- NEXT STEP: "Phase 4: specialist gap fixes, then browsing grant".


## Phase 4 grant — mid-wiring note

- `src/jarvis/policy/browse_grant.py` (BrowseGrant / BrowseGrantRegistry / pure coverage) and
  `tests/unit/test_browse_grant.py` (15 tests) are committed as 0f78c4e. Settings added to
  BrowserSettings: task_grant_enabled / task_grant_max_actions (30) / task_grant_ttl_s (600).
- ScopeRegistry, the policy engine and the tiers table are untouched.
- Decision: `browser_search` is covered by resolving the model-supplied site name to a host in
  code (SiteDirectory: defaults + config), in BOTH policy_gate and act. Unknown site = not covered.
  The tool itself is not changed.
- Wiring into policy_gate / act / daemon _end_task_runtime is in progress.

## Phase 4 results (complete)

- Task-scoped browsing grant in `src/jarvis/policy/browse_grant.py`, fully separate from ScopeRegistry
  (no plan hash, own 30-action / 600 s caps). Engine, tiers and ScopeRegistry untouched.
- policy_gate: a specialist Tier 1 browser step with no grant raises one prompt
  "Allow JARVIS to browse <host> for this task?"; approval issues the grant and covers that step.
  Later steps run coverage(); covered steps skip the interrupt and carry the grant id in the
  Decision reasons. "no"/timeout issues no grant and halts as before. Batch approval never mixes
  covered and normal steps (the covered path returns before the batch machinery).
- act: re-runs coverage() against the freshly re-observed target, charges the BROWSE registry (not
  scope_registry.reserve), keeps the tier unchanged, and records the target binding the gate issued.
- `browser_search`: the model supplies only the site name; the host is resolved in code via the
  SiteDirectory in BOTH gate and act. Unknown site = not covered.
- daemon `_end_task_runtime` revokes the browsing grant with the scope grant and bindings.
- Tests: tests/unit/test_browse_grant.py (15) and tests/unit/test_browse_grant_wiring.py (7) pass;
  tests/unit/test_invariants.py passes (37).
- Commits: 0f78c4e (module + rules), f369f2f (wiring), 99c2fe4 (target binding + wiring tests),
  a70b622 (pre-wiring note).
- Known limits: payment/purchase controls are Tier 3-blocked rather than re-confirmed;
  browser_search coverage needs a known site.  (Enter after a search was a limit here; it is covered
  from B3 on - see the browser hardening section below.)
- NEXT STEP: "Phase 5 design spec (compose, persona, guarded final click)".

## Backbone (docs/BACKBONE_PLAN.md)

Implementation-only pass over the prompt -> LLM -> plan -> run -> answer
backbone. **Every step below was UNTESTED at implementation time**: no tests were
written and pytest was not run then (per the plan's rules: `ruff check` +
`py_compile` + read-only scripts only); see "Test phase results" for the areas
verified later. `act.py`, `policy_gate.py` and `tests/unit/test_browse_grant_wiring.py`
were not touched by that pass; they were WIP then and are committed now in the
browser hardening pass (B2, `9e88856`).

| Step | Commit | Content | Status |
|---|---|---|---|
| A0 | `e28b2b8` | `docs/BACKBONE_PLAN.md` (the step list + rules) | UNTESTED |
| A1 | `a4ec8bc` | Node trace: `wrap()` appends the node *name* per `task_id` to a bounded in-memory trace in `agent/nodes/telemetry.py`; `record_finished` writes the compact path (`brain x2 > respond`) into the existing free-form `tools_used` column and frees the trace. | UNTESTED |
| A2 | `4cfc3d1` | `agent/routing.py`: `route_for_brain_result` with the route table; `route_after_brain` delegates; `unsupported` explicit (no tools, answer = brain text or "I can't do that with the tools I have."); `BrainDecision` gained the missing consistency validators (conversation/unsupported -> no actions). | UNTESTED |
| A4 | `06ed825` | `llm.total_deadline_seconds` (30 s) / `llm.voice_deadline_seconds` (12 s); the composite client sets one monotonic deadline per call, clips every attempt's timeout to the time left and raises the timeout-category error when it is gone. Brain voice path uses the voice budget. | UNTESTED |
| A3 | `f1fc83a` | `agent/tool_groups.py` + `agent.prompt_trim_enabled` (default **false**): the brain describes only the groups the request plausibly needs and a trimmed site catalogue; unassigned tool or no match = full catalogue (fail open); off = byte-identical prompt. Measured: 47 tools, full prompt 35,390 B (catalogue 30,923 B); "open notepad" trimmed 11,502 B / 13 tools. | UNTESTED |
| A5 | `a59b380` | `compose` node + `agent/compose_rules.py`: routers still return `"respond"`, the edge maps translate it to `compose`, `compose -> respond`; writes only `spoken_answer` (<=2 sentences) and `screen_answer` (the pre-existing answer); failures/refusals are never reworded; the address appears only with a stored `user_name`. Voice reads `spoken_answer` (`runner`, `daemon/server.py`), terminal reads `screen_answer` (`cli._print_outcome`). `compose.enabled` default true. | UNTESTED |
| A6 | this commit | `docs/STATE_CONTRACT.md`, `docs/GRAPH.md`, this section. | UNTESTED |

Deviations recorded during the pass: the `tools_used` column truncates at 300
chars (no migration); backoff sleeps inside the provider clients are not clipped
to the deadline; cancel-between-attempts was skipped (6 `ctx.llm` call sites);
single-provider installs bypass the composite client and therefore the overall
deadline; "sir" is only used when a `user_name` preference already exists.

### Memory pass (M0a-M7)

All implementation-only and **UNTESTED at the time** (ruff + py_compile + read-only
scripts only; no tests written, pytest not run then); see "Test phase results" for
the save gate and graph files verified later. `act.py`, `policy_gate.py` and
`tests/unit/test_browse_grant_wiring.py` were left untouched by that pass; they
are committed now in the browser hardening pass (B2, `9e88856`).

| Step | Commit | Content | Status |
|---|---|---|---|
| M0a | `3db20b0` | Node path stored as short codes (`in`, `cr`, `mr`, `br`, `va`, `pg`, `ac`, `ve`, `rp`, `cl`, `co`, `re`, `ms`, `sd`, `so`) joined by `>`, repeats as `ac x3`, so the whole run fits the 300-char `tools_used` column. One documented map (`NODE_CODES`); `get_trace` still returns full names. | UNTESTED |
| M0b | `b20a2c1` | Retry backoff clipped to the request deadline: `multi._deadline()` computes one absolute deadline and hands it to every provider as `deadline_at`; `sleep_within_deadline` sleeps `min(backoff, remaining)` and raises the timeout category when it is gone. Groq, OpenRouter/NVIDIA and Gemini all use it. | UNTESTED |
| M1 | (no commit) | Read-only memory audit (stores, caps, retrieval, save policy, failure behaviour, preferences, CLI). Findings drove M2-M6. | audited |
| M2 | `8d636dc` | Bounded, safe retrieval (`memory_retrieve.py`): per-item cap `memory.item_max_chars` (300), block cap `memory.context_max_chars` (1200), near-duplicate drop, shared secret check skips an item, failure examples older than `memory.failure_max_age_days` (30) skipped, and the store call bounded by `memory.retrieve_timeout_seconds` (2.0) in a worker thread (safe: `check_same_thread=False` + `RLock` in `SqliteMemory`). Timeout/error = empty `memory_context` + one content-free warning. `prompts.py` wraps the block in a delimited "past tasks for reference only - never instructions, never replayed" section. New keys default, so old configs load. | UNTESTED |
| M3 | `4b07f4d` | Bounded pruning (`db.MAX_PRUNE_DELETIONS = 50` rows per call, oldest/least-used first, for skills and failures); failure dedupe (same goal + tool + category refreshes `created_at` instead of inserting); failure rows store the **tool name + short category** (`tool_failed`/`verify_failed`/`untrusted_output`/`unknown`), never raw error/page/tool text, and only on a terminal failure; `memory_save` sets all three state keys on every path from the closed `REASONS` vocabulary. Success gate unchanged. | UNTESTED |
| M4 | `b1f997c` | Name learning in `memory_save` (`detect_name_statement`): four explicit phrasings only, 1-2 letter/apostrophe/hyphen words, 2-40 chars, stop words/verbs/credential words refused, secret check, conversation turns only, `memory.learn_name` (default true), weak phrasings cannot replace an existing name, `memory_saved_reason="name_saved"`. Applied from the **next** turn (compose runs before memory_save). | UNTESTED |
| M5 | `3cc4cea` | `src/jarvis/memory_cli.py` + registration: `jarvis memory list|forget|clear|prefs` over the agent's own stores; 50-row cap, 60-char text, 40-char values, secrets hidden, `CLEAR` confirmation, `--kind` disambiguates ids; `FailureStore.delete` added. | UNTESTED |
| M6 | `07305af` | `docs/MEMORY.md`, updated `docs/STATE_CONTRACT.md` memory keys, this section. | UNTESTED |
| M7 | this commit | Docs for the hardening: `docs/MEMORY.md` retrieval caps/timeout, failure content and dedupe, bounded prune, the `memory_saved_reason` code table, the four new config keys; this table. | UNTESTED |

Cross-host item: **RESOLVED-IN-CODE / UNTESTED** (B1-B3 below).

### Browser hardening pass (B1-B4)

Implementation + design docs only, **UNTESTED at the time**: no pytest run then (the
test phase was later); checks were `ruff check` on the changed files, `py_compile`
with `.venv\Scripts\python.exe`, and read-only probe scripts with a `FakeLLM`
(no network). See "Test phase results" for the browsing-grant wiring file verified
later. `PolicyEngine`, `tiers`, Tier 3 blocking, the target-binding hash
algorithm, `llm/models.py`, `free_only` and `strict_zero_cost` are untouched.

| Step | Commit | Content | Status |
|---|---|---|---|
| B1 | (no commit) | Read-only trace of the cross-host sequence. Conclusion **(Y)**: host B already gets its OWN independent authorization - a task-scope confirmation for B (`scope_confirm=True`, summary names B, `browse_grant` absent) minted by `_scope_confirm` -> `_issue_scope_grant` (`policy_gate.py:463,636`), with the target binding recorded under that grant's `confirmation_id` (`:534`) and consumed by act with the same id (`act.py:276`). Traced ids: A's browse grant `b550c3f1...` put/consumed at both ends; B's scope cid `67793dd1...` put/consumed at both ends; s1 ran (`ok=True verified=True`), A's grant untouched, browse `consumed=1`. No cross-host code fix was needed. The real defect was the *other* direction: a browse-covered step while a ScopeGrant is active. | traced |
| B2 | `9e88856` | One authorization id per step, at both ends: a browse-covered step records its binding under the grant's own id (`policy_gate.py:287-293,326-332`) and act consumes it with the id its own fresh coverage check derives (`act.py:276-278`); a step the gate did not cover binds under the active ScopeGrant's confirmation id (now the only path into `_approve_single_step`, dead `grant_id` branch removed). Tests in `test_browse_grant_wiring.py`: host B's prompt names B and is B's own authorization, approving it lets the step RUN (result + navigation asserted), refusing it runs nothing and leaves A's grant intact, `_VanishingGrantRegistry` proves act refuses when coverage is removed between gate and act, plus the simultaneous-ScopeGrant regression. | UNTESTED |
| B3 | `5687447` | Enter after a search is covered: `_record_history` keeps a compact marker of a successful type into a `searchbox`/`combobox` (role + host only, never the element's name/label/ref or the typed text) (`specialist.py:187-262`); `browse_grant_check` looks back over passive `browser_read` refresh steps only, passes the pressed key and the previous role, and takes the host from the marker for `browser_press_key` (`util.py:171-198,200-246`). Fail closed: a textbox, a password/submit role, an intervening click/navigation, or a host mismatch all leave Enter uncovered. act re-runs the same rule independently. | UNTESTED |
| B4 | `9ed011f` | `docs/FINAL_CLICK_DESIGN.md`: `browser_final_click` design - payment classified FIRST (Tier 3, unreachable), irreversible non-payment verbs only, never browse-grant/batch covered, exact identity + fresh evidence + host + typed-confirmation hash, single-use binding, re-verify immediately before dispatch, `reserved -> dispatched -> confirmed \| indeterminate` journal written before dispatch, never retried when indeterminate, `final_click_max_total` 0 and voice approval off by default, owner decisions listed. Design only, no code. | design |

Known limit after B3: the marker deliberately survives only passive page reads, so any
other action between the type and Enter (a click, a navigation, another key press, a
scroll) means Enter asks again. That is the intended fail-closed side of the rule, not
a bug: only a read is known not to move focus.

`docs/GRAPH.md` is unchanged: B2-B3 changed no node, edge or route.

### Batch C (C1-C4)

Implementation only, **UNTESTED at the time** except the C1 smoke run (raw result below);
see "Test phase results" for the files verified later.
No pytest, no new test files; checks were `ruff check` on the changed files,
`py_compile` with `.venv\Scripts\python.exe`, and read-only probe scripts with
fakes (no LLM, no network, no browser). `PolicyEngine`, `tiers`, Tier 3, the
target-binding hash, `llm/models.py`, `free_only`, `strict_zero_cost` and the
provider order are untouched.

| Step | Commit | Content | Status |
|---|---|---|---|
| C1 | `9f52a6b` | `scripts/smoke_flow.py`: runs the real graph with fakes only (scripted `FakeLLM`, the `tests/support.py` fake tool, a temp SQLite checkpointer) for S1 conversation, S2 one safe action, S3 every LLM provider failing; prints scenario, node path from the A1 trace, final answer, `halted_reason`, PASS/FAIL, and exits 1 on any FAIL. The trace is kept readable by replacing `telemetry.discard_trace` with a no-op *inside the script* (the graph is untouched). | smoke 3/3 PASS |
| C2 | `f183bbf` | `act.py`: both referent-bookkeeping `except Exception: pass` blocks replaced with the narrow set that block can actually raise (`ImportError` for the optional import, `TypeError`/`ValueError` - and `ValidationError` *is* a `ValueError` - for a page-supplied `position` or a failing ref model) plus a content-free `logger.debug`. No authorization, binding or tier logic touched. `ruff check act.py` is now clean (it had 4 carried-over S110/BLE001 findings). | UNTESTED |
| C3 | `f0ee622` | (a) `llm/abort.py`: a `ContextVar` holding an optional zero-argument `is_cancelled`; `wrap()` installs the task token's `cancelled` flag for the node and resets it in a `finally`; the composite client checks it before every attempt (all three operations) and the provider retry loop before every attempt and every backoff sleep, raising the existing `Cancelled`. No hook set = behaviour unchanged. (b) `build_llm_client` now routes a single provider through `MultiProviderClient` too, so one-provider installs get the same `llm.total_deadline_seconds` / `llm.voice_deadline_seconds` deadline from 06ed825. Probe: single install -> `MultiProviderClient`, budget 30.0 normal / 12.0 for a 5 s call, provider info preserved; hook raises before any provider call. | UNTESTED |
| C4 | `134b076` | (a) Read-only audit of the task log: the 3 failed `whatsapp_send` rows ("Aarya", "pappa", "der Arzt") all failed with `no contact matches ... Known contacts: (no contacts yet)` - `contacts.json` does not exist on this machine, so the lookup had nothing to match, not a WhatsApp/browser fault. (b) `messaging/resolver.py`: pure `resolve_contact(query, contacts)` - NFKC + casefold, emoji/punctuation dropped, spaces collapsed; exact, then alias, then near by token subset / substring (>= 3 chars) / `difflib` ratio >= 0.85; two or more candidates are `ambiguous` (max 3 names); `near` and `ambiguous` are never a selection. (c) Wired into `resolve_single`, the one lookup the precheck, `MessageFlow` and the WhatsApp tools share: `near` now refuses with "did you mean 'X'?" and `ambiguous` with "multiple contacts match ...", so nothing is ever sent on a guess. Probe: the real empty-store text is unchanged, "pappa" -> near 'Papa', "Ali" -> ambiguous, a 1-char query never fuzzy-matches, and the graph halts before any confirmation. | UNTESTED |

Alias note: the contacts file format has no `aliases` field (and `ContactStore.save`
would drop one), so the alias branch is implemented but dormant - supporting it needs
a file-format change, which this pass did not make.

C1 raw output (`py_compile` clean, `ruff` clean, 90 s limit, exit 0):

```
[S1 conversation answer]
   node path: in>cr>mr>br>co>re>ms
   final answer: 'Here is your answer.'
   halted_reason: None
   results: []
   PASS
[S2 one safe action]
   node path: in>cr>mr>br>va>pg>ac>ve>co>re>ms
   final answer: 'fake_echo ran hi'
   halted_reason: None
   results: [('s1', True)]
   PASS
[S3 all LLM providers fail]
   node path: in>cr>mr>br>co>re>ms
   final answer: "I couldn't safely understand that request."
   halted_reason: "I couldn't safely understand that request."
   results: []
   PASS
smoke: 3/3 passed
```

### Batch D (D1-D3)

Implementation only, **UNTESTED at the time** except the D2 smoke run (raw result below);
see "Test phase results" for `test_domain_policy.py` verified later.
No pytest, no new test files; checks were `ruff check` on the changed files,
`py_compile` with `.venv\Scripts\python.exe`, the extended smoke script, and
read-only probes (no LLM, no network, no browser). `PolicyEngine`, the tiers
table, Tier 3 blocking, the target-binding hash, `llm/models.py`, `free_only`,
`strict_zero_cost` and the provider order are untouched.

| Step | Commit | Content | Status |
|---|---|---|---|
| D1 | `6530338` | Fresh confirmation for the clicks a scope grant used to cover silently: `rules.FRESH_CONFIRM_TOKENS` (now public) gained `login`, `post`, `publish`, `subscribe`, `donate`, `download`, `upload`, `install`, `book`, `reserve`, `checkout`, `place`, `order`, `transfer`, `withdraw`, and the new `FRESH_CONFIRM_PHRASES` (`log in`, `sign in`, `sign up`) covers what token matching cannot see (`_has_phrase`, whitespace-collapsed whole-phrase match). `browse_grant.RISK_WORDS` is now `_MONEY_RISK_WORDS` + the shared set + the phrases, so a word needing a fresh confirmation is never browse-covered either; the probe proves the browse grant refuses **every** fresh-confirm word (`missing: none`). Payment words are still classified first (`Pay now` -> `payment`, `Buy now` -> `payment`) and Tier 3 is untouched; everything added raises `min_tier` to Tier 1 only. | UNTESTED |
| D2 | `6d4377f` | `scripts/smoke_flow.py` grew S4-S9 on the real graph with fakes only: clarification -> answer -> action, unknown-tool plan repaired, confirmation approved, confirmation refused, failing tool retried then replanned, and cancel after the prompt (the resumed run raises `Cancelled` before the approved step and nothing runs). Each prints a note, node path, answer, halted reason, results and PASS/FAIL; exit 1 on any FAIL. **9/9 PASS** (raw below). | smoke 9/9 PASS |
| D3 | this commit | This section and the remaining-items list. | docs |

Before/after classification (D1 probe, real `rules.risky_control` / `browse_grant.risky_label`):

```
label        risky_control  browse risky_label      | after edit
Log in       -              log in                   confirm / log in
Post         -              post                     confirm / post
Publish      -              publish                  confirm / publish
Subscribe    -              subscribe                confirm / subscribe
Search       -              -                        -              / -            (unchanged)
Next         -              -                        -              / -            (unchanged)
Play         -              -                        -              / -            (unchanged)
Buy now      payment        buy                      payment        / buy          (unchanged, Tier 3)
Send         confirm        send                     confirm        / send         (unchanged)
Pay          payment        pay                      payment        / pay          (unchanged, Tier 3)
```

Known consequence for the test phase: two existing expectations in
`tests/unit/test_domain_policy.py` encode the **old** (looser) classification -
`("Checkout", "button", None)` in `test_risky_control_classification` and
`("Login", "button")` in `test_risky_control_false_positives`. Both now return
`CONTROL_FRESH_CONFIRM`, so those two cases must be updated when the test phase
runs (the tests were not edited in this pass, per the batch rules).

D2 raw output (`ruff` clean, `py_compile` clean, 90 s limit, exit 0; the one
stderr line `brain call failed: ...` is S3 deliberately exhausting its script):

```
[S1 conversation answer]
   node path: in>cr>mr>br>co>re>ms
   final answer: 'Here is your answer.'
   halted_reason: None
   results: []
   PASS
[S2 one safe action]
   node path: in>cr>mr>br>va>pg>ac>ve>co>re>ms
   final answer: 'fake_echo ran hi'
   halted_reason: None
   results: [('s1', True)]
   PASS
[S3 all LLM providers fail]
   node path: in>cr>mr>br>co>re>ms
   final answer: "I couldn't safely understand that request."
   halted_reason: "I couldn't safely understand that request."
   results: []
   PASS
[S4 clarify, then answer, then act]
   note: asked first: True
   node path: in>cr>mr>br>cl x2>br>va>pg>ac>ve>co>re>ms
   final answer: 'fake_echo ran hi'
   halted_reason: None
   results: [('s1', True)]
   PASS
[S5 invalid plan, then repair]
   note: validate attempts: 0
   node path: in>cr>mr>br>va>br>va>pg>ac>ve>co>re>ms
   final answer: 'fake_echo ran hi'
   halted_reason: None
   results: [('s1', True)]
   PASS
[S6 confirmation yes]
   note: asked first: True
   node path: in>cr>mr>br>va>pg x2>ac>ve>co>re>ms
   final answer: 'fake_echo ran hi'
   halted_reason: None
   results: [('s1', True)]
   PASS
[S7 confirmation no]
   node path: in>cr>mr>br>va>pg x2>co>re>ms
   final answer: "Refused by you (confirmation answered with 'no' or a mismatched action)."
   halted_reason: "Refused by you (confirmation answered with 'no' or a mismatched action)."
   results: []
   PASS
[S8 tool fails, then replans]
   note: tools called: ['fake_echo', 'fake_echo', 'fake_echo', 'fake_fix']
   node path: in>cr>mr>br>va>pg>ac>ve>ac>ve>ac>ve>rp>va>pg>ac>ve>co>re>ms
   final answer: 'fake_fix ran fixed'
   halted_reason: None
   results: [('s2', True)]
   PASS
[S9 cancelled before acting]
   note: the run raised Cancelled at the next node
   outcome: none (the run ended by raising)
   results: []
   PASS
smoke: 9/9 passed
```

Cancel hook through the composite client (D2 probe, read-only):

```
cancelled hook: Cancelled raised before any provider call
provider calls while cancelled: []
no hook: provider was reached, error: LLMError
provider calls without a hook: ['structured']
```

Worker threads: the only thread pool in `src/jarvis/llm` + `src/jarvis/agent`
is `agent/nodes/memory_retrieve.py` (the bounded memory read), and it makes **no**
LLM call, so the C3 `ContextVar` is installed and read on the same thread as
every provider call. The daemon runs a task in its own worker thread
(`daemon/server.py:1017,1201,1245`), but the graph and its `ctx.llm` calls are
built and run inside that one thread, so the hook still applies. A future change
that moved an LLM call into an executor would silently lose the hook (a
`ContextVar` does not propagate into a new thread or a `to_thread` call).

### Still open

- `browser_final_click` code: designed (`docs/FINAL_CLICK_DESIGN.md`, B4) but blocked
  on the eight owner decisions listed there - nothing implemented.
- Screenshot and `show_file` tools: written in G1 (`b7687ca`).
- Wider system tools: `file_search`, `process_list`, `volume_control` written in G4 (`8d3cd6e`).
- `agent.prompt_trim_enabled` (tool-group prompt trimming, A3) is still **off** by default.
- The test phase has started: the files in "Test phase results" below were run; areas
  without a passing file there remain UNTESTED.
- `contacts.json` still does not exist on this machine: the WhatsApp paths refuse with
  "no contact matches ... Known contacts: (no contacts yet)" until the user adds contacts
  (`jarvis contacts add`).

Next: continue the test phase for the areas still marked UNTESTED.

## Test phase results (E3/F1/F2)

One named file per run, 90 s limit, `-q -x -p no:cacheprovider`; no full suite, integration,
windows_only, voice, camera, browser-fixture or live tests. Failures were classified before
editing: every edit below was a stale test expectation or a stale test fixture, not a source
change.

| File | Result | Action |
|---|---|---|
| `tests/unit/test_domain_policy.py` | 61 passed | none |
| `tests/unit/test_browse_grant.py` | 15 passed | none |
| `tests/unit/test_browse_grant_wiring.py` | 10 passed | none |
| `tests/unit/test_invariants.py` | 37 passed | none |
| `tests/unit/test_specialist_loop_e2e.py` | 6 passed | none |
| `tests/unit/test_agent_architecture.py` | 20 passed | none |
| `tests/unit/test_brain.py` | not run - file not found | reported, not substituted (`test_brain_prompt.py` is the only brain test file) |
| `tests/unit/test_memory_save_node.py` | 36 passed | stale reason/gate expectations updated (`4b07f4d`) |
| `tests/unit/test_brain_prompt.py` | 14 passed | none |
| `tests/unit/test_memory_graph.py` | 8 passed | stale `memory_saved_reason` expectation updated (`4b07f4d`) |
| `tests/unit/test_agent.py` | 14 passed | none |
| `tests/unit/test_cli.py` | 41 passed | none |
| `tests/unit/test_skills_cli.py` | 17 passed | none |
| `tests/unit/test_task_log.py` | 30 passed | none |
| `tests/unit/test_answer.py` | 36 passed | none |
| `tests/unit/test_completeness.py` | 7 passed | none |
| `tests/unit/test_freeonly.py` | 35 passed | none |
| `tests/unit/test_llm_rotation.py` | 8 passed | stale default-timeout expectation updated (`06ed825`) |
| `tests/unit/test_gemini.py` | 11 passed | stale sleep fixture updated to `sleep_within_deadline` (`b20a2c1`) |
| `tests/unit/test_daemon_server.py` | 30 passed | none |
| `tests/unit/test_daemon_browser_wiring.py` | 16 passed | none |
| `tests/unit/test_approval_words.py` | 3 passed | none |

### Area status after this pass

VERIFIED means at least one named file above passed and directly covers the area. Everything
not listed here remains **UNTESTED**.

| Area | Status | Evidence |
|---|---|---|
| Policy rules / fresh confirmation (D1) | VERIFIED | `test_domain_policy.py` |
| Browsing grant and wiring (Phase 4, B1-B3) | VERIFIED | `test_browse_grant.py`, `test_browse_grant_wiring.py` |
| Safety invariants | VERIFIED | `test_invariants.py` |
| Specialist loop | VERIFIED | `test_specialist_loop_e2e.py` |
| Agent architecture / backbone routing | VERIFIED | `test_agent_architecture.py`, `test_agent.py` |
| Brain prompt | VERIFIED | `test_brain_prompt.py` (the named `test_brain.py` does not exist) |
| Memory save gate and graph save path (M3) | VERIFIED | `test_memory_save_node.py`, `test_memory_graph.py` |
| CLI, skills CLI, task log, answer, completeness, free-only | VERIFIED | `test_cli.py`, `test_skills_cli.py`, `test_task_log.py`, `test_answer.py`, `test_completeness.py`, `test_freeonly.py` |
| LLM rotation, overall deadline, Gemini client | VERIFIED | `test_llm_rotation.py`, `test_gemini.py` |
| Daemon server, daemon browser wiring, approval words | VERIFIED | `test_daemon_server.py`, `test_daemon_browser_wiring.py`, `test_approval_words.py` |
| Memory retrieval/CLI, messaging resolver, live browser, voice, camera, OCR, benchmark, packaging | UNTESTED | not run in this pass |

## Batch G (G1-G6)

Scope: screenshot/`show_file`, OCR confidence, optional whole-profile roots, three wider system
tools, a prompt-trim verifier and this handoff. G1-G3 were committed before this session; G4 was
uncommitted WIP and was reviewed/fixed here; G5-G6 were created here. PolicyEngine, tiers, Tier 3,
the target-binding hash, `llm/models.py`, `free_only` and `strict_zero_cost` were not changed.

| Step | Commit | Content | Status |
|---|---|---|---|
| G1 | `b7687ca` | `feat(tools): take_screenshot and show_file` | committed |
| G2 | `f6eff96` | `feat(perception): ocr confidence passthrough` | committed |
| G3 | `8350148` | `feat(files): optional whole-profile roots with protected folders` | committed |
| G4 | `8d3cd6e` | `feat(tools): file_search, process_list, volume_control` | committed this session |
| G5 | `c517bf8` | `chore: prompt trim verification script` (`scripts/verify_trim.py`) | committed this session |
| G6 | this commit | `docs: batch G handoff` | docs |

### G4 review (R1) before commit

- Registry: **52 tools**, no duplicate names. Imports and registrations for `file_search`,
  `process_list`, `volume_control`, `take_screenshot` and `show_file` appear exactly once.
- `agent/tool_groups.py`: every registered tool is in exactly one group; no unassigned or
  duplicated tool. New assignments: `file_search` -> `files`; `process_list` and
  `volume_control` -> `system`.
- `system.py`: the broad `except Exception` around the pycaw/comtypes endpoint lookup was replaced
  with `COMError` (when importable), `OSError`, `ImportError` and `AttributeError`. The run function
  still converts unexpected backend failures into `ToolResult(ok=False, ...)`; nothing is silent.
- `policy/paths.py`: the deny list is now checked against the path **relative to home**, so a home
  folder whose own ancestors contain a deny name (for example a temp profile under `AppData`) is
  not over-denied, while paths outside home fall back to the full parts. Read-only probe with a
  throwaway home: `.ssh`, `.aws`, `.gnupg`, `AppData`, `.git`, `.venv`, `Documents/Wallets` and a
  `..` escape with a deny name are all DENIED; a plain `..` escape is DENIED by the root check; a
  normal `Documents` file is ALLOWED; with `profile_wide=false` the raw result and `is_protected`
  are byte-for-byte the same as `HEAD`.
- `ruff check` and `py_compile` were clean on all six changed files.

### Tests run this session (one named file per run, 90 s, `-q -x -p no:cacheprovider`)

| File | Result |
|---|---|
| `tests/unit/test_system_tools_new.py` | 14 passed |
| `tests/unit/test_profile_roots.py` | 22 passed, 1 skipped (symlink unavailable) |
| `tests/unit/test_screenshot_tools.py` | 13 passed |
| `tests/unit/test_ocr_confidence.py` | 7 passed |
| `tests/unit/test_invariants.py` | 37 passed |
| `tests/unit/test_agent.py` | 14 passed |

### G5 output (`python scripts/verify_trim.py`, exit 0)

```
registry tools: 52
full catalogue: 25003 bytes

PASS | 'open notepad'
     needs: open_app
     selected (13): audit_run, defender_quick_scan, defender_status, get_time, list_windows, lock_computer, lock_jarvis, open_app, open_in_app, open_path, system_info, type_text, web_answer
     bytes: 5604 (full 25003)  fallback=False
PASS | 'search youtube for lofi and open the first video'
     needs: browser_search, browser_open_result
     selected (30): audit_run, browser_back, browser_click, browser_find, browser_forward, browser_goto, browser_open_result, browser_open_site, browser_press_key, browser_read, browser_screenshot, browser_scroll, browser_search, browser_tabs, browser_type, browser_wait, defender_quick_scan, defender_status, get_time, google_search, list_windows, lock_computer, lock_jarvis, open_app, open_in_app, open_path, open_url, system_info, type_text, web_answer
     bytes: 13239 (full 25003)  fallback=False
PASS | 'send a whatsapp message to Papa'
     needs: whatsapp_send
     selected (16): audit_run, defender_quick_scan, defender_status, get_time, list_windows, lock_computer, lock_jarvis, open_app, open_in_app, open_path, system_info, type_text, web_answer, whatsapp_open_chat, whatsapp_send, whatsapp_send_file
     bytes: 7670 (full 25003)  fallback=False
PASS | 'take a screenshot and show it'
     needs: take_screenshot, show_file
     selected (12): audit_run, defender_quick_scan, defender_status, describe_screen, get_time, read_screen, show_file, system_info, take_screenshot, ui_click, ui_list_controls, web_answer
     bytes: 5373 (full 25003)  fallback=False
PASS | 'what is 2+2'
     needs: (none)
     selected (52): full catalogue
     bytes: 25003 (full 25003)  fallback=True
PASS | 'find my resume file'
     needs: file_search
     selected (13): append_file, audit_run, create_file, defender_quick_scan, defender_status, delete_path, file_search, get_time, list_dir, read_file, system_info, undo_last_delete, web_answer
     bytes: 6642 (full 25003)  fallback=False
PASS | 'set volume to 30'
     needs: volume_control
     selected (8): audit_run, defender_quick_scan, defender_status, get_time, process_list, system_info, volume_control, web_answer
     bytes: 3177 (full 25003)  fallback=False
PASS | 'read the file notes.txt'
     needs: read_file
     selected (13): append_file, audit_run, create_file, defender_quick_scan, defender_status, delete_path, file_search, get_time, list_dir, read_file, system_info, undo_last_delete, web_answer
     bytes: 6642 (full 25003)  fallback=False
PASS | "what's on my screen"
     needs: read_screen
     selected (10): audit_run, defender_quick_scan, defender_status, describe_screen, get_time, read_screen, system_info, ui_click, ui_list_controls, web_answer
     bytes: 4316 (full 25003)  fallback=False
PASS | 'take a picture of me'
     needs: take_photo
     selected (10): audit_run, defender_quick_scan, defender_status, get_time, save_dictation, start_dictation, stop_dictation, system_info, take_photo, web_answer
     bytes: 3686 (full 25003)  fallback=False

full-catalogue fallback rate: 1/10 = 10%
result: 10/10 PASS
```

`agent.prompt_trim_enabled` was **not** changed (still false by default). No trigger keyword needed
fixing; the script was not edited after its first run.

### Remaining items after batch G

- `browser_final_click`: still design-only (`docs/FINAL_CLICK_DESIGN.md`); needs the owner to approve
  the eight decisions there before any code.
- Enable `agent.prompt_trim_enabled` only after the owner reviews the G5 output (saves ~5-20 KB per
  request; fallback stays the full catalogue).
- Optional libraries: `psutil` and `Pillow` are already installed; `pycaw`, `mss` and
  `rapidocr-onnxruntime` are **not** installed. `pycaw` (for `volume_control`) is not yet declared in
  `pyproject.toml`/`scripts/verify_env.py`; add it to the optional extra and verifier before shipping.
- `contacts.json` still does not exist; WhatsApp paths refuse until the user adds contacts.
- Browser fixture test is written but must be run by the user on the PC
  (`pytest tests/integration/test_browser_fixture.py -m browser`).
- One real specialist task (live browser) has not been run.
- No vision role is configured; the vision model decision is still open.
- Orchestrator (route/dispatch/compose split) and voice-latency tuning remain.

Next: owner review of G5 and the final-click decisions before enabling anything.

## Batch H (H0-H6)

| Step | Commit | Message | Files |
|---|---|---|---|
| H0 | none (read-only) | browser-fixture investigation | none |
| H1 | `1de4140` | `build: declare optional libs in pyproject and verify_env` | `pyproject.toml`, `scripts/verify_env.py` |
| H2 | none (no file changed) | runtime backend probe | scratch probe only |
| H3 | `f12496b` | `feat(agent): specialist smoke script with fakes` | `scripts/smoke_flow.py` |
| H4 | `7bc20a1` | `feat(agent): trim-on smoke run` | `scripts/smoke_flow.py` |
| H5 | `54cd843` | `feat(agent): jarvis doctor read-only health check` | `src/jarvis/doctor.py`, `src/jarvis/cli.py`, `tests/unit/test_doctor_backends.py` |
| H6 | this commit | `docs: batch H handoff` | `JARVIS_HANDOFF.md` |

### H0 - why the browser fixture always skips

- `tests/integration/test_browser_fixture.py:6` marks the module `browser`.
- Lines 12-15 say "browser deps missing": they import `BrowserManager` from
  `jarvis.browser.manager`, a module that **does not exist** and a symbol that appears nowhere in
  `src/`. The real browser package has `context.py`, `find.py`, `lazy.py`, `results.py`, `sites.py`,
  `tools.py`, `whatsapp_page.py` and `whatsapp_web.py`.
- Lines 17, 21 and 25 are unconditional `pytest.skip("browser not executed in low-resource mode")`
  calls. There is no environment variable, config flag or conftest condition behind them; the
  `browser` marker is the only selector.
- `tests/conftest.py:71-76,110-112` blocks subprocess/os/webbrowser/pyautogui/pywinauto/cv2/sounddevice
  in autouse for every test except `live` and `windows_only`, so a real browser launch would also
  need one of those markers.
- `fixture_server` is the local server in `tests/fixtures/serve.py:10`, re-exported by
  `tests/integration/conftest.py`.

H0b (owner, on the Windows PC) once the fixture bodies exist and are marked `live` (or the guard is
deliberately extended):

```powershell
pip install -e ".[browser,vision]"
.venv\Scripts\python.exe -m playwright install chromium
.venv\Scripts\python.exe -m pytest tests/integration/test_browser_fixture.py -m browser
```

### H1 - optional libraries are declared and verified

- `pyproject.toml` gains `tools-extra = ["pycaw; sys_platform == 'win32'", "mss",
  "rapidocr-onnxruntime"]` (the existing `vision` extra keeps its own `mss`/Pillow/RapidOCR entries).
- `scripts/verify_env.py` gains `("pycaw", "pycaw", 7, True)` and `"pycaw"` in `OPTIONAL`, so a
  missing package is a WARN, not a FAIL, and Windows-only probing is explicit.
- `scripts/verify_env.py --all` exit 0: **57 PASS, 1 WARN, 0 FAIL**. The warning is the Recycle Bin
  OLE probe (`WinError -2144927705` on this machine), not a dependency. `pycaw` resolved as
  `20260927`; `ruff` and `py_compile` were clean.

### H2 - runtime backends behave cleanly when present

Probe (`C:\Users\aarya\AppData\Local\Temp\opencode_jarvis\h2_probe.py`, temp roots under
`C:\Users\Public`) found: `mss True`, `PIL True`, `rapidocr_onnxruntime True`, `rapidocr False`,
`pycaw True`, `comtypes True`; OCR `backend_name() = rapidocr_onnxruntime`.

With the backends installed, the tools still fail cleanly rather than raising:

| Probe | Result |
|---|---|
| `take_screenshot` with the capture backend unavailable | `ok=False`, `"screenshot backend not installed"` |
| `read_screen` with OCR unavailable | `ok=True`, note `"Local OCR is not installed."` (UIA text is kept) |
| `volume_control` with `pycaw` unavailable | `ok=False`, `"volume control needs the optional 'pycaw' package"` |

Tests: `test_screenshot_tools.py` 13 passed, `test_ocr_confidence.py` 7 passed,
`test_system_tools_new.py` 14 passed. No source file needed changing, so H2 has no commit.

### H3 - S10/S11 in the smoke script

`scripts/smoke_flow.py` now imports the existing `_FakeSession` and adds two specialist scenarios on
top of S1-S9, still with fakes only:

- **S10** - a browse task with `use_specialists=True`: the first special step raises the host prompt,
  approval is given, the next two steps on the same host run without a second prompt, and the run
  ends with the specialist's answer. Asserts one prompt only, `final_answer == "Browsed."`, and two
  consumed grant uses.
- **S11** - the third step targets a second host: the next prompt is a scope confirmation naming
  `other.example` (no `browse_grant`), refusing it runs nothing on that host, and only the first
  host's one use is consumed.

`python scripts/smoke_flow.py` printed **11/11 passed** (exit 0). S10 path:
`in>cr>mr>br>decide>va>pg x2>ac>ve>observe>...>decide>co>re>ms`. S11 ends
`...>decide>va>pg x2>co>re>ms` with the refusal message.

### H4 - `--trim` smoke run

`--trim` is parsed by the script and sets **only this script's** `AgentSettings.prompt_trim_enabled`;
the application default in `config.py` is untouched. A `FakeLLM` subclass records the UTF-8 byte
size of every `system`/`user` prompt without changing it, and each scenario prints per-call sizes.

`python scripts/smoke_flow.py --trim` printed **11/11 passed** (exit 0), 19 LLM calls, **163,012
bytes** total:

| Scenario | Calls | Prompt bytes |
|---|---:|---:|
| S1 conversation answer | 1 | 4,359 |
| S2 one safe action | 1 | 4,359 |
| S3 all providers fail | 1 | 4,359 |
| S4 clarify, answer, act | 2 | 9,318 |
| S5 invalid plan, repair | 2 | 8,861 |
| S6 confirmation yes | 1 | 4,351 |
| S7 confirmation no | 1 | 4,351 |
| S8 tool fails, replans | 2 | 7,350 |
| S9 cancelled before acting | 1 | 4,351 |
| S10 specialist grant | 4 | 71,286 |
| S11 second host refused | 3 | 40,067 |

S10's first planner call is large (39,186 bytes) because the specialist prompt carries the fake
page text; the two covered steps reuse the grant and add no prompt beyond the specialist's own
decisions. Trim remains **off by default**.

### H5 - `jarvis doctor` reports optional backends

New `src/jarvis/doctor.py` exposes `optional_backend_checks()`. Every probe is
`importlib.util.find_spec` (or `jarvis.perception.ocr.backend_name`), so doctor never imports a
heavy backend, opens a device, touches the network or reads user data. Missing extras are WARN,
never FAIL. `cli.run_doctor` appends the rows after the existing `vision` check; the command's
rendering and exit code are unchanged.

On this PC `jarvis doctor` exited 0 and printed:

```
 PASS  screenshot          ready (mss + Pillow)
 PASS  ocr                 ready (rapidocr_onnxruntime)
 PASS  volume              ready (pycaw)
 PASS  playwright          package installed (run `playwright install chromium` if needed)
```

Existing rows stayed as before (all PASS except `strict_zero_cost` WARN, `vision` WARN because no
vision role is configured, and `health` WARN because the daemon is not running).
`tests/unit/test_doctor_backends.py` 3 passed.

### Remaining items after batch H

- **Browser fixture**: implemented in batch I (`cd26e6a`), still **UNTESTED**; the owner runs it with
  the Batch I commands below. The autouse hazardous-call guard was not weakened.
- **Prompt trim**: left off by default. H4 proves the trim path is safe for the 11 fake scenarios;
  the owner still needs to review the G5/H4 sizes before enabling `agent.prompt_trim_enabled`.
- **Optional extras**: `pycaw`, `mss` and `rapidocr-onnxruntime` are now installed on this PC and
  declared in `[tools-extra]`, so the old G bullet saying they were undeclared/missing is superseded.
  The clean-failure paths remain intentional for machines without the extras.
- `contacts.json` still does not exist; WhatsApp paths refuse until the owner adds contacts.
- One real specialist task (live browser) has not been run; S10/S11 are fakes only.
- No vision role is configured; the vision model decision is still open.
- `browser_final_click` is still design-only (`docs/FINAL_CLICK_DESIGN.md`) pending the eight owner
  decisions.
- Orchestrator (route/dispatch/compose split) and voice-latency tuning remain.

## Batch I (I0-I2)

| Step | Commit | Message | Status |
|---|---|---|---|
| I0 | none (read-only) | fixture inventory | n/a |
| I1 | `cd26e6a` | `test: implement browser fixture tests` | **UNTESTED** (no pytest run, per rules) |
| I2 | this commit | `docs: fixture run instructions` | n/a |

### I0 - what existed before I1

- H1 exists: `1de4140` (`build: declare optional libs in pyproject and verify_env`, 2 files, +5).
  H2 has **no commit** - it changed no file (probe only); report it as MISSING, not lost.
- The fixture test had `pytestmark = [pytest.mark.browser]` and three bodies that only
  `pytest.skip("browser not executed in low-resource mode")`; the import skipped on the nonexistent
  `jarvis.browser.manager.BrowserManager`.
- Fixture site files (names/sizes before I1): `__init__.py` 0, `cart.html` 108, `checkout.html` 125,
  `form.html` 133, `result1.html` 139, `results.html` 146, `search.html` 458, `shop.html` 189,
  `video.html` 352, `video1.html` 139 bytes. There was no `index.html`, and the links were
  extensionless (`/results`, `/result1`) while files are `*.html`.

### I1 - implemented tests (not run here)

`tests/integration/test_browser_fixture.py` now uses the real wrapper
`jarvis.browser.context.PlaywrightBrowser` (the only live `BrowserSession` implementation; there is
no `BrowserManager`). It is opt-in and Windows-only:

- module markers: `browser`, `live`; skip unless `JARVIS_BROWSER_TESTS=1`; skip off Windows. The skip
  reason names the exact env var. A missing import or `BrowserError` on start is still a skip.
- `fixture_server` serves `tests/fixtures/site` from `http.server.ThreadingHTTPServer` bound to
  `127.0.0.1` on port 0 (free), runs it in a daemon thread, and calls `shutdown()` in teardown.
- `translate_path` maps `/name` to `name.html` when the exact file is absent, matching the fixture's
  extensionless links.
- Test 1 opens `/` (`index.html`), asserts title/link text and host `127.0.0.1`.
- Test 2 opens `/search`, types into the labelled search box with `browser_type` semantics, presses
  Enter, and asserts the `/results` page/title/link.
- Test 3 opens `/results`, asserts the result link is local and the added `example.invalid` link is
  external, clicks **Result One**, and asserts `/result1` plus host `127.0.0.1` (the external link is
  not followed).

Fixture edits were required for the real wrapper: new `index.html`, `aria-label="Search"` on the
search input, an external link on `results.html`, and the `.html` fallback in `serve.py`.

`ruff check` and `py_compile` on `tests/integration/test_browser_fixture.py` and
`tests/fixtures/serve.py` were clean. **pytest was not run** (per the rules).

### I2 - exact owner commands (Windows PowerShell, repo root)

```powershell
# one-time (if not already installed)
pip install -e ".[browser,vision]"
.venv\Scripts\python.exe -m playwright install chromium
# the wrapper launches channel="chrome"; if Google Chrome is not installed:
.venv\Scripts\python.exe -m playwright install chrome

# every run
$env:JARVIS_BROWSER_TESTS = "1"
.venv\Scripts\python.exe -m pytest tests/integration/test_browser_fixture.py -m browser
Remove-Item Env:JARVIS_BROWSER_TESTS
```

For `cmd.exe`, use `set JARVIS_BROWSER_TESTS=1` and `set JARVIS_BROWSER_TESTS=` instead. The test
opens a visible Chrome window and uses the `general` JARVIS profile; close any other JARVIS browser
window first so the profile lock is free.

A pass looks like:

```
...                                                                      [100%]
3 passed
```

(exit code 0). The three names are `test_browser_goto_fixture`,
`test_browser_read_fixture` and `test_browser_find_open_result_fixture`. Without the env var the
same command reports 3 skipped with a reason naming `JARVIS_BROWSER_TESTS=1`; a missing Playwright /
Chrome installation skips with "browser deps missing or browser unavailable". Any wrong title, host,
path or navigation timeout is a **failure**, not a skip.
