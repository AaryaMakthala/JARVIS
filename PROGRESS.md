# PROGRESS.md

Maintained by the coding agent. Condensed from historical log.

## Current project state

- **Agent graph**: LangGraph pipeline with brain/replan/validate/policy_gate/act/verify/respond/memory nodes; checkpoints SQLite; state typed (Pydantic).
- **Policy**: Deterministic tier engine, path resolution/symlink checks, confirmations bound to action hash, Tier 3 blocked in code, deletes go to Recycle Bin.
- **Tools**: 40 registered tools in closed registry; browser/WhatsApp/screen/UIA/camera/dictation present.
- **Daemon**: IPC on 127.0.0.1 with token, task slots, confirmations, autostart support.
- **Voice**: Wake word/stt/tts present; barge-in supports bare 'stop' per D22 (overrides old Stage 1 note).
- **Browser**: Managed Playwright browser + UIA tools; session referents exist but page referent cleared by unrelated actions (P3 added).
- **WhatsApp**: WhatsApp Web via isolated profile; content-hash binding, recipient precheck, rate limiting, W3 read-only conversation access (`whatsapp_read_messages`, Tier 1, bounded to 20 visible text messages, tainted output), and W4 confirmed text messaging (`whatsapp_send_message`, Tier 2, exact-text approval + composer read-back, operation-bound duplicate-send guard, delivery unverified).
- **Memory**: SQLite memory store (skills/failures/preferences/task_log), retrieval capped; explicit preferred-name capture persists (`user_name`), with targeted CLI deletion (`memory forget-pref`).
- **Benchmark**: Harness exists in benchmarks/; full runs not completed in condensed form.

## Safety invariants and non-negotiable decisions

- **LLM proposes, policy decides**: No LLM output can lower a tier.
- **Unknown tool rejected**: Only registry tools callable.
- **No generic shell/PowerShell/eval/exec** in v1.
- **Tier 3 blocked**: Payments/password change/Defender/firewall disable/system folder deletion/saved browser passwords blocked.
- **Paths resolved**: Absolute real paths + symlink/junction checks before policy/file ops.
- **Deletes to Recycle Bin**: Via send2trash with undo log.
- **Confirmations bound**: Exact tool+args hash; rechecked at act.
- **Secrets never in state/logs/prompts**: Password verification in daemon layer.
- **Untrusted data flagged**: Web/page/file/window text wrapped/tainted.
- **Fail closed**: If verification impossible, do not proceed.
- **IPC localhost + token**: Constant-time compare.
- **Audio local**: Only text to LLM APIs.
- **Free-tier only (effective)**: Groq/OpenRouter/NVIDIA; Gemini optional. (unverified) Pricing enforced by registry/qualify.
- **Persona/automation**: Butler tone 'Certainly, sir'; automation follows user's exact instructions.

## Known bugs, limitations, and open items

- **Stage 8 (WhatsApp/camera/browser selection)**: NOT complete. P2 (browser result selection) and P3 (reference invalidation) logic added but mandatory live checks not fully verified; camera/WhatsApp smoke tests deferred.
- **W3 WhatsApp reads**: unit-verified only. The message-container/text selectors are conservative fallbacks and have **not** been checked against a live WhatsApp DOM; summary generation is unit-tested with a capturing fake LLM, not a live provider. Name-only ambiguity remains a W2 limitation (same-display-name contacts are refused when ambiguous).
- **W3.1 read receipts**: NOT verified. The read path opens the chat through W2's number search and performs no explicit receipt action; whether WhatsApp marks messages read merely by opening the chat cannot be determined without a live account. Do not claim receipts are suppressed.
- **W4 WhatsApp text sends**: unit-verified only. The composer/Send selectors and the click-observation boundary have **not** been checked against a live WhatsApp DOM. The tool reports `verified=None` with `send_outcome="dispatched"` or `"unknown"`; delivery/read status is NOT verified, and an unknown click is never retried automatically. The exact message is intentionally present in the confirmation payload/checkpoint (required for approval) but is excluded from tool data, logs and skills.
- **Browser referents**: Page referent cleared by unrelated actions; cross-thread isolation exists in logic, live verification partial (P2/P3 NOT VERIFIED in places).
- **OCR**: Local OCR dependency not installed; OCR accuracy unvalidated.
- **Gemini/strict_zero_cost**: Under strict default Groq/Gemini would be refused; current config has strict_zero_cost=false.
- **Context windows**: Not registered numerically; overflow handled reactively from provider errors.
- **PROGRESS/roadmap naming**: Old roadmap filenames had doubled .md.md extensions historically (unverified for all links).

## Test and benchmark status

- **Unit/integration tests**: Extensive suite exists (140 files noted earlier). Last automated check for browser/referents: 118 passed, 0 failed (2026-10-23 context). W3.1 focused run (2026-10-30): 354 tests passed across 15 selected files, 0 failed. The pre-existing `test_whatsapp_browser_denylist.py` fake was missing the `wait_for_ready`/`session_state` methods introduced before W3; the fake now supplies them and all 44 denylist tests pass. `tests/unit/test_whatsapp_read_messages.py` passes 37 tests; `test_replan.py` passes 32, including the new untrusted-tool-output delimiter test. W4 focused run (2026-10-30): 37 tests in `tests/unit/test_whatsapp_send_message.py`, plus the related send-file/read/contact-search/denylist/content-binding/act/policy-gate/voice/memory/replan suites, all passed; targeted `ruff check` passed. `mypy src/jarvis/policy` passed on the earlier W4 run (no policy files touched since). A repo-wide `ruff check .` still reports 146 pre-existing errors outside this change.
- **Lint/format**: Ruff/mypy used; last recorded ruff check passed for affected files.
- **Benchmark**: Harness in benchmarks/ with tasks.yaml; full benchmark runs not recorded as complete.

## Chronological milestones (condensed)

- **Phases 0-6**: Core graph/policy/tools/daemon/voice/screen implemented and tested.
- **W0-W2 (WhatsApp)**: session-state fail-closed checks, browser denylist hardening, and verified number-based contact search committed.
- **W3 (2026-10-30)**: read-only WhatsApp conversation access added. `WhatsAppDriver.recent_messages` validates 1..20, requires an authenticated session, proves the chat header before and after the read, refuses a conversation switch, and skips media/malformed entries. `PlaywrightWhatsAppPage` reads only `#main`, the newest window, visible text (last non-empty text node), sender/timestamp when exposed. `WhatsAppWebAdapter.read_messages` opens via the W2 number-search route. `whatsapp_read_messages` is Tier 1, ContactStore-resolved, has no send/file/url args, returns `tainted=True`, and optionally summarizes through a dedicated untrusted-data prompt that escapes delimiter characters. Unit coverage: 37 tests in `tests/unit/test_whatsapp_read_messages.py`; live WhatsApp DOM and live summarizer NOT verified.
- **W3.1 (2026-10-30)**: audited the full read path. Message text reaches only the dedicated tool-free summarizer, is persisted nowhere, and never enters a planner prompt directly; the summarizer pipeline itself is tool-free. One gap found and fixed: when a later step fails, the replanner previously received prior tool output raw. Its user prompt now wraps up to 20 tool outcomes in `<untrusted_data>` and escapes `<`/`>` so a WhatsApp message cannot close the block. The stale W1 denylist test fake was updated only with `wait_for_ready`/`session_state`; the origin/identity assertions are unchanged. Read-receipt behavior remains unverified.
- **W4 (2026-10-30)**: confirmed WhatsApp text messaging added. `whatsapp_send_message` is Tier 2 (Confirm + unlocked), bound to the canonical contact + exact message hash, resolved only through the ContactStore, and sent through the W2 number-search route. The driver composes the exact text, proves the chat header and composer contents immediately before the click, requires exactly one Send control, and raises `WhatsAppSendUncertain` when the click result cannot be observed. A persisted operation-bound guard (`sha256(task_id:action_hash)`) refuses a second attempt of the same approval; a different approval may send the same text. The guard shares one process-wide lock per guard file, so concurrent dispatches of the same approval cannot both claim it; a second process remains outside that lock and relies on the persisted window. Dispatched and unknown outcomes return `verified=None` so the graph never auto-retries and memory_save never persists the message as a skill. `ToolContext.operation_id` is set by `act`; the tool is registered and in the `messaging` group. Voice Tier 2 approval intentionally does not include `whatsapp_send_message` (no voice readback was added), so it remains password-confirmation only. Unit coverage: 37 tests in `tests/unit/test_whatsapp_send_message.py`; live DOM and delivery/read status NOT verified.
- **Stage 7**: Browser DOM+UIA implemented (code complete; some live checks noted).
- **Stage 8**: Camera/WhatsApp/referents work in progress; not complete.
- **D22**: Bare 'stop' barge-in enabled (voice).
- **P3**: Browser page referent invalidation added (2026-10-23).
- **L4**: Enter now synchronizes on Playwright navigation; click rechecks the post-action origin; unauthorized cross-origin actions set the redirect block. Narrow fixture checks pass (2026-10-26).
- **L5**: Redirect-block enforcement now covers reads/actions/side effects (`resolve`, `current_url`, `scroll`, `wait`, `back`, `forward`, tabs, screenshot, WhatsApp accessor); screenshot revalidates its origin before capture. Targeted browser-context tests pass; live checks not run (2026-10-26).
- **5B**: Preferred-name capture hardened against quoted/reported/hypothetical/negated text, contextual/time phrases, three-word truncation, command-shaped suffixes, and punctuation-separated continuations; imperative statements require every candidate word to carry name-shape evidence, and an ambiguous candidate is rejected whole rather than trimmed to a prefix. Overlapping statements resolve by sentence position. Compose tests exercise the real preference-read path, and `memory forget-pref <key>` distinguishes invalid keys/storage errors from a missing preference. A fail-closed guard now prevents name learning and preference writes when the turn is classified `skipped_sensitive`; regression coverage added. Tests updated but not run in this pass (2026-10-26).
- **5C**: Consequential browser click (`browser_final_click`) now confirms against a **live observed readback**. A new `prepare_readback` graph node - one hop *before* the interrupting `policy_gate` - resolves the real target read-only, builds the readback from observed fields only (`live_dom`/`observed_origin`, each field tagged with its source), and persists it as `pending_readback` so it survives the interrupt/resume round trip (an interrupted node re-runs from the top, so an observation made inside it would be lost). The digest is sha256 over the canonical JSON (`sort_keys`, no insignificant whitespace, UTF-8) of the *verified* fields only; the model's own words travel separately under an explicit "proposed, unverified" label and can never enter the digest. On resume the gate re-verifies task/step/action-hash/target-identity/origin association, requires a well-formed 64-hex acknowledgement and compares it constant-time with the persisted digest, then writes the **trusted** digest (never the client's value) into `gated_browser_readback_hashes` and consumes `pending_readback`. `act` refuses a final click with no or malformed binding, and the tool re-observes immediately before dispatch, revalidating origin/password/payment and rebuilding the digest with the same helper (mismatch refuses before the journal reserve). A configured `final_click_spending_cap` now fails closed: the amount **and** its currency must both be observed, and missing/ambiguous/unparsable/negative/currency-mismatched/over-cap values are refused - never defaulted to 0. Tier 3 payment/purchase prohibition, `final_click_max_total = 0` default, the journal one-time reserve and indeterminate-no-retry, and voice approval (still off) are unchanged. Tests written but **NOT EXECUTED** in this pass (2026-10-27).

## Phase 5C open items (2026-10-27)

- The readback can only describe what the live control itself exposes. A consequential control whose material detail (recipient/item/message summary, amount+currency) is not observable on the control is now **refused** rather than confirmed - recoverable refusals let a specialist re-plan, but the feature is deliberately unusable until the page shows the detail. `browser_final_click` stays disabled by default (`final_click_max_total = 0`).
- `build_observed_readback` accepts a `validated_typed` source in addition (never in place of) an observation; no call site passes one yet, and the gate and the tool must always pass the same typed data or the digests will (fail closed) disagree.
- Unit tests for this slice are source-only: pytest/ruff/mypy were not run (deferred verification).
- **5C (fix)**: Two review findings closed. (1) `src/jarvis/browser/tools.py` had two module-level `_resolve_browser_element` definitions - the pre-existing `(args, ctx) -> ResolvedTarget` resolve hook bound by `browser_click`/`browser_type` and a new `(session, ref)` helper - so the later definition shadowed the hook, the resolver's `AttributeError` was swallowed by `resolve_target`, and both Tier 1 browser tools refused every step; the helper is renamed `_observe_element_for_readback`. (2) The two `verify` conditional-edge maps now route `"policy_gate"` to `prepare_readback` like the `validate` maps, so every declared edge into `policy_gate` comes from `prepare_readback` (a non-first `browser_final_click` step was previously unconfirmable). Also: every non-approval return in `_approve_single_step` (timeout, decline/cancel, typed-mismatch, locked session) now clears `pending_readback` with no readback binding written. Tests for all three added (source-only, NOT EXECUTED; 2026-10-27).
