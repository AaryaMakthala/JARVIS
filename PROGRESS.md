# PROGRESS.md

> Maintained by the coding agent. Update at the END of every session. Keep it short and factual.

## Session 2026-09-28 — Phase 8 risk classifier (`ml/`) + `jarvis undo` (no commit)

### What was asked for

Continue from `e9a7399` and implement the genuinely unfinished product capabilities rather than
stopping at analysis. Two remained, both flagged in the previous session:
`src/jarvis/ml/__init__.py` was a stub, and the undo tool existed with no CLI.

### Product capability 1 — risk classifier (`jarvis.ml`)

The policy engine already had an escalate-only hook (`max(base_tier, rules, classifier)`) and a
`RiskClassifier` Protocol, but nothing implemented or supplied it. Now:

- `ml/serialize.py` — the documented action string
  `tool=… | args={…} | user="…" | tainted=…`, with `<PATH:…>` / `<PHONE>` redaction and hard bounds
  on depth, list length, string length and total size (a hostile arg object cannot blow up the input).
- `ml/signals.py` — deterministic lexical evidence → `safe` / `sensitive` / `dangerous`. The tool's
  declared tier is one *prior vote*; signals add weighted votes; the arg-max label's margin maps to a
  softmax-comparable confidence; below the threshold the label escalates one step, clamped so
  `dangerous` never becomes Tier 3. Negation cues ("don't delete anything") cancel a signal.
- `ml/inference.py` — optional ONNX wrapper. `onnxruntime` / `tokenizers` are imported lazily inside
  `OnnxPredictor.load`; every foreseeable failure raises `ClassifierUnavailable`, so the caller
  degrades instead of crashing. No model ships, so the lexical backend is the default.
- `ml/risk.py` — `RiskClassifier` (`min_tier` / `predict` / `explain`), `build_classifier(settings,
  registry)`. `min_tier` never raises and never returns outside 0..2. `auto` uses ONNX when a model is
  present and lexical otherwise; an explicitly requested but unusable ONNX backend disables the layer
  and the rules run alone.
- Config: `[risk] enabled / backend / threshold / model_dir` in `config.py`.
- Wired into production: `cli._task_context` and `daemon/server.py` both build the registry **once**
  and derive the classifier's per-tool priors from it, so rules and classifier decide from the same specs.

`PolicyContext` gained `user_input` (read **only** by the classifier — no rule may change because the
user phrased something differently) and now receives the tainted result fragments it always declared.
`agent/nodes/util.py::policy_context_for` is the single place that builds that context, shared by
`policy_gate` and `validate`.

### Product capability 2 — `jarvis undo`

Implemented in `cli.py` as the deterministic, LLM-free path the spec's CLI table asks for
(`docs/01` line 50). It runs a hard-coded `Step` through **the same** `PolicyEngine` the graph uses,
then through the same gates in the same order as `resolve_interrupts`: approval → typed-name
confirmation → Tier-2 password. There is no `--yes`/`-y`: an approval obtainable more easily from the
command line than from the chat REPL would defeat invariant 7 (asserted by test). It opens no memory
backend and no checkpointer, so there is no persistent state to leak or leave locked.

### Real defects found and fixed (not test bugs)

| # | File | Defect | Fix |
|---|------|--------|-----|
| 1 | `agent/nodes/{policy_gate,validate}.py` | The tainted-fragment collector was **dead code**: both nodes built a bare `PolicyContext`, so `tainted_fragments` was always empty and the docs/03 §8 taint escalation could never see real untrusted output | One shared `policy_context_for(state, ctx)` helper collects tainted `StepResult.output` and threads `user_input`; both nodes use it |
| 2 | `ml/signals.py` | With no evidence and no prior, the tie-break picked the most severe of three zero-vote labels, so an unrecognised tool reading "hmm" was reported **Tier 2** | `_winner` returns `safe` when nothing fired — "no opinion" is not "dangerous" |
| 3 | `ml/signals.py` | A bare noun such as "defender" in a read-only question ("is defender on?") fired a protection-tampering signal, escalating `defender_status` to Tier 2 | Tampering is now a composite verb+noun co-occurrence; bare nouns are not signals |
| 4 | `ml/signals.py` | A prior/text tie plus the low-confidence escalation **stacked two steps** of caution, so "lock the computer now" jumped from Tier 0 to Tier 2 | Ties break toward the prior, and one ambiguous sentence escalates one label |
| 5 | `ml/signals.py` | The "no counter-evidence" confidence shortcut tested the prior tier for truthiness, so `audit_run` ("run a security audit", no keyword matched) got confidence 0.5 instead of 1.0 | Presence check is explicit (`prior_label is not None and … in LABEL_MIN_TIER`) |
| 6 | `ml/signals.py` | Taint was a heavy `dangerous` vote, which turned any web-sourced Tier-1 write into a password prompt — stricter than docs/03 §8, which escalates taint to Tier 1 and no further | Taint now raises the *floor* to Tier 1 exactly as §8 does; the flag still travels in the serialised text and the audit trail for the trained model |
| 7 | `tools/files.py` | `_verify_undo_last_delete` reported `verified=False` for a successful **no-op** ("nothing to undo"), because no `original_path` was present — a caller checking `verified` would report a phantom failure | No restore attempted ⇒ `verified=None` ("not applicable", the value dry-run already uses). A real failed restore still reports `False` |
| 8 | `pyproject.toml` | `onnxruntime` / `tokenizers` are untyped optional-extra imports, so `import-untyped` was unfixable without a new dependency — same situation the previous session fixed for `psutil` | Added a config-only `[[tool.mypy.overrides]]`; no new package |

### Design decision worth defending in the viva

`jarvis undo` passes an **empty** `user_input` to the classifier. The command is a hard-coded
single-tool invocation, not an LLM plan, so there is no user wording to classify, and inventing a
sentence there would let a fixed string decide a safety tier. The classifier still runs on the action
itself, so a future change to the tool's declared tier is still enforced; with an empty string the
documented Tier 1 stands and the mandatory approval is the gate. Feeding it the literal text
"undo the last delete" instead would have matched the destructive-verb signal and demanded a password
for a Tier-1 tool.

### Verification

- `tests/unit/test_risk_classifier.py` (50 tests) — serializer contract/redaction/bounds, signal
  arithmetic and every false-escalation regression above, backend selection, and a registry-wide
  property test that no tool's tier, `needs_confirm` or `needs_unlock` is ever lowered.
- `tests/unit/test_undo_cli.py` (18 tests) — real registry/engine/classifier/tool; only `FakeDirTrash`
  and a `tmp_path` undo log replace the irreversible edges. Covers: no `--yes`; decline restores
  nothing; approve restores and verifies; prompt shows tier + action; one prompt per run; Tier-2
  wrong/right password; no unlock manager ⇒ fail closed; typed name must match; Tier 3 never prompts;
  "nothing to undo" is exit 0; dead Recycle Bin and out-of-roots targets fail non-zero; `--dry-run`
  restores nothing; no memory/checkpointer opened; the shipped stack really yields Tier 1.
- `tests/unit/test_invariants.py` — new `test_invariant_12_risk_classifier_can_only_raise_a_tier`
  (invariant 12 now covers the classifier as well as the LLM).
- `368 passed` across the classifier, undo, invariants, files, run/CLI, config, memory, replan, agent,
  act, untrusted, daemon and schema unit suites; `30 passed` in `tests/integration`.
- `ruff check .` clean; `ruff format` clean for every file touched.
  `mypy src/jarvis/ml src/jarvis/policy` clean (the strict gate is `policy`).
- Manual: `jarvis doctor` still exits 0; `jarvis undo --help` lists only `--dry-run`.

### Known / remaining

- Phase 8 acceptance is now met on all three bullets: skill retrieval + call counting
  (`memory_retrieve`, `test_memory_graph.py`), retry→replan with an honest final message
  (`test_replan.py`), and a test proving the classifier can never lower a tier.
- No ONNX model ships. `backend = "onnx"` is implemented and tested against an injected predictor and
  against its failure paths, but exercising it end-to-end needs a trained export (docs/08 §5,
  deliberately human work).
- `lock_jarvis` is declared `base_tier=0` while docs/04 lists it as Tier 1; the classifier currently
  raises it from the wording. Left as-is rather than silently editing an unrelated tool's tier —
  worth a decision.
- `mypy src/jarvis` overall is still noisy (~150 pre-existing errors, mostly LangGraph `add_node`
  overloads); pre-existing, not a regression.
- `tests/unit/test_llm_client.py` fails `ruff format --check` in the baseline; untouched this session.

## Session 2026-09-28 — per-task telemetry (`task_log`) + audit self-review over it (no commit)

### What was asked for

Continue with the next genuine unfinished capability rather than stopping after a passing test.
Chosen: durable per-task telemetry. The `task_log` table had existed in the schema since Phase 8,
and Phase 7/§07 acceptance criteria depend on it, but **nothing ever wrote a row** — so task
success rate, latency, token spend and "which tasks fail repeatedly" were all unmeasurable.

### Files added

- `src/jarvis/memory/tasklog.py` — `TaskLogStore`, `TaskRecord`, `TaskSummary`, `safe_input`,
  bounded pruning, redacted/truncated command text, duration + RSS capture, `summary()`,
  `recent()`, `failure_repeats()`, `clear()`.
- `src/jarvis/agent/nodes/telemetry.py` — `record_started`, `record_finished`, closed-vocabulary
  `task_status`, `--dry-run` suppression, legacy-backend tolerance, exception isolation.
- `tests/unit/test_task_log.py` — 30 tests.

### Real defects found and fixed (not test bugs)

| # | File | Defect | Fix |
|---|------|--------|-----|
| 1 | `agent/runner.py` | `TaskOutcome.task_id` was generated by the runner while the graph generated its **own** `task_id` in `intake`. Every telemetry row would have been keyed to an ID the graph never used, so join-by-task-id silently matched nothing | The runner's ID is now seeded into graph state, so both agree on one task ID |
| 2 | `agent/nodes/intake.py` | Telemetry opened even under `--dry-run`, so a rehearsal mutated the very database the audit reads | Telemetry is suppressed entirely in dry-run; covered by test |
| 3 | `audit/checks.py` | The `self` audit section read **only** the JSONL log. `self_review` in production was therefore "unknown", which is exactly how the Phase 7 criterion "repeated failures are listed honestly" silently never held | The section now runs `task_history` as well, and both callers (`cli._run_audit_cli`, `tools/audit.py`) pass the real `log_path` and `memory_db`. Injectable params keep unit tests OS-independent |
| 4 | `pyproject.toml` | `psutil` has no bundled stubs and `types-psutil` is not a declared dev dep, so **4 pre-existing** `import-untyped` errors were permanently unfixable without a new dependency | Added a `[[tool.mypy.overrides]]` entry for `psutil` (config-only, no new package). All 4 errors cleared |

### New audit findings (`jarvis audit --sections self`)

`Repeated task failures` (warning at >=2x the same normalised input, `ok` below that),
`Task success rate` (warns at >=50% failed; a task that died mid-flight is counted as
**unfinished, never as a success**), and `LLM cost per task` (calls/task, tokens, replans).

### Verification

- `363 passed` across the memory, agent, audit, CLI, daemon, runtime, config and invariant suites
  (bounded selection, not the full suite).
- `30 passed` in `tests/integration`.
- Ruff check + format clean. `mypy src/jarvis/policy` clean (the strict gate). New files
  `tasklog.py` / `telemetry.py` mypy-clean.
- Read-only proven by test: row count, `user_version` and mtime unchanged after an audit, and a
  missing DB is **not** created by an audit.

### Known / remaining

- `src/jarvis/ml/__init__.py` is still a stub (Phase 8 risk classifier).
- No CLI-level `jarvis undo` yet, though the undo tool and log exist.
- `mypy src/jarvis` overall is still noisy (~150 pre-existing errors, mostly LangGraph
  `add_node` overloads); this is pre-existing and not a regression.

## Session 2026-09-27 — voice lifecycle proof, exactly-once steps, dynamic provider health (no commit)

### What was asked for

Prove the voice lifecycle on the console, make a non-idempotent step impossible to run twice,
keep provider preference dynamic (never a permanent blacklist), keep internal routing JSON out of
user-visible output, and make "2 times 2" / "2 multiplied by 2" behave identically *without*
hard-coding arithmetic in the app.

### Real defects found and fixed (not test bugs)

| # | File | Defect | Fix |
|---|------|--------|-----|
| 1 | `voice/loop.py` | Idle timeout was wall-clock only, so any stream delivering frames faster than real time (test fake, or a device replaying a buffer) satisfied it thousands of times a second and **busy-spun a CPU core** | Deadline measured in *audio seconds consumed*, with a wall-clock slack (`_IDLE_WALL_SLACK_S`) so a fully stalled stream still yields, plus a cancellable `_idle_backoff()` so re-arming cannot spin |
| 2 | `voice/status.py` | `transcribing()` was never called, so every real interaction logged an **illegal `CAPTURE_COMPLETE -> THINKING`** jump | Added `transcribing()` and called it from `_run_interaction`; transitions are now clean |
| 3 | `voice/status.py` | Console printed the enum value: `[VOICE] WAKE_DETECTED`, `[VOICE] CAPTURE_COMPLETE` — not the required text | `CONSOLE_LABELS` + `console_label()` in `voice/states.py`; JSONL keeps the machine vocabulary |
| 4 | `llm/failures.py` | `classify()` promised "Never raises" but `str(exc)` and a property-style `status_code` could raise, taking down the very fallback logic meant to catch the failure | `_safe_text()` / `_safe_status()` helpers, both failure-proof |
| 5 | `llm/health.py` | `snapshot()` read the **real** clock while every other method used the injected one, so `cooling` disagreed with `is_cooling()` and `jarvis doctor` could report the wrong state | `snapshot(now)` takes the moment; `ProviderHealth.snapshot()` passes `self._clock()` |
| 6 | `llm/health.py` | **A single 429 demoted a provider to the back of the rotation permanently.** `order()` sorted by `consecutive_failures` even after the cooldown expired, and a provider only ever got called after the others failed — so it could never win one back. This is a blacklist, which the requirement forbids | An **expired cooldown is a fresh probe** back at its configured position. The streak is deliberately *not* cleared, so failing again still earns a longer cooldown. Nothing is ever dropped |
| 7 | `voice/status.py` | The `stt` log record carried `redacted=<transcript>`. `redact()` only masks *recognised* secrets and `+`-prefixed numbers, so for ordinary speech the "redacted copy" was the **whole utterance verbatim** — the JSONL kept everything the user said | The log record now carries `chars` and `words` only. The console still shows the exact `[VOICE] STT: "…"` line, which is the channel the user opted into via `echo_transcript` |
| 8 | `agent/nodes/act.py` | The exactly-once guard called `_recorded_result` but the helper was named `_already_executed`, and `logger` was never imported | Helper renamed, `logger` defined. Guard replays a successful result and advances `step_index`; a *failed* or `verified=False` step stays retryable |

### Required console vocabulary — now exact and asserted

`READY → WAKE DETECTED → LISTENING → CAPTURE COMPLETE → TRANSCRIBING → STT → THINKING →
PROVIDER → MODEL → ANSWER → SPEAKING → RESETTING → READY`, with `[VOICE] RECOVERING` on a
recoverable failure. `READY` is printed at start-up **and** after every wake, so a missing
final `READY` is visible; `illegal_transitions == []` for a full interaction is asserted.

### Tests added (142, all green)

`tests/unit/test_voice_status.py` (36) exact terminal text, READY-restored, console/log split,
broken-stream safety, state-machine vocabulary.
`tests/unit/test_llm_health.py` (38) finite cooldowns, self-recovery, `(provider, role)`
granularity, "expired cooldown = fresh probe", stable ordering, never-dropped, classification.
`tests/unit/test_answer.py` (36) parse-driven leak detection, prose that merely says "goal" is
untouched, `safe_final_answer`, `spoken_answer`, repair-prompt text.
`tests/unit/test_act_node.py` (14) exactly-once, failed/verified-failed stay retryable, and the
guard is **not** a policy bypass.
`tests/unit/test_math_phrasings.py` (18) the four reported phrasings round-trip cleanly, and a
**structural guard fails the build** if anyone hard-codes arithmetic words in `src/jarvis`.

### `tests/unit/test_voice_loop.py` — 5 tests were rewritten, not weakened

The idle timeout no longer ends the loop (that *was* the production bug: the wake word went dead
until a restart), so `join()`-until-thread-exit could never finish. Added `_wait_until()` /
`_stop_and_join()` helpers: wait for the behaviour under test, then stop. Two tests assert
per-interaction wake-detector reset counts, so they now set `idle_timeout_s=0` to isolate the
per-interaction re-arm from the (correct) idle re-arm; `test_no_per_frame_reset_while_waiting` and
`test_wake_heartbeat_includes_audio_level` do the same for the same reason.
All 23 classes green individually.

### Decisions worth knowing

- **`CONTEXT_LENGTH` is retryable on the same provider** — a shrink-the-prompt retry can succeed,
  and a sibling model may have a bigger window. Not a bug.
- **`ToolSpec.execute()` does not verify** (that is `run_verified()`), so the act node records
  `verified=None`; the guard treats `None` as "not yet verified", not as a failure.
- **`ERROR -> READY` is deliberately illegal**; recovery from an error goes through `STARTING`.
  Resuming a listening state would hide that the mic or thread died.
- `enabled=False` on the reporter silences the **console only**; the log sink is independent,
  because turning off the on-screen status must not blind `jarvis doctor`.
- Math is the planner's job. No phrase table, no `eval`, no `ast.literal_eval` — asserted.

### Deferred / not verified

- **The full `tests/unit/test_voice_loop.py` file was never run in one process.** Every one of its
  23 classes is green individually, but the aggregate exceeds 2 minutes (real-time audio waits),
  and a whole-suite run was explicitly out of scope. The rest of `tests/unit/` is likewise unrun.
- `pytest -q` overall: **not run.** No integration, `windows_only`, voice, or benchmark run.
- `mypy src/jarvis/policy`: not run (nothing in `policy/` was touched).
- **The suspicious STT content was never traced to a source.** No audio/STT recording was captured,
  and nothing is blacklisted. It remains unexplained.
- Real microphone / real providers: not exercised. Everything above is fakes and unit tests.

### Manual smoke tests for the Windows PC

```powershell
.venv\Scripts\ruff check .
.venv\Scripts\python -m pytest -q tests/unit/test_voice_status.py tests/unit/test_llm_health.py tests/unit/test_answer.py tests/unit/test_act_node.py tests/unit/test_math_phrasings.py
.venv\Scripts\python -m pytest -q tests/unit/test_voice_loop.py     # ~3 min: use a generous timeout
.venv\Scripts\python -m pytest -q                                    # full suite
jarvis doctor
.venv\Scripts\python -m jarvis daemon --foreground
```

Say "Hey Jarvis", then **wait well past the idle timeout** (default 120 s), then say it again —
the second wake must work. Check the console shows every line in the vocabulary above, that
`READY` appears once per interaction, and that `jarvis.jsonl` contains `event=stt` records with
`chars=`/`words=` but **no transcript words**. Then unplug/replug a provider's key and confirm the
next call falls back and the provider returns on its own.

Suggested commit: `fix(voice): prove the interaction lifecycle and stop the idle busy-spin`

## Session 2026-09-25 — `keys set --visible`: explicit opt-in visible key entry (no commit)

**Status:** `jarvis keys set <provider>` keeps fully hidden `getpass` input as the default; a new
explicit `--visible` flag opts in to echoed entry and a full-key receipt. Credential storage
(`SecretStore` / Windows Credential Manager) and all LLM/provider/model config are untouched.

- New `--visible` flag on `jarvis keys set` (default off, never enabled implicitly). Visible mode
  prints `WARNING: API key visibility is enabled.` before prompting, reads through the new
  `visible_input()` (`input()` — the terminal/PowerShell line editor handles echo, paste and
  backspace), normalises any literal BS/DEL characters in the received line via
  `_apply_backspace()`, and after a successful `store.set()` prints the receipt plus
  `Key: <full key>`.
- Default mode unchanged: hidden input, masked first-4/last-4 preview only, no warning.
  `keys status` remains presence-only; `keys paste`/`clear` unchanged.
- Receipts are built strictly after storage succeeds: empty input and storage failures print no
  fragment of the key. The new code path logs nothing; the receipt is printed with rich
  `markup=False, soft_wrap=True` so a raw key is never parsed as markup or wrapped mid-line.
- Tests (`tests/unit/test_keys_cli.py`, +13): visible receipt/paste/backspace/empty/storage
  failure, default visible reader is terminal `input()`, default invocation stays hidden and
  unwarned, command-level default receipt stays masked, and caplog assertions that the full key
  never reaches logs in either mode. Test keys are obviously fake fixtures.
- Validation: focused keys/secrets/init-flow = 56 passed; full `pytest -q` = **947 passed,
  6 skipped** (pre-existing daemon async-mock warnings only); `ruff check .` +
  `ruff format --check .` clean; `mypy src/jarvis/policy` Success. `mypy src/jarvis/cli.py`
  reports 8 pre-existing errors (missing `types-pyperclip` stub + untouched password/contacts
  code); none fall in the changed lines.
- Manual smoke (user, PowerShell): `jarvis keys set groq` → hidden as before;
  `jarvis keys set nvidia --visible` → warning, characters echo while typing/pasting, receipt +
  `Key: <full>`; empty Enter in either mode → exact rejection, nothing stored; `jarvis keys
  status` → presence-only lines.

## Session 2026-09-25 — masked preview after `keys set` (no commit)

**Status:** `jarvis keys set <provider>` keeps its hidden `getpass` input and the unchanged
`SecretStore`/Windows Credential Manager path; after a successful store it now prints the receipt
followed by a masked preview line, e.g. `Key: gsk_•••…KXYZ`.

- New `masked_key_preview()` in `cli.py`: first 4 characters + one `•` per hidden character +
  last 4; keys shorter than 9 characters are shown fully masked (`•`×len) so no meaningful
  first/last split is revealed.
- The preview is built only **after** `store.set()` succeeds: empty input and storage failures
  return no fragment of the key, and no `Key:` line is printed. The complete key never appears in
  output, logs, exceptions, or doctor; keyring errors/logs carry only the secret name and length
  (unchanged). `keys status` stays presence-only.
- Input UX intentionally unchanged (spec keeps the secret fully hidden while typing/pasting; the
  earlier per-character masking idea was superseded by that requirement).
- Validation: 43 focused keys/secrets/init-flow tests pass; full `pytest -q` = 934 passed,
  6 skipped (pre-existing daemon async-mock warnings only); `ruff check .`,
  `ruff format --check .`, `mypy src/jarvis/policy` all pass.
- Manual smoke (user, PowerShell): `jarvis keys set groq` → paste a key + Enter → receipt then
  `Key:` preview showing only first/last 4; `jarvis keys set groq` + Enter on empty prompt → exact
  rejection, no `Key:` line; `jarvis keys status` → presence-only lines.

## Session 2026-09-25 — secure API-key receipt feedback (no commit)

**Status:** `jarvis keys set <provider>` now confirms receipt without exposing the value. Hidden
input still uses `getpass.getpass`, storage still uses the existing `SecretStore`/Windows
Credential Manager path, and no real credential was read, printed, or logged during validation.

- Non-empty trimmed input is stored and followed only by `✓ API key received and saved securely.`.
- Empty or whitespace-only input stores nothing and prints `✗ No API key entered. Nothing was saved.`.
- Credential-store failures print a separate safe failure and never claim that saving succeeded.
- Windows CRLF paste input is covered; command-level tests assert the entered value is absent from
  CLI output. `jarvis keys status` remains exactly `<provider>: configured|not configured`; no
  length, prefix, suffix, hash, or masked credential diagnostic was added.
- Validation: 53 focused keys/secrets/CLI tests pass; full `pytest -q` exits 0 with five skips and
  only the pre-existing daemon async-mock warnings. `ruff check .`, `ruff format --check .`, and
  `mypy src/jarvis/policy` pass.
- Manual PowerShell smoke: run `jarvis keys set groq`, paste a key and press Enter, verify only the
  receipt message appears; press Enter on an empty prompt and verify the exact rejection; run
  `jarvis keys status` and verify presence-only lines.

## Session 2026-09-25 — four-provider LLM fallback (no commit)

**Status:** Groq → OpenRouter → NVIDIA → Gemini is implemented and covered with network-free
factory, fallback, LangGraph, daemon, and voice/TTS tests. No provider is claimed live-working
without a validated credential and a real inference result.

### Configuration and policy
- Canonical order is `groq`, `openrouter`, `nvidia`, `gemini`; the existing
  `MultiProviderClient` walks this order for every structured/text operation.
- Per-provider planner/fast models remain the already registered exact IDs: Groq
  `openai/gpt-oss-120b` / `openai/gpt-oss-20b`, OpenRouter `openrouter/free` for both roles,
  NVIDIA `nvidia/nemotron-3-super-120b-a12b` /
  `nvidia/nemotron-3.5-lightning-30b-a3b`, and Gemini `gemini-3.8-flash` /
  `gemini-3.7-flash`.
- Credentials continue through Windows Credential Manager first, then the documented provider
  environment variables. No key was printed, logged, tested, or added to configuration/diffs.
- The safe generated default remains `strict_zero_cost = true`. The current runtime config uses
  the explicit `free_only = true`, `strict_zero_cost = false` opt-in so Groq/Gemini free-tier
  models can participate; their account billing state is not guaranteed zero-cost.

### Implementation and validation
- Plain doctor now validates every configured planner/fast model even when that provider's key is
  absent, while selecting only providers with both eligible models and a readable credential.
- Added four-provider tests for exact order/models, healthy Groq selection, each fallback edge,
  exhaustion, missing/invalid providers, credential non-leakage, LangGraph, daemon injection,
  and wake → STT → multi-provider agent → TTS. Existing OpenRouter route/header behaviour is
  asserted unchanged.
- Current non-live doctor: all four model checks PASS; Groq/Gemini credentials are present, while
  OpenRouter/NVIDIA are not visible to this JARVIS process. No live request was run.
- Focused provider/config/agent/daemon/voice suites, `pytest -q`, and
  `pytest -q -m "not windows_only and not slow and not voice"` pass. `ruff check .`,
  `ruff format --check .`, and `mypy src/jarvis/policy` pass. Only the pre-existing daemon
  async-mock coroutine warnings remain.

## Session 2026-09-25 — real-LLM voice pipeline diagnosis and runtime hardening (no commit)

**Status:** implementation and mocked end-to-end coverage are green. Live E2E remains blocked by
invalid stored Groq/Gemini credentials; no key or user runtime config was changed.

### Root cause
- The real config selected Groq/Gemini but had no per-provider model map; strict zero-cost mode
  also correctly blocks their free-tier models until the user explicitly opts out.
- Authentication-only provider probes rejected both stored credentials (401/400), so no live model
  could be verified or used. Current official Groq docs list both configured model IDs, but publish
  paid token pricing; selecting them therefore still requires explicit `strict_zero_cost = false`.
- The daemon started voice before building `AppContext`, allowing an immediate wake to submit with
  no context; an injected/reused voice service was not always linked into the context bridge.

### Changes
- Build the context before voice startup, refresh the bridge before/after service creation, and
  fail a pre-startup voice submission cleanly instead of raising.
- Preserve keyring-to-environment credential fallback when keyring access fails; credential
  presence probes now remain non-throwing.
- Make pricing qualification strict by default and make doctor/provider status validate every
  configured planner/fast role. `jarvis init` now warns about the Groq/Gemini strict-mode block.
- Add a fully mocked wake → STT → real `GroqClient` transport → LangGraph → TTS regression test
  plus daemon startup/bridge, credential fallback, role diagnostics, and policy-default tests.

### Validation
- Focused affected suites: 166 passed.
- `pytest -q`: exit 0, 6 skipped; only the pre-existing daemon async-mock coroutine warnings.
- `pytest -q -m "not windows_only and not slow and not voice"`: exit 0, 4 skipped; the same
  pre-existing warnings remain.
- `ruff check .`, `ruff format --check .`, and `mypy src/jarvis/policy`: pass.
- PowerShell environment-override smoke resolved Groq planner/fast models and
  `strict_zero_cost = false`; plain `jarvis doctor` then marked both model and provider checks PASS
  without inference or key output.

### Manual next step
- Replace the Groq key, configure Groq planner/fast models and `strict_zero_cost = false` as an
  explicit free-tier opt-in, run `jarvis doctor --live`, then repeat wake/STT/response/TTS.


## Session 2026-09-24 (later) — voice re-arm quiet-start gate + Phase 7 Audit DONE (no commit)

Two unrelated work items, both green on the dev PC.

### Part 1 — wake re-arm input-condition fix (completes the prior voice work)
Root cause (proven from code + fakes, not hardware): the real loop and detectors are sound
(real-loop 5/5 wakes); the defect was a **re-arm input-condition asymmetry** — after an
interaction the wake detector was re-armed while the *response TTS echo* still rang in the
mic, unlike the always-quiet first wake. So "Second interaction" hearings often lost with a
real speaker.
Fix (`voice/loop.py`): `_reset_voice_state()` order = `flush()` → **bounded quiet-start gate**
(`_quiet_start_drain`) → `wake_detector.reset()` → counters → LISTENING;
`_rearm_wake_for_confirmation()` order = speak → flush → reset → listen. The gate drains
samples until the first below-threshold (silence) chunk, bounded by drained *audio-seconds*
(`_REARM_QUIET_GATE_S=0.5`, constructor `rearm_quiet_gate_s`, 0 disables). Bounding by audio
seconds (not wall clock) is essential: fake audio calls return whole segments, so a wall-time
gate spins at CPU speed and consumes scripted feeds.
Regression proof: 4 new tests in `TestRearmQuietStartGate`; verified 3/4 fail with the fix
removed (temp backup at `%TEMP%\loop.py.fixed`, restored). Also de-flaked the pre-existing
race in `test_successful_interaction_speaks_and_keeps_listening` (poll `len(spoken) >= 2`
before asserting).

### Part 2 — Phase 7 Audit (was stubs) — READ-ONLY by design
- `audit/checks.py` (full): `Finding` dataclass + `Severity`; `SECTION_CHECKS` =
  security/performance/updates/self; fixed PowerShell whitelist **module constants** run via
  `subprocess.run([...], check=False)` — never `shell=True`, never interpolated; per-command
  timeouts (`_PS_TIMEOUTS`, updates 60 s); `_exec_safe` maps timeout/OSError to an honest
  `unknown` (fail closed — BitLocker needs admin → unknown, never a false ok). Checks:
  defender, firewall, bitlocker, listening ports (psutil), startup entries (winreg),
  scheduled tasks, performance, pending updates, self-review of the jarvis.jsonl
  (repeated ERROR/WARNING lines). `run_checks()` is injectable (executor/os_name/psutil/
  winreg) for any-OS fixture tests and **never writes to disk** (tested).
- `audit/report.py`: `render_markdown` (severity rollup, sectioned, "unknown" note) +
  `write_report` → timestamped `audit-YYYYMMDD-HHMMSS.md` in `config.reports_dir()` — the
  audit package's ONLY disk write.
- `tools/audit.py` + registry: `audit_run` (Tier 0, read-only), `defender_status` (Tier 0,
  read-only), `defender_quick_scan` (Tier 1 — starts a background `Start-MpScan`, distinct
  from the read-only audit; `CREATE_NO_WINDOW` via `getattr`, `_SPAWNED_SCANS` keeps Popen
  refs). All commands come from the shared whitelist constants (tested).
- CLI: `jarvis audit [--section/-s …] [--out DIR]` — rich severity table; exits 0 even with
  critical findings (documented in report); exit 1 only on internal failure. Smoke-tested
  for real on this PC: report written with real Defender/firewall/startup/tasks data, honest
  `unknown` for BitLocker (needs admin) and the absent failure log.

Validation (dev PC): `pytest tests/unit/test_audit.py` (26) + `test_package.py` green; full
`pytest tests -m "not windows_only and not slow and not voice"` **exit 0** (only pre-existing
daemon async-mock coroutine warnings); `ruff check .` clean; `ruff format` applied; `mypy
src/jarvis/policy` Success. NOT committed.

Manual smoke test for the PC:
1. `.venv\Scripts\jarvis.exe audit` → rich table + report path under
   `%LOCALAPPDATA%\jarvis\jarvis\reports\audit-*.md`; open and eyeball findings.
2. `jarvis audit -s security -s performance` → subset run works.
3. Voice: `jarvis daemon --foreground` → 3× "hey jarvis / command / response / hey jarvis …"
   back-to-back on a real speaker — every re-arm must detect (the old echo-deaf re-arm).

## Session 2026-09-24 — ~7 s STT_RESULT→INTERACTION_COMPLETE delay — ROOT-CAUSED & FIXED (no commit)

**Root cause (proven from the real `%LOCALAPPDATA%\jarvis\jarvis\logs\jarvis.jsonl`, not
inferred)**: the whole ~7 s sat inside TTS. Per interaction: `voice task started` →
`voice task finished (final_answer_chars=108)` only ~50 ms later (the no-LLM halt in the
brain node already fired immediately), then `TTS sapi speak start (chars=108)` →
`TTS sapi speak done (elapsed_s=6.97/6.88/6.98)` → `INTERACTION_COMPLETE`.  SAPI was
reading the 108-character terminal refusal ("JARVIS's AI backend is not configured yet.
Configure an LLM provider before asking me to reason about tasks.") aloud at ~15 chars/s.
No provider init, retries, timeouts, queue waits or sleeps were involved — the routing/agent
path was already fast.

**Fix (completes the previous session's in-progress change)**:
- `agent/nodes/brain.py`: the voice-variant no-LLM halt `LLM_NOT_CONFIGURED_VOICE = "My AI
  backend is not configured yet."` (speech-sized, same honest refusal) was present but had
  no logging and no tests; added `agent event=llm_skip` (reason=no_provider, variant=voice/
  terminal) on the no-LLM path and `agent event=llm_start` / `agent event=llm_complete`
  (finally-block) around the structured call, so future delays are attributable in the JSONL
  log next to the existing `VOICE ROUTING_START/COMPLETE` markers. The no-LLM path still
  fails immediately (no retry/timeout/probe can run).
- Regression tests: `test_replan.py::test_brain_without_llm_voice_source_gets_short_spoken_message`
  (voice source → short message, ≤60 chars) and
  `test_agent_architecture.py::test_no_llm_backend_voice_source_gets_short_spoken_answer`
  (end-to-end `run_task(source="voice")` → "My AI backend is not configured yet.", zero tool
  calls). Terminal keeps the full actionable 108-char message (existing tests unchanged).
- `voice/tts.py`: removed the PIE790 `pass` left in `SapiTTSEngine.stop` (lint-only).

**Gate**: `pytest tests/unit -m "not slow and not voice"` = **832 passed, 4 skipped, 0 failed**;
tests/integration green; `ruff check .` + `ruff format --check .` clean; `mypy src/jarvis/policy`
Success. NOT committed.

**Manual verification for the PC (real hardware, still no API key)**:
1. `jarvis daemon --foreground`, say "hey jarvis" → command.
2. Expect the spoken refusal within ~1–2 s of STT completing, then re-arm.
3. Log check: between `VOICE STT_RESULT` and `VOICE INTERACTION_COMPLETE` you should now see
   `TTS sapi speak start (chars=33)` and `elapsed_s≈1.5-2.5`, plus the new
   `agent event=llm_skip ... variant=voice` line from the agent logger.

## Current phase
Phase 6 — DONE (WhatsApp contacts + whatsapp_send, verification-before-send, rate limits; fail-closed invariant #9).
Phase 2 LLM-driver — DONE (multi-provider free-only LLM layer; see "LLM-driver architecture").
**NEW — `strict_zero_cost` ("billing-safety") rework of the free-only layer — DONE** (dated entry at
the bottom): 4-tier pricing in code, `strict_zero_cost=true` safe default with per-provider
config override, registry keys pricing so unknown/unavailable models are `unavailable` and never
selectable; doctor shows zero-cost vs free-tier vs paid; 789 pass / 0 fail / 10 skip + 1 known
WSL NTFS-ADS deselect.
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

## Session 2026-09-23 (later, same day) — Brain/ResponseSink refactor DONE (no commit)

Supercedes the deferred entry below. All three locked decisions (D1/D2/D3) implemented and green.

**Unit suite: 831 passed, 4 skipped, 0 failed** (`pytest tests/unit`). Integration: `tests/integration -q`
green. `scripts/phase3_verify_confirm.py`: ALL CHECK-11 SCENARIOS PASS (incl. new Scenario G — timeout resumes
with the honest message). WSL env; run ruff/pytest natively for the final pass.

What changed:
- **Brain node** (`agent/nodes/brain.py`): the single structured LLM classification. One `BrainDecision` call
  (temp 0.0) splits: `action` → projected `Plan`/`Step`; `conversation` → `Plan(kind=conversation)+final_answer`
  (no fast-LLM second call; blank answer halts with "I couldn't produce an answer."); `clarification` → clarify
  interrupt; `unsupported` → honest halt. `plan`/`understand_request`/`converse` retired. LLM failures map to
  fixed messages (`_llm_halted_reason`); `ctx.llm is None` → "…AI backend is not configured yet…".
- **Clarify node** (`agent/nodes/clarify.py`): `interrupt(ClarificationRequest)`; free-text validation (non-blank,
  not a `_CANCEL_RE` word); `MAX_CLARIFICATIONS = 2`; answer consumed + brain re-runs with question+answer.
- **D3**: `BrainDecision`/`ActionIntent` transient — projected before state, never on the checkpoint allowlist
  (stays {Plan, Step, Decision, StepResult, ReplanDecision}). `test_serializer.py` asserts the closed exact set.
- **D1** (`daemon/task_runtime.py`): `CONFIRMATION_TIMEOUT_S = 30.0` single-sourced (server.py's 60s constant
  deleted). `TaskRuntime.wait_for_response` is the only worker wait; `timeout_answer` resumes the graph
  (`timed_out: True`) → `policy_gate` halts "Confirmation timed out. I did not perform the action." (checkpoint
  finishes cleanly). `TerminalSink` is wait-only (no publish adapter). `_stale_confirmation_loop` synthesizes the
  same refusal.
- **D2** (`voice/loop.py`): `_YES_WORDS` split → `_LOOSE_YES_WORDS` {yes,y,approve,approved,confirm,go,ok,sure}
  for ordinary Tier-1 payloads; `_STRICT_YES_WORDS` {yes,confirm,proceed} when `typed_confirmation` present or
  `tier >= 2`. Voice still fails closed (`can_confirm_by_voice`) before words match; strict set tested directly.
- **Daemon protocol**: `ClarificationRequest`/`ClarificationResponse` messages; server `_handle_clarification`
  stashes the answer + wakes the slot; client `wait_for_event` returns both request types; voice
  `capture_free_text` for clarification answers (empty → "" fail-closed).
- **New/updated tests**: `tests/support.py` brain helpers; `test_replan.py` (28, incl. `_ExplodingLLM`
  failure-mapping), `test_agent_architecture.py` (19, incl. clarify round-trip resume), `test_invariants.py`,
  `test_agent.py`, `test_freeonly.py`, integration confirm/tier2/toctou updated; new `test_task_runtime.py`,
  `test_daemon_client.py`, daemon protocol/server clarification tests, D2 + voice-clarification tests.
- **Docs**: `docs/02_ARCHITECTURE.md` §1/§2/§4/§5/§6/§8 rewritten to the brain/TaskRuntime design;
  stale "Set by the converse node" comment fixed in `respond.py`.

Known/fragile: pre-existing async-mock "coroutine was never awaited" warnings in `test_daemon_server.py`
(not failures). 4 skips = pre-existing voice/windows guards. `src/jarvis/llm/` untouched.

Manual smoke tests for the dev PC (native PowerShell):
1. `.venv\Scripts\python.exe scripts/phase3_verify_confirm.py` → ALL CHECK-11 SCENARIOS PASS.
2. `jarvis daemon --foreground` then `jarvis chat "open notepad"` → approve → runs; refuse → "Refused…".
3. Ask something ambiguous (e.g. "summarise the file") → CLI gets `clarification_request`, inline free-text
   answer resumes; blank answer → "No clarification received."
4. `ruff check . ; ruff format --check .` and `mypy src/jarvis/policy` green on the dev PC.

## Session 2026-09-23 — Brain/ResponseSink refactor: pre-read done, implementation DEFERRED (no code changed)

**Environment**: this session ran under WSL at `/mnt/c/Users/aarya/OneDrive/Desktop/Jarvis`
(OneDrive-synced). Per past WSL+OneDrive sync issues, the human asked to restart natively
(PowerShell) and RE-VERIFY the four findings against the native filesystem before any
implementation. **No source under `src/` or `tests/` was modified this session.**

**Four pre-read findings (verified against current code; must be re-confirmed natively):**
1. Confirmation = `interrupt(payload)` in `policy_gate.py` (`ConfirmationRequest` json payload);
   `resume_task` uses `Command(resume=answer)`; node re-runs from the top and re-derives the
   decision + `action_hash`; TOCTOU via `gated_resolved_paths` (stashed in `validate.py`,
   re-checked in `policy_gate` and `act`). Already implemented + tested.
2. Voice fixes to make: bridge = `TaskSlot.owner_id`; voice tasks use `VOICE_OWNER` sentinel
   (no socket); `_voice_submit` blocks the voice-loop thread on `slot.event`; Tier-1
   confirmations driven worker-side via `_run_voice_confirmation`/`loop.confirm_by_voice`;
   Tier-2+ voice = fail closed (`tier_requires_terminal`).
3. `intake` is thin/deterministic (trim + `UserRequest` + `_CANCEL_RE`, no LLM). The single
   LLM classification today is the `plan` node → `Plan(kind, goal, steps, needs_clarification,
   clarification_question, dialog_answer)`; `understand_request` just records `request_kind`;
   `converse` makes a SECOND `text()` call (or uses `dialog_answer`). Brain single-call design
   replaces `plan`+`understand_request`+`converse`.
4. Checkpoints: `open_sqlite_checkpointer` = `SqliteSaver` + `JsonPlusSerializer` strict
   allowlist of EXACTLY {Plan, Step, Decision, StepResult, ReplanDecision}, no pickle.
   Restart: `_active=None`, worker always fresh `run_task`, no auto-resume. Confirmation expiry
   is in-memory only today (worker 60s wait + `_stale_confirmation_loop`) and marks the slot
   errored — it does NOT resume the graph with a deterministic answer.

**Three decisions locked by the human (binding constraints for the refactor):**
- **D1 — One confirmation timeout.** Remove `CONFIRMATION_TIMEOUT_S = 60` in `server.py`
  entirely. Exactly ONE timeout constant in the codebase afterwards, set to 30, owned by
  `daemon/task_runtime.py`. The timeout path must RESUME the graph with a deterministic
  "Confirmation timed out. I did not perform the action." answer (not merely mark the slot
  errored, which is today's behaviour).
- **D2 — Voice approval wordsets.** Destructive/high-risk confirmations (the tier boundary the
  PolicyEngine already uses to mean destructive) require EXACTLY "yes" — "go"/"ok"/"sure" must
  NOT approve a destructive action; "confirm"/"proceed" MAY stay as synonyms. Lower-tier
  confirmations may keep the looser existing set. Final report must state explicitly which
  tiers use strict vs loose set. (`voice/loop.py:44` `_YES_WORDS` currently includes
  go/ok/sure → must be split.)
- **D3 — Checkpoint allowlist (REVISED 2026-09-23 by human: do NOT widen the allowlist).**
  `BrainDecision`/`ActionIntent` are TRANSIENT, request-scoped objects only: the brain node
  projects `actions` into the existing `Plan`/`Step` shape BEFORE anything is written to
  AgentState/checkpoints, so neither type ever appears in checkpointed state. The allowlist in
  `build_secure_serde` therefore stays exactly {Plan, Step, Decision, StepResult,
  ReplanDecision} with `pickle_fallback=False` — no new types added, nothing opened generally.
  Instead of widening it, add a test asserting BrainDecision/ActionIntent are NEVER present in
  checkpointed state (inspect the serialized bytes / allowlist-checked object graph) and that
  the allowlist remains a closed exact set. `test_serializer.py` updated accordingly.

**Explicit design ruling (must be stated in the architecture doc, not ambiguous in the diff):**
`BrainDecision` is the single NEW **LLM-facing** schema. For the ACTION path, its `actions`
are projected into the existing `Plan`/`Step` shape BEFORE anything touches AgentState (D3:
projection happens first; `BrainDecision` itself is never persisted), then
`validate`/`policy_gate`/`act` run unchanged — there is exactly ONE action representation at
the engine boundary, not two competing ones.

**Planned implementation order (WSL session approved 2026-09-23 after OneDrive confirmed not
syncing):**
schemas (D3 revised — transient BrainDecision, projection-before-state) → `brain.py` node +
`clarify.py` node + graph rewiring (retire
`understand_request`/`converse`, one-call conversation via `response_text`, deterministic
LLM-failure mapping, action→Plan projection) → `daemon/task_runtime.py` (D1: ResponseSink +
30s timer + timeout resume) + `DaemonServer` delegation + `_YES_WORDS` split (D2) + protocol
clarification messages + client/CLI free-text + voice clarification capture → ~28 FakeLLM
tests → rewrite `docs/02_ARCHITECTURE.md` → dated PROGRESS.md entry → ruff + pytest.
`src/jarvis/llm/` stays untouched (git baseline preserved).

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

## MULTI-PROVIDER FREE-ONLY LLM layer + `keys` CLI + fixed `init` bug (WSL run)

**Task** (user requirement list): robust FREE-ONLY multi-provider LLM system. Groq
(openai/gpt-oss-120b, openai/gpt-oss-20b) → OpenRouter (openrouter/free) → Gemini
(gemini-3.8-flash, gemini-3.7-flash) → NVIDIA (nvidia/nemotron-3-super-120b-a12b,
nvidia/nemotron-3.5-lightning-30b-a3b). Never paid models; only `pricing_mode=="free"` is
selectable, enforced in code (not prompt). No API keys in config.toml/logs/doctor/state/prompts.
Fixed the `jarvis init` "infinite Gemini prompt" bug.

**Root cause of the old init bug (do not re-open)**: `typer.prompt(..., hide_input=...)` re-prompts
forever on an empty Enter (it validates non-empty); Gemini was the message the human saw, but any
empty answer looped. Fix: one `getpass.getpass` prompt per provider, empty answer == skip. Each of
the 5 providers (groq/openrouter/gemini/nvidia/tavily) is prompted exactly once.

**What changed** (uncommitted, stage it as one conventional commit when ready):
- `llm/models.py` (new): the FREE-ONLY registry — 7 unique provider/model pairs, all
  `pricing_mode="free"`, with a `Capabilities` (tools / structured-output / vision) and
  `available` flag. `qualify()` is the single enforcement point: unregistered = `unknown` pricing
  → refused even with `free_only=False`; `paid` → refused under `free_only=True`; unavailable →
  refused. `model_spec()` returns unknown-spec for anything not in the registry (fail-closed on
  renamed/retired models).
- `llm/provider.py` (rewritten): factories per provider; `_EnvBackedKeyStore` = keyring first,
  then known env var(s) (`SECRET_ENV_VARS`; Gemini also reads `GOOGLE_API_KEY`), never copies
  env→keyring; `_guard_free` runs `qualify()` on planner+fast before any client is built;
  `build_provider_client` (public, used by doctor --live), `build_llm_client` → single client or
  `MultiProviderClient`; `provider_status` (doctor rows), `provider_has_credential` (boolean only);
  every skip reason is safe and human-readable, secrets never returned.
- `llm/multi.py` (new): composite over eligible free providers — per-op fallthrough on
  transient/auth/model/capability errors, registry-capability skipping for `structured()`, bounded
  by provider count, single safe `LLMError`.
- `llm/openai_compat.py` (new) + `llm/gemini.py` (new): httpx-native OpenRouter + NVIDIA (OpenAI
  chat/completions style) and Gemini (generateContent REST) clients; typed errors
  (Transient/Auth/Model/Capability), bounded retries (default `max_retries=1`), structured=JSON
  schema + Pydantic validation + one repair retry, never log keys/URLs. No openai/google SDK
  dependency.
- `llm/client.py`: error taxonomy + GroqClient uses `settings.llm.model_for(provider, role)` with
  explicit per-provider model overrides; old GeminiClient stub deleted.
- `config.py`: default TOML now `[llm] provider_order=[groq,openrouter,gemini,nvidia]`,
  `free_only=true`, `max_retries=1`, per-provider `[llm.models.<p>] planner/fast`; legacy
  `planner_model/fast_model` still honoured via `LLMSettings.model_for()`.
- `secrets.py`: `SECRET_NAMES` + openrouter/nvidia; `SECRET_ENV_VARS`; `PROVIDER_SECRETS`.
- `cli.py`: `init` one prompt/provider (regression-tested), summary block ("Configured providers:",
  "LLM free-only mode: enabled"); new `jarvis keys {set|paste|clear|status}` sub-app — hidden
  prompt, strips whitespace, rejects empty, clipboard import then best-effort clipboard clear
  (Windows history caveat documented); status prints only `<provider>: configured|not configured`
  (effective presence incl. env); doctor rewritten with per-provider `groq_api_key`/`groq_model`
  checks, `free_only`, aggregated `llm_provider` ("No free LLM provider configured" when none),
  `vision` WARN "not wired", and `--live` probes via `_live_provider_check` (rate limit → WARN,
  auth/model → FAIL, client always closed).

**Test status (WSL)**: `ruff check .` ALL CLEAN; **771 passed, 10 skipped, 1 deselected** on the
full suite minus `tests/windows_only` and the pre-existing WSL NTFS-ADS failure
(`test_paths.py::...alternate_data_streams...`). New tests: `test_llm_models.py` (registry +
qualify incl. injected paid/unavailable entries), `test_freeonly.py` (paid/unknown refused,
missing-key skip + next provider, auth + bad key falls through, rate-limit fallthrough bounded,
all-free-fail safe joined error, capability skip for `structured()`, no secret in logs (caplog),
exceptions, pydantic dump, selection repr, and end-to-end: secret absent from the raw SQLite
checkpoint bytes after a real `run_task`), `test_keys_cli.py` (set/paste/clear/status, empty
reject, unknown provider, env fallback incl. GOOGLE_API_KEY, value never echoed),
`test_init_flow.py` (exactly one prompt per provider, Enter skips, no 2nd round, key stored once,
config written once, ipc_token generation, never in config/messages), doctor --live fake-client
tests in `test_cli.py`, `test_live_optin.py` (opt-in, skipped by default).

**Opt-in live tests** (NOT part of normal run): `pytest tests/unit/test_live_optin.py` after
`set JARVIS_LIVE_TESTS=1` + real free keys. Sounds exactly one tiny text probe per configured
provider through the real factory; asserts reachable + `sk-` never in captured output.

**Manual smoke test for the human on their PC (native PowerShell)**:
1. `.\venv\Scripts\activate; jarvis init` → verify: exactly one prompt per provider, Enter skips
   (no loop), summary prints "LLM free-only mode: enabled".
2. Put a real free key on the clipboard; `jarvis keys paste groq` → "saved"; `jarvis keys status`
   → `groq: configured` and the value is never echoed.
3. `jarvis doctor` → per-provider lines, `llm_provider` PASS; then `jarvis doctor --live` with the
   Groq key set → `live_groq PASS ... reachable (free-only)`.
4. `jarvis chat` "hi" → answered via the first *configured* free provider; to force the fallback,
   set only a Gemini key and chat again.
5. `%LOCALAPPDATA%\jarvis\jarvis\logs\jarvis.jsonl` — no `gsk-`/`sk-`/`AIza-`/`nvapi-` values at
   INFO level.

**Known bugs / notes**: NVIDIA free model IDs are the current documented free endpoints
(`nemotron-3-super-120b-a12b`, `nemotron-3.5-lightning-30b-a3b`) — verify with `doctor --live`
(the registry owns availability). openrouter has ONE registered model so its "fast" falls back to
the planner. `keys status` will raise if the OS keyring backend is genuinely unavailable (WSL-only
here: `NoKeyringError`); on Windows with the keyring service this resolves. Vision caps are
declared but not wired (doctor says so). `mypy src/jarvis/policy` still blocked by the numpy
stubs/Python-3.14 WSL env (unchanged, policy untouched). NOT committed (waiting for review).

## (bottom-entry) `strict_zero_cost` rework — 4-tier pricing + safe default — DONE
Follow-up requirement set on top of the free-only layer. Key decision honoured without
weakening the policy engine: I did **not** make `free_tier` providers pass by adding a
"WARN with valid key" exception or by trusting any billable state from a registry —
instead I added a real 4-tier pricing model in code and made the *registry* own
pricing so the engine is deterministic, then exposed the only two legitimate switches
(`strict_zero_cost`, `free_only`) as config flags.

What changed (uncommitted — stage as one conventional commit when ready):
- `config.py`: `LLMSettings.strict_zero_cost: bool = True` (safe default);
  `free_only: bool = True` retained as the second, independent gate. `provider_order`
  default is now `["openrouter","nvidia","gemini","groq"]` and `strict_zero_cost=true`
  is written in the default config TOML + doctor.
- `llm/models.py` (registry + registry owns pricing): `PricingMode` = `zero_cost_endpoint`
  | `free_tier` | `paid` | `unknown` | `unavailable`. `model_spec()` returns the mode
  (registry-issued, deterministic); every registry entry now declares `pricing_mode`
  instead of a summary flag, plus `verified_billing` per provider.
- Reclassified per verified-current facts:
  - openrouter `openrouter/free` → `zero_cost_endpoint` (endpoint never bills you; token
    allowance self-issued — proven reachable in live doctor).
  - groq `openai/gpt-oss-120b` / `openai/gpt-oss-20b` → `free_tier` (billing state
    cannot be verified).
  - gemini `gemini-3.x-flash` → `free_tier` (billing cannot be verified).
  - nvidia `nemotron-3-super-120b-a12b` → `zero_cost_endpoint` (verified free limit).
    **nvidia `nemotron-3.5-lightning-30b-a3b` → `unavailable`** (the "lightning" free
    endpoint is not verifiable → marked unavailable, never selectable; requirement #3).
  - paid models → `paid`; any unregistered/unknown model → `unknown`; spec with a
    retired flag → `unavailable`.
- `llm/provider.py`: `qualify()` now takes `strict_zero_cost: bool`. Under
  `strict_zero_cost=true`: only `zero_cost_endpoint` and `zero_cost_endpoint`-verified
  models are selectable; `free_tier` → **refused** with `reason="billing state cannot be
  verified"` **even if the API key is present** (this is the "no surprise" sign). Under
  `strict_zero_cost=false`: `free_tier` becomes selectable but `paid`/`unknown`/
  `unavailable` are still refused, and `free_only=true` still gates. `provider_status`
  now reports `pricing_mode` per provider; `llm_provider` doctor row returns the joined
  free provider names (not "No free LLM provider configured").
- `cli.py` doctor: `strict_zero_cost` health PASS/FAIL row; `groq_model`/`gemini_model`
  rows become WARN under strict with billing-unverifiable detail; zero-cost rows PASS
  with "zero-cost endpoint" detail. Secrets still never logged/printed/serialised
  (asserted in the suite).
- Tests rewritten to match the strict-default world: `test_llm_models.py`
  (pricing-mode registry incl. paid/free/zero-cost/unknown/unavailable + strict
  qualification), `test_provider.py` (selection `/` strict refusals + env fallback +
  composite which now requires `strict_zero_cost=False` for free-tier builds),
  `test_freeonly.py` (multi-provider guarantee rework, secrets never leak, composite
  never includes paid, every-provider-fails → safe joined LLMError, provider keys never
  in serialised checkpoint DB), `test_cli.py` doctor sections, `test_config.py`
  (strict default + provider_order default), `test_llm_models.py`.

**Test status (WSL)**: 789 passed / 0 failed / 10 skipped, ruff clean, ruff format
clean. One deselected/known WSL-only NTFS-ADS paths test (passes only on real Windows),
matches the pre-existing exception in `AGENTS.md §10`. `scripts/check.ps1` to be run on
the dev PC; **NOT committed** (waiting for review). Manual smoke on PC: see the doctor
walkthrough in the LLM-driver entry; the only new bit is `jarvis doctor` output now
shows `strict_zero_cost PASS` and the `zero-cost`/`free-tier` distinction on provider rows.

---

## Session: quiet-start/re-arm diagnosis — first-wake regression follow-up (no commit)

**Status: NOT committed** (per instruction; no commit/push for this follow-up).

### Reported symptom vs. evidence
Reported: after `0a5fc47`, the FIRST "Hey Jarvis" after `jarvis.exe daemon --foreground`
produced no "Yes?"/response at all. Investigated exhaustively with live diagnostics.

### Root-cause verdict
- Static trace: the re-arm quiet-start gate (`_REARM_QUIET_GATE_S=0.5`,
  `_quiet_start_drain()`) runs ONLY in `_reset_voice_state()` / `_rearm_wake_for_confirmation()`
  — i.e. post-interaction and pre-confirmation. The startup path
  (`_run()` ? `audio.open()` ? `_set_state("LISTENING")` ? `wake_detector.reset()` ?
  `_wait_for_wake()`) contains NO gate, drain, or extra reset. Empirically confirmed: daemon
  logs show `WAKE_WAIT_START` immediately after the startup reset, with no `RESETTING`,
  no drain reads, `reset_calls==1` at startup.
- `git log` proves `wake.py`/`audio_input.py` were last touched by `0f5e608`/`dab61b8`,
  BEFORE the regression window (`86b8703`, `0a5fc47` touched only `loop.py` + `tts.py`).
  Detection sensitivity is therefore byte-identical to the build that historically woke 5/5
  at score 0.966.
- Live probes (dev PC, `.venv` 3.11): replay `clip_hey_jarvis.wav` through the CURRENT
  detector in 80 ms chunks ? score 0.998. Live mic ? detector (speaker-to-mic echo) in the
  real loop ? woke (0.647, full "Yes?" ? capture ? re-arm). Ran the REAL
  `jarvis daemon --foreground` and played the clip acoustically: marginal at speaker volume
  (max_score 0.4459, below the 0.5 threshold, NO wake); at increased amplitude the real
  daemon woke (0.6337) and the full chain ran: `wake_detected` ? SAPI "Yes?" (fresh engine,
  init 0.31 s + speech 1.53 s, NO hang) ? capture ? `RESETTING` ? `LISTENING`.
- Conclusion: NOT a code regression. The first-wake pipeline is intact end-to-end. The
  reported failure is an INPUT-AMPLITUDE margin: the detector is sensitive to the level of
  the phrase in the fed stream (0.45 at x1 playback vs 0.65–0.99 at x2/human volumes);
  the fixed threshold (0.5) was never changed. Root cause in the real world — mic
  level/AGC/room at launch time, not the loop, the detector, or the TTS. Hypoteses A/B/C/D
  from the task are all closed: (A) frames reach detector (frames_seen grows at 16 k/s,
  blocks_dropped=0), (B) scores correct and turn into wakes, (C) no startup suppression,
  (D) ack TTS spoke cleanly.

### Change made
- `tests/unit/test_voice_loop.py`: added `TestFirstWakeAfterQuietStartup` (2 assertions
  locked): after a quiet startup window the FIRST wake word is fed to the detector, drives
  a full interaction (`Yes?` ? command ? re-arm), and the detector is armed exactly once at
  startup (+1 re-arm), with no extra startup reset/drain. Fails if a future startup gate
  swallows or re-resets the first wake.

### Validation (all green)
- `pytest -q -m "not windows_only and not slow and not voice"`: 0 failed (only the
  pre-existing daemon async-mock coroutine warnings).
- `ruff check .` clean; `ruff format --check .` clean; `mypy src/jarvis/policy`: Success.

### Manual test for the user (real device)
Run daemon ? wait for "voice activated" ? say "Hey Jarvis": expect a "Yes?" ack and
response. A quieter-than-usual mic (laptop AGC) can sit below threshold — check Windows
mic volume/enhancements if it doesn't react; but faces-talking at normal volume matched
historical 0.9+ scores on this exact build.

## Session: idle timeout is non-fatal (voice loop reliability)

### What was actually wrong
The non-fatal idle timeout had already been implemented in `VoiceLoop._run`, but:

1. **The regression test was racy, not slow-by-design.** It polled a list that the
   wake-ack `"Yes?"` appended to *before* capture, so it asserted on a half-finished
   interaction and failed. On failure it never called `loop.stop()`, leaving a live
   voice thread cycling `READY/RECOVERING/RESETTING` for the rest of the session —
   that was the ~40 min "stuck" run, not a slow test.
2. **The idle path reused the post-interaction reset** (`_rearm()`: flush -> quiet
   drain -> `wake_detector.reset()`), so a wake that never happened spent the
   once-per-interaction re-arm and made "re-armed exactly once per interaction"
   unobservable.
3. **~20 existing tests used the idle timeout as the loop's terminator** (`join(10)`
   + `assert not is_active()`). With a non-fatal timeout they can never pass: a fake
   mic that runs out of script keeps listening, exactly as a real one does.

### Change made
- `src/jarvis/voice/loop.py`
  - `_resume_after_idle()` replaces `_rearm()` on the idle path: reports
    `RECOVERING` -> `READY` and re-enters the wait. No flush, no quiet drain, no
    detector reset (nothing was spoken, the model was scoring silence).
  - New `idle_backoff_s` ctor knob (default unchanged at 0.5 s) so tests can shrink
    the post-timeout pause without changing production behaviour.
  - `_empty_read_backoff()`: a *zero-sample* wake read (stalled/half-closed device)
    now yields 20 ms instead of hot-spinning; with `idle_timeout_s` disabled nothing
    else bounded that loop.
- `tests/unit/test_voice_loop.py`
  - Helpers `_wait_for` / `_stop_loop` / `_run_scripted` + `_PhaseSpy` reporter:
    lifecycle phases and recoveries are asserted directly, every wait is bounded
    (5 s), and `stop()` always runs in a `finally`, so a regression fails in seconds
    and never leaks a spinning voice thread.
  - `test_idle_timeout_is_non_fatal` rewritten: first wait forced to time out, the
    wake fires only after that timeout was observed, one full interaction runs
    (ack -> capture -> STT -> agent -> answer), the loop returns to `READY`, survives
    a further idle timeout, exits only on `stop()` with reason `None`, no illegal
    phase transitions, `reset_calls == 2`.
  - Converted the ~20 `join()`-terminated tests to "wait for the effect, then
    `stop()`" (the pattern the rest of the file already used).
  - `TestVoiceLevelDiagnostics.test_wake_heartbeat_includes_audio_level` /
    `test_wake_heartbeat_is_rate_limited` and the other idle-heavy tests pass
    `idle_backoff_s=0.01`.
  - `TestVoiceClarificationCapture.test_empty_utterance_fails_closed` used the default
    30 s command window: now 0.2 s (the wait-for-speech behaviour is covered by
    `test_no_speech_capture_waits_for_speech_then_gives_up`). 30 s -> 0.2 s.

### Validation
- `pytest tests/unit/test_voice_loop.py` -> 78 passed (~2 s total, was minutes).
- `pytest tests/unit/test_voice_status.py` -> 36 passed.
- `pytest tests/unit/test_daemon_voice.py tests/unit/test_voice_wiring.py
  tests/unit/test_tools_dictation.py tests/unit/test_invariants.py` -> 92 passed, 1 skipped.
- `test_idle_timeout_is_non_fatal` alone: 0.09 s, run 5x, stable.
- `ruff check` / `ruff format --check` clean; `mypy src/jarvis/policy`: Success.

### Note for review
`tests/unit/test_voice_loop.py` is stored with CRLF line endings (unstaged diff is a
whole-file line-ending change against the index). Left as-is; normalise with
`git add --renormalize` only if the repo wants LF.


---

## Session: Phase 8 memory + `jarvis run` (one-shot command)

### Where the roadmap actually stood
`docs/05_BUILD_PLAN.md` Phase 8 (`src/jarvis/memory/*`) was still five one-line
stubs and `src/jarvis/ml/__init__.py` is still a stub, so memory was the next real
gap. `jarvis run "<command>"` (in the `docs/01_PROJECT_SPEC.md` CLI table) did not
exist at all. Both are now done except the classifier (see "Not done").

### Phase 8: memory (local, offline, no model downloads)
- `memory/db.py` - SQLite schema (`skills`, `failures`, `preferences`, `task_log`),
  WAL, `PRAGMA user_version`, bounded pruning, and the shared secret helpers
  (`secret_key`, `looks_secret`, `redact_secrets`, `safe_value`, `safe_plan`,
  `secret_free_rows`).
- `memory/embeddings.py` - deterministic 256-dim BLAKE2b lexical embeddings
  (cosine similarity). Chosen over sentence-transformers so retrieval works with
  no network and no 90 MB model; the interface is swappable later.
- `memory/skills.py` - only *verified* plans are stored; near-duplicate goals fold
  into the existing row and bump `success_count`; `trusted = fail_count <= success_count`.
- `memory/failures.py`, `memory/prefs.py` - failure history (feeds replan) and
  explicit, inspectable preferences (never inferred).
- `memory/store.py` - thread-safe `SqliteMemory`; `open_memory()` **never raises** -
  it returns `NullMemory` when memory is disabled, in `dry_run`, or if the file
  cannot be opened.
- `agent/nodes/memory_save.py` + `graph.py` - `respond -> memory_save -> END`.
  Only `ok and verified and not tainted` steps become skills; failed / unverified /
  tainted steps become failure records; a cancelled or halted task writes nothing.
  Every write is fail-soft - memory never breaks a task.
- Wiring: in-process `chat --no-daemon` and the daemon open memory lazily and close
  it in `finally`; `[memory]` settings added to config (`enabled`, `save_skills`,
  `similarity_threshold`, row caps).
- `jarvis skills list|show|delete|clear` - the user can inspect and delete anything
  JARVIS learned. `skills clear` also wipes failures and preferences.

### Audit self-review fix (Phase 7 acceptance)
`jarvis audit` and `audit_last_session` now pass `log_path=config.log_file()`. Without
it the self-review step compared against nothing and the Phase 7 criterion "audit can
review its own last session" was not actually met.

### `jarvis run "<command>"`
- Default path talks to a running daemon over IPC; `--no-daemon` runs in-process.
  Both share the *same* code as `chat`: `resolve_interrupts()` (extracted from
  `chat_loop`) answers clarification and confirmation prompts, and
  `_daemon_one_shot()` (extracted from `_daemon_chat_loop`) handles the daemon's
  `ConfirmRequest` / `ClarificationRequest` / `FinalMessage` stream.
  **There is no `--yes` and no way to skip a prompt**: an approval is bound to the
  same `action_hash` and the same typed-confirmation / password rules as the REPL,
  so the one-shot path can never be more permissive than the interactive one.
- `--dry-run` is accepted **only** with `--no-daemon`; with the daemon it exits 2 and
  says so. The daemon owns one long-lived `AppContext` shared by all its tasks, so
  honouring a per-request flag would mean mutating shared state between tasks; it
  refuses rather than silently running for real.
- Exit codes: 0 answered, 1 provider missing / daemon unreachable / task error,
  2 bad usage.

### Security fixes found while reviewing the memory code
1. `FailureStore.record` stored tool error text verbatim (only truncated). A tool
   that echoes a credential back ("login failed ... password=hunter2") would have
   persisted it. Now redacted via the shared `redact_secrets()` before the row is
   written; the useful part of the message is kept, only the value is dropped.
2. `jarvis skills show/delete <unknown id>` raised `typer.Exit` before the store was
   closed, leaking the SQLite handle. All four skills commands now go through
   `_memory_store()`, a context manager that closes on every path.
3. `_close_memory_backend(ctx)` takes a *context* and reads `ctx.memory`; passing a
   store to it silently did nothing. Split into `_close_memory_backend(ctx)` and
   `_close_store(memory)` so both call sites are honest.
4. With `[memory] enabled = false` the CLI got a `NullMemory` backend with no
   `.skills` attribute and crashed with an `AttributeError`. It now prints how to
   enable memory and exits 1.

### Validation
- `pytest tests/unit/test_run_cli.py` -> 11 passed (new).
- `pytest tests/unit/test_skills_cli.py` -> 17 passed (3 new: disabled-memory paths,
  close-on-every-exit-path, per-command connection).
- `pytest tests/unit/test_memory.py` -> 54 passed (4 new: error redaction, nested
  step redaction, `redact_secrets` prose/idempotence).
- Affected regression set (`test_memory*`, `test_skills_cli`, `test_run_cli`,
  `test_agent`, `test_replan`, `test_invariants`) and
  (`test_cli`, `test_daemon_server`, `test_daemon_client`, `test_voice_loop`) -> exit 0.
- `ruff check src/jarvis tests/unit` clean; `ruff format --check` clean on every file
  I touched; `mypy src/jarvis/policy`: Success. Full suite not run (by instruction).

### Known limitation
`jarvis run --dry-run` requires `--no-daemon`. A per-request dry-run over IPC needs
the daemon to build a per-task `AppContext` (or a `dry_run` field on `ChatMessage`
plus a derived context), which is a Phase 10 hardening item, not a one-line flag.

### Not done (deliberately)
- `src/jarvis/ml/__init__.py` risk classifier: still a stub. A trained ONNX model and
  labelled in-session data do not exist, and nothing in the v1 policy path requires
  it - the deterministic `policy/engine.py` is authoritative. Do not ship a fake
  classifier for the sake of the module existing.
- A global `jarvis --dry-run` flag. Only `jarvis run --dry-run` exists today.
- `jarvis undo` (the undo log is written by the delete path; there is no command to
  surface it yet).
