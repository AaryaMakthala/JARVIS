# PROGRESS.md

Maintained by the coding agent. Condensed from historical log.

## Current project state

- **Agent graph**: LangGraph pipeline with brain/replan/validate/policy_gate/act/verify/respond/memory nodes; checkpoints SQLite; state typed (Pydantic).
- **Policy**: Deterministic tier engine, path resolution/symlink checks, confirmations bound to action hash, Tier 3 blocked in code, deletes go to Recycle Bin.
- **Tools**: 39 registered tools in closed registry; browser/WhatsApp/screen/UIA/camera/dictation present.
- **Daemon**: IPC on 127.0.0.1 with token, task slots, confirmations, autostart support.
- **Voice**: Wake word/stt/tts present; barge-in supports bare 'stop' per D22 (overrides old Stage 1 note).
- **Browser**: Managed Playwright browser + UIA tools; session referents exist but page referent cleared by unrelated actions (P3 added).
- **WhatsApp**: WhatsApp Web via isolated profile; content-hash binding, recipient precheck, rate limiting.
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
- **Browser referents**: Page referent cleared by unrelated actions; cross-thread isolation exists in logic, live verification partial (P2/P3 NOT VERIFIED in places).
- **OCR**: Local OCR dependency not installed; OCR accuracy unvalidated.
- **Gemini/strict_zero_cost**: Under strict default Groq/Gemini would be refused; current config has strict_zero_cost=false.
- **Context windows**: Not registered numerically; overflow handled reactively from provider errors.
- **PROGRESS/roadmap naming**: Old roadmap filenames had doubled .md.md extensions historically (unverified for all links).

## Test and benchmark status

- **Unit/integration tests**: Extensive suite exists (140 files noted earlier). Last automated check for browser/referents: 118 passed, 0 failed (2026-10-23 context).
- **Lint/format**: Ruff/mypy used; last recorded ruff check passed for affected files.
- **Benchmark**: Harness in benchmarks/ with tasks.yaml; full benchmark runs not recorded as complete.

## Chronological milestones (condensed)

- **Phases 0-6**: Core graph/policy/tools/daemon/voice/screen implemented and tested.
- **Stage 7**: Browser DOM+UIA implemented (code complete; some live checks noted).
- **Stage 8**: Camera/WhatsApp/referents work in progress; not complete.
- **D22**: Bare 'stop' barge-in enabled (voice).
- **P3**: Browser page referent invalidation added (2026-10-23).
- **L4**: Enter now synchronizes on Playwright navigation; click rechecks the post-action origin; unauthorized cross-origin actions set the redirect block. Narrow fixture checks pass (2026-10-26).
- **L5**: Redirect-block enforcement now covers reads/actions/side effects (`resolve`, `current_url`, `scroll`, `wait`, `back`, `forward`, tabs, screenshot, WhatsApp accessor); screenshot revalidates its origin before capture. Targeted browser-context tests pass; live checks not run (2026-10-26).
- **5B**: Preferred-name capture hardened against quoted/reported/hypothetical/negated text, contextual/time phrases, three-word truncation, command-shaped suffixes, and punctuation-separated continuations; imperative statements require every candidate word to carry name-shape evidence, and an ambiguous candidate is rejected whole rather than trimmed to a prefix. Overlapping statements resolve by sentence position. Compose tests exercise the real preference-read path, and `memory forget-pref <key>` distinguishes invalid keys/storage errors from a missing preference. A fail-closed guard now prevents name learning and preference writes when the turn is classified `skipped_sensitive`; regression coverage added. Tests updated but not run in this pass (2026-10-26).
