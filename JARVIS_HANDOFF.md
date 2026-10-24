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
| R3 | Session context and reference resolution | IN PROGRESS |
| R4 | Browser results, refs, primitives, site catalogue | IN PROGRESS |
| R4 | Browser results, refs, primitives, site catalogue | PENDING |
| R5 | Contacts and WhatsApp matching | PENDING |
| R6 | OCR and screen perception | PENDING |
| R7 | Voice vs terminal response path | PENDING |
| R8 | Logs, task log, failure evidence | PENDING |
| R9 | State cleanup and read/write map | PENDING |
| R10 | Benchmarks and test impact | PENDING |
| R11 | Environment and dependency versions | PENDING |

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
4. **Browser**: Managed browser + UIA; page referent cleared by unrelated actions; P2/P3 added but live verification partial.
5. **WhatsApp**: Isolated Web profile, content-hash binding, recipient precheck.
6. **Tools/registry**: 39 tools, closed registry, no generic shell/exec.
7. **Stage 8 incomplete**: Camera/WhatsApp/referents live checks not fully verified.
8. **OCR**: Not installed, accuracy unvalidated.
9. **Memory**: SQLite stores skills/failures/preferences/task_log with caps.
10. **Daemon**: IPC 127.0.0.1 + token, confirmations bound to action hash.

## 6. Next step

Continue with R2

R2 is now complete; the next audit part is R3.
