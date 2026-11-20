# JARVIS P3 Referent Continuity Implementation Report

Date: 2026-11-20
Branch: `main`
Status: implemented and unit-tested; no live browser/desktop run performed.

## Milestone selection

The repository already contains mature routing, memory, task observability, voice and
WhatsApp subsystems. The comprehensive status report identifies browser page-referent
invalidation as an unfinished Stage-8/P3 acceptance item, and the focused test
`tests/unit/test_browser_referents.py` had two reproducible failures at HEAD:

- a time query after navigation left the page referent `resolved`;
- "open that page" after the time query silently injected the stale URL into the brain prompt.

That is a real multi-step reliability and context-safety bug, not a documentation gap,
so P3 referent continuity was selected and implemented end-to-end.

## What changed

### Single source of truth

- Removed the process-global `REFERENT_STORE` from `jarvis.agent.referents`.
- Added a per-`AppContext` `ReferentStore`, passed through `make_app_context`.
- `context_resolve` now reads `ctx.referents`, so the deterministic `resolved_refs`
  block used by the brain and fast paths reflects the same store the act node writes.
- Removed the duplicate `_note_navigation` helper and all nine call sites from
  `jarvis.browser.tools`. That helper wrote to a process-global store under a bogus
  key derived from `id(ctx)` and had swallowed all errors. The managed browser tools
  still publish `page_url` in `ToolResult.data`; `act` is now the only referent writer.

### Invalidation and refresh rules

- `SessionContext.note_action` gained `preserve_page`. Any successful non-browser
  action closes the artifact run and invalidates the remembered page; a browser step
  preserves it because it remains on the observed page.
- `act` publishes trusted artifacts and page/result/file refs through small helpers:
  - browser results with a valid `page_url` refresh the page in both stores;
  - navigating to a different page clears the old browser result set;
  - `browser_find` result sets, photos and files keep their existing binding rules;
  - failed and dry-run steps publish nothing and invalidate nothing;
  - unrelated successful steps clear the page in both stores.
- Page URLs continue to be validated by `SessionContext.note_page` and rendered only
  inside `<untrusted_data>`. No URL, prompt, tool argument or tool output is logged or
  persisted to state/checkpoints/task events.

### Incidental browser fixes required by this path

Two latent undefined-name bugs were present in the same browser tool module and would
break the multi-step path at runtime:

- `browser_open_result` passed an undefined `expect_identity`; it now passes the
  already-computed approved target identity from `ctx.approved_target_identity`.
- `_resolve_browser_press_key` called bare `urlsplit`; it now calls
  `urllib.parse.urlsplit`, matching the module's existing import.

## Files created/modified

Created:

- `docs/JARVIS_P3_REFERENT_CONTINUITY_IMPLEMENTATION_REPORT.md`

Modified:

- `src/jarvis/agent/referents.py`
- `src/jarvis/agent/context.py`
- `src/jarvis/agent/nodes/context_resolve.py`
- `src/jarvis/agent/session_context.py`
- `src/jarvis/agent/nodes/act.py`
- `src/jarvis/browser/tools.py`
- `tests/unit/test_browser_referents.py`
- `tests/unit/test_session_context.py`
- `tests/unit/test_referents.py`
- `PROGRESS.md`

## Verification

Commands executed with `.venv\Scripts\python.exe` in the project root:

| Command | Result |
|---|---|
| `pytest -o addopts="" -q tests/unit/test_browser_referents.py tests/unit/test_session_context.py tests/unit/test_referents.py tests/unit/test_browser_tools.py tests/unit/test_browser_context.py tests/unit/test_b9_context_discovery.py tests/unit/test_agent.py tests/unit/test_brain_prompt.py tests/unit/test_invariants.py tests/unit/test_fastpath.py` | 269 passed |
| `pytest -o addopts="" -q tests/unit/test_referents.py tests/unit/test_browser_referents.py tests/unit/test_session_context.py` | 53 passed |
| `ruff format --check` on eight changed/new Python files | 8 files already formatted |
| `ruff check` on all nine changed Python files | 9 remaining findings, all pre-existing style findings (`BLE001`, `SIM102`, `PLR1716`) in unchanged logic; no new import/undefined-name findings |

A broader browser sweep of 11 files produced 275 passed and 6 failed. The failures are
in `test_browser_find.py::test_the_new_tools_never_expose_a_press_or_hotkey_name` and
five `test_browser_sites.py` cases that expect no default configured sites; those
modules and the registry/site-loading code were not changed by this milestone. The
milestone-focused suite above passes in full.

## Limitations

- All evidence is unit-level with fakes. No live Playwright browser, Windows desktop,
  WhatsApp session, microphone or LLM provider was exercised.
- Referents remain in-memory only; a daemon restart still forgets them by design.
- `SessionContext` still uses the shared `SESSION_THREAD = "local"` key while
  `ReferentStore` uses `conversation_id`; this pre-existing split was not changed.
- The two pre-existing browser test failures above require separate cleanup of stale
  expectations/configuration and were deliberately left untouched.

## Live checks still required on the dev PC

1. `jarvis daemon --foreground`, then ask to open a known site.
2. Immediately ask "open that page again" and confirm the URL is reused.
3. Ask the time, then repeat "open that page again" and confirm JARVIS asks which page.
4. Run `jarvis tasks --task <id>` and confirm no URL appears in the event timeline.
