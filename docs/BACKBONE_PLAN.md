# JARVIS backbone plan (prompt -> LLM -> plan -> run -> answer)

## RULES FOR EVERY STEP
1. Implementation only. Do NOT write tests and do NOT run pytest. Allowed
   checks: `ruff check <changed files>`, `python -m py_compile <changed
   files>` (use .venv\Scripts\python.exe), and read-only scripts that make no
   LLM or network call. Never run the app, browser, camera, voice.
2. Do not edit or stage: src/jarvis/agent/nodes/act.py,
   src/jarvis/agent/nodes/policy_gate.py,
   tests/unit/test_browse_grant_wiring.py (uncommitted WIP). If a step needs
   them, STOP and report.
3. Never change: PolicyEngine, tiers, Tier 3 blocking, target binding,
   llm/models.py qualification, free_only, strict_zero_cost.
4. The LLM only proposes. Tiers and policy decisions never enter a prompt.
5. Changes are additive: new config keys get safe defaults and an enable
   flag. New AgentState keys must be plain str/int/bool/list/dict (the
   checkpoint serializer allowlist stays unchanged).
6. One step at a time. Use the edit tools, no Python string-replace
   one-liners. Show the full diff, then stop for review.
7. Commit per step: git add <named files only>, never -A. Verify
   git config identity first (Aarya Makthala / aaryamakthala@gmail.com).
   No trailers, no co-author lines, no push.
8. After each step add a line to JARVIS_HANDOFF.md section "Backbone":
   step id, commit hash, status UNTESTED. Never write "tests pass".
9. A step is done only when you have shown: diff, ruff output, py_compile
   output, commit hash. Treat any claim without output as unverified.

## STEPS
A1 node trace (observability, no state key)
A2 route contract (one table: LLM request_type -> route)
A4 LLM overall deadline + cancel between attempts
A3a measure prompt size, propose tool groups (no edits)
A3b implement approved tool groups + prompt trimming
A5 compose node (spoken/screen answers, butler templates)
A6 state-contract doc + handoff update
Then phase T (golden-path tests S1-S10), then memory, then browser.
