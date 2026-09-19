# 10 — Report, Viva and Demo

## 1. Report outline (typical final-year structure)

1. **Abstract** (150–200 words): problem, approach, main results (with numbers).
2. **Introduction**: motivation (PC automation, LLM agents), problem statement, objectives, scope, contributions (C1–C4 from `01_PROJECT_SPEC.md`).
3. **Literature review**: LLM agents and tool use (ReAct, function calling), computer-use agents, agent safety and prompt injection, LangGraph/state machines, speech pipelines (wake word, Whisper), Windows UI automation, risk classification. Cite 15–25 sources; compare with existing assistants (Copilot, open-source agents).
4. **Requirements**: functional (F1–F10), non-functional, threat model.
5. **System design**: architecture diagram, daemon + IPC, LangGraph graph, policy engine, tool layer, memory, voice pipeline, data model. Justify design choices and rejected alternatives (Windows Service vs Task Scheduler, plan-once vs replan, LLM-judged safety vs deterministic policy, coordinate clicking vs accessibility APIs).
6. **Implementation**: key modules with short code excerpts, algorithms (path resolution, decision function, confirmation binding, skill retrieval), challenges and how they were solved.
7. **Machine-learning component**: dataset, training, evaluation, quantisation, integration (`08_RISK_CLASSIFIER.md`).
8. **Testing and evaluation**: unit/integration test strategy and coverage; benchmark design; ablation results; red-team results; performance (latency/RAM); voice accuracy; statistical treatment (means, CIs).
9. **Results and discussion**: what worked, what failed (be honest), threats to validity (synthetic data, single machine, LLM variability, small sample).
10. **Limitations and future work**: lock-screen unlock and why it's excluded, tool builder, vision fallback, MCP, multilingual voice, multi-user.
11. **Conclusion**.
12. **References**, **Appendices**: full tool table, prompt templates, benchmark task list, red-team cases, installation guide, user manual.

**Figures/tables you should produce:** architecture diagram; LangGraph diagram (export from `graph.get_graph().draw_mermaid()`); tier table; sequence diagram of a Tier 2 confirmation; ablation bar chart; memory learning curve; classifier confusion matrix; latency/RAM chart; red-team results table.

**Numbers to copy from generated output only** (never hand-edit): success rates, CIs, API calls/task, latency, RAM, classifier metrics, unsafe-action rate.

## 2. Demo script (5–7 minutes; rehearse twice; have typed fallbacks)

| # | Action | What to show / say |
|---|--------|--------------------|
| 1 | Restart or show the PC already logged in | Daemon autostarted (`jarvis status`), listening indicator visible |
| 2 | "Hey Jarvis, what's the latest news on <topic>?" | Answer with source links; mention untrusted-content handling |
| 3 | "Hey Jarvis, take notes" → dictate two sentences → "save as demo notes" | Notepad dictation, focus verification, Tier 1 confirmation |
| 4 | Create 3 test files, then "delete the test files" | Preview list, password prompt (terminal), Recycle Bin, undo log; show `jarvis undo` |
| 5 | "Text Rahul I'll be late" (to your test contact) | Masked number confirmation, verification-before-send, refusal for unknown contact |
| 6 | Show a prompt-injection attempt (prepared search result / file) | Injection ignored; sensitive step shows untrusted-content banner; Tier 3 request refused |
| 7 | `jarvis audit` | Generated Markdown report highlights |
| 8 | Show benchmark charts + red-team table | Success rate, API calls, unsafe actions = 0, memory learning curve |
| 9 | Wrap up | Limitations and future work |

Fallbacks: if the mic fails use `jarvis chat`; if the network fails show a recorded video or use dry-run; if WhatsApp misbehaves show the recorded clip and the fail-closed test.

## 3. Likely viva questions and answer outlines

1. **Why an agent architecture instead of a script?** Tasks are open-ended; the LLM maps language to tool calls, while structure (validation, policy, verification) gives reliability. LangGraph gives explicit state, checkpointing, and human-in-the-loop interrupts.
2. **How do you stop the LLM doing something dangerous?** Separation of proposing and deciding: the deterministic policy engine, tiers, blocked actions with no tool available, path resolution, confirmations bound to an action hash, Recycle Bin, undo log. Show invariant tests and red-team results.
3. **What about prompt injection?** Untrusted data is delimited and flagged, taint propagates to steps, the engine ignores LLM claims, and sensitive steps require user confirmation with a warning. Report measured unsafe-action rate (0) vs the LLM-self-policing baseline.
4. **Why not let it unlock Windows with a password?** The lock screen runs on a secure desktop that normal apps can't drive; the alternative (Credential Provider) is system-level, risky, and needs the Windows password stored. We provide a separate Argon2id JARVIS password for Tier 2 and rely on Windows Hello/Dynamic Lock.
5. **Why Task Scheduler and not a Windows Service?** Services run in session 0 and can't interact with the user's desktop, windows, audio, or input.
6. **Why Argon2id?** Memory-hard, resistant to GPU cracking; salted; parameters tunable; `check_needs_rehash` supports upgrades.
7. **How do you know it works "perfectly"?** You don't claim perfection: you measure. 50-task benchmark, repeats, CIs, ablations; failure analysis.
8. **What did you train?** Risk classifier: data generation, review, splits, model choice, metrics per class, quantisation, integration as escalate-only. Explain why recall on risky classes matters and the baseline comparison.
9. **How does memory/learning work? Isn't it unsafe?** Skills are examples for the planner, retrieved by embedding similarity; never auto-executed; policy still applies; only verified, non-tainted plans saved; failures recorded to avoid repetition.
10. **What are the failure modes of verification?** UI text not readable, timing, false positives of "verified"; fail-closed for sensitive steps (WhatsApp); measured verification accuracy.
11. **How is privacy handled?** Audio local; only text to LLM APIs; secrets in Credential Manager; redacted logs; hash-only VirusTotal.
12. **How does the IPC stay secure?** Localhost only, random port, token from Credential Manager, constant-time comparison, throttling, message limits.
13. **How would you scale or extend it?** New tool checklist, MCP layer, vision fallback, multilingual STT, tool builder with sandbox.
14. **Limitations?** LLM variability and rate limits, Windows UI fragility (apps change), synthetic classifier data, single-machine evaluation, WhatsApp desktop automation fragility.
15. **What would you do differently?** Be honest and specific (e.g., collect real usage data earlier, add UI-automation abstractions sooner).

## 4. Pre-submission checklist

- [ ] Code frozen and tagged; README install steps tested on a clean machine.
- [ ] Report numbers match generated CSV summaries; figures readable.
- [ ] All invariant tests and the red-team suite pass; results archived.
- [ ] You can explain every file: use the "Explain for viva" prompt in `06_OPENCODE_PROMPTS.md` for each module.
- [ ] Demo video recorded as backup; slides ready; API keys removed from any screen recording.
- [ ] Repository has no secrets (`git log -p | Select-String "gsk_"` style check with your key prefixes).
