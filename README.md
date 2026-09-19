# JARVIS — Windows Desktop AI Agent (Final Year Project)

JARVIS runs in the background on your Windows PC, listens for "Hey Jarvis" (or takes typed commands
in a terminal), plans tasks with an LLM, and executes them through a **safe, verified tool layer**.

```
pipx install jarvis-agent   # (or: pip install -e . during development)
jarvis init                 # keys, password, contacts, autostart
jarvis on                   # start listening
jarvis chat                 # or type commands in a terminal
```

## What it can do (v1)

Open apps and websites · Google search · answer current questions with sources · create/delete files
(Recycle Bin, with undo) · dictate into Notepad by voice · send a single WhatsApp message (with
confirmation) · audit the PC for security and performance problems · lock the PC · remember successful
plans and get faster over time.

## What makes it a strong final-year project

1. **Deterministic policy engine separated from the LLM**: prompt-injected content cannot authorise actions.
2. **Verification + replan loop**: every step is checked; failures trigger retry or re-plan.
3. **Skill memory**: successful plans are stored and reused as examples, reducing API calls over time.
4. **Trained risk classifier** (small DistilBERT/MiniLM model, ONNX, CPU) as an extra safety layer.
5. **Real evaluation**: 50-task benchmark, ablations, red-team prompt-injection suite, numbers in the report.

## Documentation map

| File | Contents |
|------|----------|
| `AGENTS.md` | Rules OpenCode reads every session |
| `PROGRESS.md` | Living status file |
| `docs/01_PROJECT_SPEC.md` | Product spec and acceptance criteria |
| `docs/02_ARCHITECTURE.md` | Architecture, state schema, protocol, storage |
| `docs/03_SECURITY_AND_POLICY.md` | Tiers, threat model, invariants |
| `docs/04_TOOLS_SPEC.md` | Tool-by-tool specification |
| `docs/05_BUILD_PLAN.md` | Phases 0-9 |
| `docs/06_OPENCODE_PROMPTS.md` | Copy-paste prompts for each phase |
| `docs/07_TESTING_AND_BENCHMARK.md` | Tests, benchmark, ablation |
| `docs/08_RISK_CLASSIFIER.md` | The "trained model" part |
| `docs/09_SETUP_AND_DEPENDENCIES.md` | Environment, dependencies, config |
| `docs/10_REPORT_AND_VIVA.md` | Report outline, viva Q&A, demo script |
| `docs/11_DEPLOYMENT_AND_PACKAGING.md` | Wheel + pipx install, autostart, upgrades, uninstall, clean-machine test |
| `docs/12_DEPENDENCY_VERIFICATION.md` | Verified findings, fallback matrix, licences, freeze policy |
| `scripts/verify_env.py` | Run on your PC to prove every dependency works (`--all --live`) |

## How to start (5 steps)

1. Install Python 3.11/3.12 (64-bit), Git, VS Code, OpenCode. Create the project folder and `git init`.
2. Copy this whole folder's contents into the project root (keep the `docs/` folder).
3. Follow `docs/09_SETUP_AND_DEPENDENCIES.md` to create the venv and get Groq/Gemini keys.
4. Open the folder in VS Code, run `opencode` in the terminal. It reads `AGENTS.md` automatically.
5. Paste the **Phase 0 prompt** from `docs/06_OPENCODE_PROMPTS.md`. Review, test, commit. Then Phase 1, and so on.

**Golden rule:** one phase per session, tests green, commit, then continue. Understand every file: you will be asked about it in the viva.
