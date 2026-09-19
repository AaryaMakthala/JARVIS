# 09 — Setup and Dependencies

## 1. Prerequisites (about 1 hour)

| Item | Notes |
|------|-------|
| Windows 10/11, 64-bit | Primary target |
| Python **3.11 or 3.12**, 64-bit | Tick "Add to PATH". Some audio/ML wheels lag behind the newest Python; avoid 3.13+ until every dependency installs cleanly |
| Git | `git init` in the project folder |
| VS Code | + Python extension |
| OpenCode | Install per its current docs; log in / configure a provider for the coding agent (separate from JARVIS's own Groq key) |
| Microphone + speakers | For Phases 5+ |
| WhatsApp Desktop | For Phase 6 (Microsoft Store version), signed in |
| Google Chrome or Edge | Default browser |
| Optional: winget | Present on modern Windows; used by the audit |

## 2. Create the project

```powershell
mkdir jarvis ; cd jarvis
git init
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```
Copy `AGENTS.md`, `README.md`, `PROGRESS.md` and the `docs/` folder into this directory, then:
```powershell
git add . ; git commit -m "docs: project spec and agent instructions"
opencode
```

If PowerShell blocks activation: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` (only for your user).

## 3. API keys (free tiers)

1. **Groq**: create a key at `console.groq.com`. Used for planning/answering.
2. **Google AI Studio (Gemini)**: create a key at `aistudio.google.com`. Optional fallback and later vision use.
3. **Tavily** (optional): free tier for web search; DuckDuckGo (`ddgs`) works without a key.
4. **VirusTotal** (optional): free API key; hash lookups only.

**Never** put keys in code, `.env` committed to Git, chat prompts, or `PROGRESS.md`. `jarvis init` stores them in Windows Credential Manager via `keyring`.
`.env.example` documents variable names for CI/dev only (`JARVIS_LLM__PLANNER_MODEL=...`); the real `.env` is git-ignored.

## 4. Model names — verify, don't assume

Provider model names change and old ones are retired (the earlier "Gemini 1.5 Flash" plan is an example of this).
Before Phase 1:
1. Open Groq's current models list (`console.groq.com/docs/models`) and choose: a stronger model for **planning** (structured output support matters) and a fast, cheap one for **short answers**.
2. Open Google AI Studio's models page and choose a current free-tier **Flash-class** model for the optional vision/fallback role.
3. Put the names in `config.toml` (`[llm]`), and record the choice + date in `PROGRESS.md` (Decisions log).
4. Check each model's structured-output/JSON-mode support and rate limits. JSON-schema output is supported only on certain Groq models and others return an HTTP 400, so the client must fall back to plain JSON + validation. Free tiers are limited; handle 429 with backoff.
5. After storing your key, `python scripts/verify_env.py --all --live` lists the models your key can actually use.

## 5. Dependencies (`pyproject.toml`)

Do not copy version numbers from memory: let `pip` resolve the newest compatible ones, run the tests, then pin (`pip freeze > requirements.lock` or use `uv`/`pip-tools`).

```toml
[project]
name = "jarvis-agent"
version = "0.1.0"
requires-python = ">=3.11,<3.13"
dependencies = [
  # core
  "pydantic>=2", "pydantic-settings", "platformdirs", "tomli-w", "pyyaml",
  "typer", "rich", "httpx",
  # agent
  "langgraph", "langgraph-checkpoint-sqlite", "langchain-core",
  "groq",                       # or langchain-groq if you prefer LangChain wrappers
  # security / os
  "keyring", "argon2-cffi", "send2trash", "psutil",
  "pywin32; sys_platform == 'win32'", "pywinauto; sys_platform == 'win32'",
  "pyautogui", "pyperclip",
  # web + memory
  "ddgs", "numpy", "fastembed",
]

[project.optional-dependencies]
gemini  = ["langchain-google-genai"]            # or google-genai; verify the current SDK name
voice   = ["sounddevice", "openwakeword", "faster-whisper", "piper-tts", "pyttsx3"]
browser = ["playwright"]                        # then: playwright install chromium
ml      = ["onnxruntime", "tokenizers", "optimum"]  # inference needs only onnxruntime + tokenizers
dev     = ["pytest", "pytest-asyncio", "pytest-cov", "hypothesis", "ruff", "mypy", "types-PyYAML"]

[project.scripts]
jarvis = "jarvis.cli:app"

[tool.pytest.ini_options]
markers = ["windows_only", "voice", "slow"]
addopts = "-q"

[tool.ruff]
line-length = 100
[tool.mypy]
python_version = "3.11"
[[tool.mypy.overrides]]
module = "jarvis.policy.*"
strict = true
```

Install by extra as you reach each phase:
```powershell
pip install -e ".[dev]"            # Phase 0-4
pip install -e ".[voice]"          # Phase 5
pip install -e ".[browser]" ; playwright install chromium   # Phase 6b (optional)
pip install -e ".[ml]"             # Phase 8
```

### Known install gotchas (record the outcome in `PROGRESS.md`)

- **openWakeWord**: use the ONNX inference backend on Windows; it needs its pre-trained model files downloaded once (check its docs/`openwakeword.utils.download_models`). The "hey jarvis" model ships as one of the pre-trained wake words.
- **Piper**: the maintained PyPI package `piper-tts` (from OHF-Voice/piper1-gpl) is GPL-3.0-or-later, and Windows install/run success on your Python version is unconfirmed. If it fails `verify_env.py`, use the SAPI fallback (`pyttsx3`) and keep the `TTS` interface unchanged.
- **openWakeWord on Windows** uses only the onnxruntime backend (no tflite, no Speex noise suppression); the pre-trained "hey jarvis" can be over-sensitive, so tune the threshold and measure false triggers.
- Full findings, fallbacks and licences: `docs/12_DEPENDENCY_VERIFICATION.md`.
- **faster-whisper**: CPU with `compute_type="int8"`; first run downloads the model (needs internet once, then works offline).
- **pywinauto / pywin32**: install into the same venv; run `python -m pywin32_postinstall -install` only if imports fail.
- **PyAutoGUI**: keep `FAILSAFE=True`; DPI scaling can affect coordinates (v1 avoids coordinate clicking).
- **fastembed**: downloads a small ONNX embedding model on first use.

## 6. Verify the install (Phase 0 acceptance)

```powershell
python --version                    # 3.11.x or 3.12.x
python scripts/verify_env.py --phase 1
jarvis --help
jarvis doctor
pytest -q
```
Also copy `scripts/verify_env.py` from this documentation set into your project's `scripts/` folder. Re-run it with a higher `--phase` (or `--all`) as you install each phase's extras.

## 7. Workspace folder for safe file operations

Create the default allowed workspace:
```powershell
mkdir "$HOME\Documents\JarvisWorkspace"
```
JARVIS' file tools operate only inside configured `allowed_roots`. Keep test files here.

## 8. Backups and Git hygiene

- `.gitignore`: `.venv/`, `__pycache__/`, `.env`, `*.db`, `logs/`, `reports/`, `benchmarks/results/`, `benchmarks/sandbox/`, `browser_profile/`, `*.onnx` (unless using LFS), `training/data/*.jsonl` if it contains anything private.
- Commit after each working phase with a Conventional Commit message; tag milestones (`git tag phase-2`).
- Keep a copy of the project on cloud/USB before the demo.

## 9. Environment variables (dev only)

| Variable | Effect |
|----------|--------|
| `JARVIS_DRY_RUN=1` | Tools describe but don't execute |
| `JARVIS_LOG_LEVEL=DEBUG` | Verbose logs (still redacted) |
| `JARVIS_DATA_DIR=<path>` | Override data directory (used by tests and benchmarks) |
| `JARVIS_LLM__PLANNER_MODEL=<name>` | Override planner model |
