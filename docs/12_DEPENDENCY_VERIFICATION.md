# 12 — Dependency Verification

**Goal:** every dependency is known to install, import, and behave as this project needs, on *your* Windows machine, and
the exact working set is frozen. This document records what was checked, what could not be checked, and how you verify the rest.

## 1. What was checked (research done 19 Sep 2026) and what it changed

I could search the web but could **not** install packages or run code on a Windows PC from where I work, so anything
marked "verify on your PC" is not proven yet. `scripts/verify_env.py` is what proves it.

| Area | Finding | Consequence for the project |
|------|---------|-----------------------------|
| **LangGraph human-in-the-loop** | Official docs confirm the contract: a **checkpointer is mandatory**, a **`thread_id`** must be supplied, `interrupt()` surfaces its payload under `__interrupt__`, and you resume with **`Command(resume=value)`**. SQLite checkpointing lives in the separate `langgraph-checkpoint-sqlite` package. The docs also note LangGraph does **not** expire abandoned paused threads on its own. | The design in `02_ARCHITECTURE.md` matches. Added: stale-confirmation cleanup (`11_DEPLOYMENT §4.3`). `verify_env.py` runs a real interrupt→resume cycle against a SQLite checkpointer to prove the installed versions behave this way. |
| **Groq structured output** | JSON-schema structured output is supported only on **specific models**, and a model that doesn't support it returns an HTTP 400 (`does not support response format json_schema`); real projects have hit this after model or SDK changes. Third-party trackers listed models such as the gpt-oss family, Llama 4 Scout/Maverick and Kimi K2 variants as supporting it, but model lists change and some may be retired. | `LLMClient` must (a) read the planner model from config, (b) try JSON-schema mode, (c) on a 400 fall back to plain JSON mode + Pydantic validation + one repair retry, (d) log which mode was used. Choose the model from Groq's *current* docs; `verify_env.py --live` lists the models your key can actually use. |
| **openWakeWord on Windows** | On Windows only the **onnxruntime** backend is installed (tflite is unsupported). A pre-trained **"hey jarvis"** model exists. Speex noise suppression is **Linux-only**, so don't plan on it. Users report the pre-trained model can be over-sensitive (e.g., partial-word triggers). | Voice code uses the ONNX backend; tune the threshold, add a debounce, and optionally require a short STT confirmation for Tier ≥1. Measure false triggers per hour in Phase 5 (already an acceptance criterion). |
| **Piper TTS** | The maintained package on PyPI (`piper-tts`) comes from **OHF-Voice/piper1-gpl** and is licensed **GPL-3.0-or-later**; the older rhasspy/piper is archived. I saw versions in the 1.3–1.4 range but **did not confirm** that Windows wheels install cleanly on Python 3.12. | TTS stays behind an interface with a **Windows SAPI fallback (`pyttsx3`)**. Licence note in §5. If Piper fails `verify_env.py`, ship with SAPI and record it — that is an acceptable outcome. |
| **Not verified** | `ddgs`, `faster-whisper`, `fastembed`, `pywinauto`, `pywin32`, `keyring` (Windows backend), `pyautogui`, `send2trash`, `sounddevice`, exact `langchain-groq`/`groq` SDK signatures on Python 3.12/Windows. | All covered by `verify_env.py`. OpenCode must still read each installed package's docs/source before using its API (`AGENTS.md §7.4`). |

## 2. Verification workflow (do this in order)

1. Create a fresh venv on the target Windows PC (Python 3.11 or 3.12, 64-bit).
2. Install the phase's extras with **one** pip command so the resolver sees everything together:
   `pip install -e ".[dev]"` then later `".[dev,voice]"`, `".[dev,voice,browser,ml]"`.
3. Run `python scripts/verify_env.py --phase <N>` (or `--all`). Fix every **FAIL**; read every **WARN**.
4. `pip check` must be clean (the script runs it). Conflicts here are the most common hidden breakage.
5. Run `python scripts/verify_env.py --all --live` once your Groq key is stored: confirms the network path and lists usable models.
6. Do **hardware checks** the script can't fully prove: say "hey jarvis" for 2 minutes, dictate into Notepad, play TTS.
7. When everything is green: `pip freeze > requirements.lock`, commit it with the date and the output of `verify_env.py --all --json` saved as `docs/env-report-<date>.json`.
8. From then on install with constraints: `pip install -c requirements.lock -e ".[dev,voice]"`.
9. Re-run `verify_env.py` after **any** dependency change, Python upgrade, or Windows feature update.

`scripts/verify_env.py` checks: Python version/bitness/venv, `pip check`, every package (installed **and** importable, by phase),
a real LangGraph interrupt→resume round trip, keyring backend round trip (should be the Windows Credential Manager), Argon2id hash/verify,
`schtasks`/`powershell`/`winget`/`pythonw.exe`/`tkinter`, microphone and speakers, onnxruntime CPU provider, and (with `--live`) Groq reachability + model list.
It uses only the standard library so it runs even when the environment is broken, prints fixes, never prints secrets, and exits non-zero on FAIL.

## 3. Fallback matrix (decide in advance, don't panic later)

| If this fails… | Use this instead | Cost |
|----------------|------------------|------|
| Groq unavailable/rate-limited | Gemini fallback (`provider_order`), or a local model via Ollama if you have the hardware (add an `LLMClient` implementation) | Different structured-output behaviour; re-run planner tests |
| Groq model lacks JSON-schema mode | JSON mode + Pydantic + repair retry (already required) | Slightly more repair retries |
| `piper-tts` won't install/run on Windows | `pyttsx3` (SAPI) | Less natural voice |
| `openwakeword` unstable | Push-to-talk hotkey or a "listen" command in `jarvis chat`; keep wake word as stretch | Less "Jarvis-like" demo |
| `faster-whisper` too slow on your CPU | `tiny` model, `int8`, shorter segments; or Groq's hosted Whisper endpoint if allowed by your privacy claim (breaks "audio never leaves the PC"; document it) | Accuracy or privacy trade-off |
| `pywinauto` can't read WhatsApp UI | Fail closed: leave draft, user presses Enter; optional Playwright WhatsApp Web route | Less automation, still safe |
| `fastembed` problems | `sentence-transformers` (MiniLM) or TF-IDF retrieval baseline | Larger install or lower recall |
| `ddgs` blocked/rate-limited | Tavily free tier | Needs another key |
| Keyring backend issues | Stop and fix: do **not** fall back to plaintext secrets | — |

## 4. Update policy

- Until Phase 9: upgrade deliberately, one package at a time, re-run tests + `verify_env.py`.
- **Freeze at least 2 weeks before submission.** After that only security fixes; re-run the full suite after each.
- Keep `requirements.lock` from the last fully green run; if an upgrade breaks something, roll back to it.
- Run `pip-audit` (dev dependency) against the lock before releases; review findings, don't blindly upgrade.

## 5. Licences to note in the report

| Component | Licence note |
|-----------|--------------|
| `piper-tts` (piper1-gpl) | **GPL-3.0-or-later.** Fine for a course project that uses it as an optional dependency; if you *distribute* a bundled installer containing it, the GPL applies to the combination. Mention it, keep it optional, keep SAPI fallback. |
| openWakeWord | Code is open source; check the licence of each pre-trained model you use (some models have restrictions) and cite it. |
| faster-whisper / Whisper models | Open source (check the model card licence). |
| LangGraph / LangChain | Permissive open-source licences (verify in the repo). |
| Groq / Gemini APIs | Governed by their terms of service and free-tier limits; do not resell or exceed limits. |
| Your own code | Choose a licence explicitly (e.g., MIT) and include `LICENSE`; list third-party licences in `THIRD_PARTY_NOTICES.md` (generate with `pip-licenses`). |

## 6. Definition of "dependencies are good"

- [ ] `verify_env.py --all --live` = 0 FAIL, every WARN understood and documented in `PROGRESS.md`.
- [ ] `pip check` clean; `requirements.lock` committed and used by `install.ps1`.
- [ ] Clean-machine install (`11_DEPLOYMENT §11`) succeeded using only the lock file.
- [ ] Every fallback in §3 is implemented behind an interface, or explicitly marked "not needed".
- [ ] `pip-audit` reviewed; `THIRD_PARTY_NOTICES.md` generated.
- [ ] Freeze date recorded.
