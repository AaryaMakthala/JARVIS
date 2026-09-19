# 11 — Deployment and Packaging (Windows)

"Deployed" for this project means: **JARVIS is installed on your PC like a real application, starts by itself at login,
survives crashes and reboots, can be upgraded and uninstalled cleanly, and can be installed on another PC by following
the README.** This document specifies that. It is built in **Phase 10** (see `05_BUILD_PLAN.md`).

## 1. Deployment targets

| Target | Who | How |
|--------|-----|-----|
| **Dev install** | You while coding | `pip install -e ".[dev]"` in the project venv |
| **Daily-driver install** | You, using JARVIS for real | Wheel + `pipx` (isolated venv), autostart enabled |
| **Examiner/other-PC install** | Viva demo, second machine | Same wheel + `install.ps1`, tested on a clean Windows profile/VM |
| *(optional)* **Standalone installer** | Non-technical users | PyInstaller one-folder + Inno Setup (see §9). Only if your college demands a double-click installer |

**Recommendation:** ship the wheel + `pipx` route. It is the most reliable for a Python project with native dependencies
(onnxruntime, faster-whisper, pywin32) and is much easier to debug than a frozen executable.

## 2. Build the release artifact

```powershell
python -m pip install build
python -m build            # creates dist\jarvis_agent-<version>-py3-none-any.whl and .tar.gz
```
Requirements for the build to succeed and be reproducible:
- `pyproject.toml` has name, version, `requires-python`, dependencies, extras, console script (`jarvis = "jarvis.cli:app"`), and includes package data (default config template, prompt files, the ONNX risk model if bundled, the tkinter dialog assets).
- `jarvis --version` prints the version from package metadata (`importlib.metadata.version("jarvis-agent")`).
- A `requirements.lock` (from a verified environment, see `12_DEPENDENCY_VERIFICATION.md`) is shipped in the repo and used by `install.ps1` as constraints: `pip install -c requirements.lock ...`.
- Tag every release in Git (`v0.1.0`) and keep `CHANGELOG.md`.

## 3. Install on a PC (`scripts/install.ps1`, written in Phase 10)

The script must be **idempotent, non-admin, and print each step**. Outline:

1. Check Python 3.11/3.12 64-bit (`py -3.12 --version`); if missing, print the download link and stop (do not silently install).
2. Install pipx for the current user: `python -m pip install --user pipx` then `python -m pipx ensurepath` (open a new terminal afterwards). Alternative: `uv tool install` if the user has `uv`. *(Verify current pipx/uv commands and extras syntax when implementing.)*
3. `pipx install <path-to-wheel>` and inject/choose extras, e.g. `pipx install "<wheel>[voice,ml]"` (verify syntax for local wheels with extras; fallback: `pipx install <wheel>` then `pipx inject jarvis-agent <packages...>`).
4. Run `jarvis doctor` and show the result.
5. Run `jarvis models download` (§5) with a progress display.
6. Run `jarvis init` (keys, password, contacts, voice).
7. Offer `jarvis autostart enable`.
8. Print "Next steps" (how to talk to it, how to stop it, where logs are).

Also write `scripts/uninstall.ps1` calling `jarvis uninstall` then `pipx uninstall jarvis-agent`.

## 4. The daemon as a real background app

### 4.1 Autostart (Task Scheduler, current user, no admin)

`jarvis autostart enable` creates a task named `JarvisAgent` from an **XML definition** (import with `schtasks /Create /TN JarvisAgent /XML <file>`), because XML gives control over settings that command-line flags don't:

| Setting | Value |
|---------|-------|
| Trigger | At log on of the **current user** (delay 15–30 s so audio/network are ready) |
| Principal | Current user, **"Run only when user is logged on"**, run level **LimitedUser** (NOT highest privileges) |
| Action | `<venv>\Scripts\pythonw.exe -m jarvis daemon` — resolve `pythonw.exe` from the venv that contains jarvis (`Path(sys.executable).with_name("pythonw.exe")`), so it works with pipx |
| Settings | Restart on failure: every 1 min, up to 3 times · "Do not start a new instance" if already running · run on batteries · don't stop on battery · execution time limit disabled |
| Not a Windows Service | Services run in session 0 and can't reach the user's desktop, audio, or windows |

`jarvis autostart status` reads the task and reports: enabled, last run time, last result, next trigger.
`jarvis autostart disable` deletes it. Escape XML values; never build the command from untrusted input.

### 4.2 Things that break `pythonw` daemons (must be handled in code)

- **`pythonw` has no console: `sys.stdout` and `sys.stderr` are `None`.** Any `print()` or library that writes to stderr can crash the daemon. On daemon start, redirect stdout/stderr to the rotating log file (or a null writer) **before** importing anything noisy.
- Install `sys.excepthook` and `threading.excepthook` to log tracebacks; the daemon must never die silently.
- **Single instance:** use a named mutex (`win32event.CreateMutex`) or an exclusive lock file plus `daemon.json` pid check. A second start prints "already running" and exits.
- **Working directory** is not the project folder under Task Scheduler: always use absolute paths from `platformdirs`.
- **Environment:** Task Scheduler does not load your shell profile; PATH may differ. Resolve executables (`notepad`, `chrome`) via the `[apps]` allowlist with full-path lookup (`shutil.which` or known install paths).
- **Graceful shutdown:** handle `WM_QUERYENDSESSION`/logoff via the `atexit` + signal handlers that flush logs and close DBs; `jarvis stop` sends `shutdown` over IPC first, then falls back to killing the pid.
- **Audio device changes** (headset unplugged): the voice loop must catch PortAudio errors, log, and retry with backoff instead of crashing.
- **Sleep/resume:** on resume, re-open the microphone stream and re-check the network before the next LLM call.

### 4.3 Stale confirmations

LangGraph does not expire paused threads by itself. On daemon start and every 10 minutes: mark tasks waiting for confirmation longer than the confirmation timeout as `expired` (write to `task_log`), and never resume them. On version upgrade, discard all pending checkpoints (they may be incompatible with new code).

## 5. First-run assets: `jarvis models download`

Some components download files the first time they run. Make this explicit and testable instead of surprising the user mid-command:

| Asset | Used by | Notes |
|-------|---------|-------|
| faster-whisper model (`tiny`/`base`) | STT | Downloaded from Hugging Face on first use; needs internet once |
| openWakeWord pre-trained models (incl. "hey jarvis") | wake word | One-time download via the library's utility; verify the current function name in the installed version |
| Piper voice (`.onnx` + `.onnx.json`) | TTS | Choose an English voice; store in the data dir |
| fastembed / MiniLM embedding model | skill memory | Small ONNX download |
| Risk classifier (`model.onnx`) | policy | Bundled in the wheel or downloaded from your GitHub release with SHA-256 verification |
| Playwright Chromium | WhatsApp Web fallback | `playwright install chromium` (optional) |

`jarvis models download` downloads what's enabled in config, shows progress, verifies checksums where available, and is safe to re-run.
`jarvis doctor` reports which assets are present. After download, JARVIS must work **offline** for everything except LLM/web-search calls.

## 6. Upgrades and migrations

- Version in `schema_version` table (SQLite `PRAGMA user_version` is fine). On start, run ordered migrations; back up `memory.db` to `memory.db.bak-<version>` before migrating. Never modify old migrations; add new ones.
- Config: `config.toml` has `config_version`; missing keys get defaults; unknown keys produce a warning, not a crash.
- Upgrade procedure: `jarvis stop` → `pipx upgrade` (or reinstall the new wheel) → `jarvis doctor` → daemon restarts at next login (or `jarvis daemon` manually). `jarvis upgrade-check` may compare the installed version with your GitHub release (optional; no auto-update in v1).
- Downgrades are unsupported; keep the DB backup.

## 7. Uninstall (`jarvis uninstall`)

Interactive, each step asks yes/no: stop daemon → remove Task Scheduler task → remove keyring entries (`jarvis` service: API keys, IPC token, password hash) → optionally delete the data dir (`%LOCALAPPDATA%\jarvis`: memory, logs, reports, contacts) → print the final `pipx uninstall jarvis-agent` command. Test that after uninstall no scheduled task, no process, and no keyring entries remain.

## 8. Operations: health, logs, backups

| Need | Feature |
|------|---------|
| Is it running? | `jarvis status` (pid, port, uptime, voice on/off, unlocked, queue, last task) |
| Is the install healthy? | `jarvis doctor` (same checks as `scripts/verify_env.py` plus config/models/keys) |
| Why did it fail? | `jarvis logs tail [-n 100]`, `jarvis logs path`; rotating JSONL (e.g., 5 × 5 MB), redacted |
| Crash reports | Traceback logged + `crash-<timestamp>.txt` in the logs dir (no secrets) |
| Backup | `jarvis backup` zips `contacts.json`, `config.toml`, `memory.db` (not secrets) to a folder of your choice |
| Reset | `jarvis reset --memory` / `--all` with confirmation |

## 9. Optional standalone installer (only if required)

Approach: PyInstaller **one-folder** build of `jarvis` (windowed variant for the daemon, console variant for the CLI) + Inno Setup script that installs to `%LOCALAPPDATA%\Programs\Jarvis` (no admin), creates the Task Scheduler task, and adds an uninstaller.

Honest caveats — budget extra time if you choose this:
- Hidden imports and data files for `pywinauto`/`comtypes`, `pywin32`, `onnxruntime`, `faster_whisper`, `openwakeword`, `tokenizers`, `fastembed` usually need manual `--collect-all`/hooks; iterate until a clean VM works.
- Size is large (hundreds of MB). Do **not** bundle Whisper/embedding/voice models; use `jarvis models download`.
- Unsigned executables that use keyboard/UI automation are commonly flagged by antivirus/SmartScreen. Code signing needs a paid certificate; a self-signed cert won't remove warnings on other PCs. For a viva, the wheel + pipx route avoids this.
- Each dependency upgrade can break the frozen build; freeze dependencies first (see `12_DEPENDENCY_VERIFICATION.md`).

## 10. Security hardening for a daily-driver install

- Never run as administrator. The audit reports "needs admin" items instead of elevating.
- Data dir permissions: default `%LOCALAPPDATA%` is per-user; verify no world-writable files. Secrets only in Credential Manager.
- Daemon listens only on `127.0.0.1`; verify with `netstat -ano | findstr <port>` that no other interface is bound.
- Run `pip-audit` against `requirements.lock` before each release; record the result in `CHANGELOG.md`.
- Windows microphone privacy: ensure "Let desktop apps access your microphone" is on for voice; the visible listening indicator must be accurate.
- **Gradual rollout on your own PC** (protects your real files):
  1. Week 1: `--dry-run` + `allowed_roots` = only `~/Documents/JarvisWorkspace`.
  2. Week 2: real Tier 0/1 actions; review logs daily.
  3. Week 3: enable Tier 2 (delete/WhatsApp) with a test contact; keep the Recycle Bin and undo log in mind.
  4. Only then widen `allowed_roots` (e.g., Desktop, Downloads), one folder at a time.

## 11. Clean-machine test (required before the viva)

Use **Windows Sandbox** (Pro/Enterprise/Education editions), a VM, or a second Windows user account/PC:
1. Fresh profile with only Python installed. Copy `dist\*.whl`, `requirements.lock`, `install.ps1`.
2. Run `install.ps1` exactly as the README says. Everything must work without your dev environment.
3. Run `python scripts/verify_env.py --all --live` (or `jarvis doctor`), then the demo script from `10_REPORT_AND_VIVA.md`.
4. Reboot/log off and on: daemon auto-starts; `jarvis status` OK.
5. Run `jarvis uninstall` and confirm nothing remains.
6. Write every problem you hit into `PROGRESS.md` and fix the install docs, then repeat once more.

## 12. Release checklist

- [ ] All phases' acceptance tests + invariant tests + red-team suite green on the release commit.
- [ ] `scripts/verify_env.py --all` passes on the release machine; `requirements.lock` regenerated and committed.
- [ ] `pip-audit` reviewed; licences reviewed (`12_DEPENDENCY_VERIFICATION.md §5`).
- [ ] Version bumped, `CHANGELOG.md` updated, Git tag created, wheel built from the tag.
- [ ] Clean-machine test (§11) passed.
- [ ] README install steps re-tested by following them literally.
- [ ] Backup copy of the repo + wheel + lock file + report on external storage/cloud.
- [ ] **Dependency freeze** started (no upgrades except security fixes) at least 2 weeks before submission.
