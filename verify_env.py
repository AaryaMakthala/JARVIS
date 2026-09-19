#!/usr/bin/env python
"""JARVIS environment verifier.

Run on the PC where JARVIS will live:

    python scripts/verify_env.py              # Phase 0 checks
    python scripts/verify_env.py --phase 5    # everything up to Phase 5 (voice)
    python scripts/verify_env.py --all        # every phase
    python scripts/verify_env.py --all --live # also ping Groq with your key (env GROQ_API_KEY or keyring)
    python scripts/verify_env.py --all --json # machine-readable output

Exit code 0 = no FAIL. WARN means "works but read the note". Never prints secrets.
Uses only the standard library so it runs even when nothing else is installed.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import json
import os
import platform
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import asdict, dataclass

IS_WIN = sys.platform == "win32"


@dataclass
class Check:
    name: str
    status: str  # PASS | WARN | FAIL | SKIP
    detail: str = ""
    fix: str = ""
    phase: int = 0


RESULTS: list[Check] = []
MAX_PHASE = 0


def add(name: str, status: str, detail: str = "", fix: str = "", phase: int = 0) -> None:
    if phase > MAX_PHASE:
        RESULTS.append(Check(name, "SKIP", f"needed from phase {phase}", "", phase))
        return
    RESULTS.append(Check(name, status, detail, fix, phase))


def dist_version(dist: str) -> str | None:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


# (distribution name, import name, phase, windows_only)
PACKAGES: list[tuple[str, str, int, bool]] = [
    ("pydantic", "pydantic", 0, False),
    ("pydantic-settings", "pydantic_settings", 0, False),
    ("platformdirs", "platformdirs", 0, False),
    ("typer", "typer", 0, False),
    ("rich", "rich", 0, False),
    ("httpx", "httpx", 0, False),
    ("PyYAML", "yaml", 0, False),
    ("keyring", "keyring", 0, False),
    ("argon2-cffi", "argon2", 0, False),
    ("psutil", "psutil", 0, False),
    ("groq", "groq", 0, False),
    ("langgraph", "langgraph", 1, False),
    ("langgraph-checkpoint-sqlite", "langgraph.checkpoint.sqlite", 1, False),
    ("langchain-core", "langchain_core", 1, False),
    ("pyautogui", "pyautogui", 1, False),
    ("pyperclip", "pyperclip", 1, False),
    ("send2trash", "send2trash", 2, False),
    ("pywin32", "win32api", 2, True),
    ("pywinauto", "pywinauto", 2, True),
    ("ddgs", "ddgs", 4, False),
    ("numpy", "numpy", 5, False),
    ("sounddevice", "sounddevice", 5, False),
    ("onnxruntime", "onnxruntime", 5, False),
    ("faster-whisper", "faster_whisper", 5, False),
    ("openwakeword", "openwakeword", 5, False),
    ("piper-tts", "piper", 5, False),
    ("pyttsx3", "pyttsx3", 5, False),
    ("playwright", "playwright", 6, False),
    ("fastembed", "fastembed", 8, False),
    ("tokenizers", "tokenizers", 8, False),
    ("pytest", "pytest", 0, False),
    ("hypothesis", "hypothesis", 0, False),
    ("ruff", "ruff", 0, False),
    ("mypy", "mypy", 0, False),
]
# Optional packages: missing is WARN, not FAIL (they have documented fallbacks)
OPTIONAL = {"piper-tts", "pyttsx3", "playwright", "pyautogui", "tokenizers", "fastembed"}


def check_python() -> None:
    v = sys.version_info
    ok = (3, 11) <= (v.major, v.minor) < (3, 13)
    add("python version", "PASS" if ok else "FAIL", f"{platform.python_version()} ({platform.architecture()[0]})",
        "Install Python 3.11 or 3.12 (64-bit) and recreate the venv")
    add("python 64-bit", "PASS" if sys.maxsize > 2**32 else "FAIL", platform.architecture()[0],
        "Use 64-bit Python (ML wheels are 64-bit only)")
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    add("virtual environment", "PASS" if in_venv else "WARN", sys.prefix,
        "Activate .venv before installing (python -m venv .venv ; .venv\\Scripts\\activate)")
    add("operating system", "PASS" if IS_WIN else "WARN", platform.platform(),
        "JARVIS targets Windows 10/11; other OSes are for running unit tests only")
    add("sqlite3 module", "PASS", sqlite3.sqlite_version)


def check_pip() -> None:
    try:
        r = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip().splitlines()
        if r.returncode == 0:
            add("pip check (dependency conflicts)", "PASS", "no broken requirements")
        else:
            add("pip check (dependency conflicts)", "FAIL", " | ".join(out[:3]),
                "Resolve conflicts (often: reinstall with a single pip command so the resolver sees all extras)")
    except Exception as e:  # noqa: BLE001
        add("pip check (dependency conflicts)", "WARN", f"could not run: {e}")


def check_packages() -> None:
    for dist, mod, phase, win_only in PACKAGES:
        name = f"package {dist}"
        if win_only and not IS_WIN:
            add(name, "SKIP", "Windows-only", "", phase)
            continue
        ver = dist_version(dist)
        if ver is None:
            status = "WARN" if dist in OPTIONAL else "FAIL"
            add(name, status, "not installed", f"pip install {dist}  (see docs/09_SETUP_AND_DEPENDENCIES.md)", phase)
            continue
        try:
            importlib.import_module(mod)
            add(name, "PASS", ver, "", phase)
        except Exception as e:  # noqa: BLE001
            add(name, "FAIL" if dist not in OPTIONAL else "WARN", f"{ver} installed but import failed: {type(e).__name__}: {e}",
                f"Reinstall {dist}; on Windows check for missing VC++ runtime or a Python-version wheel mismatch", phase)


def check_langgraph_hitl() -> None:
    """Runs a real interrupt -> resume cycle with a SQLite checkpointer: the exact contract JARVIS relies on."""
    name = "langgraph interrupt/resume + SqliteSaver"
    try:
        from typing import TypedDict

        from langgraph.checkpoint.sqlite import SqliteSaver
        from langgraph.graph import END, START, StateGraph
        from langgraph.types import Command, interrupt

        class S(TypedDict, total=False):
            answer: str

        def node(_: S) -> S:
            return {"answer": interrupt({"question": "approve?"})}

        g = StateGraph(S)
        g.add_node("n", node)
        g.add_edge(START, "n")
        g.add_edge("n", END)
        app = g.compile(checkpointer=SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False)))
        cfg = {"configurable": {"thread_id": "verify-1"}}
        first = app.invoke({}, cfg)
        if "__interrupt__" not in first:
            add(name, "FAIL", "graph did not pause on interrupt()", "Check installed langgraph version and docs", 1)
            return
        final = app.invoke(Command(resume="yes"), cfg)
        ok = final.get("answer") == "yes"
        add(name, "PASS" if ok else "FAIL", f"langgraph {dist_version('langgraph')}",
            "" if ok else "Resume value did not reach the node; re-read LangGraph human-in-the-loop docs", 1)
    except Exception as e:  # noqa: BLE001
        add(name, "FAIL", f"{type(e).__name__}: {e}", "pip install langgraph langgraph-checkpoint-sqlite", 1)


def check_keyring() -> None:
    name = "keyring backend (Credential Manager)"
    try:
        import keyring

        backend = keyring.get_keyring()
        bname = type(backend).__name__
        good = IS_WIN and "WinVault" in bname or "Windows" in bname
        keyring.set_password("jarvis-verify", "probe", "value123")
        got = keyring.get_password("jarvis-verify", "probe")
        keyring.delete_password("jarvis-verify", "probe")
        if got != "value123":
            add(name, "FAIL", f"{bname}: round-trip mismatch", "Fix keyring backend", 0)
        elif IS_WIN and not good:
            add(name, "WARN", f"{bname} (expected the Windows Credential Manager backend)", "Uninstall extra keyring backends", 0)
        else:
            add(name, "PASS" if IS_WIN else "WARN", bname + ("" if IS_WIN else " (non-Windows: dev only)"), "", 0)
    except Exception as e:  # noqa: BLE001
        add(name, "WARN" if not IS_WIN else "FAIL", f"{type(e).__name__}: {e}", "pip install keyring", 0)


def check_argon2() -> None:
    try:
        from argon2 import PasswordHasher

        ph = PasswordHasher()
        h = ph.hash("correct horse")
        ph.verify(h, "correct horse")
        add("argon2id hash/verify", "PASS", "round-trip ok", "", 0)
    except Exception as e:  # noqa: BLE001
        add("argon2id hash/verify", "FAIL", f"{type(e).__name__}: {e}", "pip install argon2-cffi", 0)


def check_windows_tools() -> None:
    if not IS_WIN:
        for n in ("schtasks", "powershell", "winget", "WhatsApp desktop"):
            add(f"windows tool: {n}", "SKIP", "Windows-only", "", 3)
        return
    add("windows tool: schtasks", "PASS" if shutil.which("schtasks") else "FAIL", shutil.which("schtasks") or "missing", "Needed for autostart", 3)
    add("windows tool: powershell", "PASS" if shutil.which("powershell") else "FAIL", shutil.which("powershell") or "missing", "Needed for audit", 7)
    add("windows tool: winget", "PASS" if shutil.which("winget") else "WARN", shutil.which("winget") or "missing", "Optional: software-update audit will be skipped", 7)
    pyw = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
    add("pythonw.exe (hidden daemon)", "PASS" if os.path.exists(pyw) else "FAIL", pyw, "Reinstall Python with default components", 3)
    try:
        import tkinter  # noqa: F401
        add("tkinter (confirmation dialogs)", "PASS", "available", "", 3)
    except Exception as e:  # noqa: BLE001
        add("tkinter (confirmation dialogs)", "FAIL", str(e), "Reinstall Python with tcl/tk", 3)


def check_audio() -> None:
    try:
        import sounddevice as sd

        devs = sd.query_devices()
        inputs = [d["name"] for d in devs if d.get("max_input_channels", 0) > 0]
        outputs = [d["name"] for d in devs if d.get("max_output_channels", 0) > 0]
        add("microphone present", "PASS" if inputs else "FAIL", f"{len(inputs)} input device(s)", "Connect a mic; check Windows Settings > Privacy > Microphone (allow desktop apps)", 5)
        add("speakers present", "PASS" if outputs else "WARN", f"{len(outputs)} output device(s)", "", 5)
    except Exception as e:  # noqa: BLE001
        add("audio devices", "FAIL", f"{type(e).__name__}: {e}", "pip install sounddevice", 5)
    try:
        import onnxruntime as ort

        prov = ort.get_available_providers()
        add("onnxruntime CPU provider", "PASS" if "CPUExecutionProvider" in prov else "FAIL", ", ".join(prov), "", 5)
    except Exception as e:  # noqa: BLE001
        add("onnxruntime CPU provider", "FAIL", str(e), "pip install onnxruntime", 5)


def check_live_groq() -> None:
    name = "groq API reachable + model list"
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        try:
            import keyring

            key = keyring.get_password("jarvis", "groq_api_key")
        except Exception:  # noqa: BLE001
            key = None
    if not key:
        add(name, "WARN", "no key found (env GROQ_API_KEY or keyring jarvis/groq_api_key)", "Run `jarvis init`", 0)
        return
    try:
        from groq import Groq

        ids = sorted(m.id for m in Groq(api_key=key).models.list().data)
        add(name, "PASS", f"{len(ids)} models: " + ", ".join(ids[:12]) + (" ..." if len(ids) > 12 else ""),
            "Pick planner/fast models from this list; check structured-output support in Groq docs", 0)
    except Exception as e:  # noqa: BLE001
        add(name, "FAIL", f"{type(e).__name__}: {str(e)[:160]}", "Check key, network, proxy", 0)


def render(results: list[Check]) -> None:
    icon = {"PASS": "[ OK ]", "WARN": "[WARN]", "FAIL": "[FAIL]", "SKIP": "[skip]"}
    for c in results:
        line = f"{icon[c.status]} {c.name}"
        if c.detail:
            line += f" - {c.detail}"
        print(line)
        if c.status in ("FAIL", "WARN") and c.fix:
            print(f"        fix: {c.fix}")
    counts = {s: sum(1 for c in results if c.status == s) for s in icon}
    print(f"\nSummary: {counts['PASS']} pass, {counts['WARN']} warn, {counts['FAIL']} fail, {counts['SKIP']} skipped")


def main() -> int:
    global MAX_PHASE
    ap = argparse.ArgumentParser(description="Verify the JARVIS environment")
    ap.add_argument("--phase", type=int, default=0, help="verify everything needed up to this phase (0-10)")
    ap.add_argument("--all", action="store_true", help="verify all phases")
    ap.add_argument("--live", action="store_true", help="also test Groq with your key (network)")
    ap.add_argument("--json", action="store_true", help="print JSON instead of a table")
    a = ap.parse_args()
    MAX_PHASE = 10 if a.all else a.phase

    check_python()
    check_pip()
    check_packages()
    check_langgraph_hitl()
    check_keyring()
    check_argon2()
    check_windows_tools()
    check_audio()
    if a.live:
        check_live_groq()

    if a.json:
        print(json.dumps([asdict(c) for c in RESULTS], indent=2))
    else:
        render(RESULTS)
    return 1 if any(c.status == "FAIL" for c in RESULTS) else 0


if __name__ == "__main__":
    raise SystemExit(main())
