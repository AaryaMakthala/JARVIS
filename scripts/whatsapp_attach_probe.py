r"""WhatsApp Web attach probe (owner-run, Stage 8.4b).

Standalone diagnostic for ONE question: **which attach mechanism actually works
in WhatsApp Web, and what does the chat header / attachment preview look like?**
It is NOT a test, is never imported by the JARVIS app, and is never run by the
agent - only the owner runs it by hand.

It is deliberately safe:
  * it opens the *whatsapp* browser profile (B6) - never the general one;
  * it navigates to web.whatsapp.com and waits for YOU to log in;
  * it tries the attach approaches against YOUR OWN test chat;
  * it NEVER locates, clicks or presses Enter on Send. It prints the send
    button's presence after each attempt purely to prove it was not clicked.

Run (stop the daemon first so nothing else holds the profile):

    .venv\Scripts\python.exe scripts\whatsapp_attach_probe.py

Optional: probe with a real photo instead of the generated 1x1 PNG:

    .venv\Scripts\python.exe scripts\whatsapp_attach_probe.py --file "C:\path\to\photo.jpg"

WhatsApp ToS: your own test number only (ROADMAP_V2.1, risk I9).
"""

from __future__ import annotations

import argparse
import struct
import sys
import tempfile
import zlib
from pathlib import Path

sys.path.insert(0, "src")

from jarvis.browser.context import MANAGED_BROWSER_CHANNEL
from jarvis.config import browser_profile_dir

WHATSAPP_URL = "https://web.whatsapp.com/"

#: Locale-independent attributes first, aria-labels (English + Arabic) after.
ATTACH_SELECTORS = [
    '[data-testid="attach-menu-plus"]',
    '[data-testid="clip"]',
    '[aria-label="Attach"]',
    '[aria-label="attach"]',
    '[aria-label="\u0645\u0631\u0641\u0642"]',
]
PHOTO_MENU_SELECTORS = [
    '[data-testid="attach-menu-photos"]',
    '[data-testid="photos-menu-item"]',
    '[data-testid="photos"]',
]
FILE_INPUT_SELECTOR = 'input[type="file"]'
HEADER_SELECTORS = ["#main header", '[data-testid="conversation-header"]']
PREVIEW_SELECTORS = [
    "#main footer img",
    "#main footer video",
    "#main footer [data-testid*='preview']",
]
DRAFT_REMOVE_SELECTORS = [
    "#main footer [data-testid='dismissible-input']",
    '#main footer [aria-label="Remove"]',
    '#main footer [aria-label="\u0645\u0634\u0627\u062f\u0629 \u0627\u0644\u0645\u0644\u0641"]',
]
#: Read-only: printed to prove it was never clicked.
SEND_SELECTORS = ['[data-testid="send"]', '[data-testid="send-button"]']


def _make_probe_png(path: Path) -> Path:
    """Write a valid 1x1 red PNG so WhatsApp always has something to preview."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00"))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)
    return path


def _report(label: str, ok: bool, detail: str) -> None:
    tag = "WORKS" if ok else "no   "
    print(f"  [{tag}] {label}: {detail}")


def _first_match(page, selectors: list[str]) -> str | None:
    """Return the first selector with >= 1 match, printing every count."""
    found = None
    for selector in selectors:
        try:
            n = page.locator(selector).count()
        except Exception:  # noqa: BLE001 - a bad selector must not stop the probe
            n = -1
        print(f"      {n:>3}  {selector}")
        if n > 0 and found is None:
            found = selector
    return found


def _read_header(page) -> str:
    for selector in HEADER_SELECTORS:
        try:
            if page.locator(selector).count() > 0:
                return " | ".join(page.locator(selector).all_inner_texts()[:3]).replace("\n", " / ")
        except Exception:  # noqa: BLE001, S112 - probe keeps trying the next selector
            continue
    return "<no chat header found>"


def _read_preview(page) -> str:
    """Describe the composer: filename/size text plus any image/video thumbnail."""
    parts: list[str] = []
    try:
        text = page.locator("#main footer").inner_text(timeout=2000).strip()
        if text:
            parts.append(f"composer text={text!r}")
    except Exception:  # noqa: BLE001, S110 - the composer may simply be absent
        pass
    for selector in PREVIEW_SELECTORS:
        try:
            loc = page.locator(selector)
            for i in range(min(loc.count(), 3)):
                el = loc.nth(i)
                parts.append(
                    f"{selector}[{i}] alt={el.get_attribute('alt')!r} "
                    f"src={(el.get_attribute('src') or '')[:120]!r}"
                )
        except Exception:  # noqa: BLE001, S112 - keep describing the other nodes
            continue
    return "; ".join(parts) if parts else "<nothing visible in the composer>"


def _send_button_count(page) -> int:
    total = 0
    for selector in SEND_SELECTORS:
        try:
            total += page.locator(selector).count()
        except Exception:  # noqa: BLE001, S112 - count is a best-effort signal
            continue
    return total


def _try_hidden_input(page, target: str) -> bool:
    """Approach B: set the file on a hidden input[type=file] (no clicks)."""
    try:
        page.set_input_files(FILE_INPUT_SELECTOR, target)
        return True
    except Exception as exc:  # noqa: BLE001 - probe reports, never crashes
        _report("hidden input[type=file]", False, f"{type(exc).__name__}: {exc}")
        return False


def _try_file_chooser(page, target: str) -> bool:
    """Approach A: click attach -> Photos/Videos, then feed the file chooser."""
    attach = _first_match(page, ATTACH_SELECTORS)
    if attach is None:
        _report("file chooser", False, "no attach button matched any selector")
        return False
    menu = _first_match(page, PHOTO_MENU_SELECTORS)
    if menu is None:
        _report("file chooser", False, "attach found, but no Photos/Videos menu item matched")
        return False
    try:
        with page.expect_file_chooser(timeout=8000) as chooser:
            page.locator(attach).first.click()
            page.locator(menu).first.click()
        chooser.value.set_files(target)
        return True
    except Exception as exc:  # noqa: BLE001 - probe reports, never crashes
        _report("file chooser", False, f"{type(exc).__name__}: {exc}")
        return False


def _clear_draft(page) -> bool:
    """Try to drop the staged attachment so the next attempt starts clean."""
    for selector in DRAFT_REMOVE_SELECTORS:
        try:
            loc = page.locator(selector)
            if loc.count() > 0:
                loc.first.click()
                return True
        except Exception:  # noqa: BLE001, S112 - try the next removal selector
            continue
    return False


def _run(target: Path) -> None:
    print("=" * 72)
    print("WhatsApp Web ATTACH PROBE  (owner-run; NEVER presses Send)")
    print(f"test file : {target}")
    print(f"profile   : {browser_profile_dir('whatsapp')}")
    print("ToS: your own test number only.")
    print("=" * 72)

    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # noqa: BLE001
        print(f"Playwright is not installed: {exc}")
        print('Run: .venv\\Scripts\\pip install -e ".[browser]"')
        return

    profile = browser_profile_dir("whatsapp")
    with sync_playwright() as pw:
        try:
            ctx = pw.chromium.launch_persistent_context(
                str(profile), channel=MANAGED_BROWSER_CHANNEL, headless=False
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Could not open the whatsapp profile: {exc}")
            print("Is the daemon or another Chrome holding it? Try: jarvis stop")
            return
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        try:
            page.goto(WHATSAPP_URL, wait_until="domcontentloaded", timeout=45000)
            print(f"\nOpened {WHATSAPP_URL}")
            input("Log in (QR if needed), then press Enter here...")
            input("Open your TEST chat in that window, then press Enter here...")

            print("\n-- chat header --")
            print(f"  {_read_header(page)}")

            print("\n-- attach approach B: hidden input[type=file] --")
            print("  selector counts:")
            b_ok = _try_hidden_input(page, str(target))
            if b_ok:
                _report("hidden input", True, "set_input_files accepted the file")
                print(f"  preview: {_read_preview(page)}")
            if not _clear_draft(page):
                print("  could not auto-clear the draft - please remove it (X) manually.")
                input("  press Enter once the attachment is gone...")

            print("\n-- attach approach A: attach button -> Photos/Videos -> chooser --")
            a_ok = _try_file_chooser(page, str(target))
            if a_ok:
                _report("file chooser", True, "set_files accepted the file")
                print(f"  preview: {_read_preview(page)}")
            if not _clear_draft(page):
                print("  could not auto-clear the draft - please remove it (X) manually.")
                input("  press Enter once the attachment is gone...")

            print("\n-- result --")
            _report("approach B (hidden input)", b_ok, "staged a file without any click")
            _report("approach A (file chooser)", a_ok, "staged a file via attach -> Photos")
            print(f"  send button present (NOT clicked): {_send_button_count(page)}")
            print(f"  final header: {_read_header(page)}")
            if b_ok:
                print("  RECOMMENDATION: approach B (no clicks, least failure surface).")
            elif a_ok:
                print("  RECOMMENDATION: approach A only (attach->Photos menu required).")
            else:
                print("  Neither approach worked - send this whole output back before 8.4b.")
        finally:
            input("\nPress Enter to close the probe browser...")
            ctx.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="WhatsApp Web attach probe (owner-run).")
    parser.add_argument("--file", help="an image file to stage (default: a generated 1x1 PNG)")
    args = parser.parse_args()

    if args.file:
        target = Path(args.file).expanduser()
        if not target.exists():
            print(f"not found: {target}")
            return 2
    else:
        target = _make_probe_png(Path(tempfile.gettempdir()) / "jarvis_wa_probe.png")
    _run(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
