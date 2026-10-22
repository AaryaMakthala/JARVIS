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

After approach B stages a file, a read-only "preview inspection" phase dumps
the preview surface (region text, role=button accessible names, <img>/<canvas>
alt/aria-label/src, any node naming the file's basename, and the send-candidate
counts) and then states plainly whether a filename-preview gate is feasible. It
screenshots nothing and writes nothing to disk.

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
import time
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

#: Preview-inspection send candidates (Stage 8.4b): counted, never clicked.
SEND_CANDIDATES = [
    '[aria-label="Send"]',
    '[data-testid="send"]',
    '[data-icon="send"]',
    'span[data-icon="send"]',
]
#: Regions that may hold the staged-attachment preview (text only, no clicks).
PREVIEW_REGION_SELECTORS = [
    '[role="dialog"]',
    "#main footer",
    '[data-testid*="preview"]',
    '[data-testid*="composer"]',
]
#: Media nodes inspected for an alt/aria-label/src that names the file.
PREVIEW_MEDIA_SELECTORS = [
    "#main footer img",
    "#main footer canvas",
    '[role="dialog"] img',
    '[role="dialog"] canvas',
]
#: Upper bound on how many role=button nodes are enumerated (bounded output).
_MAX_BUTTONS = 40
#: Short-accessible-name cap, so a button's name is truncated on print.
_NAME_CAP = 60


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


def _clip(value: str, limit: int = 80) -> str:
    """Truncate a value for printing (text only; nothing is written to disk)."""
    text = (value or "").replace("\n", " ").replace("\r", " ").strip()
    return text if len(text) <= limit else text[:limit] + "..."


def _css_str(value: str) -> str:
    """Escape a value for use inside a double-quoted CSS attribute selector."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


#: Read-only accessible-name approximation: aria-label, then title, then text.
_NAME_JS = """(e) => (e.getAttribute('aria-label')
  || e.getAttribute('title')
  || (e.textContent || '').trim().slice(0, 60))"""


def _accessible_name(el) -> str:
    """Best-effort accessible name of an element (evaluate only; never clicks)."""
    try:
        return _clip(str(el.evaluate(_NAME_JS)), _NAME_CAP)
    except Exception:  # noqa: BLE001 - a name we cannot read is reported as empty
        return ""


def _settle(page) -> None:
    """Let the preview render: network idle, then a short fixed wait."""
    try:
        page.wait_for_load_state("networkidle", timeout=10000)
    except Exception as exc:  # noqa: BLE001 - idle may never arrive; proceed anyway
        print(f"  (networkidle wait: {type(exc).__name__}; continuing)")
    time.sleep(2.0)


def _count_matches(page, selector: str) -> int:
    try:
        return page.locator(selector).count()
    except Exception:  # noqa: BLE001 - an unusable selector counts as zero
        return 0


def _inspect_preview(page, target: Path) -> dict[str, object]:
    """Dump the staged-attachment surface WITHOUT clicking anything.

    Returns a small findings dict so :func:`_assess_signal` can judge whether
    the staged file's basename is observable in the DOM (the question behind
    Stage 8.4b's filename-preview gate).
    """
    basename = target.name
    stem = target.stem
    findings: dict[str, object] = {}

    print("\n-- preview inspection (read-only; no clicks, no screenshots) --")
    print(f"  url    : {_clip(page.url, 120)}")
    try:
        print(f"  origin : {page.evaluate('() => window.location.origin')}")
    except Exception:  # noqa: BLE001 - origin is informational only
        print("  origin : <unreadable>")
    print(f"  header : {_read_header(page)}")

    # 1. visible text of the dialog / preview region.
    region_text = ""
    for selector in PREVIEW_REGION_SELECTORS:
        n = _count_matches(page, selector)
        if n == 0:
            continue
        try:
            text = _clip(page.locator(selector).first.inner_text(timeout=3000), 400)
        except Exception:  # noqa: BLE001, S112 - try the next region selector
            continue
        if text:
            print(f"  region text [{selector}]: {text}")
            if not region_text:
                region_text = text
    if not region_text:
        print("  region text: <no preview/dialog region matched>")

    # 2. every role=button with its accessible name (bounded).
    buttons: list[tuple[str, str]] = []
    for i in range(min(_count_matches(page, '[role="button"]'), _MAX_BUTTONS)):
        el = page.locator('[role="button"]').nth(i)
        buttons.append((_accessible_name(el), f"role=button[{i}]"))
    if buttons:
        for name, where in buttons:
            print(f"  button {where}: name={name!r}")
    else:
        print("  buttons: <no role=button element found>")

    # 3. <img>/<canvas> in the preview: alt / aria-label / src prefix.
    media: list[tuple[str, str, str, str]] = []
    for selector in PREVIEW_MEDIA_SELECTORS:
        n = _count_matches(page, selector)
        for i in range(min(n, 6)):
            el = page.locator(selector).nth(i)
            try:
                alt = _clip(el.get_attribute("alt") or "", 60)
                aria = _clip(el.get_attribute("aria-label") or "", 60)
                src = _clip(el.get_attribute("src") or "", 40)
            except Exception:  # noqa: BLE001, S112 - keep describing the other nodes
                continue
            media.append((selector, alt, aria, src))
            print(f"  media {selector}[{i}]: alt={alt!r} aria-label={aria!r} src={src!r}")
    if not media:
        print("  media: <no <img>/<canvas> in the preview region>")

    # 4. elements whose text or aria-label contains the basename (with ext),
    #    plus the bare stem, so we can separate a strong signal from a noisy one.
    try:
        basename_text_hits = page.get_by_text(basename, exact=False).count()
    except Exception:  # noqa: BLE001 - a text scan we cannot run is reported as zero
        basename_text_hits = 0
    try:
        stem_text_hits = page.get_by_text(stem, exact=False).count()
    except Exception:  # noqa: BLE001
        stem_text_hits = 0
    aria_hits = _count_matches(page, f"[aria-label*={_css_str(basename)}]")
    alt_hits = _count_matches(page, f"img[alt*={_css_str(basename)}]")
    in_region = basename in region_text
    print(
        f"  basename text={basename!r}: page-text={basename_text_hits}, "
        f"aria-label={aria_hits}, img-alt={alt_hits}, inside preview region={in_region}"
    )
    print(f"  bare stem {stem!r}: page-text={stem_text_hits} (may match unrelated UI)")

    # 5. send candidates counted (never clicked) - proves nothing was sent.
    total_send = 0
    for selector in SEND_CANDIDATES:
        n = _count_matches(page, selector)
        total_send += n
        print(f"  send candidate {n:>3}  {selector}")
    named_send = 0
    for name, _where in buttons:
        if "send" in name.lower():
            named_send += 1
    total_send += named_send
    print(f"  send-named buttons: {named_send}   total send surface: {total_send} (NOT clicked)")

    findings.update(
        {
            "basename_in_text": bool(basename_text_hits) or in_region,
            "basename_in_aria": bool(aria_hits),
            "basename_in_alt": bool(alt_hits),
            "stem_only": bool(stem_text_hits) and not (basename_text_hits or in_region),
            "basename": basename,
            "stem": stem,
        }
    )
    return findings


def _assess_signal(findings: dict[str, object]) -> None:
    """Say plainly whether the filename-preview gate is feasible, and from what."""
    print("\n-- signal assessment (can we prove 'this is the expected file'?) --")
    reliable: list[str] = []
    unreliable: list[str] = []
    if findings.get("basename_in_text"):
        reliable.append("basename appears as visible text (page and/or preview region)")
    if findings.get("basename_in_alt"):
        reliable.append("basename appears in an <img alt=...>")
    if findings.get("basename_in_aria"):
        reliable.append("basename appears in an element's aria-label")
    if findings.get("stem_only"):
        unreliable.append(
            f"bare stem {findings.get('stem')!r} matched page text without the full name"
        )
    for item in reliable:
        print(f"  [RELIABLE]   {item}")
    for item in unreliable:
        print(f"  [UNRELIABLE] {item}")
    if reliable:
        print("  VERDICT: the filename-preview gate is FEASIBLE (a basename signal exists).")
    else:
        print("  VERDICT: NO reliable filename signal.")
        print(
            "    The preview exposes neither the staged file's name as text nor as"
            " alt/aria-label, so a 'filename matches expected file' gate cannot be"
            " built from the DOM. 8.4b must instead rely on the internal sha256"
            " re-check (hash at attach time vs rehash immediately before Send)."
        )


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
            b_findings: dict[str, object] = {}
            b_ok = _try_hidden_input(page, str(target))
            if b_ok:
                _report("hidden input", True, "set_input_files accepted the file")
                _settle(page)
                b_findings = _inspect_preview(page, target)
                _assess_signal(b_findings)
                # Leave the preview on screen so the owner can eyeball it; the
                # inspection above clicked nothing, and nothing is auto-clicked
                # here either - removal is manual so the evidence stays visible.
                print("  preview left on screen - please remove it manually (X); do NOT send.")
                input("  press Enter once the attachment is gone...")
            else:
                print("  nothing was staged, so there is no draft to remove.")

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
