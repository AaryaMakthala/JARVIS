"""Read-only check of tool-group prompt trimming (no LLM, no network).

Usage: ``python scripts/verify_trim.py``

For each representative request this builds the trimmed tool catalogue with the
same pure selector the brain uses, prints the selected tools and byte size, and
checks that the tool the request needs is present.  It never changes
``agent.prompt_trim_enabled`` and never calls a provider.
"""

from __future__ import annotations

import json

from jarvis.agent.tool_groups import select_catalogue
from jarvis.tools.registry import build_default_registry

#: ``(request, tool names the request must be able to use)``.  An empty tuple
#: means the request is conversational and needs no tool.
CASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("open notepad", ("open_app",)),
    (
        "search youtube for lofi and open the first video",
        ("browser_search", "browser_open_result"),
    ),
    ("send a whatsapp message to Papa", ("whatsapp_send",)),
    ("take a screenshot and show it", ("take_screenshot", "show_file")),
    ("what is 2+2", ()),
    ("find my resume file", ("file_search",)),
    ("set volume to 30", ("volume_control",)),
    ("read the file notes.txt", ("read_file",)),
    ("what's on my screen", ("read_screen",)),
    ("take a picture of me", ("take_photo",)),
)


def _bytes(entries: list[dict[str, object]]) -> int:
    """UTF-8 byte size of a catalogue as the prompt would paste it."""
    return len(
        json.dumps(entries, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )


def main() -> int:
    """Print the per-request result and return 0 only when every case passes."""
    registry = build_default_registry()
    full = registry.catalogue_for_llm()
    full_bytes = _bytes(full)
    print(f"registry tools: {len(full)}")
    print(f"full catalogue: {full_bytes} bytes")
    print()

    failures = 0
    fallbacks = 0
    for request, required in CASES:
        selected = select_catalogue(registry, {"user_input": request})
        names = [str(entry.get("name") or "") for entry in selected]
        fallback = len(selected) >= len(full)
        fallbacks += int(fallback)
        missing = [name for name in required if name not in names]
        ok = not missing
        failures += 0 if ok else 1
        needed = ", ".join(required) or "(none)"
        print(f"{'PASS' if ok else 'FAIL'} | {request!r}")
        print(f"     needs: {needed}")
        print(f"     selected ({len(names)}): {', '.join(names)}")
        print(
            f"     bytes: {_bytes(selected)} (full {full_bytes})"
            f"  fallback={fallback}"
        )
        if missing:
            print(f"     missing: {', '.join(missing)}")

    print()
    print(
        f"full-catalogue fallback rate: {fallbacks}/{len(CASES)}"
        f" = {fallbacks / len(CASES):.0%}"
    )
    print(f"result: {len(CASES) - failures}/{len(CASES)} PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
