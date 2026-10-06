"""Console echo of the lines the voice loop actually speaks.

Spoken output used to be invisible on the console: the confirmation prompt,
the plan announcement, the Tier 2 readback and the local acknowledgements
reached only the TTS engine, so a terminal-only owner could not tell what
JARVIS said.  This module prints each spoken line once, as
``[VOICE] SAY: "<text>"``.

Two rules are enforced here rather than in every caller:

* **display only** - the printed text is redacted and truncated; the caller's
  copy (what the TTS engine actually says) is never modified;
* **no transcripts at INFO** - the words never reach ``jarvis.jsonl``; at most
  a character count is logged, at DEBUG.
"""

from __future__ import annotations

import logging

from jarvis.logging_setup import redact

logger = logging.getLogger(__name__)

#: Longest spoken text shown on the console. Longer text is cut for display
#: only; the spoken text itself is untouched.
MAX_SAY_DISPLAY_CHARS = 400


def say_line(text: str) -> None:
    """Print ``[VOICE] SAY: "<text>"`` for one spoken line (console only).

    Redacts secrets and key-shaped strings via
    :func:`jarvis.logging_setup.redact`, truncates to
    :data:`MAX_SAY_DISPLAY_CHARS` for display, and never logs the words -
    only their length, at DEBUG.  Safe under ``pythonw``: a missing or broken
    stdout degrades to a debug log instead of raising.
    """
    shown = redact(str(text))
    if len(shown) > MAX_SAY_DISPLAY_CHARS:
        shown = shown[:MAX_SAY_DISPLAY_CHARS] + "…"
    logger.debug("say_line: %d chars", len(text))
    try:
        print(f'[VOICE] SAY: "{shown}"', flush=True)
    except (OSError, ValueError):
        logger.debug("say_line: console write failed", exc_info=True)
