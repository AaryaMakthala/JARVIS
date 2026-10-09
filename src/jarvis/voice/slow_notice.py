"""One-shot "taking a moment" notice for long-running voice tasks.

The daemon's voice-loop thread blocks inside ``DaemonServer._voice_submit``
while the graph works.  Once a task has run longer than ``SLOW_THRESHOLD_S``
the caller may speak ``SLOW_NOTICE_TEXT`` once so the user knows the job is
still in progress.  The notice is suppressed while anything else owns the
microphone (a confirmation or clarification dialogue), while the slot is
already done, and while the voice pipeline is stopping.

Pure module: no mutable state, no I/O beyond the caller-supplied ``tts``, so
tests use a fake TTS and a fake clock.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

#: The exact fixed phrase; the notice never says anything else.
SLOW_NOTICE_TEXT = "This is taking a moment"

#: Default threshold: speak once a task has run at least this long.
SLOW_THRESHOLD_S = 4.0


def maybe_speak(
    tts: Any,
    started: float,
    threshold: float = SLOW_THRESHOLD_S,
    *,
    confirmation_owns_mic: bool = False,
    clarification_owns_mic: bool = False,
    done: bool = False,
    stopping: bool = False,
    already_spoken: bool = False,
    clock: Callable[[], float] = time.monotonic,
) -> bool:
    """Speak the slow notice once the task has exceeded ``threshold``.

    Returns ``True`` only when the notice was spoken on this call, so the
    caller feeds it back as ``already_spoken`` to make the notice one-shot
    per task.  The notice is skipped when it was already spoken, when
    ``tts`` is ``None``, while a confirmation or clarification owns the
    microphone, when the slot is already done, and while voice is stopping.
    """
    if (
        tts is None
        or already_spoken
        or confirmation_owns_mic
        or clarification_owns_mic
        or done
        or stopping
    ):
        return False
    if clock() - started < threshold:
        return False
    tts.speak(SLOW_NOTICE_TEXT)
    return True
