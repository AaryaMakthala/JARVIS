"""Shared task runtime: the single user-answer window and resume values.

The daemon suspends a graph run on a confirmation (or clarification) and must
receive the user's answer within one bounded window.  This module owns that
window and the typed resume answers, so there is exactly one timeout constant
in the whole codebase and the timeout path RESUMES the graph (fail-closed
refusal) instead of abandoning the task.

Design notes (docs/02_ARCHITECTURE.md):
* ``CONFIRMATION_TIMEOUT_S`` is the one timeout for confirmation AND
  clarification answers.  Nothing else defines a competing window.
* ``wait_for_response`` blocks on the slot's event, then hands back whatever
  the responder stashed on ``slot.resume_answer`` (and clears it), or a
  ``timeout_answer`` refusal when the window elapsed.  A slot.resume_answer of
  ``None`` after a wake is treated as a timeout, never as approval.
* The ``ResponseSink`` protocol is how the worker offers an interrupt to the
  user: :class:`TerminalSink` waits only on the IPC-driven dialogue the
  dispatch loop already published; :class:`VoiceSink` routes to the voice
  loop.
"""

from __future__ import annotations

from typing import Any, Protocol

__all__ = [
    "CONFIRMATION_TIMEOUT_S",
    "ResponseSink",
    "TaskRuntime",
    "TerminalSink",
    "VoiceSink",
    "timeout_answer",
]

#: The single window (seconds) allowed for the user to answer a confirmation
#: or clarification prompt before the graph is resumed with a timed-out
#: refusal (D1).  Owned here alone; server.py must not define a second one.
CONFIRMATION_TIMEOUT_S = 30.0


def timeout_answer(payload: dict[str, Any]) -> dict[str, Any]:
    """The resume value that refuses a pending action after the window elapsed.

    Carries the payload's ``action_hash`` back so the policy gate can tell a
    timeout refusal from a deliberate "no" and report it honestly.  Clarification
    payloads carry no hash; the worker substitutes an empty string for them.
    """
    return {
        "approved": False,
        "action_hash": payload.get("action_hash") or "",
        "timed_out": True,
    }


class ResponseSink(Protocol):
    """Where a pending interrupt is offered to the user and its answer awaited."""

    def confirm(self, slot: Any, payload: dict[str, Any]) -> dict[str, Any]: ...

    def clarify(self, slot: Any, payload: dict[str, Any]) -> dict[str, Any]: ...


class TaskRuntime:
    """Owns the user-answer window and the resume-answer hand-off.

    ``wait_for_response`` is the only place the worker blocks on a slot's
    event; both the worker thread and the stale-confirmation sweeper drain
    through it so the window is single-sourced.
    """

    def __init__(self, confirm_timeout_s: float = CONFIRMATION_TIMEOUT_S) -> None:
        self.confirm_timeout_s = max(confirm_timeout_s, 0.0)

    def wait_for_response(self, slot: Any, payload: dict[str, Any]) -> dict[str, Any]:
        """Wait on ``slot.event`` and return the user's resume answer.

        The caller must clear ``slot.event`` *before* signalling the dispatch
        loop (and must not clear it again afterwards), so an answer that
        arrives between the signal and this call can never be lost.  A
        response the responder stashed on ``slot.resume_answer`` is returned
        once and cleared.  If the window elapses without an answer — or the
        responder stashed nothing — a ``timeout_answer`` refusal is returned.
        """
        if not slot.event.wait(timeout=self.confirm_timeout_s):
            return timeout_answer(payload)
        answer = slot.resume_answer
        slot.resume_answer = None
        if answer is None:
            return timeout_answer(payload)
        return answer


class TerminalSink:
    """Answers pending interrupts via the slot and the IPC-driven dialogue.

    Publishing is not the sink's job: the worker hands the payload to the
    dispatch loop (``slot.confirm_payload``) *before* running the sink, and
    the loop forwards it to the owner connection.  The sink only waits on the
    slot for the reply/resume value, so the window is exactly the runtime's.
    """

    def __init__(self, runtime: TaskRuntime) -> None:
        self._runtime = runtime

    def confirm(self, slot: Any, payload: dict[str, Any]) -> dict[str, Any]:
        return self._runtime.wait_for_response(slot, payload)

    def clarify(self, slot: Any, payload: dict[str, Any]) -> dict[str, Any]:
        return self._runtime.wait_for_response(slot, payload)


class VoiceSink:
    """Routes confirmations/clarifications through the active voice loop.

    Both runner methods block until the loop has finished the dialogue: a
    confirmation is a Tier-1 yes/no with a wake re-arm (the loop callback
    stashes the answer on the slot and wakes its event), and a clarification
    captures free text.  The answer is read once, right after the runner
    returns — there is no second wait, because the runner is already
    synchronous.  A missing answer fails closed as a timed-out refusal.
    """

    def __init__(self, runtime: TaskRuntime, confirm: Any, clarify: Any) -> None:
        self._runtime = runtime
        self._run_confirmation = confirm
        self._run_clarification = clarify

    def _take_answer(self, slot: Any, payload: dict[str, Any]) -> dict[str, Any]:
        answer = slot.resume_answer
        slot.resume_answer = None
        slot.event.clear()
        if answer is None:
            return timeout_answer(payload)
        return answer

    def confirm(self, slot: Any, payload: dict[str, Any]) -> dict[str, Any]:
        self._run_confirmation(slot, payload)
        return self._take_answer(slot, payload)

    def clarify(self, slot: Any, payload: dict[str, Any]) -> dict[str, Any]:
        self._run_clarification(slot, payload)
        return self._take_answer(slot, payload)
