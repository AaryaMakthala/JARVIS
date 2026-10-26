"""A per-task cancel hook for LLM calls (no call-site plumbing).

The user's cancel token belongs to one task, while the LLM client is shared by
every task.  A :class:`~contextvars.ContextVar` carries a zero-argument
``is_cancelled`` callable for the duration of one graph node, so the client can
stop *between* attempts and backoff sleeps without any of the ``ctx.llm`` call
sites passing it down.  Unset - the default - means "nothing was cancelled",
i.e. behaviour is exactly what it was before this module existed.
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from jarvis.tools.base import Cancelled

__all__ = ["abort_if_cancelled", "cancelled_by", "current_check"]

_CURRENT: contextvars.ContextVar[Callable[[], bool] | None] = contextvars.ContextVar(
    "jarvis_cancel_check", default=None
)


def current_check() -> Callable[[], bool] | None:
    """The active task's ``is_cancelled`` callable, or ``None`` outside a node."""
    return _CURRENT.get()


def abort_if_cancelled() -> None:
    """Raise :class:`Cancelled` when the active task has been cancelled."""
    check = _CURRENT.get()
    if check is not None and check():
        raise Cancelled()


@contextmanager
def cancelled_by(check: Callable[[], bool] | None) -> Iterator[None]:
    """Install ``check`` as the current task's cancel hook for this block.

    Always resets on exit (including on an exception), so one task's hook can
    never leak into the next task running in the same thread.
    """
    token = _CURRENT.set(check)
    try:
        yield
    finally:
        _CURRENT.reset(token)
