"""get_time tool: the current local time and/or date on this PC (Tier 0).

Read-only; builds the sentence from :func:`local_now`'s ``datetime`` attributes,
never from ``strftime`` flags such as ``%-I`` (not portable on Windows).  It
reports *this machine's* local clock only - there is no time-zone argument, and
the description says so.  A clock reading has no observable post-condition, so
``verify()`` returns ``None`` (not a failure).
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from jarvis.tools.base import ToolContext, ToolResult, ToolSpec


def local_now() -> datetime:
    """Return the current local time (injectable so tests are deterministic)."""
    return datetime.now().astimezone()


def _day_phrase(hour: int) -> str:
    """Day-part phrase for a 24-hour ``hour`` (noon/midnight handled by the caller).

    00-04 at night, 05-11 morning, 12-16 afternoon, 17-20 evening, 21-23 at night.
    """
    if hour < 5:
        return "at night"
    if hour < 12:
        return "in the morning"
    if hour < 17:
        return "in the afternoon"
    if hour < 21:
        return "in the evening"
    return "at night"


def spoken_date(moment: datetime) -> str:
    """Render ``moment`` as a natural date sentence ("Today is Tuesday, ...")."""
    return f"Today is {moment.strftime('%A')}, {moment.day} {moment.strftime('%B %Y')}."


def spoken_datetime(moment: datetime) -> str:
    """Render ``moment`` as a combined date and time sentence."""
    time_sentence = spoken_time(moment)
    return f"{time_sentence[:-1]} on {spoken_date(moment)[len('Today is ') :]}"


def spoken_time(moment: datetime) -> str:
    """Render ``moment`` as a natural sentence ("It is 3:45 in the afternoon.").

    Minutes 1-9 are zero-padded ("9:05"); on the hour the form is
    "3 o'clock <day part>".  Exactly 12:00 is noon, exactly 00:00 is midnight.
    """
    hour = moment.hour
    minute = moment.minute
    hour12 = hour % 12 or 12
    if hour == 12 and minute == 0:
        return "It is 12 o'clock noon."
    if hour == 0 and minute == 0:
        return "It is 12 o'clock midnight."
    if minute == 0:
        return f"It is {hour12} o'clock {_day_phrase(hour)}."
    return f"It is {hour12}:{minute:02d} {_day_phrase(hour)}."


class GetTimeArgs(BaseModel):
    """Which parts of this PC's local clock the caller wants."""

    model_config = ConfigDict(extra="forbid")

    include_date: bool = False
    include_time: bool = False


def _run_get_time(args: GetTimeArgs, ctx: ToolContext) -> ToolResult:
    if ctx.dry_run:
        return ToolResult(ok=True, output="[dry-run] would read the current local clock")
    moment = local_now()
    want_date = args.include_date
    want_time = args.include_time or not want_date
    if want_date and want_time:
        output = spoken_datetime(moment)
    elif want_date:
        output = spoken_date(moment)
    else:
        output = spoken_time(moment)
    return ToolResult(
        ok=True,
        output=output,
        data={
            "hour": moment.hour,
            "minute": moment.minute,
            "year": moment.year,
            "month": moment.month,
            "day": moment.day,
        },
    )


def _verify_get_time(args: GetTimeArgs, result: ToolResult, ctx: ToolContext) -> ToolResult:
    """A clock reading has no observable post-condition: ``verified=None``."""
    del args, ctx
    return result.model_copy(update={"verified": None})


def make_get_time_spec() -> ToolSpec:
    """Build the ``get_time`` tool (Tier 0, works on any OS)."""
    return ToolSpec(
        name="get_time",
        description=("Current local time and/or date on this PC. Cannot report other time zones."),
        args_model=GetTimeArgs,
        base_tier=0,
        timeout_s=5,
        run=_run_get_time,
        verify=_verify_get_time,
        describe=lambda args: _describe(args),
    )


def _describe(args: GetTimeArgs) -> str:
    if args.include_date and args.include_time:
        return "tell you the date and time"
    if args.include_date:
        return "tell you the date"
    return "tell you the time"
