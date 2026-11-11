"""Browser specialist loop (flagged)."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlsplit

from jarvis.agent.context import AppContext
from jarvis.agent.nodes.brain import _llm_halted_reason
from jarvis.agent.nodes.telemetry import llm_attempt_meta
from jarvis.agent.prompts import specialist_system_prompt, specialist_user_prompt
from jarvis.agent.schemas import SpecialistDecision
from jarvis.agent.state import AgentState, Plan, Step
from jarvis.browser.context import BrowserSession
from jarvis.browser.tools import _current_page_url, _session_scope

logger = logging.getLogger(__name__)

BROWSER_TOOL_NAMES = {
    "browser_goto",
    "browser_read",
    "browser_click",
    "browser_type",
    "browser_open_site",
    "browser_find",
    "browser_open_result",
    "browser_search",
    "browser_press_key",
    "browser_scroll",
    "browser_wait",
    "browser_back",
    "browser_forward",
    "browser_tabs",
    "browser_screenshot",
}


def _get_session(ctx: AppContext) -> BrowserSession | None:
    browser = getattr(ctx, "browser", None)
    if browser is None:
        return None
    if hasattr(browser, "session") and browser.session is not None:
        return browser.session
    return browser


def _build_digest(state: AgentState, ctx: AppContext) -> str:
    session = _get_session(ctx)
    parts = []
    if session is not None:
        try:
            url = _current_page_url(session)
            if url:
                parts.append(f"URL: {url}")
        except Exception:  # a digest failure must never stop the loop
            logger.debug("specialist digest: page url unavailable", exc_info=True)
        try:
            # try to read
            if hasattr(session, "read"):
                obs = session.read(max_elements=25)
                parts.append(f"Title: {getattr(obs, 'title', '')}")
                elems = []
                for e in list(getattr(obs, "elements", []))[:25]:
                    name = str(getattr(e, "name", ""))[:80]
                    role = str(getattr(e, "role", ""))
                    ref = str(getattr(e, "ref", ""))
                    elems.append(f"- {ref} [{role}] {name}")
                if elems:
                    parts.append("Elements:\n" + "\n".join(elems))
        except Exception:  # a digest failure must never stop the loop
            logger.debug("specialist digest: page elements unavailable", exc_info=True)
    # result set
    try:
        store = getattr(ctx, "browser_results", None)
        if store is not None and session is not None:
            scope = _session_scope(session)
            rs = store.get(scope)
            if rs is not None:
                items = []
                for it in list(rs.items)[:8]:
                    items.append(f"- {it.position}: {str(it.title)[:80]} ({it.url})")
                if items:
                    parts.append("Results:\n" + "\n".join(items))
    except Exception:  # a digest failure must never stop the loop
        logger.debug("specialist digest: result set unavailable", exc_info=True)
    digest = "\n\n".join(parts)
    if len(digest) > 2500:
        digest = digest[:2500]
    # wrap as untrusted
    return f"<untrusted_data>\n{digest}\n</untrusted_data>"


def _args_summary(args: dict[str, Any]) -> str:
    try:
        import json

        s = json.dumps(args)
    except Exception:  # noqa: BLE001 - fall back to str() for args that cannot be serialised
        s = str(args)
    if len(s) > 200:
        s = s[:200]
    return s


def _trim_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for h in history[-8:]:
        out.append(
            {
                "tool": str(h.get("tool", ""))[:200],
                "args-summary": str(h.get("args-summary", ""))[:200],
                "ok": bool(h.get("ok")),
                "note": str(h.get("note", ""))[:200],
            }
        )
    return out


def decide(state: AgentState, ctx: AppContext) -> dict[str, Any]:
    goal = state.get("specialist_goal") or state.get("goal") or ""
    max_calls = (
        getattr(getattr(ctx, "settings", None).agent, "specialist_max_calls", 12)
        if hasattr(getattr(ctx, "settings", None), "agent")
        else 12
    )
    calls = int(state.get("specialist_calls") or 0)
    if calls >= max_calls:
        return {
            "halted_reason": "Specialist call limit reached.",
            "final_answer": "I couldn't complete the browser task within the step limit.",
            "specialist_phase": "idle",
        }
    history = _trim_history(list(state.get("specialist_history") or []))
    if history:
        # repeat guard: same (tool,args-summary) twice in a row
        last = history[-1]
        prev = history[-2] if len(history) >= 2 else None
        if (
            prev is not None
            and prev.get("tool") == last.get("tool")
            and prev.get("args-summary") == last.get("args-summary")
        ):
            return {
                "halted_reason": "Specialist stuck in a loop.",
                "final_answer": "I got stuck repeating the same action.",
                "specialist_phase": "idle",
            }
    digest = _build_digest(state, ctx)
    # build allowed tools list (browser only)
    allowed = []
    try:
        reg = ctx.registry
        for name in sorted(BROWSER_TOOL_NAMES):
            spec = reg.get(name)
            if spec is None:
                continue
            allowed.append(
                {
                    "name": name,
                    "description": spec.description[:160],
                    "args_schema": spec.args_model.model_json_schema()
                    if hasattr(spec.args_model, "model_json_schema")
                    else {},
                }
            )
    except Exception:  # noqa: BLE001 - fall back to a bare tool list
        for name in sorted(BROWSER_TOOL_NAMES):
            allowed.append({"name": name, "description": "", "args_schema": {}})
    system = specialist_system_prompt(allowed)
    user = specialist_user_prompt(goal, history, digest)
    timeout_s = min(
        getattr(getattr(ctx, "settings", None).llm, "timeout_seconds", 30.0),
        getattr(getattr(ctx, "settings", None).agent, "specialist_call_timeout_s", 12.0),
    )
    # B8.5: every specialist turn is one real provider attempt on this task.
    attempt_meta = llm_attempt_meta(state, ctx, "action")
    attempt_budget = {"attempt_meta": attempt_meta} if attempt_meta is not None else {}
    try:
        decision, _usage = ctx.llm.structured(
            system=system,
            user=user,
            schema=SpecialistDecision,
            model_role="planner",
            temperature=0.0,
            max_retries=0,
            timeout_s=timeout_s,
            **attempt_budget,
        )
    except Exception as exc:  # noqa: BLE001 - an LLM failure stops this task only
        reason = _llm_halted_reason(exc)
        return {
            "halted_reason": reason,
            "final_answer": "I couldn't proceed with the browser task.",
            "specialist_phase": "idle",
        }
    if decision.kind == "done":
        summary = (decision.summary or "").strip() or "Done."
        return {"final_answer": summary, "specialist_phase": "idle", "halted_reason": None}
    if decision.kind == "give_up":
        reason = (decision.summary or "").strip() or "Unable to proceed."
        return {
            "final_answer": f"I couldn't complete that: {reason}",
            "specialist_phase": "idle",
            "halted_reason": None,
        }
    if decision.kind == "act":
        tool = (decision.tool or "").strip()
        if tool not in BROWSER_TOOL_NAMES:
            return {
                "halted_reason": "Specialist proposed non-browser tool.",
                "final_answer": "I can only use browser tools for this task.",
                "specialist_phase": "idle",
            }
        args = decision.args or {}
        # One step id per executed decision: ids must stay unique for the whole
        # task, because act's exactly-once guard skips a step whose id already
        # has a successful result.  ``calls`` increments once per act decision.
        step = Step(
            id=f"s{calls}",
            tool=tool,
            args=args,
            rationale=(decision.rationale or "")[:200],
            expect=(decision.expect or "")[:200],
        )
        plan = Plan(kind="tool", goal=goal or decision.rationale or "Browse", steps=[step])
        return {
            "plan": plan,
            "step_index": 0,
            "decisions": {},
            "specialist_phase": "observe",
            "specialist_calls": calls + 1,
        }
    return {
        "halted_reason": "Invalid specialist decision.",
        "final_answer": "I couldn't proceed.",
        "specialist_phase": "idle",
    }


def _record_history(state: AgentState) -> list[dict[str, Any]]:
    """Append the just-finished step's outcome to ``specialist_history``.

    The next decision must see whether the previous step worked (and why not),
    so a failed or unverified step is recorded here rather than being retried
    blindly.  Only a result that belongs to the plan's last step is recorded,
    so re-entering observe twice never duplicates an entry.
    """
    history = list(state.get("specialist_history") or [])
    plan = state.get("plan")
    results = list(state.get("results") or [])
    steps = list(getattr(plan, "steps", None) or [])
    if not steps or not results:
        return history
    step = steps[-1]
    result = results[-1]
    if str(getattr(result, "step_id", "")) != str(getattr(step, "id", "")):
        return history
    note = str(result.verify_note or result.error or "")
    if not result.ok and not note:
        note = "the step did not succeed"
    if result.ok and result.verified is False and not note:
        note = "the step could not be verified"
    role, host = _typed_target_marker(step, result)
    history.append(
        {
            "tool": str(getattr(step, "tool", "") or ""),
            "args-summary": _args_summary(getattr(step, "args", None) or {}),
            "ok": bool(result.ok and result.verified is not False),
            "note": note[:200],
            # Compact, non-sensitive marker of the element this step typed into
            # (empty for everything else).  ``browse_grant_check`` needs it to
            # decide whether a following Enter may be covered by the grant: the
            # role used to be lost here, so Enter always forced a prompt.  Only
            # a two-value role and a host are kept - never the element's name,
            # label, ref or the typed text.  Plain strings: no new state type.
            "role": role,
            "host": host,
        }
    )
    return history


#: Roles a typing step must have had for a following Enter to be coverable.
#: Mirrors the pure rule's ``_ENTER_ROLES``; a password field is never here.
_SEARCH_ROLES = frozenset({"searchbox", "combobox"})


def _typed_target_marker(step: Any, result: Any) -> tuple[str, str]:
    """``(role, host)`` for a successful type into a search field, else ``("", "")``.

    The element's role and origin are all that survives from the browser's
    answer: the tool's own ``data`` also carries page-controlled text (its name
    and label), which must never be copied into state.  Anything unexpected -
    a failed type, a field that is not a searchbox/combobox (so never a password
    or submit control), an unparseable origin - yields an empty marker, which
    makes the Enter rule fail closed.
    """
    if str(getattr(step, "tool", "") or "") != "browser_type":
        return "", ""
    if not bool(getattr(result, "ok", False)):
        return "", ""
    data = getattr(result, "data", None)
    if not isinstance(data, dict):
        return "", ""
    role = " ".join(str(data.get("role") or "").casefold().split())
    if role not in _SEARCH_ROLES:
        return "", ""
    try:
        host = (urlsplit(str(data.get("origin") or "")).hostname or "").lower().rstrip(".")
    except ValueError:  # fail closed: an unparseable origin is no host
        return "", ""
    if not host or len(host) > 253:
        return "", ""
    return role, host


def observe(state: AgentState, ctx: AppContext) -> dict[str, Any]:
    """Record the finished step, then either refresh the page or hand back."""
    history = _record_history(state)
    calls = int(state.get("specialist_calls") or 0)
    plan = state.get("plan")
    if plan is None:
        return {"specialist_phase": "decide", "specialist_history": history}
    # check last step
    try:
        steps = plan.steps if hasattr(plan, "steps") else plan.get("steps", [])
        if steps:
            last = steps[-1]
            last_tool = last.get("tool") if isinstance(last, dict) else getattr(last, "tool", None)
            if last_tool in {"browser_read", "browser_find", "browser_search"}:
                return {"specialist_phase": "decide", "specialist_history": history}
    except Exception:  # a malformed plan must not break the loop
        logger.debug("specialist observe: could not inspect the last step", exc_info=True)
    # Build a page-refresh step for act.  The phase deliberately stays
    # "observe" so verify routes the refresh back here; ``r<calls>`` keeps the
    # id out of the decision ids (``s<calls>``) and unique per iteration.
    step = Step(
        id=f"r{calls}",
        tool="browser_read",
        args={},
        rationale="Refresh page",
        expect="See current page",
    )
    new_plan = Plan(
        kind="tool",
        goal=state.get("specialist_goal") or state.get("goal") or "Browse",
        steps=[step],
    )
    return {
        "plan": new_plan,
        "step_index": 0,
        "decisions": {},
        "specialist_phase": "observe",
        "specialist_history": history,
    }


__all__ = ["BROWSER_TOOL_NAMES", "decide", "observe"]
