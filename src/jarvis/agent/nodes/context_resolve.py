"""Deterministic context resolver (no LLM)."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict

from jarvis.agent import referents
from jarvis.agent.referents import REFERENT_STORE


class ResolvedRefsState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    has_result_set: bool = False
    result_set_positions: list[int] = []
    has_page: bool = False
    page_url: str | None = None
    has_photo: bool = False
    photo_path: str | None = None
    has_file: bool = False
    file_path: str | None = None
    resolved_action: str | None = None  # e.g. open_first_result, open_picture


_ORDINALS = {
    "first": 1,
    "1st": 1,
    "second": 2,
    "2nd": 2,
    "third": 3,
    "3rd": 3,
    "fourth": 4,
    "4th": 4,
    "fifth": 5,
    "5th": 5,
}


def _extract_ordinal(text: str) -> int | None:
    t = text.lower()
    for k, v in _ORDINALS.items():
        if re.search(r"\b" + re.escape(k) + r"\b", t):
            return v
    m = re.search(r"\b(\d+)(?:st|nd|rd|th)?\b", t)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def context_resolve(state: dict[str, Any], ctx: Any) -> dict[str, Any]:
    user_input = str(state.get("user_input") or "")
    if not user_input.strip():
        return {"resolved_refs": ResolvedRefsState().model_dump()}
    cid = str(state.get("conversation_id") or "local")
    refs = REFERENT_STORE.get(cid)
    resolved = ResolvedRefsState()
    if refs.result_set and refs.result_set.items:
        resolved.has_result_set = True
        resolved.result_set_positions = [it.position for it in refs.result_set.items]
    if refs.page:
        resolved.has_page = True
        resolved.page_url = refs.page.url
    if refs.last_photo:
        resolved.has_photo = True
        resolved.photo_path = refs.last_photo.path
    if refs.last_file and not resolved.has_photo:
        resolved.has_file = True
        resolved.file_path = refs.last_file.path
    t = user_input.lower()
    # "do that again" - repeat last action if Tier 0 or 1 stored
    if re.search(r"\bdo that again\b", t):
        if refs.last_action and refs.last_action.ok:
            resolved.resolved_action = f"repeat:{refs.last_action.tool}"
    # result set
    if re.search(r"\b(first|second|third|fourth|fifth|\d+(?:st|nd|rd|th)?)\b.*\b(link|result|one)\b", t) or re.search(r"\bopen the (first|second|third)\b", t):
        pos = _extract_ordinal(t)
        if pos and resolved.has_result_set and pos in resolved.result_set_positions:
            resolved.resolved_action = f"open_result:{pos}"
    # picture/photo
    if re.search(r"\b(the\s+)?(picture|photo|screenshot)\b", t) and ("open" in t or "show" in t or resolved.resolved_action is None):
        if resolved.has_photo:
            resolved.resolved_action = "open_picture"
        elif resolved.has_file:
            resolved.resolved_action = "open_file"
    return {"resolved_refs": resolved.model_dump()}
