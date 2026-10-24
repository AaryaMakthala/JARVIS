"""Referent store for conversation-scoped entities (in-memory, never checkpointed)."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ResultItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    position: int
    title: str
    host: str
    url: str
    ref: str | None = None


class ResultSetRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    page_url: str
    items: list[ResultItem] = Field(default_factory=list)
    created_at: float


class PageRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    title: str | None = None
    origin: str | None = None
    created_at: float


class FileRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    kind: Literal["photo", "file", "screenshot", "other"] = "file"
    created_at: float


class ContactRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_initials: str | None = None
    contact_id: str | None = None
    created_at: float


class ActionRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str
    ok: bool = True
    created_at: float
    args: dict | None = None


class ResolvedRefs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result_set: ResultSetRef | None = None
    page: PageRef | None = None
    last_file: FileRef | None = None
    last_photo: FileRef | None = None
    last_contact: ContactRef | None = None
    last_action: ActionRef | None = None


class ReferentStore:
    def __init__(self, ttl_s: float = 600.0, max_one_per_kind: bool = True):
        self.ttl_s = float(ttl_s)
        self.max_one_per_kind = bool(max_one_per_kind)
        self._lock = threading.RLock()
        self._data: dict[str, ResolvedRefs] = {}

    def get(self, conversation_id: str) -> ResolvedRefs:
        with self._lock:
            refs = self._data.get(conversation_id)
            if refs is None:
                refs = ResolvedRefs()
                self._data[conversation_id] = refs
            self._prune(refs)
            return refs.model_copy(deep=True)

    def set_result_set(self, conversation_id: str, ref: ResultSetRef) -> None:
        with self._lock:
            refs = self._data.setdefault(conversation_id, ResolvedRefs())
            self._prune(refs)
            refs.result_set = ref

    def set_page(self, conversation_id: str, ref: PageRef) -> None:
        with self._lock:
            refs = self._data.setdefault(conversation_id, ResolvedRefs())
            self._prune(refs)
            refs.page = ref

    def set_file(self, conversation_id: str, ref: FileRef) -> None:
        with self._lock:
            refs = self._data.setdefault(conversation_id, ResolvedRefs())
            self._prune(refs)
            if ref.kind == "photo":
                refs.last_photo = ref
            refs.last_file = ref

    def set_contact(self, conversation_id: str, ref: ContactRef) -> None:
        with self._lock:
            refs = self._data.setdefault(conversation_id, ResolvedRefs())
            self._prune(refs)
            refs.last_contact = ref

    def set_action(self, conversation_id: str, ref: ActionRef) -> None:
        with self._lock:
            refs = self._data.setdefault(conversation_id, ResolvedRefs())
            self._prune(refs)
            refs.last_action = ref

    def clear_page(self, conversation_id: str) -> None:
        with self._lock:
            refs = self._data.setdefault(conversation_id, ResolvedRefs())
            refs.page = None

    def _prune(self, refs: ResolvedRefs) -> None:
        now = time.time()
        if refs.result_set and now - refs.result_set.created_at > self.ttl_s:
            refs.result_set = None
        if refs.page and now - refs.page.created_at > self.ttl_s:
            refs.page = None
        if refs.last_file and now - refs.last_file.created_at > self.ttl_s:
            refs.last_file = None
        if refs.last_photo and now - refs.last_photo.created_at > self.ttl_s:
            refs.last_photo = None
        if refs.last_contact and now - refs.last_contact.created_at > self.ttl_s:
            refs.last_contact = None
        if refs.last_action and now - refs.last_action.created_at > self.ttl_s:
            refs.last_action = None


# global store for process
REFERENT_STORE = ReferentStore(ttl_s=600.0)


def now_ts() -> float:
    return time.time()


def origin_from_url(url: str) -> str | None:
    try:
        from urllib.parse import urlparse

        p = urlparse(url)
        if p.scheme and p.netloc:
            return f"{p.scheme}://{p.netloc}"
        return None
    except Exception:
        return None
