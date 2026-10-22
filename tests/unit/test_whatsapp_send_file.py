"""Stage 8.4c: the guarded WhatsApp file-send and chat-open tools.

No browser, no Playwright, no network.  The real
:class:`~jarvis.browser.whatsapp_web.WhatsAppDriver` runs against a fake narrow
page, and the tool is given a fake contact store, rate limiter and resend
guard, so every refusal is deterministic.

What these tests pin down:

* ``whatsapp_send_file`` is Tier 2, costs one budget unit, declares ``path`` as
  a policy path argument and has no caption/message argument at all;
* sending stays disabled until the content-hash approval binding is wired
  (Slice 8.4d): no approved digest on the context means no send;
* the approved bytes must still be the bytes on disk immediately before Send;
* a retry/replan cannot send the same file to the same contact twice;
* ``whatsapp_open_chat`` only opens a verified chat and never attaches;
* an ambiguous contact is refused with the options (the brain asks the
  question -- a clarification answer keeps the intended send).
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from jarvis import config
from jarvis.agent.context import make_app_context
from jarvis.agent.graph import open_sqlite_checkpointer
from jarvis.agent.runner import resume_task, run_task
from jarvis.browser.whatsapp_web import (
    WHATSAPP_BASE_URL,
    WHATSAPP_SEND_URL,
    AttachmentCandidate,
    AttachmentScan,
    SearchHit,
    SendControl,
)
from jarvis.config import Settings
from jarvis.llm.client import FakeLLM
from jarvis.tools.base import ToolContext, ToolResult, ToolSpec
from jarvis.tools.registry import build_default_registry
from jarvis.tools.whatsapp import ContactStore, RateLimiter
from jarvis.tools.whatsapp_web import (
    APPROVED_HASHES_ATTR,
    FileSendGuard,
    WhatsAppOpenChatArgs,
    WhatsAppSendFileArgs,
    make_whatsapp_open_chat_spec,
    make_whatsapp_send_file_spec,
)
from support import brain_action, brain_clarify, registry_with

_PHONE = "919876543210"


@pytest.fixture(autouse=True)
def _windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test runs as if on Windows unless it overrides this."""
    monkeypatch.setattr("jarvis.platform_guard.is_windows", lambda: True)


class _Clock:
    """A deterministic clock so the resend window never depends on wall time."""

    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _scan(name: str) -> AttachmentScan:
    """A scan whose file-name evidence sits inside the composer."""
    return AttachmentScan(
        panel_found=True,
        composer=(AttachmentCandidate(scope="composer", source="aria-label", text=name),),
        panel=(),
        composer_container_below_panel=True,
    )


class _Page:
    """Fake narrow WhatsApp page (records every action the driver took)."""

    def __init__(
        self,
        *,
        header: str = "Rahul",
        scan_override: AttachmentScan | None = None,
        controls: tuple[SendControl, ...] = (SendControl(name="Send", index=0),),
        on_stage: Any = None,
    ) -> None:
        self.current_url = WHATSAPP_BASE_URL
        self.header = header
        self.scan_override = scan_override
        self.controls = controls
        self.on_stage = on_stage
        self.visited: list[str] = []
        self.staged_paths: list[str] = []
        self.staged: tuple[str, ...] = ()
        self.clicked: list[str] = []

    def url(self) -> str:
        return self.current_url

    def goto(self, url: str) -> None:
        self.visited.append(url)
        self.current_url = url

    def chat_header(self) -> str:
        return self.header

    def fill_search(self, query: str) -> None:
        del query

    def search_results(self) -> tuple[SearchHit, ...]:
        return ()

    def open_search_result(self, hit: SearchHit) -> None:
        del hit

    def set_input_files(self, path: str) -> None:
        self.staged_paths.append(path)
        self.staged = (Path(path).name,)
        if self.on_stage is not None:
            self.on_stage()

    def staged_file_names(self) -> tuple[str, ...]:
        return self.staged

    def attachment_scan(self, filename: str) -> AttachmentScan:
        return self.scan_override if self.scan_override is not None else _scan(filename)

    def send_controls(self) -> tuple[SendControl, ...]:
        return self.controls

    def click_send(self, name: str) -> None:
        self.clicked.append(name)


class _Session:
    def __init__(self, page: _Page) -> None:
        self._page = page

    def whatsapp_page(self) -> _Page:
        return self._page


class _Holder:
    """Stands in for ``ToolContext.whatsapp_browser`` (a LazyProfileBrowser)."""

    def __init__(self, page: _Page | None) -> None:
        self.page = page
        self.acquires = 0

    def acquire(self) -> _Session | None:
        self.acquires += 1
        return _Session(self.page) if self.page is not None else None


def _store(tmp_path: Path, *contacts: tuple[str, str]) -> ContactStore:
    store = ContactStore(tmp_path / "contacts.json")
    known = {contact.name.lower() for contact in store.load()}
    for name, number in contacts or (("Rahul", _PHONE),):
        if name.lower() not in known:
            store.add(name, number)
    return store


def _file(tmp_path: Path, name: str = "photo.jpg", data: bytes = b"approved bytes") -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def _ctx(holder: _Holder | None, *, max_file_bytes: int | None = None, dry_run: bool = False):
    settings = (
        config.Settings(whatsapp={"max_file_bytes": max_file_bytes})
        if max_file_bytes is not None
        else Settings()
    )
    return ToolContext(settings=settings, whatsapp_browser=holder, dry_run=dry_run)


def _limiter(tmp_path: Path) -> RateLimiter:
    return RateLimiter(tmp_path / "ratelimit.json", max_per_hour=10, min_interval_s=0.0)


def _guard(tmp_path: Path, clock: _Clock | None = None) -> FileSendGuard:
    return FileSendGuard(tmp_path / "guard.json", clock=clock or _Clock())


def _approve(ctx: ToolContext, path: Path, *, data: bytes | None = None) -> str:
    payload = path.read_bytes() if data is None else data
    digest = hashlib.sha256(payload).hexdigest()
    setattr(ctx, APPROVED_HASHES_ATTR, (digest,))
    return digest


def _spec(tmp_path: Path, *, guard: FileSendGuard | None = None, store: ContactStore | None = None):
    return make_whatsapp_send_file_spec(
        store=store or _store(tmp_path),
        limiter=_limiter(tmp_path),
        guard=guard or _guard(tmp_path),
    )


# ── spec / registration ───────────────────────────────────────────────────


def test_the_args_model_has_only_contact_and_path() -> None:
    fields = set(WhatsAppSendFileArgs.model_fields)
    assert fields == {"contact", "path"}
    assert "message" not in fields
    assert "caption" not in fields


def test_registered_as_tier2_with_budget_and_path_binding() -> None:
    registry = build_default_registry()
    spec = registry.get_optional("whatsapp_send_file")
    assert spec is not None
    assert spec.base_tier == 2
    assert spec.windows_only is True
    assert spec.budget_cost == 1
    assert spec.path_args == ("path",)
    assert spec.bind_content_hash is True
    assert "whatsapp_open_chat" in registry.names()


def test_open_chat_is_tier1_navigation_only() -> None:
    spec = build_default_registry().get_optional("whatsapp_open_chat")
    assert spec is not None
    assert spec.base_tier == 1
    assert spec.budget_cost == 1
    assert set(spec.args_model.model_fields) == {"contact"}


def test_config_defaults_include_the_file_cap() -> None:
    assert config.Settings().whatsapp.max_file_bytes == 16 * 1024 * 1024
    assert config.Settings(whatsapp={"max_file_bytes": 1024}).whatsapp.max_file_bytes == 1024


# ── send_file happy path and the binding ─────────────────────────────────


def test_send_file_happy_path(tmp_path: Path) -> None:
    target = _file(tmp_path)
    page = _Page()
    holder = _Holder(page)
    ctx = _ctx(holder)
    _approve(ctx, target)

    result = _spec(tmp_path).execute(WhatsAppSendFileArgs(contact="Rahul", path=str(target)), ctx)

    assert result.ok, result.error
    assert result.verified is True
    assert page.visited == [WHATSAPP_SEND_URL.format(digits=_PHONE)]
    assert page.staged_paths == [str(target)]
    assert page.clicked == ["Send"]
    assert result.data["file"] == "photo.jpg"
    assert result.data["chat_confirmed"] is True
    assert result.data["content_sha256"] == hashlib.sha256(b"approved bytes").hexdigest()
    assert "Rahul" in result.output
    assert _PHONE not in result.output


def test_send_is_refused_without_an_approved_content_binding(tmp_path: Path) -> None:
    target = _file(tmp_path)
    page = _Page()
    holder = _Holder(page)

    result = _spec(tmp_path).execute(
        WhatsAppSendFileArgs(contact="Rahul", path=str(target)), _ctx(holder)
    )

    assert result.ok is False
    assert "not bound" in result.error
    assert holder.acquires == 0
    assert page.staged_paths == [] and page.clicked == []


def test_a_different_file_than_the_approved_one_is_refused(tmp_path: Path) -> None:
    target = _file(tmp_path)
    page = _Page()
    ctx = _ctx(_Holder(page))
    _approve(ctx, target, data=b"some other content")

    result = _spec(tmp_path).execute(WhatsAppSendFileArgs(contact="Rahul", path=str(target)), ctx)

    assert result.ok is False
    assert "does not match the approved file" in result.error
    assert page.staged_paths == [] and page.clicked == []


def test_a_file_that_changes_while_attaching_is_not_sent(tmp_path: Path) -> None:
    target = _file(tmp_path, data=b"original")
    ctx = _ctx(_Holder(None))
    _approve(ctx, target)

    page = _Page(on_stage=lambda: target.write_bytes(b"tampered"))
    ctx.whatsapp_browser = _Holder(page)
    result = _spec(tmp_path).execute(WhatsAppSendFileArgs(contact="Rahul", path=str(target)), ctx)

    assert result.ok is False
    assert "changed while the attachment was being prepared" in result.error
    assert page.clicked == []


def test_send_file_is_refused_before_the_browser_when_rate_limited(tmp_path: Path) -> None:
    target = _file(tmp_path)
    holder = _Holder(_Page())
    ctx = _ctx(holder)
    _approve(ctx, target)
    limiter = RateLimiter(tmp_path / "rl.json", max_per_hour=1, min_interval_s=0.0)
    limiter.record()
    spec = make_whatsapp_send_file_spec(
        store=_store(tmp_path), limiter=limiter, guard=_guard(tmp_path)
    )

    result = spec.execute(WhatsAppSendFileArgs(contact="Rahul", path=str(target)), ctx)

    assert result.ok is False
    assert "rate limit" in result.error
    assert holder.acquires == 0


def test_missing_file_and_over_cap_file_are_refused(tmp_path: Path) -> None:
    ctx = _ctx(_Holder(_Page()))
    spec = _spec(tmp_path)

    missing = spec.execute(
        WhatsAppSendFileArgs(contact="Rahul", path=str(tmp_path / "nope.jpg")), ctx
    )
    assert missing.ok is False
    assert "not a regular file" in missing.error

    target = _file(tmp_path, data=b"0123456789")
    _approve(ctx, target)
    too_big = make_whatsapp_send_file_spec(
        store=_store(tmp_path),
        limiter=_limiter(tmp_path),
        guard=_guard(tmp_path),
    ).execute(
        WhatsAppSendFileArgs(contact="Rahul", path=str(target)),
        _ctx(_Holder(_Page()), max_file_bytes=4),
    )
    assert too_big.ok is False
    assert "4-byte" in too_big.error


def test_unknown_and_ambiguous_contacts_are_refused_with_options(tmp_path: Path) -> None:
    store = _store(tmp_path, ("Ali Khan", "9112345678"), ("Ali Ahmed", _PHONE))
    spec = _spec(tmp_path, store=store)
    ctx = _ctx(_Holder(_Page()))
    target = _file(tmp_path)
    _approve(ctx, target)

    unknown = spec.execute(WhatsAppSendFileArgs(contact="Nobody", path=str(target)), ctx)
    assert unknown.ok is False
    assert "no contact matches" in unknown.error
    assert "Ali Khan" in unknown.error

    ambiguous = spec.execute(WhatsAppSendFileArgs(contact="Ali", path=str(target)), ctx)
    assert ambiguous.ok is False
    assert "multiple contacts match" in ambiguous.error
    assert "Ali Khan" in ambiguous.error and "Ali Ahmed" in ambiguous.error


def test_chat_identity_and_preview_must_be_proven(tmp_path: Path) -> None:
    target = _file(tmp_path)

    wrong_chat = _Page(header="Someone Else")
    ctx = _ctx(_Holder(wrong_chat))
    _approve(ctx, target)
    mismatch = _spec(tmp_path).execute(WhatsAppSendFileArgs(contact="Rahul", path=str(target)), ctx)
    assert mismatch.ok is False
    assert "could not open the WhatsApp chat" in mismatch.error
    assert wrong_chat.staged_paths == [] and wrong_chat.clicked == []

    unscoped = _Page(scan_override=AttachmentScan(panel_found=True))
    ctx = _ctx(_Holder(unscoped))
    _approve(ctx, target)
    no_preview = _spec(tmp_path).execute(
        WhatsAppSendFileArgs(contact="Rahul", path=str(target)), ctx
    )
    assert no_preview.ok is False
    assert "could not stage the attachment" in no_preview.error
    assert unscoped.clicked == []


def test_non_windows_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("jarvis.platform_guard.is_windows", lambda: False)
    target = _file(tmp_path)
    page = _Page()
    ctx = _ctx(_Holder(page))
    _approve(ctx, target)

    result = _spec(tmp_path).execute(WhatsAppSendFileArgs(contact="Rahul", path=str(target)), ctx)

    assert result.ok is False
    assert "only supported on Windows" in result.error
    assert page.staged_paths == [] and page.clicked == []


def test_a_retry_cannot_send_the_same_file_twice(tmp_path: Path) -> None:
    target = _file(tmp_path)
    holder = _Holder(_Page())
    ctx = _ctx(holder)
    _approve(ctx, target)
    spec = _spec(tmp_path)

    first = spec.execute(WhatsAppSendFileArgs(contact="Rahul", path=str(target)), ctx)
    second = spec.execute(WhatsAppSendFileArgs(contact="Rahul", path=str(target)), ctx)

    assert first.ok, first.error
    assert second.ok is False
    assert "already sent" in second.error
    assert holder.acquires == 1  # the duplicate is refused before the browser
    assert holder.page is not None and holder.page.clicked == ["Send"]


def test_the_duplicate_guard_expires_after_its_window(tmp_path: Path) -> None:
    target = _file(tmp_path)
    clock = _Clock()
    guard = _guard(tmp_path, clock)
    spec = _spec(tmp_path, guard=guard)
    ctx = _ctx(_Holder(_Page()))
    _approve(ctx, target)

    assert spec.execute(WhatsAppSendFileArgs(contact="Rahul", path=str(target)), ctx).ok
    clock.now += 601.0
    assert spec.execute(WhatsAppSendFileArgs(contact="Rahul", path=str(target)), ctx).ok


def test_dry_run_touches_nothing(tmp_path: Path) -> None:
    target = _file(tmp_path)
    page = _Page()
    ctx = _ctx(_Holder(page), dry_run=True)
    _approve(ctx, target)

    result = _spec(tmp_path).execute(WhatsAppSendFileArgs(contact="Rahul", path=str(target)), ctx)

    assert result.ok is True
    assert result.output.startswith("[dry-run]")
    assert page.visited == [] and page.staged_paths == [] and page.clicked == []


def test_describe_masks_the_number_and_names_the_file(tmp_path: Path) -> None:
    target = _file(tmp_path)
    summary = _spec(tmp_path).describe(WhatsAppSendFileArgs(contact="Rahul", path=str(target)))
    assert "photo.jpg" in summary
    assert "Rahul" in summary
    assert "\u2022" * 8 + "10" in summary
    assert _PHONE not in summary


# ── open_chat ─────────────────────────────────────────────────────────────


def test_open_chat_verifies_the_chat_and_stages_nothing(tmp_path: Path) -> None:
    page = _Page()
    holder = _Holder(page)
    spec = make_whatsapp_open_chat_spec(store=_store(tmp_path))

    result = spec.execute(WhatsAppOpenChatArgs(contact="Rahul"), _ctx(holder))

    assert result.ok, result.error
    assert result.verified is True
    assert page.visited == [WHATSAPP_SEND_URL.format(digits=_PHONE)]
    assert page.staged_paths == [] and page.clicked == []
    assert result.data["recipient"] == "Rahul"


def test_open_chat_refuses_an_ambiguous_contact(tmp_path: Path) -> None:
    store = _store(tmp_path, ("Ali Khan", "9112345678"), ("Ali Ahmed", _PHONE))
    holder = _Holder(_Page())
    spec = make_whatsapp_open_chat_spec(store=store)

    result = spec.execute(WhatsAppOpenChatArgs(contact="Ali"), _ctx(holder))

    assert result.ok is False
    assert "multiple contacts match" in result.error
    assert holder.acquires == 0


# ── graph: a clarification answer keeps the intended send ─────────────────


def _recording_spec(record: list[dict[str, Any]]) -> ToolSpec:
    def run(args: WhatsAppSendFileArgs, ctx: ToolContext) -> ToolResult:
        del ctx
        record.append(args.model_dump())
        return ToolResult(ok=True, output="sent")

    return ToolSpec(
        name="whatsapp_send_file",
        description="recording fake for the clarification test",
        args_model=WhatsAppSendFileArgs,
        base_tier=0,
        run=run,
        describe=lambda args: "send a file",
    )


def test_a_contact_clarification_preserves_the_send_intent(tmp_path: Path) -> None:
    record: list[dict[str, Any]] = []
    target = _file(tmp_path)
    decisions = [
        brain_clarify("Which contact should I send it to?"),
        brain_action("whatsapp_send_file", {"contact": "Rahul", "path": str(target)}),
    ]
    ctx = make_app_context(
        Settings(),
        llm=FakeLLM(decisions),
        registry=registry_with(_recording_spec(record)),
    )
    conn = sqlite3.connect(tmp_path / "c.db", check_same_thread=False)
    saver = open_sqlite_checkpointer(str(tmp_path / "c.db"))

    first = run_task(ctx, saver, "send this photo to Rahul on WhatsApp")
    assert first.interrupted is True
    assert first.interrupt_kind == "clarification"
    assert record == []

    answer = "Rahul"
    second = resume_task(ctx, saver, first.task_id, answer)
    conn.close()

    assert second.interrupted is False
    assert record == [{"contact": "Rahul", "path": str(target)}]
