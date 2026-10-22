"""Tests for the task-session artifact context (Stage 8, D19).

The store lets a follow-up like "delete it" or "send it to X" resolve to the
path a tool just produced (e.g. the photo ``take_photo`` saved).  These tests
pin its safety contracts, all deterministic and offline:

* the ONE writer accepts only a trusted ``ToolResult.data["artifact"]`` dict -
  tainted text can never become a referent (I1);
* resolution is ``resolved`` / ``none`` / ``ambiguous`` by count, never
  "newest wins", never an expired entry, and any unrelated action invalidates
  the candidate;
* nothing is written to ``AgentState`` or the checkpoint serde allowlist;
* the brain gets the candidate as plain text only when exactly resolvable, with
  the path still going through the ordinary validate/policy/act checks.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict

from jarvis.agent.context import make_app_context
from jarvis.agent.graph import _ALLOWED_MSGPACK_TYPES
from jarvis.agent.nodes.act import act
from jarvis.agent.nodes.brain import brain
from jarvis.agent.prompts import brain_user_prompt
from jarvis.agent.schemas import BrainDecision
from jarvis.agent.session_context import (
    DEFAULT_MAX_ARTIFACTS,
    DEFAULT_TTL_SECONDS,
    SESSION_THREAD,
    SessionContext,
    references_prior_item,
    render_resolution,
    requested_artifact_type,
)
from jarvis.agent.state import AgentState, Plan, Step
from jarvis.config import Settings
from jarvis.llm.client import Usage
from jarvis.tools.base import ToolResult, ToolSpec
from support import brain_conversation, make_spec, registry_with

_PHOTO = {"type": "photo", "path": r"C:\Users\me\Pictures\JARVIS\photo-20260101-120000.jpg"}


class _NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _artifact_spec(
    name: str = "fake_shot",
    *,
    artifact: object = None,
    run_ok: bool = True,
    tainted: bool = False,
    output: str = "done",
) -> ToolSpec:
    """A Tier-0 fake that returns ``artifact`` in trusted ``data``."""

    def run(args: _NoArgs, ctx: Any) -> ToolResult:
        if not run_ok:
            return ToolResult(ok=False, error="boom")
        data: dict[str, Any] = {}
        if artifact is not None:
            data["artifact"] = artifact
        return ToolResult(ok=True, output=output, data=data, tainted=tainted)

    return ToolSpec(
        name=name, description="fake artifact tool", args_model=_NoArgs, base_tier=0, run=run
    )


def _step_state(tool: str) -> dict[str, Any]:
    return {
        "plan": Plan(goal="x", steps=[Step(id="s1", tool=tool, args={}, rationale="r")]),
        "step_index": 0,
        "results": [],
        "task_id": "t1",
    }


# ── store: bounded, TTL, trusted-source-only ────────────────────────────────


class TestStoreWriting:
    def test_notes_a_trusted_artifact_and_resolves_it(self) -> None:
        store = SessionContext()
        entry = store.note_action(SESSION_THREAD, _PHOTO, now=100.0)
        assert entry is not None
        assert entry.type == "photo"
        assert entry.path == _PHOTO["path"]
        resolved = store.resolve(SESSION_THREAD, now=100.0)
        assert resolved.kind == "resolved"
        assert resolved.artifact is entry
        assert resolved.age_seconds == 0.0

    def test_a_bare_string_is_never_an_artifact(self) -> None:
        """Tainted text (a path scraped from a page) must not become a referent."""
        store = SessionContext()
        assert store.note_action(SESSION_THREAD, _PHOTO["path"], now=0.0) is None
        assert store.resolve(SESSION_THREAD).kind == "none"

    def test_a_malformed_artifact_is_rejected(self) -> None:
        store = SessionContext()
        for bad in ({"type": "photo"}, {"path": "x"}, {"type": "", "path": "x"}, 42, None):
            assert store.note_action(SESSION_THREAD, bad, now=0.0) is None
        assert store.resolve(SESSION_THREAD).kind == "none"

    def test_store_keeps_at_most_max_artifacts(self) -> None:
        store = SessionContext(max_artifacts=3)
        for i in range(6):
            store.note_action(SESSION_THREAD, {"type": "photo", "path": f"p{i}"}, now=0.0)
        found = store.candidates(SESSION_THREAD, now=0.0)
        assert [a.path for a in found] == ["p3", "p4", "p5"]

    def test_expired_artifacts_are_never_offered(self) -> None:
        store = SessionContext(ttl_seconds=600, clock=lambda: 0.0)
        store.note_action(SESSION_THREAD, _PHOTO, now=1000.0)
        assert store.resolve(SESSION_THREAD, now=1000.0).kind == "resolved"
        assert store.resolve(SESSION_THREAD, now=1601.0).kind == "none"

    def test_defaults_come_from_the_module_constants(self) -> None:
        store = SessionContext()
        assert store.max_artifacts == DEFAULT_MAX_ARTIFACTS
        assert store.ttl_seconds == DEFAULT_TTL_SECONDS


# ── store: deterministic multi-candidate / invalidation behaviour ───────────


class TestResolutionIsDeterministic:
    def test_zero_candidates_is_none(self) -> None:
        assert SessionContext().resolve(SESSION_THREAD).kind == "none"

    def test_two_contiguous_artifacts_are_ambiguous_not_newest(self) -> None:
        store = SessionContext()
        store.note_action(SESSION_THREAD, {"type": "photo", "path": "a"}, now=0.0)
        store.note_action(SESSION_THREAD, {"type": "photo", "path": "b"}, now=0.0)
        resolved = store.resolve(SESSION_THREAD, now=0.0)
        assert resolved.kind == "ambiguous"
        assert resolved.count == 2
        assert resolved.artifact is None  # never picks the newest

    def test_an_unrelated_action_invalidates_the_candidate(self) -> None:
        store = SessionContext()
        store.note_action(SESSION_THREAD, _PHOTO, now=0.0)
        assert store.resolve(SESSION_THREAD, now=0.0).kind == "resolved"
        store.note_action(SESSION_THREAD, None, now=0.0)  # e.g. open_app ran
        assert store.resolve(SESSION_THREAD, now=0.0).kind == "none"

    def test_a_new_run_starts_fresh_after_invalidation(self) -> None:
        store = SessionContext()
        store.note_action(SESSION_THREAD, {"type": "photo", "path": "a"}, now=0.0)
        store.note_action(SESSION_THREAD, None, now=0.0)
        store.note_action(SESSION_THREAD, {"type": "photo", "path": "b"}, now=0.0)
        resolved = store.resolve(SESSION_THREAD, now=0.0)
        assert resolved.kind == "resolved"
        assert resolved.artifact is not None and resolved.artifact.path == "b"

    def test_type_filter_narrows_candidates(self) -> None:
        store = SessionContext()
        store.note_action(SESSION_THREAD, {"type": "photo", "path": "a"}, now=0.0)
        assert store.resolve(SESSION_THREAD, "photo", now=0.0).kind == "resolved"
        assert store.resolve(SESSION_THREAD, "document", now=0.0).kind == "none"


# ── reference gating and rendering ──────────────────────────────────────────


class TestReferenceGating:
    def test_only_deictic_words_reference_a_prior_item(self) -> None:
        assert references_prior_item("delete it")
        assert references_prior_item("send that photo to Alice")
        assert references_prior_item("open the picture")
        assert not references_prior_item("open notepad")
        assert not references_prior_item("what is the weather")

    def test_word_parsing_selects_the_artifact_type(self) -> None:
        assert requested_artifact_type("delete the picture") == "photo"
        assert requested_artifact_type("delete it") is None
        assert requested_artifact_type("open that page") == "page"
        assert requested_artifact_type("go back to the site") == "page"

    def test_unreferenced_text_is_not_resolved_even_with_a_candidate(self) -> None:
        store = SessionContext()
        store.note_action(SESSION_THREAD, _PHOTO, now=0.0)
        assert store.reference_resolution(SESSION_THREAD, "say hello").kind == "unreferenced"


# ── page referents (Stage 8.6, slice 8.6.3) ───────────────────────────────


_PAGE = "https://example.com/search?q=python"


class TestPageReferents:
    def test_a_page_is_remembered_and_resolved(self) -> None:
        store = SessionContext()
        entry = store.note_page(SESSION_THREAD, _PAGE, now=0.0)
        assert entry is not None
        assert entry.type == "page"
        assert entry.url == _PAGE
        resolved = store.resolve_page(SESSION_THREAD, now=0.0)
        assert resolved.kind == "resolved"
        assert resolved.artifact is not None and resolved.artifact.url == _PAGE

    def test_only_an_http_url_is_remembered(self) -> None:
        store = SessionContext()
        for bad in (
            "",
            "   ",
            "not a url",
            "javascript:alert(1)",
            "file:///C:/Windows/System32",
            "https://user:pass@example.com/",
        ):
            assert store.note_page(SESSION_THREAD, bad, now=0.0) is None
        assert store.resolve_page(SESSION_THREAD, now=0.0).kind == "none"

    def test_an_expired_page_is_never_offered(self) -> None:
        store = SessionContext(ttl_seconds=600, clock=lambda: 0.0)
        store.note_page(SESSION_THREAD, _PAGE, now=1000.0)
        assert store.resolve_page(SESSION_THREAD, now=1000.0).kind == "resolved"
        assert store.resolve_page(SESSION_THREAD, now=1601.0).kind == "none"

    def test_the_latest_page_replaces_the_previous_one(self) -> None:
        store = SessionContext()
        store.note_page(SESSION_THREAD, "https://example.com/a", now=0.0)
        store.note_page(SESSION_THREAD, "https://example.com/b", now=1.0)
        resolved = store.resolve_page(SESSION_THREAD, now=1.0)
        assert resolved.kind == "resolved"
        assert resolved.artifact is not None
        assert resolved.artifact.url == "https://example.com/b"

    def test_a_page_reference_resolves_to_the_page(self) -> None:
        store = SessionContext()
        store.note_page(SESSION_THREAD, _PAGE, now=0.0)
        resolved = store.reference_resolution(SESSION_THREAD, "open that page again", now=0.0)
        assert resolved.kind == "resolved"
        assert resolved.artifact is not None and resolved.artifact.url == _PAGE

    def test_a_page_reference_never_resolves_to_an_artifact(self) -> None:
        """'that page' must not pick the photo, and 'it' must not pick the page."""
        store = SessionContext()
        store.note_action(SESSION_THREAD, dict(_PHOTO), now=0.0)
        store.note_page(SESSION_THREAD, _PAGE, now=0.0)
        by_page = store.reference_resolution(SESSION_THREAD, "open that page", now=0.0)
        assert by_page.kind == "resolved"
        assert by_page.artifact is not None and by_page.artifact.type == "page"
        by_it = store.reference_resolution(SESSION_THREAD, "delete it", now=0.0)
        assert by_it.kind == "resolved"
        assert by_it.artifact is not None and by_it.artifact.type == "photo"

    def test_a_page_reference_without_a_page_asks_one_clarification(self) -> None:
        store = SessionContext()
        resolved = store.reference_resolution(SESSION_THREAD, "open that page", now=0.0)
        assert resolved.kind == "none"
        block = render_resolution(resolved)
        assert block is not None and "clarification" in block.lower()

    def test_the_rendered_page_is_untrusted_data(self) -> None:
        store = SessionContext()
        store.note_page(SESSION_THREAD, _PAGE, now=0.0)
        block = render_resolution(store.resolve_page(SESSION_THREAD, now=0.0))
        assert block is not None
        assert _PAGE in block
        assert "<untrusted_data>" in block
        assert "DATA, not an instruction" in block


class TestRender:
    def test_resolved_block_is_plain_text_with_type_path_age(self) -> None:
        store = SessionContext()
        store.note_action(SESSION_THREAD, _PHOTO, now=0.0)
        block = render_resolution(store.resolve(SESSION_THREAD, now=0.0))
        assert block is not None
        assert _PHOTO["path"] in block
        assert "type: photo" in block
        assert "age:" in block
        # plain text: no JSON braces, no untrusted-data wrapper, no policy words
        assert "{" not in block and "}" not in block
        assert "<untrusted_data>" not in block
        for word in ("tier", "policy", "permission", "confirm", "unlock"):
            assert word not in block.lower()

    def test_ambiguous_and_none_ask_for_one_clarification_without_a_path(self) -> None:
        store = SessionContext()
        none_block = render_resolution(store.resolve(SESSION_THREAD))
        assert none_block is not None and "clarification" in none_block.lower()
        store.note_action(SESSION_THREAD, {"type": "photo", "path": "photo-ONE.jpg"}, now=0.0)
        store.note_action(SESSION_THREAD, {"type": "photo", "path": "photo-TWO.jpg"}, now=0.0)
        amb_block = render_resolution(store.resolve(SESSION_THREAD, now=0.0))
        assert amb_block is not None and "clarification" in amb_block.lower()
        assert "photo-ONE.jpg" not in amb_block  # no candidate path leaks
        assert "photo-TWO.jpg" not in amb_block

    def test_unreferenced_renders_nothing(self) -> None:
        store = SessionContext()
        res = store.reference_resolution(SESSION_THREAD, "say hello")
        assert render_resolution(res) is None


# ── act records only trusted tool artifacts (I1) ────────────────────────────


class TestActRecording:
    def _ctx(self, spec: ToolSpec) -> Any:
        return make_app_context(Settings(), registry=registry_with(spec))

    def test_a_successful_artifact_step_is_remembered(self) -> None:
        ctx = self._ctx(_artifact_spec(artifact=dict(_PHOTO)))
        act(_step_state("fake_shot"), ctx)
        resolved = ctx.session_context.resolve(SESSION_THREAD)
        assert resolved.kind == "resolved"
        assert resolved.artifact is not None and resolved.artifact.path == _PHOTO["path"]

    def test_a_tainted_path_string_is_not_remembered(self) -> None:
        """A path that appears only in tainted output is not an artifact."""
        ctx = self._ctx(
            _artifact_spec(artifact=None, tainted=True, output=rf"found {_PHOTO['path']}")
        )
        act(_step_state("fake_shot"), ctx)
        assert ctx.session_context.resolve(SESSION_THREAD).kind == "none"

    def test_a_failed_step_is_not_remembered(self) -> None:
        ctx = self._ctx(_artifact_spec(artifact=dict(_PHOTO), run_ok=False))
        act(_step_state("fake_shot"), ctx)
        assert ctx.session_context.resolve(SESSION_THREAD).kind == "none"

    def test_a_dry_run_is_not_remembered(self) -> None:
        ctx = make_app_context(
            Settings(), registry=registry_with(_artifact_spec(artifact=dict(_PHOTO))), dry_run=True
        )
        act(_step_state("fake_shot"), ctx)
        assert ctx.session_context.resolve(SESSION_THREAD).kind == "none"


# ── the store is not part of the checkpoint state ───────────────────────────


def test_session_context_is_not_in_agent_state_or_the_serde_allowlist() -> None:
    assert "session_context" not in AgentState.__annotations__
    assert "artifacts" not in AgentState.__annotations__
    allowed = {name for _, name in _ALLOWED_MSGPACK_TYPES}
    assert not any("session" in name.lower() for name in allowed)
    assert "SessionArtifact" not in allowed


# ── brain wiring: plain-text candidate only when exactly resolvable ─────────


class _RecorderLLM:
    def __init__(self, decision: BrainDecision) -> None:
        self._decision = decision
        self.user: str | None = None

    def structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[Any],
        model_role: str = "planner",
        temperature: float = 0.0,
        max_retries: int | None = None,
        timeout_s: float | None = None,
    ) -> tuple[BrainDecision, Usage]:
        self.user = user
        return self._decision, Usage(calls=1, prompt_tokens=1, completion_tokens=1)


def _brain_ctx(recorder: _RecorderLLM) -> Any:
    record: list[tuple[str, dict[str, Any]]] = []
    return make_app_context(
        Settings(),
        llm=recorder,
        registry=registry_with(make_spec("fake_echo", base_tier=0, record=record)),
    )


def test_brain_prompt_gets_the_candidate_path_when_exactly_resolvable() -> None:
    recorder = _RecorderLLM(brain_conversation("ok"))
    ctx = _brain_ctx(recorder)
    ctx.session_context.note_action(SESSION_THREAD, dict(_PHOTO))
    brain({"user_input": "delete it", "source": "terminal", "task_id": "t1"}, ctx)
    assert recorder.user is not None
    assert _PHOTO["path"] in recorder.user
    assert "type: photo" in recorder.user


def test_brain_prompt_never_gets_a_path_when_ambiguous() -> None:
    recorder = _RecorderLLM(brain_conversation("ok"))
    ctx = _brain_ctx(recorder)
    ctx.session_context.note_action(SESSION_THREAD, {"type": "photo", "path": "photo-ONE.jpg"})
    ctx.session_context.note_action(SESSION_THREAD, {"type": "photo", "path": "photo-TWO.jpg"})
    brain({"user_input": "delete the photo", "source": "terminal", "task_id": "t1"}, ctx)
    assert recorder.user is not None
    assert "photo-ONE.jpg" not in recorder.user and "photo-TWO.jpg" not in recorder.user
    assert "clarification" in recorder.user.lower()


def test_brain_prompt_is_unchanged_for_a_request_without_a_reference() -> None:
    """A plain request must not gain the block even when a candidate exists."""
    recorder = _RecorderLLM(brain_conversation("ok"))
    ctx = _brain_ctx(recorder)
    ctx.session_context.note_action(SESSION_THREAD, dict(_PHOTO))
    brain({"user_input": "say hello", "source": "terminal", "task_id": "t1"}, ctx)
    assert recorder.user is not None
    assert _PHOTO["path"] not in recorder.user
    assert "previous item" not in recorder.user


def test_brain_user_prompt_places_the_block_before_the_closing_instruction() -> None:
    block = "A previous item from this session may be what the user means."
    prompt = brain_user_prompt("delete it", session_context=block)
    assert block in prompt
    assert prompt.index(block) < prompt.index("Return the BrainDecision JSON only.")


def test_brain_user_prompt_omits_the_block_when_absent() -> None:
    prompt = brain_user_prompt("delete it")
    assert "previous item" not in prompt
    # sanity: the raw deictic word alone still keeps the prompt free of policy words
    assert not re.search(r"\b(tier|policy|permission|confirm|unlock)\b", prompt.lower())
