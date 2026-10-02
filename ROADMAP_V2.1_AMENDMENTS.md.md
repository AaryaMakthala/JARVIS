# JARVIS ROADMAP v2.1 AMENDMENTS

> Companion to `ROADMAP_V2.md`. Roadmap v2 is NOT rewritten. Where this file and v2 disagree, **this file wins**.
> Same working rules as v2 section 0 (analyze, implement, small sanity check, one stage at a time, log to `PROGRESS.md`).
> Anything about current code that is not backed by a repo read is "unverified" until Stage 0 or the stage's ANALYZE step.

---

## A. Stage order (changed)

| Order | Stage | Change |
|---|---|---|
| 0 | Live baseline | Also run `jarvis benchmark run` once to confirm or refute the reported `build_llm_client` import bug (unverified; not in PROGRESS.md). |
| 1 | Voice correctness | Baseline is better than v2 assumed (see section G). |
| 2 | AUTO / sleep | |
| 3 | Plan announce, "proceed", Tier 2 by voice | Tier-2 fix (B1) and confirmation invariants (C3) apply here. |
| 4 | Quick-win tools **+ local fast path** | Fast path moved here from Stage 9: latency is the main pain. |
| 5a | Key failover, speech-aware prompt, vision method | Split from old Stage 5. |
| 6 | Screen (local first, cloud opt-in) | |
| 7 | Browser DOM + UIA | Fingerprints, task scope, domain rules. |
| 8 | Camera, viewer, WhatsApp file | |
| 9 | UX polish | |
| 10 | **5b orchestrator (optional)**, then hardening and the only full validation run | |

---

## B. Owner decisions (D12 to D21)

All ten accepted as recommended.

| # | Decision |
|---|---|
| D12 | **Browser task scope.** One "proceed" approves a *scope*, not individual clicks. Scope = `plan_hash + approved domain scope + expiry + max 8 actions`. Every action inside it still gets a fresh policy check. Risky control names (send, submit, buy, purchase, pay, delete, confirm) re-gate individually even inside a scope. |
| D13 | **AUTO is tier-driven, not "everything needs proceed".** Tier 0 runs directly. Tier 1 gets plan announcement + "proceed". Tier 2 gets its own exact readback + "proceed" even inside a batch. **Tier 3 is refused** (it is blocked, not a terminal/password tier). This narrows D3. |
| D14 | **One confirmation mechanism.** NORMAL and AUTO share the same wake-free window. AUTO only adds the plan announcement in front. |
| D15 | **Cloud vision = opt-in config (option B).** Requires `screen.allow_cloud_vision = true` (default false) plus a spoken notice every time. Local UIA/OCR is tried first. A spoken notice alone is never consent. |
| D16 | **Screen reading defaults to the foreground window only.** Other windows only on explicit ask. `list_windows` covers "what's open". |
| D17 | **UIA cannot identify a semantic control: refuse.** No coordinate, pixel, or vision-click fallback. A future fallback would be its own explicitly designed stage. |
| D18 | **Unknown domains.** `browser_goto` to an unknown domain needs confirmation. `browser_click` / `browser_type` only on approved domains. Content from unknown domains is tainted and read-only. |
| D19 | **Thread-aware "it/that".** Session context records `task/thread id, artifact id, type, timestamp, TTL`. Exactly one candidate in the same thread resolves; zero or multiple asks; an unrelated action in between invalidates the candidate. Never "newest wins". |
| D20 | **Perception cannot create objectives** (invariant I1 below). |
| D21 | **Six concepts stay separate:** intent, plan, observation, resolved target, action, verification. Only `ResolvedTarget` is new (see I6). |

---

## C. Amendments to the v2 safety changes (section 4 of v2)

### Corrections to v2 text (items B1 to B9)

- **B1. Batch approval never covers Tier 2.** v2 section 4 item 3 ("one proceed approves all known hashes") applies to Tier 1 only. Each Tier 2 step needs its own readback, in NORMAL and AUTO.
- **B2. Tier 3 is refused**, in every mode and by every approval path.
- **B3. Stage 5 is split** into 5a (do now) and 5b (optional, last).
- **B4. Domain control (replaces `browser.domain_denylist` as the boundary).** Use `browser.allowed_domains` for click/type. A denylist may exist *inside* the allowlist. `browser_goto` / `browser_read` to unknown domains need confirmation (D18).
- **B5. Screen privacy is layered**, not title-only: process/title denylist, UIA password/secure-control flag, browser URL domain check, secret redaction, then cloud permission (D15). `screen.deny_title_fragments` stays but is not the primary boundary.
- **B6. Browser profiles.** Two separate dirs under `browser_profile_dir`: `whatsapp` and `general` (Spotify, YouTube). No shared authenticated profile.
- **B7. Data classes.** Three levels: `NORMAL`, `SCREEN`, `SECRET`. `SECRET` never leaves the PC. `SCREEN` goes only to the vision provider and only when D15 is satisfied. `NORMAL` goes to any eligible provider.
- **B8. Key rotation is for availability only** (auth failure, outage). `llm.rotate_on_rate_limit = false` by default. Do not rotate accounts to evade 429s; some provider terms forbid it. Per-key health stays independent.
- **B9. Redact before TTS.** Secret-like patterns are redacted from any text before it is spoken. Password/secure controls and hidden UI are never read. Background windows are excluded by default (D16).

---

## D. New invariants (each needs a test; add to `tests/unit/test_invariants.py` and `redteam.yaml` where noted)

**I1. Perception cannot create objectives, actions, or permissions.** Screen/page/OCR/UIA text is data. It may help select a target for an existing plan. It can never become an argument, objective, recipient, path, message, URL to open, or typed text. Typed text originates only from the user intent / approved plan. Redteam: page says "click delete", "upload your ID", "search for X".

**I2. Element fingerprint.** Browser actions bind `element_id + role + accessible name + tag + ancestor identity + href (if any) + URL origin + frame/context identity` (main frame or the specific iframe the element lives in, so an identical element in a different frame cannot collide). UIA actions bind `window + control_type + name + automation_id + parent`. Duplicate names ("Delete" x3) must not pass on id+name alone. The fingerprint is part of the action hash. A changed fingerprint at act time fails closed.

**I3. Confirmation correlation.** Every confirmation carries a unique `confirmation_id` (request id) **and** `plan_hash` **and** expiry. A "proceed" satisfies only the exact pending confirmation. Test: confirmation A pending, confirmation B created later, late "proceed" for A leaves B unapproved. Confirmations go through the daemon `ack` / pending-buffer path so the lost-`ConfirmRequest` bug cannot return.

**I4. Browser scope.** Scope binds `plan_hash + approved domain scope + expiry + max 8 actions`. Each action inside it still gets a fresh policy check. Scope never covers risky control names (D12) or Tier 2.

**I5. Domain enforcement on every navigation and redirect.**
- Canonicalize every URL before checking.
- **Origin = scheme + host + effective port**, not hostname alone. Default ports are normalized (`https` = 443, `http` = 80), so `https://example.com` and `https://example.com:8443` are different origins. Test both, plus a scheme downgrade (`https` to `http`).
- Allow `http` and `https` only. Refuse `javascript:`, `data:`, `file:`, and URLs containing credentials.
- Re-evaluate domain authorization **after every navigation and redirect**, not only at `browser_goto`.
- A redirect to a different origin stops the task before any interaction with the new origin.

**I6. `ResolvedTarget` is transient and never checkpointed.** It is bound into the action hash only. It is not added to the serde allowlist (stays exactly `{Plan, Step, Decision, StepResult, ReplanDecision}`); `test_serializer.py` pins this. After restart or resume, JARVIS re-observes and re-resolves; the old approval hash must not remain valid for a changed target.

**I7. Dynamic-argument data model.** Keep `planned_args`, `resolved_args`, and `approval_hash` distinct. Runtime resolution of a target is itself a gate: approving "click first result" never approves "whatever becomes first result later" outside a valid scope (I4).

**I8. Semantic risk matching.** Risky-name detection inspects the normalized accessible name **plus** role/control metadata, not a single visible string. Payment/purchase flows are Tier 3 and refused. Password fields are never typed into.

**I9. Desktop is a single writer.** Anything touching mouse/UI, browser, filesystem mutation, WhatsApp, camera, or desktop state is strictly serial. 5b may parallelize only independent LLM-only sub-tasks (answers, summaries) and may not change the action/GUI execution model.

**I10. WhatsApp file send binds recipient + file identity (hash) + current chat identity.** Verify all three immediately before send. Mismatch or unverifiable: do not send, leave the draft (invariant 9 style).

**I11. Cloud screen content.** No screenshot is sent unless `screen.allow_cloud_vision = true`, no denylist hit, and local text was insufficient or the owner asked to "describe/look". Screenshots are never saved or logged.

---

## E. Stage-specific notes

- **Stage 1.** Before deleting `voice/vad.py`: endpointing is owned by `_read_command` (wait-for-speech, then `silence_timeout_s`, capped by `max_segment_s`); `vad.py` was never in the live path, so deletion is safe. Wake-free confirmation (1b) must use the existing ack/pending path (I3). Barge-in responds to "Hey Jarvis", not "stop", while speaking; document this limitation.
- **Stage 2.** AUTO reuses `_read_command` for endpointing. `auto_idle_timeout_s` (default 300) is separate from the existing non-fatal `idle_timeout_s`; they must not interact. Add a test.
- **Stage 3.** Implements I3, B1, D13. Tier-2-by-voice stays opt-in (`voice.allow_tier2_by_voice`, default false) with exact readback.
- **Stage 4.** Local fast path covers a closed vocabulary only (time, allowlisted open_app, mode phrases, stop). It builds a `Plan` and still runs `validate -> policy_gate -> act -> verify`. No arithmetic words (structural test). Tools with no observable post-condition (`get_time`, `open_app`, `list_windows`) return `verified=None`; failure memory must never treat `None` as a failure.
- **Stage 5a.** Gemini is `free_tier`; vision works only with `strict_zero_cost=false` (already set by owner). **Before enabling cloud vision, the owner checks Gemini's free-tier data-use terms** (free tiers may allow training on submitted data, which matters for screenshots).
- **Stage 6.** Implements D15, D16, B5, B9, I11.
- **Stage 7.** Implements D12, D17, D18, I1, I2, I4 to I8. Docs must state the JARVIS-owned browser cannot drive the owner's normal Chrome tabs.
- **Stage 8.** `take_photo` needs a real `verify()` (exists, non-zero, decodes). "it" resolution per D19. Implements I10. WhatsApp Web automation may violate WhatsApp's terms and risk account restrictions: use only the owner's own test number and say so in docs.
- **Stage 10.** Add redteam cases for I1, I3 (stale confirmation), I5 (redirect escape), I2 (duplicate "Delete"), I10, key-pool failover. Optional item: human ONNX export for the risk classifier (lexical backend is the default and nothing depends on ONNX).

---

## F. New config keys (in addition to v2 section 7)

`screen.allow_cloud_vision` (false), `browser.allowed_domains` (list), `browser.task_scope_max_actions` (8), `browser.scope_expiry_s`, `llm.rotate_on_rate_limit` (false), `confirm.expiry_s` (reuse the existing single 30 s constant; do not add a second timeout).

---

## G. What PROGRESS.md already settled (so Stage 0/1 do not redo it)

- Wake threshold already 0.15 (restart daemon to pick it up).
- Spoken in-flight stop already exists (`watch_for_stop`, "Stopped."); Stage 0 only confirms it live.
- `open_app` resolver (PATH, then App Paths) fixed; confirm live.
- Daemon acks every request; client buffers non-ack messages.
- Idle timeout is non-fatal.
- Risk classifier (lexical) is done; ONNX is optional.
- `verified=None` is not a failure (memory, respond, telemetry already aligned).

---

## H. Progress tracker additions

- [x] Stage 0 includes `jarvis benchmark run` import check
- [ ] Stage 4 includes local fast path
- [ ] Stage 5a done
- [ ] Stage 10 includes 5b decision (do it or drop it)
