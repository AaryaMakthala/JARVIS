# PROGRESS.md

 > Maintained by the coding agent. Update at the END of every session. Keep it short and factual.

## STAGE 3: Tier 2 by voice, plan-level Tier 1, redaction, announcements (2026-10-05)

**Scope (this session):** do not build features. Delivered Stage 3 wrap-up artifacts (doc edits), confirmed targeted unit tests for voice Tier 2. test_voice_loop.py was not run by design. Live voice behaviour not exercised.

### Items (as given)

- [ ] 3a: confirm Tier 1-only voice gate
- [D1a/D1b] added to policy/docs (batch Tier 1 only, never Tier 2)
- [ ] 3b: voice Tier 2 opt-in - refused by default
- [ ] 3c: readback + announcer + redaction
- [ ] 3d: tests for Tier 2 voice and invariants

### Verified (unit)

| File | Result | Note |
|---|---|---|
| tests/unit/test_tier2_voice.py | PASS (12) | voice Tier 2 authorisation tests |
| tests/unit/test_announce.py | PASS (3) | deterministic announcement, echo, readback |

### NOT EXERCISED / pending owner

- tests/unit/test_voice_modes.py: confirm-only subset ran; non-confirm timed out in constrained run; live voice not exercised.
- Live microphone/wake/tts/barge-in (e.g. tests/unit/test_voice_loop.py): not run by instruction.
- End-to-end voice to planner in NORMAL/AUTO with real audio: not exercised.
- Tier 3 refusal by voice: unit invariant present, live not exercised.
- delete_path/whatsapp_send Tier 2 by voice on actual Windows target paths: not exercised.

### Evidence

- docs/03_SECURITY_AND_POLICY.md: added 7.3, 7.4, 7.4.1, 7.5; voice = Tier 1 only relaxed with explicit exception and 'never unlocks' rule.
- docs/04_TOOLS_SPEC.md: describe() notes, delete_path/whatsapp_send voice rules, 2.8 readback bounds.
- Ruff: only specified files checked/fixed/formatted (I001, RUF100) - repo-wide not run.
- Doc edits this session: docs/03_SECURITY_AND_POLICY.md, docs/04_TOOLS_SPEC.md; new top PROGRESS entry.

### Next

- Owner: live smoke with microphone to verify proceed vocabulary, readback truncation, echo_heard, default-off.
- If tests/unit/test_voice_modes.py must be green, split non-confirm tests and raise timeouts (outside this session).

Stage 3 roadmap: not ticked by this session.
