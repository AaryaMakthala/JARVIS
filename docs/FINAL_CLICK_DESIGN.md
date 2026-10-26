# Guarded final click (`browser_final_click`) - design only

**Status: design, not implemented.** No tool, no config key and no journal table
exists yet; nothing in this file changes the current behaviour. It specifies the
one remaining browser action that a task-scoped browsing grant must never carry:
an **irreversible, non-payment** click (send / submit / post / publish / subscribe
/ book).

The current model is unchanged and stays the baseline: `PolicyEngine` decides the
tier (`policy/engine.py:143-144` blocks a payment/purchase control outright), the
gate is the only place a confirmation happens (`agent/nodes/policy_gate.py`), and
`act` re-decides and re-verifies before dispatch (`agent/nodes/act.py`).

## 1. What it is

One extra tool, `browser_final_click`, registered at **Tier 2** and never
batch-eligible, whose approval is bound to: the exact element, the host, a fresh
page observation, and a typed confirmation phrase. It exists so the agent can
finish a legitimate action the user asked for without ever turning a "browse
this host" approval into "you may send things from this host".

Arguments (validated model, no free-form text):

| Arg | Type | Meaning |
|---|---|---|
| `ref` | `str` (opaque ref the browser issued) | which observed element to click |
| `action` | `Literal["send","submit","post","publish","subscribe","book"]` | the *declared* intent, checked against the page |
| `expect` | `str`, <=120 chars | one line describing the effect, echoed in the prompt |

The model never supplies the element's label, host, URL or message text: those
come from a fresh observation and from earlier steps. A label the model wrote is
untrusted text and authorizes nothing.

## 2. Order of classification: payment is decided first

The new tool must be unable to unblock anything, so the checks run in this order
and the first one never falls through:

1. **Tier 3 payment/purchase classification** (`policy/rules.py:201-222`,
   `control is CONTROL_PAYMENT` -> Tier 3 in `policy/engine.py:143`). Buy, pay,
   purchase, checkout, place order, donate, and their multi-word forms
   (`RISK_WORDS` in `policy/browse_grant.py`) are classified **before** the action
   is looked at, using the *observed* name/role/host - not the model's claim.
   Result: hard refusal. `browser_final_click` has no argument that can reach a
   payment path, and no allowlist entry that names one.
2. **Password / credential fields** - Tier 3 as today
   (`policy/engine.py:141`).
3. **Only then** the irreversible-action allowlist, which can *raise* the tier to
   2 (never lower it): the engine's invariant `max(base, rules, classifier,
   taint)` is untouched.

If the observed label contains a payment word the step is refused even when the
user typed the confirmation phrase: classification precedes confirmation, and a
confirmation cannot re-classify. If the label carries no word from the `action`
allowlist at all, the tool refuses (the declared action must match the page's
own word) - so a "Delete" or "Remove" control is not reachable through this tool
either.

## 3. Never covered by a browse grant, never batched

- `policy/browse_grant.py`'s `BROWSE_TOOLS` is a closed set and
  `browser_final_click` is not in it. `coverage()` also refuses any step whose
  tier is not 1, so even a mis-filed tool name cannot be covered.
- The batch/plan approval path (`agent/batch_approval.py`) only folds contiguous
  **Tier 1** steps; a Tier 2 step is never batch-eligible, so plan-level "yes"
  never approves a final click.
- A scope grant (`ScopeRegistry`) does not cover it either: the gate's
  `_needs_fresh_confirm()` already returns true for a risky control and for
  `needs_unlock`, which is exactly this step.

## 4. Authorization binding (four independent parts)

A final click is approved only when all four hold:

1. **Exact target identity** - the gate resolves the element and records a
   transient `TargetBinding` (`agent/target_binding.py`) over
   `sha256(action_hash, fingerprint, origin, confirmation_id)`. `act` re-observes
   and recomputes it; any difference is a refusal (`binding: mismatch`).
2. **Fresh page evidence** - the fingerprint is derived from the element's
   stable attributes plus a hash of the *observed page state* (URL + the
   element's own serialized attributes) taken at gate time. A page that changed
   its button, its form action or its label between the prompt and the dispatch
   produces a different fingerprint, so the binding fails closed.
3. **Host binding** - the binding's `origin` is part of the hash, and the step is
   refused if the current origin's host differs, so a grant/approval for host A
   can never dispatch on host B.
4. **Typed confirmation hash** - reuse the existing typed-confirmation mechanism
   (`Decision.needs_typed_confirmation`, `policy_gate._typed_confirmation_ok`):
   the user types a phrase derived in code, e.g. `post on example.com`. The
   journal and the checkpoint store only the **sha256** of the normalized phrase,
   never the phrase itself and never the message body.

Single use: the binding is consumed on dispatch (one-shot, like every other
target binding), and the tool refuses a second call for the same journal id.

## 5. Re-verify immediately before dispatch

Between the gate's approval and the click, `act` (not the tool) repeats the
target check, and the tool itself then, inside its own `run()`:

1. re-reads the page (one observation),
2. re-derives the element's fingerprint and compares it with
   `ToolContext.approved_target_identity` (`ToolSpec.bind_target_identity`),
3. re-checks the origin/host against the approved one,
4. re-checks the payment/password classification on the freshly observed label,
5. only then dispatches exactly one click on exactly that element.

Any mismatch at any point is a refusal with no side effect. This is a narrow
window, not an atomic lock: the remaining window (page changing after step 4) is
the same one every browser automation has, and is why the journal below records
"indeterminate" rather than assuming success.

## 6. Idempotency: reserve, dispatch, confirm

A final click is the one browser action where "did it happen?" may be
unanswerable. So the tool keeps a minimal journal row in the existing SQLite
store (`memory.db`, no new dependency, same connection/lock discipline):

```
final_click_journal(
  id TEXT PRIMARY KEY,          -- random hex, one per attempt
  task_id TEXT, host TEXT,
  action TEXT,                  -- the declared class
  target_fingerprint TEXT,      -- not the element's text
  confirmation_sha256 TEXT,     -- hash of the typed phrase, never the phrase
  state TEXT,                   -- reserved | dispatched | confirmed | indeterminate
  outcome_code TEXT,            -- short code only (no page content)
  created_at TEXT, updated_at TEXT
)
```

Transitions are strictly one-way:

```
reserved  --(all pre-flight checks passed, click about to be sent)--> dispatched
reserved  --(any pre-flight check failed)-------------------------> (row deleted, refused)
dispatched --(post-click evidence: URL/title/form state observed)--> confirmed
dispatched --(no reliable evidence, timeout, browser crash, exception)--> indeterminate
```

Rules:

- The row is written **before** the click is dispatched. A crash between the
  write and the click leaves `dispatched`/`indeterminate`, never "no record".
- An `indeterminate` row is **never retried**. Not after a replan, not after a
  daemon restart, not by a second tool call: the tool refuses and the answer says
  plainly *"I could not confirm whether that was sent - check the site before
  trying again."* The user decides, not the agent.
- A `confirmed` outcome requires independent evidence (the form is gone, the URL
  changed to a confirmation page, the button no longer exists). A tool result
  that merely did not raise is **not** evidence.
- `reserved` rows older than the binding TTL are treated as abandoned and are
  cleared by the same bounded prune discipline as the other tables (<=50 rows per
  call, oldest first).
- No page content, message body, field values or element labels are stored -
  only the hashes and short codes above.

## 7. Configuration

| Key | Default | Effect |
|---|---|---|
| `browser.final_click_max_total` | `0` | Total final clicks allowed for the whole run. `0` = the tool refuses every call (disabled). Nothing may raise a hard maximum. |
| `browser.final_click_voice_approval` | `false` | Off: a voice "yes" cannot satisfy the typed confirmation or the unlock. On: reuse `voice/tier2_voice.allowed` (daemon marker + gate re-derivation), never a new bypass. |
| `browser.final_click_binding_ttl_s` | `180` | How long an approved final-click binding stays usable; expiry is a refusal. |

Typed confirmation and the unlocked-session requirement reuse the existing
settings; no new tier, no new registry and no new approval channel.

## 8. Failure modes and their answers

| Situation | Behaviour |
|---|---|
| Label contains a payment word | Tier 3 refusal (classified first, before any confirmation). |
| Label matches no allowed action word | Refusal; the prompt never offers it. |
| Element/page changed after approval | Binding mismatch, refusal, nothing dispatched. |
| Host changed mid-task | Refusal (origin is part of the binding hash). |
| Click sent, no evidence either way | `indeterminate`, honest "check manually", never retried. |
| Browser crashed mid-click | `indeterminate` (the row was already written). |
| Second call in the same task | Binding consumed/row state blocks it; `final_click_max_total` counts the total. |
| Voice approval, voice opt-in off | Refused; typed confirmation and unlock still required. |

## 9. Out of scope

- Payments, purchases, checkout, donations - Tier 3, permanently.
- Bulk or repeated sends in one task; sending from messaging apps (those tools
  keep their own identity/verification paths, e.g. `whatsapp_send`).
- Any "undo" of a final click: the web gives us none, so the design never
  pretends to.

## 10. Owner decisions still needed

1. **Ship it at all in v1?** The safest answer is "no" (Tier 3-blocked like
   payments); the value is completing legitimate tasks like "post this reply".
2. **Tier 2 (confirm + unlocked) or plain Tier 1 with typed confirmation only?**
   This design assumes Tier 2, i.e. an unlocked session is required.
3. **Which verbs are allowed?** Proposed: send, submit, post, publish, subscribe,
   book. Excluded: anything money, delete/remove, sign in.
4. **Typed phrase format** - `post on example.com`, or the element's own words?
   The first is code-derived and stable; the second quotes untrusted page text.
5. **Voice approval** - keep the default off, or opt in for accessibility?
6. **Journal location/retention** - `memory.db` (proposed) vs a separate JSONL
   file; and whether `jarvis memory clear` may delete journal rows (this design
   says no: an `indeterminate` row is evidence).
7. **What counts as post-click evidence** for `confirmed` on the sites we target
   (form disappearance vs URL change vs success banner), and whether "no evidence"
   should ever be reported as success (this design says never).
8. **Cap semantics** - is `final_click_max_total = 0` "tool refuses" (proposed) or
   "tool not registered"? The first keeps the registry and the policy honest.
