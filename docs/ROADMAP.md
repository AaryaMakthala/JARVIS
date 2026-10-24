# JARVIS Roadmap (consolidated)

## Current effective roadmap (V2 amended by V2.1 and V2.2)

| Stage | Focus | Status | Notes |
|---|---|---|---|
| 0 | Live baseline | DONE | Agent-side checks done; hardware tests noted in history. |
| 1 | Voice correctness (wake-free confirmation, stop semantics, interruptible SAPI) | DONE | D22: bare "stop" cancels in busy states (overrides old wake-word-only). |
| 2 | Policy/paths/audit basics | DONE | Path resolution, tiers, audit tools. |
| 3 | Apps/time/windows/open_in_app/Notepad/VS Code basics | DONE | Implemented. |
| 4 | Orchestrator/extensions | IN PROGRESS / PARTIAL | Some items deferred. |
| 5 | Key pools/orchestrator/speech-aware brain/vision capability | IN PROGRESS/PARTIAL | 5a done (key pools, brain, completeness) per history; 5b orchestrator optional/not started. |
| 6 | Screen reading/description (UIA-first) | DONE | UIA-first read_screen/describe_screen; live acceptance passed for tested cases. |
| 7 | Browser DOM + UIA clicking | IN PROGRESS/PARTIAL | navigate/read live-validated; click/type/UIA unit-tested; Spotify route not built. |
| 8 | Camera/viewer/delete/WhatsApp photo/referents | IN PROGRESS | Code+tests added; mandatory live checks (P2/P3/camera/WhatsApp) NOT VERIFIED. |
| 9 | UX polish (transcript visibility, latency) | PLANNED | Voice latency tuning comes last per design. |

## Decisions and constraints still in force

- **LLM proposes, policy decides** (D1): No LLM output can lower a tier.
- **Free-tier only**: Groq, OpenRouter, NVIDIA (Gemini optional). Enforced by registry/qualify.
- **Voice is Tier 1 only by default** (D22 relaxes bare-stop only; not general voice Tier 2).
- **WhatsApp Web isolated profile**: Separate from general browser profile.
- **Browser referents**: Page referent cleared by unrelated actions; untrusted until verified.
- **Policy layer untouched**: Incremental refactor only.
- **Automation follows exact instructions**: Press/click as specified.
- **Persona**: Butler tone "Certainly, sir"; use name once stated.
