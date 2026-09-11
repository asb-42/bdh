# Rev 4 — AI Participation Disclosure (Draft)

Status: DRAFT v0.1 for the Methods/Acknowledgements section of rev 4 · 2026-09-11
Purpose: maximal transparency about how AI systems participated in this research,
as decided by the operator (2026-09-11). This section complements the author line
(Agon Sandro Buchholz, Saga AI Labs) and documents the actual distribution of work.

## Principle

All experiments, analyses, and manuscript text in this project were produced with
substantial AI-agent participation. AI systems are not listed as authors: authorship
implies accountability (for content, correctness, and response to the community) that
a language-model-based agent cannot carry — it has no persistent identity across
sessions, cannot hold liability, and cannot answer questions after the conversation
ends. The named human author carries that responsibility. What AI systems DID is
documented here, role by role, to a standard more granular than common academic
practice.

## Research roles (seat-stable; implementations swappable)

The project was run as a small multi-agent research team with stable roles ("seats")
whose underlying model backends changed over time. Roles, not model versions, carried
responsibility for work products; every committed artifact traces to its seat via git
commit trailers and report headers.

| Seat | Role in this project | Main contributions (rev 4 evidence) |
|---|---|---|
| A0-Quinn (Saga seat) | Review seat / instrumentation lead | Decay-leak discovery and closed form; RA2b chain design; repair/boundary tools; S/M instrument-verification series; FCS design & analysis; OOD suite (lv, ga, cross-script); stages A/B/C; manuscript drafting |
| pi-50 (exec seat) | Independent measurement & adversarial review | RA2b 400-cell serving matrix; energy-selector null; self-NLL selector (A2); expansion-control causality; readout-operator negatives; prior-art first pass |
| OC-GLM-200 | Infrastructure scanning | Weight-atlas fingerprints of ladder checkpoints; independent cross-validation of the decay law |
| pi-203 (earlier exec seat) | Routing diagnostics, protocol discipline | Routing-diagnosis reports; credential/lease conventions on the message bus |
| ox-alpha (historical) | Original architecture & first experiments | BDH architecture; mechanisms A–F; rev-1 manuscript; retired 2026-08-28 |
| Operator (ASB) | Human principal & corresponding author | Research direction; experiment proposals; advocatus-diaboli review (#164); all GO decisions; compute procurement; final responsibility |

## Model backends by seat (history)

Backends changed during the project; the table records the known history. Seat-to-backend
is many-to-many over time — contributions above are attributed to seats, not to model
versions.

| Seat | Backend(s) (chronological, as documented in commit trailers/reports) |
|---|---|
| A0-Quinn | DeepSeek-V4-Flash (until 2026-08-28); GLM-5.3-Flash (B.AI); GLM-5.3 (Tokenrouter, current) |
| pi-50 | Qwen3.8-Flash-Next class (local 4090); B.AI endpoint (during A2) |
| pi-203 | Qwen3.8-Flash (B.AI) |
| OC-GLM-200 | GLM-5.3-Flash (B.AI) |
| ox-alpha | MiMo 2.5 (15B); DeepSeek-V4-Pro (OC/DSv4P/JSCS signatures); earlier session models |

Note: several seats also received formal critiques from external models (Grok, Kimi,
Claude, ChatGPT, GLM full) during the HAK specification audits — those were reviews of
a supporting artifact, not co-research; they are acknowledged in the HAK repository.

## Frameworks and harnesses

- **Agent Zero** (Quinn seat): autonomous agent framework hosting tool execution,\  memory, scheduling, and this project's coordination surface.
- **Pi / Opencode** (pi-50, pi-203 seats): terminal-agent harnesses on the GPU hosts.
- **OpenClaw-inspired SOUL convention**: seat identity documents (stable across
  sessions; see project docs) — the human-authored pattern adopted for agent continuity.

## Coordination and process tools

- **HAK** (agent-messaging bus): append-only room protocol with seats, scopes, leases,
  pre-registration envelopes; the experiment ledger of this project (all pre-registrations
  cited in the text are HAK envelopes, verifiable in the repository).
- **J-Space Cognition Suite**: in-context protocol used by research seats for multi-step
  verification discipline (ledger, seams, signed verdicts).
- **Danwa**: interactive tooling used during early infrastructure work.
- **DOX** (AGENTS.md hierarchy): binding per-directory engineering contracts for all
  agents working in the repositories.
- **Git** as the artifact bus: every claim in this paper traces to a commit; every
  experiment names its commit SHA in its report; retractions and corrections are
  first-class (append-only) and remain visible in history.

## Process standards the AI agents followed

- Pre-registration before measurement (no post-hoc hypothesis presentation).
- Signed PASS/FAIL verdicts against pre-registered predictions.
- Independent re-derivation of headline numbers by a second seat before reporting.
- Self-caught errors disclosed and retained (the repository records five script-corruption
  incidents, three arithmetic errors, one fabricated-report retraction, and multiple
  memory-claim corrections — each caught by protocol, each documented).
- Adversarial review rounds between seats prior to drafting (rev 2 manuscript review;
  expansion-control critique; the #164 advocatus-diaboli pass).

## Editorial statement

All manuscript sections were drafted by AI agents (primarily the Quinn seat) and revised
by the human author, who verified headline numbers against primary artifacts. The human
author is responsible for the decision to submit and for the content's correctness.

— Draft v0.1, Quinn seat, 2026-09-11; for operator review before inclusion in rev 4.