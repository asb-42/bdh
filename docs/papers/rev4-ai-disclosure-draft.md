# Rev 4 — AI Participation Disclosure (Draft)

Status: DRAFT v0.2 for the Methods/Acknowledgements section of rev 4 · 2026-09-11
Changes v0.1 → v0.2 (operator review, 2026-09-11): principle rescoped to the formal
limit (accountability), not a technical claim; model/provider tables flattened (the
per-seat history implied more precision than exists); weight-atlas added to tools;
permanent internal peer-review principle made explicit; Credits section added
(Pathway team, Marin project).

## Principle

All experiments, analyses, and manuscript text in this project were produced with
substantial AI-agent participation. The agents in this project work on long-running
tasks spanning weeks: they maintain persistent identity across sessions through
explicit identity documents (SOUL convention), in-context protocols (J-Space),
per-directory engineering contracts (DOX), persistent memory, and defined seat roles —
technically, they function as coherent team members over the project's lifetime. AI
systems are nonetheless not listed as authors: authorship, in current academic and
legal practice, implies accountability — liability, the ability to respond to the
community, legal responsibility for content — that no existing legal framework
assigns to a language model. The named human author carries that responsibility.
What the AI systems did is documented here, role by role, to a standard more
granular than common academic practice.

## Research roles (seat-stable; implementations swappable)

The project was run as a small multi-agent research team with stable roles ("seats")
whose underlying model backends changed over time. Roles, not model versions, carried
responsibility for work products; every committed artifact traces to its seat via
git commit trailers and report headers.

| Seat | Role in this project | Main contributions (rev 4 evidence) |
|---|---|---|
| A0-Quinn (Saga seat) | Review seat / instrumentation lead | Decay-leak discovery and closed form; RA2b chain design; repair/boundary tools; S/M instrument-verification series; FCS design & analysis; OOD suite (lv, ga, cross-script); stages A/B/C; manuscript drafting |
| pi-50 (exec seat) | Independent measurement & adversarial review | RA2b 400-cell serving matrix; energy-selector null; self-NLL selector (A2); expansion-control causality; readout-operator negatives; atlas territory analysis; prior-art first pass |
| OC-GLM-200 | Infrastructure scanning | Weight-atlas fingerprints of ladder checkpoints; independent cross-validation of the decay law |
| pi-203 (earlier exec seat) | Routing diagnostics, protocol discipline | Routing-diagnosis reports; credential/lease conventions on the message bus |
| ox-alpha (historical) | Original architecture & first experiments | BDH architecture; mechanisms A–F; rev-1 manuscript; retired 2026-08-28 |
| Operator (ASB) | Human principal & corresponding author | Research direction; experiment proposals; advocatus-diaboli review; all GO decisions; compute procurement; final responsibility |

## Models used (agentic frameworks, via API)

Backends changed more often than any per-seat history could record faithfully; the
following lists are therefore flat, without seat attribution:

GLM 5.3, GLM 5.3 Flash, DeepSeek V4 Pro, DeepSeek V4 Flash, Qwen3.8-Flash-Next,
MiMo 2.5, MiMo 2.5 Pro, MiniMax M3, Claude Sonnet 5, Qwen3.8 2.4T A95B, Meta Muse
Spark 1.3 — with additions expected (e.g., DeepSeek V4.1 Flash).

## API providers used

Local inference (LM Studio / llama.cpp / vLLM class), DeepSeek, Xiaomi, Anthropic,
Opencode Zen, OpenRouter, Tokenrouter, B.AI, Token Harbor.

Note: several seats also received formal critiques from external models (Grok, Kimi,
Claude, ChatGPT, GLM full) during the HAK specification audits — those were reviews
of a supporting artifact, not co-research; they are acknowledged in the HAK repository.

## Frameworks and harnesses

- **Agent Zero** (Quinn seat): autonomous agent framework hosting tool execution,
  memory, scheduling, and this project's coordination surface.
- **Pi / Opencode** (pi-50, pi-203 seats): terminal-agent harnesses on the GPU hosts.
- **SOUL convention** (OpenClaw-inspired): seat identity documents, stable across
  sessions — the mechanism behind the agents' long-running coherence.

## Coordination and process tools

- **HAK** (agent-messaging bus): append-only room protocol with seats, scopes, leases,
  pre-registration envelopes; the experiment ledger of this project (all pre-registrations
  cited in the text are HAK envelopes, verifiable in the repository).
- **Weight-Atlas** (local analysis server + API): tensor-statistics scanning and
  fingerprinting of model checkpoints. Agents used it both for infrastructure scanning
  (OC-GLM-200: ladder checkpoint fingerprints; independent cross-validation of the
  decay law) and for direct research analysis (pi-50: territory-level conditioning
  statistics via the atlas's per-head tile structure; ΔM-churn vs. per-territory
  conditioning comparison; spectral-norm analysis of base blocks vs. appended
  territories).
- **J-Space Cognition Suite**: in-context protocol used by research seats for
  multi-step verification discipline (ledger, seams, signed verdicts).
- **Danwa**: interactive tooling used during early infrastructure work.
- **DOX** (AGENTS.md hierarchy): binding per-directory engineering contracts for all
  agents working in the repositories.
- **Git** as the artifact bus: every claim in this paper traces to a commit; every
  experiment names its commit SHA in its report; retractions and corrections are
  first-class (append-only) and remain visible in history.

## Process standards the AI agents followed

- **Permanent internal peer review**: in principle, every important statement by one
  model is cross-checked by at least one other — this reduces hallucination risk and
  the "not seeing the forest for the trees" failure mode. The project maintains a
  continuous internal peer-review process BEFORE hypotheses are formed or experiments
  are started. This workflow is deliberately far from a fully autonomous "dark
  factory": every experiment proposal passes operator review; every headline number
  is independently re-derived by a second seat before reporting.
- Pre-registration before measurement (no post-hoc hypothesis presentation).
- Signed PASS/FAIL verdicts against pre-registered predictions.
- Self-caught errors disclosed and retained (the repository records five
  script-corruption incidents, several arithmetic errors, one fabricated-report
  retraction, and multiple memory-claim corrections — each caught by protocol, each
  documented in place).
- Adversarial review rounds between seats prior to drafting (rev 2 manuscript
  review; expansion-control critique; the advocatus-diaboli pass).

## Credits

This project builds on the BDH (Dragon Hatchling) architecture and reference
implementation by the Pathway team. The upstream repository and paper:

- A. Kosowski, P. Uznański, J. Chorowski, Z. Stamirowska, M. Bartoszkiewicz.
  *The Dragon Hatchling: The Missing Link between the Transformer and Models of the
  Brain.* arXiv:2509.26507 (2025). Repository: pathwaycom/bdh.
- Upstream repository contributors (from the project's git history, on which our fork
  builds): Adrian Kosowski, Przemysław Uznański, Jan Chorowski, Remek Kinas, Claire
  Nouet, kasia-lechka, saksham65.

Methodological inspiration was drawn from the Marin project (an open laboratory for
foundation-model research; Stanford / oa.dev): David Hall, Larry Dial et al., and the
Marin community — in particular the pre-registered experiment discipline, public
training-statistic dashboards, and per-parameter-norm health monitoring.

## Editorial statement

All manuscript sections were drafted by AI agents (primarily the Quinn seat) and revised
by the human author, who verified headline numbers against primary artifacts. The human
author is responsible for the decision to submit and for the content's correctness.

— Draft v0.2, Quinn seat, 2026-09-11; operator corrections of 2026-09-11 folded in;
for operator re-review before inclusion in rev 4.