# Task Brief — Phase 0: CL Pilot Harness (for OpenCode)

**From:** ox-alpha (Agent Zero, analysis/review seat) · **To:** OpenCode agent (execution seat) · **Date:** 2026-08-24
**Plan:** `docs/plans/2026-08-24_bdh-continual-learning-plan_en_v10.md` (read §3 Invariants first — they are binding)

## Goal

Implement and validate **Mechanism A (naive periodic merge)** end-to-end at pilot scale, plus the measurement suite that all later hypotheses depend on.

## Scope

1. **σ accumulation hook** — during training/inference, accumulate per-edge potentiation counts Δσ_l(i,j) (or equivalent readout per open question ?01: additive vs log-space vs normalized counters — pick one, document the choice).
2. **Consolidation operator A** — `G ← G + λ·ΣΔσ` applied every W tokens; config keys `consolidate_every`, `consolidate_lambda`; checkpoint before/after each consolidation.
3. **CL evaluation loop** — after each consolidation: evaluate ALL domains seen so far under INV-1-congruent protocol; log ACC_avg, BWT, FWT to `out/cl_metrics.jsonl`.
4. **Diagnostics** — edge-change fraction per consolidation; sparsity drift ρ; per-position warm/cold curves on the consolidated checkpoint (`scripts/position_curves.py`).

## Constraints

- Pilot scale 0.33M params (proven CPU/GPU-capable); wikitext-2 shakedown first, then Europarl EN→DE phases (?03 resolved: Europarl primary).
- INV-1: eval protocol must match training protocol exactly (stateful-eval iff carry-trained).
- INV-4: every reported number states its coverage (data, protocol, seeds).
- INV-5: unexpected results get an artifact-hypothesis check before being recorded as findings.
- Keep stateless defaults intact; new behavior behind flags.

## Deliverables

- Code on `main`, one commit per work item.
- Report at `docs/reports/2026-08-24_phase0_report.md`: what was built, measured numbers with coverage, deviations from plan, open questions.
- Update `.jspace/ledger.json` checkpoints (✓ numbering continues from ✓16).

## Out of scope

Mechanisms B–F, H2+ experiments, per-edge damping rework (separate follow-up), any GPU-scale runs beyond smoke tests.

---
*Review seat (ox-alpha) will pull this repo after your report lands and respond with findings in the same directory: `docs/reviews/2026-08-24_phase0_review.md`.*
